"""Digit count trainer using MoviNet + PyTorch Lightning.

Binary classifier: 1-digit (0-9) vs 2-digit (10-99) jersey numbers.
Each tracklet of enhanced crops is treated as a video sequence.

Usage (from repo root):
    python digit_classifier/train.py --data ./data/SoccerNet

W&B sweep:
    wandb sweep digit_classifier/conf/sweep.yaml
    wandb agent <entity>/<project>/<sweep_id>

Credentials are loaded from digit_classifier/.env (copy from .env.template).
"""

import argparse
import json
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from digit_classifier.dataset import TrackletVideoDataset, collate_fn
from digit_classifier.model import DigitCountMoviNet

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class DigitCountLit(pl.LightningModule):
    def __init__(self, model_id, lr, weight_decay, pos_weight):
        super().__init__()
        self.save_hyperparameters()
        self.model = DigitCountMoviNet(model_id)
        # Register as buffer so it moves to GPU automatically with the model
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def _step(self, batch):
        videos, labels = batch
        logits = self.model(videos)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight)
        preds = (logits > 0).float()
        acc = (preds == labels).float().mean()
        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self._step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self._step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def test_step(self, batch, _):
        loss, acc = self._step(batch)
        self.log("test/loss", loss)
        self.log("test/acc", acc)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------

class TrackletDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, n_frames, img_size, val_split, batch_size, num_workers, seed,
                 train_gt=None, test_gt=None, train_img_dir=None, test_img_dir=None, skip_test=False):
        super().__init__()
        self.data_dir = data_dir
        self.n_frames = n_frames
        self.img_size = img_size
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_gt_override = train_gt
        self.test_gt_override = test_gt
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        self.skip_test = skip_test
        self._setup_done = False

    def setup(self, stage=None):
        if self._setup_done:
            return
        self._setup_done = True
        train_gt  = self.train_gt_override or os.path.join(self.data_dir, "train", "train_gt.json")
        train_img = self.train_img_dir or os.path.join(self.data_dir, "train", "images")
        test_gt   = self.test_gt_override or os.path.join(self.data_dir, "test",  "test_gt.json")
        test_img  = self.test_img_dir  or os.path.join(self.data_dir, "test",  "images")

        # Store resolved paths for post-training inference
        self._train_gt, self._train_img = train_gt, train_img
        self._test_gt,  self._test_img  = test_gt,  test_img

        # Build full train dataset to get labels for stratified split
        full = TrackletVideoDataset(train_gt, train_img, augment=False, n_frames=self.n_frames, img_size=self.img_size)
        labels = [l for _, l in full.tracklets]
        all_idx = list(range(len(full.tracklets)))

        train_idx, val_idx = train_test_split(
            all_idx, test_size=self.val_split, stratify=labels, random_state=self.seed
        )

        self.train_ds = TrackletVideoDataset(
            train_gt, train_img, augment=True, n_frames=self.n_frames, img_size=self.img_size, indices=train_idx
        )
        self.val_ds = TrackletVideoDataset(
            train_gt, train_img, augment=False, n_frames=self.n_frames, img_size=self.img_size, indices=val_idx
        )
        if not self.skip_test:
            self.test_ds = TrackletVideoDataset(
                test_gt, test_img, augment=False, n_frames=self.n_frames, img_size=self.img_size
            )

        # Compute pos_weight from training split
        n0 = sum(1 for _, l in self.train_ds.tracklets if l == 0)
        n1 = sum(1 for _, l in self.train_ds.tracklets if l == 1)
        self.pos_weight = n0 / n1 if n1 > 0 else 1.0
        print(f"  pos_weight (n0/n1 = {n0}/{n1}): {self.pos_weight:.3f}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.environ.get("DIGIT_DATA_DIR"),
                        help="Root dir containing train/test GT files (or set DIGIT_DATA_DIR)")
    parser.add_argument("--train_gt", default=os.environ.get("DIGIT_GT"),
                        help="Path to train GT JSON (overrides <data>/train/train_gt.json, or set DIGIT_GT)")
    parser.add_argument("--test_gt", default=os.environ.get("DIGIT_TEST_GT"),
                        help="Path to test GT JSON (overrides <data>/test/test_gt.json, or set DIGIT_TEST_GT)")
    parser.add_argument("--train_img_dir", default=os.environ.get("DIGIT_TRAIN_IMG_DIR"),
                        help="Train images dir, overrides <data>/train/images (or set DIGIT_TRAIN_IMG_DIR)")
    parser.add_argument("--test_img_dir", default=os.environ.get("DIGIT_TEST_IMG_DIR"),
                        help="Test images dir, overrides <data>/test/images (or set DIGIT_TEST_IMG_DIR)")
    parser.add_argument("--output_dir", default=os.environ.get("DIGIT_OUTPUT_DIR", "./digit_classifier/checkpoints"))
    parser.add_argument("--model_id", default="a0", choices=["a0", "a1", "a2", "a3", "a4", "a5"])
    parser.add_argument("--img_size", type=int, default=None,
                        help="Input resolution (default: 172 for a0/a1, 224 for a2)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_frames", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip test set evaluation (use when test crops are unavailable)")
    args = parser.parse_args()

    if not args.data:
        raise ValueError("--data is required (or set DIGIT_DATA_DIR env var)")

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    # Recompute img_size after sweep may have changed model_id
    if args.img_size is None:
        args.img_size = {
            "a0": 172, "a1": 172, "a2": 224, "a3": 256, "a4": 290, "a5": 320,
        }[args.model_id]

    # Build meaningful run name then let W&B confirm it
    run_name = f"digit_{args.model_id}_bs{args.batch_size}_nf{args.n_frames}_lr{args.lr:.1e}_wd{args.weight_decay:.1e}"
    wandb_logger = WandbLogger(
        project=os.environ.get("WANDB_PROJECT", "jersey-digit-classifier"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=run_name,
        log_model=False,
        config=vars(args),
    )
    run_name = wandb_logger.experiment.name

    dm = TrackletDataModule(
        data_dir=args.data,
        n_frames=args.n_frames,
        img_size=args.img_size,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_gt=args.train_gt,
        test_gt=args.test_gt,
        train_img_dir=args.train_img_dir,
        test_img_dir=args.test_img_dir,
        skip_test=args.skip_test,
    )
    dm.setup()

    model = DigitCountLit(
        model_id=args.model_id,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=dm.pos_weight,
    )

    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="epoch{epoch:03d}_valacc{val/acc:.4f}",
            monitor="val/acc",
            mode="max",
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor="val/acc", mode="max", patience=args.early_stopping_patience),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        deterministic="warn",
    )

    trainer.fit(model, dm)

    best_ckpt = callbacks[0].best_model_path
    best_val_acc = callbacks[0].best_model_score
    print(f"\nTraining done. Best val/acc={best_val_acc:.4f}" if best_val_acc is not None else "\nTraining done. No checkpoint saved.")
    print(f"Best checkpoint: {best_ckpt}")

    with open(os.path.join(output_dir, "best_ckpt.txt"), "w") as f:
        f.write(best_ckpt + "\n")

    if not args.skip_test and best_ckpt:
        trainer.test(model, dm, ckpt_path=best_ckpt)

    # Save digit classifier predictions for use as p_2digit in aggregation training
    if best_ckpt:
        _save_digit_predictions(model, best_ckpt, dm, args, output_dir, wandb_logger.experiment)
    else:
        print("Skipping predictions — no checkpoint was saved.")


def _save_digit_predictions(model, ckpt_path, dm, args, output_dir, run):
    """Run the best model over the full train (and test) set and save
    {tracklet_id: p_2digit} JSONs for use as aggregation model input."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = model.__class__.load_from_checkpoint(ckpt_path).to(device)
    best.eval()

    def infer(gt_file, img_dir, split_name):
        ds = TrackletVideoDataset(gt_file, img_dir, augment=False,
                                  n_frames=args.n_frames, img_size=args.img_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
        probs = []
        with torch.no_grad():
            for videos, _ in loader:
                logits = best(videos.to(device))
                probs.extend(torch.sigmoid(logits).cpu().tolist())

        out = {tid: round(p, 6) for tid, p in zip(ds.tracklet_ids, probs)}
        save_path = os.path.join(output_dir, f"digit_predictions_{split_name}.json")
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved {len(out)} {split_name} predictions → {save_path}")
        run.save(save_path)
        return save_path

    infer(dm._train_gt, dm._train_img, "train")
    if not args.skip_test:
        infer(dm._test_gt, dm._test_img, "test")


if __name__ == "__main__":
    main()
