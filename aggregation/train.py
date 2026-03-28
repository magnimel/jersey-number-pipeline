"""
Training script for TrackletAggregator using PyTorch Lightning.

Usage (from repo root):
    python aggregation/train.py \
        data.str_results=... data.gt=... \
        data.test_str_results=... data.test_gt=...

    # Override any param on the command line (Hydra syntax):
    python aggregation/train.py training.lr=5e-4 training.epochs=50

W&B credentials are loaded from aggregation/.env (copy from aggregation/.env.template).
"""

import os
import sys

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig, OmegaConf

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aggregation.model import TrackletAggregator
from aggregation.dataset import TrackletDataset, collate_fn


class TrackletDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._setup_done = False

    def setup(self, stage=None):
        if self._setup_done:
            return
        self._setup_done = True

        self.full_dataset = TrackletDataset(self.cfg.data.str_results, self.cfg.data.gt)
        print(f"Total tracklets: {len(self.full_dataset)}")

        if self.cfg.data.val_str_results and self.cfg.data.val_gt:
            self.train_dataset = self.full_dataset
            self.val_dataset = TrackletDataset(self.cfg.data.val_str_results, self.cfg.data.val_gt)
        else:
            indices = list(range(len(self.full_dataset)))
            labels = [self.full_dataset.samples[i][3] for i in indices]

            # Singleton classes (only 1 sample) cannot be stratified — force into train
            from collections import Counter
            label_counts = Counter(labels)
            singleton_idx = [i for i, l in zip(indices, labels) if label_counts[l] == 1]
            stratify_idx = [i for i, l in zip(indices, labels) if label_counts[l] > 1]
            stratify_labels = [labels[i] for i in stratify_idx]

            if singleton_idx:
                print(f"[DataModule] {len(singleton_idx)} singleton-class tracklets forced into train.")

            split_train_idx, val_idx = train_test_split(
                stratify_idx, test_size=self.cfg.data.val_split,
                stratify=stratify_labels, random_state=self.cfg.training.seed,
            )
            train_idx = split_train_idx + singleton_idx
            self.train_dataset = Subset(self.full_dataset, train_idx)
            self.val_dataset = Subset(self.full_dataset, val_idx)

        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

        if self.cfg.data.test_str_results and self.cfg.data.test_gt:
            self.test_dataset = TrackletDataset(
                self.cfg.data.test_str_results, self.cfg.data.test_gt
            )
            print(f"Test: {len(self.test_dataset)}")
        else:
            self.test_dataset = None

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.training.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)


class TrackletAggregatorLit(pl.LightningModule):
    def __init__(self, cfg: DictConfig, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "cfg"])
        self.cfg = cfg
        self.model = TrackletAggregator(use_digit_classifier=cfg.model.use_digit_classifier)

        if cfg.training.use_class_weights and class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, padded, lengths, p_2digit=None):
        return self.model(padded, lengths, p_2digit)

    def _step(self, batch):
        padded, lengths, p_2digit, labels = batch
        p_arg = p_2digit if self.cfg.model.use_digit_classifier else None
        logits = self(padded, lengths, p_arg)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("test/loss", loss)
        self.log("test/acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.training.epochs
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.training.seed, workers=True)

    # Data
    dm = TrackletDataModule(cfg)
    dm.setup()

    # Class weights (computed from full training data)
    class_weights = None
    if cfg.training.use_class_weights:
        counts = dm.full_dataset.class_counts().float()
        weights = torch.zeros(100)
        nonzero = counts > 0
        weights[nonzero] = counts[nonzero].sum() / (counts[nonzero] * nonzero.sum().float())
        class_weights = weights

    # Model
    lit_model = TrackletAggregatorLit(cfg, class_weights=class_weights)
    print(f"Parameters: {lit_model.model.num_parameters():,}")
    print(f"use_digit_classifier: {cfg.model.use_digit_classifier}")

    # Logger — init first so we can use the W&B run name for the output dir
    wandb_logger = None
    t = cfg.training
    run_name = (
        cfg.output.run_name
        or f"agg_bs{t.batch_size}_lr{t.lr:.1e}_wd{t.weight_decay:.1e}_cw{'T' if t.use_class_weights else 'F'}"
    )
    if cfg.wandb.enabled:
        try:
            entity = cfg.wandb.entity or os.environ.get("WANDB_ENTITY")
            wandb_logger = WandbLogger(
                project=cfg.wandb.project,
                entity=entity,
                name=run_name,
                log_model=False,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            run_name = wandb_logger.experiment.name
        except Exception as e:
            print(f"[W&B] Failed to initialise: {e}. Continuing without logging.")

    # Per-run output dir named after the W&B run
    output_dir = os.path.join(cfg.output.output_dir, run_name or "unnamed")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # Callbacks
    ckpt_filename = "epoch{epoch:03d}_valacc{val/acc:.4f}"
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename=ckpt_filename,
            monitor="val/acc",
            mode="max",
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/acc",
            mode="max",
            patience=cfg.training.early_stopping_patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger or False,
        callbacks=callbacks,
        log_every_n_steps=1,
        deterministic=True,
    )

    trainer.fit(lit_model, datamodule=dm)

    best_ckpt = callbacks[0].best_model_path
    best_val_acc = callbacks[0].best_model_score
    print(f"\nTraining done. Best val_acc={best_val_acc:.4f}")
    print(f"Best checkpoint: {best_ckpt}")

    with open(os.path.join(output_dir, "best_ckpt.txt"), "w") as f:
        f.write(best_ckpt + "\n")
    print(f"Update configuration.py: dataset['SoccerNet']['aggregation_model'] = '{best_ckpt}'")

    # Test set evaluation using the best checkpoint
    if dm.test_dataset is not None:
        print("\nEvaluating best checkpoint on test set...")
        trainer.test(lit_model, datamodule=dm, ckpt_path=best_ckpt)


if __name__ == "__main__":
    main()
