"""Digit count trainer using HuggingFace Trainer.

Binary classifier: 1-digit (0-9) vs 2-digit (10-99) jersey numbers.
All hyperparameters are CLI arguments for sweep compatibility.

Usage:
    python train_digit_count.py --data ./data/SoccerNet
    python train_digit_count.py --data ./data/SoccerNet --epochs 5 --lr 1e-4 --batch_size 64
    python train_digit_count.py --data ./data/SoccerNet --eval_only --model_path ./experiments/best_model.pth
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DigitCountHFDataset(torch.utils.data.Dataset):
    """HuggingFace-compatible dataset for digit count classification."""

    def __init__(self, gt_file, img_dir, transform, balanced=False):
        with open(gt_file, "r") as f:
            gt = json.load(f)

        self.image_paths = []
        self.labels = []
        self.transform = transform

        for tracklet_id, jersey_number in gt.items():
            jersey_number = int(jersey_number)
            if jersey_number < 0:
                continue
            label = 0 if jersey_number < 10 else 1
            tracklet_dir = os.path.join(img_dir, tracklet_id)
            if not os.path.isdir(tracklet_dir):
                continue
            for img_name in os.listdir(tracklet_dir):
                self.image_paths.append(os.path.join(tracklet_dir, img_name))
                self.labels.append(label)

        n0 = sum(1 for l in self.labels if l == 0)
        n1 = sum(1 for l in self.labels if l == 1)
        print(f"  Loaded: 1-digit={n0}, 2-digit={n1}, total={len(self.labels)}")

        if balanced:
            idx0 = [i for i, l in enumerate(self.labels) if l == 0]
            idx1 = [i for i, l in enumerate(self.labels) if l == 1]
            min_n = min(len(idx0), len(idx1))
            random.shuffle(idx0)
            random.shuffle(idx1)
            keep = sorted(idx0[:min_n] + idx1[:min_n])
            self.image_paths = [self.image_paths[i] for i in keep]
            self.labels = [self.labels[i] for i in keep]
            print(f"  Balanced to {min_n} per class, total={len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(image)
        return {"pixel_values": pixel_values, "labels": torch.tensor(self.labels[idx], dtype=torch.float32)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DigitCountResNet(nn.Module):
    """ResNet34 binary classifier for digit count."""

    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, pixel_values, labels=None):
        logits = self.resnet(pixel_values).squeeze(-1)
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0).astype(int).flatten()
    labels = labels.astype(int).flatten()

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Custom data collator
# ---------------------------------------------------------------------------

def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train digit count classifier (1-digit vs 2-digit)")
    parser.add_argument("--data", required=True, help="SoccerNet root dir")
    parser.add_argument("--output_dir", default="./experiments/digit_count", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--balanced", action="store_true", help="Balance classes by undersampling majority")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    parser.add_argument("--model_path", help="Path to model weights (for eval or resume)")
    parser.add_argument("--early_stopping", type=int, default=0, help="Early stopping patience (0 = disabled)")
    args = parser.parse_args()

    set_seed(args.seed)

    # Paths
    train_gt = os.path.join(args.data, "train", "train_gt.json")
    train_img = os.path.join(args.data, "train", "images")
    val_gt = os.path.join(args.data, "val", "val_gt.json")
    val_img = os.path.join(args.data, "val", "images")
    test_gt = os.path.join(args.data, "test", "test_gt.json")
    test_img = os.path.join(args.data, "test", "images")

    # Model
    model = DigitCountResNet()
    if args.model_path:
        state = torch.load(args.model_path, map_location="cpu", weights_only=True)
        model.resnet.fc = nn.Linear(model.resnet.fc.in_features, 1)
        # Handle both old format (full resnet state) and new format
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # Old format saved model_ft keys — remap
            new_state = {}
            for k, v in state.items():
                new_key = k.replace("model_ft.", "resnet.")
                new_state[new_key] = v
            model.load_state_dict(new_state)
        print(f"Loaded weights from {args.model_path}")

    if args.eval_only:
        print("Loading test set...")
        test_ds = DigitCountHFDataset(test_gt, test_img, EVAL_TRANSFORM)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=args.batch_size,
            seed=args.seed,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
        )
        metrics = trainer.evaluate(eval_dataset=test_ds)
        print(f"\nTest results: {metrics}")
        return

    # Training
    print("Loading training set...")
    train_ds = DigitCountHFDataset(train_gt, train_img, TRAIN_TRANSFORM, balanced=args.balanced)
    print("Loading validation set...")
    val_ds = DigitCountHFDataset(val_gt, val_img, EVAL_TRANSFORM)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
    )

    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    # Save final best model as a single .pth file for easy loading
    best_path = os.path.join(args.output_dir, "best_model.pth")
    torch.save(model.state_dict(), best_path)
    print(f"\nBest model saved to {best_path}")

    # Final eval on val
    val_metrics = trainer.evaluate()
    print(f"\nVal metrics: {val_metrics}")

    # Eval on test if available
    if os.path.exists(test_gt):
        print("Loading test set...")
        test_ds = DigitCountHFDataset(test_gt, test_img, EVAL_TRANSFORM)
        test_metrics = trainer.evaluate(eval_dataset=test_ds)
        print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
