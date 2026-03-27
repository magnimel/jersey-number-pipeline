"""
Evaluate or run inference with a trained TrackletAggregator checkpoint.

Usage:
    # Evaluate against ground truth
    python aggregation/evaluate.py \
        --checkpoint aggregation/checkpoints/aggregator_epoch050_valacc0.7200.pt \
        --str_results out/SoccerNetResults/test/jersey_id_results.json \
        --gt data/SoccerNet/test/test_gt.json

    # Inference only (no GT), writes {tracklet: predicted_number} to JSON
    python aggregation/evaluate.py \
        --checkpoint aggregation/checkpoints/aggregator_epoch050_valacc0.7200.pt \
        --str_results out/SoccerNetResults/test/jersey_id_results.json \
        --output_json out/SoccerNetResults/test/aggregation_results.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

# Load W&B / any other secrets if available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aggregation.model import TrackletAggregator
from aggregation.dataset import TrackletDataset, collate_fn


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    use_digit_classifier = ckpt.get("use_digit_classifier", False)
    model = TrackletAggregator(use_digit_classifier=use_digit_classifier).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt.get('epoch', '?')}  "
          f"val_acc={ckpt.get('val_acc', 0.0):.4f}  "
          f"use_digit_classifier={use_digit_classifier}")
    return model, use_digit_classifier


def _collate_raw(batch):
    """Collate (tracklet_id, logits_seq) pairs into a padded batch."""
    tracklet_ids, logits_seqs = zip(*batch)
    lengths = torch.tensor([len(s) for s in logits_seqs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(batch), max_len, 33)
    for i, seq in enumerate(logits_seqs):
        t = len(seq)
        padded[i, :t] = torch.stack(seq)
    return list(tracklet_ids), padded, lengths


def run_inference_no_gt(model, str_results_path, device, use_digit_classifier, batch_size):
    """
    Run inference directly on a jersey_id_results JSON without needing ground truth.
    Returns {tracklet_id: predicted_jersey_number_str}.
    """
    with open(str_results_path) as f:
        str_results = json.load(f)

    tracklet_crops = defaultdict(list)
    for fname, data in str_results.items():
        raw = data.get("logits")
        if raw is None:
            continue
        tracklet = fname.split("_")[0]
        logits = torch.tensor(raw, dtype=torch.float32).flatten()
        tracklet_crops[tracklet].append((fname, logits))

    # Sort crops within each tracklet for deterministic ordering
    for t in tracklet_crops:
        tracklet_crops[t].sort(key=lambda x: x[0])

    tracklet_list = sorted(tracklet_crops.keys())
    # Build a flat list of (tracklet_id, logits_seq) for batching
    samples = [
        (t, [logits for _, logits in tracklet_crops[t]])
        for t in tracklet_list
    ]

    results = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            ids, padded, lengths = _collate_raw(batch)
            padded = padded.to(device)
            lengths = lengths.to(device)
            # p_2digit unavailable at inference without MoviNet; pass None
            logits_out = model(padded, lengths, None)
            preds = logits_out.argmax(dim=1).cpu().tolist()
            for tid, pred in zip(ids, preds):
                results[tid] = str(pred)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrackletAggregator")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--str_results", required=True,
                        help="Path to jersey_id_results*.json")
    parser.add_argument("--gt", default=None,
                        help="Path to *_gt.json. If omitted, inference-only mode is used.")
    parser.add_argument("--output_json", default=None,
                        help="Write {tracklet: predicted_number} to this JSON file.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, use_digit_classifier = load_model(args.checkpoint, device)

    if args.gt:
        # Evaluation mode: use TrackletDataset to get ground truth labels
        dataset = TrackletDataset(args.str_results, args.gt)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn
        )
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for padded, lengths, p_2digit, labels in loader:
                padded = padded.to(device)
                lengths = lengths.to(device)
                p_arg = p_2digit.to(device) if use_digit_classifier else None
                logits = model(padded, lengths, p_arg)
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(labels.tolist())

        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        print(f"Accuracy: {correct}/{len(all_preds)} = {correct / len(all_preds):.4f}")

        if args.output_json:
            result = {
                dataset.tracklet_id(i): str(all_preds[i])
                for i in range(len(dataset))
            }
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Wrote {len(result)} predictions to {args.output_json}")

    else:
        # Inference-only mode: no GT needed, no dataset class required
        result = run_inference_no_gt(
            model, args.str_results, device, use_digit_classifier, args.batch_size
        )
        print(f"Predicted {len(result)} tracklets")

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Wrote predictions to {args.output_json}")
        else:
            print("Pass --output_json to save predictions.")


if __name__ == "__main__":
    main()
