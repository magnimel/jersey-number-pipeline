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

    if "state_dict" in ckpt:
        # PyTorch Lightning checkpoint: strip "model." prefix
        raw_sd = ckpt["state_dict"]
        sd = {k[len("model."):]: v for k, v in raw_sd.items() if k.startswith("model.")}
        # Detect use_digit_classifier from classifier head input dimension
        use_digit_classifier = sd["classifier.0.weight"].shape[1] == 257
        epoch = ckpt.get("epoch", "?")
    else:
        # Legacy custom format
        sd = ckpt["model_state_dict"]
        use_digit_classifier = ckpt.get("use_digit_classifier", False)
        epoch = ckpt.get("epoch", "?")

    model = TrackletAggregator(use_digit_classifier=use_digit_classifier).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded checkpoint  epoch={epoch}  "
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


def _calibrate_batch_size(model, sample_padded, sample_lengths, sample_p2digit, device, max_bs):
    import time
    n = sample_padded.shape[0]
    candidates = [bs for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if bs <= max_bs and bs <= n]
    if not candidates:
        return max_bs

    def _infer(bs):
        with torch.no_grad():
            p = (sample_p2digit[:bs].to(device) if sample_p2digit is not None else None)
            model(sample_padded[:bs].to(device), sample_lengths[:bs].to(device), p)

    try:
        _infer(1)
        if device.type != 'cpu':
            torch.cuda.synchronize()
    except Exception:
        pass

    best_bs, best_rate = candidates[0], 0.0
    print("[Aggregation] Calibrating batch size ...")
    for bs in candidates:
        repeat = (bs // n) + 1
        padded = sample_padded.repeat(repeat, 1, 1)[:bs]
        lengths = sample_lengths.repeat(repeat)[:bs]
        p2 = sample_p2digit.repeat(repeat, 1)[:bs] if sample_p2digit is not None else None
        try:
            times = []
            for _ in range(3):
                if device.type != 'cpu':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    p = p2.to(device) if p2 is not None else None
                    model(padded.to(device), lengths.to(device), p)
                if device.type != 'cpu':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            rate = bs / sorted(times)[1]
            print(f"  batch_size={bs:4d}  →  {rate:.1f} tracklets/s")
            if rate > best_rate:
                best_rate, best_bs = rate, bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  batch_size={bs:4d}  →  OOM (skipping)")
                break
            raise
    print(f"[Aggregation] Selected batch_size={best_bs} ({best_rate:.1f} tracklets/s)")
    return best_bs


def run_inference_no_gt(model, str_results_path, device, use_digit_classifier, batch_size,
                        digit_preds=None):
    """
    Run inference directly on a jersey_id_results JSON without needing ground truth.

    Args:
        digit_preds: Optional dict {tracklet_id: p_2digit (float)} from the digit
                     classifier.  Required when use_digit_classifier=True.
    Returns:
        {tracklet_id: predicted_jersey_number_str}
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

    if device.type != 'cpu' and samples:
        n_sample = min(64, len(samples))
        _, s_padded, s_lengths = _collate_raw(samples[:n_sample])
        if use_digit_classifier:
            s_p2 = torch.tensor(
                [digit_preds.get(t, 0.5) if digit_preds else 0.5 for t, _ in samples[:n_sample]],
                dtype=torch.float32,
            ).unsqueeze(1)
        else:
            s_p2 = None
        batch_size = _calibrate_batch_size(model, s_padded, s_lengths, s_p2, device, batch_size)

    results = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            ids, padded, lengths = _collate_raw(batch)
            padded = padded.to(device)
            lengths = lengths.to(device)

            if use_digit_classifier:
                # Build p_2digit tensor from pre-computed digit classifier predictions
                p_vals = [digit_preds.get(tid, 0.5) if digit_preds else 0.5 for tid in ids]
                p_2digit = torch.tensor(p_vals, dtype=torch.float32, device=device).unsqueeze(1)
            else:
                p_2digit = None

            logits_out = model(padded, lengths, p_2digit)
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
    parser.add_argument("--digit_preds", default=None,
                        help="Path to digit_predictions.json from digit classifier.")
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
        digit_preds = None
        if args.digit_preds and os.path.exists(args.digit_preds):
            with open(args.digit_preds) as f:
                digit_preds = json.load(f)
        result = run_inference_no_gt(
            model, args.str_results, device, use_digit_classifier, args.batch_size,
            digit_preds=digit_preds,
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
