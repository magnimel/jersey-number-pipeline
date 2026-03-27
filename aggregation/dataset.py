"""
TrackletDataset - loads jersey_id_results*.json + ground truth, groups by tracklet.

Each sample is one tracklet:
  - logits_seq : list of T tensors, each (33,)  [3x11 PARSeq logits flattened]
  - p_2digit   : float, 1.0 if label has 2 digits else 0.0  (GT during train, MoviNet at test)
  - label      : int 0-99

Tracklet ID is the first underscore-delimited token of each crop filename,
e.g. "0001_000123.png" -> tracklet "0001".
"""

import json
import torch
from collections import defaultdict
from torch.utils.data import Dataset


class TrackletDataset(Dataset):
    def __init__(self, str_results_path, gt_path, tracklet_ids=None):
        """
        Args:
            str_results_path : path to jersey_id_results*.json
            gt_path          : path to *_gt.json  (tracklet_id -> label string)
            tracklet_ids     : optional set/list to restrict which tracklets are loaded
        """
        with open(str_results_path) as f:
            str_results = json.load(f)
        with open(gt_path) as f:
            gt = json.load(f)

        # Group crops by tracklet prefix
        tracklet_crops = defaultdict(list)
        for fname, data in str_results.items():
            tracklet = fname.split("_")[0]
            if tracklet in gt:
                tracklet_crops[tracklet].append((fname, data))

        if tracklet_ids is not None:
            tracklet_ids = set(tracklet_ids)
            tracklet_crops = {k: v for k, v in tracklet_crops.items() if k in tracklet_ids}

        self.samples = []
        skipped = 0
        for tracklet, crops in tracklet_crops.items():
            label_str = gt[tracklet]
            try:
                label_int = int(label_str)
            except (ValueError, TypeError):
                skipped += 1
                continue
            if not (0 <= label_int <= 99):
                skipped += 1
                continue

            # Sort crops by filename for deterministic ordering
            crops.sort(key=lambda x: x[0])

            logits_seq = []
            for _, data in crops:
                raw = data.get("logits")
                if raw is None:
                    continue
                logits = torch.tensor(raw, dtype=torch.float32).flatten()  # (33,)
                logits_seq.append(logits)

            if not logits_seq:
                skipped += 1
                continue

            p_2digit = float(len(str(label_int)) == 2)
            self.samples.append((tracklet, logits_seq, p_2digit, label_int))

        if skipped:
            print(f"[TrackletDataset] Skipped {skipped} tracklets (bad label or no logits).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tracklet, logits_seq, p_2digit, label = self.samples[idx]
        return logits_seq, p_2digit, label

    def tracklet_id(self, idx):
        return self.samples[idx][0]

    def class_counts(self):
        """Return a tensor of size 100 with the count of each jersey number class."""
        counts = torch.zeros(100, dtype=torch.long)
        for _, _, _, label in self.samples:
            counts[label] += 1
        return counts


def collate_fn(batch):
    """Pad variable-length tracklets to the longest one in the batch."""
    logits_seqs, p_2digits, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in logits_seqs], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = torch.zeros(len(batch), max_len, 33)
    for i, seq in enumerate(logits_seqs):
        t = len(seq)
        padded[i, :t] = torch.stack(seq)

    p_2digit_tensor = torch.tensor(p_2digits, dtype=torch.float32).unsqueeze(-1)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, p_2digit_tensor, labels_tensor
