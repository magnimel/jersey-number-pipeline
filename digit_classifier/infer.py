"""
Digit classifier inference script.

Loads DigitCountMoviNet and runs per-tracklet inference on a flat crops directory,
producing a JSON file mapping tracklet_id -> p_2digit (probability of 2-digit jersey).

Usage (called as subprocess by main.py --improved):
    python digit_classifier/infer.py \
        --crops_dir out/SoccerNetResults/test/crops_sr/imgs \
        --checkpoint models/digit_a3_bs16_nf256_lr4.3e-04_wd5.6e-03.ckpt \
        --output_json out/SoccerNetResults/test/digit_predictions.json \
        --batch_size 32
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from digit_classifier.model import DigitCountMoviNet

_MEAN = [0.45, 0.45, 0.45]
_STD  = [0.225, 0.225, 0.225]
_to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


def load_model(checkpoint_path, device):
    """Load DigitCountMoviNet from a PyTorch Lightning checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    hparams = ckpt.get("hyper_parameters", {})
    model_id = hparams.get("model_id", "a3")

    model = DigitCountMoviNet(model_id=model_id)

    raw_sd = ckpt.get("state_dict", ckpt)
    sd = {k[len("model."):]: v for k, v in raw_sd.items() if k.startswith("model.")}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"[DigitClassifier] Loaded model_id={model_id} from {checkpoint_path}")
    return model


class CropsDataset(Dataset):
    """Groups flat crops directory into per-tracklet video clips for MoviNet."""

    def __init__(self, crops_dir, n_frames=16, img_size=172):
        all_imgs = sorted(glob.glob(os.path.join(crops_dir, "*.jpg")))
        tracklet_frames: dict[str, list[str]] = {}
        for path in all_imgs:
            fname = os.path.basename(path)
            tracklet_id = fname.split("_")[0]
            tracklet_frames.setdefault(tracklet_id, []).append(path)

        self.tracklets = [(tid, sorted(paths)) for tid, paths in sorted(tracklet_frames.items())]
        self.n_frames = n_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.tracklets)

    def _sample_frames(self, frame_paths):
        n = len(frame_paths)
        if n >= self.n_frames:
            idx = np.linspace(0, n - 1, self.n_frames, dtype=int)
        else:
            idx = list(range(n))
        sampled = [frame_paths[i] for i in idx]
        while len(sampled) < self.n_frames:
            sampled.append(sampled[-1])
        return sampled

    def __getitem__(self, idx):
        tracklet_id, frame_paths = self.tracklets[idx]
        size = self.img_size
        resize_to = int(size * 256 / 224)
        imgs = [Image.open(p).convert("RGB") for p in self._sample_frames(frame_paths)]
        frames = [
            _to_tensor_norm(TF.center_crop(TF.resize(img, resize_to), size))
            for img in imgs
        ]
        video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
        return tracklet_id, video


def _collate(batch):
    ids, videos = zip(*batch)
    return list(ids), torch.stack(videos)


def run_inference(model, crops_dir, device, batch_size=32):
    """Run digit classifier on all tracklets in crops_dir.

    Returns:
        dict mapping tracklet_id (str) -> p_2digit (float in [0, 1])
    """
    dataset = CropsDataset(crops_dir)
    if len(dataset) == 0:
        print("[DigitClassifier] No images found in crops_dir.")
        return {}

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=_collate,
    )

    results: dict[str, float] = {}
    with torch.no_grad():
        for ids, videos in loader:
            videos = videos.to(device)
            logits = model(videos)          # (B,) raw logits
            probs = torch.sigmoid(logits).cpu().tolist()
            for tid, p in zip(ids, probs):
                results[tid] = p

    return results


def main():
    parser = argparse.ArgumentParser(description="Digit classifier inference")
    parser.add_argument("--crops_dir", required=True,
                        help="Flat directory of crop images (trackletid_frame.jpg)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to DigitCountMoviNet PL checkpoint (.ckpt)")
    parser.add_argument("--output_json", required=True,
                        help="Output JSON: {tracklet_id: p_2digit}")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DigitClassifier] Device: {device}")

    model = load_model(args.checkpoint, device)
    results = run_inference(model, args.crops_dir, device, args.batch_size)
    print(f"[DigitClassifier] Predicted {len(results)} tracklets")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DigitClassifier] Wrote predictions to {args.output_json}")


if __name__ == "__main__":
    main()
