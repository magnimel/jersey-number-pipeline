import glob
import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Kinetics normalization stats (matches MoviNet pretraining)
_MEAN = [0.45, 0.45, 0.45]
_STD  = [0.225, 0.225, 0.225]

_to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


class TrackletVideoDataset(Dataset):
    """Loads each tracklet as a fixed-length video clip for MoviNet.

    Augmentation strategy:
      Train — random temporal sampling + RandomResizedCrop applied
              consistently across all frames in the clip.
      Eval  — uniform temporal sampling + Resize(256) / CenterCrop(224).

    No horizontal flip (mirrors jersey digits) and no per-frame color
    jitter (causes temporal flicker and is not applied consistently).
    """

    def __init__(self, gt_file, img_dir, augment: bool, n_frames=16, img_size=172, indices=None):
        with open(gt_file) as f:
            gt = json.load(f)

        tracklets = []
        for tracklet_id, jersey_number in gt.items():
            jersey_number = int(jersey_number)
            if jersey_number < 0:
                continue
            label = 0 if jersey_number < 10 else 1
            tracklet_dir = os.path.join(img_dir, tracklet_id)
            if os.path.isdir(tracklet_dir):
                # Subdirectory layout: img_dir/{tracklet_id}/{frame}.jpg
                frame_paths = sorted(glob.glob(os.path.join(tracklet_dir, "*.jpg")))
            else:
                # Flat layout: img_dir/{tracklet_id}_{frame}.jpg
                frame_paths = sorted(glob.glob(os.path.join(img_dir, f"{tracklet_id}_*.jpg")))
            if not frame_paths:
                continue
            tracklets.append((tracklet_id, frame_paths, label))

        if indices is not None:
            tracklets = [tracklets[i] for i in indices]

        self.tracklets = tracklets
        self.tracklet_ids = [tid for tid, _, _ in tracklets]
        self.tracklets = [(fp, lbl) for _, fp, lbl in tracklets]
        self.augment = augment
        self.n_frames = n_frames
        self.img_size = img_size

        n0 = sum(1 for _, l in self.tracklets if l == 0)
        n1 = sum(1 for _, l in self.tracklets if l == 1)

        print(f"  Loaded: 1-digit={n0}, 2-digit={n1}, total={len(self.tracklets)}")

    def __len__(self):
        return len(self.tracklets)

    def _sample_frames(self, frame_paths):
        n = len(frame_paths)
        if self.augment:
            # Random temporal sampling during training
            if n >= self.n_frames:
                idx = sorted(np.random.choice(n, self.n_frames, replace=False))
            else:
                idx = list(range(n))
        else:
            # Uniform temporal sampling for eval
            if n >= self.n_frames:
                idx = np.linspace(0, n - 1, self.n_frames, dtype=int)
            else:
                idx = list(range(n))
        sampled = [frame_paths[i] for i in idx]
        # Pad short tracklets by repeating last frame
        while len(sampled) < self.n_frames:
            sampled.append(sampled[-1])
        return sampled

    def __getitem__(self, idx):
        frame_paths, label = self.tracklets[idx]
        imgs = [Image.open(p).convert("RGB") for p in self._sample_frames(frame_paths)]

        size = self.img_size
        if self.augment:
            # Sample crop params once, apply identically to every frame
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                imgs[0], scale=(0.5, 1.0), ratio=(3/4, 4/3)
            )
            frames = [
                _to_tensor_norm(TF.resized_crop(img, i, j, h, w, (size, size)))
                for img in imgs
            ]
        else:
            frames = [
                _to_tensor_norm(TF.center_crop(TF.resize(img, int(size * 256 / 224)), size))
                for img in imgs
            ]

        # (T, C, H, W) -> (C, T, H, W) for MoviNet
        video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        return video, torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.stack(labels)
