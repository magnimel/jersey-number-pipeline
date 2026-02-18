from pathlib import Path
import sys
import os
import argparse

ROOT = './reid/centroids-reid/'
sys.path.append(str(ROOT))  # add ROOT to PATH

import numpy as np
import torch
from tqdm import tqdm
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import cfg
from train_ctl_model import CTLModel

from datasets.transforms import ReidTransforms


class TrackletDataset(Dataset):
    """Dataset for loading tracklet images for batch processing."""
    def __init__(self, tracklet_paths, transforms):
        """
        Args:
            tracklet_paths: List of (track_id, image_path) tuples
            transforms: Image transforms to apply
        """
        self.tracklet_paths = tracklet_paths
        self.transforms = transforms
    
    def __len__(self):
        return len(self.tracklet_paths)
    
    def __getitem__(self, idx):
        track_id, img_path = self.tracklet_paths[idx]
        img = cv2.imread(img_path)
        input_img = Image.fromarray(img)
        input_img = self.transforms(input_img)
        return track_id, input_img



# Based on this repo: https://github.com/mikwieczorek/centroids-reid
# Trained model from here: https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK
CONFIG_FILE = str(ROOT+'/configs/256_resnet50.yml')
MODEL_FILE = str(ROOT+'/models/resnet50-19c8e357.pth')

# dict used to get model config and weights using model version
ver_to_specs = {}
ver_to_specs["res50_market"] = (ROOT+'/configs/256_resnet50.yml', ROOT+'/models/market1501_resnet50_256_128_epoch_120.ckpt')
ver_to_specs["res50_duke"]   = (ROOT+'/configs/256_resnet50.yml', ROOT+'/models/dukemtmcreid_resnet50_256_128_epoch_120.ckpt')


def get_specs_from_version(model_version):
    conf, weights = ver_to_specs[model_version]
    conf, weights = str(conf), str(weights)
    return conf, weights

def generate_features_old(input_folder, output_folder, model_version='res50_market'):
    """OLD SLOW VERSION - kept for reference"""
    # load model
    CONFIG_FILE, MODEL_FILE = get_specs_from_version(model_version)
    cfg.merge_from_file(CONFIG_FILE)
    opts = ["MODEL.PRETRAIN_PATH", MODEL_FILE, "MODEL.PRETRAINED", True, "TEST.ONLY_TEST", True, "MODEL.RESUME_TRAINING", False]
    cfg.merge_from_list(opts)
    
    use_cuda = True if torch.cuda.is_available() and cfg.GPU_IDS else False
    print(f"Loading model from {cfg.MODEL.PRETRAIN_PATH}...")
    model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg)

    # print("Loading from " + MODEL_FILE)
    if use_cuda:
        model.to('cuda')
        print("using GPU")
    model.eval()
    tracks = os.listdir(input_folder)
    print("ReidTransorms...")
    transforms_base = ReidTransforms(cfg)
    val_transforms = transforms_base.build_transforms(is_train=False)
    print("Generating features...")
    for track in tqdm(tracks):
        features = []
        track_path = os.path.join(input_folder, track)
        images = os.listdir(track_path)
        output_file = os.path.join(output_folder, f"{track}_features.npy")
        for img_path in images:
            img = cv2.imread(os.path.join(track_path, img_path))
            input_img = Image.fromarray(img)
            input_img = torch.stack([val_transforms(input_img)])
            with torch.no_grad():
                _, global_feat = model.backbone(input_img.cuda() if use_cuda else input_img)
                global_feat = model.bn(global_feat)
            features.append(global_feat.cpu().numpy().reshape(-1,))

        np_feat = np.array(features)
        with open(output_file, 'wb') as f:
            np.save(f, np_feat)


def generate_features(input_folder, output_folder, model_version='res50_market', batch_size=64, num_workers=2):
    """
    Optimized batch processing version of feature generation.
    
    Args:
        input_folder: Folder containing tracklet directories with images
        output_folder: Folder to store features in, one file per tracklet
        model_version: Model version to use (default: 'res50_market')
        batch_size: Batch size for processing (default 64, tune for your GPU)
        num_workers: Number of dataloader workers (default 2 for Windows compatibility)
    """
    # Load model
    CONFIG_FILE, MODEL_FILE = get_specs_from_version(model_version)
    cfg.merge_from_file(CONFIG_FILE)
    opts = ["MODEL.PRETRAIN_PATH", MODEL_FILE, "MODEL.PRETRAINED", True, "TEST.ONLY_TEST", True, "MODEL.RESUME_TRAINING", False]
    cfg.merge_from_list(opts)
    
    use_cuda = True if torch.cuda.is_available() and cfg.GPU_IDS else False
    print(f"Loading model from {cfg.MODEL.PRETRAIN_PATH}...")
    model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg)

    if use_cuda:
        model.to('cuda')
        print("using GPU")
    model.eval()
    
    # Build transforms
    print("ReidTransforms...")
    transforms_base = ReidTransforms(cfg)
    val_transforms = transforms_base.build_transforms(is_train=False)
    
    # Build list of all images with their track IDs
    print("Preparing tracklet data...")
    tracks = os.listdir(input_folder)
    tracklet_paths = []
    track_image_counts = {}
    
    for track in tracks:
        track_path = os.path.join(input_folder, track)
        images = os.listdir(track_path)
        track_image_counts[track] = len(images)
        for img_name in images:
            tracklet_paths.append((track, os.path.join(track_path, img_name)))
    
    # Create dataset and dataloader
    dataset = TrackletDataset(tracklet_paths, val_transforms)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    
    # Process all images in batches
    print(f"Generating features in batches (batch_size={batch_size}, num_workers={num_workers})...")
    track_features = {track: [] for track in tracks}
    
    with torch.no_grad():
        for track_ids, images in tqdm(dataloader, total=len(dataloader)):
            if use_cuda:
                images = images.cuda()
            
            # Extract features
            _, global_feat = model.backbone(images)
            global_feat = model.bn(global_feat)
            features = global_feat.cpu().numpy()
            
            # Group features by track ID
            for track_id, feat in zip(track_ids, features):
                track_features[track_id].append(feat.reshape(-1,))
    
    # Save features for each track
    print("Saving features...")
    for track in tracks:
        output_file = os.path.join(output_folder, f"{track}_features.npy")
        np_feat = np.array(track_features[track])
        with open(output_file, 'wb') as f:
            np.save(f, np_feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracklets_folder', help="Folder containing tracklet directories with images")
    parser.add_argument('--output_folder', help="Folder to store features in, one file per tracklet")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for processing (default: 64)")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of dataloader workers (default: 2)")
    args = parser.parse_args()

    #create if does not exist
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    generate_features(args.tracklets_folder, args.output_folder, batch_size=args.batch_size, num_workers=args.num_workers)


