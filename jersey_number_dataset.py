from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pandas as pd
import json
from PIL import Image
from torchvision import transforms

data_transforms = {
    'train': {
        'resnet':
            transforms.Compose([
            transforms.RandomGrayscale(),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Image Net
            #transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
            ]),
        'vit':
            transforms.Compose([
                transforms.RandomGrayscale(),
                transforms.ColorJitter(brightness=.5, hue=.3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Image Net
                # transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
            ]),
        },

    'val': {
        'resnet':
            transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet
           #transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
        ]),
        'vit':
            transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet
           #transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
        ])
    },
    'test': {
        'resnet':
        transforms.Compose([ # same as val
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet
        #transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
    ]),
        'vit':
        transforms.Compose([ # same as val
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet
        #transforms.Normalize(mean=[0.548, 0.529, 0.539], std=[0.268, 0.280, 0.274]) # Hockey
    ]),
    }
}

class JerseyNumberDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train'):
        self.transform = data_transforms[mode]
        self.img_labels = pd.read_csv(annotations_file)
        unqiue_ids = np.unique(self.img_labels.iloc[:, 1].to_numpy())
        print(f"Datafile:{annotations_file}, number of labels:{len(self.img_labels)}, unique ids: {len(unqiue_ids)}")
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

class JerseyNumberMultitaskDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train'):
        self.transform = data_transforms[mode]
        self.img_labels = pd.read_csv(annotations_file)
        unqiue_ids = np.unique(self.img_labels.iloc[:, 1].to_numpy())
        print(f"Datafile:{annotations_file}, number of labels:{len(self.img_labels)}, unique ids: {len(unqiue_ids)}")
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def get_digit_labels(self, label):
        if label < 10:
            return label, 10
        else:
            return label // 10, label % 10

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        digit1, digit2 = self.get_digit_labels(label)
        if not (label> 0 and label < 100 and digit1 < 10 and digit1 > 0 and digit2 > -1 and digit2 < 11):
            print(label, digit1, digit2)
        if self.transform:
            image = self.transform(image)
        return image, label, digit1, digit2

class UnlabelledJerseyNumberLegibilityDataset(Dataset):
    def __init__(self, image_paths, mode='test', arch='resnet18'):
        if 'resnet' in arch:
            arch = 'resnet'
        self.transform = data_transforms[mode][arch]
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image

class TrackletLegibilityDataset(Dataset):
    def __init__(self, annotations_file, parent_dir, mode='test', arch='resnet18'):
        if 'resnet' in arch:
            arch = 'resnet'
        self.transform = data_transforms[mode][arch]
        with open(annotations_file, 'r') as f:
            self.tracklet_labels = json.load(f)
        tracklets = self.tracklet_labels.keys()
        self.image_paths = []
        for track in tracklets:
            tracklet_dir = os.path.join(parent_dir, track)
            images = os.listdir(tracklet_dir)
            for im in images:
                label = int(self.tracklet_labels[track])
                label = 1 if label > 0 else 0
                self.image_paths.append([os.path.join(tracklet_dir, im), track, label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, track, label = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, track, label


class DigitCountDataset(Dataset):
    """Dataset for digit count classification (1-digit vs 2-digit jersey numbers).

    Reads ground truth JSON (tracklet_id -> jersey_number) and walks tracklet
    image folders. Label: 0 = 1-digit (0-9), 1 = 2-digit (10-99).
    """
    def __init__(self, gt_file, img_dir, mode='train', isBalanced=False, arch='resnet34', sample_fraction=1.0):
        if 'resnet' in arch:
            arch = 'resnet'
        self.transform = data_transforms[mode][arch]
        with open(gt_file, 'r') as f:
            gt = json.load(f)

        self.image_paths = []
        self.labels = []
        self.img_names = []

        for tracklet_id, jersey_number in gt.items():
            jersey_number = int(jersey_number)
            if jersey_number < 0:
                continue
            digit_label = 0 if jersey_number < 10 else 1
            tracklet_dir = os.path.join(img_dir, tracklet_id)
            if not os.path.isdir(tracklet_dir):
                continue
            for img_name in os.listdir(tracklet_dir):
                self.image_paths.append(os.path.join(tracklet_dir, img_name))
                self.labels.append(digit_label)
                self.img_names.append(f"{tracklet_id}/{img_name}")

        one_digit = sum(1 for l in self.labels if l == 0)
        two_digit = sum(1 for l in self.labels if l == 1)
        print(f"DigitCount dataset: 1-digit={one_digit}, 2-digit={two_digit}, total={len(self.labels)}")

        if isBalanced:
            indices_0 = [i for i, l in enumerate(self.labels) if l == 0]
            indices_1 = [i for i, l in enumerate(self.labels) if l == 1]
            min_count = min(len(indices_0), len(indices_1))
            np.random.shuffle(indices_0)
            np.random.shuffle(indices_1)
            keep = indices_0[:min_count] + indices_1[:min_count]
            self.image_paths = [self.image_paths[i] for i in keep]
            self.labels = [self.labels[i] for i in keep]
            self.img_names = [self.img_names[i] for i in keep]
            print(f"Balanced to {min_count} per class, total={len(self.labels)}")

        if sample_fraction < 1.0:
            n_keep = max(1, int(len(self.labels) * sample_fraction))
            indices = list(range(len(self.labels)))
            np.random.shuffle(indices)
            indices = indices[:n_keep]
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.img_names = [self.img_names[i] for i in indices]
            print(f"Sampled {sample_fraction*100:.0f}%: using {len(self.labels)} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], self.img_names[idx]


class JerseyNumberLegibilityDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train', isBalanced=False, arch='resnet18'):
        if 'resnet' in arch:
            arch = 'resnet'
        self.transform = data_transforms[mode][arch]
        self.img_labels = pd.read_csv(annotations_file)
        if isBalanced:
            legible =self.img_labels[self.img_labels.iloc[:,1]==1]
            count_legible = len(legible)
            illegible = self.img_labels[self.img_labels.iloc[:,1]==0]
            print(count_legible, len(illegible))
            if len(illegible) > count_legible:
                illegible = illegible.sample(n=count_legible)
            self.img_labels = pd.concat([legible, illegible])
            print(f"Balanced dataset: legibles = {count_legible} all = {len(self.img_labels)}")
        else:
            legible = self.img_labels[self.img_labels.iloc[:, 1] == 1]
            count_legible = len(legible)
            print(f"As-is dataset: legibles = {count_legible} all = {len(self.img_labels)}")

        self.img_dir = img_dir


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label, self.img_labels.iloc[idx, 0]

