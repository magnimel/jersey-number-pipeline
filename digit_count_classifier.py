from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from jersey_number_dataset import DigitCountDataset, UnlabelledJerseyNumberLegibilityDataset
from networks import DigitCountClassifier

import time
import copy
import argparse
import os
import numpy as np
from tqdm import tqdm

# SAM is optional - only imported when --sam flag is used


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, _ in dataloaders[phase]:
                labels = labels.reshape(-1, 1)
                labels = labels.type(torch.FloatTensor)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs.round()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


def train_model_with_sam(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, _ in dataloaders[phase]:
                labels = labels.reshape(-1, 1)
                labels = labels.type(torch.FloatTensor)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs.round()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.first_step(zero_grad=True)

                        criterion(model(inputs), labels).backward()
                        optimizer.second_step(zero_grad=True)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloaders, dataset_sizes, subset, device):
    model.eval()
    running_corrects = 0
    predictions = []
    gt = []
    img_names = []

    for inputs, labels, names in tqdm(dataloaders[subset]):
        inputs = inputs.to(device)
        labels_reshaped = torch.tensor(labels).reshape(-1, 1).type(torch.FloatTensor).to(device)

        torch.set_grad_enabled(False)
        outputs = model(inputs)
        preds = outputs.round()
        running_corrects += torch.sum(preds == labels_reshaped.data)
        gt += labels_reshaped.data.detach().cpu().numpy().flatten().tolist()
        predictions += preds.detach().cpu().numpy().flatten().tolist()
        img_names += list(names)

    epoch_acc = running_corrects.double() / dataset_sizes[subset]

    total, TN, TP, FP, FN = 0, 0, 0, 0, 0
    for i, true_value in enumerate(gt):
        predicted_two = predictions[i] == 1
        if true_value == 0 and not predicted_two:
            TN += 1
        elif true_value != 0 and predicted_two:
            TP += 1
        elif true_value == 0 and predicted_two:
            FP += 1
        elif true_value != 0 and not predicted_two:
            FN += 1
        total += 1

    print(f'Correct {TP+TN} out of {total}. Accuracy {100*(TP+TN)/total}%.')
    print(f'TP(2-digit correct)={TP}, TN(1-digit correct)={TN}, FP={FP}, FN={FN}')
    if TP + FP > 0:
        Pr = TP / (TP + FP)
        Recall = TP / (TP + FN)
        print(f"Precision={Pr}, Recall={Recall}")
        if Pr + Recall > 0:
            print(f"F1={2*Pr*Recall/(Pr+Recall)}")
    print(f"Accuracy {subset}: {epoch_acc}")

    return epoch_acc


def run(image_paths, model_path, threshold=0.5, batch_size=512, num_workers=2):
    """Run digit count inference on a list of image paths.
    Returns list of predictions: 0 = 1-digit, 1 = 2-digit."""
    dataset = UnlabelledJerseyNumberLegibilityDataset(image_paths, arch='resnet34')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers,
                                              pin_memory=True if torch.cuda.is_available() else False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cudnn.benchmark = True

    state_dict = torch.load(model_path, map_location=device)
    model = DigitCountClassifier()
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)

            if threshold > 0:
                outputs = (outputs > threshold).float()
            else:
                outputs = outputs.float()
            preds = outputs.cpu().detach().numpy()
            flattened_preds = preds.flatten().tolist()
            results += flattened_preds

    return results


def run_batch_tracklets(tracklet_dict, model_path, threshold=0.5, batch_size=512, num_workers=2):
    """Process multiple tracklets efficiently in batches.

    Args:
        tracklet_dict: Dictionary mapping tracklet_id -> list of image paths
        model_path: Path to model weights
        threshold: Classification threshold
        batch_size: Batch size for processing
        num_workers: Number of data loader workers

    Returns:
        Dictionary mapping tracklet_id -> list of predictions (0=1-digit, 1=2-digit)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cudnn.benchmark = True

    state_dict = torch.load(model_path, map_location=device)
    model = DigitCountClassifier()
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Flatten all images with tracklet tracking
    all_images = []
    tracklet_indices = {}
    current_idx = 0

    for tracklet_id, image_paths in tracklet_dict.items():
        start_idx = current_idx
        end_idx = current_idx + len(image_paths)
        tracklet_indices[tracklet_id] = (start_idx, end_idx)
        all_images.extend(image_paths)
        current_idx = end_idx

    dataset = UnlabelledJerseyNumberLegibilityDataset(all_images, arch='resnet34')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers,
                                              pin_memory=True if torch.cuda.is_available() else False)

    print(f"Processing {len(all_images)} images in batches of {batch_size}")

    all_results = []
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Digit count classification"):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)

            if threshold > 0:
                outputs = (outputs > threshold).float()
            else:
                outputs = outputs.float()

            preds = outputs.cpu().detach().numpy()
            all_results += preds.flatten().tolist()

    # Split results back into tracklets
    results_dict = {}
    for tracklet_id, (start_idx, end_idx) in tracklet_indices.items():
        results_dict[tracklet_id] = all_results[start_idx:end_idx]

    return results_dict


def get_majority_digit_count(tracklet_predictions):
    """For a tracklet, return the majority vote: 1 or 2 digits.

    Args:
        tracklet_predictions: list of per-image predictions (0=1-digit, 1=2-digit)

    Returns:
        1 if majority says 1-digit, 2 if majority says 2-digit
    """
    if not tracklet_predictions:
        return 2  # default to 2-digit
    two_digit_count = sum(1 for p in tracklet_predictions if p == 1)
    if two_digit_count > len(tracklet_predictions) / 2:
        return 2
    return 1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help='train model from pretrained ImageNet weights')
    parser.add_argument('--sam', action='store_true', help='Use Sharpness-Aware Minimization during training')
    parser.add_argument('--finetune', action='store_true', help='load custom weights for further training')
    parser.add_argument('--data', help='SoccerNet root dir (contains train/val/test with gt JSONs and images)')
    parser.add_argument('--trained_model_path', help='trained model to use for testing or to load for finetuning')
    parser.add_argument('--new_trained_model_path', help='path to save newly trained model')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='fraction of data to use (0.0-1.0) for testing')

    args = parser.parse_args()

    if args.train or args.finetune:
        # Training mode: use GT to create digit-count labels
        train_gt = os.path.join(args.data, 'train', 'train_gt.json')
        train_img_dir = os.path.join(args.data, 'train', 'images')
        val_gt = os.path.join(args.data, 'val', 'val_gt.json')
        val_img_dir = os.path.join(args.data, 'val', 'images')

        train_dataset = DigitCountDataset(train_gt, train_img_dir, mode='train', isBalanced=True, sample_fraction=args.sample_fraction)
        val_dataset = DigitCountDataset(val_gt, val_img_dir, mode='val', sample_fraction=args.sample_fraction)

        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2),
            'val': torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        }
        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

        model_ft = DigitCountClassifier(finetune=args.finetune)

        if args.finetune and args.trained_model_path:
            state_dict = torch.load(args.trained_model_path, map_location=device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            model_ft.load_state_dict(state_dict)

        model_ft = model_ft.to(device)
        criterion = nn.BCELoss()

        if args.sam:
            from sam.sam import SAM
            base_optimizer = torch.optim.SGD
            optimizer_ft = SAM(model_ft.parameters(), base_optimizer, lr=0.001, momentum=0.9)
            model_ft = train_model_with_sam(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, device, num_epochs=10)
        else:
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
            model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, device, num_epochs=15)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        if args.new_trained_model_path:
            save_path = args.new_trained_model_path
        else:
            os.makedirs('./experiments', exist_ok=True)
            save_path = f"./experiments/digit_count_resnet34_{timestr}.pth"

        torch.save(model_ft.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    else:
        # Test mode
        test_gt = os.path.join(args.data, 'test', 'test_gt.json')
        test_img_dir = os.path.join(args.data, 'test', 'images')

        test_dataset = DigitCountDataset(test_gt, test_img_dir, mode='test')
        dataloaders = {
            'test': torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
        }
        dataset_sizes = {'test': len(test_dataset)}

        model_ft = DigitCountClassifier()
        state_dict = torch.load(args.trained_model_path, map_location=device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        model_ft.load_state_dict(state_dict)
        model_ft = model_ft.to(device)

        test_model(model_ft, dataloaders, dataset_sizes, 'test', device)
