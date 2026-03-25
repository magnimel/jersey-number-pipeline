# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
import json
import sys
import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = './pose/ViTPose/'
sys.path.append(str(ROOT))  # add ROOT to PATH

from argparse import ArgumentParser

from xtcocotools.coco import COCO
import mmcv

from mmpose.apis import init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose


class LoadImage:
    """A simple pipeline to load an image from path or numpy array."""

    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        if isinstance(results['img_or_path'], str):
            results['image_file'] = results['img_or_path']
            img = mmcv.imread(results['img_or_path'], self.color_type,
                              self.channel_order)
        elif isinstance(results['img_or_path'], np.ndarray):
            results['image_file'] = ''
            if self.channel_order == 'rgb':
                img = mmcv.bgr2rgb(results['img_or_path'])
            else:
                img = results['img_or_path'].copy()
        else:
            raise ValueError('img_or_path must be a filepath or numpy array')
        results['img'] = img
        return results


def _xywh2cs(cfg, x, y, w, h):
    """Convert xywh bounding box to center and scale."""
    input_size = cfg.data_cfg['image_size']
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    scale = scale * 1.25
    return center, scale


def build_test_pipeline(cfg):
    """Build the test preprocessing pipeline."""
    channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)]
    test_pipeline.extend(cfg.test_pipeline[1:])
    return Compose(test_pipeline)


class PoseDataset(Dataset):
    """Map-style dataset that loads & preprocesses one crop per __getitem__.

    Workers in the DataLoader call __getitem__ in parallel, so image I/O
    and the augmentation pipeline run on 2 CPU processes concurrently while
    the main process drives the GPU.
    """

    def __init__(self, cfg, coco, img_root, dataset, flip_pairs, pipeline):
        self.cfg = cfg
        self.img_root = img_root
        self.dataset = dataset
        self.flip_pairs = flip_pairs
        self.pipeline = pipeline

        # Build a flat index: list of (image_id, file_name, bbox)
        self.items = []
        for image_id in coco.imgs:
            image = coco.loadImgs(image_id)[0]
            ann_ids = coco.getAnnIds(image_id)
            if not ann_ids:
                continue
            ann = coco.anns[ann_ids[0]]
            self.items.append((
                image_id,
                image['file_name'],
                ann['bbox'],  # [x, y, w, h]
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_id, file_name, bbox = self.items[idx]
        image_path = os.path.join(self.img_root, file_name)
        x, y, w, h = bbox
        center, scale = _xywh2cs(self.cfg, x, y, w, h)

        data = {
            'img_or_path': image_path,
            'center': center,
            'scale': scale,
            'bbox_score': 1.0,
            'bbox_id': 0,
            'dataset': self.dataset,
            'joints_3d': np.zeros(
                (self.cfg.data_cfg['num_joints'], 3), dtype=np.float32),
            'joints_3d_visible': np.zeros(
                (self.cfg.data_cfg['num_joints'], 3), dtype=np.float32),
            'rotation': 0,
            'ann_info': {
                'image_size': np.array(self.cfg.data_cfg['image_size']),
                'num_joints': self.cfg.data_cfg['num_joints'],
                'flip_pairs': self.flip_pairs,
            },
        }

        processed = self.pipeline(data)

        # Return the tensor + metadata separately so collation works
        return {
            'img': processed['img'],
            'img_metas': processed['img_metas'].data,
            'image_id': image_id,
            'file_name': file_name,
        }


def pose_collate_fn(batch):
    """Custom collate: stack img tensors, keep metas as list."""
    imgs = torch.stack([item['img'] for item in batch], dim=0)
    img_metas = [item['img_metas'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    file_names = [item['file_name'] for item in batch]
    return {
        'img': imgs,
        'img_metas': img_metas,
        'image_ids': image_ids,
        'file_names': file_names,
    }


def run_dataloader_inference(pose_model, dataloader, total, device,
                             use_fp16=False, out_json=''):
    """GPU inference loop driven by a multi-worker DataLoader.

    While the GPU processes batch N, the DataLoader workers are already
    loading and preprocessing batch N+1 on the CPU — no idle time.
    Results are flushed to disk incrementally.
    """
    total_written = 0
    results_buf = []
    flush_every = 2000

    out_fp = None
    if out_json:
        out_fp = open(out_json, 'w')
        out_fp.write('{"pose_results": [\n')

    def flush_results():
        nonlocal total_written
        if not out_fp or not results_buf:
            return
        for r in results_buf:
            if total_written > 0:
                out_fp.write(',\n')
            json.dump(r, out_fp)
            total_written += 1
        out_fp.flush()
        results_buf.clear()

    pbar = tqdm(total=total, desc="Pose inference")

    for batch in dataloader:
        imgs = batch['img'].to(device, non_blocking=True)
        img_metas = batch['img_metas']
        image_ids = batch['image_ids']
        file_names = batch['file_names']

        with torch.no_grad():
            if use_fp16:
                with torch.cuda.amp.autocast():
                    output = pose_model(
                        img=imgs,
                        img_metas=img_metas,
                        return_loss=False,
                        return_heatmap=False)
            else:
                output = pose_model(
                    img=imgs,
                    img_metas=img_metas,
                    return_loss=False,
                    return_heatmap=False)

        keypoint_preds = output['preds']  # numpy array (N, num_joints, 3)

        for j in range(len(image_ids)):
            results_buf.append({
                'img_name': file_names[j],
                'id': image_ids[j],
                'keypoints': keypoint_preds[j].tolist(),
            })

        pbar.update(len(image_ids))

        del imgs, output, keypoint_preds

        if len(results_buf) >= flush_every:
            flush_results()
            gc.collect()

    pbar.close()

    flush_results()
    if out_fp:
        out_fp.write('\n]}')
        out_fp.close()

    print(f"Wrote {total_written} results to {out_json}")
    return total_written


def main():
    """Run batched pose estimation on images with bounding box annotations."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file', type=str, default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--out-json', type=str, default='',
        help='Json file containing results.')
    parser.add_argument(
        '--show', action='store_true', default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root', type=str, default='',
        help='Root of the output img file. '
             'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for inference (default: 64). '
             'Reduce if running out of GPU memory.')
    parser.add_argument(
        '--num-workers', type=int, default=2,
        help='Number of DataLoader workers for parallel image loading.')
    parser.add_argument(
        '--fp16', action='store_true', default=False,
        help='Use FP16 mixed precision for faster inference on GPU')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3,
        help='Keypoint score threshold')
    parser.add_argument(
        '--radius', type=int, default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness', type=int, default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    coco = COCO(args.json_file)

    # Build the pose model
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset_name = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        flip_pairs = []
    else:
        dataset_info = DatasetInfo(dataset_info)
        flip_pairs = dataset_info.flip_pairs

    cfg = pose_model.cfg
    device = next(pose_model.parameters()).device

    # Build pipeline & dataset
    pipeline = build_test_pipeline(cfg)
    pose_dataset = PoseDataset(
        cfg, coco, args.img_root, dataset_name, flip_pairs, pipeline)

    total = len(pose_dataset)
    print(f"Processing {total} images "
          f"(batch_size={args.batch_size}, workers={args.num_workers}, "
          f"fp16={args.fp16})...")

    # DataLoader: workers load+preprocess images while GPU runs inference
    dataloader = DataLoader(
        pose_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pose_collate_fn,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=False,
    )

    # Run pipelined inference
    run_dataloader_inference(
        pose_model, dataloader, total, device,
        use_fp16=args.fp16, out_json=args.out_json)

    # Optional visualization
    if args.show or args.out_img_root != '':
        print("Generating visualizations...")
        with open(args.out_json, 'r') as fp:
            saved = json.load(fp)
        for i, result in enumerate(tqdm(saved['pose_results'],
                                        desc="Visualization")):
            image_name = os.path.join(args.img_root, result['img_name'])
            pose_results = [{
                'keypoints': np.array(result['keypoints']),
            }]

            if args.out_img_root == '':
                out_file = None
            else:
                os.makedirs(args.out_img_root, exist_ok=True)
                out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')

            vis_pose_result(
                pose_model,
                image_name,
                pose_results,
                dataset=dataset_name,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=args.show,
                out_file=out_file)


if __name__ == '__main__':
    main()