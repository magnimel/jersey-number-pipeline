# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
import json
import sys
import numpy as np
import torch
from tqdm import tqdm

ROOT = './pose/ViTPose/'
sys.path.append(str(ROOT))  # add ROOT to PATH

from argparse import ArgumentParser

from xtcocotools.coco import COCO
import mmcv
from mmcv.parallel import collate, scatter

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


def prepare_data_items(cfg, coco, img_root, dataset, flip_pairs):
    """Prepare data dicts for all images/annotations.

    Returns a list of (data_dict, image_id, file_name) tuples.
    """
    img_keys = list(coco.imgs.keys())
    all_items = []

    for image_id in img_keys:
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        if not ann_ids:
            continue

        # Use first annotation per image (matches original pose_results[0])
        ann = coco.anns[ann_ids[0]]
        x, y, w, h = ann['bbox']
        center, scale = _xywh2cs(cfg, x, y, w, h)

        data = {
            'img_or_path': image_name,
            'center': center,
            'scale': scale,
            'bbox_score': 1.0,
            'bbox_id': 0,
            'dataset': dataset,
            'joints_3d': np.zeros(
                (cfg.data_cfg['num_joints'], 3), dtype=np.float32),
            'joints_3d_visible': np.zeros(
                (cfg.data_cfg['num_joints'], 3), dtype=np.float32),
            'rotation': 0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs,
            },
        }

        all_items.append((data, image_id, image['file_name']))

    return all_items


def run_batched_inference(pose_model, pipeline, all_items, device,
                          batch_size=64, use_fp16=False):
    """Run pose inference in batches across all images.

    Instead of processing each image individually (batch_size=1), this
    collects crops from multiple images and runs them through the model
    in larger batches for much better GPU utilization.
    """
    # Step 1: Preprocess all items through the pipeline
    print(f"Preprocessing {len(all_items)} items...")
    processed = []
    for data, image_id, file_name in tqdm(all_items, desc="Preprocessing"):
        proc = pipeline(data)
        processed.append((proc, image_id, file_name))

    # Step 2: Run inference in batches
    results = []
    num_batches = (len(processed) + batch_size - 1) // batch_size
    print(f"Running inference: {num_batches} batches (batch_size={batch_size})...")

    for i in tqdm(range(0, len(processed), batch_size), desc="Batch inference"):
        batch_items = processed[i:i + batch_size]
        batch_data_list = [item[0] for item in batch_items]
        batch_meta = [(item[1], item[2]) for item in batch_items]

        batch = collate(batch_data_list, samples_per_gpu=len(batch_data_list))
        batch = scatter(batch, [device])[0]

        with torch.no_grad():
            if use_fp16:
                with torch.cuda.amp.autocast():
                    output = pose_model(
                        img=batch['img'],
                        img_metas=batch['img_metas'],
                        return_loss=False,
                        return_heatmap=False)
            else:
                output = pose_model(
                    img=batch['img'],
                    img_metas=batch['img_metas'],
                    return_loss=False,
                    return_heatmap=False)

        keypoint_preds = output['preds']  # (N, num_joints, 3)

        for j, (image_id, file_name) in enumerate(batch_meta):
            results.append({
                'img_name': file_name,
                'id': image_id,
                'keypoints': keypoint_preds[j].tolist(),
            })

    return results


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

    # Build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
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

    # Prepare all data items
    all_items = prepare_data_items(
        cfg, coco, args.img_root, dataset, flip_pairs)
    print(f"Processing {len(all_items)} images...")

    # Build preprocessing pipeline
    pipeline = build_test_pipeline(cfg)

    # Run batched inference
    results = run_batched_inference(
        pose_model, pipeline, all_items, device,
        batch_size=args.batch_size, use_fp16=args.fp16)

    # Optional visualization (per-image, only when explicitly requested)
    if args.show or args.out_img_root != '':
        print("Generating visualizations...")
        for i, result in enumerate(tqdm(results, desc="Visualization")):
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
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=args.show,
                out_file=out_file)

    # Save results to JSON
    if args.out_json != '':
        with open(args.out_json, 'w') as fp:
            json.dump({"pose_results": results}, fp)


if __name__ == '__main__':
    main()