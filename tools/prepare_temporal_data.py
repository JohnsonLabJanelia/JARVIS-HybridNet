#!/usr/bin/env python3
"""
Prepare temporal training data efficiently:
- Val: extract real HybridNet predictions (small set, ~5 min)
- Train: generate synthetic noisy predictions from GT (instant)

The synthetic approach adds noise matching the real error profile,
creating diverse training data for the temporal transformer.
"""
import argparse
import os
import sys
import re
import json
import numpy as np
from tqdm import tqdm

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jarvis.config.project_manager import ProjectManager
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.hybridnet.hybridnet import HybridNet


def extract_real_predictions(hybridnet, dataset, cfg, output_dir):
    """Extract real HybridNet predictions for val set."""
    os.makedirs(output_dir, exist_ok=True)
    model = hybridnet.model
    model.eval()

    for idx in tqdm(range(len(dataset)), desc=f'Extracting real predictions'):
        sample = dataset[idx]
        file_name = dataset.imgs[dataset.image_ids[idx]]['file_name']
        info_split = file_name.split('/')
        session = info_split[0]
        frame_name = info_split[-1].split('.')[0]

        with torch.no_grad():
            imgs = torch.tensor(sample[0]).unsqueeze(0).permute(
                0, 1, 4, 2, 3).float().cuda()
            centerHM = torch.tensor(sample[2]).unsqueeze(0).cuda()
            center3D = torch.tensor(sample[3]).unsqueeze(0).cuda()
            cameraMatrices = torch.tensor(sample[5]).unsqueeze(0).cuda()
            intrinsicMatrices = torch.tensor(sample[6]).unsqueeze(0).cuda()
            distCoeffs = torch.tensor(sample[7]).unsqueeze(0).cuda()
            img_size = torch.tensor(cfg.DATASET.IMAGE_SIZE).cuda()

            outputs = model(imgs, img_size, centerHM, center3D,
                           cameraMatrices, intrinsicMatrices, distCoeffs)
            points3D = outputs[2][0].cpu().numpy()
            confidences = outputs[3][0].cpu().numpy()

        np.savez(os.path.join(output_dir, f'{session}_Frame_{frame_name.split("_")[-1]}.npz'),
                 points3D=points3D.astype(np.float32),
                 confidences=confidences.astype(np.float32),
                 gt_keypoints3D=np.array(sample[1], dtype=np.float32),
                 session=session, frame_name=frame_name)


def generate_synthetic_predictions(dataset, cfg, output_dir, noise_std=12.0,
                                   outlier_prob=0.05, outlier_scale=50.0,
                                   num_augmentations=3):
    """
    Generate synthetic noisy predictions from GT keypoints.
    Creates multiple noise augmentations per frame for more training data.

    Noise model:
    - Base: Gaussian noise with std matching observed MPJPE
    - Outliers: occasional large errors (simulating detection failures)
    - Confidence: inversely correlated with error magnitude
    """
    os.makedirs(output_dir, exist_ok=True)
    total = 0

    for aug_idx in range(num_augmentations):
        for idx in tqdm(range(len(dataset)),
                       desc=f'Generating synthetic (aug {aug_idx+1}/{num_augmentations})'):
            file_name = dataset.imgs[dataset.image_ids[idx]]['file_name']
            info_split = file_name.split('/')
            session = info_split[0]
            frame_name = info_split[-1].split('.')[0]

            gt_kp = np.array(dataset.keypoints3D[idx], dtype=np.float32)
            num_joints = gt_kp.shape[0]

            # Generate noise
            noise = np.random.randn(num_joints, 3).astype(np.float32) * noise_std

            # Add outliers
            outlier_mask = np.random.random(num_joints) < outlier_prob
            noise[outlier_mask] *= outlier_scale / noise_std

            # Apply noise only to valid keypoints
            valid = np.any(gt_kp != 0, axis=-1)
            pred_kp = gt_kp.copy()
            pred_kp[valid] += noise[valid]

            # Generate confidence scores (inversely correlated with error)
            errors = np.linalg.norm(noise, axis=-1)
            conf = np.clip(1.0 - errors / 50.0, 0.1, 1.0).astype(np.float32)
            conf[~valid] = 0.0
            # Add some randomness to confidence
            conf = np.clip(conf + np.random.randn(num_joints).astype(np.float32) * 0.1,
                          0.0, 1.0)

            suffix = f'_aug{aug_idx}' if aug_idx > 0 else ''
            fname = f'{session}_{frame_name}{suffix}.npz'
            np.savez(os.path.join(output_dir, fname),
                     points3D=pred_kp,
                     confidences=conf,
                     gt_keypoints3D=gt_kp,
                     session=session, frame_name=f'{frame_name}{suffix}')
            total += 1

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--weights', default=None,
                        help='HybridNet weights for val extraction')
    parser.add_argument('--output', required=True)
    parser.add_argument('--noise_std', type=float, default=12.0,
                        help='Gaussian noise std in mm')
    parser.add_argument('--num_aug', type=int, default=3,
                        help='Number of noise augmentations per frame')
    args = parser.parse_args()

    project = ProjectManager()
    project.load(args.project)
    cfg = project.get_cfg()

    # Generate synthetic training data (fast)
    print("="*50)
    print("  Generating synthetic training predictions")
    print("="*50)
    train_dataset = Dataset3D(cfg=cfg, set='train')
    print(f"  {len(train_dataset)} training frames")
    n = generate_synthetic_predictions(
        train_dataset, cfg, os.path.join(args.output, 'train'),
        noise_std=args.noise_std, num_augmentations=args.num_aug)
    print(f"  Generated {n} synthetic training samples")

    # Extract real val predictions (slow but small)
    print("\n" + "="*50)
    print("  Extracting real validation predictions")
    print("="*50)

    if args.weights is None:
        weights_path = os.path.join(
            cfg.PARENT_DIR, 'projects', cfg.PROJECT_NAME, 'models',
            'HybridNet', 'Run_20260314-232403', 'HybridNet-medium_final.pth')
    else:
        weights_path = args.weights

    hybridnet = HybridNet('inference', cfg, weights=weights_path)
    val_dataset = Dataset3D(cfg=cfg, set='val')
    print(f"  {len(val_dataset)} validation frames")
    extract_real_predictions(hybridnet, val_dataset, cfg,
                            os.path.join(args.output, 'val'))

    print("\nDone! Data ready for temporal training.")
    print(f"  Train: {args.output}/train/")
    print(f"  Val:   {args.output}/val/")


if __name__ == '__main__':
    main()
