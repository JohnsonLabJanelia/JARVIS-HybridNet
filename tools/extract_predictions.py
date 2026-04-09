#!/usr/bin/env python3
"""
Extract HybridNet predictions for all training/validation frames.
Saves per-frame .npz files for use with the Temporal Refinement Transformer.

Usage:
    python tools/extract_predictions.py --project mouseJan30 \
        --weights latest --output predictions/mouseJan30
"""

import argparse
import os
import sys
import re
import numpy as np
from tqdm import tqdm

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jarvis.config.project_manager import ProjectManager
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.hybridnet.hybridnet import HybridNet


def extract_split(hybridnet, dataset, cfg, output_dir):
    """Run inference on all frames and save predictions."""
    os.makedirs(output_dir, exist_ok=True)

    model = hybridnet.model
    model.eval()

    saved = 0
    for idx in tqdm(range(len(dataset)), desc=f'Extracting to {output_dir}'):
        sample = dataset[idx]

        # Get frame info for naming
        file_name = dataset.imgs[dataset.image_ids[idx]]['file_name']
        info_split = file_name.split('/')
        session = info_split[0]
        frame_name = info_split[-1].split('.')[0]  # e.g. Frame_100

        with torch.no_grad():
            imgs = torch.tensor(sample[0]).unsqueeze(0).permute(
                0, 1, 4, 2, 3).float().cuda()
            keypoints_gt = sample[1]
            centerHM = torch.tensor(sample[2]).unsqueeze(0).cuda()
            center3D = torch.tensor(sample[3]).unsqueeze(0).cuda()
            cameraMatrices = torch.tensor(sample[5]).unsqueeze(0).cuda()
            intrinsicMatrices = torch.tensor(sample[6]).unsqueeze(0).cuda()
            distCoeffs = torch.tensor(sample[7]).unsqueeze(0).cuda()
            img_size = torch.tensor(cfg.DATASET.IMAGE_SIZE).cuda()

            outputs = model(imgs, img_size, centerHM, center3D,
                           cameraMatrices, intrinsicMatrices, distCoeffs)

            points3D = outputs[2][0].cpu().numpy()      # (24, 3)
            confidences = outputs[3][0].cpu().numpy()    # (24,)

        out_name = f'{session}_{frame_name}.npz'
        np.savez(os.path.join(output_dir, out_name),
                 points3D=points3D.astype(np.float32),
                 confidences=confidences.astype(np.float32),
                 gt_keypoints3D=np.array(keypoints_gt, dtype=np.float32),
                 session=session,
                 frame_name=frame_name)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(
        description='Extract HybridNet predictions for temporal training')
    parser.add_argument('--project', required=True, help='JARVIS project name')
    parser.add_argument('--weights', default='latest',
                        help='HybridNet weights path or "latest"')
    parser.add_argument('--output', required=True,
                        help='Output directory for predictions')
    args = parser.parse_args()

    project = ProjectManager()
    project.load(args.project)
    cfg = project.get_cfg()

    print(f"Loading HybridNet for project '{args.project}'...")
    hybridnet = HybridNet('inference', cfg, weights=args.weights)

    for split in ['train', 'val']:
        print(f"\nProcessing {split} split...")
        dataset = Dataset3D(cfg=cfg, set=split)
        print(f"  {len(dataset)} samples")
        output_dir = os.path.join(args.output, split)
        n = extract_split(hybridnet, dataset, cfg, output_dir)
        print(f"  Saved {n} prediction files to {output_dir}")

    print("\nDone! Predictions ready for temporal training.")
    print(f"Use --predictions_dir {args.output} with TemporalTrainer")


if __name__ == '__main__':
    main()
