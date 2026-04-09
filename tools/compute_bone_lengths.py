#!/usr/bin/env python3
"""
Compute reference bone length statistics from training ground truth.
Saves bone_stats.json to the project directory for use with BoneLengthLoss.

Usage:
    python tools/compute_bone_lengths.py --project mouseJan30
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jarvis.config.project_manager import ProjectManager
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.hybridnet.physics_loss import (
    skeleton_to_index_pairs,
    compute_bone_stats,
    save_bone_stats,
)


def main():
    parser = argparse.ArgumentParser(
        description='Compute bone length statistics from training GT'
    )
    parser.add_argument('--project', required=True, help='JARVIS project name')
    parser.add_argument('--output', default=None,
                        help='Output path (default: <project_dir>/bone_stats.json)')
    args = parser.parse_args()

    project = ProjectManager()
    project.load(args.project)
    cfg = project.get_cfg()

    print(f"Loading training dataset for project '{args.project}'...")
    dataset = Dataset3D(cfg=cfg, set='train')
    print(f"  {len(dataset)} training samples")
    print(f"  {cfg.KEYPOINTDETECT.NUM_JOINTS} keypoints")

    keypoint_names = list(cfg.KEYPOINT_NAMES)
    skeleton = list(cfg.SKELETON)
    bone_pairs = skeleton_to_index_pairs(keypoint_names, skeleton)
    print(f"  {len(bone_pairs)} skeleton bones")

    print("Computing bone length statistics...")
    stats = compute_bone_stats(dataset.keypoints3D, bone_pairs)

    output_path = args.output
    if output_path is None:
        project_dir = os.path.join(cfg.PARENT_DIR, 'projects', cfg.PROJECT_NAME)
        output_path = os.path.join(project_dir, 'bone_stats.json')

    save_bone_stats(stats, bone_pairs, keypoint_names, skeleton, output_path)
    print(f"\nSaved bone stats to: {output_path}")

    print("\nBone length summary:")
    print(f"  {'Bone':<30} {'Mean (mm)':>10} {'Std (mm)':>10} {'Count':>8}")
    print("  " + "-" * 62)
    for b_idx, (i, j) in enumerate(bone_pairs):
        name = f"{keypoint_names[i]}-{keypoint_names[j]}"
        s = stats[b_idx]
        print(f"  {name:<30} {s['mean']:>10.2f} {s['std']:>10.2f} {s['count']:>8}")


if __name__ == '__main__':
    main()
