#!/usr/bin/env python3
"""
Train the Temporal Refinement Transformer and evaluate it.

Usage:
    python tools/train_temporal.py --project mouseJan30 \
        --predictions predictions/mouseJan30 --epochs 100
"""
import argparse
import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jarvis.config.project_manager import ProjectManager
from jarvis.hybridnet.temporal_trainer import TemporalTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--predictions', required=True,
                        help='Dir with train/ and val/ .npz predictions')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--max_gap', type=int, default=20)
    args = parser.parse_args()

    project = ProjectManager()
    project.load(args.project)
    cfg = project.get_cfg()

    # Override config with CLI args
    cfg.defrost()
    cfg.TEMPORAL.PREDICTIONS_DIR = args.predictions
    cfg.TEMPORAL.NUM_EPOCHS = args.epochs
    cfg.TEMPORAL.BATCH_SIZE = args.batch_size
    cfg.TEMPORAL.WINDOW_SIZE = args.window
    cfg.TEMPORAL.MAX_LEARNING_RATE = args.lr
    cfg.TEMPORAL.STRIDE = args.stride
    cfg.TEMPORAL.MAX_GAP_FRAMES = args.max_gap
    cfg.freeze()

    trainer = TemporalTrainer(cfg, predictions_dir=args.predictions)
    trainer.train(num_epochs=args.epochs)

    # Final evaluation with detailed metrics
    print("\n" + "="*60)
    print("  FINAL EVALUATION")
    print("="*60)

    from jarvis.dataset.dataset_temporal import TemporalDataset
    from torch.utils.data import DataLoader

    val_dataset = TemporalDataset(
        args.predictions, window_size=args.window,
        stride=args.window, max_gap_frames=args.max_gap, split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)

    trainer.model.eval()
    all_raw_errors = []
    all_refined_errors = []
    all_raw_bone = []
    all_refined_bone = []
    inference_times = []

    from jarvis.hybridnet.physics_loss import skeleton_to_index_pairs
    bone_pairs = skeleton_to_index_pairs(
        list(cfg.KEYPOINT_NAMES), list(cfg.SKELETON))

    with torch.no_grad():
        for batch in val_loader:
            pred_kp = batch['pred_keypoints'].cuda()
            pred_conf = batch['pred_confidences'].cuda()
            gt_kp = batch['gt_keypoints'].cuda()
            valid_mask = batch['valid_mask'].cuda()
            gt_valid = batch['gt_valid'].cuda()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            refined = trainer.model(pred_kp, pred_conf, valid_mask)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) / pred_kp.shape[0])

            # Per-frame per-joint errors
            for b in range(pred_kp.shape[0]):
                for t in range(pred_kp.shape[1]):
                    if not valid_mask[b, t]:
                        continue
                    for j in range(pred_kp.shape[2]):
                        if not gt_valid[b, t, j]:
                            continue
                        raw_err = torch.norm(
                            pred_kp[b,t,j] - gt_kp[b,t,j]).item()
                        ref_err = torch.norm(
                            refined[b,t,j] - gt_kp[b,t,j]).item()
                        all_raw_errors.append(raw_err)
                        all_refined_errors.append(ref_err)

            # Bone lengths
            for b_idx, (a, b) in enumerate(bone_pairs):
                for bi in range(pred_kp.shape[0]):
                    for t in range(pred_kp.shape[1]):
                        if not valid_mask[bi, t]:
                            continue
                        raw_bl = torch.norm(
                            pred_kp[bi,t,a] - pred_kp[bi,t,b]).item()
                        ref_bl = torch.norm(
                            refined[bi,t,a] - refined[bi,t,b]).item()
                        all_raw_bone.append((b_idx, raw_bl))
                        all_refined_bone.append((b_idx, ref_bl))

    raw_errors = np.array(all_raw_errors)
    ref_errors = np.array(all_refined_errors)

    print(f"\n  Raw HybridNet MPJPE:     {raw_errors.mean():.2f} mm "
          f"(median {np.median(raw_errors):.2f})")
    print(f"  Refined MPJPE:           {ref_errors.mean():.2f} mm "
          f"(median {np.median(ref_errors):.2f})")
    print(f"  Improvement:             {raw_errors.mean() - ref_errors.mean():.2f} mm "
          f"({(1 - ref_errors.mean()/raw_errors.mean())*100:.1f}%)")
    print(f"  Temporal inference:       "
          f"{np.mean(inference_times)*1000:.2f} ms/batch")

    # Bone consistency
    raw_bone_dict = {}
    ref_bone_dict = {}
    for b_idx, bl in all_raw_bone:
        raw_bone_dict.setdefault(b_idx, []).append(bl)
    for b_idx, bl in all_refined_bone:
        ref_bone_dict.setdefault(b_idx, []).append(bl)
    raw_bone_stds = [np.std(v) for v in raw_bone_dict.values()]
    ref_bone_stds = [np.std(v) for v in ref_bone_dict.values()]
    print(f"  Raw bone length std:     {np.mean(raw_bone_stds):.2f} mm")
    print(f"  Refined bone length std: {np.mean(ref_bone_stds):.2f} mm")

    # Save results in same format as baseline
    kn = list(cfg.KEYPOINT_NAMES)
    # Build per-joint from refined errors (approximation - just use overall)
    results = {
        'mpjpe_mean': float(ref_errors.mean()),
        'mpjpe_median': float(np.median(ref_errors)),
        'mpjpe_std': float(ref_errors.std()),
        'mpjpe_p95': float(np.percentile(ref_errors, 95)),
        'inference_ms': float(np.mean(inference_times) * 1000),
        'fps': float(1.0 / (196.7e-3 + np.mean(inference_times))),
        'per_frame_errors': [float(x) for x in ref_errors],
        'inference_times': [float(x) for x in inference_times],
        'raw_mpjpe_mean': float(raw_errors.mean()),
        'improvement_mm': float(raw_errors.mean() - ref_errors.mean()),
        'improvement_pct': float((1 - ref_errors.mean()/raw_errors.mean())*100),
        'bone_lengths': {},
        'per_joint': {},
    }
    for b_idx, (a, b) in enumerate(bone_pairs):
        bname = f"{kn[a]}-{kn[b]}"
        results['bone_lengths'][bname] = {
            'mean': float(np.mean(ref_bone_dict.get(b_idx, [0]))),
            'std': float(np.std(ref_bone_dict.get(b_idx, [0]))),
        }

    out_path = os.path.join(
        cfg.PARENT_DIR, 'projects', cfg.PROJECT_NAME, 'temporal_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
