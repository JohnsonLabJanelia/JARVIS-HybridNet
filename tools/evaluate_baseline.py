#!/usr/bin/env python3
"""
Evaluate baseline HybridNet and save per-frame metrics for comparison.
"""
import os, sys, time, json
import numpy as np
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jarvis.config.project_manager import ProjectManager
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.hybridnet.hybridnet import HybridNet
from jarvis.hybridnet.physics_loss import skeleton_to_index_pairs


def evaluate(hybridnet, dataset, cfg):
    model = hybridnet.model
    model.eval()

    per_joint_errors = []
    per_frame_errors = []
    bone_length_errors = []
    inference_times = []

    keypoint_names = list(cfg.KEYPOINT_NAMES)
    skeleton = list(cfg.SKELETON)
    bone_pairs = skeleton_to_index_pairs(keypoint_names, skeleton)

    for idx in tqdm(range(len(dataset)), desc='Evaluating'):
        sample = dataset[idx]
        with torch.no_grad():
            imgs = torch.tensor(sample[0]).unsqueeze(0).permute(0,1,4,2,3).float().cuda()
            keypoints_gt = np.array(sample[1])
            centerHM = torch.tensor(sample[2]).unsqueeze(0).cuda()
            center3D = torch.tensor(sample[3]).unsqueeze(0).cuda()
            cameraMatrices = torch.tensor(sample[5]).unsqueeze(0).cuda()
            intrinsicMatrices = torch.tensor(sample[6]).unsqueeze(0).cuda()
            distCoeffs = torch.tensor(sample[7]).unsqueeze(0).cuda()
            img_size = torch.tensor(cfg.DATASET.IMAGE_SIZE).cuda()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(imgs, img_size, centerHM, center3D,
                           cameraMatrices, intrinsicMatrices, distCoeffs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            inference_times.append(t1 - t0)

            points3D = outputs[2][0].cpu().numpy()
            confidences = outputs[3][0].cpu().numpy()

        # Per-joint error
        joint_errors = []
        for j in range(len(keypoint_names)):
            gt = keypoints_gt[j]
            if gt[0] != 0 or gt[1] != 0 or gt[2] != 0:
                err = np.linalg.norm(points3D[j] - gt)
                joint_errors.append((j, err))

        if joint_errors:
            per_joint_errors.append(joint_errors)
            frame_err = np.mean([e for _, e in joint_errors])
            per_frame_errors.append(frame_err)

        # Bone length consistency
        for i_b, (a, b) in enumerate(bone_pairs):
            pa, pb = points3D[a], points3D[b]
            if np.abs(pa).sum() > 0 and np.abs(pb).sum() > 0:
                bl = np.linalg.norm(pa - pb)
                bone_length_errors.append((i_b, bl))

    return {
        'per_joint_errors': per_joint_errors,
        'per_frame_errors': per_frame_errors,
        'bone_length_errors': bone_length_errors,
        'inference_times': inference_times,
        'keypoint_names': keypoint_names,
        'bone_pairs': bone_pairs,
    }


def print_results(results, label=""):
    errors = results['per_frame_errors']
    times = results['inference_times']
    keypoint_names = results['keypoint_names']

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {len(errors)}")
    print(f"  MPJPE (mean):   {np.mean(errors):.2f} mm")
    print(f"  MPJPE (median): {np.median(errors):.2f} mm")
    print(f"  MPJPE (std):    {np.std(errors):.2f} mm")
    print(f"  MPJPE (95th):   {np.percentile(errors, 95):.2f} mm")
    print(f"  Inference time: {np.mean(times)*1000:.1f} ms/frame "
          f"({1.0/np.mean(times):.0f} FPS)")

    # Per-joint breakdown
    joint_accum = {}
    for frame_joints in results['per_joint_errors']:
        for j, err in frame_joints:
            if j not in joint_accum:
                joint_accum[j] = []
            joint_accum[j].append(err)

    print(f"\n  {'Joint':<20} {'MPJPE (mm)':>10} {'Std':>8} {'N':>6}")
    print(f"  {'-'*48}")
    for j in sorted(joint_accum.keys()):
        errs = joint_accum[j]
        name = keypoint_names[j] if j < len(keypoint_names) else f"Joint_{j}"
        print(f"  {name:<20} {np.mean(errs):>10.2f} {np.std(errs):>8.2f} {len(errs):>6}")

    # Bone length std (jitter proxy)
    bone_accum = {}
    for b_idx, bl in results['bone_length_errors']:
        if b_idx not in bone_accum:
            bone_accum[b_idx] = []
        bone_accum[b_idx].append(bl)
    bone_stds = [np.std(v) for v in bone_accum.values() if len(v) > 1]
    if bone_stds:
        print(f"\n  Mean bone length std: {np.mean(bone_stds):.2f} mm "
              f"(lower = more consistent)")


def save_results(results, path):
    out = {
        'mpjpe_mean': float(np.mean(results['per_frame_errors'])),
        'mpjpe_median': float(np.median(results['per_frame_errors'])),
        'mpjpe_std': float(np.std(results['per_frame_errors'])),
        'mpjpe_p95': float(np.percentile(results['per_frame_errors'], 95)),
        'inference_ms': float(np.mean(results['inference_times']) * 1000),
        'fps': float(1.0 / np.mean(results['inference_times'])),
        'per_frame_errors': [float(x) for x in results['per_frame_errors']],
        'inference_times': [float(x) for x in results['inference_times']],
    }
    # Per-joint
    joint_accum = {}
    for frame_joints in results['per_joint_errors']:
        for j, err in frame_joints:
            if j not in joint_accum:
                joint_accum[j] = []
            joint_accum[j].append(err)
    out['per_joint'] = {
        results['keypoint_names'][j]: {
            'mean': float(np.mean(v)), 'std': float(np.std(v))
        }
        for j, v in joint_accum.items()
    }
    # Bone lengths
    bone_accum = {}
    for b_idx, bl in results['bone_length_errors']:
        if b_idx not in bone_accum:
            bone_accum[b_idx] = []
        bone_accum[b_idx].append(bl)
    kn = results['keypoint_names']
    bp = results['bone_pairs']
    out['bone_lengths'] = {
        f"{kn[a]}-{kn[b]}": {
            'mean': float(np.mean(bone_accum[i])),
            'std': float(np.std(bone_accum[i]))
        }
        for i, (a, b) in enumerate(bp) if i in bone_accum
    }
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved to {path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--weights', default='latest')
    parser.add_argument('--split', default='val')
    parser.add_argument('--output', default=None)
    parser.add_argument('--label', default='Baseline')
    args = parser.parse_args()

    project = ProjectManager()
    project.load(args.project)
    cfg = project.get_cfg()

    hybridnet = HybridNet('inference', cfg, weights=args.weights)
    dataset = Dataset3D(cfg=cfg, set=args.split)
    print(f"Dataset: {len(dataset)} samples ({args.split})")

    results = evaluate(hybridnet, dataset, cfg)
    print_results(results, args.label)

    if args.output:
        save_results(results, args.output)
