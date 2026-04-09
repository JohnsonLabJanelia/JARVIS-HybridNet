"""
Physics-informed loss functions for JARVIS-HybridNet.
Provides bone length consistency constraints derived from training data.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn


class BoneLengthLoss(nn.Module):
    """
    Penalizes predicted bone lengths that deviate from reference statistics
    computed from training ground truth.

    Args:
        bone_pairs: list of (i, j) keypoint index tuples
        reference_lengths: tensor (B,) mean bone lengths in mm
        reference_stds: tensor (B,) std of bone lengths in mm
    """
    def __init__(self, bone_pairs, reference_lengths, reference_stds):
        super().__init__()
        self.bone_pairs = bone_pairs
        self.register_buffer('ref_lengths', reference_lengths)
        self.register_buffer('ref_stds', reference_stds)

    def forward(self, points3D):
        """
        Args:
            points3D: (batch, num_joints, 3) predicted keypoints in mm
        Returns:
            Scalar loss: mean normalized squared deviation of bone lengths
        """
        loss = torch.tensor(0.0, device=points3D.device)
        count = 0
        for b_idx, (i, j) in enumerate(self.bone_pairs):
            pi = points3D[:, i]  # (batch, 3)
            pj = points3D[:, j]  # (batch, 3)
            # Skip if either keypoint is at origin (invalid)
            valid = ((pi.abs().sum(-1) > 0) & (pj.abs().sum(-1) > 0))
            if valid.sum() == 0:
                continue
            pred_len = torch.norm(pi[valid] - pj[valid], dim=-1)
            deviation = (pred_len - self.ref_lengths[b_idx]) / (self.ref_stds[b_idx] + 1e-6)
            loss = loss + torch.mean(deviation ** 2)
            count += 1
        if count > 0:
            loss = loss / count
        return loss

    @classmethod
    def from_stats_file(cls, stats_path, keypoint_names, skeleton):
        """
        Build BoneLengthLoss from a bone_stats.json file.

        Args:
            stats_path: path to bone_stats.json
            keypoint_names: list of keypoint name strings
            skeleton: list of [nameA, nameB] bone definitions
        """
        bone_pairs, ref_lengths, ref_stds = load_bone_stats(
            stats_path, keypoint_names, skeleton
        )
        return cls(bone_pairs, ref_lengths, ref_stds)


def skeleton_to_index_pairs(keypoint_names, skeleton):
    """Convert skeleton name pairs to index pairs."""
    name_to_idx = {name: i for i, name in enumerate(keypoint_names)}
    pairs = []
    for bone in skeleton:
        a, b = bone[0], bone[1]
        if a in name_to_idx and b in name_to_idx:
            pairs.append((name_to_idx[a], name_to_idx[b]))
    return pairs


def compute_bone_stats(keypoints3D_list, bone_pairs):
    """
    Compute per-bone mean and std lengths from ground truth keypoints.

    Args:
        keypoints3D_list: list of (num_joints, 3) arrays, GT 3D keypoints
        bone_pairs: list of (i, j) index tuples

    Returns:
        dict mapping bone index to {mean, std, count}
    """
    bone_lengths = {b: [] for b in range(len(bone_pairs))}

    for keypoints3D in keypoints3D_list:
        for b_idx, (i, j) in enumerate(bone_pairs):
            pi = keypoints3D[i]
            pj = keypoints3D[j]
            if (np.abs(pi).sum() > 0) and (np.abs(pj).sum() > 0):
                length = np.linalg.norm(pi - pj)
                if length > 0.1:  # filter degenerate
                    bone_lengths[b_idx].append(length)

    stats = {}
    for b_idx, (i, j) in enumerate(bone_pairs):
        lengths = bone_lengths[b_idx]
        if len(lengths) > 0:
            stats[b_idx] = {
                'mean': float(np.mean(lengths)),
                'std': float(np.std(lengths)),
                'count': len(lengths),
            }
        else:
            stats[b_idx] = {'mean': 0.0, 'std': 1.0, 'count': 0}
    return stats


def save_bone_stats(stats, bone_pairs, keypoint_names, skeleton, output_path):
    """Save bone stats to JSON with human-readable bone names."""
    output = {
        'bones': [],
        'keypoint_names': keypoint_names,
    }
    for b_idx, (i, j) in enumerate(bone_pairs):
        bone_name = f"{keypoint_names[i]}-{keypoint_names[j]}"
        entry = {
            'name': bone_name,
            'idx_a': i,
            'idx_b': j,
            'mean_length_mm': stats[b_idx]['mean'],
            'std_length_mm': stats[b_idx]['std'],
            'sample_count': stats[b_idx]['count'],
        }
        output['bones'].append(entry)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def load_bone_stats(stats_path, keypoint_names, skeleton):
    """
    Load bone stats and return (bone_pairs, ref_lengths, ref_stds) tensors.
    """
    with open(stats_path, 'r') as f:
        data = json.load(f)

    bone_pairs = []
    ref_lengths = []
    ref_stds = []
    for bone in data['bones']:
        bone_pairs.append((bone['idx_a'], bone['idx_b']))
        ref_lengths.append(bone['mean_length_mm'])
        ref_stds.append(bone['std_length_mm'])

    return (
        bone_pairs,
        torch.tensor(ref_lengths, dtype=torch.float32),
        torch.tensor(ref_stds, dtype=torch.float32),
    )
