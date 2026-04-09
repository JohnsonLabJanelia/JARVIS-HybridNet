"""
Temporal dataset for the Temporal Refinement Transformer.
Loads pre-extracted HybridNet predictions and creates sliding windows
of consecutive frames from the same recording session.
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalDataset(Dataset):
    """
    Creates sliding windows of T consecutive frames from pre-extracted
    HybridNet predictions, grouped by recording session.

    Prediction files are expected as .npz with keys:
        - points3D: (24, 3) predicted keypoints in mm
        - confidences: (24,) confidence scores
        - gt_keypoints3D: (24, 3) ground truth keypoints

    File naming convention: {session}_Frame_{num}.npz

    Args:
        predictions_dir: directory containing .npz prediction files
        window_size: number of frames per window (T)
        stride: sliding window stride
        max_gap_frames: max video frame gap before splitting a run
        split: 'train' or 'val' subdirectory within predictions_dir
    """
    def __init__(self, predictions_dir, window_size=32, stride=8,
                 max_gap_frames=16, split='train'):
        self.window_size = window_size
        self.predictions_dir = os.path.join(predictions_dir, split)

        # Discover and parse prediction files
        files = sorted([f for f in os.listdir(self.predictions_dir)
                        if f.endswith('.npz')])

        # Group by session
        session_frames = {}
        for fname in files:
            # Expected: {session}_Frame_{num}.npz
            match = re.match(r'^(.+)_Frame_(\d+)\.npz$', fname)
            if match:
                session = match.group(1)
                frame_num = int(match.group(2))
                if session not in session_frames:
                    session_frames[session] = []
                session_frames[session].append((frame_num, fname))

        # Sort each session by frame number
        for session in session_frames:
            session_frames[session].sort(key=lambda x: x[0])

        # Build contiguous runs and sliding windows
        self.windows = []
        for session, frames in session_frames.items():
            runs = self._split_into_runs(frames, max_gap_frames)
            for run in runs:
                self._create_windows(run, stride)

    def _split_into_runs(self, frames, max_gap):
        """Split a sorted list of (frame_num, fname) into contiguous runs."""
        if not frames:
            return []
        runs = [[frames[0]]]
        for i in range(1, len(frames)):
            gap = frames[i][0] - frames[i - 1][0]
            if gap > max_gap:
                runs.append([])
            runs[-1].append(frames[i])
        return [r for r in runs if len(r) >= 2]

    def _create_windows(self, run, stride):
        """Create sliding windows from a contiguous run."""
        n = len(run)
        if n >= self.window_size:
            for start in range(0, n - self.window_size + 1, stride):
                window_frames = run[start:start + self.window_size]
                self.windows.append(window_frames)
        else:
            # Pad short runs with repetition of last frame
            padded = list(run) + [run[-1]] * (self.window_size - n)
            valid_len = n
            self.windows.append((padded, valid_len))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]

        # Handle padded short windows
        if isinstance(window, tuple):
            frames_list, valid_len = window
            valid_mask = np.zeros(self.window_size, dtype=bool)
            valid_mask[:valid_len] = True
        else:
            frames_list = window
            valid_mask = np.ones(self.window_size, dtype=bool)

        pred_kp = np.zeros((self.window_size, 24, 3), dtype=np.float32)
        pred_conf = np.zeros((self.window_size, 24), dtype=np.float32)
        gt_kp = np.zeros((self.window_size, 24, 3), dtype=np.float32)

        for t, (frame_num, fname) in enumerate(frames_list):
            path = os.path.join(self.predictions_dir, fname)
            data = np.load(path)
            pred_kp[t] = data['points3D']
            pred_conf[t] = data['confidences']
            gt_kp[t] = data['gt_keypoints3D']

        # Per-joint validity: GT keypoint is nonzero
        gt_valid = np.any(gt_kp != 0, axis=-1)  # (T, 24)

        return {
            'pred_keypoints': torch.from_numpy(pred_kp),
            'pred_confidences': torch.from_numpy(pred_conf),
            'gt_keypoints': torch.from_numpy(gt_kp),
            'valid_mask': torch.from_numpy(valid_mask),
            'gt_valid': torch.from_numpy(gt_valid),
        }
