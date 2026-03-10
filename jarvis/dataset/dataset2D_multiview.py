"""
Multi-view dataset for JARVIS-TR semi-supervised training.

Loads synchronized multi-camera frames (both labeled and unlabeled) for
computing the Triangulation Residual Loss. Unlike Dataset2D (single images)
or Dataset3D (with 3D heatmaps), this returns per-camera 2D crops + camera
projection matrices — the minimal data needed for TR loss.

Unlabeled frames are sampled randomly from video files at each epoch.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from jarvis.utils.reprojection import load_reprojection_tools


class Dataset2DMultiview(Dataset):
    """Multi-view dataset that loads synchronized frames from video files.

    Args:
        cfg: JARVIS config
        recording_path: Path to folder with .mp4 files (one per camera)
        dataset_name: Calibration dataset name (for loading camera matrices)
        num_frames: Number of unlabeled frames to sample per epoch
        frame_start: First frame to sample from
        frame_end: Last frame to sample from (-1 = end of video)
        crop_size: 2D crop size around detected/estimated center
    """

    def __init__(self, cfg, recording_path, dataset_name,
                 num_frames=200, frame_start=0, frame_end=-1,
                 crop_size=None):
        self.cfg = cfg
        self.recording_path = recording_path
        self.dataset_name = dataset_name
        self.num_frames = num_frames
        self.frame_start = frame_start
        self.crop_size = crop_size or cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE

        # Load calibration (same mechanism as Dataset3D)
        dataset_path = os.path.join(
            cfg.PARENT_DIR, cfg.DATASET.DATASET_ROOT_DIR,
            cfg.DATASET.DATASET_3D or cfg.DATASET.DATASET_2D)
        annotations_path = os.path.join(
            dataset_path, 'annotations', 'instances_train.json')
        with open(annotations_path) as f:
            dataset_json = json.load(f)

        calib_paths = {}
        if dataset_name in dataset_json['calibrations']:
            for cam, path in dataset_json['calibrations'][dataset_name].items():
                calib_paths[cam] = path
        else:
            # Use first available calibration
            first_key = list(dataset_json['calibrations'].keys())[0]
            for cam, path in dataset_json['calibrations'][first_key].items():
                calib_paths[cam] = path

        from jarvis.utils.reprojection import ReprojectionTool
        self.repro_tool = ReprojectionTool(dataset_path, calib_paths)
        self.camera_names = list(self.repro_tool.cameras.keys())
        self.num_cameras = len(self.camera_names)

        # Projection matrices: (N_cams, 3, 4)
        # JARVIS stores them as (N_cams, 4, 3) transposed — convert
        self.proj_matrices = self.repro_tool.cameraMatrices.transpose(1, 2).float()

        # Find video files
        self.video_paths = []
        videos = sorted(os.listdir(recording_path))
        for cam_name in self.camera_names:
            found = False
            for video in videos:
                if cam_name in video and video.endswith('.mp4'):
                    self.video_paths.append(
                        os.path.join(recording_path, video))
                    found = True
                    break
            if not found:
                raise FileNotFoundError(
                    f"No video found for camera {cam_name} in {recording_path}")

        # Get total frame count
        cap = cv2.VideoCapture(self.video_paths[0])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.frame_end = frame_end if frame_end > 0 else total_frames
        self.frame_end = min(self.frame_end, total_frames)

        # Pre-sample frame indices (re-sampled each epoch via resample())
        self.frame_indices = []
        self.resample()

        # ImageNet normalization
        self.mean = np.array(cfg.DATASET.MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(cfg.DATASET.STD, dtype=np.float32).reshape(1, 1, 3)

    def resample(self):
        """Resample random frame indices for this epoch."""
        available = self.frame_end - self.frame_start
        n = min(self.num_frames, available)
        self.frame_indices = np.sort(
            np.random.choice(range(self.frame_start, self.frame_end),
                             size=n, replace=False))

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        frame_num = self.frame_indices[idx]
        half = self.crop_size // 2

        imgs = np.zeros((self.num_cameras, self.crop_size, self.crop_size, 3),
                        dtype=np.float32)

        for c in range(self.num_cameras):
            cap = cv2.VideoCapture(self.video_paths[c])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # Crop from center of image (simple default — CenterDetect
            # would give better crops, but we don't have it during training)
            cx, cy = w // 2, h // 2
            cx = max(half, min(cx, w - half))
            cy = max(half, min(cy, h - half))

            crop = frame[cy - half:cy + half, cx - half:cx + half]
            imgs[c] = (crop.astype(np.float32) / 255.0 - self.mean) / self.std

        return (
            imgs,                           # (N_cams, crop_size, crop_size, 3)
            self.proj_matrices.clone(),     # (N_cams, 3, 4)
        )
