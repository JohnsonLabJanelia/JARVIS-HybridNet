#!/usr/bin/env python3
"""
Run HybridNet 3D prediction on dense consecutive video frames from multi-camera
video files. Saves per-frame .npz predictions for temporal transformer training.

Usage:
    python tools/predict_video_for_temporal.py \
        --project mouseJan30 \
        --video_dir /mnt/nvme1/mouse_labels/ssd/mouse_vids/2024_11_06_12_19_42 \
        --calib_dir /mnt/nvme1/mouse_labels/ssd/mouse_vids/2024_11_06_12_19_42/calibration \
        --output predictions/mouseJan30_video/train \
        --start_frame 0 --num_frames 1000 --stride 1
"""
import argparse
import os
import sys
import time
import glob
import numpy as np
from tqdm import tqdm

import torch
import cv2
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jarvis.config.project_manager import ProjectManager
from jarvis.prediction.jarvis3D import JarvisPredictor3D


def load_calibration(calib_dir, camera_names):
    """Load camera calibration from YAML files.
    Handles both JARVIS format (intrinsicMatrix/R/T) and
    OpenCV format (camera_matrix/rc_ext/tc_ext)."""
    cameraMatrices = []
    intrinsicMatrices = []
    distortionCoefficients = []

    for cam in camera_names:
        yaml_path = os.path.join(calib_dir, f'{cam}.yaml')
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)

        # Try JARVIS format first, then OpenCV format
        K = fs.getNode('intrinsicMatrix').mat()
        if K is None:
            # OpenCV format: camera_matrix is standard (not transposed)
            K_std = fs.getNode('camera_matrix').mat()
            # JARVIS stores intrinsicMatrix transposed: K.T
            K = K_std.T
            R = fs.getNode('rc_ext').mat()
            T = fs.getNode('tc_ext').mat()
            dist = fs.getNode('distortion_coefficients').mat()
        else:
            R = fs.getNode('R').mat()
            T = fs.getNode('T').mat()
            dist = fs.getNode('distortionCoefficients').mat()
        fs.release()

        # Build camera matrix same way JARVIS does:
        # cameraMatrix = (np.concatenate((R, T.reshape(1,3)), axis=0) @ K).T
        camMat = np.concatenate((R, T.reshape(1, 3)), axis=0).dot(K)
        cameraMatrices.append(camMat)  # already (4,3) after concat+dot
        intrinsicMatrices.append(K)
        distortionCoefficients.append(dist.flatten()[:5].reshape(1, 5))

    return (
        torch.tensor(np.array(cameraMatrices), dtype=torch.float32).cuda(),
        torch.tensor(np.array(intrinsicMatrices), dtype=torch.float32).cuda(),
        torch.tensor(np.array(distortionCoefficients), dtype=torch.float32).cuda(),
    )


def open_videos(video_dir, camera_names):
    """Open video captures for all cameras."""
    caps = {}
    for cam in camera_names:
        path = os.path.join(video_dir, f'{cam}.mp4')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        caps[cam] = cap
    return caps


def seek_all(caps, camera_names, frame_idx):
    """Seek all cameras to a specific frame."""
    for cam in camera_names:
        caps[cam].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


def read_next_frame(caps, camera_names):
    """Read the next frame from all cameras (sequential, no seeking)."""
    imgs = []
    for cam in camera_names:
        ret, frame = caps[cam].read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        imgs.append(frame)
    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--video_dir', required=True,
                        help='Dir with Cam*.mp4 files')
    parser.add_argument('--calib_dir', required=True,
                        help='Dir with Cam*.yaml calibration files')
    parser.add_argument('--output', required=True)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=500)
    parser.add_argument('--stride', type=int, default=1,
                        help='Frame stride (1=every frame, 2=every other)')
    parser.add_argument('--session_name', default=None,
                        help='Session name for output files')
    args = parser.parse_args()

    project = ProjectManager()
    project.load(args.project)
    cfg = project.get_cfg()

    # Camera names from config
    camera_names = [
        'Cam2002486', 'Cam2002487', 'Cam2005325', 'Cam2006050',
        'Cam2006051', 'Cam2006052', 'Cam2006054', 'Cam2006055',
        'Cam2006515', 'Cam2006516', 'Cam2008665', 'Cam2008666',
        'Cam2008667', 'Cam2008668', 'Cam2008669', 'Cam2008670',
    ]

    session_name = args.session_name or os.path.basename(args.video_dir)
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading model for project '{args.project}'...")
    predictor = JarvisPredictor3D(cfg)
    predictor.eval()

    print(f"Loading calibration from {args.calib_dir}...")
    camMats, intrMats, distCoeffs = load_calibration(args.calib_dir, camera_names)

    print(f"Opening {len(camera_names)} camera videos from {args.video_dir}...")
    caps = open_videos(args.video_dir, camera_names)

    # Check total frames
    total_in_video = int(caps[camera_names[0]].get(cv2.CAP_PROP_FRAME_COUNT))
    fps = caps[camera_names[0]].get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_in_video} frames at {fps:.0f} FPS")

    end_frame = min(args.start_frame + args.num_frames * args.stride,
                    total_in_video)
    frame_indices = list(range(args.start_frame, end_frame, args.stride))
    print(f"Processing frames {args.start_frame}-{end_frame} "
          f"(stride={args.stride}, {len(frame_indices)} frames)")

    saved = 0
    failed = 0
    times = []

    # Seek to start frame once, then read sequentially
    seek_all(caps, camera_names, args.start_frame)
    prev_frame = args.start_frame

    with torch.no_grad():
        for frame_idx in tqdm(frame_indices, desc='Predicting'):
            # Skip frames if stride > 1
            skip = frame_idx - prev_frame - 1
            for _ in range(skip):
                read_next_frame(caps, camera_names)
            prev_frame = frame_idx
            imgs_list = read_next_frame(caps, camera_names)
            if imgs_list is None:
                failed += 1
                continue

            # Stack and convert to tensor: (N_cams, 3, H, W)
            imgs_tensor = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1)
                for img in imgs_list
            ]).float().cuda()

            t0 = time.perf_counter()
            points3D, confidences = predictor(
                imgs_tensor, camMats, intrMats, distCoeffs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

            if points3D is not None:
                pts = points3D[0].cpu().numpy()   # (24, 3)
                conf = confidences[0].cpu().numpy()  # (24,)

                np.savez(
                    os.path.join(args.output,
                                f'{session_name}_Frame_{frame_idx}.npz'),
                    points3D=pts.astype(np.float32),
                    confidences=conf.astype(np.float32),
                    gt_keypoints3D=pts.astype(np.float32),  # self-training: pred=GT
                    session=session_name,
                    frame_name=f'Frame_{frame_idx}',
                )
                saved += 1
            else:
                failed += 1

    for cap in caps.values():
        cap.release()

    print(f"\nDone! Saved {saved} predictions, {failed} failed frames")
    print(f"Mean inference: {np.mean(times)*1000:.0f} ms/frame "
          f"({1.0/np.mean(times):.1f} FPS)")
    print(f"Output: {args.output}/")


if __name__ == '__main__':
    main()
