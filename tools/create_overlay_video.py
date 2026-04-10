#!/usr/bin/env python3
"""
Create a video with 3D prediction overlays projected onto a camera view.
Draws keypoints and skeleton connections on the original video frames.

Usage:
    python tools/create_overlay_video.py \
        --project mouseJan30 \
        --video /path/to/CamXXX.mp4 \
        --calib /path/to/CamXXX.yaml \
        --predictions predictions/robFeb12Talk \
        --session rob_2026_02_07 \
        --output overlay_video.mp4 \
        --start_frame 0 --num_frames 5000 --fps 30
"""
import argparse
import os
import sys
import glob
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jarvis.config.project_manager import ProjectManager
from jarvis.hybridnet.physics_loss import skeleton_to_index_pairs


def load_calibration(calib_path):
    """Load camera calibration from a single YAML file."""
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    # Try JARVIS format first
    K = fs.getNode('intrinsicMatrix').mat()
    if K is not None:
        R = fs.getNode('R').mat()
        T = fs.getNode('T').mat()
    else:
        K = fs.getNode('camera_matrix').mat().T
        R = fs.getNode('rc_ext').mat()
        T = fs.getNode('tc_ext').mat()
    fs.release()
    # Build camera matrix same as JARVIS
    cam_mat = np.concatenate((R, T.reshape(1, 3)), axis=0).dot(K)
    return cam_mat, K


def project_3d_to_2d(pts3d, cam_mat, K):
    """Project 3D points to 2D pixel coordinates."""
    pts_h = np.hstack([pts3d, np.ones((len(pts3d), 1))])
    proj = pts_h.dot(cam_mat)
    px = proj[:, 0] / proj[:, 2]
    py = proj[:, 1] / proj[:, 2]
    return np.stack([px, py], axis=1)


# Skeleton colors (warm palette)
JOINT_COLORS = {
    'head': (0, 200, 255),      # orange - Snout, EarL, EarR, Neck
    'spine': (0, 255, 100),     # green - SpineL, TailBase
    'tail': (255, 100, 0),      # blue - Tail*
    'forelimb_l': (255, 255, 0),  # cyan - ShoulderL..HandL
    'forelimb_r': (255, 0, 255),  # magenta - ShoulderR..HandR
    'hindlimb_l': (0, 255, 255),  # yellow - KneeL..FootL
    'hindlimb_r': (100, 100, 255),  # light red - KneeR..FootR
}

JOINT_GROUPS = {
    'Snout': 'head', 'EarL': 'head', 'EarR': 'head', 'Neck': 'head',
    'SpineL': 'spine', 'TailBase': 'spine',
    'TailTip': 'tail', 'TailMid': 'tail', 'Tail1Q': 'tail', 'Tail3Q': 'tail',
    'ShoulderL': 'forelimb_l', 'ElbowL': 'forelimb_l',
    'WristL': 'forelimb_l', 'HandL': 'forelimb_l',
    'ShoulderR': 'forelimb_r', 'ElbowR': 'forelimb_r',
    'WristR': 'forelimb_r', 'HandR': 'forelimb_r',
    'KneeL': 'hindlimb_l', 'AnkleL': 'hindlimb_l', 'FootL': 'hindlimb_l',
    'KneeR': 'hindlimb_r', 'AnkleR': 'hindlimb_r', 'FootR': 'hindlimb_r',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--video', required=True, help='Single camera MP4')
    parser.add_argument('--calib', required=True, help='Single camera YAML')
    parser.add_argument('--predictions', required=True, help='Dir with .npz files')
    parser.add_argument('--session', required=True, help='Session name prefix in npz files')
    parser.add_argument('--output', default='overlay_video.mp4')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Number of frames (default: all available)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS (default: 30)')
    parser.add_argument('--conf_threshold', type=float, default=0.15,
                        help='Minimum confidence to draw joint')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Output resolution scale (0.5 = half size)')
    args = parser.parse_args()

    project = ProjectManager()
    project.load(args.project)
    cfg = project.get_cfg()
    kp_names = list(cfg.KEYPOINT_NAMES)
    skeleton = list(cfg.SKELETON)
    bone_pairs = skeleton_to_index_pairs(kp_names, skeleton)

    cam_mat, K = load_calibration(args.calib)

    # Find available prediction files
    pred_files = {}
    for f in sorted(glob.glob(os.path.join(args.predictions, f'{args.session}_Frame_*.npz'))):
        fname = os.path.basename(f)
        frame_num = int(fname.split('Frame_')[1].split('.')[0])
        pred_files[frame_num] = f

    if not pred_files:
        print(f"No prediction files found for session '{args.session}' in {args.predictions}")
        return

    available = sorted(pred_files.keys())
    print(f"Found {len(available)} predictions, frames {available[0]}-{available[-1]}")

    # Determine frame range
    start = max(args.start_frame, available[0])
    if args.num_frames:
        end = min(start + args.num_frames, available[-1] + 1)
    else:
        end = available[-1] + 1

    frame_range = [f for f in available if start <= f < end]
    print(f"Rendering frames {frame_range[0]}-{frame_range[-1]} ({len(frame_range)} frames)")

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0])
    prev_frame_num = frame_range[0]

    for frame_num in tqdm(frame_range, desc='Rendering'):
        # Skip frames if needed
        skip = frame_num - prev_frame_num - 1
        for _ in range(skip):
            cap.read()
        prev_frame_num = frame_num

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        if frame_num in pred_files:
            data = np.load(pred_files[frame_num])
            pts3d = data['points3D']
            conf = data['confidences']
            pts2d = project_3d_to_2d(pts3d, cam_mat, K) * args.scale

            # Draw skeleton connections first (behind joints)
            for i, j in bone_pairs:
                if conf[i] > args.conf_threshold and conf[j] > args.conf_threshold:
                    p1 = (int(pts2d[i, 0]), int(pts2d[i, 1]))
                    p2 = (int(pts2d[j, 0]), int(pts2d[j, 1]))
                    group = JOINT_GROUPS.get(kp_names[i], 'spine')
                    color = JOINT_COLORS.get(group, (200, 200, 200))
                    alpha = min(conf[i], conf[j])
                    # Dim color by confidence
                    color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
                    cv2.line(frame, p1, p2, color, 2)

            # Draw joints
            for j_idx in range(len(pts3d)):
                if conf[j_idx] > args.conf_threshold:
                    x, y = int(pts2d[j_idx, 0]), int(pts2d[j_idx, 1])
                    if 0 <= x < width and 0 <= y < height:
                        group = JOINT_GROUPS.get(kp_names[j_idx], 'spine')
                        color = JOINT_COLORS.get(group, (200, 200, 200))
                        radius = max(3, int(6 * conf[j_idx]))
                        cv2.circle(frame, (x, y), radius, color, -1)
                        cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)

            # Info overlay
            mean_conf = conf.mean()
            n_good = (conf > 0.5).sum()
            cv2.putText(frame, f'Frame {frame_num} | conf: {mean_conf:.2f} | {n_good}/24 joints > 0.5',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"\nSaved overlay video: {args.output}")
    print(f"  Resolution: {width}x{height} @ {args.fps} FPS")
    print(f"  Duration: {len(frame_range)/args.fps:.1f}s ({len(frame_range)} frames)")


if __name__ == '__main__':
    main()
