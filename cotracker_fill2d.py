#!/usr/bin/env python3
"""
cotracker_fill2d.py

Use Meta's CoTracker3 to fill in poor/NaN 2D keypoint predictions from
JARVIS-HybridNet for single-camera recordings.

Pipeline
--------
1. Load data2D.csv  (2 header rows + N rows × 72 cols = 24 joints × [x, y, conf])
2. Find "anchor frames" where mean confidence > --conf-threshold
3. Process in overlapping windows:
     - Find best anchor inside the window
     - Run CoTracker seeded from that anchor
     - Replace low-confidence frames with CoTracker predictions
4. Save filled CSV preserving original header rows

Usage
-----
  python cotracker_fill2d.py \\
      --video /path/to/video.mp4 \\
      --csv   /path/to/Cam_data2D.csv \\
      --output /tmp/filled_data2D.csv \\
      --device cuda
"""

import argparse
import csv

import cv2
import numpy as np
import torch
from tqdm import tqdm

N_JOINTS = 24


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def load_csv(csv_path: str):
    """
    Load data2D.csv.

    Returns
    -------
    header1 : list[str]   — first header row (joint names)
    header2 : list[str]   — second header row (x/y/confidence labels)
    data    : np.ndarray  — shape (N_frames, N_JOINTS, 3) float32; NaN for missing
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header1 = next(reader)
        header2 = next(reader)
        rows = list(reader)

    n = len(rows)
    data = np.full((n, N_JOINTS * 3), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            if val not in ('NaN', '', 'nan'):
                try:
                    data[i, j] = float(val)
                except ValueError:
                    pass
    return header1, header2, data.reshape(n, N_JOINTS, 3)


def save_csv(out_path: str, header1, header2, data: np.ndarray):
    """
    Save data2D.csv preserving the two header rows.
    data: (N_frames, N_JOINTS, 3)
    """
    n = data.shape[0]
    flat = data.reshape(n, N_JOINTS * 3)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header1)
        writer.writerow(header2)
        for row in flat:
            out_row = []
            for v in row:
                if np.isnan(v):
                    out_row.append('NaN')
                else:
                    out_row.append(repr(float(v)))
            writer.writerow(out_row)


# ── Video I/O ─────────────────────────────────────────────────────────────────

def get_video_dims(video_path: str):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def load_window_frames(video_path: str, start: int, n_frames: int,
                       ct_scale: float, W_orig: int, H_orig: int):
    """
    Extract n_frames from video_path starting at frame `start`.
    Returns np.ndarray (T, H_small, W_small, 3) uint8 RGB, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    W_small = max(1, int(W_orig * ct_scale))
    H_small = max(1, int(H_orig * ct_scale))
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame_rgb, (W_small, H_small))
        frames.append(frame_small)
    cap.release()
    if not frames:
        return None
    return np.stack(frames)   # (T, H, W, 3)


# ── CoTracker ─────────────────────────────────────────────────────────────────

def load_cotracker(device: torch.device):
    model = torch.hub.load(
        'facebookresearch/co-tracker', 'cotracker3_offline',
        trust_repo=True, verbose=False,
    )
    return model.to(device).eval()


def run_cotracker(model, frames_rgb: np.ndarray, anchor_idx: int,
                  anchor_xy_small: np.ndarray, device: torch.device):
    """
    Run CoTracker on a window of frames.

    Parameters
    ----------
    frames_rgb      : (T, H, W, 3) uint8 RGB
    anchor_idx      : frame index within window used as query seed
    anchor_xy_small : (N_JOINTS, 2) keypoint xy at small (ct_scale) resolution
    device          : torch device

    Returns
    -------
    tracks : (T, N_JOINTS, 2) float32  at small resolution
    vis    : (T, N_JOINTS)   float32
    """
    video = (torch.from_numpy(frames_rgb)
             .float()
             .permute(0, 3, 1, 2)   # (T, 3, H, W)
             .unsqueeze(0)           # (1, T, 3, H, W)
             .to(device))

    N = anchor_xy_small.shape[0]
    queries = torch.zeros((1, N, 3), dtype=torch.float32, device=device)
    queries[0, :, 0] = float(anchor_idx)
    queries[0, :, 1] = torch.from_numpy(anchor_xy_small[:, 0].astype(np.float32))
    queries[0, :, 2] = torch.from_numpy(anchor_xy_small[:, 1].astype(np.float32))

    with torch.no_grad():
        pred_tracks, pred_vis = model(video, queries=queries)
    # pred_tracks: (1, T, N, 2)
    # pred_vis:    (1, T, N) or (1, T, N, 1)

    tracks = pred_tracks[0].cpu().numpy()   # (T, N, 2)
    vis = pred_vis[0]
    if vis.dim() == 3:
        vis = vis[:, :, 0]
    vis = vis.cpu().numpy()                  # (T, N)
    return tracks, vis


# ── Main ──────────────────────────────────────────────────────────────────────

def count_nan_rows(data: np.ndarray) -> int:
    """data: (N, N_JOINTS, 3).  A row is 'NaN' if all confidences are NaN."""
    confs = data[:, :, 2]   # (N, N_JOINTS)
    return int(np.all(np.isnan(confs), axis=1).sum())


def main():
    parser = argparse.ArgumentParser(
        description='Fill sparse JARVIS 2D predictions with CoTracker3.')
    parser.add_argument('--video', required=True,
                        help='Path to input video')
    parser.add_argument('--csv', required=True,
                        help='Path to data2D.csv from JARVIS predictions')
    parser.add_argument('--output', required=True,
                        help='Path for output (filled) CSV')
    parser.add_argument('--conf-threshold', type=float, default=0.15,
                        help='Min mean confidence to treat a frame as anchor (default 0.15)')
    parser.add_argument('--device', default='cuda',
                        help='Torch device (default: cuda)')
    parser.add_argument('--window', type=int, default=100,
                        help='Sliding window size in frames (default: 100)')
    parser.add_argument('--overlap', type=int, default=15,
                        help='Overlap between consecutive windows (default: 15)')
    parser.add_argument('--ct-scale', type=float, default=0.25,
                        help='Scale factor for CoTracker input resolution (default: 0.25)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')
    if str(device) != args.device:
        print(f'[warn] {args.device} not available, falling back to cpu')

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f'Loading CSV: {args.csv}')
    header1, header2, data = load_csv(args.csv)
    N_frames = data.shape[0]

    nan_before = count_nan_rows(data)
    print(f'Input:  {N_frames} frames, {nan_before} NaN rows '
          f'({100*nan_before/N_frames:.1f}%)')

    # ── Identify anchor frames ────────────────────────────────────────────────
    # mean confidence per frame (NaN joints contribute 0)
    confs = np.where(np.isnan(data[:, :, 2]), 0.0, data[:, :, 2])
    mean_conf = confs.mean(axis=1)   # (N_frames,)
    is_anchor = mean_conf > args.conf_threshold

    n_anchors = int(is_anchor.sum())
    if n_anchors == 0:
        print('[warn] No anchor frames found — nothing to propagate. '
              'Try lowering --conf-threshold.')
        save_csv(args.output, header1, header2, data)
        return

    print(f'Found {n_anchors} anchor frames ({100*n_anchors/N_frames:.1f}%)')

    # ── Load CoTracker ────────────────────────────────────────────────────────
    print('Loading CoTracker3 ...')
    model = load_cotracker(device)

    # ── Video dimensions ──────────────────────────────────────────────────────
    W_orig, H_orig = get_video_dims(args.video)
    print(f'Video size: {W_orig}x{H_orig}, '
          f'CT size: {int(W_orig*args.ct_scale)}x{int(H_orig*args.ct_scale)}')

    # ── Sliding window ────────────────────────────────────────────────────────
    out_data = data.copy()
    stride = args.window - args.overlap

    starts = list(range(0, N_frames, stride))
    # ensure last window reaches the end
    if starts[-1] + args.window < N_frames:
        starts.append(N_frames - args.window)

    for start in tqdm(starts, desc='Windows'):
        end = min(start + args.window, N_frames)
        win_len = end - start

        # find best anchor in this window
        win_conf = mean_conf[start:end]
        win_anchor_mask = is_anchor[start:end]
        if not win_anchor_mask.any():
            continue  # no anchor — skip

        # pick the highest-confidence anchor in window
        anchor_rel = int(np.argmax(win_conf * win_anchor_mask.astype(float)))

        # load video frames for this window
        frames_rgb = load_window_frames(
            args.video, start, win_len, args.ct_scale, W_orig, H_orig)
        if frames_rgb is None:
            continue
        T_actual = len(frames_rgb)
        anchor_rel = min(anchor_rel, T_actual - 1)

        # anchor keypoints at ct_scale resolution
        anchor_abs = start + anchor_rel
        anchor_xy_full = out_data[anchor_abs, :, :2]   # (N_JOINTS, 2)
        anchor_xy_small = anchor_xy_full * args.ct_scale  # scale down

        # run CoTracker
        try:
            tracks_small, vis = run_cotracker(
                model, frames_rgb, anchor_rel, anchor_xy_small, device)
        except Exception as e:
            print(f'[warn] CoTracker failed on window {start}-{end}: {e}')
            continue

        # scale tracks back to full resolution
        tracks_full = tracks_small / args.ct_scale   # (T_actual, N_JOINTS, 2)

        # fill low-confidence frames in this window
        for t in range(T_actual):
            frame_abs = start + t
            if frame_abs >= N_frames:
                break
            frame_conf = mean_conf[frame_abs]
            if frame_conf < args.conf_threshold:
                # replace with CoTracker prediction
                out_data[frame_abs, :, 0] = tracks_full[t, :, 0]
                out_data[frame_abs, :, 1] = tracks_full[t, :, 1]
                # use visibility as a proxy confidence (0–1 already)
                out_data[frame_abs, :, 2] = vis[t].astype(np.float32)

    # ── Save output ───────────────────────────────────────────────────────────
    print(f'Saving output to: {args.output}')
    save_csv(args.output, header1, header2, out_data)

    nan_after = count_nan_rows(out_data)
    print(f'Input:  {N_frames} frames, {nan_before:4d} NaN rows '
          f'({100*nan_before/N_frames:.1f}%)')
    print(f'Output: {N_frames} frames, {nan_after:4d} NaN rows '
          f'({100*nan_after/N_frames:.1f}%)')


if __name__ == '__main__':
    main()
