#!/usr/bin/env python3
"""
End-to-end pipeline:
1) Validate keypoint-mask alignment per image
2) Validate 16-camera consistency per frame
3) Optionally fix failed masks with conservative keypoint-guided refinement
4) Augment only fully passing frames
5) Keep all original frames in training manifests

Outputs:
- reports/alignment_image_metrics.csv
- reports/alignment_frame_decisions.csv
- reports/alignment_summary.json
- masks_fixed/... (only if fixes are applied)
- augmented_aligned/... (augmented images, masks, keypoints)
- manifests/original_manifest.jsonl
- manifests/augmented_manifest.jsonl
- manifests/training_manifest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


ROOT = Path("/home/user/mouse_labels/jarvis_merge/merged_output2")
ANNOT_DIR = ROOT / "annotations"
DEFAULT_MASK_ROOT = ROOT / "masks"
DEFAULT_FIXED_MASK_ROOT = ROOT / "masks_fixed"
DEFAULT_REPORT_DIR = ROOT / "reports"
DEFAULT_AUG_ROOT = ROOT / "augmented_aligned"
DEFAULT_MANIFEST_DIR = ROOT / "manifests"

IMG_W = 3208
IMG_H = 2200


@dataclass
class Thresholds:
    min_kp_inside_pct: float = 90.0
    max_outside_dist_px: float = 50.0
    min_mask_area_px: int = 1000
    min_bbox_iou: float = 0.7
    min_bbox_coverage: float = 0.8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["alignment", "augment", "all"], default="all")
    p.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    p.add_argument("--fixed-mask-root", type=Path, default=DEFAULT_FIXED_MASK_ROOT)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--aug-root", type=Path, default=DEFAULT_AUG_ROOT)
    p.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--enable-fix", action="store_true")
    p.add_argument("--num-aug", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def load_annotations(split: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Returns:
      image_lookup: rel_path_with_split -> image_info
      ann_lookup: rel_path_with_split -> annotation
    """
    ann_path = ANNOT_DIR / f"instances_{split}.json"
    with open(ann_path) as f:
        data = json.load(f)

    id_to_img = {img["id"]: img for img in data["images"]}
    id_to_ann = {ann["image_id"]: ann for ann in data["annotations"]}

    image_lookup: Dict[str, dict] = {}
    ann_lookup: Dict[str, dict] = {}
    for img_id, img_info in id_to_img.items():
        rel = f"{split}/{img_info['file_name']}"
        image_lookup[rel] = img_info
        if img_id in id_to_ann:
            ann_lookup[rel] = id_to_ann[img_id]
    return image_lookup, ann_lookup


def mask_path_from_rel(mask_root: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    return mask_root / rel.parent / f"{rel.stem}_mask.png"


def parse_frame_key(rel_path: str) -> Tuple[str, str, str]:
    """
    Returns (split, session, frame_stem) from:
      split/session/CamXXXX/Frame_YYYY.jpg
    """
    p = Path(rel_path)
    split = p.parts[0]
    session = p.parts[1]
    frame = p.stem
    return split, session, frame


def parse_camera(rel_path: str) -> str:
    return Path(rel_path).parts[2]


def extract_visible_keypoints(ann: dict, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    kps = ann.get("keypoints", [])
    if not kps:
        return np.zeros((0, 2), dtype=np.float32)

    # Detect negative-Y encoding used in some sessions.
    # Important: decide by majority of visible points, not "any negative Y",
    # because some annotations include outlier/sentinel values.
    visible_ys = [kps[i + 1] for i in range(0, len(kps), 3) if kps[i + 2] > 0]
    neg_ratio = (sum(1 for y in visible_ys if y < 0) / len(visible_ys)) if visible_ys else 0.0
    has_neg_y = neg_ratio > 0.7

    pts: List[Tuple[float, float]] = []
    for i in range(0, len(kps), 3):
        x, y, v = kps[i], kps[i + 1], kps[i + 2]
        if v <= 0:
            continue
        # Drop obvious corrupted keypoints before coordinate transforms.
        if abs(float(x)) > 1e5 or abs(float(y)) > 1e5:
            continue
        if has_neg_y:
            y = h + y
        if 0 <= x < w and 0 <= y < h:
            pts.append((float(x), float(y)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def bbox_from_points(points: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if points.size == 0:
        return None
    x1 = int(np.floor(points[:, 0].min()))
    y1 = int(np.floor(points[:, 1].min()))
    x2 = int(np.ceil(points[:, 0].max()))
    y2 = int(np.ceil(points[:, 1].max()))
    return x1, y1, x2, y2


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    b_area = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    denom = a_area + b_area - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def bbox_coverage(mask: np.ndarray, b: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(mask.shape[1] - 1, x2)
    y2 = min(mask.shape[0] - 1, y2)
    if x2 < x1 or y2 < y1:
        return 0.0
    region = mask[y1 : y2 + 1, x1 : x2 + 1]
    total = region.size
    if total == 0:
        return 0.0
    return float((region > 0).sum() / total)


def compute_alignment_metrics(mask: np.ndarray, keypoints: np.ndarray) -> dict:
    metrics = {
        "num_kp_visible": int(len(keypoints)),
        "num_kp_inside": 0,
        "kp_inside_pct": 0.0,
        "max_outside_dist_px": 0.0,
        "mask_area_px": int((mask > 0).sum()),
        "bbox_iou": 0.0,
        "bbox_coverage": 0.0,
    }
    if len(keypoints) == 0:
        return metrics

    h, w = mask.shape
    inside_flags = []
    outside_pts = []
    for x, y in keypoints:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0:
            inside_flags.append(True)
        else:
            inside_flags.append(False)
            outside_pts.append((xi, yi))

    num_inside = int(sum(inside_flags))
    metrics["num_kp_inside"] = num_inside
    metrics["kp_inside_pct"] = (100.0 * num_inside / len(keypoints)) if len(keypoints) else 0.0

    # Distance from outside keypoints to nearest foreground.
    if outside_pts:
        # distanceTransform computes dist to nearest zero pixel, so invert logic.
        # We want distance to nearest foreground (mask>0):
        # create binary where foreground is 0 and background is 1.
        inv = (mask == 0).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        max_d = 0.0
        for xi, yi in outside_pts:
            if 0 <= xi < w and 0 <= yi < h:
                max_d = max(max_d, float(dist[yi, xi]))
            else:
                max_d = max(max_d, 1e6)
        metrics["max_outside_dist_px"] = max_d

    kp_bbox = bbox_from_points(keypoints)
    m_bbox = bbox_from_mask(mask)
    if kp_bbox and m_bbox:
        metrics["bbox_iou"] = bbox_iou(kp_bbox, m_bbox)
        metrics["bbox_coverage"] = bbox_coverage(mask, kp_bbox)
    return metrics


def passes_thresholds(metrics: dict, th: Thresholds) -> bool:
    if metrics["num_kp_visible"] <= 0:
        return False
    if metrics["kp_inside_pct"] < th.min_kp_inside_pct:
        return False
    if metrics["max_outside_dist_px"] > th.max_outside_dist_px:
        return False
    if metrics["mask_area_px"] < th.min_mask_area_px:
        return False
    if metrics["bbox_iou"] < th.min_bbox_iou:
        return False
    if metrics["bbox_coverage"] < th.min_bbox_coverage:
        return False
    return True


def conservative_fix_mask(mask: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """
    Conservative fix: only small local additions near keypoints + cleanup.
    """
    fixed = (mask > 0).astype(np.uint8) * 255
    if len(keypoints) == 0:
        return fixed

    # If empty mask, bootstrap from keypoint hull.
    if fixed.sum() == 0 and len(keypoints) >= 3:
        hull = cv2.convexHull(keypoints.astype(np.float32))
        cv2.fillConvexPoly(fixed, hull.astype(np.int32), 255)
        fixed = cv2.dilate(fixed, np.ones((9, 9), np.uint8), iterations=2)
        return fixed

    # Add small circles on keypoints not already covered.
    h, w = fixed.shape
    for x, y in keypoints:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h and fixed[yi, xi] == 0:
            cv2.circle(fixed, (xi, yi), 16, 255, -1)

    fixed = cv2.morphologyEx(fixed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    fixed = cv2.morphologyEx(fixed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return fixed


def required_cameras_by_session() -> Dict[Tuple[str, str], List[str]]:
    out: Dict[Tuple[str, str], List[str]] = {}
    for split in ["train", "val"]:
        split_dir = ROOT / split
        if not split_dir.exists():
            continue
        for session_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            cams = sorted([p.name for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("Cam")])
            out[(split, session_dir.name)] = cams
    return out


def make_background(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Smooth gradient + mild noise for domain randomization.
    c1 = rng.integers(10, 220, size=(3,), dtype=np.uint8)
    c2 = rng.integers(10, 220, size=(3,), dtype=np.uint8)
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    a = (0.6 * xv + 0.4 * yv).astype(np.float32)
    bg = (c1[None, None, :] * (1 - a[..., None]) + c2[None, None, :] * a[..., None]).astype(np.float32)
    noise = rng.normal(0, 8.0, size=(h, w, 3)).astype(np.float32)
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    return bg


def affine_for_frame(seed: int, w: int, h: int) -> np.ndarray:
    rng = random.Random(seed)
    s = rng.uniform(0.95, 1.05)
    tx = rng.uniform(-40, 40)
    ty = rng.uniform(-40, 40)
    cx, cy = w / 2.0, h / 2.0
    M = np.array([[s, 0.0, tx + (1.0 - s) * cx], [0.0, s, ty + (1.0 - s) * cy]], dtype=np.float32)
    return M


def transform_keypoints(kps: np.ndarray, M: np.ndarray) -> np.ndarray:
    if len(kps) == 0:
        return kps.copy()
    xy1 = np.concatenate([kps, np.ones((len(kps), 1), dtype=np.float32)], axis=1)
    out = (M @ xy1.T).T
    return out.astype(np.float32)


def run_alignment(args: argparse.Namespace) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.fixed_mask_root.mkdir(parents=True, exist_ok=True)

    th = Thresholds()
    req_cams = required_cameras_by_session()

    image_rows: List[dict] = []
    frame_map: Dict[Tuple[str, str, str], List[dict]] = {}

    for split in ["train", "val"]:
        _, ann_lookup = load_annotations(split)
        for rel_path, ann in ann_lookup.items():
            img_abs = ROOT / rel_path
            if not img_abs.exists():
                continue

            keypoints = extract_visible_keypoints(ann, h=IMG_H, w=IMG_W)
            mask_path = mask_path_from_rel(args.mask_root, rel_path)
            fixed_path = mask_path_from_rel(args.fixed_mask_root, rel_path)

            mask_status = "orig"
            if fixed_path.exists():
                mask = cv2.imread(str(fixed_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
                mask_status = "fixed_existing"
            elif mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            else:
                mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
                mask_status = "missing"

            metrics = compute_alignment_metrics(mask, keypoints)
            passed = passes_thresholds(metrics, th)
            fixed_applied = False

            if (not passed) and args.enable_fix and len(keypoints) > 0:
                fixed = conservative_fix_mask(mask, keypoints)
                fixed_metrics = compute_alignment_metrics(fixed, keypoints)
                fixed_passed = passes_thresholds(fixed_metrics, th)
                if fixed_passed:
                    fixed_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(fixed_path), fixed)
                    metrics = fixed_metrics
                    passed = True
                    fixed_applied = True
                    mask_status = "fixed_new"

            split_k, session_k, frame_k = parse_frame_key(rel_path)
            cam_k = parse_camera(rel_path)

            row = {
                "rel_path": rel_path,
                "split": split_k,
                "session": session_k,
                "frame": frame_k,
                "camera": cam_k,
                "mask_status": mask_status,
                "fixed_applied": int(fixed_applied),
                "passed": int(passed),
                **metrics,
            }
            image_rows.append(row)
            frame_map.setdefault((split_k, session_k, frame_k), []).append(row)

    # Frame-level decisions (all required cams must pass)
    frame_rows: List[dict] = []
    for frame_key, rows in frame_map.items():
        split, session, frame = frame_key
        required = req_cams.get((split, session), [])
        by_cam = {r["camera"]: r for r in rows}
        missing_cams = [c for c in required if c not in by_cam]
        failing_cams = [c for c in required if c in by_cam and by_cam[c]["passed"] == 0]

        frame_pass = (len(missing_cams) == 0) and (len(failing_cams) == 0) and (len(required) > 0)
        frame_rows.append(
            {
                "split": split,
                "session": session,
                "frame": frame,
                "required_cam_count": len(required),
                "present_cam_count": len(rows),
                "missing_cams": ";".join(missing_cams),
                "failing_cams": ";".join(failing_cams),
                "frame_pass": int(frame_pass),
            }
        )

    # Save reports
    image_csv = args.report_dir / "alignment_image_metrics.csv"
    frame_csv = args.report_dir / "alignment_frame_decisions.csv"
    summary_json = args.report_dir / "alignment_summary.json"

    with open(image_csv, "w", newline="") as f:
        if image_rows:
            writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(image_rows)

    with open(frame_csv, "w", newline="") as f:
        if frame_rows:
            writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_rows)

    total_images = len(image_rows)
    passed_images = sum(r["passed"] for r in image_rows)
    total_frames = len(frame_rows)
    passed_frames = sum(r["frame_pass"] for r in frame_rows)
    summary = {
        "total_images": total_images,
        "passed_images": passed_images,
        "passed_images_pct": (100.0 * passed_images / total_images) if total_images else 0.0,
        "total_frames": total_frames,
        "passed_frames": passed_frames,
        "passed_frames_pct": (100.0 * passed_frames / total_frames) if total_frames else 0.0,
        "enable_fix": bool(args.enable_fix),
        "mask_root": str(args.mask_root),
        "fixed_mask_root": str(args.fixed_mask_root),
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[alignment] wrote {image_csv}")
    print(f"[alignment] wrote {frame_csv}")
    print(f"[alignment] wrote {summary_json}")
    print(f"[alignment] passed images: {passed_images}/{total_images}")
    print(f"[alignment] passed frames (all cams): {passed_frames}/{total_frames}")


def read_csv_rows(path: Path) -> List[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_ann_for_rel() -> Dict[str, dict]:
    out = {}
    for split in ["train", "val"]:
        _, ann_lookup = load_annotations(split)
        out.update(ann_lookup)
    return out


def load_image_for_rel() -> Dict[str, dict]:
    out = {}
    for split in ["train", "val"]:
        img_lookup, _ = load_annotations(split)
        out.update(img_lookup)
    return out


def run_augment(args: argparse.Namespace) -> None:
    args.aug_root.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    frame_csv = args.report_dir / "alignment_frame_decisions.csv"
    image_csv = args.report_dir / "alignment_image_metrics.csv"
    if not frame_csv.exists() or not image_csv.exists():
        raise FileNotFoundError("Run alignment first: missing report CSV files.")

    frame_rows = read_csv_rows(frame_csv)
    image_rows = read_csv_rows(image_csv)
    ann_by_rel = load_ann_for_rel()

    # Build fast lookup for per-image pass and mask preference.
    img_row_by_rel = {r["rel_path"]: r for r in image_rows}

    # Passing frames only.
    pass_frames = [
        (r["split"], r["session"], r["frame"])
        for r in frame_rows
        if int(r["frame_pass"]) == 1
    ]
    if args.max_frames > 0:
        pass_frames = pass_frames[: args.max_frames]

    # Index all image paths by frame key.
    images_by_frame: Dict[Tuple[str, str, str], List[str]] = {}
    for rel_path in img_row_by_rel.keys():
        fk = parse_frame_key(rel_path)
        images_by_frame.setdefault(fk, []).append(rel_path)

    rng = random.Random(args.seed)

    augmented_manifest = args.manifest_dir / "augmented_manifest.jsonl"
    training_manifest = args.manifest_dir / "training_manifest.jsonl"
    original_manifest = args.manifest_dir / "original_manifest.jsonl"

    # Build original manifest (all images with annotation).
    with open(original_manifest, "w") as f:
        for rel_path in sorted(img_row_by_rel.keys()):
            rec = {
                "type": "original",
                "image": str(ROOT / rel_path),
                "mask": str(mask_path_from_rel(args.mask_root, rel_path)),
                "rel_path": rel_path,
            }
            f.write(json.dumps(rec) + "\n")

    aug_count = 0
    with open(augmented_manifest, "w") as f_aug:
        for idx, fk in enumerate(pass_frames):
            split, session, frame = fk
            rels = sorted(images_by_frame.get(fk, []))
            if not rels:
                continue

            # Same background + affine for all 16 cams in a frame (per augmentation sample).
            for aug_i in range(args.num_aug):
                seed_base = args.seed + idx * 7919 + aug_i * 104729
                bg = make_background(IMG_H, IMG_W, seed_base)
                M = affine_for_frame(seed_base, IMG_W, IMG_H)

                for rel_path in rels:
                    img_path = ROOT / rel_path
                    if not img_path.exists():
                        continue

                    # Prefer fixed mask if available.
                    fixed_mask_path = mask_path_from_rel(args.fixed_mask_root, rel_path)
                    mask_path = fixed_mask_path if fixed_mask_path.exists() else mask_path_from_rel(args.mask_root, rel_path)
                    if not mask_path.exists():
                        continue

                    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    mask_bin = (mask > 0).astype(np.uint8) * 255

                    # Build RGBA foreground from original image + mask.
                    fg_rgba = np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8)
                    fg_rgba[:, :, :3] = img
                    fg_rgba[:, :, 3] = mask_bin

                    # Apply same transform for the frame.
                    warped_rgba = cv2.warpAffine(
                        fg_rgba,
                        M,
                        (IMG_W, IMG_H),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0),
                    )
                    warped_mask = warped_rgba[:, :, 3]
                    alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, None]
                    comp = (warped_rgba[:, :, :3].astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

                    ann = ann_by_rel.get(rel_path)
                    kps = extract_visible_keypoints(ann, h=IMG_H, w=IMG_W) if ann else np.zeros((0, 2), dtype=np.float32)
                    kps_aug = transform_keypoints(kps, M)

                    p = Path(rel_path)
                    out_dir = args.aug_root / p.parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_img = out_dir / f"{p.stem}_aug{aug_i:02d}.jpg"
                    out_mask = out_dir / f"{p.stem}_aug{aug_i:02d}_mask.png"
                    out_kp = out_dir / f"{p.stem}_aug{aug_i:02d}_keypoints.json"
                    out_meta = out_dir / f"{p.stem}_aug{aug_i:02d}_meta.json"

                    cv2.imwrite(str(out_img), comp)
                    cv2.imwrite(str(out_mask), warped_mask)
                    with open(out_kp, "w") as fkp:
                        json.dump({"keypoints_xy": kps_aug.tolist()}, fkp)
                    with open(out_meta, "w") as fm:
                        json.dump(
                            {
                                "source_image": str(img_path),
                                "source_mask": str(mask_path),
                                "split": split,
                                "session": session,
                                "frame": frame,
                                "camera": parse_camera(rel_path),
                                "augmentation_index": aug_i,
                                "affine": M.tolist(),
                                "background_seed": seed_base,
                            },
                            fm,
                            indent=2,
                        )

                    rec = {
                        "type": "augmented",
                        "image": str(out_img),
                        "mask": str(out_mask),
                        "keypoints": str(out_kp),
                        "meta": str(out_meta),
                        "source_rel_path": rel_path,
                    }
                    f_aug.write(json.dumps(rec) + "\n")
                    aug_count += 1

    # Combined training manifest = originals + augmented.
    with open(training_manifest, "w") as fout:
        with open(original_manifest) as fo:
            for line in fo:
                fout.write(line)
        if augmented_manifest.exists():
            with open(augmented_manifest) as fa:
                for line in fa:
                    fout.write(line)

    print(f"[augment] pass frames: {len(pass_frames)}")
    print(f"[augment] augmented images written: {aug_count}")
    print(f"[augment] wrote {original_manifest}")
    print(f"[augment] wrote {augmented_manifest}")
    print(f"[augment] wrote {training_manifest}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.phase in ("alignment", "all"):
        run_alignment(args)
    if args.phase in ("augment", "all"):
        run_augment(args)


if __name__ == "__main__":
    main()

