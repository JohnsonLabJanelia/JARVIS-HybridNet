#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


ROOT = Path("/home/user/mouse_labels/jarvis_merge/merged_output2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build JARVIS-ready augmented dataset from pass-threshold views.")
    p.add_argument("--threshold", type=int, default=16, help="pass_at_least_K threshold (1..16)")
    p.add_argument(
        "--manifests-root",
        type=Path,
        default=ROOT / "multiview_aug" / "manifests",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "multiview_aug" / "jarvis_ready_dataset",
    )
    p.add_argument(
        "--strategies",
        type=str,
        default="color_jitter,noise_blur,affine_bgmix",
        help="comma-separated: color_jitter,noise_blur,affine_bgmix,occlusion",
    )
    p.add_argument("--num-per-strategy", type=int, default=1)
    p.add_argument("--include-original", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def load_ann_lookup() -> Dict[str, dict]:
    out = {}
    for split in ["train", "val"]:
        with open(ROOT / "annotations" / f"instances_{split}.json") as f:
            data = json.load(f)
        id_to_img = {img["id"]: img for img in data["images"]}
        id_to_ann = {ann["image_id"]: ann for ann in data["annotations"]}
        for img_id, img in id_to_img.items():
            rel = f"{split}/{img['file_name']}"
            if img_id in id_to_ann:
                out[rel] = id_to_ann[img_id]
    return out


def robust_keypoints(ann: dict, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      xy: Nx2 float32 keypoint coordinates
      vis: N int visibility values
    """
    kps = ann.get("keypoints", [])
    if not kps:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    ys = [kps[i + 1] for i in range(0, len(kps), 3) if kps[i + 2] > 0]
    neg_ratio = (sum(1 for y in ys if y < 0) / len(ys)) if ys else 0.0
    has_neg_y = neg_ratio > 0.7

    xy = []
    vis = []
    for i in range(0, len(kps), 3):
        x, y, v = kps[i], kps[i + 1], kps[i + 2]
        if v <= 0:
            continue
        if abs(float(x)) > 1e5 or abs(float(y)) > 1e5:
            continue
        if has_neg_y:
            y = h + y
        if 0 <= x < w and 0 <= y < h:
            xy.append((float(x), float(y)))
            vis.append(int(v))
    if not xy:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.asarray(xy, dtype=np.float32), np.asarray(vis, dtype=np.int32)


def make_gradient_bg(h: int, w: int, rng: random.Random) -> np.ndarray:
    c1 = np.array([rng.randint(10, 230), rng.randint(10, 230), rng.randint(10, 230)], dtype=np.float32)
    c2 = np.array([rng.randint(10, 230), rng.randint(10, 230), rng.randint(10, 230)], dtype=np.float32)
    xv = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    yv = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    a = 0.6 * xv + 0.4 * yv
    bg = c1[None, None, :] * (1.0 - a[..., None]) + c2[None, None, :] * a[..., None]
    noise = np.random.normal(0, 6.0, size=(h, w, 3)).astype(np.float32)
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(bg, (5, 5), 0)


def apply_color_jitter(img: np.ndarray, rng: random.Random) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_scale = rng.uniform(0.75, 1.25)
    v_scale = rng.uniform(0.80, 1.20)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def apply_noise_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    sigma = rng.uniform(3.0, 10.0)
    noise = np.random.normal(0, sigma, size=img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.8:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def apply_occlusion(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(rng.randint(1, 3)):
        x1 = rng.randint(0, int(0.85 * w))
        y1 = rng.randint(0, int(0.85 * h))
        x2 = min(w - 1, x1 + rng.randint(int(0.05 * w), int(0.20 * w)))
        y2 = min(h - 1, y1 + rng.randint(int(0.05 * h), int(0.20 * h)))
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, -1)
    return out


def apply_gamma_contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    gamma = rng.uniform(0.75, 1.35)
    alpha = rng.uniform(0.85, 1.20)  # contrast
    beta = rng.uniform(-18, 18)      # brightness
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    out = cv2.LUT(img, lut)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    return out


def apply_motion_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice([5, 7, 9, 11, 13])
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def apply_jpeg_artifacts(img: np.ndarray, rng: random.Random) -> np.ndarray:
    q = rng.randint(20, 60)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def apply_shadow(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    # dark ellipse-like shadow region
    cx = rng.randint(int(0.2 * w), int(0.8 * w))
    cy = rng.randint(int(0.25 * h), int(0.85 * h))
    ax = rng.randint(int(0.12 * w), int(0.35 * w))
    ay = rng.randint(int(0.08 * h), int(0.22 * h))
    y, x = np.ogrid[:h, :w]
    mask = (((x - cx) / max(ax, 1)) ** 2 + ((y - cy) / max(ay, 1)) ** 2) <= 1.0
    factor = rng.uniform(0.55, 0.85)
    out[mask] *= factor
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_color_temperature(img: np.ndarray, rng: random.Random) -> np.ndarray:
    # Warm/cool cast by channel scaling
    out = img.astype(np.float32)
    warm = rng.random() < 0.5
    if warm:
        r_scale = rng.uniform(1.05, 1.25)
        b_scale = rng.uniform(0.80, 0.98)
    else:
        r_scale = rng.uniform(0.80, 0.98)
        b_scale = rng.uniform(1.05, 1.25)
    g_scale = rng.uniform(0.92, 1.08)
    out[:, :, 2] *= r_scale
    out[:, :, 1] *= g_scale
    out[:, :, 0] *= b_scale
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_background_replace(img: np.ndarray, mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    bg = make_gradient_bg(h, w, rng)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    comp = img.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(comp, 0, 255).astype(np.uint8)


def blend_foreground(base_img: np.ndarray, fg_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    out = base_img.astype(np.float32) * (1.0 - alpha) + fg_img.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_mouse_color_jitter(img: np.ndarray, mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """Change mouse fur color/saturation only (foreground only)."""
    fg = img.copy()
    hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_shift = rng.uniform(-12, 12)   # small hue shift, keep realistic
    s_scale = rng.uniform(0.75, 1.35)
    v_scale = rng.uniform(0.80, 1.20)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    fg_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return blend_foreground(img, fg_aug, mask)


def apply_mouse_relight(img: np.ndarray, mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply directional lighting changes to mouse only."""
    h, w = img.shape[:2]
    fg = img.astype(np.float32).copy()

    # Build a smooth directional light field
    theta = rng.uniform(0, 2 * np.pi)
    dx, dy = np.cos(theta), np.sin(theta)
    xv = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    yv = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    light = dx * xv + dy * yv
    light = (light - light.min()) / max(light.max() - light.min(), 1e-6)  # [0,1]
    light = 0.6 + light * rng.uniform(0.6, 1.2)  # scale roughly [0.6, 1.8]
    light = np.clip(light, 0.45, 1.9)

    # Per-channel mild scaling to simulate color temperature shifts on fur
    ch = np.array(
        [
            rng.uniform(0.90, 1.10),  # B
            rng.uniform(0.90, 1.10),  # G
            rng.uniform(0.90, 1.10),  # R
        ],
        dtype=np.float32,
    )
    fg *= light[:, :, None]
    fg *= ch[None, None, :]
    fg = np.clip(fg, 0, 255).astype(np.uint8)
    return blend_foreground(img, fg, mask)


def apply_mouse_gamma(img: np.ndarray, mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """Gamma/contrast changes on mouse only."""
    gamma = rng.uniform(0.7, 1.4)
    alpha = rng.uniform(0.85, 1.20)
    beta = rng.uniform(-20, 20)
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    fg = cv2.LUT(img, lut)
    fg = cv2.convertScaleAbs(fg, alpha=alpha, beta=beta)
    return blend_foreground(img, fg, mask)


def apply_affine_bgmix(
    img: np.ndarray,
    mask: np.ndarray,
    kps_xy: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    bg = make_gradient_bg(h, w, rng)

    s = rng.uniform(0.92, 1.08)
    rot = rng.uniform(-8.0, 8.0) * np.pi / 180.0
    tx = rng.uniform(-40.0, 40.0)
    ty = rng.uniform(-40.0, 40.0)
    cx, cy = w / 2.0, h / 2.0
    c, sn = np.cos(rot), np.sin(rot)
    M = np.array(
        [
            [s * c, -s * sn, tx + cx - s * c * cx + s * sn * cy],
            [s * sn, s * c, ty + cy - s * sn * cx - s * c * cy],
        ],
        dtype=np.float32,
    )

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img
    rgba[:, :, 3] = mask
    wrgba = cv2.warpAffine(
        rgba,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    wmask = wrgba[:, :, 3]
    alpha = (wmask.astype(np.float32) / 255.0)[:, :, None]
    out_img = (wrgba[:, :, :3].astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    if len(kps_xy) > 0:
        xy1 = np.concatenate([kps_xy, np.ones((len(kps_xy), 1), dtype=np.float32)], axis=1)
        kps_new = (M @ xy1.T).T.astype(np.float32)
    else:
        kps_new = kps_xy.copy()
    return out_img, wmask, kps_new


def save_sample(
    out_base: Path,
    rel_img: str,
    strategy: str,
    idx: int,
    img: np.ndarray,
    mask: np.ndarray,
) -> Tuple[Path, Path]:
    p = Path(rel_img)
    img_dir = out_base / "images" / p.parent
    mask_dir = out_base / "masks" / p.parent
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    img_out = img_dir / f"{p.stem}__{strategy}_{idx:02d}.jpg"
    mask_out = mask_dir / f"{p.stem}__{strategy}_{idx:02d}_mask.png"
    cv2.imwrite(str(img_out), img)
    cv2.imwrite(str(mask_out), mask)
    return img_out, mask_out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not (1 <= args.threshold <= 16):
        raise ValueError("--threshold must be in [1,16]")

    manifest_dir = args.manifests_root / f"pass_at_least_{args.threshold}"
    views_csv = manifest_dir / "views_good.csv"
    if not views_csv.exists():
        raise FileNotFoundError(f"Missing manifest file: {views_csv}")

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    valid = {
        "color_jitter",
        "noise_blur",
        "affine_bgmix",
        "occlusion",
        "gamma_contrast",
        "motion_blur",
        "jpeg_artifacts",
        "shadow",
        "color_temperature",
        "background_replace",
        "mouse_color_jitter",
        "mouse_relight",
        "mouse_gamma",
    }
    for s in strategies:
        if s not in valid:
            raise ValueError(f"Unknown strategy: {s}. valid={sorted(valid)}")

    ann_lookup = load_ann_lookup()
    out_base = args.output_root / f"pass_at_least_{args.threshold}"
    out_base.mkdir(parents=True, exist_ok=True)

    records_all: List[dict] = []
    split_counts = {"train": 0, "val": 0}

    with open(views_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, r in enumerate(rows, 1):
        rel_img = r["rel_image_path"]
        rel_mask = r["rel_mask_path"]
        split = r["split"]
        img_path = ROOT / rel_img
        mask_path = ROOT / rel_mask

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        mask_bin = ((mask > 0).astype(np.uint8) * 255)

        ann = ann_lookup.get(rel_img, {})
        kps_xy, kps_vis = robust_keypoints(ann, h=img.shape[0], w=img.shape[1])

        # Original
        if args.include_original:
            out_img, out_mask = save_sample(out_base, rel_img, "orig", 0, img, mask_bin)
            rec = {
                "split": split,
                "strategy": "orig",
                "image": str(out_img),
                "mask": str(out_mask),
                "source_image": str(img_path),
                "source_mask": str(mask_path),
                "session": r["session"],
                "camera": r["camera"],
                "frame": r["frame"],
                "kp_inside_pct": float(r["kp_inside_pct"]),
                "mask_area_px": int(r["mask_area_px"]),
                "keypoints_xy": kps_xy.tolist(),
                "keypoints_vis": kps_vis.tolist(),
            }
            records_all.append(rec)
            split_counts[split] += 1

        for s in strategies:
            for aug_idx in range(args.num_per_strategy):
                rng = random.Random(args.seed + i * 1009 + aug_idx * 7919 + hash(s) % 100000)
                if s == "color_jitter":
                    aug_img = apply_color_jitter(img, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                elif s == "noise_blur":
                    aug_img = apply_noise_blur(img, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                elif s == "occlusion":
                    aug_img = apply_occlusion(img, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                elif s == "affine_bgmix":
                    aug_img, aug_mask, aug_kps = apply_affine_bgmix(img, mask_bin, kps_xy, rng)
                elif s == "gamma_contrast":
                    aug_img = apply_gamma_contrast(img, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                elif s == "motion_blur":
                    aug_img = apply_motion_blur(img, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                elif s == "jpeg_artifacts":
                    aug_img = apply_jpeg_artifacts(img, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                elif s == "shadow":
                    aug_img = apply_shadow(img, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                elif s == "color_temperature":
                    aug_img = apply_color_temperature(img, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                elif s == "background_replace":
                    aug_img = apply_background_replace(img, mask_bin, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                elif s == "mouse_color_jitter":
                    aug_img = apply_mouse_color_jitter(img, mask_bin, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                elif s == "mouse_relight":
                    aug_img = apply_mouse_relight(img, mask_bin, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                elif s == "mouse_gamma":
                    aug_img = apply_mouse_gamma(img, mask_bin, rng)
                    aug_mask = mask_bin.copy()
                    aug_kps = kps_xy.copy()
                else:
                    continue

                out_img, out_mask = save_sample(out_base, rel_img, s, aug_idx, aug_img, aug_mask)
                rec = {
                    "split": split,
                    "strategy": s,
                    "image": str(out_img),
                    "mask": str(out_mask),
                    "source_image": str(img_path),
                    "source_mask": str(mask_path),
                    "session": r["session"],
                    "camera": r["camera"],
                    "frame": r["frame"],
                    "kp_inside_pct": float(r["kp_inside_pct"]),
                    "mask_area_px": int(r["mask_area_px"]),
                    "keypoints_xy": aug_kps.tolist(),
                    "keypoints_vis": kps_vis.tolist(),
                }
                records_all.append(rec)
                split_counts[split] += 1

        if i % 100 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] processed views", flush=True)

    # write manifests
    ann_dir = out_base / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    all_jsonl = ann_dir / "all.jsonl"
    train_jsonl = ann_dir / "train.jsonl"
    val_jsonl = ann_dir / "val.jsonl"

    with open(all_jsonl, "w") as fa, open(train_jsonl, "w") as ft, open(val_jsonl, "w") as fv:
        for rec in records_all:
            line = json.dumps(rec)
            fa.write(line + "\n")
            if rec["split"] == "train":
                ft.write(line + "\n")
            else:
                fv.write(line + "\n")

    summary = {
        "threshold": args.threshold,
        "strategies": strategies,
        "num_per_strategy": args.num_per_strategy,
        "include_original": bool(args.include_original),
        "source_views_manifest": str(views_csv),
        "source_view_count": len(rows),
        "output_root": str(out_base),
        "total_samples": len(records_all),
        "samples_by_split": split_counts,
        "manifests": {
            "all": str(all_jsonl),
            "train": str(train_jsonl),
            "val": str(val_jsonl),
        },
    }
    with open(out_base / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

