#!/usr/bin/env python3
"""
Run background augmentation using the pre-computed candidate list.

Reads: manifests/augment_candidates_pass16_pass15.jsonl  (158 frames)
Reads: annotations/instances_train.json  (for keypoints)

For each candidate frame, generates --num-aug variants:
  - random gradient background (same across all 16 cameras per variant)
  - mild affine warp (same across all cameras per variant)
  - mouse foreground composited onto new background via mask

Saves augmented images into:
  merged_output2/train/<session>/<cam>/<frame>_aug<i>.jpg  (alongside originals)

Then writes:
  annotations/instances_train_aug.json  (original + augmented entries)

Usage:
    python run_augmentation.py [--num-aug N] [--max-frames M] [--seed S]
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/user/mouse_labels/jarvis_merge/merged_output2")
CANDIDATES_JSONL = ROOT / "manifests" / "augment_candidates_pass16_pass15.jsonl"
ORIG_TRAIN_JSON = ROOT / "annotations" / "instances_train.json"
OUT_TRAIN_JSON = ROOT / "annotations" / "instances_train_aug.json"

IMG_W, IMG_H = 3208, 2200


def make_background(h, w, seed):
    rng = np.random.default_rng(seed)
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


def affine_for_frame(seed, w, h):
    rng = random.Random(seed)
    s = rng.uniform(0.95, 1.05)
    tx = rng.uniform(-40, 40)
    ty = rng.uniform(-40, 40)
    cx, cy = w / 2.0, h / 2.0
    M = np.array([[s, 0.0, tx + (1.0 - s) * cx],
                  [0.0, s, ty + (1.0 - s) * cy]], dtype=np.float32)
    return M


def transform_keypoints(kps_flat, M):
    """kps_flat: [x,y,v, x,y,v, ...] → transformed in place."""
    out = list(kps_flat)
    for i in range(0, len(out), 3):
        x, y, v = out[i], out[i+1], out[i+2]
        if v > 0:
            xn = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            yn = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            out[i], out[i+1] = float(xn), float(yn)
    return out


def load_candidates(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_ann_lookup(train_json):
    """Map rel_image_path (train/session/cam/frame.jpg) -> annotation dict."""
    data = json.load(open(train_json))
    id_to_img = {img["id"]: img for img in data["images"]}
    id_to_ann = {ann["image_id"]: ann for ann in data["annotations"]}
    lookup = {}
    for img_id, img_info in id_to_img.items():
        rel = "train/" + img_info["file_name"]
        if img_id in id_to_ann:
            lookup[rel] = id_to_ann[img_id]
    return lookup, data


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-aug", type=int, default=10,
                help="Augmented variants per frame (default: 10)")
    parser.add_argument("--max-frames", type=int, default=0,
                help="Limit number of frames processed (0=all)")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    print("Loading candidates and annotations...")
    candidates = load_candidates(CANDIDATES_JSONL)
    if args.max_frames > 0:
        candidates = candidates[:args.max_frames]
    print(f"  {len(candidates)} candidate frames, {args.num_aug} aug each")

    ann_lookup, orig_data = build_ann_lookup(ORIG_TRAIN_JSON)

    # Start from original data, we'll append to it
    new_images = list(orig_data["images"])
    new_annotations = list(orig_data["annotations"])
    new_framesets = dict(orig_data["framesets"])

    next_img_id = max(img["id"] for img in new_images) + 1
    next_ann_id = max(ann["id"] for ann in new_annotations) + 1

    total_aug = 0
    rng_global = random.Random(args.seed)

    for frame_idx, frame_rec in enumerate(candidates):
        split = frame_rec["split"]
        session = frame_rec["session"]
        frame = frame_rec["frame"]
        views = frame_rec["views"]

        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx+1}/{len(candidates)}: {session}/{frame}")

        for aug_i in range(args.num_aug):
            seed_base = args.seed + frame_idx * 7919 + aug_i * 104729
            bg = make_background(IMG_H, IMG_W, seed_base)
            M = affine_for_frame(seed_base, IMG_W, IMG_H)

            frameset_img_ids = []
            cam_order = [v["camera"] for v in views]

            for view in views:
                cam = view["camera"]
                img_abs = Path(view["image"])
                mask_abs = Path(view["mask"])

                if not img_abs.exists() or not mask_abs.exists():
                    continue

                img = cv2.imread(str(img_abs), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(mask_abs), cv2.IMREAD_GRAYSCALE)
                if img is None or mask is None:
                    continue

                mask_bin = (mask > 0).astype(np.uint8) * 255
                fg_rgba = np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8)
                fg_rgba[:, :, :3] = img
                fg_rgba[:, :, 3] = mask_bin

                warped_rgba = cv2.warpAffine(
                    fg_rgba, M, (IMG_W, IMG_H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0),
                )
                warped_mask = warped_rgba[:, :, 3]
                alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, None]
                comp = (warped_rgba[:, :, :3].astype(np.float32) * alpha
                        + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

                # Save alongside originals
                out_stem = f"{frame}_aug{aug_i:02d}"
                out_rel = f"{session}/{cam}/{out_stem}.jpg"
                out_abs = ROOT / "train" / out_rel
                out_abs.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_abs), comp)

                # Keypoints
                rel_key = f"train/{session}/{cam}/{frame}.jpg"
                ann = ann_lookup.get(rel_key)
                kps_orig = ann["keypoints"] if ann else []
                kps_aug = transform_keypoints(kps_orig, M) if kps_orig else []

                # Bbox from augmented keypoints
                xs = [kps_aug[i] for i in range(0, len(kps_aug), 3) if kps_aug[i+2] > 0]
                ys = [kps_aug[i+1] for i in range(0, len(kps_aug), 3) if kps_aug[i+2] > 0]
                if xs and ys:
                    bbox = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
                else:
                    bbox = ann["bbox"] if ann else [0, 0, 0, 0]

                new_images.append({
                    "id": next_img_id,
                    "file_name": out_rel,
                    "height": IMG_H,
                    "width": IMG_W,
                    "coco_url": "",
                    "date_captured": "",
                    "flickr_url": "",
                })
                new_annotations.append({
                    "id": next_ann_id,
                    "image_id": next_img_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "keypoints": kps_aug,
                    "iscrowd": 0,
                })
                frameset_img_ids.append(next_img_id)
                next_img_id += 1
                next_ann_id += 1
                total_aug += 1

            if len(frameset_img_ids) >= 2:
                fs_key = f"{session}/{frame}_aug{aug_i:02d}"
                new_framesets[fs_key] = {
                    "datasetName": session,
                    "frames": frameset_img_ids,
                }

    print(f"\nGenerated {total_aug} augmented images, {len(new_framesets) - len(orig_data['framesets'])} new framesets")

    out_data = {
        "keypoint_names": orig_data["keypoint_names"],
        "skeleton": orig_data["skeleton"],
        "categories": orig_data["categories"],
        "calibrations": orig_data["calibrations"],
        "images": new_images,
        "annotations": new_annotations,
        "framesets": new_framesets,
    }

    print(f"Writing {OUT_TRAIN_JSON} ...")
    print(f"  Total images: {len(new_images)}, framesets: {len(new_framesets)}")
    with open(OUT_TRAIN_JSON, "w") as f:
        json.dump(out_data, f)
    print("Done.")


if __name__ == "__main__":
    main()
