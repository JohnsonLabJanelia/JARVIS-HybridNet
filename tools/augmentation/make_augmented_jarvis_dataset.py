#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create JARVIS-compatible augmented dataset from confident views.")
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/user/mouse_labels/jarvis_merge/merged_output2"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/user/mouse_labels/jarvis_merge/augmented_merged_out2"),
    )
    p.add_argument(
        "--views-csv",
        type=Path,
        default=Path("/home/user/mouse_labels/jarvis_merge/merged_output2/multiview_aug/manifests/pass_at_least_16/views_good.csv"),
    )
    p.add_argument(
        "--strategies",
        type=str,
        default="color_jitter,noise_blur,gamma_contrast,motion_blur,jpeg_artifacts,shadow,color_temperature",
    )
    p.add_argument("--num-per-strategy", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def apply_color_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.75, 1.25), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.80, 1.20), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_noise_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(3.0, 10.0)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.8:
        out = cv2.GaussianBlur(out, (int(rng.choice([3, 5])), int(rng.choice([3, 5]))), 0)
    return out


def apply_gamma_contrast(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gamma = float(rng.uniform(0.75, 1.35))
    alpha = float(rng.uniform(0.85, 1.20))
    beta = float(rng.uniform(-18, 18))
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    out = cv2.LUT(img, lut)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    return out


def apply_motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.choice([5, 7, 9, 11, 13]))
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def apply_jpeg_artifacts(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = int(rng.integers(20, 60))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def apply_shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32).copy()
    h, w = out.shape[:2]
    cx = int(rng.integers(int(0.2 * w), int(0.8 * w)))
    cy = int(rng.integers(int(0.25 * h), int(0.85 * h)))
    ax = int(rng.integers(int(0.12 * w), int(0.35 * w)))
    ay = int(rng.integers(int(0.08 * h), int(0.22 * h)))
    y, x = np.ogrid[:h, :w]
    m = (((x - cx) / max(ax, 1)) ** 2 + ((y - cy) / max(ay, 1)) ** 2) <= 1.0
    out[m] *= float(rng.uniform(0.55, 0.85))
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_color_temperature(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)
    warm = rng.random() < 0.5
    if warm:
        r_scale = float(rng.uniform(1.05, 1.25))
        b_scale = float(rng.uniform(0.80, 0.98))
    else:
        r_scale = float(rng.uniform(0.80, 0.98))
        b_scale = float(rng.uniform(1.05, 1.25))
    g_scale = float(rng.uniform(0.92, 1.08))
    out[:, :, 2] *= r_scale
    out[:, :, 1] *= g_scale
    out[:, :, 0] *= b_scale
    return np.clip(out, 0, 255).astype(np.uint8)


STRATEGY_FUNCS = {
    "color_jitter": apply_color_jitter,
    "noise_blur": apply_noise_blur,
    "gamma_contrast": apply_gamma_contrast,
    "motion_blur": apply_motion_blur,
    "jpeg_artifacts": apply_jpeg_artifacts,
    "shadow": apply_shadow,
    "color_temperature": apply_color_temperature,
}


def hardlink_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    shutil.copytree(src, dst, copy_function=os.link)


def load_dataset_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.rebuild and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    hardlink_tree(args.source_root / "train", args.output_root / "train")
    hardlink_tree(args.source_root / "val", args.output_root / "val")
    hardlink_tree(args.source_root / "calib_params", args.output_root / "calib_params")
    (args.output_root / "annotations").mkdir(parents=True, exist_ok=True)

    train_data = load_dataset_json(args.source_root / "annotations" / "instances_train.json")
    val_data = load_dataset_json(args.source_root / "annotations" / "instances_val.json")

    train_file_to_img = {img["file_name"]: img for img in train_data["images"]}
    val_file_to_img = {img["file_name"]: img for img in val_data["images"]}
    train_imgid_to_anns = {}
    val_imgid_to_anns = {}
    for ann in train_data["annotations"]:
        train_imgid_to_anns.setdefault(ann["image_id"], []).append(ann)
    for ann in val_data["annotations"]:
        val_imgid_to_anns.setdefault(ann["image_id"], []).append(ann)

    next_img_id_train = max(img["id"] for img in train_data["images"]) + 1
    next_ann_id_train = max(ann["id"] for ann in train_data["annotations"]) + 1
    next_img_id_val = max(img["id"] for img in val_data["images"]) + 1
    next_ann_id_val = max(ann["id"] for ann in val_data["annotations"]) + 1

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in STRATEGY_FUNCS:
            raise ValueError(f"Unknown strategy: {s}")

    with open(args.views_csv) as f:
        views = list(csv.DictReader(f))

    created = 0
    for i, row in enumerate(views, 1):
        rel_img = row["rel_image_path"]  # train/session/cam/frame.jpg
        p = Path(rel_img)
        split, session, cam, orig_name = p.parts[0], p.parts[1], p.parts[2], p.name
        json_file_name = str(Path(session) / cam / orig_name)

        if split == "train":
            dataset = train_data
            file_to_img = train_file_to_img
            imgid_to_anns = train_imgid_to_anns
            next_img_id = next_img_id_train
            next_ann_id = next_ann_id_train
        else:
            dataset = val_data
            file_to_img = val_file_to_img
            imgid_to_anns = val_imgid_to_anns
            next_img_id = next_img_id_val
            next_ann_id = next_ann_id_val

        src_img_info = file_to_img.get(json_file_name)
        if src_img_info is None:
            continue
        src_img_path = args.source_root / split / json_file_name
        src_img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
        if src_img is None:
            continue

        for strat in strategies:
            fn = STRATEGY_FUNCS[strat]
            for k in range(args.num_per_strategy):
                aug = fn(src_img, rng)
                new_name = f"{Path(orig_name).stem}__{strat}_{k:02d}.jpg"
                rel_new = str(Path(session) / cam / new_name)
                out_img = args.output_root / split / rel_new
                out_img.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_img), aug)

                new_img = dict(src_img_info)
                new_img["id"] = next_img_id
                new_img["file_name"] = rel_new
                dataset["images"].append(new_img)

                for ann in imgid_to_anns.get(src_img_info["id"], []):
                    new_ann = dict(ann)
                    new_ann["id"] = next_ann_id
                    new_ann["image_id"] = next_img_id
                    dataset["annotations"].append(new_ann)
                    next_ann_id += 1

                next_img_id += 1
                created += 1

        if split == "train":
            next_img_id_train = next_img_id
            next_ann_id_train = next_ann_id
        else:
            next_img_id_val = next_img_id
            next_ann_id_val = next_ann_id

        if i % 100 == 0 or i == len(views):
            print(f"[{i}/{len(views)}] augmented images created={created}", flush=True)

    with open(args.output_root / "annotations" / "instances_train.json", "w") as f:
        json.dump(train_data, f)
    with open(args.output_root / "annotations" / "instances_val.json", "w") as f:
        json.dump(val_data, f)

    summary = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "views_csv": str(args.views_csv),
        "strategies": strategies,
        "num_per_strategy": args.num_per_strategy,
        "confident_views": len(views),
        "augmented_images_created": created,
        "train_images_total": len(train_data["images"]),
        "val_images_total": len(val_data["images"]),
        "train_annotations_total": len(train_data["annotations"]),
        "val_annotations_total": len(val_data["annotations"]),
    }
    with open(args.output_root / "augmentation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

