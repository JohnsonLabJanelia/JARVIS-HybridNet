"""
Batch SAM3 segmentation using keypoint prompts for improved accuracy.
Uses mouse keypoints as foreground point prompts + bounding box from keypoints.
Falls back to text-only prompt when no keypoints are available.
Saves binary masks as PNG files in masks_kp/ directory.
Supports resume (skips already-processed images).
"""
import os
import sys
import json
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# ── SAM3 setup ──────────────────────────────────────────────────────────────
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ROOT = Path("/home/user/mouse_labels/jarvis_merge/merged_output2")
BPE_PATH = str(ROOT / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")

# Output directory for keypoint-guided masks
MASK_ROOT = ROOT / "masks_kp"

# Image dimensions (all images are the same size)
IMG_W, IMG_H = 3208, 2200

# Padding around keypoint bbox (pixels)
BBOX_PAD = 80

# Confidence threshold for text-only fallback
TEXT_CONFIDENCE = 0.3


def build_model():
    """Build SAM3 model with instance interactivity (point/box prompts)."""
    print("Building SAM3 model with instance interactivity...", flush=True)
    model = build_sam3_image_model(
        bpe_path=BPE_PATH,
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model, confidence_threshold=TEXT_CONFIDENCE)
    print(f"Model ready on {next(model.parameters()).device}", flush=True)
    return model, processor


def load_annotations():
    """Load train and val annotations, build lookup by file_name."""
    ann_lookup = {}  # file_name -> annotation dict

    for split in ("train", "val"):
        ann_path = ROOT / "annotations" / f"instances_{split}.json"
        if not ann_path.exists():
            print(f"  Warning: {ann_path} not found", flush=True)
            continue

        with open(ann_path) as f:
            data = json.load(f)

        # Build image_id -> image info
        id_to_img = {img["id"]: img for img in data["images"]}

        # Build image_id -> annotation
        id_to_ann = {}
        for ann in data["annotations"]:
            id_to_ann[ann["image_id"]] = ann

        # Map file_name -> annotation + image info
        for img_id, img_info in id_to_img.items():
            fname = img_info["file_name"]  # e.g. "2026_02_26_13_29_50/Cam2006052/Frame_0.jpg"
            full_key = f"{split}/{fname}"
            if img_id in id_to_ann:
                ann_lookup[full_key] = {
                    "annotation": id_to_ann[img_id],
                    "image": img_info,
                }

    print(f"Loaded annotations for {len(ann_lookup)} images", flush=True)
    return ann_lookup


def extract_keypoints(ann_data, img_h=IMG_H):
    """
    Extract visible keypoints from annotation.
    Returns: points (Nx2 array of [x, y]), or None if no visible keypoints.
    Handles negative Y coordinates (session 2024_11_26_17_00_20).
    """
    ann = ann_data["annotation"]
    kps = ann.get("keypoints", [])
    if not kps or len(kps) < 3:
        return None

    points = []
    has_negative_y = False

    # First pass: check for negative Y
    for i in range(0, len(kps), 3):
        x, y, vis = kps[i], kps[i + 1], kps[i + 2]
        if vis > 0 and y < 0:
            has_negative_y = True
            break

    # Second pass: extract visible keypoints with fix
    for i in range(0, len(kps), 3):
        x, y, vis = kps[i], kps[i + 1], kps[i + 2]
        if vis > 0:
            # Fix negative Y: y_pixel = img_h + y_annotation
            if has_negative_y:
                y = img_h + y

            # Validate within image bounds
            if 0 <= x < IMG_W and 0 <= y < img_h:
                points.append([x, y])

    if len(points) < 3:
        return None

    return np.array(points, dtype=np.float32)


def bbox_from_keypoints(points, pad=BBOX_PAD):
    """Compute padded bounding box [x1, y1, x2, y2] from keypoints."""
    x1 = max(0, points[:, 0].min() - pad)
    y1 = max(0, points[:, 1].min() - pad)
    x2 = min(IMG_W, points[:, 0].max() + pad)
    y2 = min(IMG_H, points[:, 1].max() + pad)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def select_prompt_keypoints(points, max_points=12):
    """
    Select a subset of keypoints for prompting.
    We want body keypoints (not tail tips) spread across the body.
    Keypoint order: Snout, EarL, EarR, Neck, SpineL, TailBase,
                    ShoulderL, ElbowL, WristL, HandL,
                    ShoulderR, ElbowR, WristR, HandR,
                    KneeL, AnkleL, FootL, KneeR, AnkleR, FootR,
                    TailTip, TailMid, Tail1Q, Tail3Q
    We prioritize body keypoints (indices 0-5, 6-13) over tail (20-23).
    """
    if len(points) <= max_points:
        return points

    # Just take first max_points (body keypoints come first in the list)
    return points[:max_points]


def segment_with_keypoints(model, processor, img_path, points):
    """
    Segment using keypoint point prompts + bounding box.
    Returns binary mask (H, W) uint8 0/255.
    """
    image = Image.open(img_path).convert("RGB")
    W, H = image.size

    with torch.inference_mode():
        inference_state = processor.set_image(image)

        # Select subset of keypoints for prompt
        prompt_points = select_prompt_keypoints(points)

        # All points are foreground (label=1)
        point_labels = np.ones(len(prompt_points), dtype=np.int32)

        # Bounding box from keypoints
        bbox = bbox_from_keypoints(points)

        # Use predict_inst with points + box
        masks, scores, logits = model.predict_inst(
            inference_state,
            point_coords=prompt_points,
            point_labels=point_labels,
            box=bbox[None, :],  # (1, 4)
            multimask_output=True,
        )

    if masks is None or len(masks) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # Pick best mask by score
    if isinstance(scores, torch.Tensor):
        best_idx = scores.argmax().item()
    else:
        best_idx = np.argmax(scores)
    mask = masks[best_idx]

    # Handle shape: could be (1, H, W) or (H, W)
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    mask = np.asarray(mask).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    return mask * 255


def segment_with_text(model, processor, img_path):
    """
    Fallback: segment using text prompt only.
    Returns binary mask (H, W) uint8 0/255.
    """
    image = Image.open(img_path).convert("RGB")
    W, H = image.size

    with torch.inference_mode():
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt="mouse")

    masks = output["masks"]
    if len(masks) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # Combine all detected masks
    combined = np.zeros((H, W), dtype=np.uint8)
    for m in masks:
        m_np = m.squeeze(0).cpu().numpy().astype(np.uint8)
        combined = np.maximum(combined, m_np)

    return combined * 255


def mask_path_for(img_rel_path: str) -> Path:
    """
    Compute mask output path.
    img_rel_path: e.g. "train/2026_02_26_13_29_50/Cam2006052/Frame_0.jpg"
    -> masks_kp/train/2026_02_26_13_29_50/Cam2006052/Frame_0_mask.png
    """
    p = Path(img_rel_path)
    stem = p.stem
    return MASK_ROOT / p.parent / f"{stem}_mask.png"


def collect_images(splits=("train", "val")):
    """Collect all jpg paths as relative paths."""
    all_images = []
    for split in splits:
        split_dir = ROOT / split
        if not split_dir.exists():
            continue
        for jpg in sorted(split_dir.rglob("*.jpg")):
            rel = jpg.relative_to(ROOT)
            all_images.append(str(rel))
    return all_images


def main():
    # Load annotations
    print("Loading annotations...", flush=True)
    ann_lookup = load_annotations()

    # Collect all images
    all_images = collect_images()
    total = len(all_images)
    print(f"Total images found: {total}", flush=True)

    # Check resume
    todo = []
    for rel_path in all_images:
        mp = mask_path_for(rel_path)
        if not mp.exists():
            todo.append(rel_path)

    already_done = total - len(todo)
    print(f"Already processed: {already_done}", flush=True)
    print(f"Remaining: {len(todo)}", flush=True)

    if len(todo) == 0:
        print("All images already processed!", flush=True)
        return

    # Build model
    model, processor = build_model()

    # Process
    t0 = time.time()
    stats = {"kp": 0, "text": 0, "no_mouse": 0, "failed": 0}

    for i, rel_path in enumerate(todo):
        mp = mask_path_for(rel_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        img_path = ROOT / rel_path

        try:
            # Check if we have keypoints for this image
            ann_data = ann_lookup.get(rel_path)
            points = None
            if ann_data:
                points = extract_keypoints(ann_data)

            if points is not None and len(points) >= 3:
                # Use keypoint-guided segmentation
                mask = segment_with_keypoints(model, processor, img_path, points)
                stats["kp"] += 1
            else:
                # Fallback to text prompt
                mask = segment_with_text(model, processor, img_path)
                stats["text"] += 1

            if mask.max() == 0:
                stats["no_mouse"] += 1

            # Save mask
            mask_img = Image.fromarray(mask, mode="L")
            mask_img.save(str(mp))

        except Exception as e:
            print(f"  FAILED {rel_path}: {e}", flush=True)
            stats["failed"] += 1
            continue

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(todo)}] "
                f"{rate:.1f} img/s, "
                f"ETA {eta/60:.0f}min | "
                f"kp={stats['kp']} text={stats['text']} "
                f"no_mouse={stats['no_mouse']} failed={stats['failed']} | "
                f"{rel_path}",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\nDone! Processed {len(todo)} images in {elapsed/60:.1f} min", flush=True)
    print(f"  Keypoint-guided: {stats['kp']}", flush=True)
    print(f"  Text-only fallback: {stats['text']}", flush=True)
    print(f"  No mouse detected: {stats['no_mouse']}", flush=True)
    print(f"  Failed: {stats['failed']}", flush=True)
    print(f"  Masks saved to: {MASK_ROOT}/", flush=True)


if __name__ == "__main__":
    main()
