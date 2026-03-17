"""
Batch SAM3 mouse segmentation — text prompt only.
Saves:
  1. Binary mask:       masks_final/{split}/session/cam/Frame_X_mask.png
  2. Segmented mouse:   masks_final/{split}/session/cam/Frame_X_rgba.png  (RGBA, transparent bg)

The text-only prompt "mouse" gives SAM3 the best semantic understanding
of the whole animal, producing cleaner masks than point prompts.

Supports resume (skips already-processed images).
"""
import os
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ROOT = Path("/home/user/mouse_labels/jarvis_merge/merged_output2")
BPE_PATH = str(ROOT / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
MASK_ROOT = ROOT / "masks_final"
TEXT_PROMPT = "mouse"
CONFIDENCE_THRESHOLD = 0.3


def build_model():
    print("Building SAM3 model...", flush=True)
    model = build_sam3_image_model(bpe_path=BPE_PATH)
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
    print(f"Model ready on {next(model.parameters()).device}", flush=True)
    return model, processor


def collect_images(splits=("train", "val")):
    all_images = []
    for split in splits:
        split_dir = ROOT / split
        if not split_dir.exists():
            continue
        for jpg in sorted(split_dir.rglob("*.jpg")):
            all_images.append(jpg.relative_to(ROOT))
    return all_images


def out_paths(rel_path):
    """Return (mask_path, rgba_path) for a given image relative path."""
    stem = rel_path.stem
    parent = MASK_ROOT / rel_path.parent
    return (
        parent / f"{stem}_mask.png",
        parent / f"{stem}_rgba.png",
    )


def segment_image(model, processor, img_path):
    """
    Run SAM3 text prompt on one image.
    Returns: (mask_uint8 H×W 0/255, rgba H×W×4 uint8) or (None, None) if no mouse.
    """
    image = Image.open(img_path).convert("RGB")
    img_np = np.array(image)  # (H, W, 3)
    H, W = img_np.shape[:2]

    with torch.inference_mode():
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)

    masks = output["masks"]
    scores = output["scores"]

    if len(masks) == 0:
        return None, None

    # Combine all detected mouse masks
    combined = np.zeros((H, W), dtype=np.uint8)
    for m in masks:
        if hasattr(m, 'cpu'):
            m_np = m.squeeze(0).cpu().numpy()
        else:
            m_np = np.asarray(m)
            if m_np.ndim == 3:
                m_np = m_np.squeeze(0)
        combined = np.maximum(combined, m_np.astype(np.uint8))

    mask_255 = combined * 255

    # Build RGBA: mouse pixels on transparent background
    alpha = mask_255  # 0 or 255
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = img_np
    rgba[:, :, 3] = alpha

    return mask_255, rgba


def main():
    all_images = collect_images()
    total = len(all_images)
    print(f"Total images: {total}", flush=True)

    # Resume: skip images where BOTH mask and rgba exist
    todo = []
    for rel in all_images:
        mp, rp = out_paths(rel)
        if not mp.exists() or not rp.exists():
            todo.append(rel)

    done = total - len(todo)
    print(f"Already done: {done}", flush=True)
    print(f"Remaining: {len(todo)}", flush=True)

    if not todo:
        print("Nothing to do!", flush=True)
        return

    model, processor = build_model()

    t0 = time.time()
    no_mouse = 0
    failed = 0

    for i, rel in enumerate(todo):
        mp, rp = out_paths(rel)
        mp.parent.mkdir(parents=True, exist_ok=True)

        try:
            mask, rgba = segment_image(model, processor, ROOT / rel)

            if mask is None:
                # No mouse — save empty mask and transparent image
                H, W = 2200, 3208  # default
                img = Image.open(ROOT / rel)
                W, H = img.size
                mask = np.zeros((H, W), dtype=np.uint8)
                rgba = np.zeros((H, W, 4), dtype=np.uint8)
                no_mouse += 1

            Image.fromarray(mask, mode="L").save(str(mp))
            Image.fromarray(rgba, mode="RGBA").save(str(rp))

        except Exception as e:
            print(f"  FAILED {rel}: {e}", flush=True)
            failed += 1
            continue

        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(todo)}] "
                f"{rate:.1f} img/s  ETA {eta/60:.0f}min  "
                f"no_mouse={no_mouse} failed={failed} | "
                f"{rel}",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\nDone! {len(todo)} images in {elapsed/60:.1f} min", flush=True)
    print(f"  No mouse: {no_mouse}  Failed: {failed}", flush=True)
    print(f"  Output: {MASK_ROOT}/", flush=True)


if __name__ == "__main__":
    main()
