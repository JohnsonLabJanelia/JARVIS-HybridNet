# Multiview Augmentation Project

This folder is the clean workflow for:

1. Generating SAM3 masks
2. Scoring keypoint-mask alignment
3. Building pass-threshold datasets (`pass16` down to `pass1`)
4. Selecting one threshold as the active augmentation set

## Goal

Keep original dataset unchanged, and augment only confident camera views.

- A camera view is **good** if:
  - `kp_inside_pct >= 90`
  - `mask_area_px >= 1000`
  - `num_kp_visible > 0`
- A frame is in `passK` if it has at least `K` good cameras.
- For augmentation, use only **good views** from those frames.

## Scripts

- `scripts/build_threshold_manifests.py`
  - Reads `reports/alignment_image_metrics.csv`
  - Builds manifests for every threshold: `pass16 ... pass1`

- `scripts/select_threshold.py`
  - Activates a chosen threshold (e.g., `pass10`) into `multiview_aug/active/`

- `scripts/build_jarvis_aug_dataset.py`
  - Builds a training-ready augmented dataset from a chosen threshold (`passK`)
  - Writes augmented images, masks, and JSONL manifests for train/val
  - Supports multiple augmentation strategies

## Run

From repo root:

```bash
python multiview_aug/scripts/build_threshold_manifests.py
python multiview_aug/scripts/select_threshold.py --threshold 10
python multiview_aug/scripts/build_jarvis_aug_dataset.py --threshold 16 --strategies color_jitter,noise_blur,affine_bgmix,occlusion --num-per-strategy 1 --include-original
```

More augmentation strategies available in `build_jarvis_aug_dataset.py`:

- `color_jitter`
- `noise_blur`
- `affine_bgmix`
- `occlusion`
- `gamma_contrast`
- `motion_blur`
- `jpeg_artifacts`
- `shadow`
- `color_temperature`
- `background_replace`
- `mouse_color_jitter` (mouse only)
- `mouse_relight` (mouse only)
- `mouse_gamma` (mouse only)

Example using a richer set:

```bash
python multiview_aug/scripts/build_jarvis_aug_dataset.py \
  --threshold 16 \
  --strategies color_jitter,noise_blur,affine_bgmix,occlusion,gamma_contrast,motion_blur,jpeg_artifacts,shadow,color_temperature,background_replace,mouse_color_jitter,mouse_relight,mouse_gamma \
  --num-per-strategy 1 \
  --include-original
```

## Output Layout

- `multiview_aug/manifests/summary.json`
- `multiview_aug/manifests/pass_at_least_16/`
- `multiview_aug/manifests/pass_at_least_15/`
- ...
- `multiview_aug/manifests/pass_at_least_1/`

Each threshold folder contains:

- `frames.csv`: eligible frames with frame-level stats
- `views_good.csv`: camera views to augment (only good cams)
- `views_all.csv`: all cameras for eligible frames (for reference)
- `views_good.jsonl`: JSONL version for pipelines

Active selection:

- `multiview_aug/active/`
  - `frames.csv`
  - `views_good.csv`
  - `views_all.csv`
  - `views_good.jsonl`
  - `selection.json`

Training-ready augmented output:

- `multiview_aug/jarvis_ready_dataset/pass_at_least_{K}/`
  - `images/...`
  - `masks/...`
  - `annotations/all.jsonl`
  - `annotations/train.jsonl`
  - `annotations/val.jsonl`
  - `dataset_summary.json`

