# tools/

Utility scripts for the full JARVIS pipeline:
SAM3 segmentation → augmentation → dataset patching → training → monitoring.

Source data lives at:
- Local: `/home/user/mouse_labels/jarvis_merge/merged_output2/`
- Cluster (mounted): `/mnt/johnson_lab/doq/mouse_labels/jarvis_merge/`

---

## Pipeline Overview

```
1. segmentation/segment_all_sam3_kp.py              →  masks_kp/
2. segmentation/fix_empty_masks_with_sam3_kp.py     →  masks_fixed/   (repair failures)
3. augmentation/run_kp_mask_alignment_augmentation.py  →  reports/ + manifests/
4. augmentation/build_threshold_manifests.py         →  multiview_aug/manifests/pass_at_least_K/
5. augmentation/select_threshold.py                  →  multiview_aug/active/
6. augmentation/build_jarvis_aug_dataset.py          →  multiview_aug/jarvis_ready_dataset/pass_at_least_K/
7. augmentation/run_augmentation.py                  →  train/{session}/{cam}/{frame}_aug*.jpg
8. add_aug_framesets.py                              →  annotations/instances_train_aug_framesets.json
9. precompute_kp3d.py                                →  annotations/keypoints3D_{split}.npy
```

**Active dataset:** `augmented_merged_out2_v3_pass10`
- 218k images, 8 sessions, 16 cameras
- 2,427 original framesets → **16,828 framesets** after step 8
- 16,825 training samples (3 filtered by Dataset3D's `len(keypoints3D_bb) > 1` check)

---

## segmentation/

### segment_all_sam3_kp.py
Batch SAM3 segmentation using annotated keypoints as foreground point prompts + bounding box.
Falls back to text-only (`"mouse"`) when no keypoints are available.
```bash
python tools/segmentation/segment_all_sam3_kp.py
```
Output: `masks_kp/{split}/{session}/{cam}/Frame_X_mask.png`

### segment_all_sam3_final.py
Batch SAM3 segmentation using text prompt `"mouse"` only (no keypoint guidance).
```bash
python tools/segmentation/segment_all_sam3_final.py
```
Output: `masks_final/{split}/{session}/{cam}/Frame_X_mask.png` + `Frame_X_rgba.png`

### fix_empty_masks_with_sam3_kp.py
Repairs empty or failed masks identified in the alignment metrics CSV.
```bash
python tools/segmentation/fix_empty_masks_with_sam3_kp.py \
    --metrics reports/alignment_image_metrics.csv
```
Output: `masks_recovered_kp/` + `review_no_mask_fixes_sam3_kp/` (before/after overlays)

---

## augmentation/

### run_kp_mask_alignment_augmentation.py
End-to-end pipeline: validate keypoint↔mask alignment → optionally fix → build manifests.
- Phase 1: per-image alignment (keypoints must be ≥90% inside mask, mask area ≥1000 px)
- Phase 2: per-frame 16-camera consistency
- Phase 3 (optional): conservative mask repair for failed images
- Phase 4: augmentation of passing frames only
```bash
python tools/augmentation/run_kp_mask_alignment_augmentation.py \
    --phase all --num-aug 10 --seed 1337
```
Output: `reports/alignment_image_metrics.csv`, `manifests/`, augmented images.

### build_threshold_manifests.py
Builds `pass_at_least_K` manifests for every threshold K from 1–16.
A frame passes threshold K if ≥K cameras have `kp_inside_pct ≥ 90` AND `mask_area_px ≥ 1000`.
```bash
cd /path/to/merged_output2
python tools/augmentation/build_threshold_manifests.py
```
Output: `multiview_aug/manifests/pass_at_least_K/frames.csv`, `views_good.jsonl`, etc.

### select_threshold.py
Activates a threshold by copying its manifests to `multiview_aug/active/`.
```bash
python tools/augmentation/select_threshold.py --threshold 10
```

### build_jarvis_aug_dataset.py  /  make_augmented_jarvis_dataset.py
Builds a training-ready augmented dataset from a pass-threshold. Applies 13+ strategies:
`background_replace`, `motion_blur`, `noise_blur`, `gamma_contrast`, `color_temperature`,
`mouse_gamma`, `mouse_relight`, `color_jitter`, `affine_bgmix`, `occlusion`,
`jpeg_artifacts`, `shadow`, `mouse_color_jitter`.

`build_jarvis_aug_dataset.py` takes `--threshold` and runs from the dataset root.
`make_augmented_jarvis_dataset.py` is an alternative with `--source-root` / `--output-root` args.

```bash
# From merged_output2 root:
python tools/augmentation/build_jarvis_aug_dataset.py \
    --threshold 10 \
    --strategies color_jitter,noise_blur,affine_bgmix,occlusion,gamma_contrast,motion_blur,\
jpeg_artifacts,shadow,color_temperature,background_replace,mouse_color_jitter,mouse_relight,mouse_gamma \
    --num-per-strategy 1 \
    --include-original
```
Output: `multiview_aug/jarvis_ready_dataset/pass_at_least_10/{images,masks,annotations/}`

**This produced `augmented_merged_out2_v3_pass10`** (218k train images, 2,427 framesets).

### run_augmentation.py
Gradient-background augmentation using a pre-computed candidate list (158 frames).
Composites mouse foreground (via mask) onto random gradient backgrounds with mild affine warp.
```bash
python tools/augmentation/run_augmentation.py --num-aug 10 --seed 1337
```
Output: `train/{session}/{cam}/{frame}_aug{i:02d}.jpg` + `annotations/instances_train_aug.json`

---

## add_aug_framesets.py

**Fixes the critical HybridNet training gap.**

CenterDetect and KeypointDetect train on individual images, so they see all augmented data.
HybridNet requires complete 16-camera framesets — but the augmented images produced by
the pipeline above are listed in `instances_train.json` under `images`, with **no framesets**.
HybridNet's Dataset3D never sees them.

This script groups augmented images by `(session, frame, aug_variant)` and builds proper
16-entry framesets in calibration camera order. Missing cameras fall back to the original
(non-augmented) image for that camera so that `frame_idx == camera_index` alignment is
preserved.

**Key invariant:** every slot in the 16-entry list must have a valid image ID.
A `None` entry would silently shift all subsequent cameras by one index and corrupt 3D geometry.

```bash
cd /path/to/augmented_merged_out2_v3_pass10

python tools/add_aug_framesets.py \
    --input  annotations/instances_train.json \
    --output annotations/instances_train_aug_framesets.json \
    --min-aug 10   # require ≥10 cameras to have the augmented version

# Then activate:
cp annotations/instances_train.json annotations/instances_train_orig_backup.json
cp annotations/instances_train_aug_framesets.json annotations/instances_train.json
```

**Results on `augmented_merged_out2_v3_pass10`:**
- Before: 2,427 framesets (original only) — HybridNet saw none of the augmented images
- After:  16,828 framesets (+14,401 augmented) — HybridNet now trains on all strategies

---

## precompute_kp3d.py

**Offline triangulation cache for Dataset3D.**

Dataset3D triangulates 3D keypoints at runtime for every frameset on every epoch.
With 16,828 framesets across 8 sessions this adds significant startup time.
This script precomputes all 3D keypoints once and saves them to a `.npy` cache.

```bash
python tools/precompute_kp3d.py \
    --dataset /mnt/johnson_lab/doq/mouse_labels/jarvis_merge/augmented_merged_out2_v3_pass10 \
    --split train

python tools/precompute_kp3d.py \
    --dataset /mnt/johnson_lab/doq/mouse_labels/jarvis_merge/augmented_merged_out2_v3_pass10 \
    --split val
```

Output (saved inside the dataset):
- `annotations/keypoints3D_{split}.npy` — shape `(N_framesets, N_joints, 3)`, float32
- `annotations/keypoints3D_{split}_index.json` — `{frameset_key: row_index}`

**Verified results on `augmented_merged_out2_v3_pass10` (train split):**
- Shape: (16828, 24, 3), dtype float32
- 0 NaN, 0 Inf values
- 1 all-zero row (frameset `2026_02_26_13_29_50/Frame_18359` — had zero visible keypoints)
  → harmless: Dataset3D's `len(keypoints3D_bb) > 1` filter drops it automatically
- 16,828 framesets → **16,825 training samples** (3 filtered total)

Note: Dataset3D does not yet load this cache automatically. Integration is a future step.

---

## monitor_training.py

**Live training dashboard.**

Reads TensorBoard event files from any JARVIS run directory and renders a dark-themed
dashboard (Loss, Accuracy, Learning Rate curves). Optionally tails LSF job stdout via
`bpeek` over SSH. Saves a PNG on every refresh cycle.

Works for any network (HybridNet, CenterDetect, KeypointDetect) and any project.
TensorBoard events are accessible locally via the cluster NFS mount at
`/mnt/johnson_lab/doq/`.

**Requirements:** `tensorboard` package must be available. Use the `jarvis` conda env:
```bash
PYTHON=/home/user/anaconda3/envs/jarvis/bin/python3
```

```bash
# Auto-detect latest run, refresh every 5 min, save PNG to /tmp:
$PYTHON tools/monitor_training.py \
    --project mouseJan30 --net HybridNet \
    --jobid 148808346 \
    --no-show --outdir /tmp --interval 300

# Point at a specific run:
$PYTHON tools/monitor_training.py \
    --logdir /mnt/johnson_lab/doq/JARVIS-HybridNet/projects/mouseJan30/logs/HybridNet/Run_20260316-222315 \
    --jobid 148808346 --interval 300

# Interactive window (requires DISPLAY):
$PYTHON tools/monitor_training.py \
    --project mouseJan30 --net HybridNet --interval 60
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--project` | — | JARVIS project name |
| `--net` | `HybridNet` | `HybridNet` / `CenterDetect` / `KeypointDetect` |
| `--logdir` | auto | Explicit path to TF events directory |
| `--jobid` | — | LSF job ID for `bpeek` stdout tail |
| `--ssh-host` | `doq@login1` | SSH host for bpeek |
| `--interval` | `30` | Refresh interval (seconds) |
| `--no-show` | false | Save PNG only, no interactive window |
| `--outdir` | `/tmp` | Where to write dashboard PNGs |

**Note:** JARVIS writes TensorBoard scalars once per epoch. The dashboard will show
"No scalar data yet" until the first epoch completes.

---

## Cluster Training Scripts

Scripts used to submit and set up training on the Janelia cluster.
Live at `/mnt/johnson_lab/doq/` (also on the cluster at `$HOME/`).

### setup_gh200.sh
One-time environment setup for GH200 nodes (ARM aarch64 architecture).
The standard `jarvis_repro` env in `~/miniconda3` is x86 and cannot run on GH200.
```bash
bash /mnt/johnson_lab/doq/setup_gh200.sh
```
- Installs Miniforge3 aarch64 to `~/miniforge3`
- Creates `jarvis_gh200` conda env (Python 3.10)
- Installs PyTorch 2.10 + CUDA 12.8 for aarch64: `--index-url https://download.pytorch.org/whl/cu128`
- Installs JARVIS and dependencies

Two envs now exist on the cluster:
| Env | Location | Architecture | Use for |
|-----|----------|-------------|---------|
| `jarvis_repro` | `~/miniconda3` | x86_64 | A100 / L4 nodes |
| `jarvis_gh200` | `~/miniforge3` | aarch64 | GH200 nodes (480 GB HBM3) |

### train_hybridnet_gh200.sh
Submits HybridNet training on a GH200 node.
```bash
# Submit:
ssh doq@login1 "bsub -q gpu_gh200 -n 12 -gpu 'num=1' \
    -o ~/hybridnet_gh200.log \
    bash ~/train_hybridnet_gh200.sh"

# Check status:
ssh doq@login1 "bjobs <JOBID>"

# Tail output:
ssh doq@login1 "bpeek <JOBID>"
```

**Why GH200:** L4 GPU (22 GB VRAM) OOMs even at batch_size=1 — ReprojectionTools
for 8 sessions alone use ~21.6 GB leaving no room for the forward pass.
GH200 has 480 GB HBM3, training runs at ~4 it/s (~70 min/epoch).

**Current run:** job 148808346, `Run_20260316-222315`, 15 epochs, started Mar 16 22:22.
Config: `weights=None` (train from KeypointDetect init), `mode='medium'`, `batch_size=1`.
