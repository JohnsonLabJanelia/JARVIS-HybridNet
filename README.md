# JARVIS-HybridNet

<p align="center">
<img src="docs/banner_hybridnet.png" alt="banner" width="70%"/>
</p>

JARIS-HybridNet is a Python library for precise multi-view markerless 3D motion capture in complex environments. Our hybrid 2D-and 3D-CNN pose estimation network is designed to provide precise and robust tracking even under heavy occlusions.

The primary goal of JARVIS to make markerless 3D pose estimation easy to use and quick to implement. With that in mind our network architecture was specifically designed to work with small manually annotated datasets. To make the process of acquiring multi-camera recordings and annotated training data as painless as possible we also provide our [AcquisitionTool]() and our [AnnotationTool](). Assuming you have a set of [FLIR Machine Vision Cameras](https://www.flir.eu/iis/machine-vision/) those tools will enable you to set up your motion capture pipeline without writing a single line of code.  


Check out our [Getting Started Guide](https://jarvis-mocap.github.io/jarvis-docs//2021-10-28-gettingstarted.html) if you want to learn more.  

<p align="center">
<img src="docs/Pytorch_Vid.gif" alt="banner" width="70%"/>
</p>

## Temporal & Physics Extensions

This branch adds improvements to reduce keypoint jitter and improve 3D accuracy. There are two separate things here:

1. **Changes to HybridNet itself** (bone length loss, cross-view attention) — these modify how HybridNet trains and runs. The trained model weights include these changes.
2. **A separate post-processing network** (temporal transformer) — this is a completely independent network that runs AFTER HybridNet to smooth its output. It has its own weights, its own training, and is optional.

### Architecture Overview

```
                    NETWORK 1: HybridNet (modified)
                    ================================
                    Trained on labeled dataset with ground truth.
                    Runs on each frame independently.

  16 camera       EfficientTrack    ReprojectionLayer     V2VNet        3D keypoints
  images -------> 2D keypoint ----> (cross-view attn) --> 3D refine --> (x,y,z) per joint
  (per frame)     detection         + camera fusion       + bone loss   + confidence
                                                          constrains
                                                          training
                                          |
                                          | output: data3D.csv
                                          | (standard JARVIS format, one row per frame)
                                          v
                    NETWORK 2: Temporal Transformer (separate, optional)
                    =================================================
                    Trained on HybridNet's own predictions from unlabeled video.
                    Runs on sequences of frames, not individual frames.

  Window of 16    Transformer       Smoothed
  consecutive --> encoder with  --> 3D keypoints
  HybridNet       temporal +        (less jitter)
  predictions     spatial attention
  (from csv/npz)
                                          |
                                          v
                                   smoothed data3D.csv
```

**Key distinction:**
- HybridNet is the main network. It sees images and outputs 3D keypoints. You MUST have this.
- The temporal transformer is a small optional post-processing network (809K params). It never sees images — it only takes HybridNet's 3D keypoint output and smooths it across time. You can skip it entirely and just use HybridNet's output.

---

### 1. Bone Length Loss (inside HybridNet)

A new loss term added to HybridNet's training. It penalizes predicted bone lengths that deviate from reference statistics computed from the training ground truth. This does NOT change the network architecture — it only changes what the network learns during training. The resulting trained weights produce more physically plausible predictions.

**Result:** MPJPE improved from 15.3mm to 12.6mm (**-17.4%**), bone length consistency improved by **41%**, with no loss in inference speed.

**Usage:**
```bash
# 1. Compute bone stats from your training data (one time)
python tools/compute_bone_lengths.py --project mouseJan30

# 2. Train HybridNet with bone loss enabled
python tools/train_hybridnet_improved.py --project mouseJan30 \
    --bone_weight 0.1 --epochs 8 --mode 3D_only
```

Config: set `HYBRIDNET.BONE_LENGTH_LOSS_WEIGHT: 0.1` to enable.

### 2. Cross-View Attention (inside HybridNet)

Replaces the naive averaging of 2D heatmaps across camera views (in HybridNet's ReprojectionLayer) with learned per-camera per-joint attention weights. A lightweight MLP (5.2K parameters) computes camera reliability from heatmap statistics, producing softmax-normalized weights. This IS a change to the network architecture — models trained with this need `USE_CROSS_VIEW_ATTENTION: true` at inference time too.

**Usage:** Set `HYBRIDNET.USE_CROSS_VIEW_ATTENTION: true` in the project config.

Note: Requires gradient checkpointing or reduced grid resolution for training on 24GB GPUs.

### 3. Temporal Transformer (separate network, optional post-processing)

A completely separate neural network that smooths HybridNet's output over time. It does NOT modify HybridNet in any way. You can use HybridNet with or without it.

**Why it exists:** HybridNet processes each video frame independently. It has no concept of "the previous frame" or "the next frame." This means its predictions can jump around between frames even when the animal moves smoothly. The temporal transformer takes a window of 16 consecutive predictions and smooths them using self-attention.

**How it is trained (requires unlabeled video only):**

```
 Unlabeled          HybridNet             Raw 3D              Savgol filter        Temporal
 multi-camera  ---> (already trained, --> predictions    ---> (offline          --> transformer
 video              frozen)               per frame           smoothing)            learns to map
                                          (.npz files)        = training targets    raw --> smooth
```

The temporal transformer does NOT need ground truth labels. It learns from HybridNet's own predictions on any video:
1. Run HybridNet on a video to get dense per-frame predictions
2. Apply a Savitzky-Golay filter offline to create smoothed versions
3. Train the transformer to produce the smoothed version from the raw version

This means you can train it on any video you have, even without annotations.

**Training:**
```bash
# Step 1: Run HybridNet on video (outputs .npz + data3D.csv)
python tools/predict_video_for_temporal.py --project mouseJan30 \
    --video_dir /path/to/16cam/video \
    --calib_dir /path/to/calibration \
    --output predictions/my_video/train \
    --start_frame 0 --num_frames 5000

# Step 2: Create smoothed training targets
python tools/prepare_temporal_data.py --project mouseJan30 \
    --output predictions/my_video

# Step 3: Train the temporal transformer
python tools/train_temporal.py --project mouseJan30 \
    --predictions predictions/my_video \
    --epochs 100 --lr 0.0003
```

**At inference time, you have two options:**

Option A — HybridNet only (simpler):
```bash
python tools/predict_video_for_temporal.py --project mouseJan30 \
    --video_dir /path/to/video --calib_dir /path/to/calibration \
    --output my_predictions
# Output: my_predictions/data3D.csv (standard JARVIS format)
```

Option B — HybridNet + temporal transformer (smoother):
```bash
# Same as above, then run the temporal transformer on the output
# (inference script TBD — currently only training pipeline is implemented)
```

**Output formats:**
- `data3D.csv` — standard JARVIS format: header row with joint names (x4 each), subheader `x,y,z,confidence`, one row per frame. This is the usable output for downstream analysis.
- `.npz` files — per-frame numpy arrays used internally by the temporal training pipeline. Not needed for regular use.

---

### Benchmark Results (mouseJan30, 275 val samples)

| Metric | Baseline | + Bone Loss | Improvement |
|--------|----------|-------------|-------------|
| MPJPE (mean) | 15.29 mm | 12.63 mm | **-17.4%** |
| MPJPE (median) | 2.40 mm | 2.35 mm | -2.1% |
| MPJPE (95th percentile) | 44.13 mm | 38.78 mm | -12.1% |
| Bone length std | 9.05 mm | 5.38 mm | **-40.6%** |
| Inference speed | 5.1 FPS | 5.2 FPS | same |

### Tools

| Script | Purpose |
|--------|---------|
| **HybridNet training** | |
| `tools/compute_bone_lengths.py` | Compute bone length reference stats from training GT |
| `tools/train_hybridnet_improved.py` | Fine-tune HybridNet with bone loss / cross-view attention |
| `tools/evaluate_baseline.py` | Evaluate model with per-joint MPJPE, bone consistency, speed |
| **Video prediction** | |
| `tools/predict_video_for_temporal.py` | Run HybridNet 3D prediction on multi-camera MP4 videos |
| **Temporal transformer training** | |
| `tools/prepare_temporal_data.py` | Create smoothed training targets from HybridNet predictions |
| `tools/train_temporal.py` | Train the temporal refinement transformer |
| `tools/run_full_pipeline.sh` | Run full pipeline: evaluate + extract + smooth + train temporal |
| **Visualization** | |
| `tools/plot_comparison.py` | Generate comparison plots between models |

## Install Instructions

- Clone the repository with
```
git clone https://github.com/JARVIS-MoCap/JARVIS-HybridNet.git
cd JARVIS-HybridNet
```

- Make sure [Anaconda](https://www.anaconda.com/products/individual) is installed on your machine.

- Environment update:
```
conda create -n jarvis python=3.10
conda activate jarvis
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

- Setup the jarvis Anaconda environment and activate it (OUTDATED)
```
conda create -n jarvis python=3.9  pytorch=1.10.1 torchvision cudatoolkit=11.3 notebook  -c pytorch
conda activate jarvis
```

- Make sure your setuptools package is up to date \
  `pip install -U setuptools==59.5.0`

- Install JARVIS
  `pip install -e .`
  
 - To be able to use the **optional** TensorRT acceleration install [Torch-TensorRT](https://github.com/NVIDIA/Torch-TensorRT) and the [TensorRT] pip package with:
```
pip install nvidia-pyindex
pip install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases
pip install --upgrade nvidia-tensorrt
```
- If you want to be able to use TensorRT you also have to add `libnvinfer.so` to the `PATH` variable. This is not required if you're not using TensorRT acceleration.

# Contact
JARVIS was developed at the **Neurobiology Lab of the German Primate Center ([DPZ](https://www.dpz.eu/de/startseite.html))**.
If you have any questions or other inquiries related to JARVIS please contact:

Timo Hüser - [@hueser_timo](https://mobile.twitter.com/hueser_timo) - timo.hueser@gmail.com

