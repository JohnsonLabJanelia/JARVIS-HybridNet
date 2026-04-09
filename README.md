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

This branch adds three architectural improvements to reduce keypoint jitter and improve 3D accuracy, all backward-compatible and disabled by default.

### Bone Length Loss (BoneLengthLoss)

Penalizes predicted bone lengths that deviate from reference statistics computed from the training ground truth. Each of the 24 skeleton bones has a learned mean and standard deviation; deviations are penalized as normalized squared error during training.

**Result:** MPJPE improved from 15.3mm to 12.6mm (**-17.4%**), bone length consistency improved by **41%**, with no loss in inference speed.

**Usage:**
```bash
# 1. Compute bone stats from your training data
python tools/compute_bone_lengths.py --project mouseJan30

# 2. Train with bone loss enabled
python tools/train_hybridnet_improved.py --project mouseJan30 \
    --bone_weight 0.1 --epochs 8 --mode 3D_only
```

Config: set `HYBRIDNET.BONE_LENGTH_LOSS_WEIGHT: 0.1` to enable.

### Cross-View Attention (CrossViewAttention)

Replaces the naive averaging of 2D heatmaps across camera views with learned per-camera per-joint attention weights. A lightweight MLP (5.2K parameters) computes camera reliability from heatmap statistics (max, mean, std, energy), producing softmax-normalized weights. This lets the network learn which cameras provide the best signal for each keypoint.

**Usage:** Set `HYBRIDNET.USE_CROSS_VIEW_ATTENTION: true` in the project config.

Note: Requires gradient checkpointing or reduced grid resolution for training on 24GB GPUs (the 100^3 grid with 16 cameras is memory-intensive during backprop).

### Temporal Refinement Transformer (TemporalRefinementTransformer)

A 4-layer transformer encoder (809K parameters) that takes a sliding window of T consecutive frames of 3D keypoint predictions and refines them using self-attention over both temporal and spatial (joint) dimensions. Each (frame, joint) pair is a token with learned temporal and spatial positional encodings. The output is a residual correction added to the input predictions.

**Loss function:** Position MSE + bone length consistency + velocity smoothness (acceleration penalty).

**Usage:**
```bash
# 1. Extract dense predictions from a multi-camera video
python tools/predict_video_for_temporal.py --project mouseJan30 \
    --video_dir /path/to/video --calib_dir /path/to/calibration \
    --output predictions/train --num_frames 1000

# 2. Train the temporal transformer
python tools/train_temporal.py --project mouseJan30 \
    --predictions predictions/ --epochs 100
```

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
| `tools/compute_bone_lengths.py` | Compute bone length reference stats from training GT |
| `tools/train_hybridnet_improved.py` | Fine-tune HybridNet with bone loss / cross-view attention |
| `tools/evaluate_baseline.py` | Evaluate model with per-joint MPJPE, bone consistency, speed |
| `tools/predict_video_for_temporal.py` | Run 3D prediction on multi-camera MP4 videos |
| `tools/prepare_temporal_data.py` | Generate training data for temporal transformer |
| `tools/train_temporal.py` | Train the temporal refinement transformer |
| `tools/run_full_pipeline.sh` | Run full eval + extraction + temporal pipeline |
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

