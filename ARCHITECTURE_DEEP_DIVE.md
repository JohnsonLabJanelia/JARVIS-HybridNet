# JARVIS-HybridNet Architecture Deep Dive & Improvement Roadmap

## Architecture Summary

### Full Pipeline
```
Input: N camera images (e.g., 16 × 3208×2200)
  ↓
CenterDetect: resize to 256×256 → EfficientNet+BiFPN → 1-channel heatmap → find center peak
  ↓ (triangulate 2D centers → 3D center, reproject to get per-camera crop centers)
  ↓
KeypointDetect: crop 320×320 per camera → EfficientNet+BiFPN → NUM_JOINTS heatmaps (160×160)
  ↓
ReprojectionLayer: project 2D heatmaps into 3D voxel grid (200³), mean-pool across cameras
  ↓
V2V-Net: 3D encoder-decoder (Conv3d + InstanceNorm + residual blocks)
  ↓
Soft-argmax: weighted mean over 3D volume → (NUM_JOINTS, 3) coordinates + confidence
```

### Tensor Shapes (16 cameras, 4 joints, 320px crop, 120mm ROI @ 1mm spacing)
| Stage | Shape | Size |
|-------|-------|------|
| Raw images | (16, 3, 2200, 3208) | ~330 MB |
| CenterDetect input | (16, 3, 256, 256) | 12 MB |
| CenterDetect heatmap | (16, 1, 128, 128) | 1 MB |
| KeypointDetect crops | (16, 3, 320, 320) | 15 MB |
| 2D heatmaps | (16, 4, 160, 160) | 6.5 MB |
| 3D heatmap (reprojected) | (4, 60, 60, 60) | 3.5 MB |
| V2V-Net output | (4, 60, 60, 60) | 3.5 MB |
| Final 3D coords | (4, 3) | 48 bytes |

### Key Hyperparameters
- `GRID_SPACING`: mm per voxel (auto: animal_size/85)
- `ROI_CUBE_SIZE`: 3D volume extent in mm (auto: 1.25× animal extent)
- `BOUNDING_BOX_SIZE`: 2D crop size in px (auto: 1.2× 98th percentile animal size, 64-aligned)
- `MODEL_SIZE`: small/medium/large → EfficientNet-B0/B1/B3

### Architectural Assumptions
1. **Single animal** — CenterDetect finds ONE center, soft-argmax is unimodal
2. **Fixed camera count** — NUM_CAMERAS set at project init, cannot vary per frame
3. **Fixed image resolution** — all cameras must match
4. **Synchronized frames** — one frame per camera per timestamp
5. **Known calibration** — camera matrices + distortion required

## Competitor Comparison

| Tool | Stars | Approach | Multi-animal | Semi-supervised | Active Learning | 3D Method |
|------|-------|----------|-------------|-----------------|-----------------|-----------|
| DeepLabCut | 5,537 | 2D + triangulation | Yes (maDLC) | No | Yes | Post-hoc (Anipose) |
| SLEAP | 563 | 2D only | Yes | No | Yes (best GUI) | No native 3D |
| Lightning Pose | 289 | 2D + MVT | No | **Yes** | Via uncertainty | Triangulation |
| DANNCE | 247 | Volumetric 3D | Yes (s-DANNCE) | Yes (s-DANNCE) | No | End-to-end 3D CNN |
| **JARVIS** | 36 | **Hybrid 2D+3D** | No | No | No | End-to-end 3D CNN |

### JARVIS's Unique Strength
Hybrid architecture: explicit 2D heatmap generation (interpretable, debuggable) + 3D volumetric refinement (captures pose priors). Neither DLC (no 3D learning) nor DANNCE (monolithic) has this.

### JARVIS's Gaps
1. No semi-supervised learning (Lightning Pose has 3 unsupervised losses)
2. No active learning workflow (DLC, SLEAP have this built in)
3. No uncertainty quantification (Lightning Pose has Bayesian ensembles)
4. No temporal modeling (frame-by-frame, no smoothing)
5. No multi-animal support
6. No foundation model pre-training (DLC has SuperAnimal)

## Improvement Roadmap (Ranked by Impact/Effort)

### Tier 1: Low Effort, High Impact (do before paper)

**1. Triangulation Residual Loss** — self-supervised from unlabeled video
- Minimize smallest singular value of triangulation matrix A
- With 16 cameras: A is 32×4, one SVD per joint per frame
- Published: competitive accuracy with only 5% labeled data (NeurIPS 2023)
- ~50 lines in loss.py
- [arXiv: Triangulation Residual Loss](https://openreview.net/forum?id=gLwjBDsE3G)

**2. Learned per-view confidence weights** in ReprojectionLayer
- Currently: mean-pool across all cameras (occlusion adds noise)
- Fix: small MLP per camera → scalar confidence → weighted mean-pool
- ~30 lines in repro_layer.py
- From: [Iskakov et al., Learnable Triangulation](https://arxiv.org/abs/1905.05754)

**3. Best-val checkpoint selection**
- Currently: saves only `_final.pth` at last epoch
- Fix: track best_val_acc, save `_best.pth`
- 10 lines in efficienttrack.py + hybridnet.py

**4. Vectorize loss + accuracy** (10-20x faster training)
- HybridNet loss: double Python loop → single masked MSE
- Accuracy: .cpu().numpy() per batch → batched GPU ops

**5. Bone length regularization loss**
- Penalize deviations from expected bone lengths (learned from training data)
- Prevents physically impossible poses
- ~30 lines, add as auxiliary loss

### Tier 2: Moderate Effort, High Impact (strengthen paper)

**6. Lightning Pose semi-supervised losses**
- Temporal continuity: penalize frame-to-frame jumps on unlabeled video
- PCA pose plausibility: predicted poses stay in learned subspace
- Multi-view consistency: 2D predictions from different views must triangulate
- ~200 lines, requires video-level training
- [Nature Methods 2024](https://www.nature.com/articles/s41592-024-02319-1)

**7. Replace imgaug with torchvision.transforms.v2**
- imgaug: CPU-only, 5-15ms/sample (20-40% of data loading time)
- torchvision.transforms.v2: GPU-accelerated, 3-5x faster
- Add stronger augmentation: CutOut, HSV jitter, elastic deformation

**8. Ensemble training + uncertainty**
- Train 5 models with different seeds
- Ensemble variance = calibrated uncertainty per keypoint
- Feed into Kalman smoother for post-processing
- Enables confidence-based active learning frame selection

**9. Replace V2V-Net with orthographic projection** (Faster VoxelPose)
- Project 3D volume → three 2D planes (XY, XZ, YZ)
- Use 2D CNN instead of 3D CNN
- 10x speedup for 3D stage, minimal accuracy loss
- [ECCV 2022](https://arxiv.org/abs/2207.10955)

### Tier 3: Major Effort, Transformative (post-paper)

**10. Replace V2V-Net with algebraic learnable triangulation**
- Skip 3D volume entirely
- Learn per-joint confidence per view, do weighted SVD triangulation
- Much faster, much less memory, competitive accuracy
- [Iskakov et al., ICCV 2019](https://arxiv.org/abs/1905.05754)

**11. Cross-view transformer fusion** (MVGFormer-style)
- Epipolar attention between camera views
- Better generalization across camera rigs
- [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liao_Multiple_View_Geometry_Transformers_for_3D_Human_Pose_Estimation_CVPR_2024_paper.html)

**12. SuperAnimal-style pre-training**
- Pre-train 2D backbone on Quadruped-80K dataset (45+ species)
- 10-100x data efficiency when fine-tuning
- Requires backbone swap (EfficientNet → ResNet50/HRNet)
- [Nature Communications 2024](https://www.nature.com/articles/s41467-024-48792-2)

**13. Multi-animal support**
- s-DANNCE approach: skeleton-aware GNN + anatomical constraints
- Prevents chimeric predictions (body parts assigned to wrong animal)
- [Nature Machine Intelligence 2024](https://www.nature.com/articles/s42256-023-00776-5)

**14. Backbone upgrade: ViTPose-S or HRNet-W32**
- ViTPose++ has SOTA on animal benchmarks (AP-10K, APT-36K)
- HRNet maintains high-res features (no upsampling needed)
- [arXiv:2212.04246](https://arxiv.org/abs/2212.04246)

## Key Research References

- Triangulation Residual Loss — [NeurIPS 2023](https://openreview.net/forum?id=gLwjBDsE3G)
- Learnable Triangulation — [ICCV 2019](https://arxiv.org/abs/1905.05754)
- Lightning Pose — [Nature Methods 2024](https://www.nature.com/articles/s41592-024-02319-1)
- MVGFormer — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liao_Multiple_View_Geometry_Transformers_for_3D_Human_Pose_Estimation_CVPR_2024_paper.html)
- MV-SSM (Mamba) — [CVPR 2025](https://arxiv.org/abs/2509.00649)
- Faster VoxelPose — [ECCV 2022](https://arxiv.org/abs/2207.10955)
- RTMPose — [arXiv:2303.07399](https://arxiv.org/abs/2303.07399)
- ViTPose++ — [arXiv:2212.04246](https://arxiv.org/abs/2212.04246)
- SuperAnimal — [Nature Communications 2024](https://www.nature.com/articles/s41467-024-48792-2)
- s-DANNCE — [Nature Machine Intelligence 2024](https://www.nature.com/articles/s42256-023-00776-5)
- SelfPose3d — [CVPR 2024](https://arxiv.org/abs/2404.02041)
- Uncertainty-Aware Multi-View — [arXiv:2510.09903](https://arxiv.org/abs/2510.09903)
