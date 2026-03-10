# JARVIS-HybridNet Optimization Plan

## Context

We are optimizing JARVIS for the RED active learning workflow: annotate small set → train → predict → import → correct → retrain. Training must work on both Apple Silicon (local iteration) and CUDA (cluster production). The key metric is **time per active learning iteration**, not just training speed.

## Immediate Fixes (Do First — Benefits All Platforms)

### 1. Save best validation checkpoint (not just final)
**Impact: Quality++ | Effort: 1 hour**

Currently only saves `_final.pth` at last epoch. Small datasets overfit, so best val is often mid-training. Track `best_val_acc` and save `_best.pth`.

Files: `efficienttrack.py` (line 298-305), `hybridnet.py` (line 266-274)

### 2. Vectorize HybridNet loss function
**Impact: 10-20x faster loss | Effort: 30 min**

Current code (`hybridnet/loss.py:15-22`):
```python
for i,gt_batch in enumerate(gt):
    for j, gt_single in enumerate(gt_batch):
        if torch.sum(gt_single) > 1:
            loss += torch.mean(((pred[i][j] - gt_single)**2))
```

Replace with:
```python
mask = (torch.sum(gt, dim=-1) > 1).unsqueeze(-1).float()
loss = torch.mean(((pred - gt)**2) * mask)
```

### 3. Vectorize accuracy calculation
**Impact: 50-70% faster per epoch | Effort: 30 min**

EfficientTrack accuracy (`efficienttrack.py:376-389`) calls `.cpu().numpy()` every batch, forcing GPU sync. HybridNet accuracy (`hybridnet.py:225-234`) uses double Python loops.

Replace both with batched GPU operations.

### 4. Cache per-batch constants
**Impact: 15-25% less overhead | Effort: 15 min**

`img_size = torch.tensor(self.cfg.DATASET.IMAGE_SIZE).float().to(self.device)` is created every batch in HybridNet training. Cache it in `__init__` or before the epoch loop.

### 5. Pre-compute Gaussian heatmap kernels
**Impact: 2-3x faster data loading | Effort: 30 min**

`dataset2D.py` HeatmapGenerator recomputes the Gaussian kernel every `__getitem__` call despite it being constant. Move to `__init__`.

Same issue with 3D meshgrid in `dataset3D.py:322-346`.

### 6. Pre-allocate annotation arrays
**Impact: 5-10x faster annotation loading | Effort: 15 min**

`datasetBase.py:102-134` uses `np.append()` in a loop (O(n²) allocations). Pre-allocate based on `len(coco_annotations)`.

## Medium-Term Improvements (1-2 Days)

### 7. Replace imgaug with torchvision.transforms.v2
**Impact: 3-5x faster augmentation | Effort: 4-6 hours**

imgaug is the single largest bottleneck in data loading (5-15ms/sample, CPU-only). torchvision.transforms.v2 supports GPU acceleration and is actively maintained.

### 8. Add stronger augmentation for small datasets
**Impact: Quality++ | Effort: 2-4 hours**

Add: CutOut/CoarseDropout, HSV color jitter, elastic deformation. Follow RTMPose two-stage strategy (strong augmentation first 70%, weak last 30%).

### 9. Fine-tuning mode for active learning
**Impact: 15-100x faster re-training | Effort: 2 hours**

Expose and document: freeze backbone (`mode='3D_only'` or `'last_layers'`), reduce epochs to 10-20 for 2D and 5-10 for 3D, use 10x lower LR. The infrastructure exists (`finetune=True` path in `train_interface.py`) but isn't well-connected to the CLI.

### 10. Add image caching for small datasets
**Impact: Eliminates I/O bottleneck | Effort: 1-2 hours**

For datasets with <1000 images (common in active learning), cache all decoded images in memory at dataset init. Eliminates cv2.imread every epoch.

## Architectural Improvements (1-2 Weeks)

### 11. Triangulation Residual Loss (self-supervised)
**Impact: 10-20x less labeled data needed | Effort: 1-2 weeks**

Use unlabeled multi-view video as free training signal. Minimize smallest singular value of triangulation matrix across views (Feng et al.). RED has 16 calibrated cameras and thousands of unlabeled frames.

### 12. Pseudo-label self-training
**Impact: 5-10x less labeled data needed | Effort: 1 week**

After initial training, predict on unlabeled frames. Accept predictions with low multi-view reprojection error as pseudo-labels. Retrain on expanded dataset.

### 13. Replace V2V-Net with learnable triangulation
**Impact: 5-10x faster 3D stage, less memory | Effort: 2-4 weeks**

Algebraic learnable triangulation (Iskakov et al.) learns per-joint confidence weights for each view, then does weighted DLT. No voxel grid needed. Dramatically simpler, faster, and compatible with RED's existing triangulation code.

## MPS-Specific (Mac Training)

### 14. Conditional pin_memory and num_workers ✅ DONE
Already committed: pin_memory=False on MPS, num_workers=0 for small datasets.

### 15. VAL_INTERVAL = 5 ✅ DONE
Already committed: validate every 5 epochs instead of every 1.

### 16. float32 everywhere ✅ DONE
Already committed: .float() before .to(device) in HybridNet training loops.

### 17. Add persistent_workers for larger datasets
When num_workers > 0, set `persistent_workers=True` to avoid process respawn overhead on macOS (spawn vs fork).

### 18. torch.mps.empty_cache() between epochs
Releases unoccupied cached memory. Can dramatically improve iteration time in some cases.

## CUDA-Specific (Cluster Training)

### 19. Increase batch size
JARVIS defaults to batch_size=4 for 2D and 1 for 3D. On cluster GPUs with 24-80GB VRAM, increase significantly (16-64 for 2D, 4-8 for 3D).

### 20. Enable cuDNN benchmark
Already in code: `torch.backends.cudnn.benchmark = True`. Verify it's active.

### 21. Use mixed precision on CUDA
Unlike MPS, CUDA GPUs have tensor cores that benefit from FP16. Add `torch.autocast('cuda')` with GradScaler for 1.5-2x training speedup.

## CoreML Inference (Mac, Future)

### 22. Export 2D models to CoreML
Export CenterDetect + KeypointDetect via coremltools. Use ImageType input for CVPixelBuffer zero-copy from RED's VideoToolbox decoder. Expected 5-15ms per view.

### 23. Skip V2V-Net for inference
Use RED's Eigen DLT triangulation from 2D predictions instead of running the full 3D pipeline. Faster, simpler, and already implemented in red_math.h.

## References

- RTMPose: Two-stage training, CSPNeXt backbone (arxiv 2303.07399)
- Triangulation Residual Loss (OpenReview, ICLR)
- Lightning Pose: Semi-supervised animal pose (Nature Methods 2024)
- SuperAnimal: Foundation models for animal pose (Nature Communications 2024)
- Learnable Triangulation of Human Pose (Iskakov et al.)
- Active Transfer Learning for Pose Estimation (WACV 2024)
