# Keypoint-Mask Alignment & Data Augmentation Plan

## Overview
This plan addresses keypoint-mask alignment issues and establishes a robust data augmentation pipeline for training a 3D tracking model. 

**Core Principle**: 
- **For augmentation**: Only augment images where keypoints align with masks (>90% of visible keypoints inside mask). For misaligned images, either improve the mask or don't augment (but still use original for training).
- **For training**: Use ALL original frames - both augmented and non-augmented.

**Multi-View Requirement**: For 3D tracking with 16 cameras, **all 16 cameras must have good alignment** for a frame to be augmented. If even 1 camera fails alignment:
- Attempt to fix the failed camera(s)
- If fix succeeds → Augment frame
- If fix fails → Don't augment, but still include original frame in training dataset

The goal is to ensure that SAM3-generated masks align with annotated keypoints, handle occlusion cases, and create augmented datasets with **guaranteed consistent keypoint-mask pairs across all views**.

---

## Phase 1: Alignment Validation & Diagnostics

### 1. Load and Map Annotations
- Load `instances_train.json` and `instances_val.json`
- Build lookup: `image_id` → annotation (keypoints, bbox, visibility)
- Map image file paths to annotations
- Handle both train and val splits

### 2. Check Calibration/Projection Alignment
- **Locate calibration files**: Find where calibration data is stored (likely YAML files per camera/session)
- **Map sessions to calibrations**: Identify which sessions/cameras have calibration data
- **Review projection code**: Check existing projection functions (e.g., `project_3d_to_2d` in `segment_3d_model.py`)
- **Verify projection usage**: Ensure keypoints are projected using correct calibration for each session/camera
- **Identify missing calibrations**: Flag sessions/cameras without calibration data

### 3. Validate Keypoint-Mask Alignment

#### 3.1 Project Keypoints onto Mask Image
For each image with both keypoints and mask:
1. **Load mask image**: Read `masks_final/{split}/session/cam/Frame_X_mask.png`
   - Mask is binary: 0 (black/background) or 255 (white/mouse)
   - Same resolution as original image (3208×2200)

2. **Extract visible keypoints from annotation**:
   - Parse keypoints array: `[x1, y1, vis1, x2, y2, vis2, ...]`
   - Filter to visible keypoints only (visibility > 0)
   - Handle negative Y coordinates (session `2024_11_26_17_00_20`): `y_pixel = image_height + y_annotation`

3. **Project keypoints onto mask**:
   - For each visible keypoint `(x, y)`:
     - Check if `(x, y)` is within image bounds `[0, W) × [0, H)`
     - Sample mask value at `(x, y)`: `mask[y, x]`
     - Keypoint is **inside mask** if `mask[y, x] == 255` (white)
     - Keypoint is **outside mask** if `mask[y, x] == 0` (black)

4. **Compute alignment metrics**:
   - **% of visible keypoints inside mask** (target: >90%)
     - `num_inside / num_visible * 100`
   - **Distance from keypoints to nearest mask edge**:
     - For each keypoint outside mask, compute distance to nearest white pixel
     - Use distance transform or nearest neighbor search
     - Should be small (<50 pixels ideally)
   - **Bounding box overlap**: keypoint bbox vs mask bbox (IoU)
     - Compute keypoint bbox from min/max x,y of visible keypoints
     - Compute mask bbox from min/max x,y of white pixels
     - IoU = intersection / union
   - **Mask coverage**: % of keypoint bbox covered by mask
     - Count white pixels within keypoint bbox / total pixels in bbox

5. **Visual validation**:
   - Create overlay image: mask + keypoints plotted
   - Mark keypoints:
     - Green: inside mask
     - Red: outside mask
   - Save visualization for manual inspection

#### 3.2 Flag Misalignments
Categorize and flag:
- **Keypoints outside mask**: Any visible keypoint with `mask[y, x] == 0`
- **Mask doesn't cover keypoint bbox**: IoU < 0.7 or coverage < 80%
- **Large gaps**: Distance from keypoint to mask edge > 100 pixels
- **Mask includes distant regions**: Mask regions >200 pixels from nearest keypoint
- **Occlusion**: Keypoints visible but mask empty or very small (<1000 pixels)

### 4. Multi-View Alignment Validation

#### 4.1 Frame-Level Validation (16 Cameras)
For each frame (same timestamp across all 16 cameras):
1. **Validate each camera individually**:
   - Check keypoint-mask alignment for each camera
   - Compute alignment score per camera: % of keypoints inside mask

2. **Frame-level decision**:
   - **All 16 cameras pass** (>90% alignment each) → ✅ **Include frame**
   - **1+ cameras fail** (<90% alignment) → ❌ **Handle frame** (see strategies below)

3. **Track statistics**:
   - % of frames with all 16 cameras aligned
   - Distribution of failed cameras per frame (which cameras fail most often)
   - Sessions/cameras with consistent failures

#### 4.2 Strategies for Frames with Failed Cameras

**Important**: "Skip frame" means **don't augment**, but **still use original frame for training**.

**Strategy A: Fix Failed Cameras, Then Decide** (Recommended)
- For failed cameras, attempt to fix mask:
  1. Re-run SAM3 with different parameters
  2. Refine mask to include all keypoints (dilation around keypoints)
  3. Use keypoint bbox as hard constraint
- **If fix succeeds** → All 16 cameras now aligned → ✅ **Augment frame**
- **If fix fails** → ❌ **Don't augment**, but **use original frame for training**

**Strategy B: Strict - Only Augment Perfect Frames**
- If any camera fails alignment → ❌ **Don't augment**, but **use original frame for training**
- **Pros**: Only augment frames with guaranteed perfect alignment
- **Cons**: Fewer augmented frames, but safer
- **Use when**: Want maximum quality in augmented dataset

**Strategy C: Partial Augmentation** (Advanced)
- For frames with some failed cameras:
  - Augment only the cameras that pass (>90% alignment)
  - Use original (non-augmented) images for failed cameras
  - Mark in metadata which cameras were augmented
- **Pros**: Maximizes augmented data while maintaining consistency
- **Cons**: Mixed augmented/non-augmented views in same frame

**Strategy D: Camera-Specific Handling** (Hybrid)
- Identify consistently failing cameras (e.g., Cam2005325 fails often)
- For those cameras: Use stricter thresholds or always attempt fix
- For reliable cameras: Use standard thresholds
- **Pros**: Adapts to camera-specific issues
- **Cons**: Requires per-camera analysis

#### 4.3 Recommendation for Neural Network Training
**Recommended approach**: **Strategy A (Fix Failed Cameras, Then Decide)**
1. Validate all 16 cameras for each frame
2. If any camera fails → Attempt to fix mask
3. **If fix succeeds** → All cameras aligned → ✅ **Augment frame** (use in augmented dataset)
4. **If fix fails** → ❌ **Don't augment**, but **include original frame in training dataset**

**Result**:
- **Augmented dataset**: Only frames where all 16 cameras have perfect alignment (after fixes)
- **Training dataset**: Includes both augmented frames AND original frames (even if not augmented)
- **Quality guarantee**: Augmented frames have perfect multi-view consistency
- **Data utilization**: No frames are wasted - all original frames are used for training

### 5. Identify Problematic Cases
Categorize issues:
- **Occlusion**: Keypoints visible but mask missing/incomplete
- **Calibration mismatch**: Keypoints projected incorrectly (wrong camera/session calibration)
- **Partial occlusion**: Some keypoints inside mask, others outside
- **Multiple detections**: SAM3 detects multiple objects, mask includes non-mouse regions
- **Negative Y coordinates**: Session `2024_11_26_17_00_20` has negative Y values that need fixing
- **Multi-view inconsistency**: Some cameras aligned, others not (frame-level issue)

---

## Phase 2: Fix Alignment Issues

### 5. Calibration-Based Reprojection
If calibration files exist per session/camera:
- Load correct calibration for each image's session/camera
- Reproject keypoints using correct camera parameters
- Re-validate alignment after reprojection
- Update annotations if reprojection fixes misalignment

### 6. Mask Refinement Using Keypoints
If keypoints are trusted:
- **Use keypoint bbox as constraint**: Ensure mask covers keypoint bounding box
- **Expand mask to include keypoints**: Small dilation around keypoints to ensure coverage
- **Use keypoints as positive prompts**: Optionally refine SAM3 mask using keypoints as foreground points
- **Remove distant regions**: Remove mask regions far from any keypoint (outlier removal)

### 7. Handle Occlusion Cases
Detect occlusion: keypoints visible but mask empty/small

Options:
- **Skip augmentation**: Exclude heavily occluded frames from augmentation
- **Keypoint-based mask generation**: Use convex hull + dilation around keypoints
- **Mark as "occluded"**: Handle separately in training (e.g., partial supervision)
- **Hybrid approach**: Use SAM3 mask where available, fallback to keypoint-based mask

---

## Phase 3: Data Augmentation Pipeline

### 8. Quality Filtering - Keypoint-Mask Alignment Required

**Core Principle**: Only augment images where keypoints align with masks. If alignment fails, either get better masks or skip the image.

#### 8.1 Alignment Validation
For each image:
1. Project keypoints onto mask (as described in Phase 1, Section 3.1)
2. Compute alignment score:
   - **% of visible keypoints inside mask** (primary metric)
   - **Distance from keypoints to mask edge** (secondary)
   - **Bounding box IoU** (secondary)

#### 8.2 Filtering Strategy
Define strict thresholds:
- **Minimum % of visible keypoints inside mask**: **>90%** (strict requirement)
- **Maximum distance from keypoints to mask edge**: <50 pixels (for keypoints outside)
- **Minimum mask area**: >1000 pixels (avoid tiny/no detections)
- **Bounding box IoU**: >0.7 (mask should cover keypoint bbox)

**Decision Tree**:
- ✅ **Pass**: All thresholds met → Include in augmentation
- ❌ **Fail**: Thresholds not met → Two options:
  - **Option A**: Attempt to get better mask (see 8.3)
  - **Option B**: Skip image entirely (don't augment)

#### 8.3 Handling Misaligned Images
For images that fail alignment:

**Option A: Get Better Masks**
1. **Re-run SAM3 with different parameters**:
   - Lower confidence threshold (catch more of mouse)
   - Try different text prompts ("rodent", "animal", "mouse on table")
   - Use bounding box from keypoints as constraint
   - Combine multiple SAM3 outputs

2. **Refine existing mask**:
   - Expand mask to include all visible keypoints (dilation around keypoints)
   - Use keypoint bbox as hard constraint (mask must cover bbox)
   - Remove mask regions far from keypoints

3. **Re-validate**: Check alignment again after mask improvement
   - If now passes → Include in augmentation
   - If still fails → Skip (Option B)

**Option B: Skip Image**
- Mark as "misaligned" in metadata
- Exclude from augmentation dataset
- Log reason (e.g., "only 60% keypoints inside mask")

#### 8.4 Multi-View Filtering (16 Cameras)

**Frame-Level Filtering**:
1. **Group by frame**: For each frame, collect all 16 camera views
2. **Validate all cameras**: Check alignment for each camera in the frame
3. **Frame decision**:
   - **All 16 cameras pass** (>90% alignment each) → ✅ Include entire frame
   - **1+ cameras fail** → Apply chosen strategy (A, B, C, or D from Section 4.2)

**Final Filtered Dataset**:
- **Frame-level**: Only frames where all 16 cameras have validated alignment (or fixed)
- **Camera-level**: All images in filtered set have >90% keypoints inside mask
- **Multi-view consistency**: All cameras in a frame are aligned and consistent
- **Safe for augmentation**: Keypoints and masks are consistent across all views

### 9. Augmentation Strategy - Only for Aligned Images

**Prerequisite**: Only augment images from the filtered dataset (Section 8.4) where keypoints and masks are aligned.

For each **validated aligned image**:
1. **Load data**:
   - Original image
   - Mask (binary or RGBA) - **verified aligned with keypoints**
   - Keypoints (with visibility flags) - **verified inside mask**
   - Metadata (session, camera, frame, alignment score)

2. **Extract mouse**:
   - Use mask to create RGBA cutout (mouse on transparent background)
   - Keypoints are guaranteed to be on the mouse (they're inside the mask)

3. **Composite onto new background**:
   - **Multi-view consistency**: For a frame (all 16 cameras), use the **same background** across all views
   - Random backgrounds (textures, colors, scenes, indoor/outdoor)
   - Place mouse at random positions/scales (same position/scale across all 16 cameras for consistency)
   - Optionally adjust lighting/shadow to match background
   - **Keypoint coordinate adjustment**: If mouse is moved/scaled, adjust keypoint coordinates accordingly (same adjustment for all cameras)
   - Ensure adjusted keypoints remain within image bounds for all views

4. **Save augmented data**:
   - Augmented image
   - **Adjusted keypoints** (if mouse was moved/scaled, update x,y coordinates)
   - **Adjusted mask** (if mouse was scaled, scale mask accordingly)
   - Original alignment score (for reference)
   - New background metadata
   - Augmentation parameters (position, scale, background type)

**Key Guarantee**: Since we only augment aligned images, keypoints and masks remain consistent throughout augmentation.

### 10. Occlusion Handling in Augmentation
For occluded frames:
- **Option A**: Skip augmentation (use only fully visible mice)
- **Option B**: Generate synthetic occlusion (add occluders in augmentation)
- **Option C**: Use keypoint-based mask estimation for occluded parts
- **Option D**: Mark as "partially occluded" and use in training with appropriate loss weighting

---

## Phase 4: Validation & Output

### 11. Visual Inspection Tools
Generate visualization grids:
- **Original image + keypoints + mask overlay**: Show alignment visually
- **Highlight misaligned keypoints**: Mark keypoints outside mask in red
- **Show augmented result**: Display augmented image with keypoints
- **Before/after comparison**: Original vs augmented side-by-side

Create summary report:
- % of images with perfect alignment (>95% keypoints inside)
- % with minor misalignment (80-95% keypoints inside)
- % with major misalignment (<80% keypoints inside)
- % with occlusion (keypoints visible, mask missing/small)
- List of problematic sessions/cameras
- Distribution of alignment scores

### 12. Output Structure
Filtered dataset: `augmented/train/` and `augmented/val/`

Each image has:
- `Frame_X.jpg` (augmented image)
- `Frame_X_mask.png` (aligned mask)
- `Frame_X_keypoints.json` (keypoints, verified aligned)
- `Frame_X_meta.json` (original path, augmentation params, quality score, alignment metrics)

---

## Phase 5: Integration with 3D Tracking

### 13. 3D Consistency Check
For multi-view setups:
- **Verify keypoint consistency**: Ensure keypoints are consistent across cameras (same 3D point)
- **Check mask alignment**: Verify masks align with triangulated 3D keypoints
- **Multi-view mask consistency**: Ensure masks from different views are consistent
- **Background consistency**: Ensure augmented backgrounds don't break multi-view consistency (if using same background across views)

### 14. Training Data Format
Output format compatible with 3D tracking model:
- **Images**: Consistent keypoints across views
- **Masks**: Per-view masks aligned with keypoints
- **Calibration data**: For triangulation (per camera/session)
- **Metadata**: Data augmentation tracking, quality scores, alignment metrics
- **3D keypoints**: Triangulated from 2D keypoints (if available)

---

## Workflow Summary

1. **Validate Alignment** (Phase 1, Section 3):
   - For each image: Project keypoints onto mask, check if keypoints are inside mask
   - Compute alignment score: % of visible keypoints inside mask

2. **Multi-View Validation** (Phase 1, Section 4):
   - **Group by frame**: Collect all 16 cameras for each frame
   - **Validate all cameras**: Check alignment for each camera in the frame
   - **Frame decision**:
     - All 16 cameras pass (>90% each) → ✅ **Augment frame**
     - 1+ cameras fail → Attempt to fix, then:
       - If fix succeeds → ✅ **Augment frame**
       - If fix fails → ❌ **Don't augment**, but **use original frame for training**

3. **Filter Dataset** (Phase 3, Section 8):
   - **For augmentation**: Only frames where all 16 cameras have validated alignment (after fixes)
   - **For training**: Include ALL original frames (augmented + non-augmented)
   - **Camera-level**: Each image has >90% keypoints inside mask (for augmented frames)
   - **Multi-view consistency**: All cameras in augmented frames are aligned

4. **Augment Only Aligned Frames** (Phase 3, Section 9):
   - Only augment frames from filtered set (all 16 cameras aligned)
   - Extract mouse using mask for each camera, composite onto new backgrounds
   - Adjust keypoints if mouse is moved/scaled (consistent across all views)
   - Save augmented images with aligned keypoints and masks

**Result**: 
- **Augmented dataset**: Frames where all 16 cameras have perfect alignment (for data augmentation)
- **Training dataset**: Includes both augmented frames AND original frames (even if not augmented)
- **Quality guarantee**: Augmented frames have perfect multi-view consistency
- **No data waste**: All original frames are used for training

---

## Priority Order

1. **Immediate**: Phase 1 (Diagnostics) - Identify where keypoints and masks misalign
2. **Critical**: Phase 2 (Fixes) - Ensure keypoints and masks align properly
3. **Important**: Phase 3 (Augmentation) - Generate augmented dataset **only for aligned images**
4. **Final**: Phase 4-5 (Validation & Integration) - Ensure quality and compatibility

---

## Key Files to Review

- `annotations/instances_train.json` / `instances_val.json` - Keypoint annotations
- `segment_3d_model.py` - Existing projection code
- `masks_final/` - SAM3-generated masks (binary masks and RGBA cutouts)
- Calibration files (likely YAML per camera/session)
- `Ratan_mice_task/` - 3D model and IK code

---

## Implementation Notes: Keypoint-to-Mask Projection

### Approach
1. **Load mask as numpy array**:
   ```python
   mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Shape: (H, W), values: 0 or 255
   ```

2. **Extract keypoints from annotation**:
   ```python
   kps = annotation['keypoints']  # [x1, y1, vis1, x2, y2, vis2, ...]
   visible_kps = []
   for i in range(0, len(kps), 3):
       x, y, vis = kps[i], kps[i+1], kps[i+2]
       if vis > 0:
           # Fix negative Y if needed
           if y < 0:
               y = image_height + y
           visible_kps.append((x, y))
   ```

3. **Check alignment**:
   ```python
   inside_count = 0
   for x, y in visible_kps:
       if 0 <= x < W and 0 <= y < H:
           if mask[y, x] == 255:  # White = mouse
               inside_count += 1
   alignment_pct = inside_count / len(visible_kps) * 100
   ```

4. **Distance to mask edge** (for keypoints outside):
   ```python
   from scipy.ndimage import distance_transform_edt
   # Invert mask: 1 = background, 0 = foreground
   dist_to_mask = distance_transform_edt(mask == 0)
   # For keypoint at (x, y): dist_to_mask[y, x] gives distance to nearest white pixel
   ```

5. **Bounding box computation**:
   ```python
   kp_bbox = [min(x for x,y in visible_kps), min(y for x,y in visible_kps),
              max(x for x,y in visible_kps), max(y for x,y in visible_kps)]
   # Mask bbox from white pixels
   white_pixels = np.where(mask == 255)
   mask_bbox = [white_pixels[1].min(), white_pixels[0].min(),
                white_pixels[1].max(), white_pixels[0].max()]
   ```

---

## Success Criteria

- **Alignment**: >95% of visible keypoints inside masks (per camera)
- **Coverage**: Masks cover >90% of keypoint bounding boxes (per camera)
- **Multi-view consistency**: All 16 cameras aligned for each frame
- **Quality**: All augmented images have aligned keypoints and masks
- **Frame-level quality**: All frames have all 16 cameras with good alignment
- **Robustness**: 3D tracking model trains successfully on augmented dataset

---

## Answer: Should We Skip Frames with 1 Bad Camera?

**Clarification**: "Skip frame" means **don't augment**, but **still use original frame for training**.

**Short answer**: 
- **For augmentation**: Only augment frames where all 16 cameras have good alignment (after attempting fixes)
- **For training**: Use ALL original frames (both augmented and non-augmented)

**Detailed reasoning**:
- **3D tracking requires multi-view consistency**: For augmented data, all 16 cameras must be aligned
- **Neural network training**: Use all available data - augmented frames for diversity, original frames for completeness
- **Better approach**: Try to fix failed cameras (re-run SAM3, refine mask) before deciding
- **Fallback**: If fix fails, don't augment but still include original frame in training dataset

**Recommended strategy**:
1. **Validate all 16 cameras** for each frame
2. **If any camera fails**: Attempt to fix (re-run SAM3, refine mask)
3. **If fix succeeds**: All cameras aligned → ✅ **Augment frame** (add to augmented dataset)
4. **If fix fails**: ❌ **Don't augment**, but **include original frame in training dataset**
5. **Result**: 
   - **Augmented dataset**: Only frames with all 16 cameras perfectly aligned
   - **Training dataset**: All frames (augmented + original non-augmented)
   - **No data waste**: Every frame is used for training

This ensures:
- **High-quality augmented data**: Perfect multi-view consistency
- **Complete training data**: All original frames are utilized
- **Robust training**: Model sees both augmented diversity and original data
