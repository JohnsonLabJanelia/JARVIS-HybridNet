# Current Alignment Rule

This document describes the **current practical decision rule** used to create:
- `pass16` frames
- `pass15` frames
- `pass14` frames

## Per-camera rule (image-level)

A camera view is marked **good** if both are true:

1. `kp_inside_pct >= 90`
2. `mask_area_px >= 1000`

Where:
- `kp_inside_pct = (num_kp_inside / num_kp_visible) * 100`
- `num_kp_inside`: visible keypoints that fall on white mask pixels (`mask[y, x] > 0`)
- `mask_area_px`: number of white pixels in the binary mask

## Frame-level buckets (16-camera setup)

For each frame (same timestamp across cameras), count how many of the 16 cameras are **good**:

- `pass16`: 16/16 cameras good
- `pass15`: 15/16 cameras good
- `pass14`: 14/16 cameras good

Only frames with all 16 camera views present are considered for these buckets.

## Notes

- This is the rule used for the review pack in `review_alignment_examples/`.
- It is intentionally simpler than the stricter CSV `passed` field, which also includes bbox-based constraints (`bbox_iou`, `bbox_coverage`, and distance limits).
- The simpler rule tracks visual quality better for current SAM masks.
