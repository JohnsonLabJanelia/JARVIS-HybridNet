#!/usr/bin/env python3
"""
Repair empty-mask cases using SAM3 with keypoint prompts.

Inputs:
- reports/alignment_image_metrics.csv (to find empty masks)
- annotations/instances_{train,val}.json
- images under train/ and val/

Outputs:
- masks_recovered_kp/{split}/{session}/{cam}/Frame_x_mask.png
- review_no_mask_fixes_sam3_kp/ (before/after overlays + summary CSV/json)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


ROOT = Path("/home/user/mouse_labels/jarvis_merge/merged_output2")
IMG_W, IMG_H = 3208, 2200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-csv", type=Path, default=ROOT / "reports" / "alignment_image_metrics.csv")
    p.add_argument("--mask-root", type=Path, default=ROOT / "masks")
    p.add_argument("--out-mask-root", type=Path, default=ROOT / "masks_recovered_kp")
    p.add_argument("--review-dir", type=Path, default=ROOT / "review_no_mask_fixes_sam3_kp")
    p.add_argument("--max-cases", type=int, default=240)
    p.add_argument("--review-samples", type=int, default=120)
    p.add_argument("--bbox-pad", type=int, default=80)
    p.add_argument("--only-session-prefix", type=str, default="", help="e.g. 2026_ to run only matching sessions")
    p.add_argument("--use-2026-scene-prompts", action="store_true", help="For 2026 sessions, also try text prompts with cylinder context")
    p.add_argument("--use-neg-cylinder-prompts", action="store_true", help="Use positive mouse keypoints + negative cylinder/background points")
    p.add_argument("--use-neg-table-prompts", action="store_true", help="Use positive mouse keypoints + negative table/background points")
    return p.parse_args()


def load_ann_lookup() -> Dict[str, dict]:
    out = {}
    for split in ["train", "val"]:
        with open(ROOT / "annotations" / f"instances_{split}.json") as f:
            data = json.load(f)
        id_to_img = {img["id"]: img for img in data["images"]}
        id_to_ann = {ann["image_id"]: ann for ann in data["annotations"]}
        for img_id, img in id_to_img.items():
            rel = f"{split}/{img['file_name']}"
            if img_id in id_to_ann:
                out[rel] = id_to_ann[img_id]
    return out


def robust_visible_kps(ann: dict) -> np.ndarray:
    kps = ann.get("keypoints", [])
    if not kps:
        return np.zeros((0, 2), dtype=np.float32)

    ys = [kps[i + 1] for i in range(0, len(kps), 3) if kps[i + 2] > 0]
    neg_ratio = (sum(1 for y in ys if y < 0) / len(ys)) if ys else 0.0
    has_neg_y = neg_ratio > 0.7

    pts = []
    for i in range(0, len(kps), 3):
        x, y, v = kps[i], kps[i + 1], kps[i + 2]
        if v <= 0:
            continue
        if abs(float(x)) > 1e5 or abs(float(y)) > 1e5:
            continue
        if has_neg_y:
            y = IMG_H + y
        if 0 <= x < IMG_W and 0 <= y < IMG_H:
            pts.append((float(x), float(y)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def select_prompt_points(points: np.ndarray, max_points: int = 16) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]


def bbox_from_points(points: np.ndarray, pad: int) -> np.ndarray:
    x1 = max(0, points[:, 0].min() - pad)
    y1 = max(0, points[:, 1].min() - pad)
    x2 = min(IMG_W - 1, points[:, 0].max() + pad)
    y2 = min(IMG_H - 1, points[:, 1].max() + pad)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def overlay_with_kps(img_bgr: np.ndarray, mask: np.ndarray, kps: np.ndarray, title: str) -> np.ndarray:
    out = img_bgr.copy()
    idx = mask > 0
    green = np.zeros_like(out)
    green[:, :, 1] = 255
    out[idx] = (0.5 * out[idx] + 0.5 * green[idx]).astype(np.uint8)

    inside = 0
    for x, y in kps.astype(int):
        ok = 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0
        if ok:
            inside += 1
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.circle(out, (x, y), 6, color, -1)
        cv2.circle(out, (x, y), 8, (255, 255, 255), 1)

    pct = (100.0 * inside / len(kps)) if len(kps) else 0.0
    cv2.rectangle(out, (0, 0), (out.shape[1], 76), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"kp_inside={inside}/{len(kps)} ({pct:.1f}%)", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def to_numpy_mask(m) -> np.ndarray:
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 3:
        m = m.squeeze(0)
    return (m > 0).astype(np.uint8) * 255


def kp_inside_stats(mask: np.ndarray, kps: np.ndarray) -> Tuple[int, float]:
    if len(kps) == 0:
        return 0, 0.0
    inside = 0
    h, w = mask.shape
    for x, y in kps.astype(int):
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            inside += 1
    pct = 100.0 * inside / len(kps)
    return inside, pct


def choose_best_candidate(
    candidates: List[Tuple[str, np.ndarray]],
    kps: np.ndarray,
    cyl_mask: np.ndarray | None = None,
) -> Tuple[str, np.ndarray]:
    """
    Pick best mask with priority:
    1) maximize keypoints inside
    2) maximize inside percentage
    3) minimize area (avoid over-large masks like mouse+cylinder union)
    """
    best = None
    best_score = None
    for tag, mask in candidates:
        inside, pct = kp_inside_stats(mask, kps)
        area = int((mask > 0).sum())
        cyl_overlap = int(((mask > 0) & (cyl_mask > 0)).sum()) if cyl_mask is not None else 0
        score = (inside, pct, -cyl_overlap, -area)
        if best_score is None or score > best_score:
            best_score = score
            best = (tag, mask)
    assert best is not None
    return best


def keep_mouse_component(mask: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """
    Keep only the connected component most supported by keypoints.
    This removes extra objects (e.g., cylinder) when SAM returns multiple blobs.
    """
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return (bin_mask * 255).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if n_labels <= 1:
        return (bin_mask * 255).astype(np.uint8)

    # Score each component by how many keypoints fall inside it.
    comp_scores = {}
    h, w = bin_mask.shape
    for lbl in range(1, n_labels):
        comp_scores[lbl] = 0
    for x, y in kps.astype(int):
        if 0 <= x < w and 0 <= y < h:
            lbl = int(labels[y, x])
            if lbl > 0:
                comp_scores[lbl] += 1

    # If any component contains keypoints, keep the most supported one.
    best_lbl = None
    best_score = -1
    for lbl, s in comp_scores.items():
        if s > best_score:
            best_score = s
            best_lbl = lbl

    if best_lbl is not None and best_score > 0:
        out = np.where(labels == best_lbl, 255, 0).astype(np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        return out

    # Fallback: no keypoint landed on mask (rare) -> keep largest component.
    areas = [(lbl, int(stats[lbl, cv2.CC_STAT_AREA])) for lbl in range(1, n_labels)]
    largest_lbl = max(areas, key=lambda t: t[1])[0]
    out = np.where(labels == largest_lbl, 255, 0).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return out


def sample_points_from_mask(mask: np.ndarray, n: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(xs) <= n:
        idx = np.arange(len(xs))
    else:
        idx = np.linspace(0, len(xs) - 1, n).astype(int)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return pts


def sample_points_from_mask_outside_box(mask: np.ndarray, box: np.ndarray, n: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    keep = ~((xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2))
    xs = xs[keep]
    ys = ys[keep]
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(xs) <= n:
        idx = np.arange(len(xs))
    else:
        idx = np.linspace(0, len(xs) - 1, n).astype(int)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return pts


def make_bbox_ring_neg_points(box: np.ndarray, n: int = 8) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    pts = []
    # corners + edge midpoints
    pts.extend(
        [
            (x1, y1),
            (x2, y1),
            (x1, y2),
            (x2, y2),
            ((x1 + x2) / 2, y1),
            ((x1 + x2) / 2, y2),
            (x1, (y1 + y2) / 2),
            (x2, (y1 + y2) / 2),
        ]
    )
    pts = np.asarray(pts[:n], dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, IMG_W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, IMG_H - 1)
    return pts


def main() -> None:
    args = parse_args()
    args.out_mask_root.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    (args.review_dir / "samples").mkdir(parents=True, exist_ok=True)

    with open(args.metrics_csv) as f:
        rows = list(csv.DictReader(f))
    empty_rows = [r for r in rows if int(float(r["mask_area_px"])) == 0]
    if args.only_session_prefix:
        empty_rows = [r for r in empty_rows if Path(r["rel_path"]).parts[1].startswith(args.only_session_prefix)]
    if args.max_cases > 0:
        empty_rows = empty_rows[: args.max_cases]

    ann_lookup = load_ann_lookup()

    bpe_path = ROOT / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=str(bpe_path), enable_inst_interactivity=True)
    processor = Sam3Processor(model, confidence_threshold=0.3)

    repaired = 0
    failed = 0
    no_kp = 0
    review_written = 0
    summary_rows = []

    for i, rec in enumerate(empty_rows, 1):
        rel = rec["rel_path"]
        img_path = ROOT / rel
        ann = ann_lookup.get(rel)
        kps = robust_visible_kps(ann) if ann is not None else np.zeros((0, 2), dtype=np.float32)

        old_mask_path = args.mask_root / Path(rel).parent / f"{Path(rel).stem}_mask.png"
        old_mask = cv2.imread(str(old_mask_path), cv2.IMREAD_GRAYSCALE)
        if old_mask is None:
            old_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        new_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        status = "failed"
        score = -1.0
        chosen_method = "none"
        error_msg = ""

        try:
            image = Image.open(img_path).convert("RGB")
            state = processor.set_image(image)

            if len(kps) >= 3:
                candidates: List[Tuple[str, np.ndarray]] = []

                # Candidate 1: keypoint-guided interactive mask
                pts = select_prompt_points(kps, max_points=16)
                labels = np.ones(len(pts), dtype=np.int32)
                box = bbox_from_points(kps, args.bbox_pad)[None, :]
                with torch.inference_mode():
                    masks, scores, _ = model.predict_inst(
                        state,
                        point_coords=pts,
                        point_labels=labels,
                        box=box,
                        multimask_output=True,
                    )
                if masks is not None and len(masks) > 0:
                    for mi, m in enumerate(masks):
                        candidates.append((f"kp_prompt_{mi}", to_numpy_mask(m)))
                    best = int(scores.argmax()) if hasattr(scores, "argmax") else int(np.argmax(scores))
                    score = float(scores[best]) if hasattr(scores, "__len__") else -1.0

                # Candidate 2/3: 2026-specific text prompts with cylinder context
                session_name = Path(rel).parts[1]
                avoid_mask_for_scoring = None
                if args.use_2026_scene_prompts and session_name.startswith("2026_"):
                    scene_prompts = [
                        "mouse",
                        "mouse and cylinder object",
                        "mouse, cylinder object",
                    ]
                    for sp in scene_prompts:
                        with torch.inference_mode():
                            out = processor.set_text_prompt(state=state, prompt=sp)
                        t_masks = out.get("masks", [])
                        # keep a cylinder mask for scoring/negative prompts if available
                        if sp == "mouse and cylinder object" and t_masks is not None and len(t_masks) > 0:
                            avoid_mask_for_scoring = to_numpy_mask(t_masks[0])
                        if t_masks is not None and len(t_masks) > 0:
                            for mi, tm in enumerate(t_masks):
                                candidates.append((f"text_{sp}_{mi}", to_numpy_mask(tm)))

                # Candidate 4: positive mouse kps + negative points from cylinder/background
                if args.use_neg_cylinder_prompts and session_name.startswith("2026_"):
                    with torch.inference_mode():
                        cyl_out = processor.set_text_prompt(state=state, prompt="cylinder object")
                    cyl_masks = cyl_out.get("masks", [])
                    cyl_mask = (
                        to_numpy_mask(cyl_masks[0])
                        if cyl_masks is not None and len(cyl_masks) > 0
                        else np.zeros((IMG_H, IMG_W), dtype=np.uint8)
                    )
                    if avoid_mask_for_scoring is None:
                        avoid_mask_for_scoring = cyl_mask
                    else:
                        avoid_mask_for_scoring = np.maximum(avoid_mask_for_scoring, cyl_mask)

                    # Build negative points: cylinder region + bbox ring.
                    neg_pts = sample_points_from_mask(cyl_mask, n=10)
                    ring_pts = make_bbox_ring_neg_points(box[0], n=8)
                    neg_pts = np.concatenate([neg_pts, ring_pts], axis=0) if len(neg_pts) else ring_pts

                    # Keep negatives away from positive keypoints by small exclusion.
                    if len(neg_pts) and len(pts):
                        keep = []
                        for pneg in neg_pts:
                            d2 = ((pts - pneg) ** 2).sum(axis=1)
                            keep.append(bool(np.min(d2) > 25.0**2))
                        neg_pts = neg_pts[np.asarray(keep, dtype=bool)]

                    if len(neg_pts):
                        all_pts = np.concatenate([pts, neg_pts], axis=0)
                        all_labels = np.concatenate(
                            [np.ones(len(pts), dtype=np.int32), np.zeros(len(neg_pts), dtype=np.int32)],
                            axis=0,
                        )
                        with torch.inference_mode():
                            nmasks, nscores, _ = model.predict_inst(
                                state,
                                point_coords=all_pts,
                                point_labels=all_labels,
                                box=box,
                                multimask_output=True,
                            )
                        if nmasks is not None and len(nmasks) > 0:
                            for mi, nm in enumerate(nmasks):
                                candidates.append((f"kp_negcyl_{mi}", to_numpy_mask(nm)))

                # Candidate 5: positive mouse kps + negative points from table/background
                if args.use_neg_table_prompts and session_name.startswith("2026_"):
                    table_prompts = ["big round table", "round table", "table"]
                    table_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
                    max_area = 0
                    for tp in table_prompts:
                        with torch.inference_mode():
                            t_out = processor.set_text_prompt(state=state, prompt=tp)
                        t_masks = t_out.get("masks", [])
                        if t_masks is None or len(t_masks) == 0:
                            continue
                        for tm in t_masks:
                            tm_np = to_numpy_mask(tm)
                            area = int((tm_np > 0).sum())
                            if area > max_area:
                                max_area = area
                                table_mask = tm_np

                    if avoid_mask_for_scoring is None:
                        avoid_mask_for_scoring = table_mask
                    else:
                        avoid_mask_for_scoring = np.maximum(avoid_mask_for_scoring, table_mask)

                    neg_table_pts = sample_points_from_mask_outside_box(table_mask, box[0], n=12)
                    ring_pts = make_bbox_ring_neg_points(box[0], n=8)
                    neg_pts = np.concatenate([neg_table_pts, ring_pts], axis=0) if len(neg_table_pts) else ring_pts

                    if len(neg_pts):
                        keep = []
                        for pneg in neg_pts:
                            d2 = ((pts - pneg) ** 2).sum(axis=1)
                            keep.append(bool(np.min(d2) > 25.0**2))
                        neg_pts = neg_pts[np.asarray(keep, dtype=bool)]

                    if len(neg_pts):
                        all_pts = np.concatenate([pts, neg_pts], axis=0)
                        all_labels = np.concatenate(
                            [np.ones(len(pts), dtype=np.int32), np.zeros(len(neg_pts), dtype=np.int32)],
                            axis=0,
                        )
                        with torch.inference_mode():
                            tmasks2, _, _ = model.predict_inst(
                                state,
                                point_coords=all_pts,
                                point_labels=all_labels,
                                box=box,
                                multimask_output=True,
                            )
                        if tmasks2 is not None and len(tmasks2) > 0:
                            for mi, tm2 in enumerate(tmasks2):
                                candidates.append((f"kp_negtable_{mi}", to_numpy_mask(tm2)))

                if candidates:
                    chosen_method, new_mask = choose_best_candidate(candidates, kps, cyl_mask=avoid_mask_for_scoring)
                    # Enforce mouse-only mask: drop disconnected non-mouse objects.
                    new_mask = keep_mouse_component(new_mask, kps)
                    status = "repaired" if new_mask.sum() > 0 else "empty_after"
                else:
                    status = "no_sam_mask"
            else:
                no_kp += 1
                status = "no_valid_kp"
        except Exception as e:
            failed += 1
            status = "exception"
            error_msg = str(e)
            if i <= 5:
                print(f"[debug] exception on {rel}: {error_msg}", flush=True)

        out_mask_path = args.out_mask_root / Path(rel).parent / f"{Path(rel).stem}_mask.png"
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_mask_path), new_mask)

        if new_mask.sum() > 0:
            repaired += 1

        # Write sample overlays for quick review
        if review_written < args.review_samples:
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is not None:
                before = overlay_with_kps(img_bgr, old_mask, kps, f"BEFORE (empty) | {rel}")
                after = overlay_with_kps(img_bgr, new_mask, kps, f"AFTER (SAM3+KP) | {rel} | status={status} score={score:.3f}")
                before = cv2.resize(before, (1200, 824), interpolation=cv2.INTER_AREA)
                after = cv2.resize(after, (1200, 824), interpolation=cv2.INTER_AREA)
                side = cv2.hconcat([before, after])
                outp = args.review_dir / "samples" / f"{review_written+1:04d}_{Path(rel).parts[0]}_{Path(rel).parts[1]}_{Path(rel).parts[2]}_{Path(rel).stem}.jpg"
                cv2.imwrite(str(outp), side)
                review_written += 1

        summary_rows.append(
            {
                "rel_path": rel,
                "num_kp_visible_reparsed": int(len(kps)),
                "old_mask_area_px": int((old_mask > 0).sum()),
                "new_mask_area_px": int((new_mask > 0).sum()),
                "status": status,
                "chosen_method": chosen_method,
                "sam_score": score,
                "error_msg": error_msg,
                "out_mask_path": str(out_mask_path),
            }
        )

        if i % 25 == 0 or i == len(empty_rows):
            print(f"[{i}/{len(empty_rows)}] repaired={repaired} no_kp={no_kp} failed={failed}", flush=True)

    # save CSV + summary
    csv_path = args.review_dir / "repair_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["rel_path"])
        w.writeheader()
        w.writerows(summary_rows)

    js = {
        "processed_cases": len(empty_rows),
        "repaired_nonempty_masks": repaired,
        "repair_rate_pct": (100.0 * repaired / len(empty_rows)) if empty_rows else 0.0,
        "no_valid_kp_cases": no_kp,
        "failed_cases": failed,
        "review_samples_written": review_written,
        "out_mask_root": str(args.out_mask_root),
        "review_dir": str(args.review_dir),
    }
    with open(args.review_dir / "repair_summary.json", "w") as f:
        json.dump(js, f, indent=2)

    print("Done.")
    print(json.dumps(js, indent=2))


if __name__ == "__main__":
    main()

