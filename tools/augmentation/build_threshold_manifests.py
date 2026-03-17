#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/user/mouse_labels/jarvis_merge/merged_output2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=ROOT / "reports" / "alignment_image_metrics.csv",
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "multiview_aug" / "manifests")
    p.add_argument("--kp-threshold", type=float, default=90.0)
    p.add_argument("--min-mask-area", type=int, default=1000)
    p.add_argument("--min-threshold", type=int, default=1)
    p.add_argument("--max-threshold", type=int, default=16)
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        r = csv.DictReader(f)
        for x in r:
            x["kp_inside_pct"] = float(x["kp_inside_pct"])
            x["mask_area_px"] = int(float(x["mask_area_px"]))
            x["num_kp_visible"] = int(float(x["num_kp_visible"]))
            x["num_kp_inside"] = int(float(x["num_kp_inside"]))
            x["max_outside_dist_px"] = float(x["max_outside_dist_px"])
            rows.append(x)
    return rows


def is_good_cam(row: dict, kp_threshold: float, min_mask_area: int) -> bool:
    return (
        row["num_kp_visible"] > 0
        and row["kp_inside_pct"] >= kp_threshold
        and row["mask_area_px"] >= min_mask_area
    )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.metrics_csv)
    by_frame: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_frame[(r["split"], r["session"], r["frame"])].append(r)

    frame_records: list[dict] = []
    for (split, session, frame), views in by_frame.items():
        views = sorted(views, key=lambda z: z["camera"])
        good = [v for v in views if is_good_cam(v, args.kp_threshold, args.min_mask_area)]
        bad = [v for v in views if not is_good_cam(v, args.kp_threshold, args.min_mask_area)]
        rec = {
            "split": split,
            "session": session,
            "frame": frame,
            "total_present_cams": len(views),
            "good_cams": len(good),
            "bad_cams": len(bad),
            "failing_cameras": ";".join(v["camera"] for v in bad),
            "min_kp_inside_pct": min(v["kp_inside_pct"] for v in views) if views else 0.0,
            "avg_kp_inside_pct": (sum(v["kp_inside_pct"] for v in views) / len(views)) if views else 0.0,
        }
        frame_records.append(rec)

    # Global frame summary
    frame_summary_csv = args.output_root / "frame_summary.csv"
    with open(frame_summary_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "session",
                "frame",
                "total_present_cams",
                "good_cams",
                "bad_cams",
                "failing_cameras",
                "min_kp_inside_pct",
                "avg_kp_inside_pct",
            ],
        )
        w.writeheader()
        w.writerows(sorted(frame_records, key=lambda z: (z["split"], z["session"], z["frame"])))

    summary = {
        "rule": {
            "num_kp_visible_gt_0": True,
            "kp_inside_pct_gte": args.kp_threshold,
            "mask_area_px_gte": args.min_mask_area,
        },
        "source_metrics_csv": str(args.metrics_csv),
        "thresholds": {},
    }

    for thr in range(args.max_threshold, args.min_threshold - 1, -1):
        out_dir = args.output_root / f"pass_at_least_{thr}"
        out_dir.mkdir(parents=True, exist_ok=True)

        eligible_frame_keys = set()
        for fr in frame_records:
            if fr["good_cams"] >= thr:
                eligible_frame_keys.add((fr["split"], fr["session"], fr["frame"]))

        frames_csv = out_dir / "frames.csv"
        with open(frames_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "session",
                    "frame",
                    "total_present_cams",
                    "good_cams",
                    "bad_cams",
                    "failing_cameras",
                    "min_kp_inside_pct",
                    "avg_kp_inside_pct",
                ],
            )
            w.writeheader()
            for fr in sorted(frame_records, key=lambda z: (z["split"], z["session"], z["frame"])):
                key = (fr["split"], fr["session"], fr["frame"])
                if key in eligible_frame_keys:
                    w.writerow(fr)

        # Good views only (augmentation views)
        views_good_csv = out_dir / "views_good.csv"
        with open(views_good_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "rel_image_path",
                    "rel_mask_path",
                    "split",
                    "session",
                    "camera",
                    "frame",
                    "kp_inside_pct",
                    "mask_area_px",
                ]
            )
            for r in rows:
                key = (r["split"], r["session"], r["frame"])
                if key not in eligible_frame_keys:
                    continue
                if not is_good_cam(r, args.kp_threshold, args.min_mask_area):
                    continue
                p = Path(r["rel_path"])
                rel_mask = str(Path("masks") / p.parent / f"{p.stem}_mask.png")
                w.writerow(
                    [
                        r["rel_path"],
                        rel_mask,
                        r["split"],
                        r["session"],
                        r["camera"],
                        r["frame"],
                        f"{r['kp_inside_pct']:.3f}",
                        r["mask_area_px"],
                    ]
                )

        # All views in eligible frames (for context/reference)
        views_all_csv = out_dir / "views_all.csv"
        with open(views_all_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "rel_image_path",
                    "rel_mask_path",
                    "split",
                    "session",
                    "camera",
                    "frame",
                    "is_good_cam",
                    "kp_inside_pct",
                    "mask_area_px",
                ]
            )
            for r in rows:
                key = (r["split"], r["session"], r["frame"])
                if key not in eligible_frame_keys:
                    continue
                p = Path(r["rel_path"])
                rel_mask = str(Path("masks") / p.parent / f"{p.stem}_mask.png")
                w.writerow(
                    [
                        r["rel_path"],
                        rel_mask,
                        r["split"],
                        r["session"],
                        r["camera"],
                        r["frame"],
                        int(is_good_cam(r, args.kp_threshold, args.min_mask_area)),
                        f"{r['kp_inside_pct']:.3f}",
                        r["mask_area_px"],
                    ]
                )

        # JSONL for augmentation pipeline
        views_good_jsonl = out_dir / "views_good.jsonl"
        good_views = 0
        with open(views_good_jsonl, "w") as f:
            for r in rows:
                key = (r["split"], r["session"], r["frame"])
                if key not in eligible_frame_keys:
                    continue
                if not is_good_cam(r, args.kp_threshold, args.min_mask_area):
                    continue
                p = Path(r["rel_path"])
                rec = {
                    "image": str(ROOT / r["rel_path"]),
                    "mask": str(ROOT / "masks" / p.parent / f"{p.stem}_mask.png"),
                    "split": r["split"],
                    "session": r["session"],
                    "camera": r["camera"],
                    "frame": r["frame"],
                    "kp_inside_pct": r["kp_inside_pct"],
                    "mask_area_px": r["mask_area_px"],
                    "threshold": thr,
                }
                f.write(json.dumps(rec) + "\n")
                good_views += 1

        summary["thresholds"][f"pass_at_least_{thr}"] = {
            "eligible_frames": len(eligible_frame_keys),
            "augmentable_views_good_only": good_views,
            "folder": str(out_dir),
        }

    with open(args.output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

