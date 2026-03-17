#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path("/home/user/mouse_labels/jarvis_merge/merged_output2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=int, required=True, help="Choose pass threshold 1..16")
    p.add_argument("--manifests-root", type=Path, default=ROOT / "multiview_aug" / "manifests")
    p.add_argument("--active-root", type=Path, default=ROOT / "multiview_aug" / "active")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (1 <= args.threshold <= 16):
        raise ValueError("threshold must be in [1, 16]")

    src = args.manifests_root / f"pass_at_least_{args.threshold}"
    if not src.exists():
        raise FileNotFoundError(f"Manifest folder not found: {src}")

    args.active_root.mkdir(parents=True, exist_ok=True)
    files = ["frames.csv", "views_good.csv", "views_all.csv", "views_good.jsonl"]
    for fn in files:
        s = src / fn
        d = args.active_root / fn
        if s.exists():
            shutil.copy2(s, d)

    meta = {
        "selected_threshold": args.threshold,
        "source_folder": str(src),
        "active_folder": str(args.active_root),
        "files": files,
    }
    with open(args.active_root / "selection.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

