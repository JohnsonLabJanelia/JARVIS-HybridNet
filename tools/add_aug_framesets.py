"""
add_aug_framesets.py
--------------------
Patch a JARVIS instances JSON so that HybridNet can train on augmented images.

The problem: augmented images (e.g. background_replace, motion_blur, ...) are
listed in `images` but have no framesets, so Dataset3D / HybridNet never sees
them.  CenterDetect and KeypointDetect train on individual images (fine), but
HybridNet needs full 16-camera framesets where frame_idx == camera_index in
the calibration order.

Fix: for each (session, frame, strategy_variant) group of augmented images:
  - Build a 16-entry frameset in calibration camera order
  - Augmented image where available; fall back to original for missing cameras
  - Require every camera slot to have a valid image (no gaps → no index skew)
  - Write a new JSON with the extra framesets appended

Usage:
    python add_aug_framesets.py \\
        --input  annotations/instances_train.json \\
        --output annotations/instances_train_aug_framesets.json \\
        --min-aug 10        # min cameras that must have the augmented version
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_stem_lookup(images):
    """(session, cam, stem) -> image_id"""
    lookup = {}
    for img in images:
        parts = img['file_name'].split('/')
        session, cam = parts[0], parts[1]
        stem = parts[2].rsplit('.', 1)[0]
        lookup[(session, cam, stem)] = img['id']
    return lookup


def build_aug_groups(images):
    """(session, aug_stem) -> {cam: image_id}  — only augmented images (contain __)"""
    groups = defaultdict(dict)
    for img in images:
        parts = img['file_name'].split('/')
        session, cam = parts[0], parts[1]
        stem = parts[2].rsplit('.', 1)[0]
        if '__' in stem:
            groups[(session, stem)][cam] = img['id']
    return groups


def add_aug_framesets(in_path, out_path, min_aug=10):
    print(f"Loading {in_path} ...")
    data = json.load(open(in_path))

    stem_lookup = build_stem_lookup(data['images'])
    aug_groups  = build_aug_groups(data['images'])
    print(f"  {len(aug_groups)} augmented (session, variant) groups found")

    cal_cam_order = {session: list(cams.keys())
                     for session, cams in data['calibrations'].items()}

    new_framesets = dict(data['framesets'])
    added = skipped = 0

    for (session, aug_stem), cam_aug in aug_groups.items():
        if session not in cal_cam_order:
            continue

        base_stem  = aug_stem.split('__')[0]   # e.g. Frame_30
        cam_order  = cal_cam_order[session]     # 16 cameras in calibration order

        # Build 16-entry list: augmented > original > None
        frame_ids = []
        for cam in cam_order:
            if cam in cam_aug:
                frame_ids.append(cam_aug[cam])
            else:
                orig_id = stem_lookup.get((session, cam, base_stem))
                frame_ids.append(orig_id)   # None if missing

        # Require no gaps (None would misalign frame_idx ↔ camera_index)
        if None in frame_ids:
            skipped += 1
            continue

        # Also require at least min_aug cameras to have the augmented version
        if sum(1 for cam in cam_order if cam in cam_aug) < min_aug:
            skipped += 1
            continue

        fs_key = f'{session}/{aug_stem}'
        new_framesets[fs_key] = {'datasetName': session, 'frames': frame_ids}
        added += 1

    print(f"  Added {added} framesets  |  Skipped {skipped}")
    print(f"  Total framesets: {len(new_framesets)}")

    out_data = dict(data)
    out_data['framesets'] = new_framesets
    print(f"Writing {out_path} ...")
    with open(out_path, 'w') as f:
        json.dump(out_data, f)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',   required=True, help='Input instances JSON')
    parser.add_argument('--output',  required=True, help='Output instances JSON')
    parser.add_argument('--min-aug', type=int, default=10,
                        help='Min cameras that must have the augmented image (default: 10)')
    args = parser.parse_args()
    add_aug_framesets(args.input, args.output, args.min_aug)
