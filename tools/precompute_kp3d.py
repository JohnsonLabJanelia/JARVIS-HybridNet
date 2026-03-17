"""
Precompute 3D keypoints for all framesets in instances_train.json and save
to a cache file. Dataset3D can load this instead of triangulating at runtime.

Run once:
    python precompute_kp3d.py --dataset /path/to/pass10 --split train
    python precompute_kp3d.py --dataset /path/to/pass10 --split val

Saves: {dataset}/annotations/keypoints3D_{split}.npy
       {dataset}/annotations/keypoints3D_{split}_index.json  (frameset_key -> row index)
"""

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, '/home/user/src/JARVIS-HybridNet')
os.chdir('/home/user/src/JARVIS-HybridNet')

from jarvis.config.project_manager import ProjectManager
from jarvis.dataset.utils import ReprojectionTool

DATASET = '/mnt/johnson_lab/doq/mouse_labels/jarvis_merge/augmented_merged_out2_v3_pass10'
SPLIT   = 'train'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=DATASET)
parser.add_argument('--split',   default=SPLIT)
args = parser.parse_args()

project = ProjectManager()
project.load('mouseJan30')
cfg = project.get_cfg()
cfg.defrost()
cfg.DATASET.DATASET_3D = args.dataset

ann_path = os.path.join(args.dataset, 'annotations', f'instances_{args.split}.json')
print(f"Loading {ann_path} ...")
data = json.load(open(ann_path))

id_to_img = {img['id']: img for img in data['images']}
id_to_ann = {a['image_id']: a for a in data['annotations']}

# Build reprojection tools per session
repro_tools = {}
for session, cams in data['calibrations'].items():
    calib_paths = {cam: path for cam, path in cams.items()}
    repro_tools[session] = ReprojectionTool(args.dataset, calib_paths)

N_joints = data['categories'][0]['num_keypoints']
framesets = data['framesets']
n = len(framesets)
print(f"Precomputing 3D keypoints for {n} framesets x {N_joints} joints ...")

kp3d_all  = np.zeros((n, N_joints, 3), dtype=np.float32)
index_map  = {}  # frameset_key -> row

for row, (fs_key, fs) in enumerate(framesets.items()):
    if row % 500 == 0:
        print(f"  {row}/{n} ...")

    session = fs['datasetName']
    repro   = repro_tools[session]
    num_cams = repro.num_cameras
    frame_ids = fs['frames']

    # Load 2D keypoints for each camera
    kps_list = []
    for img_id in frame_ids:
        ann = id_to_ann.get(img_id)
        if ann is None:
            kps_list.append(np.zeros((N_joints, 3)))
        else:
            kp = np.array(ann['keypoints']).reshape(N_joints, 3)
            kps_list.append(kp)

    # Triangulate each joint
    for j in range(N_joints):
        points2D  = np.zeros((num_cams, 2))
        cams_to_use = []
        for cam_idx in range(min(len(kps_list), num_cams)):
            x, y, v = kps_list[cam_idx][j]
            if x != 0 or y != 0:
                points2D[cam_idx] = [x, y]
                cams_to_use.append(cam_idx)
        if len(cams_to_use) >= 2:
            kp3d_all[row, j] = repro.reconstructPoint(points2D.T, cams_to_use)

    index_map[fs_key] = row

out_npy  = os.path.join(args.dataset, 'annotations', f'keypoints3D_{args.split}.npy')
out_idx  = os.path.join(args.dataset, 'annotations', f'keypoints3D_{args.split}_index.json')
np.save(out_npy, kp3d_all)
with open(out_idx, 'w') as f:
    json.dump(index_map, f)

print(f"Saved {out_npy}  ({kp3d_all.shape})")
print(f"Saved {out_idx}")
print("Done.")
