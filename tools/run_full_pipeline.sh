#!/bin/bash
# Full evaluation + temporal pipeline after HybridNet training completes
set -e

PROJECT=mouseJan30
NEW_WEIGHTS="$1"
if [ -z "$NEW_WEIGHTS" ]; then
    echo "Usage: $0 <path_to_new_weights>"
    exit 1
fi

echo "============================================"
echo "  Step 1: Evaluate improved HybridNet"
echo "============================================"
python -u tools/evaluate_baseline.py \
    --project $PROJECT \
    --weights "$NEW_WEIGHTS" \
    --split val \
    --output projects/$PROJECT/improved_results.json \
    --label "HybridNet + Bone Loss (continued)"

echo ""
echo "============================================"
echo "  Step 2: Extract video predictions (train)"
echo "============================================"
rm -rf predictions/${PROJECT}_video_v2/train
python -u tools/predict_video_for_temporal.py \
    --project $PROJECT \
    --video_dir /mnt/nvme1/mouse_labels/ssd/mouse_vids/2024_11_06_12_19_42 \
    --calib_dir /mnt/nvme1/mouse_labels/ssd/mouse_vids/2024_11_06_12_19_42/calibration \
    --output predictions/${PROJECT}_video_v2/train \
    --start_frame 1000 --num_frames 800 --stride 1 \
    --session_name vid_2024_11_06

echo ""
echo "============================================"
echo "  Step 3: Extract video predictions (val)"
echo "============================================"
rm -rf predictions/${PROJECT}_video_v2/val
python -u tools/predict_video_for_temporal.py \
    --project $PROJECT \
    --video_dir /mnt/nvme1/mouse_labels/ssd/mouse_vids/2024_11_06_12_19_42 \
    --calib_dir /mnt/nvme1/mouse_labels/ssd/mouse_vids/2024_11_06_12_19_42/calibration \
    --output predictions/${PROJECT}_video_v2/val \
    --start_frame 5000 --num_frames 300 --stride 1 \
    --session_name vid_2024_11_06_val

echo ""
echo "============================================"
echo "  Step 4: Create smoothed training targets"
echo "============================================"
python3 -u << 'PYEOF'
import os, numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm

for split in ['train', 'val']:
    d = f'predictions/mouseJan30_video_v2/{split}'
    files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
    pts = np.array([np.load(os.path.join(d, f))['points3D'] for f in files])
    smoothed = np.zeros_like(pts)
    for j in range(24):
        for ax in range(3):
            smoothed[:, j, ax] = savgol_filter(pts[:, j, ax], 15, 3)
    all_data = [dict(np.load(os.path.join(d, f), allow_pickle=True)) for f in files]
    for i, f in enumerate(files):
        np.savez(os.path.join(d, f),
                 points3D=all_data[i]['points3D'],
                 confidences=all_data[i]['confidences'],
                 gt_keypoints3D=smoothed[i].astype(np.float32),
                 session=str(all_data[i]['session']),
                 frame_name=str(all_data[i]['frame_name']))
    raw_acc = np.linalg.norm(np.diff(np.diff(pts, axis=0), axis=0), axis=-1).mean()
    smo_acc = np.linalg.norm(np.diff(np.diff(smoothed, axis=0), axis=0), axis=-1).mean()
    print(f"  {split}: {len(files)} frames, jitter {raw_acc:.1f} -> {smo_acc:.1f} mm/f^2")
PYEOF

echo ""
echo "============================================"
echo "  Step 5: Train temporal transformer"
echo "============================================"
python -u tools/train_temporal.py \
    --project $PROJECT \
    --predictions predictions/${PROJECT}_video_v2 \
    --epochs 150 --batch_size 32 --window 16 --stride 2 \
    --max_gap 3 --lr 0.001

echo ""
echo "============================================"
echo "  Step 6: Measure jitter reduction"
echo "============================================"
python3 -u << 'PYEOF'
import os, numpy as np, torch, sys
sys.path.insert(0, '.')
from jarvis.config.project_manager import ProjectManager
from jarvis.hybridnet.temporal_transformer import TemporalRefinementTransformer
from jarvis.hybridnet.physics_loss import skeleton_to_index_pairs

project = ProjectManager()
project.load('mouseJan30')
cfg = project.get_cfg()

model = TemporalRefinementTransformer(num_joints=24, d_model=128, nhead=8, num_layers=4, window_size=16).cuda()
ckpt_dir = 'projects/mouseJan30/models/Temporal'
latest = sorted(os.listdir(ckpt_dir))[-1]
best = os.path.join(ckpt_dir, latest, 'Temporal_best.pth')
model.load_state_dict(torch.load(best))
model.eval()

val_dir = 'predictions/mouseJan30_video_v2/val'
files = sorted([f for f in os.listdir(val_dir) if f.endswith('.npz')])
raw = np.array([np.load(os.path.join(val_dir, f))['points3D'] for f in files])
T = 16
refined = np.zeros_like(raw); counts = np.zeros(len(raw))
with torch.no_grad():
    for s in range(0, len(raw)-T+1, 2):
        kp = torch.from_numpy(raw[s:s+T]).unsqueeze(0).float().cuda()
        c = torch.ones(1,T,24).float().cuda()
        m = torch.ones(1,T,dtype=torch.bool).cuda()
        refined[s:s+T] += model(kp,c,m)[0].cpu().numpy()
        counts[s:s+T] += 1
v = counts > 0; refined[v] /= counts[v,None,None]; refined[~v] = raw[~v]

raw_j = np.linalg.norm(np.diff(np.diff(raw,axis=0),axis=0),axis=-1)
ref_j = np.linalg.norm(np.diff(np.diff(refined,axis=0),axis=0),axis=-1)
bp = skeleton_to_index_pairs(list(cfg.KEYPOINT_NAMES), list(cfg.SKELETON))
raw_bs = np.mean([np.std(np.linalg.norm(raw[:,a]-raw[:,b],axis=-1)) for a,b in bp])
ref_bs = np.mean([np.std(np.linalg.norm(refined[:,a]-refined[:,b],axis=-1)) for a,b in bp])

print(f"  Jitter: {raw_j.mean():.2f} -> {ref_j.mean():.2f} mm/f^2 ({(1-ref_j.mean()/raw_j.mean())*100:.1f}% reduction)")
print(f"  Bone std: {raw_bs:.2f} -> {ref_bs:.2f} mm ({(1-ref_bs/raw_bs)*100:.1f}% reduction)")
PYEOF

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
