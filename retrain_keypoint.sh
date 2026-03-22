#!/usr/bin/env bash
# Continues KeypointDetect training from Run_20260311 for 100 more epochs.
# Logs to retrain_keypoint.log (launch with nohup to run in the background).

REPO=/home/user/src/JARVIS-HybridNet
WEIGHTS=$REPO/projects/mouseJan30/models/KeypointDetect/Run_20260311-151206/EfficientTrack-medium_final.pth
LOG=$REPO/retrain_keypoint.log

cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jarvis

echo "[$(date)] Starting KeypointDetect retraining from $WEIGHTS" | tee -a "$LOG"

python train_aug.py \
    --stage keypoint \
    --weights_keypoint "$WEIGHTS" \
    --epochs_keypoint 100 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Training complete." | tee -a "$LOG"
