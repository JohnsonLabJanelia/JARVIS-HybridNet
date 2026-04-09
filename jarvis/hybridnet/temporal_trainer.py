"""
Training loop for the Temporal Refinement Transformer.
Operates independently from HybridNet on pre-extracted predictions.
"""

import os
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .temporal_transformer import TemporalRefinementTransformer
from .temporal_loss import TemporalLoss
from .physics_loss import BoneLengthLoss
from jarvis.dataset.dataset_temporal import TemporalDataset
from jarvis.utils.logger import NetLogger, AverageMeter
import jarvis.utils.clp as clp


class TemporalTrainer:
    """
    Standalone trainer for the Temporal Refinement Transformer.

    Args:
        cfg: global config
        predictions_dir: directory with pre-extracted HybridNet predictions
        run_name: optional run name for logging/saving
    """
    def __init__(self, cfg, predictions_dir=None, run_name=None):
        self.cfg = cfg
        tcfg = cfg.TEMPORAL

        if predictions_dir is None:
            predictions_dir = tcfg.PREDICTIONS_DIR

        if run_name is None:
            run_name = "Temporal_" + time.strftime("%Y%m%d-%H%M%S")

        self.model_savepath = os.path.join(
            cfg.PARENT_DIR, 'projects', cfg.PROJECT_NAME,
            'models', 'Temporal', run_name)
        os.makedirs(self.model_savepath, exist_ok=True)

        log_path = os.path.join(
            cfg.PARENT_DIR, 'projects', cfg.PROJECT_NAME,
            'logs', 'Temporal', run_name)
        self.logger = NetLogger(log_path)

        # Build model
        self.model = TemporalRefinementTransformer(
            num_joints=cfg.KEYPOINTDETECT.NUM_JOINTS,
            d_model=tcfg.D_MODEL,
            nhead=tcfg.NHEAD,
            num_layers=tcfg.NUM_LAYERS,
            window_size=tcfg.WINDOW_SIZE,
            dropout=tcfg.DROPOUT,
        ).cuda()

        # Build datasets
        self.train_dataset = TemporalDataset(
            predictions_dir, window_size=tcfg.WINDOW_SIZE,
            stride=tcfg.STRIDE, max_gap_frames=tcfg.MAX_GAP_FRAMES,
            split='train')
        self.val_dataset = TemporalDataset(
            predictions_dir, window_size=tcfg.WINDOW_SIZE,
            stride=tcfg.WINDOW_SIZE,  # non-overlapping for val
            max_gap_frames=tcfg.MAX_GAP_FRAMES,
            split='val')

        clp.info(f'Temporal train windows: {len(self.train_dataset)}')
        clp.info(f'Temporal val windows: {len(self.val_dataset)}')

        # Build loss
        bone_criterion = None
        bone_stats_path = os.path.join(
            cfg.PARENT_DIR, 'projects', cfg.PROJECT_NAME, 'bone_stats.json')
        if os.path.isfile(bone_stats_path) and tcfg.LAMBDA_BONE > 0:
            bone_criterion = BoneLengthLoss.from_stats_file(
                bone_stats_path,
                list(cfg.KEYPOINT_NAMES),
                list(cfg.SKELETON)).cuda()
            clp.info(f'Loaded bone stats for temporal loss')

        self.criterion = TemporalLoss(
            bone_criterion=bone_criterion,
            lambda_pos=tcfg.LAMBDA_POS,
            lambda_bone=tcfg.LAMBDA_BONE,
            lambda_vel=tcfg.LAMBDA_VEL,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=tcfg.MAX_LEARNING_RATE)

        self.lossMeter = AverageMeter()
        self.accuracyMeter = AverageMeter()

    def train(self, num_epochs=None):
        tcfg = self.cfg.TEMPORAL
        if num_epochs is None:
            num_epochs = tcfg.NUM_EPOCHS

        train_loader = DataLoader(
            self.train_dataset, batch_size=tcfg.BATCH_SIZE,
            shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(
            self.val_dataset, batch_size=tcfg.BATCH_SIZE,
            shuffle=False, num_workers=4, pin_memory=True)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, tcfg.MAX_LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs, div_factor=100)

        best_val_acc = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            self.lossMeter.reset()
            self.accuracyMeter.reset()

            progress_bar = tqdm(train_loader,
                                desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch in progress_bar:
                pred_kp = batch['pred_keypoints'].cuda()
                pred_conf = batch['pred_confidences'].cuda()
                gt_kp = batch['gt_keypoints'].cuda()
                valid_mask = batch['valid_mask'].cuda()
                gt_valid = batch['gt_valid'].cuda()

                self.optimizer.zero_grad()
                refined = self.model(pred_kp, pred_conf, valid_mask)
                loss, loss_dict = self.criterion(
                    refined, gt_kp, gt_valid, valid_mask)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                scheduler.step()

                # MPJPE on valid keypoints
                with torch.no_grad():
                    diff = torch.norm(refined - gt_kp, dim=-1)  # (B,T,J)
                    mpjpe = (diff * gt_valid).sum() / (gt_valid.sum() + 1e-6)

                self.lossMeter.update(loss.item())
                self.accuracyMeter.update(mpjpe.item())
                progress_bar.set_postfix(
                    loss=f'{self.lossMeter.read():.4f}',
                    mpjpe=f'{self.accuracyMeter.read():.2f}mm')

            train_loss = self.lossMeter.read()
            train_acc = self.accuracyMeter.read()
            self.logger.update_train_loss(train_loss)
            self.logger.update_train_accuracy(train_acc)
            self.logger.update_learning_rate(
                self.optimizer.param_groups[0]['lr'])

            # Validation
            val_acc = self._validate(val_loader)

            if (epoch + 1) % tcfg.CHECKPOINT_SAVE_INTERVAL == 0:
                self._save_checkpoint(
                    f'Temporal_Epoch_{epoch+1}.pth')

            if val_acc < best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint('Temporal_best.pth')

        self._save_checkpoint('Temporal_final.pth')
        clp.info(f'Best validation MPJPE: {best_val_acc:.2f} mm')

    def _validate(self, val_loader):
        self.model.eval()
        self.lossMeter.reset()
        self.accuracyMeter.reset()

        # Also track raw (unrefined) MPJPE for comparison
        raw_meter = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                pred_kp = batch['pred_keypoints'].cuda()
                pred_conf = batch['pred_confidences'].cuda()
                gt_kp = batch['gt_keypoints'].cuda()
                valid_mask = batch['valid_mask'].cuda()
                gt_valid = batch['gt_valid'].cuda()

                refined = self.model(pred_kp, pred_conf, valid_mask)
                loss, _ = self.criterion(
                    refined, gt_kp, gt_valid, valid_mask)

                # Refined MPJPE
                diff = torch.norm(refined - gt_kp, dim=-1)
                mpjpe = (diff * gt_valid).sum() / (gt_valid.sum() + 1e-6)

                # Raw MPJPE (before refinement)
                raw_diff = torch.norm(pred_kp - gt_kp, dim=-1)
                raw_mpjpe = (raw_diff * gt_valid).sum() / (gt_valid.sum() + 1e-6)

                self.lossMeter.update(loss.item())
                self.accuracyMeter.update(mpjpe.item())
                raw_meter.update(raw_mpjpe.item())

        val_loss = self.lossMeter.read()
        val_acc = self.accuracyMeter.read()
        raw_acc = raw_meter.read()

        self.logger.update_val_loss(val_loss)
        self.logger.update_val_accuracy(val_acc)

        print(f'  Val Loss: {val_loss:.4f} | '
              f'Raw MPJPE: {raw_acc:.2f}mm | '
              f'Refined MPJPE: {val_acc:.2f}mm | '
              f'Improvement: {raw_acc - val_acc:.2f}mm')

        self.model.train()
        return val_acc

    def _save_checkpoint(self, name):
        path = os.path.join(self.model_savepath, name)
        torch.save(self.model.state_dict(), path)
