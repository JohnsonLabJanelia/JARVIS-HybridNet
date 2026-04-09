"""
Combined loss functions for the Temporal Refinement Transformer.
Includes position MSE, bone length consistency, and velocity smoothness.
"""

import torch
import torch.nn as nn

from .physics_loss import BoneLengthLoss


class TemporalLoss(nn.Module):
    """
    Combined loss for temporal refinement:
        L = lambda_pos * L_pos + lambda_bone * L_bone + lambda_vel * L_vel

    Args:
        bone_criterion: BoneLengthLoss instance (or None to skip)
        lambda_pos: weight for position MSE loss
        lambda_bone: weight for bone length loss
        lambda_vel: weight for velocity/acceleration loss
    """
    def __init__(self, bone_criterion=None, lambda_pos=1.0,
                 lambda_bone=0.05, lambda_vel=0.01):
        super().__init__()
        self.bone_criterion = bone_criterion
        self.lambda_pos = lambda_pos
        self.lambda_bone = lambda_bone
        self.lambda_vel = lambda_vel

    def position_loss(self, refined, gt, gt_valid):
        """
        MSE on valid keypoints.

        Args:
            refined: (B, T, J, 3) refined predictions
            gt: (B, T, J, 3) ground truth
            gt_valid: (B, T, J) boolean mask for valid GT keypoints
        """
        diff = (refined - gt) ** 2  # (B, T, J, 3)
        diff = diff.sum(dim=-1)     # (B, T, J)
        return (diff * gt_valid).sum() / (gt_valid.sum() + 1e-6)

    def velocity_loss(self, refined, valid_mask):
        """
        Penalizes acceleration (second derivative) for smooth trajectories.

        Args:
            refined: (B, T, J, 3) refined predictions
            valid_mask: (B, T) boolean, True for valid frames
        """
        # Velocity: first difference
        vel = refined[:, 1:] - refined[:, :-1]      # (B, T-1, J, 3)
        # Acceleration: second difference
        accel = vel[:, 1:] - vel[:, :-1]             # (B, T-2, J, 3)

        # Mask: need 3 consecutive valid frames
        mask_3 = (valid_mask[:, :-2] & valid_mask[:, 1:-1]
                  & valid_mask[:, 2:])  # (B, T-2)

        accel_mag = (accel ** 2).sum(dim=-1)  # (B, T-2, J)
        masked = accel_mag * mask_3.unsqueeze(-1)
        num_valid = mask_3.sum() * accel_mag.shape[-1]
        return masked.sum() / (num_valid + 1e-6)

    def forward(self, refined, gt, gt_valid, valid_mask):
        """
        Args:
            refined: (B, T, J, 3) refined predictions
            gt: (B, T, J, 3) ground truth keypoints
            gt_valid: (B, T, J) boolean mask for valid GT keypoints
            valid_mask: (B, T) boolean mask for valid frames

        Returns:
            total_loss, dict of individual loss components
        """
        l_pos = self.position_loss(refined, gt, gt_valid)
        l_vel = self.velocity_loss(refined, valid_mask)

        total = self.lambda_pos * l_pos + self.lambda_vel * l_vel

        losses = {'pos': l_pos.item(), 'vel': l_vel.item()}

        if self.bone_criterion is not None and self.lambda_bone > 0:
            B, T, J, _ = refined.shape
            l_bone = self.bone_criterion(
                refined.reshape(B * T, J, 3))
            total = total + self.lambda_bone * l_bone
            losses['bone'] = l_bone.item()

        losses['total'] = total.item()
        return total, losses
