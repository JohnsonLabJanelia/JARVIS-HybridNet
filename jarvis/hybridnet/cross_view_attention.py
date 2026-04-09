"""
Cross-view attention module for JARVIS-HybridNet ReprojectionLayer.
Replaces naive camera averaging with learned per-joint per-camera weights.
"""

import torch
import torch.nn as nn


class CrossViewAttention(nn.Module):
    """
    Learns per-camera per-joint attention weights from heatmap statistics,
    replacing torch.mean across cameras with a weighted sum.

    For each joint, computes summary statistics (max, mean, std, energy)
    of each camera's heatmap values across the 3D volume, then uses a
    small MLP to produce softmax-normalized camera weights.

    Args:
        num_cameras: number of camera views
        num_joints: number of keypoints
        hidden_dim: MLP hidden dimension
    """
    def __init__(self, num_cameras=16, num_joints=24, hidden_dim=64):
        super().__init__()
        self.num_cameras = num_cameras
        # Input: 4 statistics per camera (max, mean, std, energy), concatenated
        # across all cameras → num_cameras * 4 features per joint
        self.attn_net = nn.Sequential(
            nn.Linear(4 * num_cameras, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_cameras),
        )
        self.temperature = nn.Parameter(torch.ones(1))
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize near-uniform: small weights so softmax ≈ 1/C (like mean)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, cam_heatmaps):
        """
        Args:
            cam_heatmaps: (J, C, G, G, G) per-camera heatmap values at each
                          voxel in the 3D grid

        Returns:
            (J, G, G, G) attention-weighted combination across cameras
        """
        J, C, G = cam_heatmaps.shape[0], cam_heatmaps.shape[1], cam_heatmaps.shape[2]

        # Compute per-camera per-joint summary statistics
        flat = cam_heatmaps.reshape(J, C, -1)  # (J, C, G^3)
        stat_max = flat.max(dim=2).values       # (J, C)
        stat_mean = flat.mean(dim=2)            # (J, C)
        stat_std = flat.std(dim=2)              # (J, C)
        stat_energy = (flat ** 2).mean(dim=2)   # (J, C)

        stats = torch.cat([stat_max, stat_mean, stat_std, stat_energy],
                          dim=1)  # (J, C*4)

        # Compute attention weights
        attn_logits = self.attn_net(stats)  # (J, C)
        attn_weights = torch.softmax(
            attn_logits / (self.temperature + 1e-6), dim=-1)  # (J, C)

        # Weighted sum across cameras
        # cam_heatmaps: (J, C, G, G, G)
        # attn_weights: (J, C) → (J, C, 1, 1, 1)
        weighted = cam_heatmaps * attn_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return weighted.sum(dim=1)  # (J, G, G, G)
