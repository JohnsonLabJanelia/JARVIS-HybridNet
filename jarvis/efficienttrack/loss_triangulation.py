"""
Triangulation Residual Loss for semi-supervised multi-view pose estimation.

Based on: "Triangulation Residual Loss for Data-efficient 3D Pose Estimation"
(Zhao et al., NeurIPS 2023). Trains the 2D detector to produce geometrically
consistent predictions across camera views on unlabeled frames.

The key idea: for unlabeled multi-view frames, the 2D predictions from all
cameras should triangulate to a single consistent 3D point. The TR loss
measures this consistency as the smallest singular value of the triangulation
matrix — no 3D ground truth needed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftArgmax2d(nn.Module):
    """Differentiable 2D soft-argmax: heatmap → continuous (x, y) coordinates.

    Unlike hard argmax, gradients flow through this operation, enabling
    end-to-end training with geometric losses.

    Args:
        temperature: Controls sharpness. Lower = sharper (closer to argmax).
                     Default 1.0 is a good starting point.
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        # Cache coordinate grids (created on first forward)
        self._grid_x = None
        self._grid_y = None
        self._grid_shape = None

    def forward(self, heatmaps):
        """
        Args:
            heatmaps: (B, N_joints, H, W)
        Returns:
            coords: (B, N_joints, 2) — (x, y) in heatmap pixel coordinates
        """
        B, N, H, W = heatmaps.shape
        device = heatmaps.device

        # Create/cache coordinate grids
        if self._grid_shape != (H, W) or self._grid_x is None \
                or self._grid_x.device != device:
            pos_x = torch.arange(W, dtype=heatmaps.dtype, device=device)
            pos_y = torch.arange(H, dtype=heatmaps.dtype, device=device)
            grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')
            self._grid_x = grid_x.reshape(-1)  # (H*W,)
            self._grid_y = grid_y.reshape(-1)
            self._grid_shape = (H, W)

        # Spatial softmax
        flat = heatmaps.view(B, N, -1) / self.temperature
        weights = F.softmax(flat, dim=-1)  # (B, N, H*W)

        # Weighted sum = expected coordinates
        exp_x = torch.sum(self._grid_x * weights, dim=-1)  # (B, N)
        exp_y = torch.sum(self._grid_y * weights, dim=-1)

        return torch.stack([exp_x, exp_y], dim=-1)  # (B, N, 2)


class TriangulationResidualLoss(nn.Module):
    """Triangulation Residual Loss for multi-view geometric consistency.

    For each keypoint, constructs the DLT triangulation matrix A from
    2D predictions across all cameras, then minimizes sigma_min(A).

    When predictions are geometrically consistent (view rays converge),
    sigma_min → 0. No 3D labels needed.

    Args:
        temperature: Soft-argmax temperature (default 1.0)
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.soft_argmax = SpatialSoftArgmax2d(temperature=temperature)

    def forward(self, heatmaps, proj_matrices, heatmap_to_image_scale=2.0):
        """
        Args:
            heatmaps: (B, N_cams, N_joints, H, W) — 2D heatmaps per camera
            proj_matrices: (N_cams, 3, 4) — camera projection matrices
            heatmap_to_image_scale: Scale factor from heatmap coords to image
                                    coords (e.g., 2.0 if heatmap is half-res)
        Returns:
            tr_loss: scalar — mean TR loss across batch and joints
        """
        B, N_cams, N_joints, H, W = heatmaps.shape

        # Step 1: Extract differentiable 2D coordinates via soft-argmax
        hm_flat = heatmaps.view(B * N_cams, N_joints, H, W)
        coords_hm = self.soft_argmax(hm_flat)  # (B*N_cams, N_joints, 2)
        coords_hm = coords_hm.view(B, N_cams, N_joints, 2)

        # Scale to image coordinates
        coords = coords_hm * heatmap_to_image_scale  # (B, N_cams, N_joints, 2)

        # Get confidence from heatmap peak values (detached — don't backprop through)
        with torch.no_grad():
            conf = hm_flat.view(B * N_cams, N_joints, -1).max(dim=-1)[0]
            conf = conf.view(B, N_cams, N_joints)
            conf = conf / (conf.max() + 1e-8)  # normalize to [0, 1]
            conf = conf.clamp(min=0.3, max=0.7)  # prevent degenerate weighting

        # Step 2: Build triangulation matrix A for each joint
        # A[joint] is (2*N_cams, 4): for each camera, two rows
        #   row1 = u * P[2,:] - P[0,:]
        #   row2 = v * P[2,:] - P[1,:]
        u = coords[..., 0]  # (B, N_cams, N_joints)
        v = coords[..., 1]

        P = proj_matrices  # (N_cams, 3, 4)
        P0 = P[:, 0, :].unsqueeze(0).unsqueeze(2).expand(B, -1, N_joints, -1)
        P1 = P[:, 1, :].unsqueeze(0).unsqueeze(2).expand(B, -1, N_joints, -1)
        P2 = P[:, 2, :].unsqueeze(0).unsqueeze(2).expand(B, -1, N_joints, -1)

        row1 = u.unsqueeze(-1) * P2 - P0  # (B, N_cams, N_joints, 4)
        row2 = v.unsqueeze(-1) * P2 - P1

        # Apply confidence weighting
        w = conf.unsqueeze(-1)  # (B, N_cams, N_joints, 1)
        row1 = row1 * w
        row2 = row2 * w

        # Interleave rows: (B, N_cams, N_joints, 2, 4)
        rows = torch.stack([row1, row2], dim=3)

        # Reshape to (B, N_joints, 2*N_cams, 4)
        A = rows.permute(0, 2, 1, 3, 4).reshape(B, N_joints, 2 * N_cams, 4)

        # Step 3: TR loss = smallest singular value (always stable gradients)
        A_flat = A.reshape(B * N_joints, 2 * N_cams, 4)

        # svdvals not yet on MPS — compute on CPU (fast: matrices are tiny)
        if A_flat.device.type == 'mps':
            A_cpu = A_flat.detach().cpu().requires_grad_(True)
            sigma = torch.linalg.svdvals(A_cpu)
            sigma_min = sigma[:, -1]
            loss_cpu = sigma_min.mean()
            # Manual backward: d(sigma_min)/dA = u_min * v_min^T
            loss_cpu.backward()
            # Propagate gradient back to A_flat on MPS
            return (A_flat * A_cpu.grad.to(A_flat.device)).sum()
        else:
            sigma = torch.linalg.svdvals(A_flat)  # (B*N_joints, 4)
            sigma_min = sigma[:, -1]
            return sigma_min.mean()


def tr_loss_weight(epoch, total_epochs, max_weight=1.0, ramp_fraction=0.3):
    """Gaussian ramp-up schedule for TR loss weight.

    Slowly increases the TR loss contribution to prevent degenerate solutions
    in early training when the 2D detector is still learning.

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total training epochs
        max_weight: Maximum TR loss weight
        ramp_fraction: Fraction of epochs for ramp-up (default 0.3 = 30%)
    """
    ramp_epochs = int(total_epochs * ramp_fraction)
    if ramp_epochs == 0 or epoch >= ramp_epochs:
        return max_weight
    t = epoch / ramp_epochs
    return max_weight * math.exp(-5.0 * (1.0 - t) ** 2)
