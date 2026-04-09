"""
Temporal Refinement Transformer for JARVIS-HybridNet.
Takes a window of T frames of 3D keypoint predictions and refines them
using self-attention over both temporal and spatial (joint) dimensions.

Inspired by OptiPose (Patel et al., IJCV 2023).
"""

import torch
import torch.nn as nn
import math


class TemporalRefinementTransformer(nn.Module):
    """
    Refines a sequence of 3D keypoint predictions using a transformer encoder.

    Each (frame, joint) pair is treated as a token. The transformer learns
    temporal smoothness and skeletal structure through self-attention.
    Output is a residual correction added to the input predictions.

    Args:
        num_joints: number of keypoints (default 24)
        d_model: transformer embedding dimension (default 128)
        nhead: number of attention heads (default 8)
        num_layers: number of transformer encoder layers (default 4)
        dim_feedforward: FFN hidden dimension (default 512)
        window_size: maximum temporal window size T (default 32)
        dropout: dropout rate (default 0.1)
    """
    def __init__(self, num_joints=24, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, window_size=32, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model
        self.window_size = window_size

        # Input projection: (x, y, z, confidence) → d_model
        self.input_proj = nn.Linear(4, d_model)

        # Learned positional encodings
        self.temporal_embed = nn.Embedding(window_size, d_model)
        self.spatial_embed = nn.Embedding(num_joints, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head: predict residual (dx, dy, dz) per token
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize output head near zero so initial output ≈ input
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, keypoints, confidences, valid_mask=None):
        """
        Args:
            keypoints: (B, T, J, 3) predicted 3D keypoints in mm
            confidences: (B, T, J) confidence scores [0, 1]
            valid_mask: (B, T) boolean, True for valid frames.
                        If None, all frames are valid.

        Returns:
            refined: (B, T, J, 3) refined 3D keypoints
        """
        B, T, J = keypoints.shape[:3]
        device = keypoints.device

        # Build input tokens: (B, T, J, 4) = (x, y, z, conf)
        tokens_in = torch.cat([
            keypoints,
            confidences.unsqueeze(-1)
        ], dim=-1)  # (B, T, J, 4)

        # Reshape to (B, T*J, 4) and project
        tokens = tokens_in.reshape(B, T * J, 4)
        tokens = self.input_proj(tokens)  # (B, T*J, d_model)

        # Add positional encodings
        frame_idx = torch.arange(T, device=device)
        joint_idx = torch.arange(J, device=device)
        # Temporal: (T, d_model) → repeat for each joint → (T*J, d_model)
        t_pe = self.temporal_embed(frame_idx).unsqueeze(1).expand(T, J, -1)
        t_pe = t_pe.reshape(T * J, self.d_model)
        # Spatial: (J, d_model) → repeat for each frame → (T*J, d_model)
        s_pe = self.spatial_embed(joint_idx).unsqueeze(0).expand(T, J, -1)
        s_pe = s_pe.reshape(T * J, self.d_model)
        tokens = tokens + t_pe + s_pe  # (B, T*J, d_model)

        # Build attention mask for invalid frames
        src_key_padding_mask = None
        if valid_mask is not None:
            # Expand frame mask to per-token: (B, T) → (B, T*J)
            frame_mask = ~valid_mask  # True = masked (invalid)
            src_key_padding_mask = frame_mask.unsqueeze(-1).expand(
                B, T, J).reshape(B, T * J)

        # Transformer encoding
        encoded = self.encoder(
            tokens, src_key_padding_mask=src_key_padding_mask
        )  # (B, T*J, d_model)

        # Predict residuals
        residuals = self.output_head(encoded)  # (B, T*J, 3)
        residuals = residuals.reshape(B, T, J, 3)

        # Refine: add residual to input
        refined = keypoints + residuals
        return refined
