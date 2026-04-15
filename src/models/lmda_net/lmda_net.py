"""LMDA-Net — Lightweight Multi-Dimensional Attention Network for EEG.

Reference: Miao et al. (2023) "LMDA-Net: A lightweight multi-dimensional
attention network for general EEG-based brain-computer interfaces."
NeuroImage, 276, 120209.

Architecture
------------
Input : (B, 1, C, T)

Temporal Conv
  Conv2d(1, F1, (1, kern_len)) → BN

Channel Attention (squeeze-and-excitation style)
  GlobalAvgPool over time → FC → Sigmoid → scale channels

Depthwise Spatial Conv
  Conv2d(F1, F1*D, (C, 1), groups=F1) → BN → ELU → AvgPool → Dropout

Depth Attention (across feature depth dimension)
  AdaptiveAvgPool → Conv1d → Sigmoid → scale features

Separable Conv
  Depthwise temporal + Pointwise → BN → ELU → AvgPool → Dropout

Output
  Flatten → embedding of shape (B, F2 * T')
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention for EEG channels."""

    def __init__(self, n_channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(n_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, n_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F1, C, T) → (B, F1, C, T) attention-weighted."""
        # Global average pool over time
        attn = x.mean(dim=-1)              # (B, F1, C)
        attn = self.fc(attn)               # (B, F1, C)
        attn = attn.unsqueeze(-1)          # (B, F1, C, 1)
        return x * attn


class _DepthAttention(nn.Module):
    """Attention across the depth (feature/filter) dimension."""

    def __init__(self, depth: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(depth // reduction, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # (B, depth, 1, 1)
            nn.Flatten(1),                 # (B, depth)
            nn.Linear(depth, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, depth, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, 1, T) → (B, D, 1, T) attention-weighted."""
        w = self.attn(x)                   # (B, D)
        w = w.unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
        return x * w


class LMDANet(nn.Module):
    """LMDA-Net: Lightweight Multi-Dimensional Attention Network EEG feature extractor.

    Args:
        channels:   Number of EEG channels.
        samples:    Number of time samples.
        F1:         Number of temporal filters.
        D:          Depth multiplier.
        F2:         Pointwise filter count (default F1*D).
        kern_len:   Temporal filter length.
        dropout:    Dropout probability.

    Attributes:
        feature_dim: Size of the output embedding vector.
    """

    def __init__(
        self,
        channels: int = 128,
        samples: int = 500,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_len: int = 125,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if F2 is None:
            F2 = F1 * D

        # --- Temporal convolution ---
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len),
                      padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
        )

        # --- Channel Attention ---
        self.channel_attn = _ChannelAttention(channels, reduction=4)

        # --- Depthwise Spatial Conv ---
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # --- Depth Attention ---
        self.depth_attn = _DepthAttention(F1 * D, reduction=4)

        # --- Separable Conv ---
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        t_after_pool = (samples // 4) // 8
        self.feature_dim = F2 * t_after_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) or (B, 1, C, T).
        Returns:
            embedding (B, feature_dim).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.temporal_conv(x)       # (B, F1, C, T)
        x = self.channel_attn(x)        # (B, F1, C, T) — channel-weighted
        x = self.spatial_conv(x)        # (B, F1*D, 1, T//4)
        x = self.depth_attn(x)          # (B, F1*D, 1, T//4) — depth-weighted
        x = self.separable_conv(x)      # (B, F2, 1, T//32)
        return x.flatten(1)             # (B, feature_dim)
