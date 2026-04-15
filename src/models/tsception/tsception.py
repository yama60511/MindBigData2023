"""TSception — Multi-scale Temporal-Spatial CNN for EEG.

Reference: Ding et al. (2022) "TSception: Capturing Temporal Dynamics and
Spatial Asymmetry from EEG for Emotion Recognition."
IEEE Trans. Affective Computing, 14(3), 2238-2250.

Official repo: https://github.com/yi-ding-cs/TSception

Architecture
------------
Input : (B, 1, C, T)

Dynamic Temporal Layer
  Parallel multi-scale 1D convolutions:
    kern1 = sfreq//2 = 125  (captures ~2 Hz and above)
    kern2 = sfreq//4 = 62   (captures ~4 Hz and above)
    kern3 = sfreq//8 = 31   (captures ~8 Hz and above)
  Each branch: Conv2d(1, F1, (1, kern)) → BN → ELU → AvgPool → Dropout

Asymmetric Spatial Layer
  Global kernel:     Conv2d(F1, F1, (C, 1))   — all channels
  Hemisphere kernel: Conv2d(F1, F1, (C//2, 1)) — left/right hemisphere
  Each branch: → BN → ELU → AvgPool → Dropout

Fusion Layer
  Concatenate temporal × spatial branches → Conv2d → BN → ELU → AvgPool

Output
  Flatten → embedding of shape (B, fused_channels * T'')
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _TemporalBranch(nn.Module):
    """Single temporal convolution branch at a specific scale."""

    def __init__(self, kern_len: int, F1: int, dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len),
                      padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _SpatialBranch(nn.Module):
    """Spatial convolution branch (global or hemisphere)."""

    def __init__(self, F1: int, n_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(F1, F1, (n_channels, 1), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TSception(nn.Module):
    """TSception: Multi-scale Temporal-Spatial CNN EEG feature extractor.

    Args:
        channels:   Number of EEG channels.
        samples:    Number of time samples.
        sfreq:      Sampling frequency (used to compute kernel sizes).
        F1:         Filters per temporal branch.
        dropout:    Dropout probability.

    Attributes:
        feature_dim: Size of the output embedding vector.
    """

    def __init__(
        self,
        channels: int = 128,
        samples: int = 500,
        sfreq: int = 250,
        F1: int = 9,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.channels = channels

        # --- Dynamic Temporal Layer: 3 scales ---
        kern_sizes = [sfreq // 2, sfreq // 4, sfreq // 8]  # [125, 62, 31]
        self.temp_branches = nn.ModuleList([
            _TemporalBranch(k, F1, dropout) for k in kern_sizes
        ])
        n_temp_branches = len(kern_sizes)

        # After temporal conv + pool: T' = samples // 4
        # Concatenate along filter dim: n_temp_branches * F1

        # --- Asymmetric Spatial Layer ---
        total_F1 = n_temp_branches * F1
        half_ch = channels // 2

        # Global spatial: covers all channels
        self.spatial_global = _SpatialBranch(total_F1, channels, dropout)
        # Left hemisphere
        self.spatial_left = _SpatialBranch(total_F1, half_ch, dropout)
        # Right hemisphere
        self.spatial_right = _SpatialBranch(total_F1, half_ch, dropout)

        # --- Fusion Layer ---
        n_spatial_branches = 3  # global + left + right
        fused_channels = n_spatial_branches * total_F1

        # Compute temporal size after pooling
        t_after_pool = samples // 4

        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, (1, 1), bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        t_final = t_after_pool // 8
        self.feature_dim = fused_channels * t_final

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) or (B, 1, C, T).
        Returns:
            logits (B, nb_classes).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # --- Temporal branches (parallel, multi-scale) ---
        temp_outs = [branch(x) for branch in self.temp_branches]

        # All branches produce (B, F1, C, T//4)
        # Unify temporal dimension (may differ slightly due to padding)
        min_t = min(t.size(-1) for t in temp_outs)
        temp_outs = [t[:, :, :, :min_t] for t in temp_outs]

        # Concatenate along filter dimension
        x = torch.cat(temp_outs, dim=1)    # (B, total_F1, C, T')

        # --- Spatial branches ---
        # Global
        s_global = self.spatial_global(x)  # (B, total_F1, 1, T')

        # Hemispheres (split channels)
        half = self.channels // 2
        s_left = self.spatial_left(x[:, :, :half, :])   # (B, total_F1, 1, T')
        s_right = self.spatial_right(x[:, :, half:, :]) # (B, total_F1, 1, T')

        # Concatenate spatial branches along filter dimension
        x = torch.cat([s_global, s_left, s_right], dim=1)  # (B, 3*total_F1, 1, T')

        # --- Fusion ---
        x = self.fusion(x)                # (B, fused, 1, T'')
        return x.flatten(1)               # (B, feature_dim)
