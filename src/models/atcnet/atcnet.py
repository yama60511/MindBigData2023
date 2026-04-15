"""ATCNet — Attention Temporal Convolutional Network for EEG.

Reference: Altaheri et al. (2022) "Physics-Informed Attention Temporal
Convolutional Network for EEG-Based Motor Imagery Classification."
IEEE Trans. Industrial Informatics.

Architecture
------------
Input : (B, 1, C, T)

Conv Block (EEGNet-inspired)
  Temporal Conv → BN → Depthwise Spatial Conv → BN → ELU → AvgPool → Dropout
  Separable Conv → BN → ELU → AvgPool → Dropout

Sliding Window
  Split temporal feature map into n_windows overlapping windows

Per-Window: Attention + TCN
  Multi-Head Self-Attention (on temporal tokens)
  TCN: dilated causal 1D convolutions with residual connections

Fusion
  Concatenate all window outputs → embedding of shape (B, n_windows * tcn_channels)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """EEGNet-inspired convolutional feature extractor."""

    def __init__(
        self,
        channels: int = 128,
        F1: int = 16,
        D: int = 2,
        kern_len: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        F2 = F1 * D
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len),
                      padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8),
                      groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)   # (B, F2, 1, T//4)
        x = self.block2(x)   # (B, F2, 1, T//16)
        return x


class _MultiHeadAttention(nn.Module):
    """Simplified multi-head self-attention for temporal tokens."""

    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class _TCNBlock(nn.Module):
    """Single Temporal Convolutional Network block with residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        dilation: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        # Trim to match input length (causal padding)
        out = out[:, :, :x.size(2)]
        return F.elu(out + self.residual(x))


class _AttentionTCN(nn.Module):
    """Attention + TCN block applied to each sliding window."""

    def __init__(self, d_model: int, tcn_channels: int = 32,
                 n_heads: int = 2, dropout: float = 0.3):
        super().__init__()
        self.attention = _MultiHeadAttention(d_model, n_heads, dropout)
        self.tcn = nn.Sequential(
            _TCNBlock(d_model, tcn_channels, dilation=1, dropout=dropout),
            _TCNBlock(tcn_channels, tcn_channels, dilation=2, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = self.attention(x)           # (B, T, d_model)
        x = x.transpose(1, 2)           # (B, d_model, T)
        x = self.tcn(x)                 # (B, tcn_channels, T)
        x = x.mean(dim=-1)              # (B, tcn_channels)  global avg pool
        return x


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------

class ATCNet(nn.Module):
    """ATCNet: Attention + Temporal Convolutional Network for EEG.

    Args:
        channels:     Number of EEG channels.
        samples:      Number of time samples.
        F1:           Temporal filters in conv block.
        D:            Depth multiplier.
        kern_len:     Temporal kernel length.
        n_windows:    Number of sliding windows.
        tcn_channels: Hidden channels in TCN blocks.
        n_heads:      Attention heads.
        dropout:      Dropout probability.

    Attributes:
        feature_dim: Size of the output embedding vector.
    """

    def __init__(
        self,
        channels: int = 128,
        samples: int = 500,
        F1: int = 16,
        D: int = 2,
        kern_len: int = 64,
        n_windows: int = 5,
        tcn_channels: int = 32,
        n_heads: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_windows = n_windows
        F2 = F1 * D

        self.conv_block = _ConvBlock(
            channels=channels, F1=F1, D=D,
            kern_len=kern_len, dropout=dropout,
        )

        # Attention-TCN applied per window
        self.attn_tcn = _AttentionTCN(
            d_model=F2, tcn_channels=tcn_channels,
            n_heads=n_heads, dropout=dropout,
        )

        self.feature_dim = n_windows * tcn_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv_block(x)               # (B, F2, 1, T')
        x = x.squeeze(2)                     # (B, F2, T')
        B, F2, T = x.shape

        # Sliding window with overlap
        win_len = T // self.n_windows
        stride = max(1, (T - win_len) // max(1, self.n_windows - 1))

        window_outputs = []
        for i in range(self.n_windows):
            start = i * stride
            end = min(start + win_len, T)
            if end <= start:
                end = start + 1
            window = x[:, :, start:end]      # (B, F2, win_len)
            window = window.transpose(1, 2)  # (B, win_len, F2)
            out = self.attn_tcn(window)       # (B, tcn_channels)
            window_outputs.append(out)

        return torch.cat(window_outputs, dim=1)  # (B, feature_dim)
