"""EEGNet — compact EEG classification CNN.

Reference: Lawhern et al. (2018) "EEGNet: a compact convolutional neural
network for EEG-based brain–computer interfaces."
https://iopscience.iop.org/article/10.1088/1741-2552/aace8c

Original Keras repo: https://github.com/aliasvishnu/EEGNet

Adapted for MindBigData2023
----------------------------
  channels  : 128
  samples   : 500
  sfreq     : 250 Hz
  nb_classes: 11  (digits 0–9  +  black-screen label remapped to 10)

Architecture
------------
Input : (B, 1, C, T)   where C=channels, T=samples

Block 1 — temporal + spatial convolution
  Conv2d(1,  F1, (1, kern_len), padding='same', bias=False)   # temporal
  BatchNorm2d(F1)
  DepthwiseConv2d(F1, F1*D, (C, 1), groups=F1, bias=False)    # spatial
  BatchNorm2d(F1*D)
  ELU
  AvgPool2d((1, 4))
  Dropout(p)

Block 2 — separable convolution
  DepthwiseConv2d(F1*D, F1*D, (1,16), groups=F1*D, padding='same', bias=False)
  PointwiseConv2d(F1*D, F2, (1,1), bias=False)
  BatchNorm2d(F2)
  ELU
  AvgPool2d((1, 8))
  Dropout(p)

Head
  Flatten → Linear(F2 * T', nb_classes)
  where T' = floor(floor(T / 4) / 8)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    """EEGNet convolutional EEG classifier.

    Args:
        nb_classes:  Number of output classes.
        channels:    Number of EEG channels (C).
        samples:     Number of time samples (T).
        F1:          Number of temporal filters in Block 1.
        D:           Depth multiplier for spatial filters.
        F2:          Number of pointwise filters in Block 2 (typically F1*D).
        kern_len:    Temporal filter length (paper recommends sfreq // 2).
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        nb_classes: int = 11,
        channels: int = 128,
        samples: int = 500,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_len: int = 125,   # 250 Hz // 2
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.nb_classes = nb_classes
        self.channels = channels
        self.samples = samples

        # --- Block 1 ---
        self.block1 = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, F1, kernel_size=(1, kern_len),
                      padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial convolution
            nn.Conv2d(F1, F1 * D, kernel_size=(channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # --- Block 2 ---
        self.block2 = nn.Sequential(
            # Depthwise temporal convolution
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            # Pointwise convolution
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # --- Classification head ---
        # Compute flattened feature size from the two pooling stages
        t_after_pool = (samples // 4) // 8   # floor divisions
        self.classifier = nn.Linear(F2 * t_after_pool, nb_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape (B, C, T) or (B, 1, C, T).
        Returns:
            logits of shape (B, nb_classes).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)   # (B, C, T) → (B, 1, C, T)

        x = self.block1(x)       # (B, F1*D, 1, T//4)
        x = self.block2(x)       # (B, F2,   1, T//32)
        x = x.flatten(1)         # (B, F2 * T//32)
        return self.classifier(x)
