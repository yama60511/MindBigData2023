"""CTNet — Convolutional Transformer Network for EEG classification.

Reference: Zhao et al. "CTNet: A Convolutional Transformer Network for
EEG-Based Motor Imagery Classification."

Official repo: https://github.com/snailpt/CTNet

Adapted for MindBigData2023
----------------------------
  channels  : 128
  samples   : 500
  sfreq     : 250 Hz
  nb_classes: 10

Architecture
------------
Input : (B, 1, C, T)

CNN Block (lightweight, EEGNet-like)
  Temporal Conv → BN → Depthwise Spatial Conv → BN → ELU → AvgPool → Dropout
  Separable Conv → BN → ELU → AvgPool → Dropout

Reshape to Sequence
  (B, F2, 1, T') → (B, T', F2)  — temporal tokens

Transformer Encoder (lightweight)
  2 layers × (LayerNorm → MultiHeadAttention → FFN → residual)

Classification
  Global average pool over tokens → FC → nb_classes
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl


class CTNet(nn.Module):
    """CTNet: Lightweight CNN + Transformer Encoder.

    Args:
        nb_classes: Number of output classes.
        channels:   Number of EEG channels.
        samples:    Number of time samples.
        F1:         Temporal filters.
        D:          Depth multiplier.
        F2:         Pointwise filters.
        kern_len:   Temporal kernel length.
        n_heads:    Attention heads in transformer.
        depth:      Number of transformer layers.
        ff_dim:     Feed-forward hidden dimension.
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        nb_classes: int = 10,
        channels: int = 128,
        samples: int = 500,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_len: int = 125,
        n_heads: int = 4,
        depth: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # --- CNN Block 1: temporal + spatial ---
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len),
                      padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # --- CNN Block 2: separable conv ---
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=F2,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth,
        )
        self.norm = nn.LayerNorm(F2)

        # --- Classifier ---
        self.classifier = nn.Linear(F2, nb_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) or (B, 1, C, T).
        Returns:
            logits (B, nb_classes).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # CNN feature extraction
        x = self.block1(x)                 # (B, F1*D, 1, T//4)
        x = self.block2(x)                 # (B, F2, 1, T//16)

        # Reshape to sequence of temporal tokens
        x = x.squeeze(2)                  # (B, F2, T')
        x = x.transpose(1, 2)             # (B, T', F2)

        # Transformer encoder
        x = self.transformer(x)           # (B, T', F2)
        x = self.norm(x)

        # Global average pool over tokens
        x = x.mean(dim=1)                 # (B, F2)

        return self.classifier(x)
