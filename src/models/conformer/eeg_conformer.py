"""EEG Conformer — Convolutional Transformer for EEG decoding.

Reference: Song et al. (2022) "EEG Conformer: Convolutional Transformer
for EEG Decoding and Visualization."
https://ieeexplore.ieee.org/document/9991178

Architecture
------------
Input : (B, 1, C, T)

PatchEmbedding (CNN front-end)
  Conv2d → Conv2d → BN → ELU → AvgPool → Dropout → Conv2d → Rearrange

TransformerEncoder (nn.TransformerEncoder, pre-norm)
  depth × (LayerNorm → MultiHeadAttention → FFN → residual)

ClassificationHead
  Flatten → FC layers → nb_classes
"""
from __future__ import annotations

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """CNN front-end that converts raw EEG into a sequence of patch tokens."""

    def __init__(
        self,
        channels: int = 128,
        emb_size: int = 40,
        temp_kern: int = 25,
        pool_kern: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, temp_kern), (1, 1)),
            nn.Conv2d(40, 40, (channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kern), (1, pool_stride)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class EEGConformer(nn.Module):
    """EEG Conformer: CNN + Transformer for EEG classification.

    Args:
        nb_classes: Number of output classes.
        channels:   Number of EEG channels.
        samples:    Number of time samples.
        emb_size:   Embedding dimension for transformer tokens.
        depth:      Number of transformer encoder layers.
        num_heads:  Number of attention heads (must divide emb_size).
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        nb_classes: int = 10,
        channels: int = 128,
        samples: int = 500,
        emb_size: int = 40,
        depth: int = 6,
        num_heads: int = 10,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            channels=channels, emb_size=emb_size, dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=4 * emb_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth,
        )

        # Compute number of tokens from the patch embedding
        t_after_conv = samples - 25 + 1
        t_after_pool = (t_after_conv - 75) // 15 + 1
        n_tokens = t_after_pool

        # Classification head
        flat_size = n_tokens * emb_size
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, nb_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = x.contiguous().view(x.size(0), -1)
        return self.classifier(x)
