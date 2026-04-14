"""EEG Conformer — Convolutional Transformer for EEG decoding.

Reference: Song et al. (2022) "EEG Conformer: Convolutional Transformer
for EEG Decoding and Visualization."
https://ieeexplore.ieee.org/document/9991178

Original repo: https://github.com/eeyhsong/EEG-Conformer

Adapted for MindBigData2023
----------------------------
  channels  : 128
  samples   : 500
  sfreq     : 250 Hz
  nb_classes: 10

Architecture
------------
Input : (B, 1, C, T)   where C=channels, T=samples

PatchEmbedding (CNN front-end)
  Conv2d(1, 40, (1, 25))            # temporal convolution
  Conv2d(40, 40, (C, 1))            # spatial convolution
  BatchNorm2d → ELU
  AvgPool2d((1, 75), (1, 15))       # temporal pooling / patch slicing
  Dropout
  Conv2d(40, emb_size, (1,1))       # projection to embedding dim
  Rearrange to sequence tokens

TransformerEncoder × depth
  LayerNorm → MultiHeadAttention → residual
  LayerNorm → FFN(expansion=4) → residual

ClassificationHead
  Flatten → FC layers → nb_classes
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

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


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention."""

    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = rearrange(
            self.queries(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        keys = rearrange(
            self.keys(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        values = rearrange(
            self.values(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    """Wraps a sub-module with a residual connection."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    """Position-wise feed-forward network."""

    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    """Single transformer encoder block (pre-norm)."""

    def __init__(
        self,
        emb_size: int,
        num_heads: int = 10,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
    ):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion,
                                 drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class TransformerEncoder(nn.Sequential):
    """Stacked transformer encoder blocks."""

    def __init__(self, depth: int, emb_size: int, num_heads: int = 10):
        super().__init__(
            *[TransformerEncoderBlock(emb_size, num_heads=num_heads)
              for _ in range(depth)]
        )


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------

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
        self.transformer = TransformerEncoder(
            depth=depth, emb_size=emb_size, num_heads=num_heads,
        )

        # Compute number of tokens from the patch embedding
        # After Conv2d(1,40,(1,25)): T' = samples - 25 + 1 = 476
        # After Conv2d(40,40,(C,1)): spatial dim = 1
        # After AvgPool2d((1,75),(1,15)): T'' = (T' - 75) // 15 + 1
        t_after_conv = samples - 25 + 1
        t_after_pool = (t_after_conv - 75) // 15 + 1
        n_tokens = t_after_pool  # number of sequence tokens

        # Classification head: flatten all tokens → FC layers
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
        """
        Args:
            x: (B, C, T) or (B, 1, C, T).
        Returns:
            logits of shape (B, nb_classes).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.patch_embedding(x)     # (B, n_tokens, emb_size)
        x = self.transformer(x)         # (B, n_tokens, emb_size)
        x = x.contiguous().view(x.size(0), -1)
        logits = self.classifier(x)
        return logits
