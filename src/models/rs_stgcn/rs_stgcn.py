"""RS-STGCN — Regional-Synergy Spatio-Temporal Graph Convolutional Network.

Reference: Yunqi et al. (2025) "RS-STGCN: Regional-Synergy Spatio-Temporal
Graph Convolutional Network for Emotion Recognition."

Architecture
------------
Input: (B, C, F)  where C=128 channels, F=5 DE features

Regional Grouping
  Split channels into 5 brain regions (frontal, central, temporal,
  parietal, occipital)

Temporal Conv Block (per region)
  1D conv on feature dimension to capture intra-regional patterns

Spatial GCN Block
  Learnable adjacency for inter-regional synergy
  Graph convolution on region-level features

Output
  Flatten regions → FC(n_regions * spatial_dim, 64) → ReLU → Dropout
  embedding of shape (B, 64)

NOTE: This model expects DE features as input, not raw time-series.
      Use DEFeatureTransform from data.transforms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .channel_groups import get_group_indices


class _RegionalConv(nn.Module):
    """1D convolution within a regional group of channels."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=1),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_channels_in_region, in_features) → (B, n_ch, out_features)"""
        x = x.transpose(1, 2)   # (B, in_features, n_ch)
        x = self.conv(x)        # (B, out_features, n_ch)
        return x.transpose(1, 2)


class _SpatialGCN(nn.Module):
    """Graph convolution for inter-regional synergy."""

    def __init__(self, n_regions: int, in_features: int, out_features: int):
        super().__init__()
        self.adj = nn.Parameter(torch.randn(n_regions, n_regions))
        self.weight = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_regions, in_features) → (B, n_regions, out_features)"""
        adj = F.softmax(self.adj, dim=-1)
        x = torch.matmul(adj, x)       # (B, R, F_in)
        x = self.weight(x)             # (B, R, F_out)
        B, R, Feat = x.shape
        x = self.bn(x.reshape(B * R, Feat)).reshape(B, R, Feat)
        return F.relu(x)


class RSSTGCN(nn.Module):
    """RS-STGCN: Regional-Synergy Spatio-Temporal GCN EEG feature extractor.

    Args:
        n_channels:     Number of EEG channels.
        in_features:    Input feature dim per node (5 for DE bands).
        regional_dim:   Hidden dim for intra-regional conv.
        spatial_dim:    Hidden dim for inter-regional GCN.
        n_regions:      Number of brain regions.
        region_indices: List of list of channel indices per region.
                        If None, falls back to the mapping in channel_groups.py.
        dropout:        Dropout probability.

    Attributes:
        feature_dim: Size of the output embedding vector (64).
    """

    def __init__(
        self,
        n_channels: int = 128,
        in_features: int = 5,
        regional_dim: int = 32,
        spatial_dim: int = 32,
        n_regions: int = 5,
        region_indices: list[list[int]] | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_regions = n_regions

        # Use anatomically correct groupings by default
        if region_indices is None:
            region_indices = get_group_indices(n_channels)

        self.register_buffer(
            "_region_indices",
            torch.zeros(0),  # placeholder, actual indices stored as list
        )
        self.region_indices = region_indices

        # Intra-regional convolution (shared weights)
        self.regional_conv = _RegionalConv(in_features, regional_dim)

        # Inter-regional spatial GCN
        self.spatial_gcn1 = _SpatialGCN(n_regions, regional_dim, spatial_dim)
        self.spatial_gcn2 = _SpatialGCN(n_regions, spatial_dim, spatial_dim)

        self.dropout = nn.Dropout(dropout)

        # Projection to compact embedding
        self.projection = nn.Sequential(
            nn.Linear(n_regions * spatial_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.feature_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, F) — DE features per channel.
        Returns:
            logits (B, nb_classes).
        """
        B = x.size(0)

        # Extract per-region features via avg pooling over channels in each region
        region_features = []
        for indices in self.region_indices:
            region_x = x[:, indices, :]                 # (B, n_ch, F)
            region_x = self.regional_conv(region_x)     # (B, n_ch, regional_dim)
            region_x = region_x.mean(dim=1)             # (B, regional_dim)
            region_features.append(region_x)

        # Stack regions: (B, n_regions, regional_dim)
        x = torch.stack(region_features, dim=1)

        # Inter-regional GCN
        x = self.spatial_gcn1(x)            # (B, R, spatial_dim)
        x = self.dropout(x)
        x = self.spatial_gcn2(x)            # (B, R, spatial_dim)
        x = self.dropout(x)

        # Flatten and project
        x = x.reshape(B, -1)               # (B, R * spatial_dim)
        return self.projection(x)          # (B, feature_dim)
