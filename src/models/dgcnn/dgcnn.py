"""DGCNN — Dynamical Graph Convolutional Neural Network for EEG.

Reference: Song et al. (2018) "EEG Emotion Recognition Using Dynamical
Graph Convolutional Neural Networks."
IEEE Trans. Affective Computing.

Architecture
------------
Input: (B, C, F)  where C=128 channels, F=5 frequency-band DE features

Learnable Adjacency Matrix
  nn.Parameter(128, 128), normalized via softmax

Graph Convolutional Layers × K
  X' = ReLU(BN(A · X · W))

Output
  Global average pooling → FC(hidden_dim, 64) → ReLU → Dropout
  embedding of shape (B, 64)

NOTE: This model expects Differential Entropy (DE) features as input,
      NOT raw time-series. Use DEFeatureTransform from data.transforms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Single graph convolutional layer: X' = σ(A · X · W + b)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=True)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, N, F_in)  — node features.
            adj: (N, N)        — normalized adjacency matrix.
        Returns:
            (B, N, F_out)
        """
        # Graph convolution: A · X
        x = torch.matmul(adj, x)          # (B, N, F_in)
        # Feature transform: (A·X) · W + b
        x = self.weight(x)                # (B, N, F_out)
        # BatchNorm over feature dim
        B, N, Feat = x.shape
        x = self.bn(x.reshape(B * N, Feat)).reshape(B, N, Feat)
        return F.relu(x)


class DGCNN(nn.Module):
    """Dynamical Graph CNN EEG feature extractor.

    Args:
        n_channels:    Number of EEG channels (graph nodes).
        in_features:   Input feature dim per node (5 for DE bands).
        hidden_dim:    Hidden dimension in graph conv layers.
        n_layers:      Number of graph conv layers.
        dropout:       Dropout probability.

    Attributes:
        feature_dim: Size of the output embedding vector (64).
    """

    def __init__(
        self,
        n_channels: int = 128,
        in_features: int = 5,
        hidden_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels

        # Learnable adjacency matrix — dynamically learned during training
        self.adj = nn.Parameter(torch.randn(n_channels, n_channels))

        # Graph convolutional layers
        layers = []
        dims = [in_features] + [hidden_dim] * n_layers
        for i in range(n_layers):
            layers.append(GraphConvLayer(dims[i], dims[i + 1]))
        self.gc_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

        # Projection to compact embedding
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.feature_dim = 64

    def _get_adj(self) -> torch.Tensor:
        """Normalize adjacency matrix via row-wise softmax."""
        return F.softmax(self.adj, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, F) — DE features per channel.
               C = n_channels, F = in_features (5 bands).
        Returns:
            logits of shape (B, nb_classes).
        """
        adj = self._get_adj()              # (N, N)

        for gc in self.gc_layers:
            x = gc(x, adj)                 # (B, N, hidden_dim)
            x = self.dropout(x)

        # Global average pooling over nodes
        x = x.mean(dim=1)                 # (B, hidden_dim)
        return self.projection(x)         # (B, feature_dim)
