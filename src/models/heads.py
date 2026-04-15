"""Task heads — maps backbone embeddings to task outputs."""
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Linear classification head.

    Args:
        feature_dim: Output dimension of the backbone.
        num_classes:  Number of target classes.
    """

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
