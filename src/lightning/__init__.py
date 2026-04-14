"""Export lightning wrappers and registry."""
from .wrappers import (
    LitEEGNet,
    LitEEGConformer,
    LitATCNet,
    LitDGCNN,
    LitRSSTGCN,
    LitLMDANet,
    LitTSception,
    LitCTNet,
)

# Model registry: maps CLI name → LightningModule wrapper
MODEL_REGISTRY = {
    "eegnet":    LitEEGNet,
    "conformer": LitEEGConformer,
    "atcnet":    LitATCNet,
    "dgcnn":     LitDGCNN,
    "rs_stgcn":  LitRSSTGCN,
    "lmda_net":  LitLMDANet,
    "tsception": LitTSception,
    "ctnet":     LitCTNet,
}

__all__ = [
    "LitEEGNet",
    "LitEEGConformer",
    "LitATCNet",
    "LitDGCNN",
    "LitRSSTGCN",
    "LitLMDANet",
    "LitTSception",
    "LitCTNet",
    "MODEL_REGISTRY",
]
