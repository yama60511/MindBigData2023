"""Export core physiological AI models."""
from .eegnet.eegnet import EEGNet
from .conformer.eeg_conformer import EEGConformer
from .atcnet.atcnet import ATCNet
from .dgcnn.dgcnn import DGCNN
from .rs_stgcn.rs_stgcn import RSSTGCN
from .lmda_net.lmda_net import LMDANet
from .tsception.tsception import TSception
from .ctnet.ctnet import CTNet

# Models that require DE features instead of raw EEG
GRAPH_MODELS = {"dgcnn", "rs_stgcn"}

__all__ = [
    "EEGNet",
    "EEGConformer",
    "ATCNet",
    "DGCNN",
    "RSSTGCN",
    "LMDANet",
    "TSception",
    "CTNet",
    "GRAPH_MODELS",
]
