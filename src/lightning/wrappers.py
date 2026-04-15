"""PyTorch Lightning architectural wrappers.

This module isolates all PyTorch Lightning boilerplate logic securely
away from the pure mathematical PyTorch network definitions.
"""
from models import (
    EEGNet, EEGConformer, ATCNet, DGCNN,
    RSSTGCN, LMDANet, TSception, CTNet,
    ClassificationHead,
)

from .base import BaseEEGLightningModule


class LitEEGNet(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = EEGNet(**kwargs)
        head = ClassificationHead(model.feature_dim, nb_classes)
        super().__init__(model, head, lr)
        self.save_hyperparameters(ignore=["model", "head"])


class LitEEGConformer(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 2e-4, **kwargs):
        model = EEGConformer(**kwargs)
        head = ClassificationHead(model.feature_dim, nb_classes)
        super().__init__(model, head, lr)
        self.save_hyperparameters(ignore=["model", "head"])


class LitATCNet(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = ATCNet(**kwargs)
        head = ClassificationHead(model.feature_dim, nb_classes)
        super().__init__(model, head, lr)
        self.save_hyperparameters(ignore=["model", "head"])


class LitDGCNN(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = DGCNN(**kwargs)
        head = ClassificationHead(model.feature_dim, nb_classes)
        super().__init__(model, head, lr)
        self.save_hyperparameters(ignore=["model", "head"])


class LitRSSTGCN(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = RSSTGCN(**kwargs)
        head = ClassificationHead(model.feature_dim, nb_classes)
        super().__init__(model, head, lr)
        self.save_hyperparameters(ignore=["model", "head"])


class LitLMDANet(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = LMDANet(**kwargs)
        head = ClassificationHead(model.feature_dim, nb_classes)
        super().__init__(model, head, lr)
        self.save_hyperparameters(ignore=["model", "head"])


class LitTSception(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = TSception(**kwargs)
        head = ClassificationHead(model.feature_dim, nb_classes)
        super().__init__(model, head, lr)
        self.save_hyperparameters(ignore=["model", "head"])


class LitCTNet(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = CTNet(**kwargs)
        head = ClassificationHead(model.feature_dim, nb_classes)
        super().__init__(model, head, lr)
        self.save_hyperparameters(ignore=["model", "head"])
