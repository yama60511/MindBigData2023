"""PyTorch Lightning architectural wrappers.

This module isolates all PyTorch Lightning boilerplate logic securely
away from the pure mathematical PyTorch network definitions.
"""
from __future__ import annotations

from models import (
    EEGNet, EEGConformer, ATCNet, DGCNN,
    RSSTGCN, LMDANet, TSception, CTNet
)

from .base import BaseEEGLightningModule


class LitEEGNet(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = EEGNet(nb_classes=nb_classes, **kwargs)
        super().__init__(model, lr)
        self.save_hyperparameters(ignore=["model"])


class LitEEGConformer(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 2e-4, **kwargs):
        model = EEGConformer(nb_classes=nb_classes, **kwargs)
        super().__init__(model, lr)
        self.save_hyperparameters(ignore=["model"])


class LitATCNet(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = ATCNet(nb_classes=nb_classes, **kwargs)
        super().__init__(model, lr)
        self.save_hyperparameters(ignore=["model"])


class LitDGCNN(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = DGCNN(nb_classes=nb_classes, **kwargs)
        super().__init__(model, lr)
        self.save_hyperparameters(ignore=["model"])


class LitRSSTGCN(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = RSSTGCN(nb_classes=nb_classes, **kwargs)
        super().__init__(model, lr)
        self.save_hyperparameters(ignore=["model"])


class LitLMDANet(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = LMDANet(nb_classes=nb_classes, **kwargs)
        super().__init__(model, lr)
        self.save_hyperparameters(ignore=["model"])


class LitTSception(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = TSception(nb_classes=nb_classes, **kwargs)
        super().__init__(model, lr)
        self.save_hyperparameters(ignore=["model"])


class LitCTNet(BaseEEGLightningModule):
    def __init__(self, nb_classes: int = 10, lr: float = 1e-3, **kwargs):
        model = CTNet(nb_classes=nb_classes, **kwargs)
        super().__init__(model, lr)
        self.save_hyperparameters(ignore=["model"])
