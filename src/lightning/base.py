import torch
import torch.nn as nn
import pytorch_lightning as pl

from .utils import build_optimizer, build_scheduler

class BaseEEGLightningModule(pl.LightningModule):
    """Shared PyTorch Lightning boilerplate for all EEG models.
    
    Subclasses must only initialize their inner `nn.Module` and pass it
    along with the learning rate to `super().__init__(model, lr)`.
    """
    def __init__(self, model: nn.Module, lr: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch: tuple, stage: str) -> torch.Tensor:
        eeg, labels = batch
        logits = self.model(eeg)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            f"{stage}_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt_cfg = getattr(self, "optimizer_cfg", None)
        sched_cfg = getattr(self, "scheduler_cfg", None)

        # Extract name + hyperparams from optimizer config
        opt_name = opt_cfg.name if opt_cfg else "AdamW"
        opt_kwargs = {k: v for k, v in opt_cfg.items() if k != "name"} if opt_cfg else {}

        # Extract name, warmup, + hyperparams from scheduler config
        sched_name = sched_cfg.name if sched_cfg else "CosineAnnealingLR"
        warmup = int(sched_cfg.warmup_epochs) if sched_cfg else 0
        sched_kwargs = {k: v for k, v in sched_cfg.items() if k not in ("name", "warmup_epochs")} if sched_cfg else {}

        optimizer = build_optimizer(self.parameters(), name=opt_name, lr=self.lr, **opt_kwargs)
        scheduler = build_scheduler(
            optimizer, name=sched_name, warmup_epochs=warmup, **sched_kwargs
        )

        if scheduler is None:
            return optimizer
        # build_scheduler returns a list when warmup + ReduceLROnPlateau
        if isinstance(scheduler, list):
            return [optimizer], scheduler
        return [optimizer], [scheduler]
