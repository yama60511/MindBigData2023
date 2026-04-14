"""Train EEG models on MindBigData2023.

Experiment infrastructure:
  - Hydra    → config management (configs/)
  - W&B      → experiment tracking
  - Lightning → training / testing pipeline

Usage
-----
# Default (EEGNet, W&B enabled)
python main.py

# Switch model
python main.py model=conformer

# Override hyperparameters
python main.py model=atcnet model.lr=5e-4 trainer.max_epochs=100

# Disable W&B
python main.py wandb.enabled=false

# Quick sanity check
python main.py trainer.fast_dev_run=true

# Multi-run sweep
python main.py --multirun model=eegnet,conformer,atcnet,lmda_net,tsception,ctnet

# Override experiment name
python main.py experiment.name=baseline_comparison

Inside Docker
-------------
docker exec mbd2023 bash -c "cd /workspace && python main.py model=conformer"

Output structure
----------------
outputs/<model_name>/<YYYY-MM-DD>/<HH-MM-SS>/
  ├── checkpoints/          # top-k model checkpoints
  ├── .hydra/               # config snapshots
  └── train.log             # console log
"""
import sys
import os
import logging
from pathlib import Path

import torch.multiprocessing as mp

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import MindBigDataModule
from models import GRAPH_MODELS
from lightning import MODEL_REGISTRY
from data.transforms import DEFeatureTransform

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model_kwargs(cfg: DictConfig) -> dict:
    """Extract model constructor kwargs from Hydra model config.

    Strips internal keys (prefixed with '_') and 'lr' (handled separately
    by the Lightning wrapper).
    """
    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    # Remove Hydra metadata keys
    return {k: v for k, v in model_dict.items() if not k.startswith("_")}


def _get_model_name(cfg: DictConfig) -> str:
    """Get the model CLI name from config."""
    return cfg.model._model_name_


def _build_callbacks(cfg: DictConfig, model_name: str) -> list:
    """Build Lightning callbacks."""
    callbacks = []

    # --- Model Checkpoint ---
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoints"),
            filename=model_name + "-{epoch:02d}-{val_acc:.3f}",
            monitor="val_acc",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        )
    )

    # --- Early Stopping ---
    if cfg.trainer.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_acc",
                mode="max",
                patience=cfg.trainer.patience,
            )
        )

    # --- LR Monitor (logged to W&B) ---
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    return callbacks


def _build_logger(cfg: DictConfig, model_name: str):
    """Build experiment logger (W&B or TensorBoard fallback)."""
    if cfg.wandb.enabled:
        run_name = cfg.wandb.name or model_name
        if cfg.experiment.name:
            run_name = f"{cfg.experiment.name}/{run_name}"

        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=run_name,
            tags=list(cfg.wandb.tags) + [model_name],
            notes=cfg.wandb.notes or f"Training {model_name} on MindBigData2023",
            log_model=cfg.wandb.log_model,
            save_dir=os.getcwd(),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        return wandb_logger
    else:
        return TensorBoardLogger(
            save_dir=os.getcwd(),
            name="tb_logs",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Use file-based tensor sharing so num_workers > 0 works in Docker
    mp.set_sharing_strategy("file_system")

    # Print resolved config
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Seed
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Model name
    model_name = _get_model_name(cfg)
    log.info("Model: %s", model_name)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # Resolve data path relative to original working directory
    # (Hydra changes cwd to the output dir)
    hdf5_dir = cfg.data.hdf5_dir
    if not os.path.isabs(hdf5_dir):
        hdf5_dir = os.path.join(hydra.utils.get_original_cwd(), hdf5_dir)

    transform = None
    if model_name in GRAPH_MODELS:
        sfreq = cfg.data.get("sfreq", 250)
        transform = DEFeatureTransform(sfreq=sfreq)
        log.info("Using DE feature transform for graph model '%s'", model_name)

    dm = MindBigDataModule(
        hdf5_dir=hdf5_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        transform=transform,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cls = MODEL_REGISTRY[model_name]
    model_kwargs = _build_model_kwargs(cfg)
    model = model_cls(**model_kwargs)

    # Inject trainer-level optimizers directly into the Lightning wrapper
    model.optimizer_name = cfg.trainer.get("optimizer", "Adam")
    model.scheduler_name = cfg.trainer.get("scheduler", "ReduceLROnPlateau")
    model.warmup_epochs = cfg.trainer.get("warmup_epochs", 0)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable parameters: %s", f"{n_params:,}")

    # ------------------------------------------------------------------
    # Logger & Callbacks
    # ------------------------------------------------------------------
    logger = _build_logger(cfg, model_name)
    callbacks = _build_callbacks(cfg, model_name)

    # Log model metadata to W&B
    if cfg.wandb.enabled and isinstance(logger, WandbLogger):
        logger.experiment.config.update({
            "n_params": n_params,
            "model_name": model_name,
        }, allow_val_change=True)

    # Watch model gradients in W&B
    if cfg.wandb.enabled and cfg.wandb.watch_model and isinstance(logger, WandbLogger):
        logger.watch(model, log=cfg.wandb.watch_log, log_freq=cfg.wandb.watch_freq)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        callbacks=callbacks,
        logger=logger,
        deterministic=cfg.trainer.deterministic,
        fast_dev_run=cfg.trainer.fast_dev_run,
        default_root_dir=os.getcwd(),       # use explicit absolute Hydra output dir
    )

    # ------------------------------------------------------------------
    # Fit → Test
    # ------------------------------------------------------------------
    try:
        log.info("Starting training...")
        trainer.fit(model, dm)

        if not cfg.trainer.fast_dev_run:
            log.info("Running test evaluation...")
            trainer.test(model, dm, ckpt_path="best")
    finally:
        if cfg.wandb.enabled and isinstance(logger, WandbLogger):
            import wandb
            wandb.finish()

    log.info("Done! Outputs saved to: %s", os.getcwd())


if __name__ == "__main__":
    main()
