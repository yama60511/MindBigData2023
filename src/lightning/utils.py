"""Utility functions for building optimizers and schedulers.

Centralises the optimizer/scheduler creation logic so it can be
reused across different Lightning modules and easily extended
with new strategies.

Supported optimizers : Adam, AdamW, SGD
Supported schedulers : ReduceLROnPlateau, CosineAnnealingLR, StepLR, None
Warmup              : Optional linear warmup (recommended for Transformers)
"""

import torch
import torch.nn as nn


# ── Optimizer registry ─────────────────────────────────────────────
OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {
    "Adam":  torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD":   torch.optim.SGD,
}

# Default kwargs per optimizer (applied when the user doesn't override)
_OPTIMIZER_DEFAULTS: dict[str, dict] = {
    "Adam":  {},
    "AdamW": {"weight_decay": 1e-2},
    "SGD":   {"momentum": 0.9},
}


# ── Scheduler registry ────────────────────────────────────────────
SCHEDULER_REGISTRY: dict[str, type] = {
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "StepLR":            torch.optim.lr_scheduler.StepLR,
}

# Default kwargs per scheduler
_SCHEDULER_DEFAULTS: dict[str, dict] = {
    "ReduceLROnPlateau": {"mode": "max", "factor": 0.5, "patience": 5},
    "CosineAnnealingLR": {"T_max": 100},
    "StepLR":            {"step_size": 20, "gamma": 0.5},
}

# Schedulers that require a ``monitor`` metric
_MONITOR_SCHEDULERS = {"ReduceLROnPlateau"}


def build_optimizer(
    params,
    name: str = "Adam",
    lr: float = 1e-3,
    **kwargs,
) -> torch.optim.Optimizer:
    """Create an optimizer by name.

    Parameters
    ----------
    params : iterable
        Model parameters (e.g. ``model.parameters()``).
    name : str
        Key in ``OPTIMIZER_REGISTRY``.
    lr : float
        Learning rate.
    **kwargs
        Extra keyword arguments forwarded to the optimizer constructor.
        These take precedence over the built-in defaults.
    """
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Choose from: {list(OPTIMIZER_REGISTRY.keys())}"
        )

    merged = {**_OPTIMIZER_DEFAULTS.get(name, {}), **kwargs}
    return OPTIMIZER_REGISTRY[name](params, lr=lr, **merged)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str | None = "ReduceLROnPlateau",
    monitor: str = "val_acc",
    warmup_epochs: int = 0,
    **kwargs,
) -> dict | None:
    """Create a Lightning-compatible scheduler dict by name.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer to wrap.
    name : str or None
        Key in ``SCHEDULER_REGISTRY``, or ``None`` / ``"None"`` to skip.
    monitor : str
        Metric to monitor (only used for ReduceLROnPlateau).
    warmup_epochs : int
        Number of epochs for linear LR warmup (0 = disabled).
        During warmup, LR linearly ramps from ~0 to the base LR.
        Recommended for Transformer-based models (Conformer, CTNet, ATCNet).
    **kwargs
        Extra keyword arguments forwarded to the scheduler constructor.
        These take precedence over the built-in defaults.

    Returns
    -------
    dict or None
        A Lightning scheduler config dict, or ``None`` if disabled.
    """
    # ── Warmup-only (no main scheduler) ────────────────────────────
    if (name is None or name == "None") and warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=warmup_epochs
        )
        return {"scheduler": warmup_sched, "interval": "epoch"}

    if name is None or name == "None":
        return None

    if name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            f"Choose from: {list(SCHEDULER_REGISTRY.keys())} or None"
        )

    merged = {**_SCHEDULER_DEFAULTS.get(name, {}), **kwargs}

    # ── ReduceLROnPlateau special case (cannot use SequentialLR) ────
    # SequentialLR doesn't support ReduceLROnPlateau because it needs
    # a metric value. Use a simple LinearLR warmup only for this case.
    if name == "ReduceLROnPlateau":
        if warmup_epochs > 0:
            # Use ChainedSchedulerLR: warmup first, then plateau
            # We achieve this by returning multiple schedulers to Lightning
            warmup_sched = {
                "scheduler": torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-3, total_iters=warmup_epochs
                ),
                "interval": "epoch",
            }
            plateau_sched = {
                "scheduler": SCHEDULER_REGISTRY[name](optimizer, **merged),
                "monitor": monitor,
                "interval": "epoch",
                "frequency": 1,
            }
            # Return both as a list (Lightning supports multiple schedulers)
            return [warmup_sched, plateau_sched]

        scheduler_obj = SCHEDULER_REGISTRY[name](optimizer, **merged)
        return {
            "scheduler": scheduler_obj,
            "monitor": monitor,
            "interval": "epoch",
            "frequency": 1,
        }

    # ── Standard schedulers (support SequentialLR with warmup) ──────
    main_sched = SCHEDULER_REGISTRY[name](optimizer, **merged)

    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=warmup_epochs
        )
        combined = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[warmup_epochs],
        )
        return {"scheduler": combined, "interval": "epoch"}

    return {"scheduler": main_sched, "interval": "epoch"}

