"""PyTorch Lightning DataModule for MindBigData2023 (HDF5 backend).

Expects pre-built HDF5 files under data/processed/hdf5/:
    train.h5  — 88,954 samples
    val.h5    — 31,046 samples
    test.h5   — 20,000 samples

Build them with:
    python scripts/build_hdf5.py
"""
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import HDF5Dataset


class MindBigDataModule(pl.LightningDataModule):
    """LightningDataModule backed by pre-filtered HDF5 files.

    Args:
        hdf5_dir:    Directory containing train.h5, val.h5, test.h5.
        batch_size:  Samples per batch.
        num_workers: DataLoader worker processes.
                     0 is safe in Docker without --shm-size.
                     Increase if you restart the container with --shm-size=8g.
        transform:   Optional callable applied to each EEG sample.
                     Forwarded to HDF5Dataset.
    """

    def __init__(
        self,
        hdf5_dir: str | Path = "data/processed/hdf5",
        batch_size: int = 64,
        num_workers: int = 4,
        transform=None,
    ) -> None:
        super().__init__()
        self.hdf5_dir   = Path(hdf5_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        self.train_dataset: HDF5Dataset | None = None
        self.val_dataset:   HDF5Dataset | None = None
        self.test_dataset:  HDF5Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = HDF5Dataset(self.hdf5_dir / "train.h5",
                                             transform=self.transform)
            self.val_dataset   = HDF5Dataset(self.hdf5_dir / "val.h5",
                                             transform=self.transform)
        if stage in ("test", None):
            self.test_dataset  = HDF5Dataset(self.hdf5_dir / "test.h5",
                                             transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

    def __repr__(self) -> str:
        n_train = len(self.train_dataset) if self.train_dataset else "?"
        n_val   = len(self.val_dataset)   if self.val_dataset   else "?"
        n_test  = len(self.test_dataset)  if self.test_dataset  else "?"
        return (
            f"MindBigDataModule(train={n_train}, val={n_val}, test={n_test}, "
            f"batch_size={self.batch_size}, num_workers={self.num_workers})"
        )
