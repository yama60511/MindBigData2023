"""HDF5-backed PyTorch Dataset for MindBigData2023.

Each HDF5 file (train/val/test) contains pre-filtered data:
    eeg    : (N, 128, 500)  float32  — 1–40 Hz bandpass, µV
    labels : (N,)           int64    — digit 0–9 or -1 (black-screen)

File handles are opened lazily per DataLoader worker so num_workers > 0
works safely (each worker gets its own independent h5py file handle).
"""
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """Random-access dataset backed by a single HDF5 file.

    Args:
        h5_path:   Path to the HDF5 file (train.h5 / val.h5 / test.h5).
        transform: Optional callable applied to the raw EEG tensor.
                   Receives a FloatTensor of shape (128, 500) and should
                   return a tensor (shape may differ, e.g. (128, 5) for DE).
    """

    def __init__(
        self,
        h5_path: str | Path,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.transform = transform
        self._file: h5py.File | None = None  # opened lazily per worker

        # Read length once at init (does not keep the file open)
        with h5py.File(self.h5_path, "r") as f:
            self._len = int(f["eeg"].shape[0])
            # support both key names
            self._label_key = "label" if "label" in f else "labels"

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return (eeg, label) for sample `idx`.

        Returns:
            eeg:   FloatTensor — shape depends on transform:
                   (128, 500) if no transform, or whatever the transform yields.
            label: int, digit 0–9 or -1 for black-screen trials.
        """
        if self._file is None:
            # Opened here so each DataLoader worker gets its own handle
            self._file = h5py.File(self.h5_path, "r")

        eeg   = torch.from_numpy(self._file["eeg"][idx])          # (128, 500)
        label = int(self._file[self._label_key][idx])

        if self.transform is not None:
            eeg = self.transform(eeg)

        return eeg, label

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
