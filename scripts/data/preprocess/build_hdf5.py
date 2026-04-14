#!/usr/bin/env python
"""Build processed HDF5 files from interim per-date npz files.

Source : data/interim/by_date/train/*.npz  (76 files, ~1668 samples each)
         data/interim/by_date/test/*.npz   ( 9 files, ~1024 samples each)

For each file:
  - Loads EEG (N, 128, 500) and labels from meta['label']  (images skipped)
  - Converts ADC counts → µV  (÷ 1000)
  - Applies 1–40 Hz IIR bandpass filter (order-4 Butterworth, sosfiltfilt)
  - Appends to HDF5

Train/val split (temporal)
--------------------------
Files are sorted by date.  The last val_split fraction of train dates
becomes val — same strategy as the DataModule.

    76 files, val_split=0.2  →  61 train files + 15 val files

Output
------
    data/processed/train.h5
    data/processed/val.h5
    data/processed/test.h5

HDF5 layout (each file)
-----------------------
    eeg    : (N, 128, 500)  float32   — filtered, µV
    labels : (N,)           int64     — digit 0–9 or -1 (black-screen)

Usage
-----
    python scripts/build_hdf5.py
    python scripts/build_hdf5.py --val_split 0.2 --l_freq 1 --h_freq 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sos(l_freq: float, h_freq: float, sfreq: float,
              order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    return butter(order, [l_freq / nyq, h_freq / nyq],
                  btype="bandpass", output="sos")


def _open_h5(path: Path, n_ch: int = 128, n_t: int = 500) -> tuple[h5py.File, h5py.Dataset, h5py.Dataset]:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = h5py.File(path, "w")
    eeg_ds = h.create_dataset(
        "eeg", shape=(0, n_ch, n_t), maxshape=(None, n_ch, n_t),
        dtype="float32", chunks=(1, n_ch, n_t),  # one sample per chunk → O(1) random reads
    )
    lbl_ds = h.create_dataset(
        "labels", shape=(0,), maxshape=(None,),
        dtype="int64",
    )
    return h, eeg_ds, lbl_ds


def _append(eeg_ds: h5py.Dataset, lbl_ds: h5py.Dataset,
            eeg: np.ndarray, labels: np.ndarray) -> None:
    n = len(labels)
    cur = eeg_ds.shape[0]
    eeg_ds.resize(cur + n, axis=0)
    lbl_ds.resize(cur + n, axis=0)
    eeg_ds[cur:] = eeg
    lbl_ds[cur:] = labels


def process_files(
    npz_files: list[Path],
    out_path: Path,
    sos: np.ndarray,
    split_name: str,
) -> None:
    h, eeg_ds, lbl_ds = _open_h5(out_path)
    try:
        for fp in tqdm(npz_files, desc=split_name, unit="file"):
            with np.load(fp, allow_pickle=True) as f:
                eeg = f["eeg"].copy()    # (N, 128, 500) float32, ADC counts
                meta = f["meta"].copy()

            labels = meta["label"].astype(np.int64)

            # ADC counts → µV, then bandpass filter along time axis
            eeg = (eeg / 1000.0).astype(np.float32)
            eeg = sosfiltfilt(sos, eeg, axis=-1).astype(np.float32)

            _append(eeg_ds, lbl_ds, eeg, labels)

        n_total = eeg_ds.shape[0]
    finally:
        h.close()

    print(f"  {split_name:5s} → {out_path}  ({n_total:,} samples, {len(npz_files)} files)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="data")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--l_freq",    type=float, default=1.0)
    p.add_argument("--h_freq",    type=float, default=40.0)
    p.add_argument("--sfreq",     type=float, default=250.0)
    args = p.parse_args()

    data_dir  = Path(args.data_dir)
    npz_train = sorted((data_dir / "interim/by_date/train").glob("*.npz"))
    npz_test  = sorted((data_dir / "interim/by_date/test").glob("*.npz"))
    out_dir   = data_dir / "processed" / "hdf5"

    sos = _make_sos(args.l_freq, args.h_freq, args.sfreq)

    # Temporal train/val split
    n_val = max(1, round(len(npz_train) * args.val_split))
    val_files   = npz_train[-n_val:]
    train_files = npz_train[:-n_val]

    print(f"Filter   : {args.l_freq}–{args.h_freq} Hz  IIR Butterworth order-4")
    print(f"Train    : {len(train_files)} files  |  Val: {len(val_files)} files  |  Test: {len(npz_test)} files\n")

    process_files(train_files, out_dir / "train.h5", sos, "train")
    process_files(val_files,   out_dir / "val.h5",   sos, "val")
    process_files(npz_test,    out_dir / "test.h5",  sos, "test")

    print("\nDone. Final sizes:")
    for name in ("train", "val", "test"):
        path = out_dir / f"{name}.h5"
        with h5py.File(path) as f:
            e, l = f["eeg"], f["labels"]
            print(f"  {name:5s}.h5  eeg={e.shape}  labels={l.shape}"
                  f"  {path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
