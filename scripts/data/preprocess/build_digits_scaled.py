"""Build interim HDF5 files: digits only (label != -1), EEG scaled to µV.

No bandpass filter applied.

Output: data/interim/digits_scaled/{train,val,test}.h5

Each file contains:
    eeg          : (N, 128, 500) float32  — ADC ÷ 1000 → µV
    label        : (N,)          int64
    label_source : (N,)          str
    label_pos    : (N,)          int64
    timestamp    : (N,)          float64
    sessionnum   : (N,)          int64
    blocknum     : (N,)          int64
    blockpos     : (N,)          int64

Usage
-----
    python scripts/build_digits_scaled.py
"""
from pathlib import Path

import h5py
import numpy as np

VAL_SPLIT  = 0.2
DATA_DIR   = Path("data")
SRC_DIR    = DATA_DIR / "interim/by_date"
OUT_DIR    = DATA_DIR / "interim/digits_scaled"


def write_split(npz_files: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as h:
        eeg_ds  = h.create_dataset("eeg",          shape=(0, 128, 500), maxshape=(None, 128, 500), dtype="float32", chunks=(1, 128, 500))
        lbl_ds  = h.create_dataset("label",        shape=(0,), maxshape=(None,), dtype="int64")
        src_ds  = h.create_dataset("label_source", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
        pos_ds  = h.create_dataset("label_pos",    shape=(0,), maxshape=(None,), dtype="int64")
        ts_ds   = h.create_dataset("timestamp",    shape=(0,), maxshape=(None,), dtype="float64")
        ses_ds  = h.create_dataset("sessionnum",   shape=(0,), maxshape=(None,), dtype="int64")
        blk_ds  = h.create_dataset("blocknum",     shape=(0,), maxshape=(None,), dtype="int64")
        blkp_ds = h.create_dataset("blockpos",     shape=(0,), maxshape=(None,), dtype="int64")

        for fp in npz_files:
            with np.load(fp, allow_pickle=True) as f:
                eeg  = f["eeg"]    # (N, 128, 500)
                meta = f["meta"]   # structured array

            # Keep digits only
            mask = meta["label"] != -1
            eeg  = eeg[mask]
            meta = meta[mask]
            n    = len(meta)
            if n == 0:
                continue

            # Scale ADC → µV
            eeg = (eeg / 1000.0).astype(np.float32)

            # Append to each dataset
            for ds, arr in [
                (eeg_ds,  eeg),
                (lbl_ds,  meta["label"].astype(np.int64)),
                (src_ds,  meta["label_source"].astype(str)),
                (pos_ds,  meta["label_pos"].astype(np.int64)),
                (ts_ds,   meta["timestamp"].astype(np.float64)),
                (ses_ds,  meta["sessionnum"].astype(np.int64)),
                (blk_ds,  meta["blocknum"].astype(np.int64)),
                (blkp_ds, meta["blockpos"].astype(np.int64)),
            ]:
                cur = ds.shape[0]
                ds.resize(cur + n, axis=0)
                ds[cur:] = arr

            print(f"  {fp.name}  →  {n} digit samples")

    with h5py.File(out_path) as h:
        total = h["eeg"].shape[0]
    print(f"  Saved {out_path}  ({total:,} samples)\n")


def main() -> None:
    train_files = sorted((SRC_DIR / "train").glob("*.npz"))
    test_files  = sorted((SRC_DIR / "test").glob("*.npz"))

    n_val       = max(1, round(len(train_files) * VAL_SPLIT))
    val_files   = train_files[-n_val:]
    train_files = train_files[:-n_val]

    print(f"train: {len(train_files)} files")
    write_split(train_files, OUT_DIR / "train.h5")

    print(f"val: {len(val_files)} files")
    write_split(val_files, OUT_DIR / "val.h5")

    print(f"test: {len(test_files)} files")
    write_split(test_files, OUT_DIR / "test.h5")


if __name__ == "__main__":
    main()
