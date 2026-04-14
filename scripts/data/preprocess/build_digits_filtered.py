"""Apply 1-40 Hz bandpass filter to interim/digits_scaled and save to processed.

Pipeline:
    interim/digits_scaled  →  processed/digits_1_40hz
    (scaled, no filter)        (scaled + filtered)

Usage
-----
    python scripts/build_digits_filtered.py
"""
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt

DATA_DIR = Path("data")
SRC_DIR  = DATA_DIR / "interim/digits_scaled"
OUT_DIR  = DATA_DIR / "processed/digits_1_40hz"

SFREQ, L_FREQ, H_FREQ = 250.0, 1.0, 40.0
CHUNK_SIZE = 1024  # samples processed at a time to limit memory


def make_sos() -> np.ndarray:
    nyq = SFREQ / 2.0
    return butter(4, [L_FREQ / nyq, H_FREQ / nyq], btype="bandpass", output="sos")


def process(src_path: Path, out_path: Path, sos: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src, h5py.File(out_path, "w") as dst:
        n = src["eeg"].shape[0]

        # Create datasets mirroring the source
        eeg_ds  = dst.create_dataset("eeg",          shape=(n, 128, 500), dtype="float32", chunks=(1, 128, 500))
        lbl_ds  = dst.create_dataset("label",        data=src["label"][:])
        src_ds  = dst.create_dataset("label_source", data=src["label_source"][:])
        pos_ds  = dst.create_dataset("label_pos",    data=src["label_pos"][:])
        ts_ds   = dst.create_dataset("timestamp",    data=src["timestamp"][:])
        ses_ds  = dst.create_dataset("sessionnum",   data=src["sessionnum"][:])
        blk_ds  = dst.create_dataset("blocknum",     data=src["blocknum"][:])
        blkp_ds = dst.create_dataset("blockpos",     data=src["blockpos"][:])

        # Filter EEG in chunks to limit peak memory
        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            eeg = src["eeg"][start:end]                              # (chunk, 128, 500)
            eeg = sosfiltfilt(sos, eeg, axis=-1).astype(np.float32)
            eeg_ds[start:end] = eeg
            print(f"  {start + len(eeg)}/{n}", end="\r")

    print(f"  Saved {out_path}  ({n:,} samples)")


def main() -> None:
    sos = make_sos()
    print(f"Filter: {L_FREQ}–{H_FREQ} Hz  IIR Butterworth order-4\n")

    for split in ("train", "val", "test"):
        print(f"{split}:")
        process(SRC_DIR / f"{split}.h5", OUT_DIR / f"{split}.h5", sos)
        print()


if __name__ == "__main__":
    main()
