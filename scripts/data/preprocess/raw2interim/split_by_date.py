"""
Combine MindBigData2023 train.csv + test.csv into per-date .npz files.

Two phases:
  1. Stream both CSVs in chunks → append matching rows to per-date temp CSVs
  2. For each temp CSV → convert to compressed .npz and delete temp file

Memory usage at any time: ~one chunk (1024 rows) during phase 1,
~one date's rows during phase 2.

Output: data/interim/by_date/{date}.npz

Each .npz contains:
  - label:  array (n,)
  - meta:   structured array with label_source, label_pos,
            timestamp, sessionnum, blocknum, blockpos
  - eeg:    float32 array (n, 128, 500)  — scaled to µV (ADC ÷ 1000)
  - images: float32 array (n, 28, 28)

Usage:
    python scripts/data/preprocess/raw2interim/split_by_date.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR  = PROJECT_ROOT / "data" / "raw"
OUT_DIR  = PROJECT_ROOT / "data" / "interim" / "by_date"
TEMP_DIR = PROJECT_ROOT / "data" / "interim" / "by_date_tmp"

N_CHANNELS = 128
N_SAMPLES  = 500
IMG_PIXELS = 784
CHUNKSIZE  = 1024

LABEL_COL = "label"
META_COLS = ["label_source", "label_pos", "timestamp", "sessionnum", "blocknum", "blockpos"]
IMG_COLS  = [f"label_imgpix_{i}" for i in range(IMG_PIXELS)]


def get_eeg_cols(columns):
    excluded = {LABEL_COL, *META_COLS, *IMG_COLS}
    return [c for c in columns if c not in excluded]

def to_date(ts_series):
    ts = pd.to_numeric(ts_series, errors="coerce")
    unit = "ms" if ts.median() > 1e12 else "s"
    return pd.to_datetime(ts, unit=unit, utc=True).dt.strftime("%Y-%m-%d")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = [p for p in [RAW_DIR / "train.csv", RAW_DIR / "test.csv"] if p.exists()]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    header_cols = pd.read_csv(csv_paths[0], nrows=0).columns.tolist()
    eeg_cols = get_eeg_cols(header_cols)

    # Phase 1: stream CSVs, append rows to per-date temp CSVs
    for p in csv_paths:
        print(f"Phase 1: streaming {p.name}...")
        n_rows = 0
        for chunk in pd.read_csv(p, chunksize=CHUNKSIZE, low_memory=False):
            dates = to_date(chunk["timestamp"])
            for date, group in chunk.groupby(dates):
                temp_file = TEMP_DIR / f"{date}.csv"
                group.to_csv(temp_file, mode="a", header=not temp_file.exists(), index=False)
            n_rows += len(chunk)
            print(f"  {n_rows:,} rows processed...", end="\r")
        print(f"  {n_rows:,} rows processed.   ")

    # Phase 2: convert each temp CSV to .npz
    temp_files = sorted(TEMP_DIR.glob("*.csv"))
    print(f"\nPhase 2: saving {len(temp_files)} date(s)...")
    for i, temp_file in enumerate(temp_files, 1):
        date = temp_file.stem
        df = pd.read_csv(temp_file, low_memory=False)

        label = df[LABEL_COL].to_numpy()
        meta = df[META_COLS].reset_index(drop=True)
        images = df[IMG_COLS].to_numpy(dtype=np.float32).reshape(-1, 28, 28)
        eeg = df[eeg_cols].to_numpy(dtype=np.float32).reshape(-1, N_CHANNELS, N_SAMPLES) / 1000.0

        out_path = OUT_DIR / f"{date}.npz"
        np.savez_compressed(out_path, label=label, meta=meta.to_records(index=False), images=images, eeg=eeg)
        temp_file.unlink()
        print(f"  [{i}/{len(temp_files)}] {date}: {len(meta):>6} rows → {out_path.name}")

    TEMP_DIR.rmdir()
    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
