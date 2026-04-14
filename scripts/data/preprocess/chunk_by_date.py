"""
Split MindBigData2023 train/test CSVs into per-date chunks.

Processes one date at a time to keep memory usage low:
  1. Quick scan: read only the timestamp column to discover all unique dates
  2. For each date: read the file in chunks, keep only that date's rows,
     and save as a compressed .npz file under data/interim/by_date/

This ensures all rows for the same date are collected together,
even if they are scattered throughout the file.

Each .npz contains:
  - meta:   structured array with label, label_source, label_pos,
            timestamp, sessionnum, blocknum, blockpos
  - eeg:    float32 array (n, 128, 500)
  - images: float32 array (n, 28, 28)

Usage:
    python scripts/chunk_by_date.py              # both train and test
    python scripts/chunk_by_date.py --split train
    python scripts/chunk_by_date.py --split test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────
N_CHANNELS = 128
N_SAMPLES = 500
IMG_PIXELS = 784
EEG_SCALE = 1000.0  # ADC counts → µV

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "by_date"

META_COLS = [
    "label", "label_source", "label_pos",
    "timestamp", "sessionnum", "blocknum", "blockpos",
]
IMG_COLS = [f"label_imgpix_{i}" for i in range(IMG_PIXELS)]


def get_eeg_cols(columns: list[str]) -> list[str]:
    """Return EEG column names from the full header."""
    img_set = set(IMG_COLS)
    return [c for c in columns if "_" in c and c.split("_")[-1].isdigit() and c not in img_set]


def get_channel_names(eeg_cols: list[str]) -> list[str]:
    return list(dict.fromkeys(c.rsplit("_", 1)[0] for c in eeg_cols))


def detect_ts_unit(csv_path: Path) -> str:
    """Read a small sample to determine if timestamps are seconds or milliseconds."""
    sample = pd.read_csv(csv_path, usecols=["timestamp"], nrows=100)
    median_ts = pd.to_numeric(sample["timestamp"], errors="coerce").median()
    return "ms" if median_ts > 1e12 else "s"


def scan_dates(csv_path: Path, ts_unit: str) -> list[str]:
    """Quick scan: read only the timestamp column to find all unique dates."""
    dates = set()
    for chunk in pd.read_csv(csv_path, usecols=["timestamp"], chunksize=10000):
        ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
        d = pd.to_datetime(ts, unit=ts_unit, utc=True).dt.strftime("%Y-%m-%d")
        dates.update(d.dropna().unique())
    return sorted(dates)


def process_one_date(
    csv_path: Path, target_date: str, ts_unit: str,
    eeg_cols: list[str], out_dir: Path, chunksize: int,
) -> int:
    """Read the full CSV, keep only rows matching target_date, save as .npz."""
    meta_parts, img_parts, eeg_parts = [], [], []

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
        dates = pd.to_datetime(ts, unit=ts_unit, utc=True).dt.strftime("%Y-%m-%d")
        mask = dates == target_date
        if not mask.any():
            continue

        group = chunk.loc[mask]
        meta_parts.append(group[META_COLS].reset_index(drop=True))
        img_parts.append(group[IMG_COLS].to_numpy(dtype=np.float32).reshape(-1, 28, 28))
        eeg = group[eeg_cols].to_numpy(dtype=np.float32).reshape(-1, N_CHANNELS, N_SAMPLES)
        eeg /= EEG_SCALE
        eeg_parts.append(eeg)

    if not meta_parts:
        return 0

    meta = pd.concat(meta_parts, ignore_index=True)
    images = np.concatenate(img_parts, axis=0)
    eeg = np.concatenate(eeg_parts, axis=0)

    out_path = out_dir / f"{target_date}.npz"
    np.savez_compressed(
        out_path,
        meta=meta.to_records(index=False),
        images=images,
        eeg=eeg,
    )
    print(f"    {target_date}: {len(meta):>5} rows -> {out_path.name}")
    return len(meta)


def process_split(split: str, chunksize: int = 256) -> None:
    csv_path = RAW_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read header to determine columns
    header = pd.read_csv(csv_path, nrows=0)
    all_cols = header.columns.tolist()
    eeg_cols = get_eeg_cols(all_cols)
    channel_names = get_channel_names(eeg_cols)
    assert len(channel_names) == N_CHANNELS, f"Expected {N_CHANNELS} channels, got {len(channel_names)}"

    # Detect timestamp unit
    ts_unit = detect_ts_unit(csv_path)
    print(f"  Timestamp unit: {ts_unit}")

    # Step 1: quick scan to find all dates
    print(f"  Scanning dates...")
    dates = scan_dates(csv_path, ts_unit)
    print(f"  Found {len(dates)} dates: {dates}")

    # Step 2: process one date at a time
    total_rows = 0
    for i, date_str in enumerate(dates, 1):
        print(f"  [{i}/{len(dates)}] Collecting {date_str}...")
        n = process_one_date(csv_path, date_str, ts_unit, eeg_cols, out_dir, chunksize)
        total_rows += n

    print(f"  {split}: done. {total_rows} total rows across {len(dates)} dates.")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into per-date chunks")
    parser.add_argument("--split", choices=["train", "test"], default=None,
                        help="Process only this split (default: both)")
    parser.add_argument("--chunksize", type=int, default=256,
                        help="CSV read chunk size (default: 256)")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["train", "test"]
    for split in splits:
        print(f"Processing {split}...")
        process_split(split, chunksize=args.chunksize)

    print("All done.")


if __name__ == "__main__":
    main()
