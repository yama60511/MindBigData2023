"""Verify forward pass shape for all 8 EEG models.

Run inside Docker:
    docker exec mbd2023 python test_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from models import (
    EEGNet, EEGConformer, ATCNet, DGCNN,
    RSSTGCN, LMDANet, TSception, CTNet
)


def test_model(name, model, input_shape, expected_output):
    """Test a single model's forward pass."""
    try:
        x = torch.randn(*input_shape)
        with torch.no_grad():
            out = model(x)
        if out.shape == expected_output:
            print(f"  {name:15s}  input={str(list(input_shape)):20s}  "
                  f"output={list(out.shape)}  ✓")
            return True
        else:
            print(f"  {name:15s}  input={str(list(input_shape)):20s}  "
                  f"output={list(out.shape)}  ✗ expected {list(expected_output)}")
            return False
    except Exception as e:
        print(f"  {name:15s}  ✗ ERROR: {e}")
        return False


def main():
    B = 2
    C = 128
    T = 500
    N_CLASSES = 10
    N_BANDS = 5

    print("=" * 70)
    print("Shape tests for all EEG models")
    print("=" * 70)
    print()

    results = []

    # --- Raw EEG models: input (B, C, T) → output (B, N_CLASSES) ---
    raw_models = [
        ("EEGNet",       EEGNet(nb_classes=N_CLASSES, channels=C, samples=T)),
        ("Conformer",    EEGConformer(nb_classes=N_CLASSES, channels=C, samples=T)),
        ("ATCNet",       ATCNet(nb_classes=N_CLASSES, channels=C, samples=T)),
        ("LMDA-Net",     LMDANet(nb_classes=N_CLASSES, channels=C, samples=T)),
        ("TSception",    TSception(nb_classes=N_CLASSES, channels=C, samples=T)),
        ("CTNet",        CTNet(nb_classes=N_CLASSES, channels=C, samples=T)),
    ]

    print("Raw EEG models (input: B×C×T)")
    print("-" * 70)
    for name, model in raw_models:
        ok = test_model(name, model, (B, C, T), torch.Size([B, N_CLASSES]))
        results.append((name, ok))

    # --- Graph models: input (B, C, N_BANDS) → output (B, N_CLASSES) ---
    graph_models = [
        ("DGCNN",        DGCNN(nb_classes=N_CLASSES, n_channels=C, in_features=N_BANDS)),
        ("RS-STGCN",     RSSTGCN(nb_classes=N_CLASSES, n_channels=C, in_features=N_BANDS)),
    ]

    print()
    print("Graph models (input: B×C×N_BANDS with DE features)")
    print("-" * 70)
    for name, model in graph_models:
        ok = test_model(name, model, (B, C, N_BANDS), torch.Size([B, N_CLASSES]))
        results.append((name, ok))

    # --- Summary ---
    print()
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        failed = [name for name, ok in results if not ok]
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All shape tests passed! ✓")


if __name__ == "__main__":
    main()
