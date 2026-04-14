"""Channel grouping for regional brain areas.

Maps the 128-channel EEG montage into anatomical regions for RS-STGCN.

Assumptions
-----------
We assume a standard Geodesic Sensor Net (GSN) HydroCel 128-channel layout
(EGI / NetStation convention).  The grouping below is approximate and based
on standard anatomical correspondences.

If the actual montage differs, update the `_GSN128_GROUPS` dictionary.
"""
from __future__ import annotations


# Regional groups for ~128-channel GSN layout (0-indexed channel numbers).
# Grouped by scalp region based on standard EGI GSN-128 electrode positions.
_GSN128_GROUPS: dict[str, list[int]] = {
    # Prefrontal & Frontal (Fp, AF, F positions)
    "frontal": list(range(0, 26)),         # channels 0–25  (≈26 channels)

    # Central (C, FC positions)
    "central": list(range(26, 52)),        # channels 26–51 (≈26 channels)

    # Temporal (T, FT, TP positions)
    "temporal": list(range(52, 78)),       # channels 52–77 (≈26 channels)

    # Parietal (P, CP positions)
    "parietal": list(range(78, 104)),      # channels 78–103 (≈26 channels)

    # Occipital (O, PO positions)
    "occipital": list(range(104, 128)),    # channels 104–127 (≈24 channels)
}

REGION_NAMES = list(_GSN128_GROUPS.keys())
N_REGIONS = len(REGION_NAMES)


def get_channel_groups(n_channels: int = 128) -> dict[str, list[int]]:
    """Return channel-index groups for regional brain areas.

    Args:
        n_channels: Total number of EEG channels. If different from 128,
                    channels are split into 5 approximately equal groups.

    Returns:
        Dictionary mapping region name → list of channel indices.
    """
    if n_channels == 128:
        return _GSN128_GROUPS.copy()

    # Fallback: equal-sized groups
    group_size = n_channels // N_REGIONS
    remainder = n_channels % N_REGIONS
    groups = {}
    start = 0
    for i, name in enumerate(REGION_NAMES):
        size = group_size + (1 if i < remainder else 0)
        groups[name] = list(range(start, start + size))
        start += size
    return groups


def get_group_indices(n_channels: int = 128) -> list[list[int]]:
    """Return a list of channel-index lists (ordered by region).

    Convenience wrapper for models that need an ordered list of groups
    rather than a named dictionary.
    """
    groups = get_channel_groups(n_channels)
    return [groups[name] for name in REGION_NAMES]
