"""On-disk layout of the aggregate localization room map."""

import json
from pathlib import Path

import numpy as np

from wavetrace.domain.localize import Localization


def save_localization(room_map: Localization, out_dir) -> Path:
    """Writes heatmap.npy (R×G joint pseudo-spectrum), angles.npy + ranges.npy (its grids) and
    meta.json (the peak fix) under out_dir. The per-frame track is published over the Publisher wire
    schema (RecognitionResult), not written here. O(R·G)."""
    localization_dir = Path(out_dir)
    localization_dir.mkdir(parents=True, exist_ok=True)
    np.save(localization_dir / "angles.npy", room_map.angles_deg.astype(np.float32))
    np.save(localization_dir / "ranges.npy", room_map.ranges_m.astype(np.float32))
    np.save(localization_dir / "heatmap.npy", room_map.heatmap.astype(np.float32))
    with open(localization_dir / "meta.json", "w") as meta_file:
        json.dump({
            "peak_angle_deg": room_map.peak_angle_deg,
            "peak_range_m": (None if np.isnan(room_map.peak_range_m) else room_map.peak_range_m),
            "x_m": room_map.x_m, "y_m": room_map.y_m, "confidence": room_map.confidence,
            "num_angles": int(room_map.angles_deg.size), "num_ranges": int(room_map.ranges_m.size),
        }, meta_file, indent=2)
    return localization_dir
