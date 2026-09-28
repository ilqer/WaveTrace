"""Adapt to a new room without labels, and without touching any model weights.

`refresh_norm_stats` updates the CNN's input normalization from recent windows; `recalibrate` re-runs
the quiet-room calibration. Changing the model itself means retraining and uploading.
"""

import numpy as np


def refresh_norm_stats(head, recent_images, *, blend=0.5) -> tuple[float, float]:
    """Update CNN norm (mean, std) from recent unlabeled images, blended with training stats. Returns new (mean, std)."""
    backend = getattr(head, "_backend", head)  # WeaponHead delegates cnn state to its backend
    if getattr(backend, "_normalization_stats", None) is None:
        raise ValueError("head has no normalization stats — not a CNN head")
    images = np.asarray(recent_images, dtype=np.float32)
    new_mean, new_std = float(images.mean()), float(images.std()) or 1.0
    old_mean, old_std = backend._normalization_stats
    mean = (1 - blend) * old_mean + blend * new_mean
    std = max((1 - blend) * old_std + blend * new_std, 1e-6)
    backend._normalization_stats = (mean, std)
    return backend._normalization_stats


def recalibrate(calibrate_fn, source, out_dir, *, baseline_packets=300, use_gain_lock=True) -> str:
    """Re-run quiet-room calibration (gain, baseline, NBVI) with weights intact. Returns path.

    calibrate_fn takes the same arguments as wavetrace.application.calibrate.calibrate_source.
    """
    calibration_path, _ = calibrate_fn(source, out_dir, baseline_packets=baseline_packets,
                                       use_gain_lock=use_gain_lock)
    return str(calibration_path)
