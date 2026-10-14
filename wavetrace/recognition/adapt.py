"""Adapt to a new room without labels, and without touching any model weights.

`refresh_norm_stats` updates the CNN's input normalization from recent windows; `recalibrate` re-runs
the quiet-room calibration.
"""

import numpy as np


def refresh_norm_stats(head, recent_images, *, blend=0.5) -> tuple[float, float]:
    """Blend the CNN norm (mean, std) toward recent unlabeled images in place; returns the new (mean, std)."""
    backend = getattr(head, "_backend", head)  # WeaponHead delegates cnn state to its backend
    if getattr(backend, "_normalization_stats", None) is None:
        raise ValueError("head has no normalization stats — not a CNN head")
    images = np.asarray(recent_images, dtype=np.float32)
    new_mean, new_standard_deviation = float(images.mean()), float(images.std()) or 1.0
    old_mean, old_standard_deviation = backend._normalization_stats
    mean = (1 - blend) * old_mean + blend * new_mean
    standard_deviation = max((1 - blend) * old_standard_deviation + blend * new_standard_deviation, 1e-6)
    backend._normalization_stats = (mean, standard_deviation)
    return backend._normalization_stats


def recalibrate(calibrate_function, source, out_dir, *, baseline_packets=300, use_gain_lock=True) -> str:
    """Re-run quiet-room calibration (gain, baseline, NBVI) with weights intact. Returns path.

    calibrate_function takes the same arguments as wavetrace.pipeline.calibrate_source.
    """
    calibration_path, _ = calibrate_function(source, out_dir, baseline_packets=baseline_packets,
                                       use_gain_lock=use_gain_lock)
    return str(calibration_path)
