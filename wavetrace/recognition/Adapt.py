"""Label-free in-room adaptation.

1. refresh_norm_stats: Update CNN input norm (mean/std) from recent windows. Test-time only, no weight changes.
2. recalibrate: Re-run quiet-room calibration (gain lock, baseline, NBVI). No weight changes.

For model changes, retrain and upload.
"""

import numpy as np


def refresh_norm_stats(head, recent_images, *, blend=0.5) -> tuple[float, float]:
    """Update CNN norm (mean, std) from recent unlabeled images, blended with training stats. Returns new (mean, std)."""
    if getattr(head, "_norm", None) is None:
        raise ValueError("head has no normalization stats — not a CNN head")
    images = np.asarray(recent_images, dtype=np.float32)
    newMean, newStd = float(images.mean()), float(images.std()) or 1.0
    oldMean, oldStd = head._norm
    mean = (1 - blend) * oldMean + blend * newMean
    std = max((1 - blend) * oldStd + blend * newStd, 1e-6)
    head._norm = (mean, std)
    return head._norm


def recalibrate(source, out_dir, *, baseline_packets=300, use_gain_lock=True) -> str:
    """Re-run quiet-room calibration (gain, baseline, NBVI) with weights intact. Returns path."""
    from wavetrace.Cli import calibrate_source
    calPath, _ = calibrate_source(source, out_dir, baseline_packets=baseline_packets,
                                use_gain_lock=use_gain_lock)
    return str(calPath)
