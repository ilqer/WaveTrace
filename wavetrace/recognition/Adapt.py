"""Label-free in-room adaptation.

1. refreshNormStats: Update CNN input norm (mean/std) from recent windows. Test-time only, no weight changes.
2. recalibrate: Re-run quiet-room calibration (gain lock, baseline, NBVI). No weight changes.

For model changes, retrain and upload.
"""

import numpy as np


def refreshNormStats(head, recent_images, *, blend=0.5) -> tuple[float, float]:
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


def recalibrate(calibrate_fn, source, out_dir, *, baseline_packets=300, use_gain_lock=True) -> str:
    """Re-run quiet-room calibration (gain, baseline, NBVI) with weights intact. Returns path.

    calibrate_fn takes the same arguments as wavetrace.Cli.calibrateSource and is injected by the
    caller so this module never imports the Cli entrypoint directly.
    """
    calPath, _ = calibrate_fn(source, out_dir, baseline_packets=baseline_packets,
                                use_gain_lock=use_gain_lock)
    return str(calPath)
