"""Label-free in-room adaptation — the legitimate alternative to fine-tuning on unlabeled data.

Two mechanisms, both safe (no backprop, no labels):

  1. refresh_norm_stats — recompute ONLY the CNN head's input normalization (mean, std) from recent
     UNLABELED windows. Re-centres inputs to the new room without touching the learned weights, so
     it cannot shift the decision boundary. This is test-time normalization, not training.

  2. recalibrate — re-run quiet-room calibration (gain lock + baseline + NBVI); the strongest and
     cheapest room adaptation. Model weights are completely untouched.

If the model itself must change, retrain on a PC and upload via /api/model/upload.
"""

import numpy as np


def refresh_norm_stats(head, recent_images, *, blend=0.5) -> tuple[float, float]:
    """Update a CNN head's stored (mean, std) from recent unlabeled window images, blended with
    the training-time stats so one noisy batch can't overcorrect.

    recent_images: (m, C, K, W) array of recent live windows (no labels needed).
    blend: weight of the new stats vs existing (0 = ignore new, 1 = replace entirely).
    Returns the new (mean, std) tuple (also updated in-place on head._norm)."""
    if getattr(head, "_norm", None) is None:
        raise ValueError("head has no normalization stats — not a CNN head")
    X = np.asarray(recent_images, dtype=np.float32)
    new_mean, new_std = float(X.mean()), float(X.std()) or 1.0
    old_mean, old_std = head._norm
    mean = (1 - blend) * old_mean + blend * new_mean
    std = max((1 - blend) * old_std + blend * new_std, 1e-6)
    head._norm = (mean, std)
    return head._norm


def recalibrate(source, out_dir, *, baseline_packets=300, use_gain_lock=True) -> str:
    """Re-run quiet-room calibration on a fresh empty-room capture. This is the primary in-room
    adaptation — it re-centres the WHOLE front-end (gain + baseline + NBVI) with all model weights
    intact. Returns the saved calibration path as a string."""
    from wavetrace.Cli import calibrate_source
    path, _ = calibrate_source(source, out_dir, baseline_packets=baseline_packets,
                                use_gain_lock=use_gain_lock)
    return str(path)
