"""Per-link serving math shared by every live front end (presence, count and weapon)."""

import json

import numpy as np

from wavetrace.frontend import iter_windows
from wavetrace.domain.dsp.resample import resample_uniform


def required_subcarrier_width(result):
    """Subcarrier width the calibration needs = highest index it references + 1."""
    subcarrier_indices = [int(i) for i in list(result.subcarriers) + list(result.image_subcarriers)]
    return 1 + max(subcarrier_indices)


def load_logo_accuracy(metrics_path):
    """A node head's honest (LOGO) accuracy — session axis preferred, subject fallback; None if absent."""
    try:
        with open(metrics_path) as metrics_file:
            logo = json.load(metrics_file).get("logo", {})
    except (OSError, ValueError):
        return None
    for axis in ("session", "subject"):
        accuracy = logo.get(axis, {}).get("accuracy")
        if accuracy is not None:
            return float(accuracy)
    return None


def last_window_proba(frames, sample_rate_hz, result, gain_lock, config, intercarrier, pick, session):
    """Last window's class probabilities, or None when there are too few frames to fill a window.
    O(n) front-end over the frames, but a single prediction."""
    resampled_frames = resample_uniform(frames, sample_rate_hz)
    if len(resampled_frames) < config.window:
        return None
    # iter_windows yields its reused buffers and keeps pushing frames after the last yield, so each
    # window's input is copied into one buffer; only the last one is ever predicted
    last_input = None
    for _timestamp, features, image, intercarrier_features in iter_windows(
        resampled_frames, result.subcarriers, gain_lock,
        window=config.window, hop=config.hop, intercarrier=intercarrier,
        image_subcarriers=result.image_subcarriers,
    ):
        window_input = pick(features, image, intercarrier_features)
        if last_input is None:
            last_input = np.empty_like(window_input)
        np.copyto(last_input, window_input)
    return None if last_input is None else session.predict_window_proba(last_input)
