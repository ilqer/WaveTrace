from collections.abc import Callable

import numpy as np


def plan_input(mode: str, head) -> tuple[bool, bool, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
    """Return (should_apply_lock, intercarrier, pick) for the serving loop.
    `pick` returns the row fed to `InferenceSession.predict_window`.
    A presence head always takes the plain feature vector; a weapon head is self-describing via
    `head.feature_mode`, falling back on `head.default_feature_mode` for models saved before that
    was recorded."""
    if mode == "presence":
        return True, False, (lambda features, image, intercarrier: features)
    feature_mode = getattr(head, "feature_mode", None) or head.default_feature_mode
    if feature_mode == "cnn":
        return False, False, (lambda features, image, intercarrier: image.reshape(-1))
    if feature_mode == "fusion":
        return True, True, (lambda features, image, intercarrier: np.hstack([intercarrier, features]))
    return False, True, (lambda features, image, intercarrier: intercarrier)  # ic27 / variance
