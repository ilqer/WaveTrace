import glob
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from wavetrace import GainLock
from wavetrace.adapters.recognition.session import InferenceSession, open_session
from wavetrace.adapters.store.calibration import load_calibration
from wavetrace.application.serve import WindowView
from wavetrace.domain.calibration import CalibrationResult
from wavetrace.domain.dsp.frontend import iter_windows
from wavetrace.domain.dsp.resample import resample_uniform
from wavetrace.domain.model_config import ModelConfig
from wavetrace.domain.recognition.input_plan import plan_input


@dataclass(slots=True)
class NodeServer:
    """One rx node's calibration and head as the live loops serve it. `global_class_columns`,
    `accuracy` and `weight` are filled in after all nodes load, so it stays mutable."""
    calibration: CalibrationResult
    gain_lock: GainLock | None
    should_use_intercarrier: bool
    pick_input: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    session: InferenceSession
    config: ModelConfig
    min_num_subcarriers: int = 0
    classes: list[int] = field(default_factory=list)
    global_class_columns: list[int] = field(default_factory=list)  # `classes` positions in the global class space
    positive_class_index: int = -1
    intercarrier_baseline: np.ndarray | None = None  # subtracted from the inter-carrier path; must match training
    accuracy: float = 1.0
    weight: float = 1.0


def required_subcarrier_width(result):
    """Subcarrier width the calibration needs = highest index it references + 1."""
    subcarrier_indices = [int(i) for i in list(result.subcarriers) + list(result.image_subcarriers)]
    return 1 + max(subcarrier_indices)


def discover_node_dirs(calibration_root, model_root):
    """Yield (node_id, model_dir, calibration_dir) for each `<model_root>/node<id>` that has a
    `<calibration_root>/node<id>` directory, in sorted directory-name order."""
    for model_dir in sorted(glob.glob(os.path.join(model_root, "node*"))):
        base = os.path.basename(model_dir)
        if not base[len("node"):].isdigit():
            continue
        calibration_dir = os.path.join(calibration_root, base)
        if os.path.isdir(calibration_dir):
            yield int(base[len("node"):]), model_dir, calibration_dir


def load_presence_node_servers(calibration_root, model_root):
    """({rx node_id: NodeServer}, {rx node_id: LOGO accuracy or None}) for every node with a calibration
    and a `model.joblib` presence head (a count head is a multi-class presence head). `weight` stays
    1.0 for the caller to set."""
    nodes = {}
    logo_accuracy_by_node = {}
    for node_id, model_dir, calibration_dir in discover_node_dirs(calibration_root, model_root):
        model_path = os.path.join(model_dir, "model.joblib")
        if not os.path.exists(model_path):
            continue
        calibration, gain_lock = load_calibration(calibration_dir)
        session = open_session("presence", model_path)
        should_apply_lock, intercarrier, pick = plan_input("presence", session.head)
        classes = [int(class_id) for class_id in session.head.classes_]
        nodes[node_id] = NodeServer(
            calibration=calibration, gain_lock=gain_lock if should_apply_lock else None,
            should_use_intercarrier=intercarrier, pick_input=pick, session=session,
            config=session.head.config, min_num_subcarriers=required_subcarrier_width(calibration),
            classes=classes, positive_class_index=classes.index(1) if 1 in classes else -1,
        )
        logo_accuracy_by_node[node_id] = load_logo_accuracy(os.path.join(model_dir, "metrics.json"))
    return nodes, logo_accuracy_by_node


def load_logo_accuracy(metrics_path):
    """Session-axis LOGO accuracy preferred, subject fallback; None if absent."""
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


def copy_window(features, image, intercarrier_features):
    """The last window an exhausted `iter_windows` yielded, as a WindowView owning its arrays. Its views
    still hold that window: the extractors rewrite their outputs only when they emit, so the frames pushed
    after the last yield leave them alone. O(K·W)."""
    return WindowView(features.copy(), image.copy(),
                      None if intercarrier_features is None else intercarrier_features.copy())


def last_window_proba(frames, sample_rate_hz, result, gain_lock, config, intercarrier, pick, session):
    """(last window's class probabilities, its WindowView), or (None, None) when there are too few frames
    to fill a window. O(n) front-end over the frames, one O(K·W) copy and a single prediction."""
    resampled_frames = resample_uniform(frames, sample_rate_hz)
    if len(resampled_frames) < config.window:
        return None, None
    features = image = intercarrier_features = None
    for _timestamp, features, image, intercarrier_features in iter_windows(
        resampled_frames, result.subcarriers, gain_lock,
        window=config.window, hop=config.hop, intercarrier=intercarrier,
        image_subcarriers=result.image_subcarriers,
    ):
        pass
    if features is None:
        return None, None
    window_view = copy_window(features, image, intercarrier_features)
    window_input = pick(window_view.features, window_view.image, window_view.intercarrier_features)
    return session.predict_window_proba(window_input), window_view
