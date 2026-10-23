"""Live multi-link weapon-detection serving: discover the per-link/per-node weapon heads a training
run wrote to disk, and the per-link math `WeaponServeMode` scores them with."""

import glob
import os

import numpy as np

from wavetrace.adapters.recognition.session import open_session
from wavetrace.adapters.store.calibration import load_calibration
from wavetrace.domain.dsp.frontend import iter_windows
from wavetrace.domain.dsp.resample import resample_uniform
from wavetrace.domain.recognition.input_plan import plan_input
from wavetrace.recognition.link import accuracy_weights
from wavetrace.recognition.serving import (NodeServer, copy_window, discover_node_dirs, load_logo_accuracy,
                                            required_subcarrier_width)


def dwell_proba(frames, sample_rate_hz, entry):
    """(mean class probability over every window in the buffer, soft so it composes with the cross-link
    LinkVoter; the last window's WindowView), or (None, None) when no full window fits. `entry` is a
    NodeServer from `load_weapon_links`. O(windows) predictions and one O(K·W) copy."""
    resampled = resample_uniform(frames, sample_rate_hz)
    config = entry.config
    if len(resampled) < config.window:
        return None, None
    window_probas = []
    features = image = intercarrier_features = None
    for _window_timestamp_s, features, image, intercarrier_features in iter_windows(
        resampled, entry.calibration.subcarriers, entry.gain_lock,
        window=config.window, hop=config.hop, intercarrier=entry.should_use_intercarrier,
        image_subcarriers=entry.calibration.image_subcarriers, intercarrier_baseline=entry.intercarrier_baseline,
    ):
        window_probas.append(
            entry.session.predict_window_proba(entry.pick_input(features, image, intercarrier_features))
        )
    if not window_probas:
        return None, None
    return np.mean(window_probas, axis=0), copy_window(features, image, intercarrier_features)


def load_weapon_links(calibration_root, model_root):
    """Discover weapon heads -> {(tx_tag|None, rx_node): entry}. Two layouts, auto-detected per node:
      * PER-LINK: model_weapon/node<id>/link<link_tag>/model.joblib -> one entry per directed
        (tx->rx) view, keyed (link_tag, node_id). The weapon signal is per-direction; this lets each
        NLOS-scatter link vote on its own merit instead of being pooled (and sign-flipped) per node.
      * PER-NODE (fallback): model_weapon/node<id>/model.joblib -> a single (None, node_id) entry
        that serves ANY tx into that node. Used when a node wasn't trained --per-link.
    Calibration is per RX node (shared across that node's link heads). Each entry carries a static
    LinkVoter `weight` from its OWN LOGO accuracy (accuracy_weights; chance->0 so a weak direction
    drops out without being explicitly removed) and the serving plan matching how it was trained."""
    entries = {}
    accuracy_by_key = {}
    for node_id, model_dir, calibration_dir in discover_node_dirs(calibration_root, model_root):
        link_models = sorted(glob.glob(os.path.join(model_dir, "link*", "model.joblib")))
        if link_models:
            found = [
                (os.path.basename(os.path.dirname(path))[len("link"):], path) for path in link_models
            ]
        elif os.path.exists(os.path.join(model_dir, "model.joblib")):
            found = [(None, os.path.join(model_dir, "model.joblib"))]
        else:
            continue
        result, gain_lock = load_calibration(calibration_dir)
        for link_tag, model_path in found:
            session = open_session("weapon", model_path)
            should_apply_lock, intercarrier, pick = plan_input("weapon", session.head)
            classes = [int(class_id) for class_id in session.head.classes_]
            # rebuilt from the node's calibration, so serving subtracts the baseline training did
            intercarrier_baseline = (result.baseline_mag
                                     if getattr(session.head.config, "subtract_ic_baseline", False) else None)
            key = (link_tag, node_id)
            entries[key] = NodeServer(
                calibration=result, gain_lock=gain_lock if should_apply_lock else None,
                should_use_intercarrier=intercarrier, pick_input=pick, session=session,
                config=session.head.config, min_num_subcarriers=required_subcarrier_width(result),
                classes=classes, positive_class_index=classes.index(1) if 1 in classes else -1,
                intercarrier_baseline=intercarrier_baseline,
            )
            accuracy_by_key[key] = load_logo_accuracy(
                os.path.join(os.path.dirname(model_path), "metrics.json"))
    weights = accuracy_weights(
        {key: accuracy for key, accuracy in accuracy_by_key.items() if accuracy is not None})
    for key in entries:
        entries[key].weight = weights.get(key, 1.0)
    orders = {tuple(entry.classes) for entry in entries.values()}
    if len(orders) > 1:
        raise ValueError(f"weapon heads disagree on class ordering {orders}; retrain consistently")
    return entries


def make_link_tag(tx_short):
    """The ':'-free tag ('4f9c') of a tx mac-short ('4f:9c'), usable as a path segment."""
    return tx_short.replace(":", "")


def parse_link_tag(dataset_dir):
    """Tx tag of a weapon_ds dir named `<sess>_<cond>_link<tag>`, or None for a per-node dir."""
    base = os.path.basename(dataset_dir)
    return base.split("_link")[-1] if "_link" in base else None


def find_weapon_entry(entries, tx_short, rx_node_id):
    """The per-link head for a live link (tx_short '4f:9c' is keyed by tag '4f9c'), else the rx node's
    per-node head (None, rx_node_id), else None."""
    link_tag = make_link_tag(tx_short) if tx_short else None
    return entries.get((link_tag, rx_node_id)) or entries.get((None, rx_node_id))


def link_health(frames):
    """(delivered_hz, missing_fraction) from frame timestamps — the per-link Missing-Rate metric
    (measure DELIVERED rate, not configured). Median inter-arrival is the nominal period; a gap of
    ~k periods counts k-1 missing frames. O(n). (0,0) if too few frames."""
    if len(frames) < 3:
        return 0.0, 0.0
    timestamps = np.array([frame.timestamp for frame in frames], dtype=np.float64)
    inter_arrival = np.diff(timestamps)
    inter_arrival = inter_arrival[inter_arrival > 0]
    if inter_arrival.size == 0:
        return 0.0, 0.0
    span = timestamps[-1] - timestamps[0]
    delivered_hz = (len(frames) - 1) / span if span > 0 else 0.0
    median_inter_arrival = float(np.median(inter_arrival))
    if median_inter_arrival <= 0:
        return delivered_hz, 0.0
    missing = float(np.clip(np.round(inter_arrival / median_inter_arrival) - 1.0, 0.0, None).sum())
    expected = missing + inter_arrival.size
    return delivered_hz, (missing / expected if expected > 0 else 0.0)
