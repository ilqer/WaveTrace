"""Live multi-link weapon-detection serving: discover the per-link/per-node weapon heads a training
run wrote to disk, and the per-window/per-link math shared by every weapon-serving front end
(`scripts/run_weapon.py`'s standalone all-pairs loop and the web dashboard's mesh path)."""

import glob
import json
import os

import numpy as np

from wavetrace.Calibration import loadCalibration
from wavetrace.Frontend import iterWindows
from wavetrace.Source import resampleUniform
from wavetrace.recognition.Infer import modeSession, planInferenceInput
from wavetrace.recognition.Link import accuracyWeights


def _min_width(result):
    """Subcarrier width the calibration needs = highest index it references + 1."""
    subcarrier_indices = [int(i) for i in list(result.subcarriers) + list(result.image_subcarriers)]
    return 1 + max(subcarrier_indices)


def _logo_accuracy(metrics_path):
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


def dwellProbaDetailed(frames, sample_rate_hz, entry):
    """One link's serving math, shared by `run_weapon` and the web streamer. `entry` is a serving
    entry — the dict from `loadWeaponLinks` or a mesh node dict; both carry
    result/lock/cfg/intercarrier/pick/session and an optional ic_baseline.

    Resamples to sample_rate_hz, windows the dwell, and returns the TEMPORAL VOTE — the mean
    class-proba over every window in the buffer (BUFFER_S of history), not just the last
    (per-crossing aggregation lifted single-window 51% -> 93%). Soft mean (not hard majority) so it
    composes with the soft cross-link LinkVoter. Also returns the LAST window's (image, features,
    intercarrier_features) for the web spectrogram, and the window count. Returns
    (None, None, None, None, 0) if no full window fits. ic_baseline: subtracted from the IC path when
    this head was trained that way — MUST match training (carried on the entry)."""
    resampled = resampleUniform(frames, sample_rate_hz)
    config = entry["cfg"]
    if len(resampled) < config.window:
        return None, None, None, None, 0
    probas = []
    image = features = intercarrier_features = None
    for _window_timestamp_s, features, image, intercarrier_features in iterWindows(
        resampled, entry["result"].subcarriers, entry["lock"],
        window=config.window, hop=config.hop, intercarrier=entry["intercarrier"],
        image_subcarriers=entry["result"].image_subcarriers, ic_baseline=entry.get("ic_baseline"),
    ):
        probas.append(
            entry["session"].predictProbaWindow(entry["pick"](features, image, intercarrier_features))
        )
    if not probas:
        return None, None, None, None, 0
    return np.mean(probas, axis=0), image, features, intercarrier_features, len(probas)  # temporal vote


def loadWeaponLinks(cal_root, model_root):
    """Discover weapon heads -> {(tx_tag|None, rx_node): entry}. Two layouts, auto-detected per node:
      * PER-LINK: model_weapon/node<id>/link<link_tag>/model.joblib -> one entry per directed
        (tx->rx) view, keyed (link_tag, node_id). The weapon signal is per-direction; this lets each
        NLOS-scatter link vote on its own merit instead of being pooled (and sign-flipped) per node.
      * PER-NODE (fallback): model_weapon/node<id>/model.joblib -> a single (None, node_id) entry
        that serves ANY tx into that node. Used when a node wasn't trained --per-link.
    Calibration is per RX node (shared across that node's link heads). Each entry carries a static
    LinkVoter `weight` from its OWN LOGO accuracy (accuracyWeights; chance->0 so a weak direction
    drops out without being explicitly removed) and the serving plan matching how it was trained."""
    entries = {}
    accs = {}
    for model_dir in sorted(glob.glob(os.path.join(model_root, "node*"))):
        base = os.path.basename(model_dir)
        if not base[len("node"):].isdigit():
            continue
        node_id = int(base[len("node"):])
        cal_dir = os.path.join(cal_root, base)
        if not os.path.isdir(cal_dir):
            continue
        link_models = sorted(glob.glob(os.path.join(model_dir, "link*", "model.joblib")))
        if link_models:  # per-link: tag from the link<tag> dir name
            found = [
                (os.path.basename(os.path.dirname(path))[len("link"):], path) for path in link_models
            ]
        elif os.path.exists(os.path.join(model_dir, "model.joblib")):
            found = [(None, os.path.join(model_dir, "model.joblib"))]  # per-node fallback
        else:
            continue
        result, gain_lock = loadCalibration(cal_dir)
        for link_tag, model_path in found:
            session = modeSession("weapon", model_path)
            apply_lock, intercarrier, pick = planInferenceInput("weapon", session.head)
            classes = list(session.head.classes_)
            # rebuilt from the node's calibration so train/serve subtract the same baseline (Item 10).
            ic_baseline = (result.baseline_mag
                           if getattr(session.head.config, "subtract_ic_baseline", False) else None)
            key = (link_tag, node_id)
            entries[key] = dict(
                result=result, lock=gain_lock if apply_lock else None,
                intercarrier=intercarrier, pick=pick, session=session, cfg=session.head.config,
                min_width=_min_width(result), weapon_i=classes.index(1) if 1 in classes else -1,
                ic_baseline=ic_baseline,
            )
            accs[key] = _logo_accuracy(os.path.join(os.path.dirname(model_path), "metrics.json"))
    weights = accuracyWeights({key: accuracy for key, accuracy in accs.items() if accuracy is not None})
    for key in entries:
        entries[key]["weight"] = weights.get(key, 1.0)
    orders = {tuple(int(class_id) for class_id in e["session"].head.classes_) for e in entries.values()}
    if len(orders) > 1:
        raise ValueError(f"weapon heads disagree on class ordering {orders}; retrain consistently")
    return entries


def linkHealth(frames):
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
    hz = (len(frames) - 1) / span if span > 0 else 0.0
    median_inter_arrival = float(np.median(inter_arrival))
    if median_inter_arrival <= 0:
        return hz, 0.0
    missing = float(np.clip(np.round(inter_arrival / median_inter_arrival) - 1.0, 0.0, None).sum())
    expected = missing + inter_arrival.size
    return hz, (missing / expected if expected > 0 else 0.0)
