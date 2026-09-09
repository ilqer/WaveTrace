"""Live multi-link weapon-detection serving: discover the per-link/per-node weapon heads a training
run wrote to disk, and the per-window/per-link math shared by every weapon-serving front end
(`scripts/run_weapon.py`'s standalone all-pairs loop and the web dashboard's mesh path) so it is
defined once instead of twice."""

import glob
import json
import os

import numpy as np

from wavetrace.Calibration import loadCalibration
from wavetrace.Frontend import iterWindows
from wavetrace.Source import resampleUniform
from wavetrace.recognition.Infer import modeSession, planInferenceInput
from wavetrace.recognition.Link import accuracyWeights


def _minWidth(result):
    """Subcarrier width the calibration needs = highest index it references + 1."""
    idx = [int(i) for i in list(result.subcarriers) + list(result.image_subcarriers)]
    return 1 + max(idx)


def _logoAcc(metrics_path):
    """A node head's honest (LOGO) accuracy — session axis preferred, subject fallback; None if absent."""
    try:
        with open(metrics_path) as f:
            logo = json.load(f).get("logo", {})
    except (OSError, ValueError):
        return None
    for axis in ("session", "subject"):
        acc = logo.get(axis, {}).get("accuracy")
        if acc is not None:
            return float(acc)
    return None


def dwellProbaDetailed(frames, fs, m):
    """SINGLE SOURCE OF TRUTH for one link's serving math, shared by run_weapon and the web streamer
    (don't reimplement the vote loop twice). `m` is a serving entry — the dict from loadWeaponLinks OR
    a mesh node dict; both carry result/lock/cfg/intercarrier/pick/session and an optional
    ic_baseline.

    Resamples to fs, windows the dwell, and returns the TEMPORAL VOTE — the mean class-proba over every
    window in the buffer (BUFFER_S of history), not just the last (per-crossing aggregation lifted
    single-window 51% -> 93%). Soft mean (not hard majority) so it composes with the soft cross-link
    LinkVoter. Also returns the LAST window's (image, features, ic) for the web spectrogram, and the
    window count. Returns (None, None, None, None, 0) if no full window fits. ic_baseline: subtracted
    from the IC path when this head was trained that way — MUST match training (carried on the entry)."""
    res = resampleUniform(frames, fs)
    cfg = m["cfg"]
    if len(res) < cfg.window:
        return None, None, None, None, 0
    probas = []
    image = features = ic = None
    for _t, features, image, ic in iterWindows(
        res, m["result"].subcarriers, m["lock"],
        window=cfg.window, hop=cfg.hop, intercarrier=m["intercarrier"],
        image_subcarriers=m["result"].image_subcarriers, ic_baseline=m.get("ic_baseline"),
    ):
        probas.append(m["session"].predictProbaWindow(m["pick"](features, image, ic)))
    if not probas:
        return None, None, None, None, 0
    return np.mean(probas, axis=0), image, features, ic, len(probas)  # temporal (soft) vote


def loadWeaponLinks(cal_root, model_root):
    """Discover weapon heads -> {(tx_tag|None, rx_node): entry}. Two layouts, auto-detected per node:
      * PER-LINK: model_weapon/node<id>/link<tag>/model.joblib -> one entry per directed (tx->rx)
        view, keyed (tag, nid). The weapon signal is per-direction; this lets each NLOS-scatter link
        vote on its own merit instead of being pooled (and sign-flipped) per node.
      * PER-NODE (fallback): model_weapon/node<id>/model.joblib -> a single (None, nid) entry that
        serves ANY tx into that node. Used when a node wasn't trained --per-link.
    Calibration is per RX node (shared across that node's link heads). Each entry carries a static
    LinkVoter `weight` from its OWN LOGO accuracy (accuracyWeights; chance->0 so a weak direction
    drops out without being explicitly removed) and the serving plan matching how it was trained."""
    entries = {}
    accs = {}
    for model_dir in sorted(glob.glob(os.path.join(model_root, "node*"))):
        base = os.path.basename(model_dir)
        if not base[len("node"):].isdigit():
            continue
        nid = int(base[len("node"):])
        calDir = os.path.join(cal_root, base)
        if not os.path.isdir(calDir):
            continue
        linkModels = sorted(glob.glob(os.path.join(model_dir, "link*", "model.joblib")))
        if linkModels:  # per-link: tag from the link<tag> dir name
            found = [(os.path.basename(os.path.dirname(p))[len("link"):], p) for p in linkModels]
        elif os.path.exists(os.path.join(model_dir, "model.joblib")):
            found = [(None, os.path.join(model_dir, "model.joblib"))]  # per-node fallback
        else:
            continue
        result, gainLock = loadCalibration(calDir)
        for tag, modelPath in found:
            session = modeSession("weapon", modelPath)
            applyLock, intercarrier, pick = planInferenceInput("weapon", session.head)
            classes = list(session.head.classes_)
            # rebuilt from the node's calibration so train/serve subtract the same baseline (Item 10).
            icBaseline = (result.baseline_mag
                           if getattr(session.head.config, "subtract_ic_baseline", False) else None)
            key = (tag, nid)
            entries[key] = dict(
                result=result, lock=gainLock if applyLock else None,
                intercarrier=intercarrier, pick=pick, session=session, cfg=session.head.config,
                min_width=_minWidth(result), weapon_i=classes.index(1) if 1 in classes else -1,
                ic_baseline=icBaseline,
            )
            accs[key] = _logoAcc(os.path.join(os.path.dirname(modelPath), "metrics.json"))
    weights = accuracyWeights({k: a for k, a in accs.items() if a is not None})
    for key in entries:
        entries[key]["weight"] = weights.get(key, 1.0)
    orders = {tuple(int(c) for c in e["session"].head.classes_) for e in entries.values()}
    if len(orders) > 1:
        raise ValueError(f"weapon heads disagree on class ordering {orders}; retrain consistently")
    return entries


def linkHealth(frames):
    """(delivered_hz, missing_fraction) from frame timestamps — the per-link Missing-Rate metric
    (measure DELIVERED rate, not configured). Median inter-arrival is the nominal period; a gap of
    ~k periods counts k-1 missing frames. O(n). (0,0) if too few frames."""
    if len(frames) < 3:
        return 0.0, 0.0
    ts = np.array([f.timestamp for f in frames], dtype=np.float64)
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0, 0.0
    span = ts[-1] - ts[0]
    hz = (len(frames) - 1) / span if span > 0 else 0.0
    med = float(np.median(dt))
    if med <= 0:
        return hz, 0.0
    missing = float(np.clip(np.round(dt / med) - 1.0, 0.0, None).sum())
    expected = missing + dt.size
    return hz, (missing / expected if expected > 0 else 0.0)
