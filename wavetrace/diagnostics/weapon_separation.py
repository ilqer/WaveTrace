"""Weapon-detection go/no-go check shared by `experiments/weapon_litmus.py` and the web
`/api/weapon/litmus` route.

Checks whether per-packet inter-subcarrier variance σ²[p] separates clear vs weapon (see Yousaf
Fig 17). If the PDFs overlap, the signal is lost at the radio/geometry level — fix hardware before
training ML. σ²[p] mirrors InterCarrierExtractor: per frame, antenna-collapse magnitude
`|grid|.mean(antennas)`, then sample-variance (ddof=1) over all subcarriers. Physics: metal weapons
lower σ².

Reads the recordings collect_weapon.py already saves:
    <root>/weapon_rec/<session>/<clear|weapon>/node<id>/link_<tag>/grid.npy

Per-node breakdown matters: gain=LOCK and gain=SKIP boards have different amplitude scales, so a
pooled PDF blurs node separation — evaluate per node (or per per-link, see gather_sigma2)."""

import glob
import os

import numpy as np


def sigma2_per_frame(grid):
    """(F,A,S) complex CSI -> (F,) per-frame σ²[p]."""
    magnitude = np.abs(np.asarray(grid)).mean(axis=1)   # (F, S) antenna-collapsed magnitude
    return magnitude.var(axis=1, ddof=1)                # (F,) inter-subcarrier variance per packet


def _node_of(path):
    """From a .../node<id>/... recording path, or None."""
    for part in path.split(os.sep):
        if part.startswith("node") and part[len("node"):].isdigit():
            return int(part[len("node"):])
    return None


def _link_of(path):
    """TX tag from a .../link_<tag>/... recording path, or None."""
    for part in path.split(os.sep):
        if part.startswith("link_"):
            return part[len("link_"):]
    return None


def key_label(key):
    """'2' for a node key, '64b8->2' for a (rx, tx) link key."""
    return f"{key[1]}->{key[0]}" if isinstance(key, tuple) else str(key)


def gather_sigma2(root, node_id=None, per_link=False):
    """{key: {"clear": arr, "weapon": arr}} from <root>/weapon_rec grids.
    per_link=True keys by (rx_node, tx_tag): only NLOS-scatter carries weapon signals; pooling washes
    them out."""
    frame_variances_by_key = {}
    for capture_condition in ("clear", "weapon"):
        for grid_path in glob.glob(
                os.path.join(root, "weapon_rec", "**", capture_condition, "**", "grid.npy"), recursive=True):
            grid_node_id = _node_of(grid_path)
            if grid_node_id is None or (node_id is not None and grid_node_id != node_id):
                continue
            key = (grid_node_id, _link_of(grid_path)) if per_link else grid_node_id
            if per_link and key[1] is None:
                continue
            frame_variance = sigma2_per_frame(np.load(grid_path))
            frame_variances_by_key.setdefault(key, {}).setdefault(capture_condition, []).append(
                frame_variance)
    return {key: {condition: np.concatenate(values) for condition, values in values_by_condition.items()}
            for key, values_by_condition in frame_variances_by_key.items()}


def separation(clear, weapon):
    """Separability of σ²[p]. AUC is direction-folded to >=0.5 (orientation flips metal shift sign)."""
    if clear.size == 0 or weapon.size == 0:
        return None
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(clear.size), np.ones(weapon.size)])
    sigma2_values = np.concatenate([clear, weapon])
    auc = roc_auc_score(y, sigma2_values)
    clear_count, weapon_count = clear.size, weapon.size
    pooled_std = np.sqrt(
        ((clear_count - 1) * clear.var(ddof=1) + (weapon_count - 1) * weapon.var(ddof=1))
        / (clear_count + weapon_count - 2)
    )
    cohens_d = (weapon.mean() - clear.mean()) / pooled_std if pooled_std > 0 else 0.0
    return {
        "auc": max(auc, 1.0 - auc),         # separability, direction-folded
        "lower_when_armed": bool(weapon.mean() < clear.mean()),  # True = matches metal physics
        "cohens_d": cohens_d,
        "clear_med": float(np.median(clear)), "weapon_med": float(np.median(weapon)),
        "n_clear": int(clear_count), "n_weapon": int(weapon_count),
    }


def json_histogram(clear, weapon, bins=20):
    """Density-normalised heights on a shared edge grid. O(N log N)."""
    value_min = float(min(clear.min(), weapon.min()))
    value_max = float(max(clear.max(), weapon.max()))
    edges = np.linspace(value_min, value_max, bins + 1)
    clear_histogram, _ = np.histogram(clear, edges, density=True)
    weapon_histogram, _ = np.histogram(weapon, edges, density=True)
    return {"edges": edges.tolist(), "clear": clear_histogram.tolist(), "weapon": weapon_histogram.tolist()}


def verdict(auc):
    """Expects direction-folded AUC (max(auc, 1 - auc), >= 0.5)."""
    if auc < 0.55:
        return "NO SEPARATION — radio/geometry problem; do NOT train (fix hardware first)"
    if auc < 0.65:
        return "WEAK — borderline; needs more controlled geometry before ML is worth it"
    return "PROMISING — signal present; ML is justified on this node"
