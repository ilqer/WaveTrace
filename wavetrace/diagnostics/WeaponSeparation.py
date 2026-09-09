"""Static σ²[p] PDF separation diagnostics — the weapon-detection go/no-go check shared by
`experiments/weapon_litmus.py`'s CLI and the web dashboard's `/api/weapon/litmus` route (one
definition instead of two).

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
    """(F,A,S) complex CSI -> (F,) per-frame σ²[p]. Sample variance of antenna-collapsed subcarrier magnitudes."""
    mag = np.abs(np.asarray(grid)).mean(axis=1)        # (F, S) antenna-collapsed magnitude
    return mag.var(axis=1, ddof=1)                     # (F,) inter-subcarrier variance per packet


def _node_of(path):
    """Extract the node id from a .../node<id>/... recording path, or None."""
    for part in path.split(os.sep):
        if part.startswith("node") and part[len("node"):].isdigit():
            return int(part[len("node"):])
    return None


def _link_of(path):
    """Extract the TX tag from a .../link_<tag>/... recording path, or None (the directed link's TX)."""
    for part in path.split(os.sep):
        if part.startswith("link_"):
            return part[len("link_"):]
    return None


def key_label(key):
    """Human label for a group key: '2' for a node, '64b8->2' for a tx->rx link."""
    return f"{key[1]}->{key[0]}" if isinstance(key, tuple) else str(key)


def gather_sigma2(root, node=None, per_link=False):
    """Walk <root>/weapon_rec for clear/weapon grids -> {key: {"clear": arr, "weapon": arr}}.
    If per_link=True, key is (rx_node, tx_tag) to score directions separately. Only NLOS-scatter carries weapon signals; pooling washes them out."""
    out = {}
    for cond in ("clear", "weapon"):
        for gpath in glob.glob(os.path.join(root, "weapon_rec", "**", cond, "**", "grid.npy"),
                               recursive=True):
            nid = _node_of(gpath)
            if nid is None or (node is not None and nid != node):
                continue
            key = (nid, _link_of(gpath)) if per_link else nid
            if per_link and key[1] is None:
                continue
            s2 = sigma2_per_frame(np.load(gpath))
            out.setdefault(key, {}).setdefault(cond, []).append(s2)
    return {key: {c: np.concatenate(v) for c, v in conds.items()}
            for key, conds in out.items()}


def separation(clear, weapon):
    """Separability of σ²[p]. AUC is direction-folded to >=0.5 (orientation flips metal shift sign)."""
    if clear.size == 0 or weapon.size == 0:
        return None
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(clear.size), np.ones(weapon.size)])
    x = np.concatenate([clear, weapon])
    auc = roc_auc_score(y, x)
    nc, nw = clear.size, weapon.size
    pooled_sd = np.sqrt(((nc - 1) * clear.var(ddof=1) + (nw - 1) * weapon.var(ddof=1)) / (nc + nw - 2))
    d = (weapon.mean() - clear.mean()) / pooled_sd if pooled_sd > 0 else 0.0
    return {
        "auc": max(auc, 1.0 - auc),         # separability, direction-folded
        "lower_when_armed": bool(weapon.mean() < clear.mean()),  # True = matches metal physics
        "cohens_d": d,
        "clear_med": float(np.median(clear)), "weapon_med": float(np.median(weapon)),
        "n_clear": int(nc), "n_weapon": int(nw),
    }


def json_hist(clear, weapon, bins=20):
    """JSON-serializable overlaid σ²[p] histogram for the web litmus card.
    Returns density-normalised heights on a shared edge grid. O(N log N)."""
    lo = float(min(clear.min(), weapon.min()))
    hi = float(max(clear.max(), weapon.max()))
    edges = np.linspace(lo, hi, bins + 1)
    hc, _ = np.histogram(clear, edges, density=True)
    hw, _ = np.histogram(weapon, edges, density=True)
    return {"edges": edges.tolist(), "clear": hc.tolist(), "weapon": hw.tolist()}


def verdict(auc):
    """Map direction-folded AUC to a go/no-go call."""
    if auc < 0.55:
        return "NO SEPARATION — radio/geometry problem; do NOT train (fix hardware first)"
    if auc < 0.65:
        return "WEAK — borderline; needs more controlled geometry before ML is worth it"
    return "PROMISING — signal present; ML is justified on this node"
