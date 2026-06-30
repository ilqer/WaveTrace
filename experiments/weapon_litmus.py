"""Static-σ²[p] PDF litmus — the pre-ML go/no-go test for weapon detection (diagnosis CAUSE 5A).

Before training ANY weapon model, answer one physics question on YOUR hardware: does the per-packet
inter-subcarrier variance σ²[p] separate the no-weapon (clear) and weapon conditions at all? Yousaf
Fig 17 / Hanif Fig 5 plot exactly this. If the two PDFs overlap, NO classifier can recover the
signal — the problem is upstream (radio/geometry), and you save weeks of model tuning by knowing it.

σ²[p] is computed the SAME way the live InterCarrierExtractor sees it (Frontend.py:76): per frame,
antenna-collapse the magnitude `|grid|.mean(antennas)`, then sample-variance (ddof=1) over ALL
subcarriers — not the presence subset (diagnosis 5B). Metal physics: weapon -> LOWER σ².

Reads the recordings collect_weapon.py already saves:
    <root>/weapon_rec/<session>/<clear|weapon>/node<id>/link_<tag>/grid.npy

    .venv/bin/python experiments/weapon_litmus.py                 # all nodes under data/
    .venv/bin/python experiments/weapon_litmus.py --root data/5g_ht80 --node 2
    .venv/bin/python experiments/weapon_litmus.py --plot          # also write PNG PDFs if matplotlib present

Per-node breakdown is deliberate: gain=LOCK vs gain=SKIP boards live on different amplitude scales
(diagnosis CAUSE 10), so a pooled PDF can blur a node that actually separates. Judge each node.
"""

import argparse
import glob
import os

import numpy as np


def sigma2_per_frame(grid):
    """(F,A,S) complex CSI -> (F,) per-frame σ²[p]: sample variance (ddof=1) of the antenna-collapsed
    subcarrier magnitudes. Mirrors the live IC extractor's input exactly. O(F·A·S)."""
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


def _key_nid(key):
    """RX node id for a group key (int node, or (node, tx_tag) link)."""
    return key[0] if isinstance(key, tuple) else key


def _key_label(key):
    """Human label for a group key: '2' for a node, '64b8->2' for a tx->rx link."""
    return f"{key[1]}->{key[0]}" if isinstance(key, tuple) else str(key)


def gather_sigma2(root, node=None, per_link=False):
    """Walk <root>/weapon_rec for clear/weapon grids -> {key: {"clear": arr, "weapon": arr}}.
    key is the RX node id, or (rx_node, tx_tag) per directed link when per_link=True — the latter
    scores each of the round-robin's directions separately (a node-as-RX sees several TX angles, and
    only the NLOS-scatter ones carry weapon signal; pooling them per node washes that out).
    Concatenates σ²[p] across every session of each condition. O(total frames)."""
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
    """Single-feature separability of σ²[p] between the two conditions. Returns None if either is
    empty. AUC is direction-folded to >=0.5 (orientation can flip the sign of the metal shift, so
    |0.5-AUC| is the honest 'how separable', not which way). Cohen's d uses the pooled SD."""
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


def _verdict(auc):
    """Map direction-folded AUC to a go/no-go call."""
    if auc < 0.55:
        return "NO SEPARATION — radio/geometry problem; do NOT train (fix hardware first)"
    if auc < 0.65:
        return "WEAK — borderline; needs more controlled geometry before ML is worth it"
    return "PROMISING — signal present; ML is justified on this node"


def ascii_hist(clear, weapon, bins=24, width=40):
    """Overlaid terminal histogram of the two σ²[p] PDFs on a shared bin grid (C=clear, W=weapon)."""
    lo = float(min(clear.min(), weapon.min()))
    hi = float(max(clear.max(), weapon.max()))
    edges = np.linspace(lo, hi, bins + 1)
    hc, _ = np.histogram(clear, edges, density=True)
    hw, _ = np.histogram(weapon, edges, density=True)
    peak = max(hc.max(), hw.max(), 1e-12)
    lines = []
    for i in range(bins):
        cbar = "C" * int(round(hc[i] / peak * width))
        wbar = "W" * int(round(hw[i] / peak * width))
        lines.append(f"{edges[i]:>10.3g} | {cbar}")
        lines.append(f"{'':>10} | {wbar}")
    return "\n".join(lines)


def _maybe_plot(data, out_path):
    """Write overlaid PDF histograms per node to a PNG if matplotlib is available; else warn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping PNG (pip install matplotlib)")
        return
    keys = sorted(data, key=_key_label)
    fig, axes = plt.subplots(len(keys), 1, figsize=(7, 3 * len(keys)), squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        c = data[key].get("clear", np.array([]))
        w = data[key].get("weapon", np.array([]))
        if c.size:
            ax.hist(c, bins=40, density=True, alpha=0.5, label="clear")
        if w.size:
            ax.hist(w, bins=40, density=True, alpha=0.5, label="weapon")
        ax.set_title(f"{_key_label(key)} — σ²[p] PDF")
        ax.set_xlabel("σ²[p]"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Static σ²[p] PDF litmus — weapon go/no-go before ML.")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root holding weapon_rec/ (default: data)")
    parser.add_argument("--node", type=int, default=None, help="Only this node (default: all)")
    parser.add_argument("--per-link", action="store_true", dest="per_link",
                        help="Score each directed tx->rx link separately (the 30 round-robin "
                             "directions), not pooled per RX node — find which directions separate")
    parser.add_argument("--no-hist", action="store_true", help="Skip the ASCII histograms")
    parser.add_argument("--plot", action="store_true", help="Also write PNG PDFs (needs matplotlib)")
    args = parser.parse_args()

    data = gather_sigma2(args.root, args.node, per_link=args.per_link)
    if not data:
        print(f"[ERROR] No weapon recordings under {args.root}/weapon_rec/*/<clear|weapon>/node*/.\n"
              f"        Run collect_weapon.py first (it saves the grids this tool reads).")
        return

    unit = "tx->rx link" if args.per_link else "node"
    # sort by separability when per-link (best directions first), else by node id
    print(f"Static σ²[p] litmus over {args.root}/weapon_rec  (metal physics: weapon -> LOWER σ²)\n")
    print(f"{unit:>8}  {'AUC':>6}  {'dir':>4}  {'cohen_d':>8}  {'clear~':>10}  {'weapon~':>10}  "
          f"{'n(c/w)':>13}  verdict")

    def _sortkey(key):
        s = separation(data[key].get("clear", np.array([])), data[key].get("weapon", np.array([])))
        return (-s["auc"], _key_label(key)) if (args.per_link and s) else (0.0, str(key))

    pooled = {"clear": [], "weapon": []}
    for key in sorted(data, key=_sortkey):
        c = data[key].get("clear", np.array([]))
        w = data[key].get("weapon", np.array([]))
        pooled["clear"].append(c); pooled["weapon"].append(w)
        label = _key_label(key)
        s = separation(c, w)
        if s is None:
            print(f"{label:>8}  {'-':>6}  {'-':>4}  {'-':>8}  {'-':>10}  {'-':>10}  "
                  f"{c.size}/{w.size:>6}  (need BOTH clear and weapon captures)")
            continue
        direction = "ok" if s["lower_when_armed"] else "INV"  # INV = armed σ² higher (anti-physics)
        print(f"{label:>8}  {s['auc']:>6.3f}  {direction:>4}  {s['cohens_d']:>8.2f}  "
              f"{s['clear_med']:>10.3g}  {s['weapon_med']:>10.3g}  "
              f"{s['n_clear']}/{s['n_weapon']:<7}  {_verdict(s['auc'])}")

    pc = np.concatenate(pooled["clear"]) if any(a.size for a in pooled["clear"]) else np.array([])
    pw = np.concatenate(pooled["weapon"]) if any(a.size for a in pooled["weapon"]) else np.array([])
    ps = separation(pc, pw)
    if ps is not None:
        print(f"\nPOOLED (all {unit}s — blurred by per-board gain scale, read with care): "
              f"AUC={ps['auc']:.3f}  {_verdict(ps['auc'])}")

    if not args.no_hist:
        for key in sorted(data, key=_sortkey):
            c = data[key].get("clear", np.array([]))
            w = data[key].get("weapon", np.array([]))
            if c.size and w.size:
                print(f"\n--- {unit} {_key_label(key)} σ²[p] PDF (C=clear  W=weapon) ---")
                print(ascii_hist(c, w))

    if args.plot:
        _maybe_plot(data, os.path.join(args.root, "weapon_litmus.png"))


if __name__ == "__main__":
    main()
