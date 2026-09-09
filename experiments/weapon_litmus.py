"""Static-σ²[p] PDF litmus — weapon detection go/no-go test.
Checks if per-packet inter-subcarrier variance σ²[p] separates clear vs weapon (see Yousaf Fig 17).
If PDFs overlap, signal is lost at radio/geometry level. Fix hardware before training ML.

σ²[p] computation mirrors InterCarrierExtractor: per frame, antenna-collapse magnitude `|grid|.mean(antennas)`,
then sample-variance (ddof=1) over all subcarriers.
Physics: metal weapons lower σ².

Reads the recordings collect_weapon.py already saves:
    <root>/weapon_rec/<session>/<clear|weapon>/node<id>/link_<tag>/grid.npy

    .venv/bin/python experiments/weapon_litmus.py                 # all nodes under data/
    .venv/bin/python experiments/weapon_litmus.py --root data/5g_ht80 --node 2
    .venv/bin/python experiments/weapon_litmus.py --plot          # also write PNG PDFs if matplotlib present

Per-node breakdown is required: gain=LOCK and gain=SKIP boards have different amplitude scales.
A pooled PDF blurs node separation. Evaluate per node.
"""

import argparse
import os

import numpy as np

from wavetrace.diagnostics import gather_sigma2, key_label, separation, verdict

# pre-existing dead code (no caller here or elsewhere) — left in place rather than deleted, per
# CLAUDE.md §3 ("don't fix unrelated dead code, mention it"); not part of this move.
def _key_nid(key):
    """RX node id for a group key (int node, or (node, tx_tag) link)."""
    return key[0] if isinstance(key, tuple) else key


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
    keys = sorted(data, key=key_label)
    fig, axes = plt.subplots(len(keys), 1, figsize=(7, 3 * len(keys)), squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        c = data[key].get("clear", np.array([]))
        w = data[key].get("weapon", np.array([]))
        if c.size:
            ax.hist(c, bins=40, density=True, alpha=0.5, label="clear")
        if w.size:
            ax.hist(w, bins=40, density=True, alpha=0.5, label="weapon")
        ax.set_title(f"{key_label(key)} — σ²[p] PDF")
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
        print(f"[ERROR] no weapon recordings under {args.root}/weapon_rec/*/<clear|weapon>/node*/.\n"
              f"        run collect_weapon.py first (it saves the grids this tool reads).")
        return

    unit = "tx->rx link" if args.per_link else "node"
    # per-link: sort by separability, best first; else by node id
    print(f"static σ²[p] litmus over {args.root}/weapon_rec  (metal physics: weapon -> lower σ²)\n")
    print(f"{unit:>8}  {'AUC':>6}  {'dir':>4}  {'cohen_d':>8}  {'clear~':>10}  {'weapon~':>10}  "
          f"{'n(c/w)':>13}  verdict")

    def _sortkey(key):
        s = separation(data[key].get("clear", np.array([])), data[key].get("weapon", np.array([])))
        return (-s["auc"], key_label(key)) if (args.per_link and s) else (0.0, str(key))

    pooled = {"clear": [], "weapon": []}
    for key in sorted(data, key=_sortkey):
        c = data[key].get("clear", np.array([]))
        w = data[key].get("weapon", np.array([]))
        pooled["clear"].append(c); pooled["weapon"].append(w)
        label = key_label(key)
        s = separation(c, w)
        if s is None:
            print(f"{label:>8}  {'-':>6}  {'-':>4}  {'-':>8}  {'-':>10}  {'-':>10}  "
                  f"{c.size}/{w.size:>6}  (need BOTH clear and weapon captures)")
            continue
        direction = "ok" if s["lower_when_armed"] else "INV"  # INV = armed σ² higher, wrong direction
        print(f"{label:>8}  {s['auc']:>6.3f}  {direction:>4}  {s['cohens_d']:>8.2f}  "
              f"{s['clear_med']:>10.3g}  {s['weapon_med']:>10.3g}  "
              f"{s['n_clear']}/{s['n_weapon']:<7}  {verdict(s['auc'])}")

    pc = np.concatenate(pooled["clear"]) if any(a.size for a in pooled["clear"]) else np.array([])
    pw = np.concatenate(pooled["weapon"]) if any(a.size for a in pooled["weapon"]) else np.array([])
    ps = separation(pc, pw)
    if ps is not None:
        print(f"\nPOOLED (all {unit}s — blurred by per-board gain scale, read with care): "
              f"AUC={ps['auc']:.3f}  {verdict(ps['auc'])}")

    if not args.no_hist:
        for key in sorted(data, key=_sortkey):
            c = data[key].get("clear", np.array([]))
            w = data[key].get("weapon", np.array([]))
            if c.size and w.size:
                print(f"\n--- {unit} {key_label(key)} σ²[p] PDF (C=clear  W=weapon) ---")
                print(ascii_hist(c, w))

    if args.plot:
        _maybe_plot(data, os.path.join(args.root, "weapon_litmus.png"))


if __name__ == "__main__":
    main()
