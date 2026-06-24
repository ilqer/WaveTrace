"""Independent live WEAPON detection — every (tx->rx) link served through its RX node's cal + weapon
head (inter-carrier features), fused into one armed/clear verdict. Standalone from run_live_mesh /
run_count (imports only library code), reads the weapon models from data/model_weapon.

Each link's RX-node WeaponHead emits P(weapon); LinkVoter blends them weighted by static reliability
(per-node LOGO accuracy via accuracy_weights) x live decision margin. A node validated at/below chance
gets weight 0 and drops out.

    .venv/bin/python run_weapon.py
"""

import argparse
import collections
import glob
import json
import os
import socket
import time

import numpy as np

from wavetrace.Source import parse_batch_links, resample_uniform, bind_udp
from wavetrace.Calibration import load_calibration
from wavetrace.Frontend import iter_windows
from wavetrace.recognition import mode_session
from wavetrace.recognition.Link import LinkVoter, accuracy_weights
from wavetrace.Cli import _serving_plan


def _min_width(result):
    """Subcarrier width the calibration needs = highest index it references + 1."""
    idx = [int(i) for i in list(result.subcarriers) + list(result.image_subcarriers)]
    return 1 + max(idx)


def _logo_acc(metrics_path):
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


def _dwell_proba(frames, fs, result, gain_lock, cfg, intercarrier, pick, session, ic_baseline=None):
    """Resample one link's frames to fs, window them, and return the TEMPORAL VOTE across the whole
    dwell — the mean class-proba over every window in the buffer (BUFFER_S of history), not just the
    last one (diagnosis CAUSE 5C: Zhou's per-crossing aggregation lifted single-window 51% -> 93%).
    Soft mean (not hard majority) so it composes with the soft cross-link LinkVoter. None if no full
    window fits. ic_baseline (Item 10/CAUSE 2B): the node's quiet-room baseline, subtracted from the
    IC path when this head was trained that way — MUST match training (set from the head config)."""
    res = resample_uniform(frames, fs)
    if len(res) < cfg.window:
        return None
    probas = []
    for _t, features, image, ic in iter_windows(
        res, result.subcarriers, gain_lock,
        window=cfg.window, hop=cfg.hop, intercarrier=intercarrier,
        image_subcarriers=result.image_subcarriers, ic_baseline=ic_baseline,
    ):
        probas.append(session.predict_proba_window(pick(features, image, ic)))
    if not probas:
        return None
    return np.mean(probas, axis=0)  # temporal (soft) majority vote over the dwell


def load_weapon_nodes(cal_root, model_root):
    """Discover per-RX-node calibrations + weapon heads -> {node_id: dict(...)}. Each node carries a
    static vote `weight` from its LOGO accuracy (binary chance 0.5) and the serving plan for its head's
    feature_mode (ic27/fusion/cnn), so the live feature path matches how it was trained."""
    nodes = {}
    accs = {}
    for model_dir in sorted(glob.glob(os.path.join(model_root, "node*"))):
        base = os.path.basename(model_dir)
        if not base[len("node"):].isdigit():
            continue
        nid = int(base[len("node"):])
        cal_dir = os.path.join(cal_root, base)
        model_path = os.path.join(model_dir, "model.joblib")
        if not (os.path.isdir(cal_dir) and os.path.exists(model_path)):
            continue
        result, gain_lock = load_calibration(cal_dir)
        session = mode_session("weapon", model_path)
        apply_lock, intercarrier, pick = _serving_plan("weapon", session.head)
        classes = list(session.head.classes_)
        # IC background subtraction is a property of how THIS head was trained (head config), rebuilt
        # from the same per-node calibration so train/serve subtract the identical baseline (Item 10).
        ic_baseline = (result.baseline_mag
                       if getattr(session.head.config, "subtract_ic_baseline", False) else None)
        nodes[nid] = dict(
            result=result, lock=gain_lock if apply_lock else None,
            intercarrier=intercarrier, pick=pick, session=session, cfg=session.head.config,
            min_width=_min_width(result), weapon_i=classes.index(1) if 1 in classes else -1,
            ic_baseline=ic_baseline,
        )
        accs[nid] = _logo_acc(os.path.join(model_dir, "metrics.json"))
    weights = accuracy_weights({nid: a for nid, a in accs.items() if a is not None})
    for nid in nodes:
        nodes[nid]["weight"] = weights.get(nid, 1.0)
    orders = {tuple(int(c) for c in m["session"].head.classes_) for m in nodes.values()}
    if len(orders) > 1:
        raise ValueError(f"per-node heads disagree on class ordering {orders}; retrain consistently")
    return nodes


def main():
    parser = argparse.ArgumentParser(description="Live ALL-PAIRS weapon detection (per-link, per-RX-node cal+head).")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Weapon model root (default: <root>/model_weapon)")
    args = parser.parse_args()
    if args.cal is None:
        args.cal = f"{args.root}/cal"
    if args.model is None:
        args.model = f"{args.root}/model_weapon"

    TARGET_FS = 100.0      # uniform resample grid; MUST match collect_weapon.TARGET_FS
    CHUNK_S = 1.5          # fuse + print at this cadence
    LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
    BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing

    nodes = load_weapon_nodes(args.cal, args.model)
    if not nodes:
        print(f"[ERROR] No weapon models under {args.model}/node*/model.joblib with a matching "
              f"{args.cal}/node*/. Run collect_baseline.py then collect_weapon.py first.")
        return
    weapon_i = next(iter(nodes.values()))["weapon_i"]  # ordering validated equal in load_weapon_nodes

    buffers = collections.defaultdict(collections.deque)  # keyed by (tx_short, rx_node)
    last_seen = {}
    link_ids = {}

    sock = bind_udp(args.port, timeout=0.5)
    wsummary = "  ".join(f"N{nid}:w={nodes[nid]['weight']:.2f}" for nid in sorted(nodes))
    print(f"WEAPON detection on udp/{args.port} (fs={TARGET_FS:g}Hz, rx nodes={sorted(nodes)}; "
          f"vote weights {wsummary}). Ctrl+C to stop.\n")

    next_fuse = time.time() + CHUNK_S
    try:
        while True:
            now = time.time()
            try:
                payload, _ = sock.recvfrom(65535)
                for key, frames in parse_batch_links(payload).items():
                    m = nodes.get(key[1])  # key = (tx_short, rx_node); cal+head belong to the RX node
                    if m is not None and frames[0].num_subcarriers >= m["min_width"]:
                        buffers[key].extend(frames)
                        last_seen[key] = now
                        link_ids.setdefault(key, len(link_ids))
            except socket.timeout:
                pass

            if now < next_fuse:
                continue
            next_fuse = now + CHUNK_S

            for buf in buffers.values():  # trim each buffer to the last BUFFER_S seconds
                if buf:
                    cutoff = buf[-1].timestamp - BUFFER_S
                    while buf and buf[0].timestamp < cutoff:
                        buf.popleft()

            # static per-node reliability x live margin (LinkVoter multiplies them); uniform fallback.
            link_static = {lid: nodes[key[1]]["weight"] for key, lid in link_ids.items()}
            static = link_static if any(w > 0 for w in link_static.values()) else None
            voter = LinkVoter(static)
            breakdown = []
            for key in sorted(buffers):
                if now - last_seen.get(key, 0) > LINK_TIMEOUT_S or len(buffers[key]) < 2:
                    continue
                m = nodes[key[1]]
                proba = _dwell_proba(list(buffers[key]), TARGET_FS, m["result"], m["lock"],
                                     m["cfg"], m["intercarrier"], m["pick"], m["session"],
                                     ic_baseline=m["ic_baseline"])
                if proba is None:
                    continue
                wi = m["weapon_i"]
                p_weapon = float(proba[wi]) if wi >= 0 else 0.0
                quality = abs(p_weapon - 0.5) * 2.0  # decision margin -> 0 (unsure) .. 1 (confident)
                voter.add(link_ids[key], proba, quality=quality)
                breakdown.append(f"{key[0]}->{key[1]}:{p_weapon:.2f}")

            if not breakdown:
                print("\r(no live links with a full window yet)            ", end="", flush=True)
                continue
            try:
                _cls, blended = voter.finalize()
            except ValueError:
                print("\r(live links present, but all from chance-level nodes)   ", end="", flush=True)
                continue
            p_weapon = float(blended[weapon_i]) if weapon_i >= 0 else 0.0
            label = "WEAPON" if p_weapon >= 0.5 else "clear "
            bar = "#" * int(p_weapon * 20)
            print(f"{label}  P {p_weapon:0.2f}  {bar:<20}  [{len(breakdown)} links] "
                  + " ".join(breakdown))
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
