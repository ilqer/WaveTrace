"""Live ALL-PAIRS presence: every (tx->rx) link served through its RX node's own cal+head; vote.

Max-info path. Splits the batched-UDP stream into per-(tx,rx)-link streams (parse_batch_links), and
serves EACH link through the calibration + presence head of its RX node — gain is an RX property
(gain=LOCK vs gain=SKIP differ per board), so calibration/normalization is per-RX-node, while the
features stay per-link to keep all N(N-1) views. LinkVoter blends every live link weighted by the
head's decision margin, so a blocked/confused link down-weights itself and a node that drops just
removes its links from the vote (redundancy). Ctrl+C to stop.

    .venv/bin/python scripts/run_live_mesh.py
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


def _last_window_proba(frames, fs, result, gain_lock, cfg, intercarrier, pick, session):
    """Resample one node's frames to fs, window them, return the LAST window's class-proba or None
    (None when there aren't enough frames to fill a window)."""
    res = resample_uniform(frames, fs)
    if len(res) < cfg.window:
        return None
    last = None
    for _t, features, image, ic in iter_windows(
        res, result.subcarriers, gain_lock,
        window=cfg.window, hop=cfg.hop, intercarrier=intercarrier,
        image_subcarriers=result.image_subcarriers,
    ):
        last = session.predict_proba_window(pick(features, image, ic))
    return last


def _logo_accuracy(metrics_path):
    """A node head's HONEST (LOGO) balanced accuracy — session axis preferred, subject as fallback.
    None when the model was trained without a foldable group (no validated number exists)."""
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


def load_node_models(cal_root, model_root, mode="presence"):
    """Discover per-RX-node calibrations + heads -> {node_id: dict(...)}. Each node serves alone.
    Each node carries a static vote `weight` from its validated LOGO balanced accuracy (via
    accuracy_weights: chance->0, perfect->1), so a more reliable node counts more in the live vote."""
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
        session = mode_session(mode, model_path)
        apply_lock, intercarrier, pick = _serving_plan(mode, session.head)
        classes = list(session.head.classes_)
        nodes[nid] = dict(
            result=result, lock=gain_lock if apply_lock else None,
            intercarrier=intercarrier, pick=pick, session=session, cfg=session.head.config,
            min_width=_min_width(result), present_i=classes.index(1) if 1 in classes else -1,
        )
        accs[nid] = _logo_accuracy(os.path.join(model_dir, "metrics.json"))
    # Per-node reliability prior from the honest balanced accuracy; a node trained without a foldable
    # group (no LOGO) defaults to weight 1.0 — "no reliability discount until it's been validated".
    weights = accuracy_weights({nid: a for nid, a in accs.items() if a is not None})
    for nid in nodes:
        nodes[nid]["weight"] = weights.get(nid, 1.0)
    # The voter blends raw proba vectors across heads, so they MUST share class ordering; otherwise
    # class-1 of one head would be averaged with class-0 of another. Hard-fail on a mismatch rather
    # than silently fuse garbage (per-link p_present still uses each head's own present_i below).
    orders = {tuple(int(c) for c in m["session"].head.classes_) for m in nodes.values()}
    if len(orders) > 1:
        raise ValueError(f"per-node heads disagree on class ordering {orders}; retrain consistently")
    return nodes


def main():
    parser = argparse.ArgumentParser(description="Live ALL-PAIRS presence (per-link, per-RX-node cal+head).")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Per-node model root (default: <root>/model)")
    args = parser.parse_args()
    if args.cal is None:
        args.cal = f"{args.root}/cal"
    if args.model is None:
        args.model = f"{args.root}/model"

    TARGET_FS = 100.0      # uniform resample grid (the locked live cadence the collect scripts assume)
    CHUNK_S = 1.5          # fuse + print at this cadence
    LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
    BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing

    nodes = load_node_models(args.cal, args.model)
    if not nodes:
        print(f"[ERROR] No per-node models found under {args.model}/node*/model.joblib with a matching "
              f"{args.cal}/node*/. Run collect_baseline.py then collect_presence.py first.")
        return
    present_i = next(iter(nodes.values()))["present_i"]  # ordering validated equal in load_node_models

    # buffers keyed by (tx_short, rx_node); each link served via its RX node's cal+head
    buffers = collections.defaultdict(collections.deque)
    last_seen = {}
    link_ids = {}  # stable int id per link for LinkVoter

    sock = bind_udp(args.port, timeout=0.5)
    wsummary = "  ".join(f"N{nid}:w={nodes[nid]['weight']:.2f}" for nid in sorted(nodes))
    print(f"ALL-PAIRS presence on udp/{args.port} (fs={TARGET_FS:g}Hz, rx nodes={sorted(nodes)}; "
          f"vote weights {wsummary}). move in and out of the links. Ctrl+C to stop.\n")

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

            # trim each buffer to the last BUFFER_S seconds (by frame timestamp)
            for buf in buffers.values():
                if buf:
                    cutoff = buf[-1].timestamp - BUFFER_S
                    while buf and buf[0].timestamp < cutoff:
                        buf.popleft()

            # static per-node reliability prior x live margin (LinkVoter multiplies them): a node that
            # validated well AND is confident right now dominates; a weak or blocked link fades out.
            # Uniform fallback if no node carries a usable weight, so the vote stays defined.
            link_static = {lid: nodes[k[1]]["weight"] for k, lid in link_ids.items()}
            static = link_static if any(w > 0 for w in link_static.values()) else None
            voter = LinkVoter(static)
            breakdown = []
            for key in sorted(buffers):
                if now - last_seen.get(key, 0) > LINK_TIMEOUT_S or len(buffers[key]) < 2:
                    continue
                m = nodes[key[1]]  # serve each (tx,rx) link through ITS RX node's cal+head
                proba = _last_window_proba(list(buffers[key]), TARGET_FS, m["result"], m["lock"],
                                           m["cfg"], m["intercarrier"], m["pick"], m["session"])
                if proba is None:
                    continue
                pi = m["present_i"]  # index of class 1 in THIS rx-node's head (defensive; orderings equal)
                p_present = float(proba[pi]) if pi >= 0 else 0.0
                quality = abs(p_present - 0.5) * 2.0  # decision margin -> 0 (unsure) .. 1 (confident)
                voter.add(link_ids[key], proba, quality=quality)
                breakdown.append(f"{key[0]}->{key[1]}:{p_present:.2f}")

            if not breakdown:
                print("\r(no live links with a full window yet)            ", end="", flush=True)
                continue
            try:
                _cls, blended = voter.finalize()
            except ValueError:
                # all active links belong to node(s) validated at/below chance (weight 0) -> no vote
                print("\r(live links present, but all from chance-level nodes)   ", end="", flush=True)
                continue
            p_present = float(blended[present_i]) if present_i >= 0 else 0.0
            label = "PRESENT" if p_present >= 0.5 else "absent "
            bar = "#" * int(p_present * 20)
            print(f"{label}  P {p_present:0.2f}  {bar:<20}  [{len(breakdown)} links] "
                  + " ".join(breakdown))
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
