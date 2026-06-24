"""Independent live PEOPLE-COUNT — every (tx->rx) link served through its RX node's own cal + count
head, fused into one people-count. Standalone from run_live_mesh (imports only library code), reads the
count models from data/model_count.

Each link's RX-node head emits a probability over its count classes; per-node heads may have learned
different class subsets, so each proba is expanded into the GLOBAL class space (union over nodes) before
fusion. LinkVoter blends them weighted by static reliability (per-node LOGO accuracy, chance=1/K) x live
decision margin; the blended vector's argmax is the reported count, plus an expected-value estimate.

    .venv/bin/python run_count.py --max-count 3
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
from wavetrace.recognition.Link import LinkVoter
from wavetrace.Cli import _serving_plan
from collect_count import count_name  # shared label formatting (count module is internally DRY)


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


def _expand_proba(proba, col_map, k):
    """Place a node head's per-class proba into the GLOBAL k-class vector via col_map (head col -> global
    col). Classes the head never saw stay 0. Sum is preserved (each head's proba already sums to 1)."""
    g = np.zeros(k, dtype=np.float64)
    for j, col in enumerate(col_map):
        g[col] = proba[j]
    return g


def _last_window_proba(frames, fs, result, gain_lock, cfg, intercarrier, pick, session):
    """Resample one link's frames to fs, window them, return the LAST window's class-proba or None."""
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


def load_count_nodes(cal_root, model_root):
    """Discover per-RX-node calibrations + count heads -> ({node_id: dict(...)}, global_classes).

    Builds the global class space (union over nodes) and, per node: a col_map (head class -> global col)
    and a static weight from LOGO accuracy with chance = 1/K (so a multi-class head isn't zeroed out)."""
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
        session = mode_session("presence", model_path)  # count head is a multi-class PresenceHead
        apply_lock, intercarrier, pick = _serving_plan("presence", session.head)
        nodes[nid] = dict(
            result=result, lock=gain_lock if apply_lock else None,
            intercarrier=intercarrier, pick=pick, session=session, cfg=session.head.config,
            min_width=_min_width(result), classes=[int(c) for c in session.head.classes_],
        )
        accs[nid] = _logo_acc(os.path.join(model_dir, "metrics.json"))

    classes = sorted(set().union(*[set(m["classes"]) for m in nodes.values()])) if nodes else []
    k = len(classes)
    col_of = {c: i for i, c in enumerate(classes)}
    chance = 1.0 / k if k else 0.5
    for nid, m in nodes.items():
        m["col_map"] = [col_of[c] for c in m["classes"]]
        a = accs[nid]
        # static reliability prior; chance-aware so a decent K-class head isn't zeroed; None -> 1.0.
        m["weight"] = max(a - chance, 0.0) / max(1.0 - chance, 1e-9) if a is not None else 1.0
    return nodes, classes


def main():
    parser = argparse.ArgumentParser(description="Live ALL-PAIRS people-count (per-link, per-RX-node cal+head).")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Count model root (default: <root>/model_count)")
    parser.add_argument("--max-count", type=int, default=3, help="Top count level for 'N+' formatting (default: 3)")
    args = parser.parse_args()
    if args.cal is None:
        args.cal = f"{args.root}/cal"
    if args.model is None:
        args.model = f"{args.root}/model_count"

    TARGET_FS = 100.0      # uniform resample grid; MUST match collect_count.TARGET_FS
    CHUNK_S = 1.5          # fuse + print at this cadence
    LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
    BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing

    nodes, classes = load_count_nodes(args.cal, args.model)
    if not nodes:
        print(f"[ERROR] No count models under {args.model}/node*/model.joblib with a matching "
              f"{args.cal}/node*/. Run collect_baseline.py then collect_count.py first.")
        return
    k = len(classes)
    labels = [count_name(c, args.max_count) for c in classes]
    cls_arr = np.asarray(classes, dtype=np.float64)

    buffers = collections.defaultdict(collections.deque)  # keyed by (tx_short, rx_node)
    last_seen = {}
    link_ids = {}

    sock = bind_udp(args.port, timeout=0.5)
    wsummary = "  ".join(f"N{nid}:w={nodes[nid]['weight']:.2f}" for nid in sorted(nodes))
    print(f"PEOPLE-COUNT on udp/{args.port} (fs={TARGET_FS:g}Hz, classes={labels}, rx nodes={sorted(nodes)}; "
          f"vote weights {wsummary}). vary the headcount. Ctrl+C to stop.\n")

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
                proba = _last_window_proba(list(buffers[key]), TARGET_FS, m["result"], m["lock"],
                                           m["cfg"], m["intercarrier"], m["pick"], m["session"])
                if proba is None:
                    continue
                g = _expand_proba(proba, m["col_map"], k)
                top = np.sort(proba)[::-1]
                quality = float(top[0] - top[1]) if proba.size > 1 else float(top[0])  # decision margin
                voter.add(link_ids[key], g, quality=quality)
                breakdown.append(f"{key[0]}->{key[1]}:{classes[int(np.argmax(g))]}")

            if not breakdown:
                print("\r(no live links with a full window yet)            ", end="", flush=True)
                continue
            try:
                _cls, blended = voter.finalize()
            except ValueError:
                print("\r(live links present, but all from chance-level nodes)   ", end="", flush=True)
                continue
            blended = np.asarray(blended, dtype=np.float64)
            count = classes[int(np.argmax(blended))]
            expected = float((cls_arr * blended).sum())  # soft estimate (handles 'N+' as N)
            print(f"PEOPLE {count_name(count, args.max_count):>3}  (~{expected:0.1f})  "
                  f"[{len(breakdown)} links] " + " ".join(breakdown))
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
