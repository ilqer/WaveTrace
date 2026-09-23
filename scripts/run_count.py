"""Live people-count fusion: every (tx->rx) link is served through its rx node's cal and count head.

Heads may have learned different class sets, so each one's probabilities are expanded into a global
class space before LinkVoter blends them, weighted by LOGO accuracy times live decision margin.
"""

import argparse
import collections
import glob
import json
import os
import socket
import time

import numpy as np

from wavetrace.Source import parseBatchLinks, resampleUniform, bindUdp
from wavetrace.Calibration import loadCalibration
from wavetrace.Frontend import iterWindows
from wavetrace.recognition import countName, modeSession, planInferenceInput
from wavetrace.recognition.Link import LinkVoter


def _minWidth(result):
    """Calibration subcarrier width (highest index + 1)."""
    idx = [int(i) for i in list(result.subcarriers) + list(result.image_subcarriers)]
    return 1 + max(idx)


def _logoAcc(metrics_path):
    """LOGO accuracy of a node head; session axis preferred, subject as fallback, None if neither."""
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


def _expandProba(proba, col_map, k):
    """Expand node head proba into global k-class vector. Unseen classes stay 0."""
    g = np.zeros(k, dtype=np.float64)
    for j, col in enumerate(col_map):
        g[col] = proba[j]
    return g


def _lastWindowProba(frames, fs, result, gainLock, cfg, intercarrier, pick, session):
    """Last window's class probabilities, or None when there are too few frames to fill a window."""
    res = resampleUniform(frames, fs)
    if len(res) < cfg.window:
        return None
    last = None
    for _t, features, image, ic in iterWindows(
        res, result.subcarriers, gainLock,
        window=cfg.window, hop=cfg.hop, intercarrier=intercarrier,
        image_subcarriers=result.image_subcarriers,
    ):
        last = session.predictProbaWindow(pick(features, image, ic))
    return last


def loadCountNodes(cal_root, model_root):
    """Per rx-node calibration and count head, plus the global class space, each head's column map
    into it, and a static vote weight rescaled from LOGO accuracy against 1/k chance."""
    nodes = {}
    accs = {}
    for model_dir in sorted(glob.glob(os.path.join(model_root, "node*"))):
        base = os.path.basename(model_dir)
        if not base[len("node"):].isdigit():
            continue
        nid = int(base[len("node"):])
        calDir = os.path.join(cal_root, base)
        modelPath = os.path.join(model_dir, "model.joblib")
        if not (os.path.isdir(calDir) and os.path.exists(modelPath)):
            continue
        result, gainLock = loadCalibration(calDir)
        session = modeSession("presence", modelPath)  # count head is a multi-class PresenceHead
        applyLock, intercarrier, pick = planInferenceInput("presence", session.head)
        nodes[nid] = dict(
            result=result, lock=gainLock if applyLock else None,
            intercarrier=intercarrier, pick=pick, session=session, cfg=session.head.config,
            min_width=_minWidth(result), classes=[int(c) for c in session.head.classes_],
        )
        accs[nid] = _logoAcc(os.path.join(model_dir, "metrics.json"))

    classes = sorted(set().union(*[set(m["classes"]) for m in nodes.values()])) if nodes else []
    k = len(classes)
    colOf = {c: i for i, c in enumerate(classes)}
    chance = 1.0 / k if k else 0.5
    for nid, m in nodes.items():
        m["col_map"] = [colOf[c] for c in m["classes"]]
        a = accs[nid]
        # rescale accuracy so chance maps to 0 and perfect to 1; no LOGO number means 1.0
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

    CHUNK_S = 1.5          # fuse + print at this cadence
    LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
    BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing

    nodes, classes = loadCountNodes(args.cal, args.model)
    if not nodes:
        print(f"[ERROR] no count models under {args.model}/node*/model.joblib with a matching "
              f"{args.cal}/node*/. Run collect_baseline.py then collect_count.py first.")
        return
    k = len(classes)
    labels = [countName(c, args.max_count) for c in classes]
    clsArr = np.asarray(classes, dtype=np.float64)
    # read off a head's own contract rather than re-declared here; every head is trained at the same
    # rate today, so any one of them stands in for the banner
    sample_rate_hz = next(iter(nodes.values()))["session"].head.contract.target_sample_rate_hz

    buffers = collections.defaultdict(collections.deque)  # keyed by (tx_short, rx_node)
    lastSeen = {}
    linkIds = {}

    sock = bindUdp(args.port, timeout=0.5)
    wsummary = "  ".join(f"N{nid}:w={nodes[nid]['weight']:.2f}" for nid in sorted(nodes))
    print(f"PEOPLE-COUNT on udp/{args.port} (fs={sample_rate_hz:g}Hz, "
          f"classes={labels}, rx nodes={sorted(nodes)}; "
          f"vote weights {wsummary}). vary the headcount. Ctrl+C to stop.\n")

    nextFuse = time.time() + CHUNK_S
    try:
        while True:
            now = time.time()
            try:
                payload, _ = sock.recvfrom(65535)
                for key, frames in parseBatchLinks(payload).items():
                    m = nodes.get(key[1])  # key = (tx_short, rx_node); cal+head belong to the RX node
                    if m is not None and frames[0].num_subcarriers >= m["min_width"]:
                        buffers[key].extend(frames)
                        lastSeen[key] = now
                        linkIds.setdefault(key, len(linkIds))
            except socket.timeout:
                pass

            if now < nextFuse:
                continue
            nextFuse = now + CHUNK_S

            for buf in buffers.values():
                if buf:
                    cutoff = buf[-1].timestamp - BUFFER_S
                    while buf and buf[0].timestamp < cutoff:
                        buf.popleft()

            # LinkVoter multiplies static reliability by live margin; uniform when nothing is weighted
            linkStatic = {lid: nodes[key[1]]["weight"] for key, lid in linkIds.items()}
            static = linkStatic if any(w > 0 for w in linkStatic.values()) else None
            voter = LinkVoter(static)
            breakdown = []
            for key in sorted(buffers):
                if now - lastSeen.get(key, 0) > LINK_TIMEOUT_S or len(buffers[key]) < 2:
                    continue
                m = nodes[key[1]]
                proba = _lastWindowProba(list(buffers[key]), m["session"].head.contract.target_sample_rate_hz,
                                          m["result"], m["lock"], m["cfg"], m["intercarrier"],
                                          m["pick"], m["session"])
                if proba is None:
                    continue
                g = _expandProba(proba, m["col_map"], k)
                top = np.sort(proba)[::-1]
                quality = float(top[0] - top[1]) if proba.size > 1 else float(top[0])  # decision margin
                voter.add(linkIds[key], g, quality=quality)
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
            expected = float((clsArr * blended).sum())  # soft estimate; the 'N+' class counts as N
            rounded = int(round(expected))
            label = countName(min(rounded, max(classes)), args.max_count)
            print(f"PEOPLE ~{expected:0.1f}  ({label})  "
                  f"[{len(breakdown)} links] " + " ".join(breakdown))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
