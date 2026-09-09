"""Live ALL-PAIRS presence: every (tx->rx) link served through its RX node's own cal+head; vote.

Max-info path. Splits the batched-UDP stream into per-(tx,rx)-link streams (parseBatchLinks), and
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

from wavetrace.Source import parseBatchLinks, resampleUniform, bindUdp
from wavetrace.Calibration import loadCalibration
from wavetrace.Frontend import iterWindows
from wavetrace.recognition import modeSession, planInferenceInput
from wavetrace.recognition.Link import LinkVoter, accuracyWeights


def _minWidth(result):
    """Subcarrier width the calibration needs = highest index it references + 1."""
    idx = [int(i) for i in list(result.subcarriers) + list(result.image_subcarriers)]
    return 1 + max(idx)


def _lastWindowProba(frames, fs, result, gainLock, cfg, intercarrier, pick, session):
    """Resample one node's frames to fs, window them, return the LAST window's class-proba or None
    (None when there aren't enough frames to fill a window)."""
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


def _logoAccuracy(metrics_path):
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


def loadNodeModels(cal_root, model_root, mode="presence"):
    """Discover per-RX-node calibrations + heads -> {node_id: dict(...)}. Each node serves alone.
    Each node carries a static vote `weight` from its validated LOGO balanced accuracy (via
    accuracyWeights: chance->0, perfect->1), so a more reliable node counts more in the live vote."""
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
        session = modeSession(mode, modelPath)
        applyLock, intercarrier, pick = planInferenceInput(mode, session.head)
        classes = list(session.head.classes_)
        nodes[nid] = dict(
            result=result, lock=gainLock if applyLock else None,
            intercarrier=intercarrier, pick=pick, session=session, cfg=session.head.config,
            min_width=_minWidth(result), present_i=classes.index(1) if 1 in classes else -1,
        )
        accs[nid] = _logoAccuracy(os.path.join(model_dir, "metrics.json"))
    # Reliability prior from LOGO balanced accuracy; a node with no foldable group defaults to weight 1.0.
    weights = accuracyWeights({nid: a for nid, a in accs.items() if a is not None})
    for nid in nodes:
        nodes[nid]["weight"] = weights.get(nid, 1.0)
    # Heads must share class ordering to blend raw proba vectors; hard-fail rather than silently fuse garbage.
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

    CHUNK_S = 1.5          # fuse + print at this cadence
    LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
    BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing

    nodes = loadNodeModels(args.cal, args.model)
    if not nodes:
        print(f"[ERROR] no per-node models found under {args.model}/node*/model.joblib with a matching "
              f"{args.cal}/node*/. Run collect_baseline.py then collect_presence.py first.")
        return
    presentI = next(iter(nodes.values()))["present_i"]  # ordering validated equal in loadNodeModels
    # the artifact's own resample rate, not a re-declared constant -- every node's head is trained
    # at the same rate today, so any node's contract stands in for the print banner.
    sample_rate_hz = next(iter(nodes.values()))["session"].head.contract.target_sample_rate_hz

    # buffers keyed by (tx_short, rx_node); each link served via its RX node's cal+head
    buffers = collections.defaultdict(collections.deque)
    lastSeen = {}
    linkIds = {}  # stable int id per link for LinkVoter

    sock = bindUdp(args.port, timeout=0.5)
    wsummary = "  ".join(f"N{nid}:w={nodes[nid]['weight']:.2f}" for nid in sorted(nodes))
    print(f"ALL-PAIRS presence on udp/{args.port} (fs={sample_rate_hz:g}Hz, "
          f"rx nodes={sorted(nodes)}; vote weights {wsummary}). "
          "move in and out of the links. Ctrl+C to stop.\n")

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

            # static reliability x live margin (LinkVoter multiplies them); uniform fallback keeps the vote defined.
            linkStatic = {lid: nodes[k[1]]["weight"] for k, lid in linkIds.items()}
            static = linkStatic if any(w > 0 for w in linkStatic.values()) else None
            voter = LinkVoter(static)
            breakdown = []
            for key in sorted(buffers):
                if now - lastSeen.get(key, 0) > LINK_TIMEOUT_S or len(buffers[key]) < 2:
                    continue
                m = nodes[key[1]]  # serve each (tx,rx) link through ITS RX node's cal+head
                proba = _lastWindowProba(list(buffers[key]), m["session"].head.contract.target_sample_rate_hz,
                                          m["result"], m["lock"], m["cfg"], m["intercarrier"],
                                          m["pick"], m["session"])
                if proba is None:
                    continue
                pi = m["present_i"]  # index of class 1 in THIS rx-node's head (defensive; orderings equal)
                pPresent = float(proba[pi]) if pi >= 0 else 0.0
                quality = abs(pPresent - 0.5) * 2.0  # decision margin -> 0 (unsure) .. 1 (confident)
                voter.add(linkIds[key], proba, quality=quality)
                breakdown.append(f"{key[0]}->{key[1]}:{pPresent:.2f}")

            if not breakdown:
                print("\r(no live links with a full window yet)            ", end="", flush=True)
                continue
            try:
                _cls, blended = voter.finalize()
            except ValueError:
                # all active links belong to node(s) validated at/below chance (weight 0) -> no vote
                print("\r(live links present, but all from chance-level nodes)   ", end="", flush=True)
                continue
            pPresent = float(blended[presentI]) if presentI >= 0 else 0.0
            label = "PRESENT" if pPresent >= 0.5 else "absent "
            bar = "#" * int(pPresent * 20)
            print(f"{label}  P {pPresent:0.2f}  {bar:<20}  [{len(breakdown)} links] "
                  + " ".join(breakdown))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
