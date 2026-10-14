"""Live people-count fusion: every (tx->rx) link is served through its rx node's cal and count head.

Heads may have learned different class sets, so each one's probabilities are expanded into a global
class space before LinkVoter blends them, weighted by LOGO accuracy times live decision margin.
"""

import argparse
import collections
import glob
import os
import socket
import time

import numpy as np

from wavetrace.adapters.recognition.session import open_session
from wavetrace.adapters.source import parse_batch_by_link, bind_udp_socket
from wavetrace.adapters.store.calibration import load_calibration
from wavetrace.domain.recognition.input_plan import plan_input
from wavetrace.recognition import count_name
from wavetrace.recognition.link import LinkVoter
from wavetrace.recognition.serving import last_window_proba, load_logo_accuracy, required_subcarrier_width


CHUNK_S = 1.5          # fuse + print at this cadence
LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing


def _expand_proba(proba, col_map, k):
    """Scatter a head's proba into the global k-class vector; classes it never saw stay 0."""
    expanded = np.zeros(k, dtype=np.float64)
    for j, column in enumerate(col_map):
        expanded[column] = proba[j]
    return expanded


def load_count_nodes(calibration_root, model_root):
    """Per rx-node calibration and count head, plus the global class space, each head's column map
    into it, and a static vote weight rescaled from LOGO accuracy against 1/k chance."""
    nodes = {}
    logo_accuracy_by_node = {}
    for model_dir in sorted(glob.glob(os.path.join(model_root, "node*"))):
        base = os.path.basename(model_dir)
        if not base[len("node"):].isdigit():
            continue
        node_id = int(base[len("node"):])
        calibration_dir = os.path.join(calibration_root, base)
        model_path = os.path.join(model_dir, "model.joblib")
        if not (os.path.isdir(calibration_dir) and os.path.exists(model_path)):
            continue
        result, gain_lock = load_calibration(calibration_dir)
        session = open_session("presence", model_path)  # count head is a multi-class PresenceHead
        should_apply_lock, intercarrier, pick = plan_input("presence", session.head)
        nodes[node_id] = dict(
            result=result, lock=gain_lock if should_apply_lock else None,
            intercarrier=intercarrier, pick=pick, session=session, cfg=session.head.config,
            min_width=required_subcarrier_width(result),
            classes=[int(class_id) for class_id in session.head.classes_],
        )
        logo_accuracy_by_node[node_id] = load_logo_accuracy(os.path.join(model_dir, "metrics.json"))

    classes = sorted(set().union(*[set(node_model["classes"])
                                   for node_model in nodes.values()])) if nodes else []
    class_count = len(classes)
    column_by_class = {class_id: i for i, class_id in enumerate(classes)}
    chance = 1.0 / class_count if class_count else 0.5
    for node_id, node_model in nodes.items():
        node_model["col_map"] = [column_by_class[class_id] for class_id in node_model["classes"]]
        logo_accuracy = logo_accuracy_by_node[node_id]
        # rescale accuracy so chance maps to 0 and perfect to 1; no LOGO number means 1.0
        node_model["weight"] = (max(logo_accuracy - chance, 0.0) / max(1.0 - chance, 1e-9)
                                if logo_accuracy is not None else 1.0)
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

    nodes, classes = load_count_nodes(args.cal, args.model)
    if not nodes:
        print(f"[ERROR] no count models under {args.model}/node*/model.joblib with a matching "
              f"{args.cal}/node*/. Run collect_baseline.py then collect_count.py first.")
        return
    class_count = len(classes)
    labels = [count_name(class_id, args.max_count) for class_id in classes]
    class_values = np.asarray(classes, dtype=np.float64)
    # banner only: every head is trained at the same rate today (not enforced)
    sample_rate_hz = next(iter(nodes.values()))["session"].head.contract.target_sample_rate_hz

    buffers = collections.defaultdict(collections.deque)  # keyed by (tx_short, rx_node)
    last_seen_time_s = {}
    link_ids = {}

    sock = bind_udp_socket(args.port, timeout_s=0.5)
    weight_summary = "  ".join(f"N{node_id}:w={nodes[node_id]['weight']:.2f}" for node_id in sorted(nodes))
    print(f"PEOPLE-COUNT on udp/{args.port} (fs={sample_rate_hz:g}Hz, "
          f"classes={labels}, rx nodes={sorted(nodes)}; "
          f"vote weights {weight_summary}). vary the headcount. Ctrl+C to stop.\n")

    next_fuse_time_s = time.time() + CHUNK_S
    try:
        while True:
            now = time.time()
            try:
                payload, _ = sock.recvfrom(65535)
                for key, frames in parse_batch_by_link(payload).items():
                    # key = (tx_short, rx_node); cal+head belong to the RX node
                    node_model = nodes.get(key[1])
                    if node_model is not None and frames[0].num_subcarriers >= node_model["min_width"]:
                        buffers[key].extend(frames)
                        last_seen_time_s[key] = now
                        link_ids.setdefault(key, len(link_ids))
            except socket.timeout:
                pass

            if now < next_fuse_time_s:
                continue
            next_fuse_time_s = now + CHUNK_S

            for link_frames in buffers.values():
                if link_frames:
                    cutoff = link_frames[-1].timestamp - BUFFER_S
                    while link_frames and link_frames[0].timestamp < cutoff:
                        link_frames.popleft()

            # LinkVoter multiplies static reliability by live margin; uniform when nothing is weighted
            static_link_weights = {link_id: nodes[key[1]]["weight"] for key, link_id in link_ids.items()}
            static = (static_link_weights
                      if any(weight > 0 for weight in static_link_weights.values()) else None)
            voter = LinkVoter(static)
            breakdown = []
            for key in sorted(buffers):
                if now - last_seen_time_s.get(key, 0) > LINK_TIMEOUT_S or len(buffers[key]) < 2:
                    continue
                node_model = nodes[key[1]]
                proba = last_window_proba(
                    list(buffers[key]), node_model["session"].head.contract.target_sample_rate_hz,
                    node_model["result"], node_model["lock"], node_model["cfg"], node_model["intercarrier"],
                    node_model["pick"], node_model["session"])
                if proba is None:
                    continue
                expanded_proba = _expand_proba(proba, node_model["col_map"], class_count)
                top = np.sort(proba)[::-1]
                quality = float(top[0] - top[1]) if proba.size > 1 else float(top[0])  # decision margin
                voter.add(link_ids[key], expanded_proba, quality=quality)
                breakdown.append(f"{key[0]}->{key[1]}:{classes[int(np.argmax(expanded_proba))]}")

            if not breakdown:
                print("\r(no live links with a full window yet)            ", end="", flush=True)
                continue
            try:
                _cls, blended = voter.finalize()
            except ValueError:
                print("\r(live links present, but all from chance-level nodes)   ", end="", flush=True)
                continue
            blended = np.asarray(blended, dtype=np.float64)
            expected = float((class_values * blended).sum())  # soft estimate; the 'N+' class counts as N
            rounded = int(round(expected))
            label = count_name(min(rounded, max(classes)), args.max_count)
            print(f"PEOPLE ~{expected:0.1f}  ({label})  "
                  f"[{len(breakdown)} links] " + " ".join(breakdown))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
