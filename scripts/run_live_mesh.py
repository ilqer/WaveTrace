"""Live presence over every (tx->rx) link, each served through its rx node's own cal and head.

Gain is a property of the receiver (gain=LOCK on one board, gain=SKIP on another), so calibration is
per rx node, while the features stay per link to keep all N(N-1) views. LinkVoter blends every live
link weighted by its head's decision margin, so a blocked link down-weights itself and a node that
drops just removes its links from the vote.
"""

import argparse
import collections
import glob
import os
import socket
import time

from wavetrace.adapters.recognition.session import open_session
from wavetrace.adapters.source import parse_batch_by_link, bind_udp_socket
from wavetrace.adapters.store.calibration import load_calibration
from wavetrace.domain.recognition.input_plan import plan_input
from wavetrace.recognition.link import LinkVoter, accuracy_weights
from wavetrace.recognition.serving import (NodeServer, last_window_proba, load_logo_accuracy,
                                            required_subcarrier_width)


CHUNK_S = 1.5          # fuse + print at this cadence
LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing


def load_node_models(calibration_root, model_root, mode="presence"):
    """Per rx-node calibration and head, each carrying a static vote `weight` rescaled from its LOGO
    balanced accuracy so that chance maps to 0 and perfect to 1."""
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
        session = open_session(mode, model_path)
        should_apply_lock, intercarrier, pick = plan_input(mode, session.head)
        classes = list(session.head.classes_)
        nodes[node_id] = NodeServer(
            calibration=result, gain_lock=gain_lock if should_apply_lock else None,
            should_use_intercarrier=intercarrier, pick_input=pick, session=session,
            config=session.head.config, min_num_subcarriers=required_subcarrier_width(result),
            positive_class_index=classes.index(1) if 1 in classes else -1,
        )
        logo_accuracy_by_node[node_id] = load_logo_accuracy(os.path.join(model_dir, "metrics.json"))
    weights = accuracy_weights({node_id: logo_accuracy
                                for node_id, logo_accuracy in logo_accuracy_by_node.items()
                                if logo_accuracy is not None})
    for node_id in nodes:
        nodes[node_id].weight = weights.get(node_id, 1.0)
    # blending raw proba vectors is only valid when the heads share a class ordering
    orders = {tuple(int(class_id) for class_id in node.session.head.classes_)
              for node in nodes.values()}
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

    nodes = load_node_models(args.cal, args.model)
    if not nodes:
        print(f"[ERROR] no per-node models found under {args.model}/node*/model.joblib with a matching "
              f"{args.cal}/node*/. Run collect_baseline.py then collect_presence.py first.")
        return
    # ordering validated equal in load_node_models
    present_class_index = next(iter(nodes.values())).positive_class_index
    # banner only: every head is trained at the same rate today (not enforced)
    sample_rate_hz = next(iter(nodes.values())).session.head.contract.target_sample_rate_hz

    # keyed by (tx_short, rx_node)
    buffers = collections.defaultdict(collections.deque)
    last_seen_time_s = {}
    link_ids = {}

    sock = bind_udp_socket(args.port, timeout_s=0.5)
    weight_summary = "  ".join(f"N{node_id}:w={nodes[node_id].weight:.2f}" for node_id in sorted(nodes))
    print(f"ALL-PAIRS presence on udp/{args.port} (fs={sample_rate_hz:g}Hz, "
          f"rx nodes={sorted(nodes)}; vote weights {weight_summary}). "
          "move in and out of the links. Ctrl+C to stop.\n")

    next_fuse_time_s = time.time() + CHUNK_S
    try:
        while True:
            now = time.time()
            try:
                payload, _ = sock.recvfrom(65535)
                for key, frames in parse_batch_by_link(payload).items():
                    # key = (tx_short, rx_node); cal+head belong to the RX node
                    node = nodes.get(key[1])
                    if node is not None and frames[0].num_subcarriers >= node.min_num_subcarriers:
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
            static_link_weights = {link_id: nodes[link_key[1]].weight
                                   for link_key, link_id in link_ids.items()}
            static = (static_link_weights
                      if any(weight > 0 for weight in static_link_weights.values()) else None)
            voter = LinkVoter(static)
            breakdown = []
            for key in sorted(buffers):
                if now - last_seen_time_s.get(key, 0) > LINK_TIMEOUT_S or len(buffers[key]) < 2:
                    continue
                node = nodes[key[1]]
                proba = last_window_proba(
                    list(buffers[key]), node.session.head.contract.target_sample_rate_hz,
                    node.calibration, node.gain_lock, node.config, node.should_use_intercarrier,
                    node.pick_input, node.session)
                if proba is None:
                    continue
                present_index = node.positive_class_index
                present_proba = float(proba[present_index]) if present_index >= 0 else 0.0
                quality = abs(present_proba - 0.5) * 2.0  # decision margin -> 0 (unsure) .. 1 (confident)
                voter.add(link_ids[key], proba, quality=quality)
                breakdown.append(f"{key[0]}->{key[1]}:{present_proba:.2f}")

            if not breakdown:
                print("\r(no live links with a full window yet)            ", end="", flush=True)
                continue
            try:
                _cls, blended = voter.finalize()
            except ValueError:
                # every live link belongs to a node weighted 0, so there is no vote to take
                print("\r(live links present, but all from chance-level nodes)   ", end="", flush=True)
                continue
            present_proba = float(blended[present_class_index]) if present_class_index >= 0 else 0.0
            label = "PRESENT" if present_proba >= 0.5 else "absent "
            bar = "#" * int(present_proba * 20)
            print(f"{label}  P {present_proba:0.2f}  {bar:<20}  [{len(breakdown)} links] "
                  + " ".join(breakdown))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
