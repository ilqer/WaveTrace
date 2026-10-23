"""Capture a quiet baseline over UDP and calibrate every mesh node.

Each node needs its own calibration because the boards are not interchangeable: one may run
gain=LOCK (gain frozen in the chip) while another runs gain=SKIP (the host normalizes instead).
"""

import argparse
import collections
import os
import sys
import time

from wavetrace.adapters.source import (
    RecordingSource, UdpSource, UdpSourceOptions, detect_node_ids, save_recording,
)
from wavetrace.pipeline import calibrate_source


def capture_per_node(max_frames_per_node, port, node_ids, timeout_seconds=20.0, max_capture_s=60.0):
    """Returns {node_id: [frames]}, each node reduced to its dominant subcarrier width (widths differ
    across boards and bands). The wall-clock deadline is needed because the receive timeout only
    fires on total silence: one node going quiet while the others stream would loop forever."""
    frames_by_node = {node_id: [] for node_id in node_ids}
    source = UdpSource(UdpSourceOptions(port=port, timeout_seconds=timeout_seconds, max_frames=None))
    start_time_s = time.time()
    last_print_time_s = start_time_s
    for received_frame in source.frames():
        node_frames = frames_by_node.get(received_frame.node_id)
        if node_frames is not None and len(node_frames) < max_frames_per_node:
            node_frames.append(received_frame)
        if frames_by_node and all(len(queued_frames) >= max_frames_per_node
                                  for queued_frames in frames_by_node.values()):
            break
        now = time.time()
        if now - start_time_s > max_capture_s:
            short_node_ids = [reported_node_id
                              for reported_node_id, captured_frames in sorted(frames_by_node.items())
                              if len(captured_frames) < max_frames_per_node]
            print(f"\n[WARN] deadline {max_capture_s:g}s hit; "
                  f"nodes short of {max_frames_per_node}: {short_node_ids}")
            break
        if now - last_print_time_s >= 1.0:
            counts = "  ".join(f"N{reported_node_id}:{len(captured_frames)}"
                               for reported_node_id, captured_frames in sorted(frames_by_node.items()))
            print(f"   {counts}  (target {max_frames_per_node}/node)...", end="\r")
            last_print_time_s = now
    print()
    for node_id in frames_by_node:
        if frames_by_node[node_id]:
            subcarrier_count = collections.Counter(
                frame.num_subcarriers for frame in frames_by_node[node_id]).most_common(1)[0][0]
            frames_by_node[node_id] = [frame for frame in frames_by_node[node_id]
                                       if frame.num_subcarriers == subcarrier_count]
    print('\a', end='', flush=True)
    return frames_by_node


def main():
    parser = argparse.ArgumentParser(description="Capture quiet baseline and calibrate every mesh node.")
    parser.add_argument("--node", type=int, default=None, help="Calibrate ONLY this node (default: all detected)")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--frames", type=int, default=3000, help="Baseline frames per node (default: 3000)")
    parser.add_argument("--min-frames", type=int, default=300, help="Skip a node with fewer frames than this (default: 300)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    args = parser.parse_args()

    os.makedirs(f"{args.root}/cal", exist_ok=True)

    nodes = detect_node_ids(args.port)
    if not nodes:
        print(f"\n[ERROR] no active nodes on UDP port {args.port}. "
              "Check the mesh boards are powered and flooding, or run `scripts/mesh_verify.py`.", file=sys.stderr)
        return
    if args.node is not None:
        if args.node not in nodes:
            print(f"warning: node {args.node} not seen in scan (seen: {nodes}); waiting for it anyway.")
        nodes = [args.node]
    print(f"calibrating nodes: {nodes}")

    print("\nkeep the room quiet and still, then press Enter to start capturing...", flush=True)
    input()
    for seconds_left in range(5, 0, -1):
        print(f"   starting in {seconds_left}s...", end="\r")
        time.sleep(1)
    print("\n   [CAPTURING] keep the room still and empty...")

    frames_by_node = capture_per_node(args.frames, args.port, nodes)

    calibrated = []
    for node_id in nodes:
        node_frames = frames_by_node.get(node_id, [])
        if len(node_frames) < args.min_frames:
            print(f"   [SKIP] node {node_id}: only {len(node_frames)} frames (< {args.min_frames}), "
                  "not calibrated.")
            continue
        save_recording(node_frames, f"{args.root}/baseline_raw/node{node_id}")
        calibrate_source(RecordingSource(f"{args.root}/baseline_raw/node{node_id}"),
                         f"{args.root}/cal/node{node_id}", baseline_packets=min(2000, len(node_frames)))
        print(f"   [OK]   node {node_id}: {len(node_frames)} frames, "
              f"{node_frames[0].num_subcarriers} subcarriers -> {args.root}/cal/node{node_id}")
        calibrated.append(node_id)

    if not calibrated:
        print(f"\n[ERROR] no node reached {args.min_frames} frames. Check the boards or mesh_verify.py.",
              file=sys.stderr)
        return
    print(f"\ncalibration written for nodes {calibrated} -> {args.root}/cal/node*/")

    print('\a', end='', flush=True)

if __name__ == "__main__":
    main()
