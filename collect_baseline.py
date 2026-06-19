"""Step 1 of live bring-up: capture a quiet baseline over UDP and calibrate EVERY mesh node.

Per-RX-node models: each ESP32 self-calibrates (its own NBVI subcarriers + gain regime), because the
boards are not interchangeable — one may run gain=LOCK (in-chip frozen gain) while another runs
gain=SKIP (host CV normalization). One quiet capture pass feeds all nodes; each gets its own cal dir.

Prereqs: mesh nodes powered + flooding on RD-WIN1, `mesh_verify.py` shows arrivals.
Produces data/cal/node{id}/ for every detected node.
"""

import argparse
import collections
import os
import sys
import time

from wavetrace.Source import UdpSource, RecordingSource, save_recording
from wavetrace.Cli import calibrate_source


def detect_nodes(port, timeout_s=3.0):
    """Briefly listen to detect the active Node IDs in the live UDP stream. Returns sorted list."""
    print("Listening to detect active nodes...")
    detected = collections.Counter()
    source = UdpSource(port, timeout_s=timeout_s, max_frames=150)
    for fr in source.frames():
        detected[fr.node_id] += 1
    return sorted(detected.keys())


def capture_all(n, port, node_ids, timeout_s=20.0, max_capture_s=60.0):
    """Collect up to n frames PER node in ONE listening pass. Returns {node_id: [frames]} (dominant
    subcarrier width kept per node, since widths can differ across boards/bands).

    Stops when every node has n frames OR max_capture_s elapses — the wall-clock deadline is essential
    because the per-recv timeout only fires on TOTAL silence: if one node stays quiet while others keep
    streaming, the all-nodes-reached check never trips and the loop would otherwise run forever."""
    frames = {nid: [] for nid in node_ids}
    source = UdpSource(port, timeout_s=timeout_s, max_frames=None)
    start = time.time()
    last_print = start
    for fr in source.frames():
        buf = frames.get(fr.node_id)
        if buf is not None and len(buf) < n:
            buf.append(fr)
        if frames and all(len(b) >= n for b in frames.values()):
            break
        now = time.time()
        if now - start > max_capture_s:
            short = [k for k, v in sorted(frames.items()) if len(v) < n]
            print(f"\n[WARN] capture deadline {max_capture_s:g}s hit; nodes short of {n}: {short}")
            break
        if now - last_print >= 1.0:
            counts = "  ".join(f"N{k}:{len(v)}" for k, v in sorted(frames.items()))
            print(f"   {counts}  (target {n}/node)...", end="\r")
            last_print = now
    print()
    for nid in frames:
        if frames[nid]:
            S = collections.Counter(f.num_subcarriers for f in frames[nid]).most_common(1)[0][0]
            frames[nid] = [f for f in frames[nid] if f.num_subcarriers == S]
    return frames


def main():
    parser = argparse.ArgumentParser(description="Capture quiet baseline and calibrate every mesh node.")
    parser.add_argument("--node", type=int, default=None, help="Calibrate ONLY this node (default: all detected)")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--frames", type=int, default=3000, help="Baseline frames per node (default: 3000)")
    parser.add_argument("--min-frames", type=int, default=300, help="Skip a node with fewer frames than this (default: 300)")
    args = parser.parse_args()

    os.makedirs("data/cal", exist_ok=True)

    nodes = detect_nodes(args.port)
    if not nodes:
        print(f"\n[ERROR] No active nodes detected on UDP port {args.port}. "
              "Are mesh boards powered and flooding? Run `mesh_verify.py` to confirm.", file=sys.stderr)
        return
    if args.node is not None:
        if args.node not in nodes:
            print(f"Warning: --node {args.node} not seen in scan (seen: {nodes}); will still wait for it.")
        nodes = [args.node]
    print(f"Calibrating nodes: {nodes}")

    input("\nEnsure room is QUIET and still. Press Enter to start capturing the baseline...")
    for d in range(5, 0, -1):
        print(f"   capturing baseline in {d}s...", end="\r")
        time.sleep(1)
    print("\n   [CAPTURING] keep the room still and empty...")

    frames = capture_all(args.frames, args.port, nodes)

    calibrated = []
    for nid in nodes:
        fr = frames.get(nid, [])
        if len(fr) < args.min_frames:
            print(f"   [SKIP] Node {nid}: only {len(fr)} frames (< {args.min_frames}). Not calibrated.")
            continue
        save_recording(fr, f"data/baseline_raw/node{nid}")
        calibrate_source(RecordingSource(f"data/baseline_raw/node{nid}"), f"data/cal/node{nid}",
                         baseline_packets=min(2000, len(fr)))
        print(f"   [OK]   Node {nid}: {len(fr)} frames, {fr[0].num_subcarriers} subcarriers "
              f"-> data/cal/node{nid}")
        calibrated.append(nid)

    if not calibrated:
        print(f"\n[ERROR] No node reached {args.min_frames} frames. Check the boards / mesh_verify.py.",
              file=sys.stderr)
        return
    print(f"\ncalibration written for nodes {calibrated} -> data/cal/node*/")


if __name__ == "__main__":
    main()
