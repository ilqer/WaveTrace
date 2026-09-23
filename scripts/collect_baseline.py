"""Capture a quiet baseline over UDP and calibrate every mesh node.

Each node needs its own calibration because the boards are not interchangeable: one may run
gain=LOCK (gain frozen in the chip) while another runs gain=SKIP (the host normalizes instead).
"""

import argparse
import collections
import os
import sys
import time

from wavetrace.Source import RecordingSource, UdpSource, UdpSourceOptions, saveRecording
from wavetrace.application.calibrate import calibrate_source


def detectNodes(port, timeout_seconds=3.0):
    """Node ids heard in one short listen, sorted."""
    print("listening for active nodes...")
    detected = collections.Counter()
    source = UdpSource(UdpSourceOptions(port=port, timeout_seconds=timeout_seconds, max_frames=150))
    for fr in source.frames():
        detected[fr.node_id] += 1
    return sorted(detected.keys())


def captureAll(n, port, node_ids, timeout_seconds=20.0, max_capture_s=60.0):
    """Collect up to n frames per node in one pass. Returns {node_id: [frames]}, each node reduced to
    its dominant subcarrier width, because widths differ across boards and bands. The wall-clock
    deadline is needed because the receive timeout only fires on total silence, so one node going
    quiet while the others stream would otherwise leave the loop running forever."""
    frames = {nid: [] for nid in node_ids}
    source = UdpSource(UdpSourceOptions(port=port, timeout_seconds=timeout_seconds, max_frames=None))
    start = time.time()
    lastPrint = start
    for fr in source.frames():
        buf = frames.get(fr.node_id)
        if buf is not None and len(buf) < n:
            buf.append(fr)
        if frames and all(len(b) >= n for b in frames.values()):
            break
        now = time.time()
        if now - start > max_capture_s:
            short = [k for k, v in sorted(frames.items()) if len(v) < n]
            print(f"\n[WARN] deadline {max_capture_s:g}s hit; nodes short of {n}: {short}")
            break
        if now - lastPrint >= 1.0:
            counts = "  ".join(f"N{k}:{len(v)}" for k, v in sorted(frames.items()))
            print(f"   {counts}  (target {n}/node)...", end="\r")
            lastPrint = now
    print()
    for nid in frames:
        if frames[nid]:
            S = collections.Counter(f.num_subcarriers for f in frames[nid]).most_common(1)[0][0]
            frames[nid] = [f for f in frames[nid] if f.num_subcarriers == S]
    print('\a', end='', flush=True)
    return frames


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

    nodes = detectNodes(args.port)
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
    for d in range(5, 0, -1):
        print(f"   starting in {d}s...", end="\r")
        time.sleep(1)
    print("\n   [CAPTURING] keep the room still and empty...")

    frames = captureAll(args.frames, args.port, nodes)

    calibrated = []
    for nid in nodes:
        fr = frames.get(nid, [])
        if len(fr) < args.min_frames:
            print(f"   [SKIP] node {nid}: only {len(fr)} frames (< {args.min_frames}), not calibrated.")
            continue
        saveRecording(fr, f"{args.root}/baseline_raw/node{nid}")
        calibrate_source(RecordingSource(f"{args.root}/baseline_raw/node{nid}"), f"{args.root}/cal/node{nid}",
                          baseline_packets=min(2000, len(fr)))
        print(f"   [OK]   node {nid}: {len(fr)} frames, {fr[0].num_subcarriers} subcarriers "
              f"-> {args.root}/cal/node{nid}")
        calibrated.append(nid)

    if not calibrated:
        print(f"\n[ERROR] no node reached {args.min_frames} frames. Check the boards or mesh_verify.py.",
              file=sys.stderr)
        return
    print(f"\ncalibration written for nodes {calibrated} -> {args.root}/cal/node*/")

    print('\a', end='', flush=True)

if __name__ == "__main__":
    main()
