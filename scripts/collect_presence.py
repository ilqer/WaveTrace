"""Step 2 of live bring-up: capture labeled empty/present sessions over UDP and train a presence
model PER mesh node. Run AFTER collect_baseline.py (needs data/cal/node{id}/ for each node).

Per-RX-node models: one capture pass per condition feeds all nodes; each node's frames are labeled
and trained independently into data/model/node{id}/. Live, run_live_mesh.py votes across nodes, so a
node that drops/crashes just lowers the vote weight instead of taking the system down.

Per-LINK training (matches serving): capture splits the stream per (tx->rx) link and each link is
resampled onto the shared pipeline rate and windowed on its OWN clean grid, then all of a node's
links are POOLED into that node's single head. This mirrors run_live_mesh (per-link, 100 Hz),
instead of interleaving both transmitters into one window, and teaches the head the human signal
common to every link (generalize) rather than the round-robin slot-switching artifact of a merged
stream.

Each session = part A (zone EMPTY) then part B (you stand + MOVE in the zone).
"""

import argparse
import collections
import glob
import os
import socket
import sys
import time

from wavetrace.Source import (UdpSource, UdpSourceOptions, RecordingSource, saveRecording,
                              parseBatchLinks, resampleUniform, bindUdp)
from wavetrace.Cli import collectSource
from wavetrace.recognition import trainPresence
from wavetrace.domain.contracts import DEFAULT_TARGET_SAMPLE_RATE_HZ, DEFAULT_WINDOW_FRAMES

SUBJECT = "u0"


def detectNodes(port, timeout_seconds=3.0):
    """Briefly listen to detect the active Node IDs in the live UDP stream. Returns sorted list."""
    print("listening for active nodes...")
    detected = collections.Counter()
    source = UdpSource(UdpSourceOptions(port=port, timeout_seconds=timeout_seconds, max_frames=150))
    for fr in source.frames():
        detected[fr.node_id] += 1
    return sorted(detected.keys())


def captureAll(prompt, n, port, node_ids, countdown=0, max_capture_s=60.0):
    """Collect up to n frames PER (tx->rx) LINK in ONE pass. Returns {(tx_short, rx_node): [frames]}.

    Per-LINK (not per-node) so training matches the per-link serving path (run_live_mesh): each directed
    link is its own clean single-channel stream instead of both transmitters merged. Keeps the dominant
    subcarrier width per link. Stops when every expected RX node has appeared AND all known links reach
    n, OR max_capture_s elapses (the per-recv timeout only fires on TOTAL silence, so the wall-clock
    deadline is what stops a lone quiet link from stalling the loop forever)."""
    print(f"\n>> {prompt}\n   press Enter to start...", flush=True)
    input()
    if countdown:
        for d in range(countdown, 0, -1):
            print(f"   starting in {d}s...", end="\r")
            time.sleep(1)
        print()
    print("   [CAPTURING] hold the condition steady...")

    links = collections.defaultdict(list)  # (tx_short, rx_node) -> [frames]
    want = set(node_ids)
    sock = bindUdp(port, timeout=15.0)
    start = time.time()
    lastPrint = start
    try:
        while True:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                break  # total silence
            for key, frames in parseBatchLinks(payload).items():
                buf = links[key]
                if len(buf) < n:
                    buf.extend(frames[: n - len(buf)])
            now = time.time()
            if links and want <= {k[1] for k in links} and all(len(b) >= n for b in links.values()):
                break
            if now - start > max_capture_s:
                short = [k for k, v in sorted(links.items()) if len(v) < n]
                print(f"\n[WARN] deadline {max_capture_s:g}s hit; links short of {n}: {short}")
                break
            if now - lastPrint >= 1.0:
                perNode = dict(sorted(collections.Counter(k[1] for k in links).items()))
                mn = min((len(v) for v in links.values()), default=0)
                print(f"   links/node {perNode}  min {mn}/{n}...", end="\r")
                lastPrint = now
    finally:
        sock.close()
    print()
    for key in links:  # dominant subcarrier width per link (widths can differ across boards/bands)
        if links[key]:
            S = collections.Counter(f.num_subcarriers for f in links[key]).most_common(1)[0][0]
            links[key] = [f for f in links[key] if f.num_subcarriers == S]
    print('\a', end='', flush=True)
    return dict(links)


def main():
    parser = argparse.ArgumentParser(description="Capture presence sessions and train a model per node.")
    parser.add_argument("--node", type=int, default=None, help="Train ONLY this node (default: all calibrated)")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--sessions", type=int, default=3, help="Number of sessions to capture (default: 3)")
    parser.add_argument("--frames", type=int, default=1500, help="Frames per condition per node (default: 1500)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    args = parser.parse_args()
    if args.cal is None:
        args.cal = f"{args.root}/cal"

    # nodes with a calibration from collect_baseline, optionally narrowed by --node
    calNodes = sorted(int(os.path.basename(d)[len("node"):])
                       for d in glob.glob(os.path.join(args.cal, "node*"))
                       if os.path.basename(d)[len("node"):].isdigit())
    if not calNodes:
        print(f"\n[ERROR] no per-node calibrations in {args.cal}/node*. Run collect_baseline.py first.",
              file=sys.stderr)
        return
    if args.node is not None:
        calNodes = [args.node] if args.node in calNodes else []
        if not calNodes:
            print(f"\n[ERROR] node {args.node} has no calibration in {args.cal}.", file=sys.stderr)
            return
    print(f"training nodes: {calNodes} (calibrated)")

    os.makedirs(f"{args.root}/model", exist_ok=True)
    dsDirs = {nid: [] for nid in calNodes}

    for i in range(args.sessions):
        empty = captureAll(f"session {i+1}/{args.sessions} — part A: keep the zone empty and still.",
                            args.frames, args.port, calNodes, countdown=5)
        present = captureAll(f"session {i+1}/{args.sessions} — part B: stand and move in the zone.",
                              args.frames, args.port, calNodes)
        for nid in calNodes:
            # every (tx->rx) link on this node, windowed on its own grid, pooled into one head (same session_id)
            keys = sorted(k for k in set(empty) | set(present) if k[1] == nid)
            used = 0
            for key in keys:
                # resample separately: a single resample across both would interpolate fake frames across the gap
                e = resampleUniform(empty.get(key, []), DEFAULT_TARGET_SAMPLE_RATE_HZ)
                p = resampleUniform(present.get(key, []), DEFAULT_TARGET_SAMPLE_RATE_HZ)
                # a link segment shorter than this emits no window
                if len(e) < DEFAULT_WINDOW_FRAMES or len(p) < DEFAULT_WINDOW_FRAMES:
                    continue  # too short on this grid to emit a window in each class
                span = (p[0].timestamp, p[-1].timestamp + 1.0)
                tag = key[0].replace(":", "")  # tx mac short, ':'-free for a path segment
                rec, ds = f"{args.root}/sess_{i}/node{nid}/link_{tag}", f"{args.root}/ds_{i}/node{nid}/link_{tag}"
                saveRecording(e + p, rec)
                collectSource(RecordingSource(rec), f"{args.cal}/node{nid}", ds, [span],
                               stage="presence", session_id=f"sess{i}", subject_id=SUBJECT)
                dsDirs[nid].append(ds)
                used += 1
            if used == 0:
                print(f"   [SKIP] node {nid} session {i}: "
                      f"no link had >= {DEFAULT_WINDOW_FRAMES} frames/class.")

    print("\ntraining per-node presence models...")
    trained = []
    for nid in calNodes:
        if not dsDirs[nid]:
            print(f"   [SKIP] node {nid}: no usable sessions.")
            continue
        _, m = trainPresence(dsDirs[nid], out_dir=f"{args.root}/model/node{nid}")
        logo = m.get("logo", {}).get("session")
        line = (f"   [OK]   node {nid}: samples={m['n_samples']} class_counts={m['class_counts']} "
                f"train_acc={m['train_accuracy']:.3f}")
        if logo:
            line += (f"  LOGO={logo['accuracy']:.3f} (majority {logo['majority_accuracy']:.3f}, "
                     f"TPR {logo.get('tpr', 0):.3f}, FP {logo.get('fp_rate', 0):.3f})")
        print(line)
        trained.append(nid)

    if not trained:
        print("\n[ERROR] no node trained — check the boards or run mesh_verify.py.", file=sys.stderr)
        return
    print(f"\nmodels saved for nodes {trained} -> {args.root}/model/node*/  "
          "(LOGO is the honest number: it must clearly beat the majority baseline.)")

    print('\a', end='', flush=True)

if __name__ == "__main__":
    main()
