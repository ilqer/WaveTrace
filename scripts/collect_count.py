"""Independent PEOPLE-COUNT pipeline — capture labeled count sessions over UDP and train a per-node
count model. Standalone from the presence mesh scripts: imports only library code (wavetrace.*), never
run_live_mesh / collect_presence, and writes to its own data/model_count root so presence is untouched.

You choose the counts up front with --max-count N: you capture levels 0,1,...,N where the top level N
means "N or more people". Per-(tx->rx)-LINK + TARGET_FS resample, pooled into each node's single
multi-class head (same train/serve parity as the presence path). Calibration is shared and reused from
collect_baseline (data/cal/node{id}) — gain/NBVI are per-node, count-independent.

    .venv/bin/python scripts/collect_count.py --max-count 3 --sessions 3
"""

import argparse
import collections
import glob
import os
import socket
import sys
import time

import numpy as np

from wavetrace.Source import RecordingSource, saveRecording, parseBatchLinks, resampleUniform, bindUdp
from wavetrace.Cli import collectSource
from wavetrace.recognition import trainPresence
from wavetrace.groundtruth.CameraLabeler import ScriptedLabeler

SUBJECT = "u0"
TARGET_FS = 100.0   # resample grid; MUST match run_count.TARGET_FS so train and serve windows align
WINDOW = 128        # front-end window (frames); a link shorter than this on the grid emits no window


def countName(c, max_count):
    """Display label for a count class: the top level is the open-ended 'N+' bin."""
    return f"{c}+" if c >= max_count else str(c)


def captureLinks(prompt, n, port, node_ids, countdown=0, max_capture_s=60.0):
    """Collect up to n frames PER (tx->rx) LINK in ONE pass. Returns {(tx_short, rx_node): [frames]}.

    Per-link so training matches per-link serving (run_count): each directed link is its own clean
    single-channel stream. Keeps the dominant subcarrier width per link. Stops when every expected RX
    node has appeared AND all known links reach n, OR max_capture_s elapses (the recv timeout fires only
    on TOTAL silence, so the wall-clock deadline is what stops a lone quiet link from stalling)."""
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
    parser = argparse.ArgumentParser(description="Capture people-count sessions and train a model per node.")
    parser.add_argument("--node", type=int, default=None, help="Train ONLY this node (default: all calibrated)")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--sessions", type=int, default=3, help="Number of sessions to capture (default: 3)")
    parser.add_argument("--frames", type=int, default=1500, help="Frames per count per node (default: 1500)")
    parser.add_argument("--max-count", type=int, default=3, help="Top count level; captures 0..N, N='N+' (default: 3)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Count model root (default: <root>/model_count)")
    args = parser.parse_args()
    if args.cal is None:
        args.cal = f"{args.root}/cal"
    if args.model is None:
        args.model = f"{args.root}/model_count"

    if args.max_count < 1:
        print("[ERROR] --max-count must be >= 1 (need at least empty vs one person).", file=sys.stderr)
        return
    counts = list(range(args.max_count + 1))  # 0,1,...,N; class_id == count

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
    labels = [countName(c, args.max_count) for c in counts]
    print(f"training nodes: {calNodes}; count classes: {labels}")

    os.makedirs(args.model, exist_ok=True)
    dsDirs = {nid: [] for nid in calNodes}

    for i in range(args.sessions):
        for c in counts:
            label = countName(c, args.max_count)
            cap = captureLinks(
                f"session {i+1}/{args.sessions} — put {label} people in the zone and have them move.",
                args.frames, args.port, calNodes, countdown=5 if c == 0 else 0)
            print('\a\a\a', end='', flush=True)  # 3 beeps = done, stop moving
            for nid in calNodes:
                # every (tx->rx) link on this node, resampled on its own grid, pooled into one head
                for key in sorted(k for k in cap if k[1] == nid):
                    fr = resampleUniform(cap.get(key, []), TARGET_FS)
                    if len(fr) < WINDOW:
                        continue
                    span = (fr[0].timestamp, fr[-1].timestamp + 1.0)
                    tag = key[0].replace(":", "")  # tx mac short, ':'-free for a path segment
                    rec = f"{args.root}/count_sess_{i}/c{c}/node{nid}/link_{tag}"
                    ds = f"{args.root}/count_ds_{i}/c{c}/node{nid}/link_{tag}"
                    saveRecording(fr, rec)
                    # constant-count labeler: every window in this segment gets class_id = c
                    lab = ScriptedLabeler([(span[0], span[1], True)],
                                          label_fn=lambda raw, t, _c=c, _n=label: (_c, _n))
                    collectSource(RecordingSource(rec), f"{args.cal}/node{nid}", ds, [span],
                                   stage="presence", session_id=f"sess{i}", subject_id=SUBJECT,
                                   labeler=lab)
                    dsDirs[nid].append(ds)

    print("\ntraining per-node count models...")
    trained = []
    for nid in calNodes:
        if not dsDirs[nid]:
            print(f"   [SKIP] node {nid}: no usable segments.")
            continue
        _, m = trainPresence(dsDirs[nid], out_dir=f"{args.model}/node{nid}")
        logo = m.get("logo", {}).get("session")
        line = (f"   [OK]   node {nid}: samples={m['n_samples']} class_counts={m['class_counts']} "
                f"train_acc={m['train_accuracy']:.3f}")
        if logo:
            line += f"  LOGO={logo['accuracy']:.3f} (majority {logo['majority_accuracy']:.3f})"
        print(line)
        if logo and "confusion" in logo:
            cm = np.asarray(logo["confusion"])
            rowSums = cm.sum(axis=1)
            classNames = [countName(c, args.max_count) for c in counts]
            per = {classNames[i]: f"{cm[i,i]/rowSums[i]:.0%}" if rowSums[i] > 0 else "n/a"
                   for i in range(min(len(classNames), cm.shape[0]))}
            print(f"          per-class: {per}")
        trained.append(nid)

    if not trained:
        print("\n[ERROR] no node trained — check the boards or run mesh_verify.py.", file=sys.stderr)
        return
    print(f"\ncount models saved for nodes {trained} -> {args.model}/node*/  "
          "(LOGO must clearly beat the majority baseline, or it's not real.)")

    print('\a', end='', flush=True)

if __name__ == "__main__":
    main()
