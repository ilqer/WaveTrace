"""Capture labeled empty/present sessions over UDP and train a presence model per mesh node.

Each node is trained on its own, so live a node that drops just lowers the vote weight instead of
taking the system down. Each session is part A (zone empty) then part B (stand and move in the zone).
"""

import argparse
import glob
import os
import sys

from wavetrace.adapters.source import RecordingSource, save_recording
from wavetrace.domain.dsp.resample import resample_uniform
from wavetrace.application.collect import collect_source
from wavetrace.recognition import train_presence
from wavetrace.domain.contracts import DEFAULT_TARGET_SAMPLE_RATE_HZ, DEFAULT_WINDOW_FRAMES
from link_capture import capture_links

SUBJECT = "u0"


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

    calibrated_node_ids = sorted(int(os.path.basename(d)[len("node"):])
                                 for d in glob.glob(os.path.join(args.cal, "node*"))
                                 if os.path.basename(d)[len("node"):].isdigit())
    if not calibrated_node_ids:
        print(f"\n[ERROR] no per-node calibrations in {args.cal}/node*. Run collect_baseline.py first.",
              file=sys.stderr)
        return
    if args.node is not None:
        calibrated_node_ids = [args.node] if args.node in calibrated_node_ids else []
        if not calibrated_node_ids:
            print(f"\n[ERROR] node {args.node} has no calibration in {args.cal}.", file=sys.stderr)
            return
    print(f"training nodes: {calibrated_node_ids} (calibrated)")

    os.makedirs(f"{args.root}/model", exist_ok=True)
    dataset_dirs = {nid: [] for nid in calibrated_node_ids}

    for i in range(args.sessions):
        empty = capture_links(f"session {i+1}/{args.sessions} — part A: keep the zone empty and still.",
                            args.frames, args.port, calibrated_node_ids, countdown=5)
        present = capture_links(f"session {i+1}/{args.sessions} — part B: stand and move in the zone.",
                              args.frames, args.port, calibrated_node_ids)
        for nid in calibrated_node_ids:
            keys = sorted(k for k in set(empty) | set(present) if k[1] == nid)
            used = 0
            for key in keys:
                # resampled apart: one resample across both would interpolate frames over the gap
                e = resample_uniform(empty.get(key, []), DEFAULT_TARGET_SAMPLE_RATE_HZ)
                p = resample_uniform(present.get(key, []), DEFAULT_TARGET_SAMPLE_RATE_HZ)
                if len(e) < DEFAULT_WINDOW_FRAMES or len(p) < DEFAULT_WINDOW_FRAMES:
                    continue  # too short on this grid to emit a window in each class
                span = (p[0].timestamp, p[-1].timestamp + 1.0)
                tag = key[0].replace(":", "")  # tx mac short, ':'-free for a path segment
                rec, ds = f"{args.root}/sess_{i}/node{nid}/link_{tag}", f"{args.root}/ds_{i}/node{nid}/link_{tag}"
                save_recording(e + p, rec)
                collect_source(RecordingSource(rec), f"{args.cal}/node{nid}", ds, [span],
                               stage="presence", session_id=f"sess{i}", subject_id=SUBJECT)
                dataset_dirs[nid].append(ds)
                used += 1
            if used == 0:
                print(f"   [SKIP] node {nid} session {i}: "
                      f"no link had >= {DEFAULT_WINDOW_FRAMES} frames/class.")

    print("\ntraining per-node presence models...")
    trained = []
    for nid in calibrated_node_ids:
        if not dataset_dirs[nid]:
            print(f"   [SKIP] node {nid}: no usable sessions.")
            continue
        _, m = train_presence(dataset_dirs[nid], out_dir=f"{args.root}/model/node{nid}")
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
