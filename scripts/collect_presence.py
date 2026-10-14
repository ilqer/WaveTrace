"""Capture labeled empty/present sessions over UDP and train a presence model per mesh node.

Each node is trained on its own, so live a node that drops only lowers the vote weight instead of
taking the system down. Each session is part A (zone empty) then part B (stand and move in
the zone).
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

    calibrated_node_ids = sorted(int(os.path.basename(node_dir)[len("node"):])
                                 for node_dir in glob.glob(os.path.join(args.cal, "node*"))
                                 if os.path.basename(node_dir)[len("node"):].isdigit())
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
    dataset_dirs = {node_id: [] for node_id in calibrated_node_ids}

    for i in range(args.sessions):
        empty_capture = capture_links(
            f"session {i+1}/{args.sessions} — part A: keep the zone empty and still.",
            args.frames, args.port, calibrated_node_ids, countdown=5)
        present_capture = capture_links(
            f"session {i+1}/{args.sessions} — part B: stand and move in the zone.",
            args.frames, args.port, calibrated_node_ids)
        for node_id in calibrated_node_ids:
            keys = sorted(link_key for link_key in set(empty_capture) | set(present_capture)
                          if link_key[1] == node_id)
            used_link_count = 0
            for key in keys:
                # resampled apart: one resample across both would interpolate frames over the gap
                empty_frames = resample_uniform(empty_capture.get(key, []), DEFAULT_TARGET_SAMPLE_RATE_HZ)
                present_frames = resample_uniform(present_capture.get(key, []), DEFAULT_TARGET_SAMPLE_RATE_HZ)
                if len(empty_frames) < DEFAULT_WINDOW_FRAMES or len(present_frames) < DEFAULT_WINDOW_FRAMES:
                    continue  # too short on this grid to emit a window in each class
                span = (present_frames[0].timestamp, present_frames[-1].timestamp + 1.0)
                tag = key[0].replace(":", "")  # tx mac short, ':'-free for a path segment
                recording_dir, dataset_dir = (f"{args.root}/sess_{i}/node{node_id}/link_{tag}",
                                              f"{args.root}/ds_{i}/node{node_id}/link_{tag}")
                save_recording(empty_frames + present_frames, recording_dir)
                collect_source(RecordingSource(recording_dir), f"{args.cal}/node{node_id}", dataset_dir,
                               [span], stage="presence", session_id=f"sess{i}", subject_id=SUBJECT)
                dataset_dirs[node_id].append(dataset_dir)
                used_link_count += 1
            if used_link_count == 0:
                print(f"   [SKIP] node {node_id} session {i}: "
                      f"no link had >= {DEFAULT_WINDOW_FRAMES} frames/class.")

    print("\ntraining per-node presence models...")
    trained = []
    for node_id in calibrated_node_ids:
        if not dataset_dirs[node_id]:
            print(f"   [SKIP] node {node_id}: no usable sessions.")
            continue
        _, metrics = train_presence(dataset_dirs[node_id], out_dir=f"{args.root}/model/node{node_id}")
        logo = metrics.get("logo", {}).get("session")
        line = (f"   [OK]   node {node_id}: samples={metrics['n_samples']} "
                f"class_counts={metrics['class_counts']} train_acc={metrics['train_accuracy']:.3f}")
        if logo:
            line += (f"  LOGO={logo['accuracy']:.3f} (majority {logo['majority_accuracy']:.3f}, "
                     f"TPR {logo.get('tpr', 0):.3f}, FP {logo.get('fp_rate', 0):.3f})")
        print(line)
        trained.append(node_id)

    if not trained:
        print("\n[ERROR] no node trained — check the boards or run mesh_verify.py.", file=sys.stderr)
        return
    print(f"\nmodels saved for nodes {trained} -> {args.root}/model/node*/  "
          "(LOGO is the honest number: it must clearly beat the majority baseline.)")

    print('\a', end='', flush=True)

if __name__ == "__main__":
    main()
