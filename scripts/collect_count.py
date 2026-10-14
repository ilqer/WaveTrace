"""Capture labeled people-count sessions over UDP and train a count model per node.

Levels 0..N are captured in turn, with the top level N meaning "N or more people". Calibration is
shared with the presence path (<root>/cal/node{id}): gain and NBVI are per node and do not depend on
how many people are in the zone.
"""

import argparse
import glob
import os
import sys

import numpy as np

from wavetrace.adapters.source import RecordingSource, save_recording
from wavetrace.domain.dsp.resample import resample_uniform
from wavetrace.application.collect import collect_source
from wavetrace.recognition import count_name, train_presence
from wavetrace.groundtruth.camera_labeler import ScriptedLabeler
from wavetrace.domain.contracts import DEFAULT_TARGET_SAMPLE_RATE_HZ, DEFAULT_WINDOW_FRAMES
from link_capture import capture_links

SUBJECT = "u0"


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
    counts = list(range(args.max_count + 1))  # class_id == the count

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
    labels = [count_name(count, args.max_count) for count in counts]
    print(f"training nodes: {calibrated_node_ids}; count classes: {labels}")

    os.makedirs(args.model, exist_ok=True)
    dataset_dirs = {node_id: [] for node_id in calibrated_node_ids}

    for i in range(args.sessions):
        for count in counts:
            label = count_name(count, args.max_count)
            capture = capture_links(
                f"session {i+1}/{args.sessions} — put {label} people in the zone and have them move.",
                args.frames, args.port, calibrated_node_ids, countdown=5 if count == 0 else 0)
            print('\a\a\a', end='', flush=True)  # 3 beeps = done, stop moving
            for node_id in calibrated_node_ids:
                for key in sorted(link_key for link_key in capture if link_key[1] == node_id):
                    frames = resample_uniform(capture.get(key, []), DEFAULT_TARGET_SAMPLE_RATE_HZ)
                    if len(frames) < DEFAULT_WINDOW_FRAMES:  # a link shorter than this emits no window
                        continue
                    span = (frames[0].timestamp, frames[-1].timestamp + 1.0)
                    tag = key[0].replace(":", "")  # tx mac short, ':'-free for a path segment
                    recording_dir = f"{args.root}/count_sess_{i}/c{count}/node{node_id}/link_{tag}"
                    dataset_dir = f"{args.root}/count_ds_{i}/c{count}/node{node_id}/link_{tag}"
                    save_recording(frames, recording_dir)
                    labeler = ScriptedLabeler(
                        [(span[0], span[1], True)],
                        label_function=lambda raw, timestamp_s, class_id=count, class_name=label: (
                            class_id, class_name))
                    collect_source(RecordingSource(recording_dir), f"{args.cal}/node{node_id}", dataset_dir,
                                   [span], stage="presence", session_id=f"sess{i}", subject_id=SUBJECT,
                                   labeler=labeler)
                    dataset_dirs[node_id].append(dataset_dir)

    print("\ntraining per-node count models...")
    trained = []
    for node_id in calibrated_node_ids:
        if not dataset_dirs[node_id]:
            print(f"   [SKIP] node {node_id}: no usable segments.")
            continue
        _, metrics = train_presence(dataset_dirs[node_id], out_dir=f"{args.model}/node{node_id}")
        logo = metrics.get("logo", {}).get("session")
        line = (f"   [OK]   node {node_id}: samples={metrics['n_samples']} "
                f"class_counts={metrics['class_counts']} train_acc={metrics['train_accuracy']:.3f}")
        if logo:
            line += f"  LOGO={logo['accuracy']:.3f} (majority {logo['majority_accuracy']:.3f})"
        print(line)
        if logo and "confusion" in logo:
            confusion = np.asarray(logo["confusion"])
            row_sums = confusion.sum(axis=1)
            class_names = [count_name(count, args.max_count) for count in counts]
            per_class_recall = {class_names[i]: (f"{confusion[i,i]/row_sums[i]:.0%}"
                                                 if row_sums[i] > 0 else "n/a")
                                for i in range(min(len(class_names), confusion.shape[0]))}
            print(f"          per-class: {per_class_recall}")
        trained.append(node_id)

    if not trained:
        print("\n[ERROR] no node trained — check the boards or run mesh_verify.py.", file=sys.stderr)
        return
    print(f"\ncount models saved for nodes {trained} -> {args.model}/node*/  "
          "(LOGO must clearly beat the majority baseline, or it's not real.)")

    print('\a', end='', flush=True)

if __name__ == "__main__":
    main()
