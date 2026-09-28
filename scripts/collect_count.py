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
    labels = [count_name(c, args.max_count) for c in counts]
    print(f"training nodes: {calibrated_node_ids}; count classes: {labels}")

    os.makedirs(args.model, exist_ok=True)
    dataset_dirs = {nid: [] for nid in calibrated_node_ids}

    for i in range(args.sessions):
        for c in counts:
            label = count_name(c, args.max_count)
            cap = capture_links(
                f"session {i+1}/{args.sessions} — put {label} people in the zone and have them move.",
                args.frames, args.port, calibrated_node_ids, countdown=5 if c == 0 else 0)
            print('\a\a\a', end='', flush=True)  # 3 beeps = done, stop moving
            for nid in calibrated_node_ids:
                for key in sorted(k for k in cap if k[1] == nid):
                    fr = resample_uniform(cap.get(key, []), DEFAULT_TARGET_SAMPLE_RATE_HZ)
                    if len(fr) < DEFAULT_WINDOW_FRAMES:  # a link shorter than this emits no window
                        continue
                    span = (fr[0].timestamp, fr[-1].timestamp + 1.0)
                    tag = key[0].replace(":", "")  # tx mac short, ':'-free for a path segment
                    rec = f"{args.root}/count_sess_{i}/c{c}/node{nid}/link_{tag}"
                    ds = f"{args.root}/count_ds_{i}/c{c}/node{nid}/link_{tag}"
                    save_recording(fr, rec)
                    lab = ScriptedLabeler([(span[0], span[1], True)],
                                          label_fn=lambda raw, t, _c=c, _n=label: (_c, _n))
                    collect_source(RecordingSource(rec), f"{args.cal}/node{nid}", ds, [span],
                                   stage="presence", session_id=f"sess{i}", subject_id=SUBJECT,
                                   labeler=lab)
                    dataset_dirs[nid].append(ds)

    print("\ntraining per-node count models...")
    trained = []
    for nid in calibrated_node_ids:
        if not dataset_dirs[nid]:
            print(f"   [SKIP] node {nid}: no usable segments.")
            continue
        _, m = train_presence(dataset_dirs[nid], out_dir=f"{args.model}/node{nid}")
        logo = m.get("logo", {}).get("session")
        line = (f"   [OK]   node {nid}: samples={m['n_samples']} class_counts={m['class_counts']} "
                f"train_acc={m['train_accuracy']:.3f}")
        if logo:
            line += f"  LOGO={logo['accuracy']:.3f} (majority {logo['majority_accuracy']:.3f})"
        print(line)
        if logo and "confusion" in logo:
            cm = np.asarray(logo["confusion"])
            row_sums = cm.sum(axis=1)
            class_names = [count_name(c, args.max_count) for c in counts]
            per = {class_names[i]: f"{cm[i,i]/row_sums[i]:.0%}" if row_sums[i] > 0 else "n/a"
                   for i in range(min(len(class_names), cm.shape[0]))}
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
