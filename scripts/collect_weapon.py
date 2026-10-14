"""Capture labeled no-weapon/weapon sessions over UDP and train a weapon model per node.

The head uses the inter-carrier feature block (stage="weapon", feature_mode="ic27") rather than the
amplitude features presence uses: a concealed object shows up in inter-subcarrier structure, not in
body motion, and the subject stands still throughout. Every run appends its datasets to the
cumulative pool at <root>/weapon_ds and retrains each node on the whole pool, so subject and
carry-position diversity builds up over many runs.
"""

import argparse
import collections
import glob
import os
import sys

from wavetrace.adapters.source import RecordingSource, save_recording
from wavetrace.domain.dsp.resample import resample_uniform
from wavetrace.pipeline import collect_source
from wavetrace.recognition import make_link_tag, parse_link_tag, train_weapon
from wavetrace.domain.contracts import DEFAULT_TARGET_SAMPLE_RATE_HZ, DEFAULT_WINDOW_FRAMES
from link_capture import capture_links


def _emit(capture, root, calibration_root, node_id, session_id, subject, carry, condition, is_weapon,
          should_subtract_baseline=False):
    """Writes one dataset per link of `node_id`, class 1 if `is_weapon` else 0; returns their directories."""
    dataset_dirs = []
    for key in sorted(link_key for link_key in capture if link_key[1] == node_id):
        frames = resample_uniform(capture.get(key, []), DEFAULT_TARGET_SAMPLE_RATE_HZ)
        if len(frames) < DEFAULT_WINDOW_FRAMES:  # a link shorter than this on the grid emits no window
            continue
        span = (frames[0].timestamp, frames[-1].timestamp + 1.0)
        tag = make_link_tag(key[0])
        recording_dir = f"{root}/weapon_rec/{session_id}/{condition}/node{node_id}/link_{tag}"
        dataset_dir = f"{root}/weapon_ds/node{node_id}/{session_id}_{condition}_link{tag}"
        save_recording(frames, recording_dir)
        # a span marks class 1 over the segment and no span marks class 0; should_subtract_baseline takes the
        # quiet-room level out of σ²[p]
        collect_source(RecordingSource(recording_dir), f"{calibration_root}/node{node_id}", dataset_dir,
                       [span] if is_weapon else [],
                       stage="weapon", session_id=session_id, subject_id=subject,
                       subtract_ic_baseline=should_subtract_baseline)
        dataset_dirs.append(dataset_dir)
    return dataset_dirs


def _train_and_report(dataset_dirs, out_dir, label):
    """Trains one weapon head on `dataset_dirs`, prints its LOGO line under `label`, and returns whether
    it trained."""
    if not dataset_dirs:
        print(f"   [SKIP] {label}: no datasets in pool.")
        return False
    try:
        _, metrics = train_weapon(dataset_dirs, out_dir=out_dir, feature_mode="ic27")
    except ValueError as error:  # WeaponHead.fit needs BOTH classes in the pool
        print(f"   [SKIP] {label}: {error}")
        return False
    logo = metrics.get("logo", {})
    session_logo = logo.get("session") or logo.get("subject")
    line = (f"   [OK]   {label}: samples={metrics['n_samples']} class_counts={metrics['class_counts']} "
            f"train_acc={metrics['train_accuracy']:.3f}")
    if session_logo:
        line += (f"  LOGO={session_logo['accuracy']:.3f} (majority {session_logo['majority_accuracy']:.3f}, "
                 f"TPR {session_logo.get('tpr', 0):.3f}, FP {session_logo.get('fp_rate', 0):.3f})")
    carry = logo.get("carry")  # does it generalize across carry pose, or ride that confound
    if carry:
        line += f"  carry-LOGO={carry['accuracy']:.3f} (maj {carry['majority_accuracy']:.3f})"
    print(line)
    return True


def main():
    parser = argparse.ArgumentParser(description="Capture no-weapon/weapon sessions and train a weapon model per node.")
    parser.add_argument("--node", type=int, default=None, help="Train ONLY this node (default: all calibrated)")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--sessions", type=int, default=3, help="Sessions to capture THIS run (default: 3)")
    parser.add_argument("--frames", type=int, default=1500, help="Frames per condition per node (default: 1500)")
    parser.add_argument("--subject", default="p0", help="Subject id for this run (vary it across people!)")
    parser.add_argument("--carry", default="na", help="Carry position label, e.g. waist/chest/ankle (default: na)")
    parser.add_argument("--bg-subtract", action=argparse.BooleanOptionalAction, dest="bg_subtract",
                        default=True,
                        help="Subtract the quiet-room baseline from σ²[p]; serving mirrors it. "
                             "Default ON (matches the web frontend); pass --no-bg-subtract to disable.")
    parser.add_argument("--per-link", action="store_true", dest="per_link",
                        help="Train ONE head per (tx->rx) DIRECTION -> model_weapon/node<id>/link<tag>/ "
                             "instead of pooling a node's directions into one head. "
                             "The signal is per-direction; pooling sign-flips the good NLOS link. "
                             "Bad directions are NOT dropped — run_weapon's LinkVoter zeroes a "
                             "sub-chance link via its own LOGO weight (accuracy_weights).")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Weapon model root (default: <root>/model_weapon)")
    args = parser.parse_args()
    if args.cal is None:
        args.cal = f"{args.root}/cal"
    if args.model is None:
        args.model = f"{args.root}/model_weapon"

    calibrated_node_ids = sorted(int(os.path.basename(directory)[len("node"):])
                                 for directory in glob.glob(os.path.join(args.cal, "node*"))
                                 if os.path.basename(directory)[len("node"):].isdigit())
    if not calibrated_node_ids:
        print(f"\n[ERROR] no per-node calibrations in {args.cal}/node*. Run collect_baseline.py first.",
              file=sys.stderr)
        return
    if args.node is not None:
        calibrated_node_ids = [args.node] if args.node in calibrated_node_ids else []
        if not calibrated_node_ids:
            print(f"\n[ERROR] node {args.node} has no calibration in {args.cal}.", file=sys.stderr)
            return
    print(f"training nodes: {calibrated_node_ids}  (subject={args.subject}, carry={args.carry})")

    os.makedirs(args.model, exist_ok=True)
    for i in range(args.sessions):
        session_id = f"{args.subject}_{args.carry}_s{i}"
        clear = capture_links(f"session {i+1}/{args.sessions} — stand still, no weapon on you.",
                              args.frames, args.port, calibrated_node_ids, countdown=5,
                              capture_hint="hold the condition steady (stand STILL)...")
        armed = capture_links(f"session {i+1}/{args.sessions} — stand still, weapon concealed on you.",
                              args.frames, args.port, calibrated_node_ids, countdown=5,
                              capture_hint="hold the condition steady (stand STILL)...")
        for node_id in calibrated_node_ids:
            _emit(clear, args.root, args.cal, node_id, session_id, args.subject, args.carry, "clear",
                  is_weapon=False, should_subtract_baseline=args.bg_subtract)
            _emit(armed, args.root, args.cal, node_id, session_id, args.subject, args.carry, "weapon",
                  is_weapon=True, should_subtract_baseline=args.bg_subtract)

    unit = "per-link (tx->rx)" if args.per_link else "per-node"
    print(f"\ntraining {unit} weapon models (ic27) on the cumulative pool...")
    trained = []
    for node_id in calibrated_node_ids:
        dataset_dirs = sorted(glob.glob(f"{args.root}/weapon_ds/node{node_id}/*"))
        if not args.per_link:
            if _train_and_report(dataset_dirs, f"{args.model}/node{node_id}", f"Node {node_id}"):
                trained.append(node_id)
            continue
        dataset_dirs_by_tag = collections.defaultdict(list)
        for directory in dataset_dirs:
            tag = parse_link_tag(directory)
            if tag is not None:
                dataset_dirs_by_tag[tag].append(directory)
        if not dataset_dirs_by_tag:
            print(f"   [SKIP] node {node_id}: no per-link datasets, re-capture with collect_weapon.")
            continue
        for tag in sorted(dataset_dirs_by_tag):
            if _train_and_report(dataset_dirs_by_tag[tag], f"{args.model}/node{node_id}/link{tag}",
                                 f"Node {node_id} link{tag}"):
                trained.append((node_id, tag))

    if not trained:
        print("\n[ERROR] nothing trained — need both no-weapon and weapon captures in the pool.",
              file=sys.stderr)
        return
    destination = f"{args.model}/node*/link*/" if args.per_link else f"{args.model}/node*/"
    print(f"\nweapon models saved ({len(trained)} {unit} heads) -> {destination}  "
          "(LOGO must clearly beat majority; expect this to need many subjects/positions/objects.)")

    print('\a', end='', flush=True)

if __name__ == "__main__":
    main()
