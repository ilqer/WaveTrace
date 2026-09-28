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
from wavetrace.application.collect import collect_source
from wavetrace.recognition import make_link_tag, parse_link_tag, train_weapon
from wavetrace.domain.contracts import DEFAULT_TARGET_SAMPLE_RATE_HZ, DEFAULT_WINDOW_FRAMES
from link_capture import capture_links


def _emit(cap, root, cal_root, nid, sess_id, subject, carry, cond, weapon, bg_subtract=False):
    """Writes one labeled dataset per link of node `nid` (class 1 if `weapon` else 0) and returns
    their directories."""
    out = []
    for key in sorted(k for k in cap if k[1] == nid):
        fr = resample_uniform(cap.get(key, []), DEFAULT_TARGET_SAMPLE_RATE_HZ)
        if len(fr) < DEFAULT_WINDOW_FRAMES:  # a link shorter than this on the grid emits no window
            continue
        span = (fr[0].timestamp, fr[-1].timestamp + 1.0)
        tag = make_link_tag(key[0])
        rec = f"{root}/weapon_rec/{sess_id}/{cond}/node{nid}/link_{tag}"
        ds = f"{root}/weapon_ds/node{nid}/{sess_id}_{cond}_link{tag}"
        save_recording(fr, rec)
        # a span marks class 1 over the segment and no span marks class 0; bg_subtract takes the
        # quiet-room level out of σ²[p]
        collect_source(RecordingSource(rec), f"{cal_root}/node{nid}", ds,
                       [span] if weapon else [],
                       stage="weapon", session_id=sess_id, subject_id=subject,
                       subtract_ic_baseline=bg_subtract)
        out.append(ds)
    return out


def _train_and_report(ds_dirs, out_dir, label):
    """Trains one weapon head on `ds_dirs`, prints its LOGO line under `label`, and returns whether
    it trained."""
    if not ds_dirs:
        print(f"   [SKIP] {label}: no datasets in pool.")
        return False
    try:
        _, m = train_weapon(ds_dirs, out_dir=out_dir, feature_mode="ic27")
    except ValueError as e:  # WeaponHead.fit needs BOTH classes in the pool
        print(f"   [SKIP] {label}: {e}")
        return False
    logo = m.get("logo", {})
    sess = logo.get("session") or logo.get("subject")
    line = (f"   [OK]   {label}: samples={m['n_samples']} class_counts={m['class_counts']} "
            f"train_acc={m['train_accuracy']:.3f}")
    if sess:
        line += (f"  LOGO={sess['accuracy']:.3f} (majority {sess['majority_accuracy']:.3f}, "
                 f"TPR {sess.get('tpr', 0):.3f}, FP {sess.get('fp_rate', 0):.3f})")
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
        for nid in calibrated_node_ids:
            _emit(clear, args.root, args.cal, nid, session_id, args.subject, args.carry, "clear",
                  weapon=False, bg_subtract=args.bg_subtract)
            _emit(armed, args.root, args.cal, nid, session_id, args.subject, args.carry, "weapon",
                  weapon=True, bg_subtract=args.bg_subtract)

    unit = "per-link (tx->rx)" if args.per_link else "per-node"
    print(f"\ntraining {unit} weapon models (ic27) on the cumulative pool...")
    trained = []
    for nid in calibrated_node_ids:
        dataset_dirs = sorted(glob.glob(f"{args.root}/weapon_ds/node{nid}/*"))
        if not args.per_link:
            if _train_and_report(dataset_dirs, f"{args.model}/node{nid}", f"Node {nid}"):
                trained.append(nid)
            continue
        dataset_dirs_by_tag = collections.defaultdict(list)
        for d in dataset_dirs:
            tag = parse_link_tag(d)
            if tag is not None:
                dataset_dirs_by_tag[tag].append(d)
        if not dataset_dirs_by_tag:
            print(f"   [SKIP] node {nid}: no per-link datasets, re-capture with collect_weapon.")
            continue
        for tag in sorted(dataset_dirs_by_tag):
            if _train_and_report(dataset_dirs_by_tag[tag], f"{args.model}/node{nid}/link{tag}",
                                 f"Node {nid} link{tag}"):
                trained.append((nid, tag))

    if not trained:
        print("\n[ERROR] nothing trained — need both no-weapon and weapon captures in the pool.",
              file=sys.stderr)
        return
    dest = f"{args.model}/node*/link*/" if args.per_link else f"{args.model}/node*/"
    print(f"\nweapon models saved ({len(trained)} {unit} heads) -> {dest}  "
          "(LOGO must clearly beat majority; expect this to need many subjects/positions/objects.)")

    print('\a', end='', flush=True)

if __name__ == "__main__":
    main()
