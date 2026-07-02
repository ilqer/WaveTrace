"""Independent WEAPON pipeline — capture labeled no-weapon/weapon sessions over UDP and train a
per-node Stage-E weapon model. Standalone from the presence/count scripts: imports only library code
(wavetrace.*), writes to its own data/model_weapon root, and reuses the shared calibration (data/cal).

Stage-E uses the INTER-CARRIER feature block (stage="weapon" -> intercarrier dataset, trainWeapon
feature_mode="ic27"), NOT the amplitude features the presence head uses — the concealed-object signal
lives in inter-subcarrier structure, not in "is a body moving". Per-(tx->rx)-LINK + TARGET_FS resample,
pooled into each node's single head (same parity as the presence/count paths).

CUMULATIVE: every run appends its datasets under data/weapon_ds/ and RETRAINS each node on the whole
pool, so you build subject/position diversity over many runs. Run once per subject AND carry position:

    .venv/bin/python scripts/collect_weapon.py --subject p1 --carry waist --sessions 3
    .venv/bin/python scripts/collect_weapon.py --subject p2 --carry chest --sessions 3   # adds to the pool

WEAPON detection is hard and data-hungry: vary SUBJECTS, CARRY POSITIONS and OBJECTS, and judge it by
LOGO (held-out session/subject), never train accuracy. Static-first: the subject stands STILL.
"""

import argparse
import collections
import glob
import os
import socket
import sys
import time

from wavetrace.Source import RecordingSource, saveRecording, parseBatchLinks, resampleUniform, bindUdp
from wavetrace.Cli import collectSource
from wavetrace.recognition import trainWeapon

TARGET_FS = 100.0   # resample grid; MUST match run_weapon.TARGET_FS so train and serve windows align
WINDOW = 128        # front-end window (frames); a link shorter than this on the grid emits no window
# Cumulative dataset pool lives at <root>/weapon_ds (one subdir tree per run), globbed at train time.


def captureLinks(prompt, n, port, node_ids, countdown=0, max_capture_s=60.0):
    """Collect up to n frames PER (tx->rx) LINK in ONE pass. Returns {(tx_short, rx_node): [frames]}.

    Per-link so training matches per-link serving (run_weapon). Keeps the dominant subcarrier width per
    link. Stops when every expected RX node has appeared AND all known links reach n, OR max_capture_s
    elapses (the recv timeout fires only on TOTAL silence, so the deadline is what stops a quiet link)."""
    print(f"\n>> {prompt}\n   press Enter to start...", flush=True)
    input()
    if countdown:
        for d in range(countdown, 0, -1):
            print(f"   starting in {d}s...", end="\r")
            time.sleep(1)
        print()
    print("   [CAPTURING] hold the condition steady (stand STILL)...")

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


def _emit(cap, root, cal_root, nid, sess_id, subject, carry, cond, weapon, bg_subtract=False):
    """Window every link of node `nid` from one captured condition and persist a labeled dataset per
    link (class 1 if `weapon` else 0). Returns the list of dataset dirs written."""
    out = []
    for key in sorted(k for k in cap if k[1] == nid):
        fr = resampleUniform(cap.get(key, []), TARGET_FS)
        if len(fr) < WINDOW:
            continue
        span = (fr[0].timestamp, fr[-1].timestamp + 1.0)
        tag = key[0].replace(":", "")  # tx mac short, ':'-free for a path segment
        rec = f"{root}/weapon_rec/{sess_id}/{cond}/node{nid}/link_{tag}"
        ds = f"{root}/weapon_ds/node{nid}/{sess_id}_{cond}_link{tag}"
        saveRecording(fr, rec)
        # spans=[span] -> class 1 over the segment, spans=[] -> class 0; bg_subtract nulls σ²[p]'s quiet-room channel (Item 10/CAUSE 2B)
        collectSource(RecordingSource(rec), f"{cal_root}/node{nid}", ds,
                       [span] if weapon else [],
                       stage="weapon", session_id=sess_id, subject_id=subject,
                       subtract_ic_baseline=bg_subtract)
        out.append(ds)
    return out


def _linkTag(ds_dir):
    """TX link tag from a weapon_ds dir name `<sess>_<cond>_link<tag>` (':'-free tx mac-short)."""
    base = os.path.basename(ds_dir)
    return base.split("_link")[-1] if "_link" in base else None


def _trainAndReport(ds_dirs, out_dir, label):
    """Train one weapon head (ic27) on `ds_dirs` -> `out_dir`, print its LOGO line, return True on
    success. Shared by the per-node and per-link paths so both report identically. `label` names the
    unit in the log (e.g. 'Node 2' or 'Node 2 link4f9c')."""
    if not ds_dirs:
        print(f"   [SKIP] {label}: no datasets in pool.")
        return False
    try:
        _, m = trainWeapon(ds_dirs, out_dir=out_dir, feature_mode="ic27")
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
    carry = logo.get("carry")  # confound axis: generalization across carry pose (diagnosis 5E)
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
                        help="Subtract the quiet-room baseline from σ²[p] (Item 10/CAUSE 2B); serving mirrors it. "
                             "Default ON (matches the web frontend); pass --no-bg-subtract to disable.")
    parser.add_argument("--per-link", action="store_true", dest="per_link",
                        help="Train ONE head per (tx->rx) DIRECTION -> model_weapon/node<id>/link<tag>/ "
                             "instead of pooling a node's directions into one head (WEAPON_NLOS_PLAN §4). "
                             "The signal is per-direction; pooling sign-flips the good NLOS link. "
                             "Bad directions are NOT dropped — run_weapon's LinkVoter zeroes a "
                             "sub-chance link via its own LOGO weight (accuracyWeights).")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Weapon model root (default: <root>/model_weapon)")
    args = parser.parse_args()
    if args.cal is None:
        args.cal = f"{args.root}/cal"
    if args.model is None:
        args.model = f"{args.root}/model_weapon"

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
    print(f"training nodes: {calNodes}  (subject={args.subject}, carry={args.carry})")

    os.makedirs(args.model, exist_ok=True)
    for i in range(args.sessions):
        sessId = f"{args.subject}_{args.carry}_s{i}"
        clear = captureLinks(f"session {i+1}/{args.sessions} — stand still, no weapon on you.",
                              args.frames, args.port, calNodes, countdown=5)
        armed = captureLinks(f"session {i+1}/{args.sessions} — stand still, weapon concealed on you.",
                              args.frames, args.port, calNodes, countdown=5)
        for nid in calNodes:
            _emit(clear, args.root, args.cal, nid, sessId, args.subject, args.carry, "clear",
                  weapon=False, bg_subtract=args.bg_subtract)
            _emit(armed, args.root, args.cal, nid, sessId, args.subject, args.carry, "weapon",
                  weapon=True, bg_subtract=args.bg_subtract)

    unit = "per-link (tx->rx)" if args.per_link else "per-node"
    print(f"\ntraining {unit} weapon models (ic27) on the cumulative pool...")
    trained = []
    for nid in calNodes:
        dsDirs = sorted(glob.glob(f"{args.root}/weapon_ds/node{nid}/*"))
        if not args.per_link:
            if _trainAndReport(dsDirs, f"{args.model}/node{nid}", f"Node {nid}"):
                trained.append(nid)
            continue
        # per-link: group this node's datasets by tx tag, one head per (tx->rx) direction
        byTag = collections.defaultdict(list)
        for d in dsDirs:
            tag = _linkTag(d)
            if tag is not None:
                byTag[tag].append(d)
        if not byTag:
            print(f"   [SKIP] node {nid}: no per-link datasets, re-capture with collect_weapon.")
            continue
        for tag in sorted(byTag):
            if _trainAndReport(byTag[tag], f"{args.model}/node{nid}/link{tag}",
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
