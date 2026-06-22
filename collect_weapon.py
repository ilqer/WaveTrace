"""Independent WEAPON pipeline — capture labeled no-weapon/weapon sessions over UDP and train a
per-node Stage-E weapon model. Standalone from the presence/count scripts: imports only library code
(wavetrace.*), writes to its own data/model_weapon root, and reuses the shared calibration (data/cal).

Stage-E uses the INTER-CARRIER feature block (stage="weapon" -> intercarrier dataset, train_weapon
feature_mode="ic27"), NOT the amplitude features the presence head uses — the concealed-object signal
lives in inter-subcarrier structure, not in "is a body moving". Per-(tx->rx)-LINK + TARGET_FS resample,
pooled into each node's single head (same parity as the presence/count paths).

CUMULATIVE: every run appends its datasets under data/weapon_ds/ and RETRAINS each node on the whole
pool, so you build subject/position diversity over many runs. Run once per subject AND carry position:

    .venv/bin/python collect_weapon.py --subject p1 --carry waist --sessions 3
    .venv/bin/python collect_weapon.py --subject p2 --carry chest --sessions 3   # adds to the pool

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

from wavetrace.Source import RecordingSource, save_recording, parse_batch_links, resample_uniform
from wavetrace.Cli import collect_source
from wavetrace.recognition import train_weapon

TARGET_FS = 100.0   # resample grid; MUST match run_weapon.TARGET_FS so train and serve windows align
WINDOW = 128        # front-end window (frames); a link shorter than this on the grid emits no window
DS_ROOT = "data/weapon_ds"   # cumulative dataset pool (one subdir tree per run), globbed at train time


def capture_links(prompt, n, port, node_ids, countdown=0, max_capture_s=60.0):
    """Collect up to n frames PER (tx->rx) LINK in ONE pass. Returns {(tx_short, rx_node): [frames]}.

    Per-link so training matches per-link serving (run_weapon). Keeps the dominant subcarrier width per
    link. Stops when every expected RX node has appeared AND all known links reach n, OR max_capture_s
    elapses (the recv timeout fires only on TOTAL silence, so the deadline is what stops a quiet link)."""
    input(f"\n>> {prompt}\n   Press Enter to start...")
    if countdown:
        for d in range(countdown, 0, -1):
            print(f"   starting in {d}s...", end="\r")
            time.sleep(1)
        print()
    print("   [CAPTURING] hold the condition steady (stand STILL)...")

    links = collections.defaultdict(list)  # (tx_short, rx_node) -> [frames]
    want = set(node_ids)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(15.0)
    start = time.time()
    last_print = start
    try:
        while True:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                break  # total silence
            for key, frames in parse_batch_links(payload).items():
                buf = links[key]
                if len(buf) < n:
                    buf.extend(frames[: n - len(buf)])
            now = time.time()
            if links and want <= {k[1] for k in links} and all(len(b) >= n for b in links.values()):
                break
            if now - start > max_capture_s:
                short = [k for k, v in sorted(links.items()) if len(v) < n]
                print(f"\n[WARN] capture deadline {max_capture_s:g}s hit; links short of {n}: {short}")
                break
            if now - last_print >= 1.0:
                per_node = dict(sorted(collections.Counter(k[1] for k in links).items()))
                mn = min((len(v) for v in links.values()), default=0)
                print(f"   links/node {per_node}  min {mn}/{n}...", end="\r")
                last_print = now
    finally:
        sock.close()
    print()
    for key in links:  # dominant subcarrier width per link (widths can differ across boards/bands)
        if links[key]:
            S = collections.Counter(f.num_subcarriers for f in links[key]).most_common(1)[0][0]
            links[key] = [f for f in links[key] if f.num_subcarriers == S]
    return dict(links)


def _emit(cap, cal_root, nid, sess_id, subject, carry, cond, weapon):
    """Window every link of node `nid` from one captured condition and persist a labeled dataset per
    link (class 1 if `weapon` else 0). Returns the list of dataset dirs written."""
    out = []
    for key in sorted(k for k in cap if k[1] == nid):
        fr = resample_uniform(cap.get(key, []), TARGET_FS)
        if len(fr) < WINDOW:
            continue
        span = (fr[0].timestamp, fr[-1].timestamp + 1.0)
        tag = key[0].replace(":", "")  # tx mac-short, ':'-free for a path segment
        rec = f"data/weapon_rec/{sess_id}/{cond}/node{nid}/link_{tag}"
        ds = f"{DS_ROOT}/node{nid}/{sess_id}_{cond}_link{tag}"
        save_recording(fr, rec)
        # stage="weapon" -> intercarrier dataset + weapon_label_fn; spans=[span] -> class 1 over the
        # whole segment, spans=[] -> class 0. Same session_id for both conditions so LOGO folds cleanly.
        collect_source(RecordingSource(rec), f"{cal_root}/node{nid}", ds,
                       [span] if weapon else [],
                       stage="weapon", session_id=sess_id, subject_id=subject)
        out.append(ds)
    return out


def main():
    parser = argparse.ArgumentParser(description="Capture no-weapon/weapon sessions and train a weapon model per node.")
    parser.add_argument("--node", type=int, default=None, help="Train ONLY this node (default: all calibrated)")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--sessions", type=int, default=3, help="Sessions to capture THIS run (default: 3)")
    parser.add_argument("--frames", type=int, default=1500, help="Frames per condition per node (default: 1500)")
    parser.add_argument("--subject", default="p0", help="Subject id for this run (vary it across people!)")
    parser.add_argument("--carry", default="na", help="Carry position label, e.g. waist/chest/ankle (default: na)")
    parser.add_argument("--cal", default="data/cal", help="Calibration root (default: data/cal)")
    parser.add_argument("--model", default="data/model_weapon", help="Weapon model root (default: data/model_weapon)")
    args = parser.parse_args()

    cal_nodes = sorted(int(os.path.basename(d)[len("node"):])
                       for d in glob.glob(os.path.join(args.cal, "node*"))
                       if os.path.basename(d)[len("node"):].isdigit())
    if not cal_nodes:
        print(f"\n[ERROR] No per-node calibrations in {args.cal}/node*. Run collect_baseline.py first.",
              file=sys.stderr)
        return
    if args.node is not None:
        cal_nodes = [args.node] if args.node in cal_nodes else []
        if not cal_nodes:
            print(f"\n[ERROR] --node {args.node} has no calibration in {args.cal}.", file=sys.stderr)
            return
    print(f"Will train nodes: {cal_nodes}  (subject={args.subject}, carry={args.carry})")

    os.makedirs(args.model, exist_ok=True)
    for i in range(args.sessions):
        sess_id = f"{args.subject}_{args.carry}_s{i}"
        clear = capture_links(f"Session {i+1}/{args.sessions} — stand STILL, NO weapon on you.",
                              args.frames, args.port, cal_nodes, countdown=5)
        armed = capture_links(f"Session {i+1}/{args.sessions} — stand STILL, weapon concealed on you.",
                              args.frames, args.port, cal_nodes, countdown=5)
        for nid in cal_nodes:
            _emit(clear, args.cal, nid, sess_id, args.subject, args.carry, "clear", weapon=False)
            _emit(armed, args.cal, nid, sess_id, args.subject, args.carry, "weapon", weapon=True)

    print("\nTraining per-node weapon models (ic27) on the CUMULATIVE pool...")
    trained = []
    for nid in cal_nodes:
        ds_dirs = sorted(glob.glob(f"{DS_ROOT}/node{nid}/*"))
        if not ds_dirs:
            print(f"   [SKIP] Node {nid}: no datasets in pool.")
            continue
        try:
            _, m = train_weapon(ds_dirs, out_dir=f"{args.model}/node{nid}", feature_mode="ic27")
        except ValueError as e:  # WeaponHead.fit needs BOTH classes in the pool
            print(f"   [SKIP] Node {nid}: {e}")
            continue
        logo = m.get("logo", {})
        sess = logo.get("session") or logo.get("subject")
        line = (f"   [OK]   Node {nid}: samples={m['n_samples']} class_counts={m['class_counts']} "
                f"train_acc={m['train_accuracy']:.3f}")
        if sess:
            line += (f"  LOGO={sess['accuracy']:.3f} (majority {sess['majority_accuracy']:.3f}, "
                     f"TPR {sess.get('tpr', 0):.3f}, FP {sess.get('fp_rate', 0):.3f})")
        print(line)
        trained.append(nid)

    if not trained:
        print("\n[ERROR] No node trained — need both no-weapon AND weapon captures in the pool.",
              file=sys.stderr)
        return
    print(f"\nweapon models saved for nodes {trained} -> {args.model}/node*/  "
          "(LOGO must clearly beat majority; expect this to need many subjects/positions/objects.)")


if __name__ == "__main__":
    main()
