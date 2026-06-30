"""Camera-supervised collection for the WHOLE mesh — capture every ESP's CSI + the MacBook webcam
together, run YOLO ONLINE (live) to label each frame, and build BOTH datasets from one pass:

  * PRESENCE  — per-RX-node (person -> present/absent), feeds the per-node presence heads + vote.
  * HEATMAP   — all nodes stacked as channels (n, N, K, W) + the camera's occupancy "where" mask,
                feeds the camera-supervised HeatmapHead (occupancy grid).

One camera supervises the entire test: its labels are time-aligned to every node's CSI windows.

    .venv/bin/python scripts/collect_camera.py --duration 30 --train          # presence + heatmap, then train both
    .venv/bin/python scripts/collect_camera.py --stage weapon --duration 30   # open-carry weapon + weapon "where"

Training-only: the deployed detector needs no camera. HONEST SCOPE: a camera can't see a CONCEALED
weapon (use collect_weapon.py for that) and stock COCO has no firearm class (knife=43; pass a custom
--weights for guns). Presence + occupancy heatmap is the solid, immediate win.
"""

import argparse
import collections
import glob
import os
import socket
import threading
import time

import numpy as np

from wavetrace.Source import (parse_batch_links, resample_uniform, bind_udp, save_recording,
                              RecordingSource)
from wavetrace.Calibration import load_calibration
from wavetrace.Cli import collect_source
from wavetrace.Config import ModelConfig
from wavetrace.groundtruth.DatasetBuilder import build_dataset_stacked, save_dataset
from wavetrace.recognition import train_presence
from wavetrace.groundtruth.CameraLabeler import YoloSegLabeler, presence_label_fn, weapon_label_fn
from wavetrace.groundtruth.Webcam import (WebcamCapture, record_labels_online,
                                          COCO_WEAPON_CLASSES)

TARGET_FS = 100.0   # resample grid; matches the other collectors so train/serve windows align
WINDOW = 128


def capture_csi(duration_s, port, node_ids):
    """Drain CSI for `duration_s`, bucketing frames by RX node (every tx link merged into its node).
    Frames keep node_id + wall-clock timestamps so they stack/align. Returns {rx_node: [frames]}."""
    per_node = collections.defaultdict(list)
    sock = bind_udp(port, timeout=1.0)
    t_end = time.monotonic() + duration_s
    try:
        while time.monotonic() < t_end:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            for (_tx, rx), frames in parse_batch_links(payload).items():
                if not node_ids or rx in node_ids:
                    per_node[rx].extend(frames)
    finally:
        sock.close()
    return dict(per_node)


def main():
    p = argparse.ArgumentParser(description="Camera-supervised mesh collection (webcam + YOLO -> presence + heatmap).")
    p.add_argument("--stage", choices=["presence", "weapon"], default="presence",
                   help="presence (person) or weapon (open-carry, visible). Default presence.")
    p.add_argument("--duration", type=float, default=30.0, help="Seconds to capture (default: 30)")
    p.add_argument("--fps", type=float, default=15.0, help="Live label rate (default 15; < align tol 0.05 s)")
    p.add_argument("--grid", type=int, default=16, help="Occupancy heatmap resolution G (G×G, default 16)")
    p.add_argument("--cam-index", type=int, default=0, help="Webcam index (0 = MacBook FaceTime cam)")
    p.add_argument("--weights", default=None, help="YOLO-seg weights (default yolov8n-seg.pt; custom weapon model here)")
    p.add_argument("--weapon-class", type=int, action="append", dest="weapon_classes",
                   help="COCO class id to treat as a visible weapon; repeatable. Default knife (43)")
    p.add_argument("--conf", type=float, default=0.35, help="Detector confidence floor")
    p.add_argument("--subject", default="cam", help="Subject id for LOGO grouping")
    p.add_argument("--port", type=int, default=9876)
    p.add_argument("--root", default="data/2g4_ht40", help="Capture-profile root (default: data/2g4_ht40)")
    p.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    p.add_argument("--model", default=None, help="Presence model root (default: <root>/model)")
    p.add_argument("--train", action="store_true", help="Train presence (per node) + heatmap after capture")
    args = p.parse_args()
    args.cal = args.cal or f"{args.root}/cal"
    args.model = args.model or f"{args.root}/model"
    weapon_classes = tuple(args.weapon_classes) if args.weapon_classes else COCO_WEAPON_CLASSES

    calibs = {}
    for d in sorted(glob.glob(f"{args.cal}/node*")):
        base = os.path.basename(d)
        if base[4:].isdigit():
            calibs[int(base[4:])] = load_calibration(d)
    if not calibs:
        print(f"[ERROR] no calibrations in {args.cal}/node* — run collect_baseline.py first.")
        return
    cal_nodes = sorted(calibs)
    print(f"Nodes: {cal_nodes}.  Loading YOLO-seg (first run downloads weights)...")

    label_fn = weapon_label_fn if args.stage == "weapon" else presence_label_fn
    labeler = YoloSegLabeler(args.weights or "yolov8n-seg.pt", weapon_classes=weapon_classes,
                             conf=args.conf, grid=args.grid, label_fn=label_fn)

    # ----- capture: webcam (online YOLO) in a thread WHILE the main thread drains all nodes' CSI ----
    pos = {"n": 0, "tot": 0, "last": time.monotonic()}
    def on_label(lab):
        pos["tot"] += 1
        pos["n"] += int(lab.class_id == 1)
        now = time.monotonic()
        if now - pos["last"] >= 1.0:
            print(f"   live: {pos['n']}/{pos['tot']} frames "
                  f"{'weapon' if args.stage=='weapon' else 'present'}", end="\r")
            pos["last"] = now

    box = {}
    def cam_worker():
        try:
            with WebcamCapture(index=args.cam_index) as cap:
                box["labels"] = record_labels_online(cap.read, labeler, args.duration,
                                                     fps=args.fps, on_label=on_label)
        except Exception as e:  # camera permission / busy — report after join
            box["error"] = e

    print(f"\n>> Capturing {args.duration:g}s — subject in the camera's view AND the mesh zone. Press Enter...")
    input()
    th = threading.Thread(target=cam_worker, daemon=True)
    th.start()
    csi = capture_csi(args.duration, args.port, cal_nodes)
    th.join()
    print()
    if "error" in box:
        print(f"[ERROR] webcam: {box['error']}")
        return
    labels = box.get("labels", [])
    if not labels:
        print("[ERROR] no webcam frames labeled (camera permission? --cam-index?).")
        return
    n_pos = sum(l.class_id == 1 for l in labels)
    print(f"Labeled {n_pos}/{len(labels)} frames positive "
          f"({'weapon' if args.stage=='weapon' else 'present'}).")

    # resample each node once (uniform grid) + keep node_id; reused by both dataset builds.
    res = {}
    for nid, frs in csi.items():
        rf = resample_uniform(frs, TARGET_FS)
        for f in rf:
            f.node_id = nid
        res[nid] = rf

    sess = f"{args.subject}_cam_s0"
    # ----- 1) per-node PRESENCE/weapon datasets (class label only) ---------------------------------
    pres_built = []
    for nid in cal_nodes:
        fr = res.get(nid, [])
        if len(fr) < WINDOW:
            print(f"   [SKIP presence] node {nid}: {len(fr)} frames (< {WINDOW}).")
            continue
        rec = f"{args.root}/cam_rec/{sess}/node{nid}"
        ds = f"{args.root}/cam_ds/{args.stage}/node{nid}/{sess}"
        save_recording(fr, rec)
        collect_source(RecordingSource(rec), f"{args.cal}/node{nid}", ds, [], stage=args.stage,
                       labeler=labels, session_id=sess, subject_id=args.subject,
                       subtract_ic_baseline=(args.stage == "weapon"))
        pres_built.append(nid)
        print(f"   [OK presence] node {nid} -> {ds}")

    # ----- 2) all-node STACKED heatmap dataset (occupancy "where" mask) ----------------------------
    merged = [f for nid in cal_nodes for f in res.get(nid, [])]
    hm_dir = f"{args.root}/cam_ds/heatmap/{sess}"
    hm_ds = None
    if merged:
        hm_ds = build_dataset_stacked(merged, calibs, labels, window=WINDOW, hop=32,
                                      session_id=sess, subject_id=args.subject)
        save_dataset(hm_ds, hm_dir)
        n_mask = sum(1 for lb in hm_ds.labels if lb.mask)
        print(f"   [OK heatmap]  stacked {len(cal_nodes)} nodes -> {hm_dir} "
              f"({hm_ds.X_image.shape[0]} windows, {n_mask} with masks)")

    if not pres_built and hm_ds is None:
        print("[ERROR] no usable CSI — is the mesh streaming on this port?")
        return

    # ----- optional training -----------------------------------------------------------------------
    if args.train:
        print("\nTraining...")
        for nid in pres_built:
            dirs = sorted(glob.glob(f"{args.root}/cam_ds/{args.stage}/node{nid}/*"))
            if dirs:
                train_presence(dirs, out_dir=f"{args.model}/node{nid}")
                print(f"   [OK] presence node {nid} -> {args.model}/node{nid}")
        if hm_ds is not None:
            _train_heatmap(hm_dir, f"{args.model}/heatmap.joblib", args.grid)

    print(f"\nDone. presence nodes {pres_built}; heatmap {'built' if hm_ds is not None else 'skipped'}.")


def _train_heatmap(dataset_dir, out_path, grid):
    """Train the camera-supervised occupancy HeatmapHead from a stacked dataset's Label.masks."""
    from wavetrace.groundtruth import load_dataset
    from wavetrace.recognition.Heatmap import HeatmapHead
    ds = load_dataset(dataset_dir)
    masks = [lb.mask for lb in ds.labels if lb.mask]
    if not masks:
        print("   [SKIP] heatmap: no masks (need a person/weapon visible to the camera).")
        return
    Y = np.asarray(masks, dtype=np.float32)
    cfg = ModelConfig(stage="presence", k=int(ds.meta["K"]))
    HeatmapHead(cfg, grid=grid).fit(ds.X_image[:len(masks)], Y).save(out_path)
    print(f"   [OK] heatmap ({grid}x{grid}, {len(masks)} masks) -> {out_path}")

    print('\a', end='', flush=True)

if __name__ == "__main__":
    main()
