"""Capture every node's CSI and the webcam together, label each camera frame with YOLO as it
arrives, and build both a per-node presence dataset and an all-node stacked heatmap dataset
((n, N, K, W) plus the camera's occupancy mask) from the one pass.

The camera is for training only. It cannot see a concealed weapon (use collect_weapon.py for that),
and stock COCO has no firearm class — knife is 43, so a custom --weights is needed for guns.
"""

import argparse
import collections
import glob
import os
import socket
import threading
import time

import numpy as np

from wavetrace.adapters.source import parse_batch_by_link, bind_udp_socket, save_recording, RecordingSource
from wavetrace.domain.dsp.resample import resample_uniform
from wavetrace.calibration import load_calibration
from wavetrace.application.collect import collect_source
from wavetrace.config import ModelConfig
from wavetrace.groundtruth.dataset_builder import build_stacked_dataset, save_dataset
from wavetrace.recognition import train_presence
from wavetrace.groundtruth.camera_labeler import (YoloLabelerOptions, YoloSegLabeler, presence_label_fn,
                                                 weapon_label_fn)
from wavetrace.groundtruth.webcam import (WebcamCapture, WebcamOptions, record_labels_online,
                                          COCO_WEAPON_CLASSES)
from wavetrace.domain.contracts import (DEFAULT_HOP_FRAMES, DEFAULT_TARGET_SAMPLE_RATE_HZ,
                                        DEFAULT_WINDOW_FRAMES)


def capture_csi(duration_s, port, node_ids):
    """Returns {rx_node: [frames]} for `duration_s` of capture, every tx link merged into its rx node."""
    frames_by_node = collections.defaultdict(list)
    sock = bind_udp_socket(port, timeout_s=1.0)
    end_time_s = time.monotonic() + duration_s
    try:
        while time.monotonic() < end_time_s:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            for (_tx, rx), frames in parse_batch_by_link(payload).items():
                if not node_ids or rx in node_ids:
                    frames_by_node[rx].extend(frames)
    finally:
        sock.close()
    return dict(frames_by_node)


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
    calibrated_node_ids = sorted(calibs)
    print(f"nodes: {calibrated_node_ids}. loading YOLO-seg (first run downloads weights)...")

    label_fn = weapon_label_fn if args.stage == "weapon" else presence_label_fn
    labeler = YoloSegLabeler(
        options=YoloLabelerOptions(weights_path=args.weights, weapon_classes=weapon_classes),
        conf=args.conf, grid=args.grid, label_fn=label_fn)

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
    def run_camera_labeler():
        try:
            with WebcamCapture(WebcamOptions(index=args.cam_index)) as cap:
                box["labels"] = record_labels_online(cap.read, labeler, args.duration,
                                                     fps=args.fps, on_label=on_label)
        except Exception as e:  # camera permission or busy; reported after the join
            box["error"] = e

    print(f"\n>> capturing {args.duration:g}s — keep the subject in camera view and the mesh zone. Press Enter...")
    input()
    th = threading.Thread(target=run_camera_labeler, daemon=True)
    th.start()
    csi = capture_csi(args.duration, args.port, calibrated_node_ids)
    th.join()
    print()
    if "error" in box:
        print(f"[ERROR] webcam: {box['error']}")
        return
    labels = box.get("labels", [])
    if not labels:
        print("[ERROR] no webcam frames labeled — check camera permission and --cam-index.")
        return
    positive_count = sum(l.class_id == 1 for l in labels)
    print(f"labeled {positive_count}/{len(labels)} frames positive "
          f"({'weapon' if args.stage=='weapon' else 'present'}).")

    res = {}
    for nid, frs in csi.items():
        rf = resample_uniform(frs, DEFAULT_TARGET_SAMPLE_RATE_HZ)
        for f in rf:
            f.node_id = nid
        res[nid] = rf

    sess = f"{args.subject}_cam_s0"
    presence_node_ids = []
    for nid in calibrated_node_ids:
        fr = res.get(nid, [])
        if len(fr) < DEFAULT_WINDOW_FRAMES:
            print(f"   [SKIP presence] node {nid}: {len(fr)} frames (< {DEFAULT_WINDOW_FRAMES})")
            continue
        rec = f"{args.root}/cam_rec/{sess}/node{nid}"
        ds = f"{args.root}/cam_ds/{args.stage}/node{nid}/{sess}"
        save_recording(fr, rec)
        collect_source(RecordingSource(rec), f"{args.cal}/node{nid}", ds, [], stage=args.stage,
                       labeler=labels, session_id=sess, subject_id=args.subject,
                       subtract_ic_baseline=(args.stage == "weapon"))
        presence_node_ids.append(nid)
        print(f"   [OK presence] node {nid} -> {ds}")

    merged = [f for nid in calibrated_node_ids for f in res.get(nid, [])]
    heatmap_dir = f"{args.root}/cam_ds/heatmap/{sess}"
    heatmap_dataset = None
    if merged:
        heatmap_dataset = build_stacked_dataset(merged, calibs, labels, window=DEFAULT_WINDOW_FRAMES,
                                                hop=DEFAULT_HOP_FRAMES,
                                                session_id=sess, subject_id=args.subject)
        save_dataset(heatmap_dataset, heatmap_dir)
        masked_window_count = sum(1 for lb in heatmap_dataset.labels if lb.mask)
        print(f"   [OK heatmap]  stacked {len(calibrated_node_ids)} nodes -> {heatmap_dir} "
              f"({heatmap_dataset.X_image.shape[0]} windows, {masked_window_count} with masks)")

    if not presence_node_ids and heatmap_dataset is None:
        print("[ERROR] no usable CSI — is the mesh streaming on this port?")
        return

    if args.train:
        print("\ntraining...")
        for nid in presence_node_ids:
            dirs = sorted(glob.glob(f"{args.root}/cam_ds/{args.stage}/node{nid}/*"))
            if dirs:
                train_presence(dirs, out_dir=f"{args.model}/node{nid}")
                print(f"   [OK] presence node {nid} -> {args.model}/node{nid}")
        if heatmap_dataset is not None:
            _train_heatmap(heatmap_dir, f"{args.model}/heatmap.joblib", args.grid)

    print(f"\ndone. presence nodes {presence_node_ids}; "
          f"heatmap {'built' if heatmap_dataset is not None else 'skipped'}.")


def _train_heatmap(dataset_dir, out_path, grid):
    from wavetrace.groundtruth import load_dataset
    from wavetrace.recognition.heatmap import HeatmapHead
    ds = load_dataset(dataset_dir)
    masks = [lb.mask for lb in ds.labels if lb.mask]
    if not masks:
        print("   [SKIP] heatmap: no masks — need a person/weapon visible to the camera.")
        return
    Y = np.asarray(masks, dtype=np.float32)
    cfg = ModelConfig(stage="presence", k=int(ds.meta["K"]))
    HeatmapHead(cfg, grid=grid).fit(ds.X_image[:len(masks)], Y).save(out_path)
    print(f"   [OK] heatmap ({grid}x{grid}, {len(masks)} masks) -> {out_path}")

    print('\a', end='', flush=True)

if __name__ == "__main__":
    main()
