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

from wavetrace.Source import (parseBatchLinks, resampleUniform, bindUdp, saveRecording,
                              RecordingSource)
from wavetrace.Calibration import loadCalibration
from wavetrace.Cli import collectSource
from wavetrace.Config import ModelConfig
from wavetrace.groundtruth.DatasetBuilder import buildDatasetStacked, saveDataset
from wavetrace.recognition import trainPresence
from wavetrace.groundtruth.CameraLabeler import (YoloLabelerOptions, YoloSegLabeler, presenceLabelFn,
                                                 weaponLabelFn)
from wavetrace.groundtruth.Webcam import (WebcamCapture, WebcamOptions, recordLabelsOnline,
                                          COCO_WEAPON_CLASSES)
from wavetrace.domain.contracts import (DEFAULT_HOP_FRAMES, DEFAULT_TARGET_SAMPLE_RATE_HZ,
                                        DEFAULT_WINDOW_FRAMES)


def captureCsi(duration_s, port, node_ids):
    """Drain CSI for `duration_s`, bucketing frames by RX node (every tx link merged into its node).
    Frames keep node_id + wall-clock timestamps so they stack/align. Returns {rx_node: [frames]}."""
    perNode = collections.defaultdict(list)
    sock = bindUdp(port, timeout=1.0)
    tEnd = time.monotonic() + duration_s
    try:
        while time.monotonic() < tEnd:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            for (_tx, rx), frames in parseBatchLinks(payload).items():
                if not node_ids or rx in node_ids:
                    perNode[rx].extend(frames)
    finally:
        sock.close()
    return dict(perNode)


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
    weaponClasses = tuple(args.weapon_classes) if args.weapon_classes else COCO_WEAPON_CLASSES

    calibs = {}
    for d in sorted(glob.glob(f"{args.cal}/node*")):
        base = os.path.basename(d)
        if base[4:].isdigit():
            calibs[int(base[4:])] = loadCalibration(d)
    if not calibs:
        print(f"[ERROR] no calibrations in {args.cal}/node* — run collect_baseline.py first.")
        return
    calNodes = sorted(calibs)
    print(f"nodes: {calNodes}. loading YOLO-seg (first run downloads weights)...")

    labelFn = weaponLabelFn if args.stage == "weapon" else presenceLabelFn
    labeler = YoloSegLabeler(
        options=YoloLabelerOptions(weights_path=args.weights, weapon_classes=weaponClasses),
        conf=args.conf, grid=args.grid, label_fn=labelFn)

    # capture: webcam (online YOLO) runs in a thread while the main thread drains all nodes' CSI
    pos = {"n": 0, "tot": 0, "last": time.monotonic()}
    def onLabel(lab):
        pos["tot"] += 1
        pos["n"] += int(lab.class_id == 1)
        now = time.monotonic()
        if now - pos["last"] >= 1.0:
            print(f"   live: {pos['n']}/{pos['tot']} frames "
                  f"{'weapon' if args.stage=='weapon' else 'present'}", end="\r")
            pos["last"] = now

    box = {}
    def camWorker():
        try:
            with WebcamCapture(WebcamOptions(index=args.cam_index)) as cap:
                box["labels"] = recordLabelsOnline(cap.read, labeler, args.duration,
                                                     fps=args.fps, onLabel=onLabel)
        except Exception as e:  # camera permission / busy — report after join
            box["error"] = e

    print(f"\n>> capturing {args.duration:g}s — keep the subject in camera view and the mesh zone. Press Enter...")
    input()
    th = threading.Thread(target=camWorker, daemon=True)
    th.start()
    csi = captureCsi(args.duration, args.port, calNodes)
    th.join()
    print()
    if "error" in box:
        print(f"[ERROR] webcam: {box['error']}")
        return
    labels = box.get("labels", [])
    if not labels:
        print("[ERROR] no webcam frames labeled — check camera permission and --cam-index.")
        return
    nPos = sum(l.class_id == 1 for l in labels)
    print(f"labeled {nPos}/{len(labels)} frames positive "
          f"({'weapon' if args.stage=='weapon' else 'present'}).")

    # resample each node once onto a uniform grid, keep node_id, reuse for both dataset builds
    res = {}
    for nid, frs in csi.items():
        rf = resampleUniform(frs, DEFAULT_TARGET_SAMPLE_RATE_HZ)
        for f in rf:
            f.node_id = nid
        res[nid] = rf

    sess = f"{args.subject}_cam_s0"
    # 1) per-node presence/weapon datasets (class label only)
    presBuilt = []
    for nid in calNodes:
        fr = res.get(nid, [])
        if len(fr) < DEFAULT_WINDOW_FRAMES:
            print(f"   [SKIP presence] node {nid}: {len(fr)} frames (< {DEFAULT_WINDOW_FRAMES})")
            continue
        rec = f"{args.root}/cam_rec/{sess}/node{nid}"
        ds = f"{args.root}/cam_ds/{args.stage}/node{nid}/{sess}"
        saveRecording(fr, rec)
        collectSource(RecordingSource(rec), f"{args.cal}/node{nid}", ds, [], stage=args.stage,
                       labeler=labels, session_id=sess, subject_id=args.subject,
                       subtract_ic_baseline=(args.stage == "weapon"))
        presBuilt.append(nid)
        print(f"   [OK presence] node {nid} -> {ds}")

    # 2) all-node stacked heatmap dataset (occupancy "where" mask)
    merged = [f for nid in calNodes for f in res.get(nid, [])]
    hmDir = f"{args.root}/cam_ds/heatmap/{sess}"
    hmDs = None
    if merged:
        hmDs = buildDatasetStacked(merged, calibs, labels, window=DEFAULT_WINDOW_FRAMES,
                                    hop=DEFAULT_HOP_FRAMES,
                                    session_id=sess, subject_id=args.subject)
        saveDataset(hmDs, hmDir)
        nMask = sum(1 for lb in hmDs.labels if lb.mask)
        print(f"   [OK heatmap]  stacked {len(calNodes)} nodes -> {hmDir} "
              f"({hmDs.X_image.shape[0]} windows, {nMask} with masks)")

    if not presBuilt and hmDs is None:
        print("[ERROR] no usable CSI — is the mesh streaming on this port?")
        return

    if args.train:
        print("\ntraining...")
        for nid in presBuilt:
            dirs = sorted(glob.glob(f"{args.root}/cam_ds/{args.stage}/node{nid}/*"))
            if dirs:
                trainPresence(dirs, out_dir=f"{args.model}/node{nid}")
                print(f"   [OK] presence node {nid} -> {args.model}/node{nid}")
        if hmDs is not None:
            _trainHeatmap(hmDir, f"{args.model}/heatmap.joblib", args.grid)

    print(f"\ndone. presence nodes {presBuilt}; heatmap {'built' if hmDs is not None else 'skipped'}.")


def _trainHeatmap(dataset_dir, out_path, grid):
    """Train the camera-supervised occupancy HeatmapHead from a stacked dataset's Label.masks."""
    from wavetrace.groundtruth import loadDataset
    from wavetrace.recognition.Heatmap import HeatmapHead
    ds = loadDataset(dataset_dir)
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
