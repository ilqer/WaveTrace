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

from wavetrace.adapters.source import parse_batch_by_link, bind_udp_socket, save_recording, RecordingSource
from wavetrace.adapters.store.calibration import load_calibration
from wavetrace.adapters.store.dataset import save_dataset
from wavetrace.pipeline import collect_source
from wavetrace.domain.contracts import (DEFAULT_HOP_FRAMES, DEFAULT_TARGET_SAMPLE_RATE_HZ,
                                        DEFAULT_WINDOW_FRAMES)
from wavetrace.domain.dsp.resample import resample_uniform
from wavetrace.domain.groundtruth.dataset import build_stacked_dataset
from wavetrace.domain.groundtruth.labeler import presence_label_function, weapon_label_function
from wavetrace.groundtruth.camera_labeler import YoloLabelerOptions, YoloSegLabeler
from wavetrace.groundtruth.webcam import (WebcamCapture, WebcamOptions, record_labels_online,
                                          COCO_WEAPON_CLASSES)
from wavetrace.recognition import train_presence


def capture_csi(duration_s, port, node_ids):
    """Every tx link is merged into its rx node: {rx_node: [frames]}."""
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
    parser = argparse.ArgumentParser(description="Camera-supervised mesh collection (webcam + YOLO -> presence + heatmap).")
    parser.add_argument("--stage", choices=["presence", "weapon"], default="presence",
                        help="presence (person) or weapon (open-carry, visible). Default presence.")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to capture (default: 30)")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Live label rate (default 15; < align tol 0.05 s)")
    parser.add_argument("--grid", type=int, default=16,
                        help="Occupancy heatmap resolution G (G×G, default 16)")
    parser.add_argument("--cam-index", type=int, default=0, help="Webcam index (0 = MacBook FaceTime cam)")
    parser.add_argument("--weights", default=None, help="YOLO-seg weights (default yolov8n-seg.pt; custom weapon model here)")
    parser.add_argument("--weapon-class", type=int, action="append", dest="weapon_classes",
                        help="COCO class id to treat as a visible weapon; repeatable. Default knife (43)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detector confidence floor")
    parser.add_argument("--subject", default="cam", help="Subject id for LOGO grouping")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--root", default="data/2g4_ht40",
                        help="Capture-profile root (default: data/2g4_ht40)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Presence model root (default: <root>/model)")
    parser.add_argument("--train", action="store_true",
                        help="Train presence (per node) + heatmap after capture")
    args = parser.parse_args()
    args.cal = args.cal or f"{args.root}/cal"
    args.model = args.model or f"{args.root}/model"
    weapon_classes = tuple(args.weapon_classes) if args.weapon_classes else COCO_WEAPON_CLASSES

    calibrations = {}
    for node_dir in sorted(glob.glob(f"{args.cal}/node*")):
        node_dir_name = os.path.basename(node_dir)
        if node_dir_name[4:].isdigit():
            calibrations[int(node_dir_name[4:])] = load_calibration(node_dir)
    if not calibrations:
        print(f"[ERROR] no calibrations in {args.cal}/node* — run collect_baseline.py first.")
        return
    calibrated_node_ids = sorted(calibrations)
    print(f"nodes: {calibrated_node_ids}. loading YOLO-seg (first run downloads weights)...")

    label_function = weapon_label_function if args.stage == "weapon" else presence_label_function
    labeler = YoloSegLabeler(
        options=YoloLabelerOptions(weights_path=args.weights, weapon_classes=weapon_classes),
        confidence_threshold=args.conf, grid=args.grid, label_function=label_function)

    live_progress = {"n": 0, "tot": 0, "last": time.monotonic()}
    def on_label(frame_label):
        live_progress["tot"] += 1
        live_progress["n"] += int(frame_label.class_id == 1)
        now = time.monotonic()
        if now - live_progress["last"] >= 1.0:
            print(f"   live: {live_progress['n']}/{live_progress['tot']} frames "
                  f"{'weapon' if args.stage=='weapon' else 'present'}", end="\r")
            live_progress["last"] = now

    camera_result = {}
    def run_camera_labeler():
        try:
            with WebcamCapture(WebcamOptions(index=args.cam_index)) as webcam:
                camera_result["labels"] = record_labels_online(webcam.read, labeler, args.duration,
                                                     frame_rate_hz=args.fps, on_label=on_label)
        except Exception as error:  # camera permission or busy; reported after the join
            camera_result["error"] = error

    print(f"\n>> capturing {args.duration:g}s — keep the subject in camera view and the mesh zone. Press Enter...")
    input()
    camera_thread = threading.Thread(target=run_camera_labeler, daemon=True)
    camera_thread.start()
    csi = capture_csi(args.duration, args.port, calibrated_node_ids)
    camera_thread.join()
    print()
    if "error" in camera_result:
        print(f"[ERROR] webcam: {camera_result['error']}")
        return
    labels = camera_result.get("labels", [])
    if not labels:
        print("[ERROR] no webcam frames labeled — check camera permission and --cam-index.")
        return
    positive_count = sum(label.class_id == 1 for label in labels)
    print(f"labeled {positive_count}/{len(labels)} frames positive "
          f"({'weapon' if args.stage=='weapon' else 'present'}).")

    resampled_by_node = {}
    for node_id, frames in csi.items():
        resampled_frames = resample_uniform(frames, DEFAULT_TARGET_SAMPLE_RATE_HZ)
        for frame in resampled_frames:
            frame.node_id = node_id
        resampled_by_node[node_id] = resampled_frames

    session_id = f"{args.subject}_cam_s0"
    presence_node_ids = []
    for node_id in calibrated_node_ids:
        node_frames = resampled_by_node.get(node_id, [])
        if len(node_frames) < DEFAULT_WINDOW_FRAMES:
            print(f"   [SKIP presence] node {node_id}: {len(node_frames)} frames (< {DEFAULT_WINDOW_FRAMES})")
            continue
        recording_dir = f"{args.root}/cam_rec/{session_id}/node{node_id}"
        dataset_dir = f"{args.root}/cam_ds/{args.stage}/node{node_id}/{session_id}"
        save_recording(node_frames, recording_dir)
        collect_source(RecordingSource(recording_dir), f"{args.cal}/node{node_id}", dataset_dir, [],
                       stage=args.stage,
                       labeler=labels, session_id=session_id, subject_id=args.subject,
                       subtract_ic_baseline=(args.stage == "weapon"))
        presence_node_ids.append(node_id)
        print(f"   [OK presence] node {node_id} -> {dataset_dir}")

    merged_frames = [frame for node_id in calibrated_node_ids for frame in resampled_by_node.get(node_id, [])]
    heatmap_dir = f"{args.root}/cam_ds/heatmap/{session_id}"
    heatmap_dataset = None
    if merged_frames:
        heatmap_dataset = build_stacked_dataset(merged_frames, calibrations, labels,
                                                window=DEFAULT_WINDOW_FRAMES,
                                                hop=DEFAULT_HOP_FRAMES,
                                                session_id=session_id, subject_id=args.subject)
        save_dataset(heatmap_dataset, heatmap_dir)
        masked_window_count = sum(1 for window_label in heatmap_dataset.labels if window_label.mask)
        print(f"   [OK heatmap]  stacked {len(calibrated_node_ids)} nodes -> {heatmap_dir} "
              f"({heatmap_dataset.X_image.shape[0]} windows, {masked_window_count} with masks)")

    if not presence_node_ids and heatmap_dataset is None:
        print("[ERROR] no usable CSI — is the mesh streaming on this port?")
        return

    if args.train:
        print("\ntraining...")
        for node_id in presence_node_ids:
            dirs = sorted(glob.glob(f"{args.root}/cam_ds/{args.stage}/node{node_id}/*"))
            if dirs:
                train_presence(dirs, out_dir=f"{args.model}/node{node_id}")
                print(f"   [OK] presence node {node_id} -> {args.model}/node{node_id}")
        if heatmap_dataset is not None:
            _train_heatmap(heatmap_dir, f"{args.model}/heatmap.joblib")

    print(f"\ndone. presence nodes {presence_node_ids}; "
          f"heatmap {'built' if heatmap_dataset is not None else 'skipped'}.")


def _train_heatmap(dataset_dir, out_path):
    from wavetrace.adapters.store.dataset import load_dataset
    from wavetrace.recognition.heatmap import train_heatmap
    dataset = load_dataset(dataset_dir)
    try:
        head, masked_window_count = train_heatmap(dataset)
    except ValueError as error:
        print(f"   [SKIP] heatmap: {error}")
        return
    head.save(out_path)
    print(f"   [OK] heatmap ({head.grid}x{head.grid}, {masked_window_count} masks) -> {out_path}")

    print('\a', end='', flush=True)

if __name__ == "__main__":
    main()
