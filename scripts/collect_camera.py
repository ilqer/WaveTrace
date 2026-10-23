"""Record every mesh link's CSI while the webcam labels each frame with YOLO-seg, and write one
camera-labelled dataset per (tx->rx) link plus one all-node stacked heatmap dataset ((n, N, K, W) with
the camera's occupancy masks) from the one pass. Each run is the next session, <subject>_cam_s<n>.

The camera is for training only. It cannot see a concealed weapon (use collect_weapon.py for that), and
stock COCO has no firearm class — knife is 43, so a custom --weights is needed for guns. Weapon-stage
datasets join collect_weapon.py's cumulative pool at <root>/weapon_ds.
"""

import argparse
import threading

from wavetrace.adapters.store.calibration import find_node_calibration_dirs
from wavetrace.groundtruth.webcam import COCO_WEAPON_CLASSES, WebcamLabelCapture
from wavetrace.pipeline import collect_camera
from wavetrace.recognition.camera_collect_modes import default_camera_model_root


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
    parser.add_argument("--model", default=None,
                        help="Model root for --train (default: <root>/model_camera for presence, "
                             "<root>/model_weapon for weapon)")
    parser.add_argument("--train", action="store_true",
                        help="After capture, train the stage's per-node heads on their whole pool + the heatmap")
    args = parser.parse_args()
    calibration_root = args.cal or f"{args.root}/cal"
    calibration_dir_by_node_id = find_node_calibration_dirs(calibration_root)
    if not calibration_dir_by_node_id:
        print(f"[ERROR] no calibrations in {calibration_root}/node* — run collect_baseline.py first.")
        return
    print(f"nodes: {sorted(calibration_dir_by_node_id)}. loading YOLO-seg (first run downloads weights)...")
    camera = WebcamLabelCapture(stage=args.stage, camera_index=args.cam_index, weights_path=args.weights,
                                weapon_classes=tuple(args.weapon_classes or COCO_WEAPON_CLASSES),
                                confidence_threshold=args.conf, grid=args.grid, frame_rate_hz=args.fps)
    print(f"\n>> capturing {args.duration:g}s — keep the subject in camera view and the mesh zone. Press Enter...")
    input()
    model_root = (args.model or default_camera_model_root(args.root, args.stage)) if args.train else None
    collect_camera(camera, stage=args.stage, port=args.port, root=args.root,
                   calibration_dir_by_node_id=calibration_dir_by_node_id, subject_id=args.subject,
                   duration_s=args.duration, stop=threading.Event(), model_root=model_root, log=print,
                   on_capture_end=lambda: print("\a", end="", flush=True))


if __name__ == "__main__":
    main()
