# `wavetrace/groundtruth/`

Tools for producing labels during data collection. A label tells the model what was actually happening in the room at a given moment so it has something to learn from.

## Files

| File | What it does |
|---|---|
| `vision_labeler.py` | `VisionLabeler`: labels each camera frame from any object detector (`detector(image) -> list[Detection]`): the highest-confidence person gives present + bbox + keypoints, and a detection whose class is in `LabelerOptions.weapon_classes` flags a weapon; also `Detection` and `LabelerOptions` |
| `segmentation_labeler.py` | `SegmentationLabeler`: labels from a segmentation model (`segmenter(image) -> list[Segment]`); accepts a weapon only when its mask overlaps the person's, and average-pools the supervised mask into a `grid`×`grid` occupancy heatmap for the heatmap head; also `Segment` |
| `yolo_labeler.py` | Ultralytics adapters: `YoloLabeler` (detection, pose keypoints) and `YoloSegLabeler` (segmentation masks), set by `YoloLabelerOptions` (`device`, `weights_path`); ultralytics is imported only when no `model=` is passed |
| `replay_labeler.py` | `ReplayLabeler`: replays pre-computed detections (a synthetic fixture or a recorded vision-model output stream) |
| `location_chip_labeler.py` | `LocationChipLabeler`: labels from a BLE/UWB tag track on the CSI clock (nearest sample in time), with the tag position as a hint |
| `thermal_labeler.py` | `ThermalLabeler`: seam for a thermal camera model; raises `NotImplementedError` until `_detect` is overridden |
| `webcam.py` | ffmpeg (e.g. MacBook) webcam frame source whose timestamps share the CSI wall clock; `record_frames` buffers frames during capture, `stream_labels` runs YOLO offline, `record_labels_online` labels live; `WebcamLabelCapture` is the camera-collect adapter (`CameraLabelCapture` port) both `scripts/collect_camera.py` and the dashboard use |

Timestamp alignment (`align`, `estimate_clock_offset`) lives in `domain/groundtruth/alignment.py`; the `Labeler` interface, `ScriptedLabeler` and the label functions in `domain/groundtruth/labeler.py`; dataset building in `domain/groundtruth/dataset.py` and its persistence in `adapters/store/dataset.py`.

## Ground-truth tiers (weapon mode)

1. **Open + static weapon** — camera-labeled (the weapon is visible; same as the proven Wi-Fi material-ID setup).
2. **See-through-wrapped + carried** — camera labels through transparent cloth (the label is clean; the radio sees nearly the same as tier 1).
3. **Truly concealed (opaque)** — camera cannot see it → **scripted ground truth**: the weapon was placed by the operator, so the time-span label is known in advance. Pass `--label-spans "start:end"` to `collect-data`.

Each tier is a feasibility gate. Do not move to the next until the current tier beats chance on held-out data.
