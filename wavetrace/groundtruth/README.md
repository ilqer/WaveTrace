# `wavetrace/groundtruth/`

Tools for producing labels during data collection. A label tells the model what was actually happening in the room at a given moment so it has something to learn from.

## Files

| File | What it does |
|---|---|
| `camera_labeler.py` | Reads an MJPEG stream from the Pi camera and runs a YOLO or SAM vision model to produce bounding-box or segmentation labels at each frame |
| `webcam.py` | OpenCV (e.g. MacBook) webcam frame source whose timestamps share the CSI wall clock; `record_frames` buffers frames during capture, `stream_labels` runs YOLO offline |

Timestamp alignment (`align`, `estimate_clock_offset`) lives in `domain/groundtruth/alignment.py`; the `Labeler` interface, `ScriptedLabeler` and the label functions in `domain/groundtruth/labeler.py`; dataset building in `domain/groundtruth/dataset.py` and its persistence in `adapters/store/dataset.py`.

## Ground-truth tiers (weapon mode)

1. **Open + static weapon** — camera-labeled (the weapon is visible; same as the proven Wi-Fi material-ID setup).
2. **See-through-wrapped + carried** — camera labels through transparent cloth (the label is clean; the radio sees nearly the same as tier 1).
3. **Truly concealed (opaque)** — camera cannot see it → **scripted ground truth**: the weapon was placed by the operator, so the time-span label is known in advance. Pass `--label-spans "start:end"` to `collect-data`.

Each tier is a feasibility gate. Do not move to the next until the current tier beats chance on held-out data.
