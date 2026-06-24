# `wavetrace/` — Python library

The Python side of the pipeline: reads CSI frames, runs calibration, builds datasets, trains models, and serves live predictions. The heavy signal processing happens in `src/` (C++); this package orchestrates it.

## Main files

| File | What it does |
|---|---|
| `Source.py` | CSI frame sources: live UDP, recorded file replay, or synthetic (for tests) |
| `Frontend.py` | Shared pipeline loop — pulls from a source, preprocesses, emits windowed feature vectors or spectrograms |
| `Calibration.py` | Saves and loads per-session calibration: gain-lock scalar, quiet baseline, NBVI subcarrier mask |
| `Cli.py` | `wavetrace` CLI: `calibrate`, `collect-data`, `train`, `run` subcommands |
| `Config.py` | Runtime configuration: mode (`presence`/`weapon`), backend, head, subcarrier count, window/hop |

## Subfolders

### `recognition/`

All machine learning: training, evaluation, inference, voting, and fusion.

| File | What it does |
|---|---|
| `Train.py` | Trains a presence or weapon head from a dataset; saves `model.joblib` |
| `Evaluate.py` | Leave-one-session-out and leave-one-subject-out cross-validation |
| `Infer.py` | Runs a trained model on a feature window; returns class + confidence |
| `Model.py` | Head definitions: `PresenceHead` (MLP/SVM) and `WeaponHead` (variance threshold / sklearn / CNN) |
| `Weapon.py` | Weapon-specific feature logic and the σ²[p] variance baseline |
| `Vote.py` | `SegmentVoter` — soft majority vote over an active motion segment for a stable verdict |
| `Fusion.py` | `fuse()` — concatenates per-node feature vectors for multi-node inference |
| `Link.py` | Per-link bookkeeping (each (tx, rx) pair has its own model and calibration) |
| `Resample.py` | Uniform-grid resampling to handle timing jitter between nodes |
| `Stack.py` | Stacks per-node windows into a single tensor for the CNN path |
| `Cir.py` | Offline CIR (channel impulse response) super-resolution via L1/ISTA |
| `Heatmap.py` | Camera-supervised occupancy heatmap head (built; weapon-heatmap head is a future step) |
| `Adapt.py` | Session adaptation utilities |
| `Explain.py` | Feature importance and model introspection |
| `Occupancy.py` | Occupancy-specific model helpers |

### `groundtruth/`

Used during data collection to produce labels for training.

| File | What it does |
|---|---|
| `CameraLabeler.py` | Reads an MJPEG camera stream and runs a vision model (YOLO/SAM) to produce bounding-box or segmentation labels |
| `DatasetBuilder.py` | Pairs CSI windows with labels and serializes them to `.npy` + `meta.json` |
| `Align.py` | Aligns camera timestamps to CSI timestamps; measures residual clock skew |

### `output/`

Pushes final predictions out of the pipeline.

| File | What it does |
|---|---|
| `Publisher.py` | `Publisher` ABC + `JsonlPublisher` (default, zero dependencies) + WebSocket seam |
| `Guard.py` | Rate-limits and deduplicates output events |

### `diagnostics/`

| File | What it does |
|---|---|
| `Telemetry.py` | Per-node health metrics (frames/s, heap, uptime) consumed by `health_monitor.py` |
