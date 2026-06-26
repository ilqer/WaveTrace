# `wavetrace/` — Python library

Reads CSI frames from a source, runs calibration, builds labeled datasets, trains models, and serves live predictions. The heavy signal processing is in `src/` (C++); this package orchestrates it.

## Main files

| File | What it does |
|---|---|
| `Source.py` | CSI frame sources: live UDP (`UdpSource`), recorded file replay (`RecordingSource`), synthetic frames for tests (`SyntheticSource`). Also the binary UDP wire format parser and `parse_batch_links`. |
| `Frontend.py` | Shared pipeline loop — pulls from a source, preprocesses, emits windowed feature vectors or spectrograms. Used by both training and serving so the features always match. |
| `Calibration.py` | Saves and loads per-session calibration: gain-lock scalar, quiet baseline, NBVI subcarrier mask. |
| `Cli.py` | `wavetrace` CLI: `capture`, `calibrate`, `collect-data`, `train`, `localize`, `run` subcommands. |
| `Config.py` | Runtime config: mode (`presence`/`weapon`), backend, head, subcarrier count, window/hop. |
| `Localize.py` | AoA spatial heatmap (parked — needs ≥ 2 phase-coherent antennas; ESP32-S3 has one receive chain). |

## Subfolders

### `recognition/`

All machine learning: training, evaluation, inference, voting, and fusion.

| File | What it does |
|---|---|
| `Train.py` | Trains a presence or weapon head from a dataset; saves `model.joblib` |
| `Evaluate.py` | Leave-one-session-out and leave-one-subject-out cross-validation; `tier_verdict` gate |
| `Infer.py` | Runs a trained model on a feature window; returns class + confidence |
| `Model.py` | Head classes: `PresenceHead` (MLP/SVM) and `WeaponHead` (variance threshold / sklearn / CNN) |
| `Weapon.py` | Weapon-specific features: σ²[p] inter-subcarrier variance, block-average decimation |
| `Vote.py` | `SegmentVoter` — soft majority vote over a motion segment for a stable verdict |
| `Fusion.py` | `fuse()` — concatenates per-node feature vectors for multi-node inference |
| `Link.py` | Per-(tx, rx)-link model and calibration bookkeeping; `LinkVoter` |
| `Resample.py` | Uniform-grid resampler to handle timing jitter between nodes (100 Hz target) |
| `Stack.py` | Stacks per-node windows into a batch tensor for the CNN path |
| `Cir.py` | Offline CIR super-resolution via L1/ISTA (optional; verify subcarrier pattern on hardware first) |
| `Heatmap.py` | Camera-supervised G×G occupancy heatmap head |
| `Adapt.py` | Session adaptation: norm-stat refresh on new calibration data |
| `Explain.py` | Feature importance and CNN channel weight introspection |

### `groundtruth/`

Produces labels during data collection.

| File | What it does |
|---|---|
| `CameraLabeler.py` | Reads an MJPEG stream and runs YOLO/SAM to produce bounding-box or segmentation labels |
| `DatasetBuilder.py` | Pairs CSI windows with labels and serializes them to `.npy` + `meta.json` |
| `Align.py` | Aligns camera timestamps to CSI timestamps; measures residual clock skew |

### `output/`

Pushes predictions out of the pipeline.

| File | What it does |
|---|---|
| `Publisher.py` | `Publisher` ABC + `JsonlPublisher` (default, zero extra dependencies) + WebSocket seam. Each prediction is one JSON line: `{t, class, conf, mode, bbox, keypoints}`. |
| `Guard.py` | `AlertGuard` deduplicates and rate-limits output events. `DriftMonitor` watches for baseline drift. |

### `diagnostics/`

| File | What it does |
|---|---|
| `Telemetry.py` | Collects per-node metrics (frames/s, free heap, uptime) from UDP health port 9877 for `health_monitor.py` and the web dashboard. |
