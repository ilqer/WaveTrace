# `wavetrace/` — Python library

Reads CSI frames from a source, runs calibration, builds labeled datasets, trains models, and serves live predictions. The heavy signal processing is in `native/` (C++); this package orchestrates it.

## Main files

| File | What it does |
|---|---|
| `source.py` | Compatibility path only: re-exports `adapters/source/` and `domain/dsp/` under the old `wavetrace.source` name, which the golden test imports. New code imports the new paths. |
| `frontend.py` | Shared pipeline loop — pulls from a source, preprocesses, emits windowed feature vectors or spectrograms. Used by both training and serving so the features always match. |
| `calibration.py` | Saves and loads per-session calibration: gain-lock scalar, quiet baseline, NBVI subcarrier mask. |
| `cli.py` | `wavetrace` CLI: `capture`, `calibrate`, `collect-data`, `train`, `localize`, `run` subcommands. |
| `config/` | Runtime config: mode (`presence`/`weapon`), backend, head, subcarrier count, window/hop. |
| `localize.py` | AoA spatial heatmap (parked — needs ≥ 2 phase-coherent antennas; ESP32-S3 has one receive chain). |
| `synthetic.py` | Synthetic CSI stream and paired camera observations with a skewed clock, behind the CLI's `--synthetic` mode. |
| `_wavetrace.pyi` | Type stubs for the native `_wavetrace` extension. |

## Subfolders

### `recognition/`

All machine learning: training, evaluation, inference, voting, and fusion.

| File | What it does |
|---|---|
| `train.py` | Trains a presence or weapon head from a dataset; saves `model.joblib` |
| `evaluate.py` | Leave-one-session-out and leave-one-subject-out cross-validation; `tier_verdict` gate |
| `infer.py` | Runs a trained model on a feature window; returns class + confidence |
| `vote.py` | `SegmentVoter` — soft majority vote over a motion segment for a stable verdict |
| `fusion.py` | `fuse()` — concatenates per-node feature vectors for multi-node inference |
| `link.py` | Per-(tx, rx)-link model and calibration bookkeeping; `LinkVoter` |
| `stack.py` | Stacks per-node windows into a batch tensor for the CNN path |
| `cir.py` | Offline CIR super-resolution via L1/ISTA (optional; verify subcarrier pattern on hardware first) |
| `heatmap.py` | Camera-supervised G×G occupancy heatmap head |
| `adapt.py` | Session adaptation: norm-stat refresh on new calibration data |
| `explain.py` | Feature importance and CNN channel weight introspection |
| `occupancy.py` | Temporal fusion of heatmap frames: per-cell Bayesian filter + 2D position Kalman smoother |
| `count.py` | Display naming for the people-count head's classes |
| `serving.py` | Per-link serving math shared by every live front end (presence, count and weapon) |
| `weapon_serving.py` | Live multi-link weapon serving: loads per-link/per-node weapon heads, shared per-window/per-link math |

### `groundtruth/`

Produces labels during data collection.

| File | What it does |
|---|---|
| `camera_labeler.py` | Reads an MJPEG stream and runs YOLO/SAM to produce bounding-box or segmentation labels |
| `dataset_builder.py` | Pairs CSI windows with labels and serializes them to `.npy` + `meta.json` |
| `alignment.py` | Aligns camera timestamps to CSI timestamps; measures residual clock skew |
| `webcam.py` | OpenCV webcam frame source on the CSI wall clock; records frames live, runs YOLO labeling offline |

### `output/`

Pushes predictions out of the pipeline.

| File | What it does |
|---|---|
| `publisher.py` | `Publisher` ABC + `JsonlPublisher` (default, zero extra dependencies) + WebSocket seam. Each prediction is one JSON line: `{t, class, conf, mode, bbox, keypoints}`. |
| `guard.py` | `AlertGuard` deduplicates and rate-limits output events. `DriftMonitor` watches for baseline drift. |

### `diagnostics/`

| File | What it does |
|---|---|
| `telemetry.py` | Collects per-node metrics (frames/s, free heap, uptime) from UDP health port 9877 for `health_monitor.py` and the web dashboard. |
| `weapon_separation.py` | Static σ²[p] PDF separation diagnostics — the weapon go/no-go check behind `experiments/weapon_litmus.py` and `/api/weapon/litmus`. |

### `domain/`

Pure descriptions of the pipeline's shape, no I/O. `adapters/` and `recognition/` may import it; it imports neither (checked by `lint-imports`).

| File | What it does |
|---|---|
| `contracts.py` | `PipelineContract`: a trained model's resample rate, window/hop framing and subcarrier width, declared once |
| `link_key.py` | `LinkKey` value object for one directed (tx, rx) link |
| `dsp/resample.py` | `resample_uniform`: a jittery CsiFrame stream onto a uniform time grid |
| `recognition/presence_head.py` | `PresenceHead`: is a person in the room? One verdict per window of CSI features |
| `recognition/weapon_head.py` | `WeaponHead`: is the person carrying metal? One verdict per window of inter-carrier features |

### `application/`

Use-cases and ports, no concrete I/O.

| File | What it does |
|---|---|
| `calibrate.py` | `calibrate_source`: quiet-baseline calibration -> a persisted calibration dir |
| `collect.py` | `collect_source`: a source + a label source -> a serialized labeled dataset |
| `serve.py` | `run_inference`: stream a source through the front-end and publish one verdict per window |
| `localize.py` | `localize_source`: stream a source through the AoA `Localizer` -> a published track |
| `ports.py` | The Protocols concrete adapters implement |

### `adapters/`

Concrete implementations of the ports in `application/ports.py`.

| File | What it does |
|---|---|
| `source/` | CSI frame sources behind the `CsiSource` port: `udp.py` (`UdpSource`, the mesh backhaul), `serial_reader.py` (`SerialReader`), `nexmon.py` (`NexmonSource`, Pi 5 GHz), `recording.py` (`RecordingSource`, `save_recording`/`load_recording`), `synthetic.py` (`SyntheticSource`), `wire_format.py` (esp-csi CSV and binary UDP parsers, `parse_batch_by_link`), `options.py` (`SourceOptions`); `build_csi_source` is in `__init__.py` |
| `recognition/heads.py` | The only place a recognition head is built or loaded from the backend registry |
| `recognition/mlp_backend.py` | Default backend: `StandardScaler` + a single-hidden-layer `MLPClassifier` |
| `recognition/svm_backend.py` | SVM backend from the CSI-sensing literature |
| `recognition/cnn_backend.py` | Small 2D-CNN over the CSI image (subcarriers x window) |
| `recognition/variance_backend.py` | Weapon baseline: one threshold over an inter-carrier variance feature |
| `recognition/_sklearn_pipeline.py` | Shared scaler + sklearn-classifier plumbing for the `mlp` and `svm` backends |
