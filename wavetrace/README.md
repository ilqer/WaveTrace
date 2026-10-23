# `wavetrace/` — Python library

Reads CSI frames from a source, runs calibration, builds labeled datasets, trains models, and serves live predictions. The heavy signal processing is in `native/` (C++); this package orchestrates it.

## Main files

| File | What it does |
|---|---|
| `source.py` | Compatibility path only: re-exports `adapters/source/` and `domain/dsp/` under the old `wavetrace.source` name, which the golden test imports. New code imports the new paths. |
| `cli.py` | `wavetrace` CLI: `capture`, `calibrate`, `collect-data`, `train`, `localize`, `run` subcommands. |
| `pipeline.py` | Directory-based entry points (`calibrate_source`, `collect_source`, `run_inference`, `localize_source`): load from disk, run an `application/` use-case, save to disk; `serve_mesh` binds the mesh UDP port and runs `ServeMeshUseCase` with the live tunables (`CHUNK_S`, `LINK_TIMEOUT_S`, `BUFFER_S`); `collect_mesh` finds the calibrated nodes and runs `CollectMeshUseCase` over `UdpLinkCapture`, writing each link's recording and dataset through `write_link_dataset` (`save_recording` + `collect_source`); `collect_camera` picks the stage's camera mode and the next free session and runs `CollectCameraUseCase` over `UdpTimedLinkCapture` and an injected `CameraLabelCapture`, stacking the heatmap dataset with `build_stacked_dataset`. The CLI, `web/` and `scripts/` call these. |
| `synthetic.py` | Synthetic CSI stream and paired camera observations with a skewed clock, behind the CLI's `--synthetic` mode. |
| `_wavetrace.pyi` | Type stubs for the native `_wavetrace` extension. |

## Subfolders

### `recognition/`

All machine learning: training, evaluation, inference, and fusion.

| File | What it does |
|---|---|
| `train.py` | Trains a presence or weapon head from a dataset; saves `model.joblib` |
| `evaluate.py` | Leave-one-session-out and leave-one-subject-out cross-validation; `tier_verdict` gate |
| `infer.py` | `measure_latency` (budget 8 ms per window) |
| `fusion.py` | `fuse()` — concatenates per-node feature vectors for multi-node inference |
| `link.py` | `accuracy_weights` (LOGO accuracy -> static link priors) and `evaluate_link_fusion` (offline replay of the `LinkVoter` blend) |
| `stack.py` | Stacks per-node windows into a batch tensor for the CNN path |
| `cir.py` | Offline CIR super-resolution via L1/ISTA (optional; verify subcarrier pattern on hardware first) |
| `heatmap.py` | Camera-supervised G×G occupancy heatmap head |
| `adapt.py` | Session adaptation: norm-stat refresh on new calibration data |
| `explain.py` | Feature importance and CNN channel weight introspection |
| `occupancy.py` | Temporal fusion of heatmap frames: per-cell Bayesian filter + 2D position Kalman smoother |
| `count.py` | Display naming for the people-count head's classes |
| `serving.py` | Per-link serving math shared by every live front end (presence, count and weapon); `NodeServer` holds one rx node's calibration, head and vote state; `discover_node_dirs` and `load_presence_node_servers` find and load the per-node heads |
| `collect_modes.py` | The collect modes `CollectMeshUseCase` runs, each its own `CollectPlan` and `TrainingPolicy`: `PresenceCollectMode`, `CountCollectMode`, `WeaponCollectMode` (session conditions and prompts, per-link dataset labels and directory layout, head training and report lines); `WeaponTrainingPolicy`, the weapon training half `WeaponCollectMode` and `CameraWeaponMode` extend; each pool's `find_dataset_dirs_by_node()`, `find_collected_max_count`, `default_model_root` (the dashboard's Train uses all of them) |
| `camera_collect_modes.py` | The camera modes `CollectCameraUseCase` runs, each its `CameraCollectPlan` and `CameraTrainingPolicy`: `CameraPresenceMode` (per-link datasets under `cam_ds/presence`, presence heads on the whole camera pool through `CameraPresenceTrainingPolicy`) and `CameraWeaponMode` (per-link datasets into the `weapon_ds` pool, trained as a `WeaponTrainingPolicy`); both train the heatmap head (`train_heatmap_head`); `next_camera_session_id`, `default_camera_model_root`, `find_heatmap_capture_root` |
| `serve_modes.py` | The `ServeMode` strategies `ServeMeshUseCase` runs: `PresenceServeMode`, `CountServeMode`, `WeaponServeMode` (load, per-link score, banner, verdict line) |
| `weapon_serving.py` | Live multi-link weapon serving: loads per-link/per-node weapon heads, shared per-window/per-link math |

### `groundtruth/`

Produces labels during data collection.

| File | What it does |
|---|---|
| `vision_labeler.py` | `VisionLabeler`: per-frame labels from any object detector (`Detection`, `LabelerOptions`) |
| `segmentation_labeler.py` | `SegmentationLabeler`: per-frame labels from a segmentation model, with a mask-overlap weapon gate and a pooled occupancy grid (`Segment`) |
| `yolo_labeler.py` | Ultralytics adapters `YoloLabeler` and `YoloSegLabeler` (`YoloLabelerOptions`) |
| `replay_labeler.py` | `ReplayLabeler`: replays pre-computed detections |
| `location_chip_labeler.py` | `LocationChipLabeler`: labels from a BLE/UWB tag track on the CSI clock |
| `thermal_labeler.py` | `ThermalLabeler`: thermal camera seam, not implemented |
| `webcam.py` | ffmpeg webcam frame source on the CSI wall clock; records frames live and labels them offline or live; `WebcamLabelCapture` (the `CameraLabelCapture` adapter) labels each frame live with YOLO-seg |

### `output/`

Pushes predictions out of the pipeline.

| File | What it does |
|---|---|
| `publisher.py` | `Publisher` ABC + `JsonlPublisher` (default, zero extra dependencies) + WebSocket seam. Each prediction is one JSON line: `{t, class, conf, mode, bbox, keypoints}`. |

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
| `calibration.py` | Empty-room calibration: `Calibration` (gain-lock scale, NBVI subcarrier selection, quiet baseline), `CalibrationResult`, `reflection_signature`, `build_image_baseline` |
| `localize.py` | AoA spatial heatmap: `Localizer` (MUSIC/Bartlett angle spectrum, joint delay-angle room map) and `Localization`. Parked: needs ≥ 2 phase-coherent antennas; the ESP32-S3 has one receive chain |
| `tracking.py` | The Kalman `Tracker` over a `Localization` stream (angle, range), with its `TrackState`. Parked with `localize.py` |
| `model_config.py` | `ModelConfig`: a trained head's stage, backend, subcarrier count and window/hop framing |
| `guard.py` | `AlertGuard` and `DriftMonitor`: pure alert-debounce and baseline-drift state machines with their event classes |
| `dsp/frontend.py` | Shared pipeline loop (`iter_windows`, `iter_stacked_windows`): source → preprocess → features → windows |
| `dsp/resample.py` | `resample_uniform`: a jittery CsiFrame stream onto a uniform time grid |
| `groundtruth/alignment.py` | `align`: matches label timestamps to CSI window timestamps; `estimate_clock_offset` measures residual clock skew |
| `groundtruth/labeler.py` | `Labeler` interface, `ScriptedLabeler`, `FrameDetection`, `presence_label_function`, `weapon_label_function` |
| `groundtruth/dataset.py` | `Dataset`, `build_dataset`, `build_stacked_dataset`: pair CSI windows with labels |
| `recognition/presence_head.py` | `PresenceHead`: is a person in the room? One verdict per window of CSI features |
| `recognition/weapon_head.py` | `WeaponHead`: is the person carrying metal? One verdict per window of inter-carrier features |
| `recognition/vote.py` | `SegmentVoter`: soft majority vote over a motion segment for a stable verdict |
| `recognition/link_vote.py` | `LinkVoter`: blends one probability vector per link, weighted by static prior x live quality |
| `recognition/verdict.py` | `decide_class`: one window's class probabilities -> (class id, confidence) |

### `application/`

Use-cases and ports, no concrete I/O (console output through `print`): each use-case takes loaded values and returns results; `pipeline.py` does the loading and saving.

| File | What it does |
|---|---|
| `calibrate.py` | `calibrate`: a source + a `Calibration` -> a `CalibrationResult` |
| `collect.py` | `collect_dataset`: a source + a loaded calibration + a label source -> a labeled `Dataset` |
| `serve.py` | `iter_window_verdicts`: stream a source through the front-end and an inference session, yielding one `WindowVerdict` per window (one forward pass each) with its `WindowView` (features, image, inter-carrier block); `infer_windows` publishes each verdict and each window behaviour's events right after it, then the behaviours' extra rows after the last window |
| `behaviours/window.py` | `WindowBehaviour` hooks run after each window's verdict: `AlertGuardBehaviour` (debounced weapon alerts; `build_alert_behaviours(mode)` adds it in weapon mode only), `SegmentVoteBehaviour` (one soft-vote verdict at the end) |
| `behaviours/drift.py` | `DriftMonitoredSource`: a `CsiSource` wrapper that publishes recalibration advisories from raw per-frame magnitudes |
| `collect_mesh.py` | `CollectMeshUseCase`: the labelled collect loop (capture each condition through the `LinkCapture` port, resample each link per condition, drop links under one window, write one dataset per link, report nodes with no usable link per session, then train and report) behind a per-mode `CollectPlan` + `TrainingPolicy` |
| `collect_camera.py` | `CollectCameraUseCase`: camera-labelled collection (the camera labels on a worker thread through the `CameraLabelCapture` port while the `TimedLinkCapture` port records every link; one bell when the capture ends; resample and write one dataset per link; stack each node's merged links into the heatmap dataset; optionally train through a `CameraTrainingPolicy`), every message through one injected `log` |
| `serve_mesh.py` | `ServeMeshUseCase`: the live all-pairs loop (per-link buffers, fuse cadence, `LinkVoter` blend) behind a per-mode `ServeMode` strategy; `iter_fused` yields one `FusedVerdict` (blend + one `LinkReport` per live link) per fuse for the web dashboard, `run` prints them |
| `localize.py` | `localize`: stream a source through the AoA `Localizer` (and optional `Tracker`) -> a published track + the room map |
| `ports.py` | The Protocols concrete adapters implement |

### `adapters/`

Concrete implementations of the ports in `application/ports.py`.

| File | What it does |
|---|---|
| `source/` | CSI frame sources behind the `CsiSource` port: `udp.py` (`UdpSource`, the mesh backhaul, and `UdpLinkBatchReceiver` behind the `LinkBatchReceiver` port), `link_capture.py` (`UdpLinkCapture` behind the `LinkCapture` port, one operator-prompted capture of every link; `UdpTimedLinkCapture` behind the `TimedLinkCapture` port, every link for a fixed time or until stopped; `detect_node_ids`), `serial_reader.py` (`SerialReader`), `nexmon.py` (`NexmonSource`, Pi 5 GHz), `recording.py` (`RecordingSource`, `save_recording`/`load_recording`), `synthetic.py` (`SyntheticSource`, typed `build_synthetic_source`), `wire_format.py` (esp-csi CSV and binary UDP parsers, `parse_batch_by_link`), `options.py` (`SourceOptions`); `build_csi_source` is in `__init__.py` |
| `store/dataset.py` | `save_dataset` / `load_dataset`: the labeled-dataset dir (`features.npy`, `images.npy`, `manifest.jsonl`, `meta.json`); `load_dataset_labels` reads only the manifest and meta (classes and LOGO groups) |
| `store/calibration.py` | `save_calibration` / `load_calibration`: the calibration dir (`baseline_mag.npy`, `baseline_diff.npy`, `meta.json`); `discover_calibrated_node_ids` lists the `<cal>/node<id>` dirs, `find_node_calibration_dirs` maps each id to its dir |
| `store/localization.py` | `save_localization`: the aggregate room-map dir (`heatmap.npy`, `angles.npy`, `ranges.npy`, `meta.json`) |
| `store/scripted_spans.py` | `load_scripted_spans`: reads the JSON span sidecar (`{"spans": [{"start", "end", "present"}]}`) that `ScriptedLabeler` takes |
| `recognition/heads.py` | The only place a recognition head is built or loaded from the backend registry |
| `recognition/mlp_backend.py` | Default backend: `StandardScaler` + a single-hidden-layer `MLPClassifier` |
| `recognition/svm_backend.py` | SVM backend from the CSI-sensing literature |
| `recognition/cnn_backend.py` | Small 2D-CNN over the CSI image (subcarriers x window) |
| `recognition/variance_backend.py` | Weapon baseline: one threshold over an inter-carrier variance feature |
| `recognition/_sklearn_pipeline.py` | Shared scaler + sklearn-classifier plumbing for the `mlp` and `svm` backends |
