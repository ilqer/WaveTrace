# `wavetrace/` — Python library

Reads CSI frames from a source, runs calibration, builds labeled datasets, trains models, and serves live predictions. The heavy signal processing is in `native/` (C++); this package orchestrates it.

## Main files

| File | What it does |
|---|---|
| `source.py` | Compatibility path only: re-exports `adapters/source/` and `domain/dsp/` under the old `wavetrace.source` name, which the golden test imports. New code imports the new paths. |
| `pipeline.py` | Directory-based entry points (`calibrate_source`, `collect_source`, `run_inference`, `localize_source`): load from disk, run an `application/` use-case, save to disk; `serve_mesh` binds the mesh UDP port and runs `ServeMeshUseCase` with the live tunables (`CHUNK_S`, `LINK_TIMEOUT_S`, `BUFFER_S`) and the caller's `rate_gate` and `min_sample_rate_fraction`; `run_inference` takes `min_sample_rate_fraction`; `collect_mesh` finds the calibrated nodes and runs `CollectMeshUseCase` over `UdpLinkCapture`, writing each link's recording and dataset through `write_link_dataset` (`save_recording` + `collect_source`); `collect_camera` picks the stage's camera mode and the next free session and runs `CollectCameraUseCase` over `UdpTimedLinkCapture` and an injected `CameraLabelCapture`, stacking the heatmap dataset with `build_stacked_dataset`. The CLI, the dashboard API and `scripts/` call these. |
| `synthetic.py` | Synthetic CSI stream and paired camera observations with a skewed clock, behind the CLI's `--synthetic` mode. |
| `_wavetrace.pyi` | Type stubs for the native `_wavetrace` extension. |

## Subfolders

### `recognition/`

All machine learning: training, evaluation, inference, and fusion.

| File | What it does |
|---|---|
| `train.py` | Trains a presence or weapon head from a dataset; saves `model.joblib`, `metrics.json` and `run_manifest.json` as a new current run of the model dir |
| `evaluate.py` | Leave-one-session-out and leave-one-subject-out cross-validation; `tier_verdict` gate |
| `infer.py` | `measure_latency` (budget 8 ms per window) |
| `fusion.py` | `fuse()` — concatenates per-node feature vectors for multi-node inference |
| `link.py` | `accuracy_weights` (LOGO accuracy -> static link priors) and `evaluate_link_fusion` (offline replay of the `LinkVoter` blend) |
| `stack.py` | Stacks per-node windows into a batch tensor for the CNN path |
| `cir.py` | Offline CIR super-resolution via L1/ISTA (optional; verify subcarrier pattern on hardware first); `cir_from_taps` gives the same statistics for UWB taps without ISTA |
| `heatmap.py` | Camera-supervised G×G occupancy heatmap head |
| `adapt.py` | Session adaptation: norm-stat refresh on new calibration data |
| `explain.py` | Feature importance and CNN channel weight introspection |
| `occupancy.py` | Temporal fusion of heatmap frames: per-cell Bayesian filter + 2D position Kalman smoother |
| `count.py` | Display naming for the people-count head's classes |
| `serving.py` | Per-link serving math shared by every live front end (presence, count and weapon); `NodeServer` holds one rx node's calibration, head and vote state; `discover_node_dirs` and `load_presence_node_servers` find and load the per-node heads |
| `collect_modes.py` | The collect modes `CollectMeshUseCase` runs, each its own `CollectPlan` and `TrainingPolicy`: `PresenceCollectMode`, `CountCollectMode`, `WeaponCollectMode` (session conditions and prompts, per-link dataset labels and directory layout, head training and report lines); `WeaponTrainingPolicy`, the weapon training half `WeaponCollectMode` and `CameraWeaponMode` extend; each pool's `find_dataset_dirs_by_node()`, `find_collected_max_count`, `default_model_root` (the dashboard's Train uses all of them) |
| `camera_collect_modes.py` | The camera modes `CollectCameraUseCase` runs, each its `CameraCollectPlan` and `CameraTrainingPolicy`: `CameraPresenceMode` (per-link datasets under `cam_ds/presence`, presence heads on the whole camera pool through `CameraPresenceTrainingPolicy`) and `CameraWeaponMode` (per-link datasets into the `weapon_ds` pool, trained as a `WeaponTrainingPolicy`); both train the heatmap head (`train_heatmap_head`); `next_camera_session_id`, `default_camera_model_root`, `find_heatmap_capture_root` |
| `serve_modes.py` | The `ServeMode` strategies `ServeMeshUseCase` runs: `PresenceServeMode`, `CountServeMode`, `WeaponServeMode` (load, per-link score, banner, verdict line); `load` refuses heads with different class orderings (presence), heads trained at different sample rates, a calibration wider than its model's capture width and one made from another width |
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
| `webcam.py` | ffmpeg webcam frame source on the CSI wall clock; records frames live and labels them offline or live; `WebcamLabelCapture` (the `CameraLabelCapture` adapter) labels each frame live with YOLO-seg; a set `stop` ends a stalled read within 0.1 s, and closing the capture terminates ffmpeg (killed after 2 s) and reaps it |

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
| `contracts.py` | `PipelineContract`: a trained model's resample rate, window/hop framing and subcarrier width, declared once; `RunManifest` / `LibraryVersions`: where a trained or uploaded model run came from |
| `cir_record.py` | `CirRecord`: one UWB packet's CIR (R RX chains x N taps, first-path indices, distance, AoA, PDoA) with `first_path_columns` for `reanchor_taps` |
| `link_key.py` | `LinkKey` value object for one directed (tx, rx) link |
| `rules.py` | `BusinessRule` and the named serving rules: `SubcarrierWidthMustCoverCalibration`, `SubcarrierWidthMustMatchContract` (a legacy model without a recorded width accepts every width), `SubcarrierWidthMustMatchCalibration` (likewise for a legacy calibration), `SubcarrierWidthMustStayConstantPerLink`, `LinkMustHaveServingHead`, `HeadsMustShareClassOrdering`, `HeadsMustShareSampleRate`, `SampleRateMustBeWithinTolerance` (delivered rate at least `min_sample_rate_fraction` of the head's target rate), `SampleRateMustKeepPaceWithPeers` (at least that fraction of the median delivered rate of the links measured at the same fuse); `measure_delivered_rate_hz` is the one rate measurement both use; `require` raises `BusinessRuleViolationError`, whose message starts with the rule's name |
| `calibration.py` | Empty-room calibration: `Calibration` (gain-lock scale, NBVI subcarrier selection, quiet baseline), `CalibrationResult` (records the raw capture width it was made from, `subcarrier_width`), `reflection_signature`, `build_image_baseline` |
| `localize.py` | AoA spatial heatmap: `Localizer` (MUSIC/Bartlett angle spectrum, joint delay-angle room map; `tone_layout=` for FFT-order frames) and `Localization`. Parked: needs ≥ 2 phase-coherent antennas; the ESP32-S3 has one receive chain |
| `tracking.py` | The Kalman `Tracker` over a `Localization` stream (angle, range), with its `TrackState`. Parked with `localize.py` |
| `model_config.py` | `ModelConfig`: a trained head's stage, backend, subcarrier count and window/hop framing |
| `guard.py` | `AlertGuard` and `DriftMonitor`: pure alert-debounce and baseline-drift state machines with their event classes |
| `tone_layout.py` | `ToneField` / `ToneLayout`: which signed OFDM tone each CSI column holds per capture layout (`ESP32S3_HT20`, `ESP32S3_HT40`, `NEXMON_VHT80`), `tone_layout_by_name`, `select_field` (carrying tones of the layout's `default_field`, sorted by tone), legacy `infer_tone_layout` by width |
| `dsp/frontend.py` | Shared pipeline loop (`iter_windows`, `iter_stacked_windows`): source → preprocess → features → windows |
| `dsp/resample.py` | `resample_uniform`: a jittery CsiFrame stream onto a uniform time grid |
| `dsp/cir_alignment.py` | UWB tap alignment on plain arrays: `reanchor_taps` (each RX's first path onto one column) and `cross_spectrum` (H_r·conj(H_0), cancels a common window shift) |
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
| `serve.py` | `iter_window_verdicts`: stream a source through the front-end and an inference session, yielding one `WindowVerdict` per window (one forward pass each) with its `WindowView` (features, image, inter-carrier block); a model that records its capture width refuses a calibration wider than, or made from another width than, that width before the source is read, and frames of another width than the model's (else the calibration's) are dropped and counted under `SubcarrierWidthMustMatchContract` (`SubcarrierWidthMustMatchCalibration`); a window whose first-to-last frame timestamps give under `min_sample_rate_fraction` (default 80 %) of the head's target rate is skipped and counted under `SampleRateMustBeWithinTolerance` (O(1) per window); `infer_windows` publishes each verdict and each window behaviour's events right after it, then the behaviours' extra rows after the last window |
| `behaviours/window.py` | `WindowBehaviour` hooks run after each window's verdict: `AlertGuardBehaviour` (debounced weapon alerts; `build_alert_behaviours(mode)` adds it in weapon mode only), `SegmentVoteBehaviour` (one soft-vote verdict at the end) |
| `behaviours/drift.py` | `DriftMonitoredSource`: a `CsiSource` wrapper that publishes recalibration advisories from raw per-frame magnitudes |
| `behaviours/rule_violations.py` | `RuleViolationCounter`: counts dropped frames per `domain/rules.py` rule and logs a WARNING (INFO for the rule types it is told are routine) naming the rule, the link and the counts at most once per rule per 10 s; `iter_frames_satisfying_width` drops and counts frames of another width |
| `collect_mesh.py` | `CollectMeshUseCase`: the labelled collect loop (capture each condition through the `LinkCapture` port, resample each link per condition, drop links under one window, write one dataset per link, report nodes with no usable link per session, then train and report) behind a per-mode `CollectPlan` + `TrainingPolicy` |
| `collect_camera.py` | `CollectCameraUseCase`: camera-labelled collection (the camera labels on a worker thread through the `CameraLabelCapture` port while the `TimedLinkCapture` port records every link; the thread is told to stop and joined, and one still running after its grace period is logged as an error, joined, and fails the session; one bell when the capture ends; resample and write one dataset per link; stack each node's merged links into the heatmap dataset; optionally train through a `CameraTrainingPolicy`), every message through one injected `log` |
| `serve_mesh.py` | `ServeMeshUseCase`: the live all-pairs loop (per-link buffers, fuse cadence, `LinkVoter` blend) behind a per-mode `ServeMode` strategy; a link's batches are admitted by `domain/rules.py` rules (a head for its rx node, the calibration's and the model's width, one width per link); a link is left out of a fuse when it delivers too slowly, by the `rate_gate` option: `RateGate.TARGET` (default) under `min_sample_rate_fraction` (default 80 %) of its head's target rate, counted under `SampleRateMustBeWithinTolerance`; `RateGate.PEERS` under that fraction of the median delivered rate of the live links with 2 or more buffered frames, counted under `SampleRateMustKeepPaceWithPeers` and dropping nothing while fewer than 2 links are measured (the median is computed once per fuse, O(links log links)); every dropped frame is counted per rule (`violation_counts`) and logged through a `RuleViolationCounter` (a link without a head at INFO, the rest WARNING); `iter_fused` yields one `FusedVerdict` (blend + one `LinkReport` per live link) per fuse for the web dashboard, `run` prints them |
| `localize.py` | `localize`: stream a source through the AoA `Localizer` (and optional `Tracker`) -> a published track + the room map |
| `ports.py` | The Protocols concrete adapters implement |

### `adapters/`

Concrete implementations of the ports in `application/ports.py`.

| File | What it does |
|---|---|
| `source/` | CSI frame sources behind the `CsiSource` port: `udp.py` (`UdpSource`, the mesh backhaul, and `UdpLinkBatchReceiver` behind the `LinkBatchReceiver` port, a context manager that closes its socket), `link_capture.py` (`UdpLinkCapture` behind the `LinkCapture` port, one operator-prompted capture of every link; `UdpTimedLinkCapture` behind the `TimedLinkCapture` port, every link for a fixed time or until stopped; `detect_node_ids`), `serial_reader.py` (`SerialReader`), `nexmon.py` (`NexmonSource`, Pi 5 GHz), `recording.py` (`RecordingSource`, `save_recording`/`load_recording`, `load_tone_layout` from `meta.json`), `synthetic.py` (`SyntheticSource`, typed `build_synthetic_source`), `wire_format.py` (esp-csi CSV and binary UDP parsers, `parse_batch_by_link`), `uwb_wire_format.py` (`parse_cir_batch`, the v4 UWB CIR parser) and `uwb_udp.py` (`UdpCirSource` behind the `CirSource` port, default port 9879), `options.py` (`SourceOptions`); `build_csi_source` is in `__init__.py` |
| `store/dataset.py` | `save_dataset` / `load_dataset`: the labeled-dataset dir (`features.npy`, `images.npy`, `manifest.jsonl`, `meta.json`); `load_dataset_labels` reads only the manifest and meta (classes and LOGO groups) |
| `store/calibration.py` | `save_calibration` / `load_calibration`: the calibration dir (`baseline_mag.npy`, `baseline_diff.npy`, `meta.json`, whose `subcarrier_width` is absent in a calibration saved before it was recorded); `discover_calibrated_node_ids` lists the `<cal>/node<id>` dirs, `find_node_calibration_dirs` maps each id to its dir |
| `store/localization.py` | `save_localization`: the aggregate room-map dir (`heatmap.npy`, `angles.npy`, `ranges.npy`, `meta.json`) |
| `store/scripted_spans.py` | `load_scripted_spans`: reads the JSON span sidecar (`{"spans": [{"start", "end", "present"}]}`) that `ScriptedLabeler` takes |
| `store/model_runs.py` | `write_model_run`: writes a run dir `<model_dir>/<run_id>/` atomically and points `<model_dir>/current` (the `CURRENT` file on Windows or without symlinks) at it; `resolve_run_dir` / `resolve_model_path`: the current run, else the flat layout of models trained before runs; `find_model_dir`: the model dir a run dir belongs to |
| `store/run_manifest.py` | `build_run_manifest`: the `RunManifest` saved as `run_manifest.json` (git state, config and dataset sha256, seed, library versions, contract); `build_upload_manifest` for an uploaded model; `RunProvenance`: the git state and library versions one training call reads once; `hash_files` streams in 1 MiB chunks |
| `recognition/heads.py` | The only place a recognition head is built or loaded from the backend registry |
| `recognition/mlp_backend.py` | Default backend: `StandardScaler` + a single-hidden-layer `MLPClassifier` |
| `recognition/svm_backend.py` | SVM backend from the CSI-sensing literature |
| `recognition/cnn_backend.py` | Small 2D-CNN over the CSI image (subcarriers x window) |
| `recognition/variance_backend.py` | Weapon baseline: one threshold over an inter-carrier variance feature |
| `recognition/_sklearn_pipeline.py` | Shared scaler + sklearn-classifier plumbing for the `mlp` and `svm` backends |

### `entrypoints/`

The composition roots: each wires adapters into the use-cases. Nothing else imports this package (checked by
`lint-imports`).

| File | What it does |
|---|---|
| `cli/main.py` | `wavetrace` CLI: `capture`, `calibrate`, `collect-data`, `train`, `localize`, `run` subcommands. |
| `api/` | The web dashboard's FastAPI backend (`app.py`, the routers, `WaveTraceRunner`, `DeviceHub`); its routes and files are listed in [`web/README.md`](../web/README.md). |
