# `tests/`

Automated test suite. All tests run on synthetic or recorded data — no hardware required.

```bash
pytest tests/ -q    # run everything; expect 631 tests, all passing
```

## Layout

| Folder | What lives there |
|---|---|
| `unit/` | One module under test, in milliseconds. Mirrors the source tree: `unit/domain/dsp/test_resample.py` tests `wavetrace/domain/dsp/resample.py`; `unit/native/<dir>/` tests the pybind11 API of `native/<dir>/`, `unit/firmware/pi/` tests `firmware/pi/`, `unit/entrypoints/api/` and `unit/entrypoints/cli/` test the dashboard and the CLI |
| `integration/` | Several layers wired together end to end (calibrate → collect → train → serve on a synthetic recording) |
| `contract/` | Each adapter matches its port in `wavetrace/application/ports.py` (member names, `inspect.signature` parameter names, kinds, order and defaults; new adapters go into `PORT_ADAPTERS`), and no package, module, class or function under `wavetrace/`, `web/`, `scripts/`, `firmware/pi/` is named `util(s)`, `helper(s)`, `common`, `misc` or `manager(s)` |
| `golden/` | End-to-end characterization tests and their committed fixtures and expected output |

`tests/` and every folder under `unit/`, `integration/` and `contract/` carry an `__init__.py`, so two test files
with the same name in different mirrored folders do not collide under pytest's default import mode. `golden/` has
none on purpose: its tests import their replay helpers as top-level modules (`from mesh_replay import ...`).

## What is tested

| File(s) | Coverage |
|---|---|
| `unit/native/core/test_core.py`, `unit/native/hardware/test_frame_parser.py` | C++ `CsiFrame` type and UDP datagram parsing |
| `unit/native/signal/test_preprocess.py`, `unit/native/signal/test_features.py` | Conjugate multiply, Hampel filter, phase unwrap, gain lock, NBVI, feature extraction |
| `unit/native/signal/test_gain_lock.py` | Gain lock apply / relock / coefficient-of-variation fallback |
| `unit/domain/test_calibration.py` | Save/load calibration, NBVI subcarrier selection |
| `unit/domain/dsp/test_resample.py`, `unit/native/signal/test_subcarrier_select.py` | Timing-jitter resampler, subcarrier ranking |
| `unit/native/signal/test_spectrogram.py` | Spectrogram builder shape and values |
| `unit/recognition/test_recognition.py`, `integration/test_pipeline.py` | Presence model train + infer on synthetic data |
| `unit/recognition/test_weapon.py`, `unit/domain/recognition/test_weapon_pipeline.py` | Weapon head train + σ²[p] baseline |
| `unit/domain/dsp/test_weapon_background_subtract.py`, `unit/recognition/test_weapon_confound.py`, `unit/diagnostics/test_weapon_litmus.py` | Weapon-specific edge cases |
| `unit/recognition/test_count.py` | People-count model train + infer |
| `unit/recognition/test_cir.py` | CIR super-resolution (L1/ISTA) |
| `unit/adapters/source/test_udp_source.py`, `unit/adapters/source/test_mesh_links.py` | UDP source and multi-node link handling |
| `unit/domain/dsp/test_multi_node.py`, `unit/recognition/test_node_weights.py`, `unit/recognition/test_link_voter.py` | Per-node voting and per-link vote weights |
| `unit/recognition/test_weapon_per_link.py` | Per-link weapon heads and link health metrics |
| `unit/recognition/test_serving.py` | Per-link serving math: calibration width and the single last-window prediction |
| `unit/domain/groundtruth/test_ground_truth.py`, `unit/groundtruth/test_segmentation_labeler.py`, `unit/groundtruth/test_vision_labeler.py` | Camera labeler and dataset builder |
| `unit/groundtruth/test_webcam.py` | Webcam frame recording and offline label streaming with a stub detector |
| `unit/recognition/test_heatmap_training.py` | Heatmap trainer pairs each mask with its own window |
| `unit/native/signal/test_presence_segment.py` | Motion-segment gate used inside the voter |
| `unit/domain/test_localize.py`, `unit/domain/dsp/test_frontend_extensions.py` | Localization scaffold and frontend extensions |
| `unit/adapters/recognition/test_regression.py` | Regression guards: heads reject single-class data, a zero-depth synthetic weapon warns, `build_dataset` accepts a generator |
| `unit/firmware/pi/test_pi_publisher.py` | Round-trips the Pi wire format through `wavetrace/adapters/source/wire_format.py` |
| `unit/domain/test_guard.py` | `AlertGuard` fire / clear / cooldown, `DriftMonitor` advisory, guard-event JSONL bytes |
| `golden/test_golden_collect_scripts.py` | Characterization tests pinning the stdout, stderr and a sha256 manifest of the written tree of `collect_presence`, `collect_count` and `collect_weapon` over a replayed UDP mesh (`golden/expected/collect_{presence,count,weapon}.txt`); the replay helpers, mesh trainers, fake UDP socket and tree hasher shared with the serving, web and camera goldens live in `golden/mesh_replay.py` |
| `golden/test_golden_collect_camera.py` | Characterization tests pinning the stdout and a sha256 manifest of the written tree of `collect_camera` (presence, weapon, each with and without `--train`) over a replayed mesh and a fake YOLO-seg webcam (`golden/expected/collect_camera_*.txt`); the fake net, webcam, synchronous camera thread and monotonic clock live in `golden/camera_replay.py` |
| `golden/test_golden_web_camera.py` | The same for the dashboard's `WaveTraceRunner.run_camera_collection_request`: its logs, published messages and their keys, and the written tree (`golden/expected/web_camera_*.txt`) |
| `golden/test_golden_web_training.py` | The same for `WaveTraceRunner.run_training_request`, per stage and dataset choice, on one capture-profile root built in the collect modes' own layout (presence, count, the weapon pool, a dashboard camera session): logs without traceback frames, payloads with sorted keys and without `fit_seconds`, and only the files training wrote (`golden/expected/web_training_*.txt`) |
| `golden/test_golden_web_inference.py` | Characterization tests pinning the web dashboard's `WaveTraceRunner.run_inference` payloads (presence, count and weapon mesh roots, and a single-node file model on the synthetic source): key paths per message type (`golden/expected/web_*_keys.txt`) and rounded values in emission order (`web_*_values.txt`); plus weapon alerts staying weapon-mode only, and a stop or a socket error ending a mesh run with the socket closed |
| `golden/test_golden_web_routes.py` | Characterization tests pinning the dashboard's route table in registration order (`golden/expected/web_routes.txt`) and its OpenAPI schema with sorted keys (`golden/expected/web_openapi.json`) |
| `golden/test_golden_pipeline.py` | Characterization tests pinning calibrate → collect → train → infer on committed fixture recordings (`golden/fixtures/`, `golden/expected/`) |
| `contract/test_port_conformance.py` | Every adapter in `PORT_ADAPTERS` has each member of its port with the port's parameter names, kinds, order and defaults (an adapter may spell out keyword-only parameters with defaults where the port ends in `**kwargs`), and every port has a case |
| `contract/test_no_dumping_ground_names.py` | No dumping-ground name (`util(s)`, `helper(s)`, `common`, `misc`, `manager(s)`, in any casing) on a package, module, class or function under the Python source roots |
| `unit/` (other files) | Unit tests for backend registry, contracts, node-health gain drift against calibration, YOLO labeler options, occupancy, pipeline-contract persistence, serving artifact contract, source options, web server options, the collect and serve mesh use-cases and their modes, the camera collect use-case and its next-free-session rule, `UdpLinkCapture` / `UdpTimedLinkCapture` / `detect_node_ids`, calibrated-node discovery, `pipeline.collect_mesh`, `pipeline.serve_mesh`, `pipeline.collect_camera` and the `collect_camera` script's model roots, the dashboard's camera-collection bind error, the dashboard's Train (pools per stage, default model roots, one `done` then `pipeline_done`, stop, the heatmap skip) and `/api/paths/scan`, the dashboard's HTTP and WebSocket routes through `TestClient` inside the app's lifespan (`unit/entrypoints/api/test_routes.py`), Calib and Collect always ending with `pipeline_done`, the `wavetrace train --stage weapon` backend flag (`unit/entrypoints/cli/test_cli_train.py`), each pool's dataset finder, the training backends, the manifest-only dataset labels and the pooled `dataset_report`, `build_synthetic_source` |

Plain helpers shared across test files live in `fixtures/synthetic_csi.py`: raw wire-format frames
(`encode_frame`, `generate_raw_frames`) and the quiet empty-room baseline several calibrating tests share
(`generate_quiet_baseline`, `calibrate_quiet_baseline`).

Tests import those helpers as `fixtures.synthetic_csi` and an operator script as `scripts.<name>`
(`from scripts import collect_count`). `pythonpath = ["."]` in `pyproject.toml` puts the repo root on the
test path, so both resolve under plain `pytest` as well as `python -m pytest`.
