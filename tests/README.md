# `tests/`

Automated test suite. All tests run on synthetic or recorded data — no hardware required.

```bash
pytest tests/ -q    # run everything; expect 609 tests, all passing
```

## What is tested

| File(s) | Coverage |
|---|---|
| `test_core.py`, `test_frame_parser.py` | C++ `CsiFrame` type and UDP datagram parsing |
| `test_preprocess.py`, `test_features.py` | Conjugate multiply, Hampel filter, phase unwrap, gain lock, NBVI, feature extraction |
| `test_gain_lock.py` | Gain lock apply / relock / coefficient-of-variation fallback |
| `test_calibration.py` | Save/load calibration, NBVI subcarrier selection |
| `test_resample.py`, `test_subcarrier_select.py` | Timing-jitter resampler, subcarrier ranking |
| `test_spectrogram.py` | Spectrogram builder shape and values |
| `test_recognition.py`, `test_pipeline.py` | Presence model train + infer on synthetic data |
| `test_weapon.py`, `test_weapon_pipeline.py` | Weapon head train + σ²[p] baseline |
| `test_weapon_background_subtract.py`, `test_weapon_confound.py`, `test_weapon_litmus.py` | Weapon-specific edge cases |
| `test_count.py` | People-count model train + infer |
| `test_cir.py` | CIR super-resolution (L1/ISTA) |
| `test_udp_source.py`, `test_mesh_links.py` | UDP source and multi-node link handling |
| `test_multi_node.py`, `test_node_weights.py`, `test_link_voter.py` | Per-node voting and per-link vote weights |
| `test_weapon_per_link.py` | Per-link weapon heads and link health metrics |
| `test_serving.py` | Per-link serving math: calibration width and the single last-window prediction |
| `test_ground_truth.py`, `test_segmentation_labeler.py`, `test_vision_labeler.py` | Camera labeler and dataset builder |
| `test_webcam.py` | Webcam frame recording and offline label streaming with a stub detector |
| `test_heatmap_training.py` | Heatmap trainer pairs each mask with its own window |
| `test_presence_segment.py` | Motion-segment gate used inside the voter |
| `test_localize.py`, `test_frontend_extensions.py` | Localization scaffold and frontend extensions |
| `test_regression.py` | End-to-end regression: full pipeline on a synthetic recording |
| `test_pi_publisher.py` | Round-trips the Pi wire format through `wavetrace/adapters/source/wire_format.py` |
| `test_guard.py` | `AlertGuard` fire / clear / cooldown, `DriftMonitor` advisory, guard-event JSONL bytes |
| `golden/test_golden_collect_scripts.py` | Characterization tests pinning the stdout, stderr and a sha256 manifest of the written tree of `collect_presence`, `collect_count` and `collect_weapon` over a replayed UDP mesh (`golden/expected/collect_{presence,count,weapon}.txt`); the replay helpers, mesh trainers, fake UDP socket and tree hasher shared with the serving, web and camera goldens live in `golden/mesh_replay.py` |
| `golden/test_golden_collect_camera.py` | Characterization tests pinning the stdout and a sha256 manifest of the written tree of `collect_camera` (presence, weapon, each with and without `--train`) over a replayed mesh and a fake YOLO-seg webcam (`golden/expected/collect_camera_*.txt`); the fake net, webcam, synchronous camera thread and monotonic clock live in `golden/camera_replay.py` |
| `golden/test_golden_web_camera.py` | The same for the dashboard's `WaveTraceRunner.run_camera_collection_request`: its logs, published messages and their keys, and the written tree (`golden/expected/web_camera_*.txt`) |
| `golden/test_golden_web_training.py` | The same for `WaveTraceRunner.run_training_request`, per stage and dataset choice, on one capture-profile root built in the collect modes' own layout (presence, count, the weapon pool, a dashboard camera session): logs without traceback frames, payloads with sorted keys and without `fit_seconds`, and only the files training wrote (`golden/expected/web_training_*.txt`) |
| `golden/test_golden_web_inference.py` | Characterization tests pinning the web dashboard's `WaveTraceRunner.run_inference` payloads (presence, count and weapon mesh roots, and a single-node file model on the synthetic source): key paths per message type (`golden/expected/web_*_keys.txt`) and rounded values in emission order (`web_*_values.txt`); plus weapon alerts staying weapon-mode only, and a stop or a socket error ending a mesh run with the socket closed |
| `golden/test_golden_web_routes.py` | Characterization tests pinning the dashboard's route table in registration order (`golden/expected/web_routes.txt`) and its OpenAPI schema with sorted keys (`golden/expected/web_openapi.json`) |
| `golden/test_golden_pipeline.py` | Characterization tests pinning calibrate → collect → train → infer on committed fixture recordings (`golden/fixtures/`, `golden/expected/`) |
| `unit/` | Unit tests for backend registry, contracts, node-health gain drift against calibration, YOLO labeler options, occupancy, pipeline-contract persistence, serving artifact contract, source options, web server options, the collect and serve mesh use-cases and their modes, the camera collect use-case and its next-free-session rule, `UdpLinkCapture` / `UdpTimedLinkCapture` / `detect_node_ids`, calibrated-node discovery, `pipeline.collect_mesh`, `pipeline.serve_mesh`, `pipeline.collect_camera` and the `collect_camera` script's model roots, the dashboard's camera-collection bind error, the dashboard's Train (pools per stage, default model roots, one `done` then `pipeline_done`, stop, the heatmap skip) and `/api/paths/scan`, the dashboard's HTTP and WebSocket routes through `TestClient` inside the app's lifespan (`unit/web/test_routes.py`), Calib and Collect always ending with `pipeline_done`, the `wavetrace train --stage weapon` backend flag (`unit/test_cli_train.py`), each pool's dataset finder, the training backends, the manifest-only dataset labels and the pooled `dataset_report`, `build_synthetic_source` |

Plain helpers shared across test files live in `fixtures/synthetic_csi.py`: raw wire-format frames
(`encode_frame`, `generate_raw_frames`) and the quiet empty-room baseline several calibrating tests share
(`generate_quiet_baseline`, `calibrate_quiet_baseline`).

Tests import those helpers as `fixtures.synthetic_csi` and an operator script as `scripts.<name>`
(`from scripts import collect_count`). `pythonpath = ["."]` in `pyproject.toml` puts the repo root on the
test path, so both resolve under plain `pytest` as well as `python -m pytest`.
