# `tests/`

Automated test suite. All tests run on synthetic or recorded data — no hardware required.

```bash
pytest tests/ -q    # run everything; expect 378 tests, all passing
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
| `golden/test_golden_pipeline.py` | Characterization tests pinning calibrate → collect → train → infer on committed fixture recordings (`golden/fixtures/`, `golden/expected/`) |
| `unit/` | Unit tests for backend registry, contracts, node-health gain drift against calibration, YOLO labeler options, occupancy, pipeline-contract persistence, serving artifact contract, source options, web server options |
| `conftest.py` | Shared fixtures (synthetic CSI, synthetic recordings) |

Plain helpers shared across test files live in `fixtures/synthetic_csi.py`: raw wire-format frames
(`encode_frame`, `generate_raw_frames`) and the quiet empty-room baseline several calibrating tests share
(`generate_quiet_baseline`, `calibrate_quiet_baseline`).
