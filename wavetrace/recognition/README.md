# `wavetrace/recognition/`

Machine learning for WaveTrace. Takes feature vectors or spectrograms from the pipeline and produces a class label + confidence.

## Files

| File | What it does |
|---|---|
| `train.py` | Trains a presence or weapon head from a `.npy` dataset; saves `model.joblib` |
| `evaluate.py` | LOGO cross-validation (leave-one-session-out, leave-one-subject-out); reports confusion matrix and per-tier verdicts |
| `infer.py` | `measure_latency` (budget 8 ms per window) |
| `fusion.py` | `fuse()` — concatenates per-node feature vectors; O(m) for m nodes |
| `link.py` | Per-(tx, rx)-link model and calibration bookkeeping |
| `stack.py` | Stacks per-node windows into a batch tensor for the CNN path |
| `cir.py` | Offline CIR super-resolution via L1/ISTA (optional Stage-E tool) |
| `heatmap.py` | Camera-supervised G×G occupancy heatmap head |
| `adapt.py` | Session adaptation: norm-stat refresh on new calibration data |
| `explain.py` | Feature importance and SHAP-style model introspection |
| `occupancy.py` | Temporal fusion of heatmap frames: per-cell Bayesian filter (`OccupancyGrid`) + 2D position Kalman smoother (`HeatmapTrack`) |
| `count.py` | Display naming for the people-count head's classes (`count_name`) |
| `serving.py` | Per-link serving math shared by every live front end (presence, count and weapon) |
| `weapon_serving.py` | Live multi-link weapon serving: loads the per-link/per-node weapon heads a training run wrote, plus shared per-window/per-link math |
| `serve_modes.py` | The presence, count and weapon `ServeMode`s `ServeMeshUseCase` runs |
| `collect_modes.py` | The presence, count and weapon collect modes `CollectMeshUseCase` runs (session plan, per-link dataset layout, head training); `WeaponTrainingPolicy` is the weapon training half (backend and epoch report optional); each pool's `find_dataset_dirs_by_node()` finds what the mode writes; `default_model_root` is every stage's default head dir, shared with the dashboard's Train |
| `camera_collect_modes.py` | The camera collect modes `CollectCameraUseCase` runs: per-link presence datasets under `cam_ds/presence` with heads in `model_camera` (`CameraPresenceTrainingPolicy`), or per-link weapon datasets into the `weapon_ds` pool trained like `collect_weapon`; both train the heatmap head (`train_heatmap_head`); `default_camera_model_root`, `find_heatmap_capture_root` |

## Evaluation rule

Always use **leave-one-session-out AND leave-one-subject-out** (`evaluate.py`). A random within-session split leaks 97–99 % because consecutive frames are nearly identical. Report the cross-session/subject number or it doesn't count.
