# `wavetrace/recognition/`

Machine learning for WaveTrace. Takes feature vectors or spectrograms from the pipeline and produces a class label + confidence.

## Files

| File | What it does |
|---|---|
| `train.py` | Trains a presence or weapon head from a `.npy` dataset; saves `model.joblib` |
| `evaluate.py` | LOGO cross-validation (leave-one-session-out, leave-one-subject-out); reports confusion matrix and per-tier verdicts |
| `infer.py` | Runs `predict_window()` on a feature window; target latency < 8 ms |
| `vote.py` | `SegmentVoter` — accumulates per-window predictions over one motion segment and emits a single stable verdict |
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

## Evaluation rule

Always use **leave-one-session-out AND leave-one-subject-out** (`evaluate.py`). A random within-session split leaks 97–99 % because consecutive frames are nearly identical. Report the cross-session/subject number or it doesn't count.
