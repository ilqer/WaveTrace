# `wavetrace/recognition/`

Machine learning for WaveTrace. Takes feature vectors or spectrograms from the pipeline and produces a class label + confidence.

## Files

| File | What it does |
|---|---|
| `Train.py` | Trains a presence or weapon head from a `.npy` dataset; saves `model.joblib` |
| `Evaluate.py` | LOGO cross-validation (leave-one-session-out, leave-one-subject-out); reports confusion matrix and per-tier verdicts |
| `Infer.py` | Runs `predict_window()` on a feature window; target latency < 8 ms |
| `Model.py` | Head classes: `PresenceHead` (MLP default / SVM selectable) and `WeaponHead` (variance threshold / sklearn / CNN) |
| `Weapon.py` | Weapon-specific features: σ²[p] inter-subcarrier variance, `reconstruct_complex_csi`, block-average decimation |
| `Vote.py` | `SegmentVoter` — accumulates per-window predictions over one motion segment and emits a single stable verdict |
| `Fusion.py` | `fuse()` — concatenates per-node feature vectors; O(m) for m nodes |
| `Link.py` | Per-(tx, rx)-link model and calibration bookkeeping |
| `Resample.py` | Uniform-grid resampler to handle timing jitter between nodes (100 Hz target) |
| `Stack.py` | Stacks per-node windows into a batch tensor for the CNN path |
| `Cir.py` | Offline CIR super-resolution via L1/ISTA (optional Stage-E tool) |
| `Heatmap.py` | Camera-supervised G×G occupancy heatmap head |
| `Adapt.py` | Session adaptation: norm-stat refresh on new calibration data |
| `Explain.py` | Feature importance and SHAP-style model introspection |
| `Occupancy.py` | Occupancy-specific head helpers |

## Evaluation rule

Always use **leave-one-session-out AND leave-one-subject-out** (`Evaluate.py`). A random within-session split leaks 97–99 % because consecutive frames are nearly identical. Report the cross-session/subject number or it doesn't count.
