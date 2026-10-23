# `wavetrace/diagnostics/`

Runtime health monitoring for the WaveTrace system.

## Files

| File | What it does |
|---|---|
| `telemetry.py` | `NodeHealthMeter`: rolling per-node link health for the dashboard's `/ws/telemetry` (rate, loss, SNR, cross-subcarrier variation). Given each node's loaded calibration it also reports `gain_drift` (live mean \|H\| / calibrated quiet-room mean \|H\|; 1.0 = gain unchanged, the UI flags \|drift − 1\| > 0.2) and that node's NBVI `subcarriers`; a node with no calibration reports `null` / `[]`. Also `cluster_sync`, `baseline_drift`, and `dataset_report`, the pooled class balance, sessions and subjects of several datasets (the dashboard's `train_init`). |
| `weapon_separation.py` | Static σ²[p] PDF separation diagnostics — the weapon-detection go/no-go check shared by `experiments/weapon_litmus.py` and the web dashboard's `/api/weapon/litmus` route. |
