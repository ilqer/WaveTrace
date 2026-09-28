# `wavetrace/diagnostics/`

Runtime health monitoring for the WaveTrace system.

## Files

| File | What it does |
|---|---|
| `telemetry.py` | Collects per-node metrics (frames/s, free heap, uptime) from the UDP health-monitor port (9877) and exposes them for `health_monitor.py` and the web dashboard. |
| `weapon_separation.py` | Static σ²[p] PDF separation diagnostics — the weapon-detection go/no-go check shared by `experiments/weapon_litmus.py` and the web dashboard's `/api/weapon/litmus` route. |
