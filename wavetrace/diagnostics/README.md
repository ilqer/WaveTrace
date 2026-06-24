# `wavetrace/diagnostics/`

Runtime health monitoring for the WaveTrace system.

## Files

| File | What it does |
|---|---|
| `Telemetry.py` | Collects per-node metrics (frames/s, free heap, uptime) from the UDP health-monitor port (9877) and exposes them for `health_monitor.py` and the web dashboard. |
