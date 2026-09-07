# `wavetrace/output/`

Handles what happens after the model produces a prediction.

## Files

| File | What it does |
|---|---|
| `Publisher.py` | `Publisher` abstract base class + `JsonlPublisher` (default, zero extra dependencies) + WebSocket seam. Each prediction is serialized as one JSON line: `{t, class, conf, mode, bbox, keypoints}`. |
| `Guard.py` | Deduplicates and rate-limits output events. Prevents rapid oscillation (e.g. Present → Empty → Present within one second) from generating noisy output. |
