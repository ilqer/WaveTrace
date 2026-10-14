# `wavetrace/output/`

Handles what happens after the model produces a prediction.

## Files

| File | What it does |
|---|---|
| `publisher.py` | `Publisher` abstract base class + `JsonlPublisher` (default, zero extra dependencies) + WebSocket seam. Each prediction is serialized as one JSON line: `{t, class, conf, mode, bbox, keypoints}`. |
