# `src/hardware/`

Turns raw UDP bytes into `CsiFrame` objects.

| File | What it does |
|---|---|
| `FrameParser.hpp` | Decodes a binary UDP datagram into a `CsiFrame` (handles wire format v1/v2/v3, checks length and version byte) |
| `NodeAggregator.hpp` | Buffers frames from multiple nodes and re-orders them by timestamp so downstream code sees a consistent view across all links |
