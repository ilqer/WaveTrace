# `src/core/`

Shared types used across the entire C++ backend.

| File | What it defines |
|---|---|
| `CsiFrame.hpp` | One snapshot of CSI data: complex amplitude per antenna per subcarrier, timestamp, node ID |
| `Types.hpp` | Common type aliases (e.g. `complex64` vectors) |
| `Errors.hpp` | Error types for parse failures and out-of-range values |
