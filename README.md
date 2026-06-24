# WaveTrace

WaveTrace uses Wi-Fi Channel State Information (CSI) to detect human presence and concealed weapons in a room without a camera. Nodes (ESP32 boards + a Raspberry Pi) stream CSI over UDP; a Mac host processes the signal in C++ and runs machine learning models in Python.

For a full explanation, architecture diagram, bring-up steps, and glossary, read [Documentation.md](Documentation.md).

## Quick start

```bash
# 1. create and activate the Python venv
python3 -m venv .venv
source .venv/bin/activate

# 2. build the C++ extension and install the package
pip install -e .

# 3. flash the ESP32 boards (see firmware/README.md), then verify
python mesh_verify.py       # expect links 1->2 and 2->1 at a non-zero rate

# 4. calibrate → collect → run
python collect_baseline.py  --root data/2g4_ht40
python collect_presence.py  --root data/2g4_ht40
python run_live_mesh.py     --root data/2g4_ht40
```

Full walkthrough with expected outputs and hardware setup: [Documentation.md](Documentation.md).

## Prerequisites

| Dependency | Purpose |
|---|---|
| Python 3.10+ + venv | ML and glue scripts |
| CMake + C++ compiler | builds `src/` into the `wavetrace` package |
| ESP-IDF v5.x | flashing ESP32 firmware |
| Node.js | web dashboard (optional) |

## Structure

| Path | What it is |
|---|---|
| `firmware/esp32_node/` | unified mesh firmware for every ESP32 board |
| `firmware/pi/` | Nexmon CSI capture + UDP stream scripts for the Pi |
| `src/` | C++ signal processing (conjugate-multiply, Hampel, NBVI, features) |
| `wavetrace/` | Python library: calibration, pipeline, training, inference, CLI |
| `collect_*.py` / `run_*.py` | top-level scripts you run at the terminal |
| `web/` | Flask + WebSocket backend and React dashboard |
| `tests/` | pytest suite (~271 tests, all offline) |
| `data/` | captured data, models, calibration (git-ignored) |

## Tests

```bash
pytest tests/ -q
```

All tests run on synthetic/recorded data — no hardware needed.
