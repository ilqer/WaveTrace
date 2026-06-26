# WaveTrace

Wi-Fi CSI sensing for human presence detection and concealed-weapon detection using an ESP32 mesh.

## Prerequisites

```
Python 3.10+
CMake 3.16+ and a C++ compiler (gcc or clang)
ESP-IDF v5.3      firmware flashing
Node.js 18+       web dashboard (optional)
ffmpeg            camera features (optional — brew install ffmpeg)
```

## Installation

```bash
git clone <repo-url>
cd WaveTrace
python3 -m venv .venv && source .venv/bin/activate
pip install -e .     # builds the C++ extension and installs the wavetrace package
brew bundle          # macOS system dependencies (Brewfile)
```

## Usage

Flash the firmware first (see [firmware/README.md](firmware/README.md)), then:

```bash
python ntp_server.py                              # keep running — boards use this as their clock
python mesh_verify.py                             # confirm CSI is arriving from each node
python collect_baseline.py --root data/2g4_ht40  # calibrate (empty room, ~30 s)
python collect_presence.py --root data/2g4_ht40  # collect data and train
python run_live_mesh.py    --root data/2g4_ht40  # live presence detection
```

Full walkthrough with expected outputs: [Documentation.md](Documentation.md).

## What it does

WaveTrace reads Wi-Fi Channel State Information (CSI) from a mesh of ESP32 boards. When a person enters the room, the signal changes in a measurable way. A trained model classifies each time window as occupied or empty.

A second independent mode detects whether the person is carrying concealed metal. It uses per-packet inter-subcarrier variance (σ²[p]) as the main discriminator. The two modes share the same signal-processing front-end and differ only in the classification head.

## Structure

| Path | What it is |
|---|---|
| `firmware/esp32_node/` | unified mesh firmware; `NODE_ID` is the only per-board difference |
| `firmware/pi/` | Nexmon CSI capture and UDP stream scripts for the Raspberry Pi |
| `src/` | C++ signal processing compiled into the package via pybind11 |
| `wavetrace/` | Python library: calibration, pipeline, training, inference, CLI |
| `collect_*.py` / `run_*.py` | top-level scripts you run at the terminal |
| `web/` | FastAPI backend and React dashboard |
| `tests/` | pytest suite (~295 tests, all offline) |
| `data/` | captured data, models, calibration (git-ignored) |

## Tests

```bash
pytest tests/ -q    # no hardware needed
```
