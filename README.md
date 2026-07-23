# WaveTrace

Camera-free Wi-Fi CSI sensing over a self-organizing ESP32 mesh, with four operating modes on one shared signal front-end: **presence detection**, **people counting**, **camera-supervised occupancy heatmap**, and **concealed-weapon research**. They differ only in the classification head.

## Prerequisites

```
Python 3.10+
CMake 3.16+ and a C++ compiler (gcc or clang)
ESP-IDF v5.3      firmware flashing
Node.js 18+       web dashboard (optional)
ffmpeg            camera features (optional — brew install ffmpeg)
ultralytics       camera ground-truth labeling only (optional — pip install ultralytics)
```

`ultralytics` is only needed for `scripts/collect_camera.py` (YOLO-based labeling). It's not a core dependency — `pip install -e .` won't install it, and nothing else in the pipeline imports it. If you run `collect_camera.py` without it, you'll get an `ImportError` telling you to `pip install ultralytics`. There's no weights file to download by hand either: passing a model name like `yolov8n-seg.pt` triggers an automatic download the first time it's used.

## Installation

```bash
git clone <repo-url>
cd WaveTrace
python3 -m venv .venv && source .venv/bin/activate
pip install -e .     # builds the C++ extension and installs the wavetrace package
brew bundle          # macOS system dependencies (Brewfile)
```

`data/` is git-ignored, so a fresh clone won't have it. The `collect_*.py` scripts create it (and the profile subfolders under it) automatically the first time you pass `--root data/<profile>` — you don't need to create it by hand, just make sure the scripts are run from the project root so the relative path resolves correctly.

## Usage

Flash the firmware first (see [firmware/README.md](firmware/README.md)), then run everything from the project root:

```bash
python scripts/ntp_server.py                              # keep running — boards use this as their clock
python scripts/mesh_verify.py                             # confirm CSI is arriving from each node
python scripts/collect_baseline.py --root data/2g4_ht40  # calibrate (empty room, ~30 s)
python scripts/collect_presence.py --root data/2g4_ht40  # collect data and train
python scripts/run_live_mesh.py    --root data/2g4_ht40  # live presence detection
```

Full walkthrough with expected outputs: [Documentation.md](Documentation.md).

## What it does

WaveTrace reads Wi-Fi Channel State Information (CSI) from a mesh of ESP32 boards. When people move through the room, the signal changes in a measurable way. Every mode shares the same C++ front-end (preprocess → features) and swaps only the head:

- **Presence** — is a person in the room? Yes or no. This is the fully proven mode: **98.5% accuracy** on real captures under leave-one-session-out validation. It runs live by voting across the nodes, with each node weighted by its own measured accuracy, and each verdict takes about 0.05 ms.
- **People counting** — how many people are moving? A four-class head (0, 1, 2, 3+) runs on each node and the links are combined into one count. On real hardware it reaches **53–61% accuracy where random guessing scores 25%**, so it more than doubles the baseline. Collecting more sessions raises it further.
- **Occupancy heatmap** — where in the room is the person? A small CNN turns the CSI into a 16×16 occupancy grid. During collection a webcam labels the data on its own with a YOLO segmentation model, and those labels are time-aligned to each node's CSI windows, so no hand-labeling is needed. The camera is only used for training; the running system needs no camera. The pieces are built and covered by tests, and the live labeling works during capture. What remains is one full webcam-to-model run on real hardware; until then the live view uses a simpler energy-based estimate.
- **Weapon (research)** — is the person carrying concealed metal? An open research problem. It measures how flat the signal is across frequencies, since metal reflects evenly. At 2.4 GHz with the omnidirectional antennas in line of sight, no reliable body-worn signal has appeared yet (see [Documentation.md](Documentation.md) §11).

The modes are independent. There is no gate between them.

## Structure

| Path | What it is |
|---|---|
| `firmware/esp32_node/` | unified mesh firmware; `NODE_ID` is the only per-board difference |
| `firmware/pi/` | Nexmon CSI capture and UDP stream scripts for the Raspberry Pi |
| `src/` | C++ signal processing compiled into the package via pybind11 |
| `wavetrace/` | Python library: calibration, pipeline, training, inference, CLI |
| `scripts/` | terminal scripts: calibrate, collect data, run live detection (see [scripts/README.md](scripts/README.md)) |
| `experiments/` | offline analysis: σ²[p] litmus check, model bake-off |
| `web/` | FastAPI backend and React dashboard |
| `tests/` | pytest suite (~295 tests, all offline) |
| `data/` | captured data, models, calibration (git-ignored) |

## Tests

```bash
pytest tests/ -q    # no hardware needed
```
