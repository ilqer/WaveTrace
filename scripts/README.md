# scripts/

These are the terminal scripts you run to do things: calibrate the hardware, collect training data, and run live detection. They are not part of the `wavetrace` library — they are just entry points you call from the command line.

## Run them from the project root

Always run these from the top-level `WaveTrace/` folder, not from inside `scripts/`:

```bash
# correct — from WaveTrace/
python scripts/mesh_verify.py

# wrong — from inside scripts/
cd scripts && python mesh_verify.py   # wavetrace imports will fail
```

The scripts import from the `wavetrace` package, which lives one level up. If you run them from inside `scripts/`, Python won't find it.

---

## Before anything else

Get the venv active and the NTP server running:

```bash
source .venv/bin/activate
python scripts/ntp_server.py    # keep this in its own terminal — boards use it as a clock
python scripts/health_monitor.py   # optional, shows per-node status
python scripts/mesh_verify.py      # confirm CSI links are arriving before you collect anything
```

`mesh_verify.py` should print something like `aa:bb->1:45  cc:dd->2:38` — one line per second, with non-zero frame rates. If it's silent, the boards aren't reaching the Mac (wrong `PC_IP` in `config.h`, firewall, or boards not powered).

---

## The usual session order

Every session needs a fresh calibration. If you moved hardware, changed the band, or started a new day — recalibrate before collecting.

### 1. Calibrate

```bash
python scripts/collect_baseline.py --root data/2g4_ht40
```

Room must be completely empty and still. Takes about 30 seconds. Saves to `data/2g4_ht40/cal/node{id}/`.

### 2a. Presence

```bash
python scripts/collect_presence.py --root data/2g4_ht40
python scripts/run_live_mesh.py    --root data/2g4_ht40
```

`collect_presence.py` walks you through it — it prompts you to keep the room empty, then to walk around. It trains after each session pair. `run_live_mesh.py` loads whatever model is there and runs live; expect `PRESENT` / `absent` printed every ~1.5 s.

### 2b. Weapon

```bash
python scripts/collect_weapon.py --root data/2g4_ht40 --subject p0 --carry chest
python scripts/run_weapon.py     --root data/2g4_ht40
```

Run `collect_weapon.py` once per person and carry position. It appends to the cumulative pool and retrains from scratch each time. Vary `--subject` and `--carry` across runs — that's what makes the model generalize.

### 2c. Camera-supervised (presence + heatmap)

```bash
python scripts/collect_camera.py --root data/2g4_ht40 --duration 30 --train
```

Records CSI and webcam at the same time. YOLO-seg labels each webcam frame in real time and aligns those labels to the CSI windows. Pass `--stage weapon` for open-carry weapon data (YOLO can see a knife if it's visible — it can't see a concealed one).

### 2d. People count

```bash
python scripts/collect_count.py --root data/2g4_ht40 --max-count 3
python scripts/run_count.py     --root data/2g4_ht40 --max-count 3
```

You capture levels 0, 1, 2, 3 (and 3 means "3 or more"). The script prompts for each count level in turn.

---

## What each script does

| Script | What it does |
|---|---|
| `collect_baseline.py` | Records a quiet empty room and saves the per-node calibration: gain-lock scalar, NBVI subcarrier mask, quiet baseline magnitude. Run this first. |
| `collect_presence.py` | Walks through empty/occupied session pairs and trains a per-node presence head. |
| `collect_weapon.py` | Records clear/weapon session pairs and retrains the weapon head on the full cumulative pool. |
| `collect_count.py` | Records one session per people count (0…N) and trains a per-node multi-class count head. |
| `collect_camera.py` | Simultaneous CSI + webcam capture with live YOLO-seg labeling. Builds both per-node presence datasets and a stacked heatmap dataset. |
| `run_live_mesh.py` | Live presence detection over all active (tx, rx) links. Votes are weighted by each node's LOGO accuracy. |
| `run_weapon.py` | Live weapon detection. Loads per-link or per-node weapon heads and fuses them via LinkVoter. |
| `run_count.py` | Live people count. Same architecture as presence, multi-class. |
| `mesh_verify.py` | Listens on UDP 9876 and prints which directed links are arriving and at what rate. Use this to confirm hardware is working before collecting. |
| `health_monitor.py` | Live table of per-node uptime, CSI rate, gain state, RSSI, free heap. Refreshes every second. |
| `ntp_server.py` | SNTP server on UDP 123. The ESP32 firmware uses the Mac as its clock source for cross-node timestamp alignment. Keep it running during every session. |

---

## The --root flag

Every collect and run script takes `--root` to select a capture profile. Data from different profiles cannot be mixed — a model trained on one is invalid on another.

```
data/2g4_ht40/   ← current default (2.4 GHz, 40 MHz, ~114 subcarriers)
data/2g4_ht20/   ← legacy (2.4 GHz, 20 MHz, ~52 subcarriers)
data/5g_ht40/    ← Pi/Nexmon, 5 GHz HT40 (not yet validated)
data/5g_ht80/    ← target for weapon (5 GHz, 80 MHz, ~256 subcarriers)
```

`--root` defaults to `data/` if you omit it — data lands at the top level. Always pass the profile explicitly so calibration, datasets, and models all nest under the same directory.

`--cal` and `--model` derive from `--root` automatically unless you override them.
