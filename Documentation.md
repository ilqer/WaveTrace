# WaveTrace — Project Documentation

---

## 1. What WaveTrace is

WaveTrace uses Wi-Fi Channel State Information (CSI) to sense what is happening in a room without a camera. Wi-Fi signals bounce off everything in a room. When a person or object moves, those reflections change in a measurable way. By analyzing those changes — per-frequency, per-packet — you can detect whether a person is present, and eventually whether they are carrying concealed metal.

The project has two independent operating modes:

- **Presence** — is a person in the room? Binary output: occupied / empty.
- **Weapon** — is a person carrying a concealed weapon (knife, gun, or similar metal object)?

Weapon detection is the main goal. Presence is the easier preliminary that proves the pipeline works before you tackle the harder problem.

---

## 2. The two modes

**Presence** is a binary classification problem. The model gets a short window of CSI frames, extracts statistical features, and outputs 0 or 1. It trains in a few minutes on a laptop and achieves good accuracy on real data. This is the starting point.

**Weapon detection** is harder for a physical reason: a concealed metal object is much smaller than the human body it is attached to, so its signal rides on top of the person's much stronger reflection. The key discriminator is per-packet inter-subcarrier variance σ²[p]. Metal reflects all subcarriers evenly, which produces lower variance. A human body scatters the signal unevenly, which produces higher variance. This difference is detectable but subtle — sensor placement and geometry matter a lot.

The two modes are **fully independent**. Weapon mode does not use the presence model as a pre-filter. They share the signal processing front-end (preprocess → features) and differ only in the classification head.

---

## 3. How it fits together

```
ESP32 mesh (2.4 GHz)  ─┐
                        ├──► UDP :9876 ──► Mac host
Pi 5 GHz nexmon       ─┘         │
                                  ▼
                        C++ front-end
                        (conjugate-multiply, Hampel filter,
                         gain lock, NBVI subcarrier select,
                         feature extraction / spectrogram)
                                  │
                                  ▼
                        Python model
                        (presence head OR weapon head)
                                  │
                                  ▼
                        verdict + confidence
                        (printed / published to dashboard)
```

**Nodes (ESP32):** The mesh runs a time-division round-robin. In each turn, one node broadcasts an ESP-NOW burst; every other node captures CSI from it. The TX role rotates. N nodes produce N·(N−1) directed (tx, rx) links per cycle. Every board runs the same firmware — `NODE_ID` is the only per-board difference.

**Pi node:** The Raspberry Pi captures 5 GHz CSI on its onboard CYW43455 chip via Nexmon and streams it to the Mac as node 5, on the same UDP port and in the same wire format as the ESP nodes. The host treats it like any other node.

**Mac host:** Decodes the binary UDP datagrams in C++ (fast path), preprocesses, extracts features, and runs a scikit-learn or PyTorch model. Training is also done on the Mac from recorded data.

**Training flow:** You record sessions with `collect_*.py`, which saves raw CSI to disk. After recording, the script trains a model and saves it as `model.joblib`. The `run_*.py` scripts load that file and run inference on live UDP data.

**Web dashboard:** An optional React UI (`web/ui/`) that shows live spectrograms, per-node health, and predictions over a WebSocket. Useful for demos and debugging, not required for the sensing pipeline.

---

## 4. Prerequisites

### Hardware

| Item | Qty | Notes |
|---|---|---|
| ESP32-S3-DevKitC-1 | 6 | All six run the same firmware. The only per-board difference is `NODE_ID`. Status LED is GPIO 38 (v1.1 board). |
| 8dBi RP-SMA omnidirectional antenna | 6 | 160 mm whip, 2.4/5.8 GHz dual-band, vertical polarization, 50 Ω, RP-SMA male connector. One per board. These are **omnidirectional** (radiate in all horizontal directions), not directional — see the note below. |
| Raspberry Pi 5 | 1 | For the 5 GHz arm. Nexmon CSI on Pi 5 requires a community patch — it is not in the official Nexmon repo. See [`firmware/pi/README.md`](firmware/pi/README.md) for the current status. |
| Dedicated Wi-Fi router | 1 | Locked to channel 6, 2.4 GHz, 40 MHz (HT40). Do not use a router shared with regular traffic — bursty traffic corrupts the calibration baseline. |
| USB cables | 1 per ESP32 | For flashing only. After flashing, boards run from any 5 V supply. |
| 5 V power supply | 1 per ESP32 | Standard USB power banks or phone chargers work fine. |
| Ethernet cable | 1 | Wired connection from the Pi to the router. The Pi's Wi-Fi radio is in monitor mode for CSI — it cannot simultaneously carry the data backhaul. |

**Antenna note:** The 8dBi omnidirectional antennas are a significant upgrade over the bare PCB trace antennas the boards ship with, and they work well for presence detection. For weapon detection, the research literature uses directional antennas (horn/dish, ≥9 dBi, narrow beam) to focus the signal on the subject and reduce background reflections. The round-robin TX rotation across 6 boards provides spatial diversity across 30 directed links (6×5), which partially compensates — different links cover different angles. The system is likely to work for presence with the current antennas; weapon detection performance relative to the published numbers (which used directional hardware) is an open question to validate on the first real dataset.

### Software

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Main host language |
| CMake + C++ compiler (GCC or Clang) | CMake 3.16+ | Compiles `src/` into the Python package |
| ESP-IDF | v5.3 | Flashes the ESP32 firmware |
| Node.js | 18+ | Web dashboard only (optional) |
| numpy, scikit-learn | via `pip install -e .` | Signal processing and ML |
| torch | via `pip install 'wavetrace[cnn]'` | CNN backend only (optional) |

### Network setup

- The router is locked to **channel 6, 2.4 GHz, 40 MHz (HT40)**. All ESP32 boards join it as STAs; this automatically locks every board to channel 6 so they hear each other's ESP-NOW traffic. Do not change the channel after flashing without a full wipe-and-reflash of the boards (stale Wi-Fi NVS caches the old channel).
- Give the Mac a **static DHCP lease** on the router so the IP is always the same. Set that IP as `PC_IP` in `config.h`. If the IP changes, the boards send CSI into the void.
- The Mac firewall must allow incoming UDP on these ports. Check under `System Settings → Network → Firewall → Options`:
  - **9876** — CSI datagrams from all nodes
  - **9877** — per-node health heartbeats
  - **9878** — node discovery (boards ping this port to learn the current PC IP)
- Run `python ntp_server.py` on the Mac **before** running any collect or live script. The ESP32 firmware uses the Mac as its SNTP clock source (`SNTP_SERVER = PC_IP` in `config.h`). Without it, nodes fall back to their own monotonic clock — single-link presence still works, but cross-node timestamp alignment for fusion is unreliable.
- Keep the sensing network on a **dedicated router** — not your home or office network. Bursty traffic from other devices causes AGC swings that corrupt the calibration baseline.

---

## 5. Where things live

```
firmware/
├── esp32_node/     unified mesh firmware (one binary; NODE_ID is the only per-board difference)
└── pi/             Nexmon CSI capture + UDP stream scripts for the Raspberry Pi

src/                C++ signal processing (compiled into the wavetrace package via pybind11)
├── core/           CsiFrame type (one frame = complex amplitude per subcarrier + timestamp + node_id)
├── hardware/       UDP datagram parser + multi-node frame assembler
├── signal/         conjugate-multiply, Hampel filter, phase unwrap, gain lock, NBVI, features, spectrogram
└── util/           radix-2 FFT, ring buffers

wavetrace/          the Python library
├── Source.py       CSI frame sources: live UDP, recorded file, synthetic (tests)
├── Frontend.py     shared pipeline loop: source → preprocess → features → emit windows
├── Calibration.py  saves/loads per-session calibration: gain-lock scalar, NBVI mask, quiet baseline
├── Cli.py          `wavetrace` CLI: calibrate / collect-data / train / run
├── Config.py       runtime config: mode, backend, head, subcarrier count, window/hop sizes
├── recognition/    training, inference, voting, multi-node fusion, evaluation
├── groundtruth/    camera labeler, timestamp alignment, dataset serializer
├── output/         result publisher (JSONL default; WebSocket seam for the dashboard)
└── diagnostics/    per-node health telemetry

collect_baseline.py   record quiet empty room → saves calibration data under data/<profile>/cal/
collect_presence.py   record presence sessions → trains per-node presence models
collect_weapon.py     record weapon sessions (open/wrapped/concealed) → trains weapon model
collect_count.py      record people-count sessions → trains count model

run_live_mesh.py      live presence detection (all nodes voted)
run_weapon.py         live weapon detection
run_count.py          live people count

mesh_verify.py        listen on UDP 9876 and print which (tx, rx) links are arriving and at what rate
health_monitor.py     per-node uptime, free heap, frame rate — refreshes every second

web/
├── streamer.py       WebSocket backend: pushes live CSI + predictions to the browser
├── app.py            Flask routes
├── device_ctl.py     device control endpoints (flash, recalibrate, upload model)
└── ui/               React + Vite + TypeScript dashboard

tests/                pytest suite (~283 tests; all offline on synthetic data — no hardware needed)
data/                 git-ignored; created by the collect scripts
```

---

## 6. Run the simplest thing end to end

All commands run from the repo root. Start with two ESP32-S3 boards and a dedicated router.

---

### A. Python setup (Mac — once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .               # compiles the C++ extension and installs the wavetrace package
pip install 'wavetrace[cnn]'   # optional: enables the CNN training backend
```

Run `source .venv/bin/activate` again whenever you open a new terminal.

---

### B. Flash the ESP32 boards

Full steps are in [`firmware/README.md`](firmware/README.md). Summary:

1. Install ESP-IDF v5.3 (once, ~30–60 min).
2. Open `firmware/esp32_node/main/config.h`. Set `WIFI_SSID`, `WIFI_PASS`, `PC_IP` (your Mac's IP — run `ipconfig getifaddr en0`), and `MESH_NODES 2`.
3. Flash board 1 (node 1), then board 2 (node 2). Plug in one at a time.

After flashing, the boards only need a 5 V power supply. The USB cable is only needed for flashing.

---

### C. Start the NTP server (keep this running throughout)

```bash
python ntp_server.py
```

Keep this running in a dedicated terminal whenever the boards are on. Boards use the Mac as their SNTP source. Without it they fall back to their own clock — presence on a single link still works, but multi-node timestamp fusion becomes unreliable.

### D. Verify hardware

```bash
python mesh_verify.py
```

With 6 nodes, expect 30 links (6×5 directed pairs), each at a non-zero frame rate. Two nodes gives 2 links (`1->2` and `2->1`). Typical rate per link: 20–100 Hz depending on channel load.

If nothing appears: wrong `PC_IP` in `config.h`, or the Mac firewall is blocking UDP 9876, or the boards haven't associated yet (give them ~10 s after power-on).

```bash
python health_monitor.py    # per-node uptime and free heap; Ctrl-C to stop
```

---

### E. Calibrate (empty room)

```bash
python collect_baseline.py --root data/2g4_ht40
```

The room must be **completely empty and still** — no people, no movement, no fans. This is not optional. The calibration records what the empty room looks like and locks the AGC gain so later captures are comparable. It takes about 30 seconds.

Output goes into `data/2g4_ht40/cal/`. The `--root` flag selects the capture profile (see §7). Use the same `--root` for every step in a session. If you move the hardware, change the band, or change the room layout, you must recalibrate.

---

### F. Collect presence data and train

```bash
python collect_presence.py --root data/2g4_ht40
```

The script walks you through it: it records the empty room first, then tells you to walk around. Follow the on-screen prompts. When finished, it trains a per-node model and saves it to `data/2g4_ht40/model/`.

---

### G. Run live presence detection

```bash
python run_live_mesh.py --root data/2g4_ht40
```

Expected output: `PRESENT` or `EMPTY` printed every ~0.3 s. Votes from all active links are combined into one verdict.

---

### H. Weapon mode (only after presence works end to end)

```bash
python collect_weapon.py --root data/2g4_ht40 --subject p0 --carry chest
python run_weapon.py     --root data/2g4_ht40
```

`--subject p0` is an arbitrary label for the person (used for leave-one-subject-out evaluation). `--carry chest` is the carry position label for this session. Collect multiple sessions before training.

Do not skip to weapon mode if presence is not working. The pipeline is identical — if presence fails, something is wrong with calibration, gain lock, or the UDP stream, not the model.

---

### I. Web dashboard (optional)

```bash
python web/streamer.py      # WebSocket backend on port 8765 — keep this running
```

In a separate terminal:

```bash
cd web/ui
npm install                 # once
npm run dev                 # Vite dev server at http://localhost:5173
```

Note: the Train button in the dashboard currently fakes metrics. Real training is done from the terminal with the `collect_*.py` scripts.

---

## 7. RF configuration — which profile to use

Data from different radio configurations cannot be mixed. A model trained on HT20 data is invalid on HT40, and vice versa. Every `collect_*.py` and `run_*.py` script takes `--root` to pick a profile; all data (calibration, recordings, datasets, models) for that profile lives under that directory.

| Profile (`--root`) | Band | Bandwidth | Subcarriers | Hardware | Status |
|---|---|---|---|---|---|
| `data/2g4_ht20` | 2.4 GHz | 20 MHz | ~52 | ESP32-S3 | legacy only |
| `data/2g4_ht40` | 2.4 GHz | 40 MHz | ~114 | ESP32-S3 | **current active** |
| `data/5g_ht40` | 5 GHz | 40 MHz | ~114 | Pi / Nexmon | not yet used |
| `data/5g_ht80` | 5 GHz | 80 MHz | ~256 | Pi / Nexmon | target for weapon |

**Current setup:** The firmware has `WT_BW_HT40 1` enabled and the router is set to 40 MHz, so all ESP32 captures go into `data/2g4_ht40`. Use this profile for all collect and run commands until the Pi node is ready.

**For weapon detection:** 5 GHz HT80 (`data/5g_ht80`) is better suited because the shorter wavelength (~60 mm vs 125 mm at 2.4 GHz) is closer to the size of a knife or gun, making the metal reflection more measurable. The Pi is configured for channel 36 / HT80 / 256 subcarriers and emits wire format v3 (int16, fixed scale) to preserve absolute amplitude. However, Pi 5 Nexmon support is a community patch and is not yet validated on this hardware — see [`firmware/pi/README.md`](firmware/pi/README.md).

**Do not mix profiles.** If you switch band or bandwidth, recalibrate from scratch. The old `cal/` directory is for a different subcarrier layout and will produce wrong gain-lock values.

**2.4 GHz vs 5 GHz tradeoff:** 2.4 GHz diffracts around bodies better, so links are less likely to be completely blocked in a crowded room. 5 GHz attenuates harder around bodies but gives a stronger material signature for small metal objects. For a crowded-entry deployment (airport/school) the plan is to collect on both and pick the operating point from real data.

---

## 8. Data collection protocol

Bad data collection produces a model that looks good on paper and fails in the real world. Follow these rules.

### Room and hardware setup

- **Geometry: non-LOS by default.** Do not place TX and RX facing each other with the subject walking between them (LOS-blocking doorway). A strong direct path masks a small weapon's signal. Instead, arrange nodes so the measured signal comes from reflections off walls and the subject, not the direct TX→RX path.
- **Antenna.** The 8dBi omnidirectional whip antennas provide a solid RF improvement over bare PCB traces and cover all horizontal directions. For weapon detection, the round-robin TX rotation gives 30 directed links from 6 different TX positions, each illuminating the subject from a different angle — this spatial diversity is a partial substitute for a fixed directional TX. If weapon-tier accuracy plateaus, adding a directional antenna on the fixed TX node is the first hardware change to try.
- **Fixed hardware.** Do not move any node, antenna, or the router between calibration and data collection. Even a few centimeters changes the multipath signature. If you move anything, recalibrate.
- **Channel lock.** The router is locked to channel 6, 40 MHz. Do not change this between sessions. Confirm no nearby AP is on channel 6 (channels 4–8 overlap with it in 2.4 GHz) — interference causes amplitude spikes that the Hampel filter will mostly catch but a heavily congested channel degrades signal quality.

### Calibration rules

- Run `collect_baseline.py` at the start of every new session (new day, moved hardware, changed room, changed band).
- During calibration, the room must be completely empty. No people in the room, no movement visible through windows, no fans or AC vents causing air movement near the antennas.
- The calibration takes about 30 seconds. Do not touch or walk near the equipment during this time.

### Presence collection

- Collect at least 3–5 separate empty/occupied session pairs. More is better.
- "Occupied" sessions: walk naturally, change direction, sit down, stand up. Cover the whole area the system is expected to monitor.
- Each empty session and each occupied session should be roughly the same duration (~2–5 minutes each).
- Label the person ID (`--subject`) consistently across sessions. This enables leave-one-subject-out evaluation.

### Weapon collection

Follow the tier order (§9). Do not attempt tier 2 or 3 until tier 1 works.

- **Tier 1 (static, open weapon):** Subject stands still in the beam for ~30 seconds with the object held openly at chest level. This reproduces the lab setup from the literature and should be the easiest to detect. If this fails, fix the hardware (antenna, geometry) before proceeding.
- **Tier 2 (moving, real weapon):** Subject walks through the sensing area carrying the weapon concealed in clothing or a bag.
- **Tier 3 (truly concealed, scripted):** The weapon is in place before the session starts. Labels come from the operator's script (`--label-spans "start:end"`), not from a camera.

Collect multiple subjects and multiple carry positions (`chest`, `waist`, `bag`) for each tier.

---

## 9. Weapon detection — the three tiers

Weapon detection is feasibility-gated. Each tier is a checkpoint. If a tier fails, fix the hardware before moving to the next one — adding a more complex model on top of a broken sensor geometry will not help.

### Tier 1 — flat metal plate, non-LOS, static

Reproduce the setup from Yousaf et al. (2025): directional TX, non-LOS geometry, subject stands still with a flat metal plate at chest level. Run the σ²[p] variance threshold baseline:

```bash
python collect_weapon.py --root data/2g4_ht40 --subject p0 --carry chest
wavetrace train ./dataset --stage weapon --backend variance --out ./weapon_var
```

Target: the held-out session should show clearly separated σ²[p] distributions for metal vs no-metal. If the distributions overlap completely, the problem is hardware (antenna, geometry, gain lock) — not the model.

### Tier 2 — real weapon, static, non-LOS

Replace the metal plate with a real knife or gun. Use the CNN backend:

```bash
wavetrace train ./dataset --stage weapon --backend cnn --feature-mode cnn --out ./weapon_cnn
```

Target: FP ≤ 10% and TPR ≥ 90% on held-out sessions (`tier_verdict` checks this automatically).

### Tier 3 — moving subject, weapon concealed

The hardest case. The person is walking while carrying a concealed weapon. Requires:
- At least 2–3 RX nodes at different angles (angle-dependent multipath)
- CNN + soft majority voting (`--vote` flag in `run_weapon.py`)
- Multiple training sessions with the weapon concealed in different positions

```bash
wavetrace run --calibration ./calib --model ./weapon_cnn/model.joblib --head-mode weapon --vote
```

The voting gain is large only when the subject is moving. Static detection works without it.

---

## 10. How to evaluate results

### Use leave-one-session-out (LOGO)

A random train/test split on CSI data is meaningless. Consecutive frames in the same session are nearly identical — a model can learn to "recognize" the session rather than the class. This produces fake 97–99% accuracy that collapses on new data.

The correct evaluation is **leave-one-session-out (LOGO)**: train on all sessions except one, test on the held-out session, repeat for every session, and report the average. The `Evaluate.py` module does this automatically.

For weapon mode, also do **leave-one-subject-out**: hold out all sessions from one person, train on everyone else. This tests whether the model generalizes to a new person.

### What numbers to report

| Metric | How to compute | Minimum bar |
|---|---|---|
| Presence accuracy | LOGO cross-validation, all sessions | Clearly above majority-class baseline |
| Weapon TPR (true positive rate) | `tier_verdict` in `Evaluate.py` | ≥ 90% |
| Weapon FP rate (false positive rate) | `tier_verdict` | ≤ 10% |
| Per-tier verdict | Run separately for each tier (plate → gun → moving) | Beat chance on each tier before moving to the next |

### What "works" means at each stage

- **Presence**: the LOGO accuracy is clearly above the majority-class baseline (e.g. 60% empty → baseline is 60%; your model should be ≥ 85%).
- **Weapon tier 1**: σ²[p] distributions for metal vs no-metal are visually separated on held-out data.
- **Weapon tier 2**: `tier_verdict` returns True (FP ≤ 10% ∧ TPR ≥ 90%) on held-out data.
- **Weapon tier 3**: Same verdict gate, but evaluated on moving-subject sessions from held-out subjects.

Do not report accuracy from a random within-session split. It will look impressive and mean nothing.

---

## 11. Project status

### What is built and working (on synthetic/recorded data)

- Full C++ signal processing pipeline: frame parsing, conjugate-multiply, Hampel filter, phase unwrap, EMA detrend, gain lock, NBVI subcarrier selection, FFT, 9-feature extractor, inter-subcarrier σ²[p], spectrogram builder.
- Full Python presence pipeline: calibration, dataset builder, MLP/SVM training, LOGO evaluation, live inference, per-node voting, multi-node fusion, result publishing.
- Weapon pipeline (plumbing): σ²[p] variance baseline, sklearn head, CNN head (lazy-load), `SegmentVoter`, `tier_verdict` gate.
- Ground-truth tools: camera labeler (YOLO/SAM), segmentation labeler, scripted labeler, timestamp alignment, dataset serializer.
- CLI: `wavetrace calibrate / collect-data / train / run`.
- Web dashboard: spectrograms, node health, live predictions. (The Train button in the UI currently returns fake metrics — real training is done from the terminal.)
- Pi 5 GHz node: `firmware/pi/` is implemented and tested against `wavetrace/Source.py` via `TestPiPublisher.py`. Not yet validated on real Pi hardware.
- All 283 pytest tests pass offline.

### What is blocked on hardware

None of the accuracy numbers from synthetic data carry over to real hardware. The following must be verified on a real capture before trusting any results:

- **I/Q byte order** — esp-csi assumes `[imag, real]` pairs. A swap makes all phase data garbage. Verify against a known-still capture (amplitude should be stable; phase should not spin).
- **Subcarrier count and pattern** — with `WT_BW_HT40 1` the expected count is ~114 (HT40 on 2.4 GHz). Verify the actual buffer size on the first real capture. This also determines whether compressed sensing is usable later.
- **AGC / PHY gain lock** — verify the lock is engaged by checking that amplitude is stable across two back-to-back empty captures taken minutes apart.
- **Actual CSI sample rate** — the firmware targets ~250 Hz per link (burst of 10 frames, 2 ms apart, rotating among 6 nodes). The pipeline estimates `fs` from timestamps; confirm it is within range of the 100 Hz resample target.
- **Antenna performance on weapon detection** — the 8dBi omnidirectional whip is an upgrade over bare PCB but is not the directional horn used in the published weapon-detection papers. Tier 1 (flat metal plate, static) is the hardware reality check: if σ²[p] distributions don't separate, the antenna or geometry needs adjustment.
- **Pi 5 Nexmon** — Pi 5 is not in the official Nexmon CSI repo. A community patch exists but is not validated on this hardware. Pi setup is deferred until the ESP mesh baseline is confirmed working.
- **All weapon accuracy tiers (7a–7d)** — synthetic numbers are plumbing only. Real accuracy requires real recordings with real metal objects.

### What is not built and not planned

- AoA localization (`Localize.py` exists but is parked — needs ≥ 2 phase-coherent antennas on one radio; the ESP32-S3 has one receive chain).
- People counting (`collect_count.py` exists; pipeline is built as presence-with-N-classes; no accuracy on real hardware yet).
- Through-wall sensing.
- Vitals detection (breathing, heartbeat).
- On-device model updates (LoRA, EWC on the ESP32 or Pi) — the "adapt to this room" mechanism is recalibration, not weight updates.
- HomeKit / Matter / MQTT integrations.
- Compressed sensing (deferred; only useful if the subcarrier pattern is scattered/incoherent — verify on hardware first).

---

## 12. Troubleshooting

### `mesh_verify.py` shows nothing

1. Check `PC_IP` in `firmware/esp32_node/main/config.h`. It must match your Mac's current LAN IP (`ipconfig getifaddr en0`). If the IP changed, reflash with the correct value.
2. Check the Mac firewall: `System Settings → Network → Firewall → Options`. UDP 9876 must be allowed.
3. Check that the boards associated to the router. In the serial monitor after flashing, you should see a log line confirming the Wi-Fi connection. If not, `WIFI_SSID`/`WIFI_PASS` are wrong, or the router is on a different band than expected.
4. If one specific node is missing (others are fine), that board has stale NVS. Wipe and reflash: `idf.py erase-flash`, then `rm -rf build` and flash again.

### Boards won't associate to the router

The most common cause is stale NVS (saved Wi-Fi credentials from a previous network). Run `idf.py erase-flash` to wipe the entire chip, then reflash. See the wipe-and-reflash instructions in [`firmware/README.md`](firmware/README.md).

### Frame rate is very low (< 5 Hz per link)

Two known causes:
- ESP-NOW is sending at the legacy Wi-Fi rate instead of the HT rate. Check the `rate_config` in the firmware.
- FreeRTOS tick rate is 100 Hz instead of 1000 Hz. Check `sdkconfig.defaults` — `CONFIG_FREERTOS_HZ` should be 1000.

### Presence model accuracy is poor

In order of likelihood:
1. **Did not recalibrate after moving hardware.** Recalibrate; collect fresh data.
2. **Mixed capture profiles.** Check that you used the same `--root` for baseline, collection, and training.
3. **Room not empty during calibration.** Re-run `collect_baseline.py` with the room actually empty.
4. **Too few training sessions.** Collect more; 3–5 session pairs is the minimum.
5. **Wrong evaluation method.** Make sure you are using LOGO, not a random split. Random splits on CSI data give artificially high accuracy.

### `pip install -e .` fails with a C++ build error

- Make sure CMake and a C++ compiler are installed: `cmake --version`, `g++ --version` (or `clang++ --version` on Mac).
- On macOS, run `xcode-select --install` if the compiler is missing.
- Check that the Python venv is active before running `pip install -e .`.

### `idf.py build` fails with `cmake -E touch` error

This is a leftover stamp file from a partial `fullclean`. Fix: `rm -rf build` (not `idf.py fullclean`). If it persists, also delete `sdkconfig`, run `idf.py set-target esp32s3`, then build again.

### Pi `mesh_verify` entry at very low rate (< 10 Hz)

The illuminator is not running, or the Mac is not associated to modem B. An idle AP only sends ~10 beacons/s. Run `bash firmware/pi/illuminate.sh <modem_B_ip>` from the Mac while associated to modem B's 5 GHz network.

### Web dashboard shows no data

1. Make sure `python web/streamer.py` is running.
2. Check that the WebSocket port 8765 is not blocked. Both processes (streamer and browser) must be on the same machine, or the browser must be able to reach port 8765 on the Mac.

---

## 13. Do not do these things

These are mistakes that look reasonable but will waste time or break the system silently.

**Do not use a random train/test split on CSI data.** Consecutive frames in one session are nearly identical. A random split will leak session-level identity into the test set and give you 97–99% accuracy on paper. Use LOGO (leave-one-session-out). See §10.

**Do not run presence and weapon on the same `--root` with different subcarrier counts.** Calibration files are tied to a specific subcarrier layout. If you change the bandwidth or band, recalibrate.

**Do not mix 2.4 GHz and 5 GHz data in the same model.** They have different subcarrier counts (K). You cannot stack their feature vectors or tensor-merge their spectrograms. Fuse at the decision level (vote) if you want to combine them.

**Do not try to implement AoA (angle-of-arrival) localization with the ESP32 nodes.** Each ESP32-S3 has one receive chain. The "diversity switch" on the board selects between two antennas on that one chain — it is not a 2-antenna MIMO receiver. AoA requires two coherent receive chains on one radio. The code for this (`Localize.py`) exists but is parked until a suitable radio is added.

**Do not use LOS-blocking ("doorway") geometry for weapon detection.** Placing TX and RX on opposite sides of a door so the person walks between them creates a strong direct path. That direct path masks the weapon's signal. Use non-LOS geometry instead: put TX and RX on the same side of the room so the measured signal comes from reflections.

**Do not implement on-device model updates (LoRA, EWC) on the ESP32 or Pi.** The adaptation mechanism is per-session recalibration (`collect_baseline.py`) plus retraining on the Mac. That is cheap, reliable, and already built.

**Do not attempt weapon tier 2 or 3 before tier 1 works.** If the σ²[p] distributions for a flat metal plate are not separated from empty room on held-out data, no CNN complexity will fix it. The problem is in the sensor (geometry, antenna, gain lock), not the model. Fix the hardware first.

**Do not rename the identifiers listed in the "Skipped — crosses a boundary" section at the bottom of this document.** They cross a firmware ↔ host boundary. A rename that touches only one side will corrupt the wire format silently.

---

## 14. Development

### Running the tests

```bash
source .venv/bin/activate
pytest tests/ -q
```

All tests run on synthetic data. No hardware is needed. Expected: ~283 passed. If a test fails after a code change, fix the test and the code in the same commit — do not leave a failing test open.

### Rebuilding the C++ extension after a change in `src/`

```bash
pip install -e . --no-build-isolation
```

### Adding a new ESP32 node

1. Set `MESH_NODES` in `config.h` to the new total.
2. Flash the new board with the next available `NODE_ID`.
3. No Python code change needed — the aggregator reads `node_id` from the UDP header and handles any count dynamically.

### Changing the model backend (MLP → SVM → CNN)

Pass `--backend` to `wavetrace train`:

```bash
wavetrace train ./dataset --stage presence --backend mlp    # default
wavetrace train ./dataset --stage presence --backend svm
wavetrace train ./dataset --stage weapon   --backend cnn --feature-mode cnn
```

The saved `model.joblib` is backend-agnostic at load time — `Infer.py` detects the type automatically.

### Switching modes (presence ↔ weapon)

Pass `--head-mode` to `wavetrace run`:

```bash
wavetrace run --calibration ./calib --model ./model.joblib --head-mode presence
wavetrace run --calibration ./calib --model ./model.joblib --head-mode weapon --vote
```

The `--vote` flag enables `SegmentVoter`, which accumulates per-window predictions over a motion segment and emits one stable verdict per segment. Recommended for weapon mode.

### Capture profiles for new hardware

If you add hardware with a different bandwidth (e.g. HT80 on the Pi), create a new directory under `data/` and pass it as `--root`. The pipeline is profile-agnostic — `K` (subcarrier count) is read from the calibration file.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| CSI | Channel State Information — complex amplitude + phase per subcarrier, measured from the preamble of every incoming Wi-Fi packet. Encodes how the room shaped the signal. |
| Subcarrier | OFDM splits the Wi-Fi channel into narrow frequency bins. HT20 ≈ 52 subcarriers; HT40 ≈ 114; HT80 ≈ 256. Each bin is an independent amplitude + phase measurement. |
| Link | A directed (TX node, RX node) pair. N mesh nodes → N·(N−1) links per cycle. Each link trains and runs its own model. |
| Node | One hardware unit (ESP32 or Pi) sending UDP CSI. Identified by `node_id` in the datagram header. |
| Gain lock | Fixing the ESP32's AGC amplification to the value measured during calibration. Without this, the receiver adjusts its own "volume" between sessions and the amplitude baseline shifts. Implemented in C++ `GainLock`; CV (σ/μ) is the fallback if the firmware lock is unavailable. |
| HT20/40/80 | 802.11 channel-width modes: 20/40/80 MHz. Wider = more subcarriers = more frequency resolution, but uses more spectrum. |
| Calibration | One `collect_baseline.py` run: records a quiet empty room, stores the gain-lock scalar, NBVI mask, and quiet baseline. Must be redone when the room layout, antenna position, or radio band changes. |
| LOGO | Leave-One-Session-Out — the correct cross-validation for CSI data. Train on all sessions except one, test on the held-out session, repeat for every session. Prevents session-identity leakage. |
| LOSO | Leave-One-Subject-Out — same idea but held-out unit is one person's sessions. Tests generalization to new individuals. |
| σ²[p] | Per-packet inter-subcarrier variance — variance of CSI amplitude across subcarriers within one packet. Metal reflects evenly → lower σ²; human body scatters unevenly → higher σ². The primary weapon discriminator. |
| NBVI | Narrowband variance index — ranks subcarriers by how much their amplitude changes when a person enters. The top-K are kept; the rest are discarded to reduce noise and computation. Run offline during calibration. |
| ESP-NOW | Espressif's connectionless 802.11 protocol used for the TX burst. No association required; RX nodes capture CSI from the packet preamble without decoding the payload. |
| Non-LOS | Non-line-of-sight geometry: TX and RX are positioned so there is no direct unobstructed path between them. The signal measured is reflections off walls, floor, and objects. Recommended for weapon detection because a strong direct path masks a small weapon's signature. |
| Wire format v2/v3 | Binary UDP datagram layout shared between `firmware/pi/publisher.py` and `wavetrace/Source.py`. v2 = float32 I/Q; v3 = int16 I/Q with fixed scale (preserves absolute amplitude for weapon detection). |
| ESP-IDF | Espressif IoT Development Framework — the build toolchain and SDK used to compile and flash the ESP32 firmware. |
| Nexmon CSI | A firmware patch for Broadcom/Cypress Wi-Fi chips (including the Pi's CYW43455) that exposes raw CSI over a local UDP socket. |
| ISTA / L1 / CS | Compressed sensing: recovers missing subcarriers from the measured ones by solving a ℓ1-minimization in the delay domain. Optional deferred tool; only valid if the measured subcarrier pattern is incoherent (scattered). Verify the pattern on hardware before enabling. |

---

## 16. Where to look next

| Topic | File |
|---|---|
| Physics, hardware constraints, design decisions | `MINDMAP.md` |
| Current build state (done / blocked / next step) | `STATUS.md` |
| Full CLI run-book with all flags | `QUICKSTART.md` |
| DSP and ML algorithm details with pseudocode | `REFERENCE_DIGEST.md` |
| ESP32 mesh flash + wipe steps | `firmware/README.md` |
| Pi 5 GHz Nexmon node setup | `firmware/pi/README.md` |
| Data directory layout | `data/README.md` |
| Locked design decisions and why | `plan.md §7` |

---

## Skipped — crosses a boundary (do not rename without updating both sides)

The following identifiers cross a hardware or wire-format boundary. Renaming one side without the other corrupts the data stream silently, and no test will catch it.

- UDP datagram struct fields and header layout — `wavetrace/Source.py` ↔ `firmware/pi/publisher.py` ↔ `firmware/esp32_node/main/main.cpp`
- Wire format version bytes and JSON header keys: `v`, `node`, `tx`, `ntp_ms`
- `config.h` macro names: `NODE_ID`, `MESH_NODES`, `PC_IP`, `CHANNEL`, `DEST_IP`
- ESP-IDF sdkconfig keys in `sdkconfig.defaults`
- CLI flag names: `--root`, `--node`, `--carry`, `--subject`, `--head-mode`, `--stage`, `--vote`
- `data/` subdirectory names: `cal/`, `model/`, `model_weapon/`, `model_count/`, `2g4_ht20/`, `2g4_ht40/`, `5g_ht40/`, `5g_ht80/`
- pybind11 binding names in `src/Bindings.cpp` and `wavetrace/_wavetrace.pyi`
- UDP ports: 9876 (CSI mesh), 5566 (legacy 1-TX rig), 9877 (health monitor), 9878 (node discovery), 8765 (WebSocket)
