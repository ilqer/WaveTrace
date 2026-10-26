# WaveTrace — Project Documentation

---

## 1. What WaveTrace is

WaveTrace uses Wi-Fi Channel State Information (CSI) to sense what is happening in a room without a camera. Wi-Fi signals bounce off everything in the room. When a person or object moves, those reflections change in a measurable way across each frequency bin of the signal. By analyzing those per-frequency changes — one packet at a time — the system can tell whether a person is present and, eventually, whether they are carrying concealed metal.

The project has four independent operating modes. They all share one CSI front-end (preprocess → features) and differ only in the head:

- **Presence** — is a person in the room? Occupied or empty. This is the fully proven mode: 98.5% accuracy on real captures under leave-one-session-out validation.
- **People counting** — how many people are moving in the room? A four-class head (0, 1, 2, 3+) runs on each node, and the mesh links are combined into one count. On real hardware it reaches 53–61% accuracy where random guessing scores 25%, so it more than doubles the baseline. Collecting more sessions raises it further.
- **Occupancy heatmap** — where in the room is the person? A small CNN turns the CSI into a 16×16 occupancy grid. During collection a webcam labels the data on its own with a YOLO segmentation model, and those labels are time-aligned to each node's CSI windows, so no hand-labeling is needed. The camera is used only for training; the running system needs no camera. The pieces are built and covered by tests, and the live labeling works during capture. What remains is one full webcam-to-model run on real hardware (see §11); until then the live view uses a simpler energy-based estimate.
- **Weapon** — is a person carrying a concealed weapon (knife, gun, or similar metal object)? This is a research mode. It measures how flat the signal is across frequencies, since metal reflects evenly. At 2.4 GHz with the omnidirectional antennas, no reliable body-worn signal has appeared yet (see §11).

Weapon detection was the original headline goal, and presence is the easier step that proves the pipeline works. Presence is what works today. People counting also runs on real hardware. The occupancy heatmap is built and tested but has not had its one full end-to-end run on a real webcam yet.

---

## 2. Presence and weapon in depth

This section goes deep on presence and weapon, the two modes with the most physics behind them. The other two heads reuse the same machinery: **people counting** is the presence head widened to a 4-class output (0, 1, 2, 3+) — see projectProgress.md §9.10 — and the **occupancy heatmap** is a separate camera-supervised CNN covered in §11 here and in projectProgress.md §9.11.

**Presence** is a binary classification problem. The model gets a short window of CSI frames, extracts statistical features, and outputs 0 or 1. It trains in a few minutes on a laptop and achieves good accuracy on real data (LOGO 0.985 on real hardware). Start here.

**Weapon detection** is harder for a physical reason: a concealed metal object is much smaller than the human body it is attached to, so its signal rides on top of the person's much stronger reflection. The key discriminator is per-packet inter-subcarrier variance σ²[p]. Metal reflects all subcarriers evenly → lower variance. A human body scatters the signal unevenly → higher variance. This difference is detectable but subtle — sensor placement and geometry matter a lot.

Presence and weapon are **fully independent** (as are all four modes). Weapon mode does not use the presence model as a pre-filter. Every mode shares the signal-processing front-end (preprocess → features) and differs only in the classification head.

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
                        (JSONL / WebSocket to dashboard)
```

**Nodes (ESP32):** The mesh runs a time-division round-robin. In each turn, one node broadcasts an ESP-NOW burst; every other node captures CSI from it. The TX role rotates. N nodes produce N·(N−1) directed (tx, rx) links per cycle. Every board runs the same firmware — `NODE_ID` (set by `flash.sh`) is the only per-board difference.

**Pi node:** The Raspberry Pi captures 5 GHz CSI on its onboard CYW43455 chip via Nexmon and streams it to the Mac as node 5 on the same UDP port and in the same wire format as the ESP nodes. The host treats it like any other node.

**Mac host:** Decodes the binary UDP datagrams in C++ (fast path), preprocesses, extracts features, and runs a scikit-learn or PyTorch model. Training is also done on the Mac from recorded data.

**Training flow:** Record sessions with `collect_*.py`, which saves raw CSI to disk and trains a model. The `run_*.py` scripts load that model and run inference on live UDP data.

**Web dashboard:** An optional React UI (`web/ui/`) that shows live spectrograms, per-node health, and predictions over WebSockets. Useful for demos and debugging. Per-node health includes gain drift against that node's calibration (live mean |H| / calibrated quiet-room mean |H|, flagged when it moves more than 20%).

---

## Pipeline internals

Skip this if you just want to run the scripts. Read it if you want to understand why the code is structured the way it is.

### Frames

Every ESP32 burst is decoded into a `CsiFrame` by `wavetrace/adapters/source/wire_format.py`:
- `grid` — shape `(A, S)` complex64: amplitude + phase per antenna × subcarrier
- `timestamp` — float seconds
- `node_id` — which node TX'd the burst

`save_recording` writes these to disk as `grid.npy` (F, A, S), `t.npy` (F,), `node_id.npy` (F,), `meta.json`.

**Column is not tone.** A `grid` column is a position in the radio's buffer, not a signed OFDM tone.
`wavetrace/domain/tone_layout.py` holds one `ToneLayout` per capture layout (Δf = 312.5 kHz for all):

| Layout | Width | Fields (columns → signed tones, carrying) |
|---|---|---|
| `esp32s3_ht20` | 128 | `l_ltf` cols 0..63 FFT order (0..31, −32..−1), ±1..±26; `ht_ltf` cols 64..127 FFT order, ±1..±28 |
| `esp32s3_ht40` | 192 | `l_ltf` cols 0..63 centred (col c = primary-relative tone c−32), ±1..±26; `ht_ltf` cols 64..191 FFT order on the 128-point grid, ±2..±58 |
| `nexmon_vht80` | 256 | `vht_ltf` one field, FFT order assumed; carrying set not measured yet, so every non-DC tone counts |

The ESP32 carrying sets match the always-zero columns measured on `data/2g4_ht20` and `data/2g4_ht40`.
Anything working in the delay domain must use the signed tone: `select_field(csi, layout)` returns the
carrying tones of the layout's `default_field` (`ht_ltf` on ESP32, `vht_ltf` on Nexmon) sorted by tone, plus their indices, for `cir_from_csi(subcarrier_indices=...)`, and
`Localizer(tone_layout=...)` (also `localize_source(tone_layout=...)`) uses them for the range profile
and joint delay-angle MUSIC. With a layout, `Localizer` refuses a `subcarrier_spacing_hz` other than the
layout's and a frame whose width is not the layout's `num_columns`. Without a layout `Localizer` treats
columns as contiguous tones (synthetic grids only). Adjacent-column products (calibration, conjugate multiply) are unaffected: every FFT wrap
and field seam sits between null columns.

`save_recording(..., tone_layout=...)` writes `"tone_layout": "<name>"` into `meta.json`;
`load_tone_layout(recording_dir)` reads it back. A recording without the key (every recording made
before 2026-10-26, all ESP32-S3) gets its layout from its width (128 → HT20, 192 → HT40) with one warning.

**UWB CIR taps.** The UWB shield (NXP SR150, 2 RX chains, one LO) sends the CIR itself: per packet, per
RX, N complex taps (tap spacing ≈ 1 ns, unverified), one `window_start_index` and a per-RX
`first_path_index` counted in the chip's whole CIR. The window start moves between packets, so the
same path lands on different taps. All of these take and return plain numpy arrays:

- `recognition/cir.py` `cir_from_taps(taps, tap_spacing_s)`: the same statistics as `cir_from_csi`
  (dominant tap and its basin ratio, RMS delay spread, active taps) without the ISTA step;
  `subcarrier_spacing_hz` is 1/(N·tap_spacing_s), the FFT bin spacing of the taps.
- `domain/dsp/cir_alignment.py` `reanchor_taps(taps, first_path_columns, anchor_column, out=None)`:
  shifts each RX row so column `first_path_index − window_start_index` lands on `anchor_column`, edges
  zero-filled. Use it before any per-tap feature. O(R·N); residual jitter ≤ ½ tap.
- `cross_spectrum(taps, out=None)`: X_r[k] = H_r[k]·conj(H_0[k]) with H_r = fft(taps_r), on the taps
  as received. A common window shift (whole or fractional) and the shared-LO phase cancel. Exact only
  while no path leaves the window; per-path delay is lost. O(R·N log N).

**UWB CIR wire format (v4).** The Portenta C33 repacks the SR150's CIR notification into one UDP datagram
(default port 9879) per batch. `adapters/source/uwb_wire_format.py` `parse_cir_batch` is the reference
parser; it raises `ValueError` naming the field on any length or value violation. All integers are
little-endian. A datagram is one header followed by `num_records` records, with no padding and no
trailing bytes.

| Field | Type | Unit / meaning |
|---|---|---|
| **Header (13 bytes)** | | |
| `magic` | u8 | `0x57` |
| `version` | u8 | `4` (v2/v3 are the CSI formats) |
| `node` | u8 | node id |
| `ntp_ms` | u64 | NTP-synced wall clock, ms |
| `num_records` | u16 | records that follow |
| **Per record, in this order** | | |
| `peer_short_addr` | u16 | UWB short address of the peer |
| `ts_us` | u32 | device timestamp, µs |
| `num_rx` | u8 | R, RX chains; must be > 0 |
| `num_taps` | u16 | N, taps per RX; must be > 0 |
| `tap_spacing_fs` | u32 | tap spacing, femtoseconds (1 ns = 1 000 000) |
| `window_start_index` | i16 | chip CIR index of tap column 0 |
| `first_path_index` | i16 x R | per RX, in the chip's whole CIR (not the window) |
| `distance_cm` | u16 | cm |
| `aoa_azimuth_q9_7` | i16 | Q9.7 degrees (value / 128) |
| `aoa_elevation_q9_7` | i16 | Q9.7 degrees |
| `pdoa_q9_7` | i16 | Q9.7 degrees |
| `pdoa_index` | u16 | chip PDoA index |
| `snr_first_path_db` | u8 x R | dB per RX |
| `taps` | i16 x R x N x 2 | RX-major, then tap, then **real first, imag second** (ESP32 CSI is imag first) |

A record is `15 + 2R + 10 + R + 4RN` bytes. `parse_cir_batch` returns `domain/cir_record.py` `CirRecord`s;
`CirRecord.first_path_columns` (`first_path_index − window_start_index`) feeds `reanchor_taps`.
`UdpCirSource` (`adapters/source/uwb_udp.py`, behind the `CirSource` port) yields them with the same
idle-timeout behaviour as `UdpSource`.

### Calibration

Calibrate on an empty-room recording before collecting labeled data. Calibration does two things:

1. **GainLock** — fits a per-node AGC normalizer on the empty-room amplitudes. This corrects for hardware gain differences between boards.
2. **NBVI** — picks the K=12 subcarriers with the highest variance-to-noise ratio. These carry the most presence signal.

Output goes to `output/calib/`:
- `baseline_mag.npy` — mean amplitude per subcarrier in the empty baseline
- `baseline_diff.npy` — mean conjugate-phase difference across antennas
- `meta.json` — the K NBVI subcarrier indices + `image_subcarriers` (all noise-gate-passing indices, used for spectrograms; different set from NBVI K) + `subcarrier_width` (the raw capture width the baseline was made from; absent in older calibrations, which then skip the width checks against the stream and the head)

### Feature window loop

`frontend.iter_windows` slides a window of `window` frames with stride `hop` across the recording. Per window it extracts three things in this order:

1. **IC features** — 27 numbers from **raw (pre-lock)** magnitudes. Structure: 3 series (µ, σ², CV) × 9 statistics = 27. This must be computed before gain lock. Gain lock removes the per-packet amplitude variance that distinguishes metal from body tissue — apply it first and σ²[p] disappears.
2. **Presence features** — gain lock applied, then K=12 NBVI subcarriers selected, then 9 statistics per subcarrier → 9·K = 108 numbers.
3. **Spectrogram image** — shape `(K_img, window)` using all noise-gate-passing subcarriers. K_img is not the same as the NBVI K.

### Dataset

`build_dataset` runs the window loop with a labeler attached and collects:

| field | shape | content |
|---|---|---|
| `X_features` | `(n, 9·K)` | presence features (post-lock, NBVI) |
| `X_image` | `(n, K_img, window)` | spectrograms (all valid subcarriers) |
| `X_intercarrier` | `(n, 27)` | weapon IC features (pre-lock) |
| `y` | `(n,)` | class labels |
| `session_ids`, `subject_ids` | `(n,)` | group IDs for LOGO cross-validation |

`save_dataset` writes: `features.npy`, `images.npy`, `features_ic.npy`, `manifest.jsonl`, `meta.json`.

### Training

Presence and weapon each train a separate head. They don't share the same features.

**Presence**: trains on `X_features`. Default backend is an MLP (`StandardScaler` + `MLPClassifier`, `ModelConfig.backend = "mlp"`). Also available: `svm` (`CalibratedClassifierCV(SVC())`), `variance`.

**Weapon**: 4 backends. All except `cnn` train on `X_intercarrier`:
- `variance` — threshold on column 9 of the IC block (mean σ² of the σ² series). No model file, just a threshold.
- `mlp`, `svm` — full 27-feature IC head.
- `cnn` — trains on `X_image` instead.

### Model storage

Every training run, and every model uploaded through `/api/model/upload`, writes one run dir,
`<model_dir>/<run_id>/`, holding the head (`model.joblib`, or `heatmap.joblib` for the heatmap head),
`metrics.json` with LOGO accuracy (trained runs only) and `run_manifest.json`. The run id is the UTC creation time
plus a random suffix (`20261026T153012Z-a1b2c3`), so run dirs sort by time. The dir is written in full under a
hidden temp name (`.<run_id>.partial`) and renamed into place; only then does the pointer switch to it. On macOS
and Linux the pointer is the `<model_dir>/current` symlink, swapped with `os.replace`; on Windows, and wherever a
symlink cannot be made, it is a `CURRENT` text file holding the run id, also replaced atomically (Windows cannot
rename onto a directory symlink, so a symlink copied there is dropped just before `CURRENT` replaces it). The
new pointer is complete before the old one changes, so a crash or Ctrl+C mid-train leaves the previous run
current, never a model without its metrics; a pointer move that fails raises and leaves no temp file behind.
Only two windows leave no pointer at all (loaders then serve the flat files, if any): between moving a copied
`current` dir aside and the final rename, and, on Windows, between dropping a copied-in `current` symlink and
the replace; both happen only on the first retrain of such a copy. Old runs are kept; nothing prunes them.

Copying a model dir with `shutil.copytree`, `scp` or `zip` follows the symlink, so the copy holds a real
`current` dir. Loaders serve the run dir it duplicates (the one run of its creation second with a byte-equal
`run_manifest.json`), so a retrain cannot split one load across two runs; when that run was not copied along,
the `current` dir itself is served. The next training run in the copy renames it to
`current.copied-<new run_id>` (kept, never deleted) before pointing `current` at the new run.

Loaders take the `current` symlink (or a copied `current` dir), then the `CURRENT` file, then the flat
`<model_dir>/model.joblib` that every model trained before runs has. A pointer naming a run that no longer exists
logs one warning per process naming the pointer and the missing run, then falls back to the flat file. A path
written as `<model_dir>/model.joblib` (`--model`, dashboard paths) loads the current run, and each load reads the
head and its `metrics.json` from that one run. A current run hides any flat file beside it, so one model dir
holds one head kind (the heatmap's model root holds the per-node heads only in its `node<id>/` subdirs).

`run_manifest.json` records where the run came from. Its keys are a file format and never change:
`origin` (`"trained"` or `"uploaded"`), `git_commit` and `has_uncommitted_changes` (tracked files only; null
outside a checkout), `config_hash` (sha256 of the `ModelConfig`'s canonical JSON), `dataset_hash` (sha256 over
every dataset file the run read, sorted by relative path, each path hashed with its size and bytes), `seed`,
`library_versions` (`python`, `numpy`, `scikit_learn`, `torch`, `wavetrace`), `contract` (the
`PipelineContract`), `created_timestamp_s` (UTC Unix seconds) and `schema_version`. Dataset files that cannot
be hashed (unreadable, or pooled from two Windows drives) leave `dataset_hash` null with a warning; the fitted
head is still saved. An uploaded run knows only its origin and creation time; every other key is null. An
upload's `dest` must name a file inside `output/`; a `dest` inside a run dir or through `current` uploads a new
run of that model dir. One training call (a `wavetrace train`, a collect
script's training, a dashboard Train) reads the git state and library versions once for all the heads it
trains, and each head hashes only the dataset files it trained on.

Presence, weapon and heatmap heads check their saved `schema_version` on load: a head without one loads as
legacy version 1 with a warning (once per file per process, however the path is spelled or linked), and any
version other than the code's
`SCHEMA_VERSION` fails with `SchemaVersionMismatchError`, which names the file, both versions and how to
retrain it into its model dir (the `wavetrace train` command for presence and weapon heads).

### Serving

`open_session(mode, model_path)` loads the trained head and returns an `InferenceSession`. `plan_input(mode, head)` tells the front-end what to extract:

| backend | input | gain lock? |
|---|---|---|
| presence / mlp / svm | `X_features` (NBVI stats) | yes |
| weapon / mlp / svm | `X_intercarrier` (IC 27) | no |
| weapon / cnn | `X_image` reshaped | no |
| weapon / variance | IC column 9 only | no |

`session.predict(window)` returns `(class_id, confidence)`. Multi-node serving in `run_live_mesh` votes across all nodes weighted by each node's LOGO accuracy.

---

## 4. Prerequisites

### Hardware

| Item | Qty | Notes |
|---|---|---|
| ESP32-S3-DevKitC-1 | 2–6 | All run the same firmware. `NODE_ID` is injected per-board by `flash.sh`. Status LED is GPIO 38 (v1.1 board). |
| 8dBi RP-SMA omnidirectional antenna | one per board | 160 mm whip, 2.4/5.8 GHz dual-band, vertical polarization, 50 Ω, RP-SMA male. These are omnidirectional — they cover all horizontal directions. |
| Raspberry Pi 5 | 1 (optional) | For the 5 GHz arm. Pi 5 Nexmon is a community patch — not in the official Nexmon repo. See [`firmware/pi/README.md`](firmware/pi/README.md). |
| Dedicated Wi-Fi router | 1 | Locked to channel 6, 2.4 GHz, 40 MHz (HT40). Do not share with regular traffic — bursty traffic corrupts the calibration baseline. |
| USB cables | one per ESP32 | For flashing only. After flashing, boards run from any 5 V supply. |

**Antenna note:** The 8dBi omnidirectional antennas work well for presence detection. For weapon detection, the research literature uses directional antennas (horn/dish, ≥9 dBi) to focus the signal on the subject. The round-robin TX rotation across 6 boards gives 30 directed links from 6 different TX positions, which provides partial spatial diversity. Whether that is enough for weapon detection is an open question to answer from the first real dataset.

### Software

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Main host language |
| CMake + C++ compiler | CMake 3.16+ | Compiles `native/` into the Python package |
| ESP-IDF | v5.3 | Flashes the ESP32 firmware |
| Node.js | 18+ | Web dashboard only (optional) |
| numpy, scikit-learn | via `pip install -e .` | Signal processing and ML |
| torch | via `pip install 'wavetrace[cnn]'` | CNN backend only (optional) |
| ultralytics | via `pip install ultralytics` | YOLO-based camera ground-truth labeling only (optional, used by `scripts/collect_camera.py`) |

### `data/` folder

`data/` is git-ignored, so it does not exist right after cloning. Don't create it by hand — the `collect_*.py` scripts create it, and the profile subfolder under it, automatically the first time you pass `--root data/<profile>`. It only needs to exist before you run something that *reads* from it (e.g. `run_live_mesh.py` after training), which will already be true if you followed the calibrate → collect → train order in section 6.

### Optional: YOLO model for camera labeling

`ultralytics` (YOLOv8) is only needed for camera-supervised ground-truth labeling (`scripts/collect_camera.py`), not for the core CSI pipeline. It is not a project dependency — `pip install -e .` does not install it, and nothing else imports it. Install it separately with `pip install ultralytics` if you want that feature; running `collect_camera.py` without it raises an `ImportError` telling you to install it. There is no weights file to download manually either — passing a model name like `yolov8n-seg.pt` triggers an automatic one-time download the first time it's used.

### Network setup

- Lock the router to **channel 6, 2.4 GHz, 40 MHz (HT40)**. All ESP32 boards join it as STAs, which automatically locks every board to channel 6 so they hear each other's ESP-NOW traffic. Do not change the channel after flashing without a full wipe-and-reflash.
- Give the Mac a **static DHCP lease** on the router. Set that IP as `PC_IP` in `Config.h`. If the IP changes, boards send CSI into the void.
- Allow incoming UDP on these ports through the Mac firewall (`System Settings → Network → Firewall → Options`):
  - **9876** — CSI datagrams from all nodes
  - **9877** — per-node health heartbeats
  - **9878** — node discovery
- Run `python scripts/ntp_server.py` on the Mac before any collect or live script. The ESP32 firmware uses the Mac as its SNTP clock source. Without it, nodes fall back to their own monotonic clock — single-link presence still works, but cross-node timestamp alignment is unreliable.
- Keep the sensing network on a **dedicated router**. Traffic from other devices causes AGC swings that corrupt the calibration baseline.

---

## 5. Where things live

```
firmware/
├── esp32_node/     unified mesh firmware (one binary; NODE_ID set per-board by flash.sh)
├── esp32_rx/ esp32_tx/  legacy 1-TX/6-RX rig
└── pi/             Nexmon CSI capture + UDP stream scripts for the Raspberry Pi

native/             C++ signal processing (compiled into the wavetrace package via pybind11)
├── core/           CsiFrame type (one frame = complex amplitude per subcarrier + timestamp + node_id)
├── hardware/       UDP datagram parser + multi-node frame assembler
├── signal/         conjugate-multiply, Hampel filter, phase unwrap, gain lock, NBVI, features, spectrogram
└── util/           radix-2 FFT, ring buffers

wavetrace/          the Python library
├── entrypoints/    composition roots, imported by nothing: cli/main.py (the `wavetrace` CLI: capture / calibrate / collect-data / train / localize / run) and api/ (the dashboard backend, below)
├── synthetic.py    synthetic CSI + paired labels for `--synthetic` mode
├── domain/         pure pipeline descriptions, no I/O: PipelineContract, ModelConfig, LinkKey, Calibration (gain-lock scale, NBVI mask, quiet baseline), presence/weapon heads, `recognition/input_plan.py` (`plan_input`), `groundtruth/` (`alignment.py`: timestamp alignment; `labeler.py`: `Labeler`, `ScriptedLabeler`, label functions; `dataset.py`: `Dataset`, `build_dataset`, `build_stacked_dataset`), SegmentVoter, AlertGuard/DriftMonitor (guard.py), `localize.py` (AoA/delay MUSIC `Localizer`), `tone_layout.py` (CSI column → signed tone per capture layout), `tracking.py` (Kalman `Tracker`), dsp/ (resampling and frontend.py, the shared pipeline loop: source → preprocess → features → emit windows)
├── application/    use-cases (calibrate, collect, localize, serve, plus the live mesh loops serve_mesh, collect_mesh and collect_camera behind per-mode strategies in recognition/) on loaded values, no concrete I/O (console output through print, or collect_camera's injected log) + ports (the Protocols adapters implement) + behaviours/ (the serving guard and vote, composed by `pipeline.run_inference` from `--guard` / `--vote`)
├── pipeline.py     directory-based entry points (calibrate_source, collect_source, run_inference, localize_source, serve_mesh, collect_mesh, collect_camera): load → use-case → save
├── adapters/       behind the ports: source/ (CSI frame sources: live UDP, serial, Pi, recorded file, synthetic;
│                   wire-format parsers), store/ (save/load the calibration dir; `dataset.py`: `save_dataset`, `load_dataset`; `localization.py`: `save_localization`; `scripted_spans.py`: `load_scripted_spans`; `model_runs.py`: the atomic `<model_dir>/<run_id>/` + `current` model layout; `run_manifest.py`: `build_run_manifest`, `build_upload_manifest`, `RunProvenance`) and recognition/ (MLP, SVM, CNN, variance backends; head construction; `session.py`: `InferenceSession`, `open_session`)
├── source.py       compatibility path for the old `wavetrace.source` imports (the golden test uses it)
├── recognition/    heads (presence, people-count, occupancy heatmap, weapon) + training, multi-node fusion, evaluation, CIR; `infer.py` holds only `measure_latency`
├── groundtruth/    concrete labelers, one per file: `vision_labeler.py`, `segmentation_labeler.py`, `yolo_labeler.py`
│                   (Ultralytics), `replay_labeler.py`, `location_chip_labeler.py`, `thermal_labeler.py`; `webcam.py`
├── output/         result publisher (JSONL default; WebSocket seam for the dashboard)
└── diagnostics/    per-node health telemetry; `weapon_separation.py` (σ²[p] go/no-go check)

scripts/              terminal scripts you run at the command line (always from the project root)
├── collect_baseline.py   record quiet empty room → saves calibration under data/<profile>/cal/
├── collect_presence.py   record presence sessions → trains per-node presence models
├── collect_weapon.py     record weapon sessions (no-weapon/weapon, per node or --per-link) → trains weapon model
├── collect_count.py      record people-count sessions → trains count model
├── collect_camera.py     simultaneous CSI + webcam capture with live YOLO-seg labeling → one dataset per link (+ heatmap)
├── run_live_mesh.py      live presence detection (all nodes voted)
├── run_weapon.py         live weapon detection
├── run_count.py          live people count
├── mesh_verify.py        listen on UDP 9876 and print which (tx, rx) links are arriving and at what rate
├── health_monitor.py     per-node uptime, free heap, frame rate — refreshes every second
└── ntp_server.py         SNTP server the ESP32 nodes use as their shared clock source

experiments/          offline analysis scripts — run after you have data, not part of the live pipeline
├── weapon_litmus.py      σ²[p] PDF go/no-go check: do the clear and weapon distributions actually separate?
└── weapon_experiments.py model bake-off (per-node vs per-link vs combined CNN) with LOGO evaluation

wavetrace/entrypoints/api/   the dashboard's FastAPI backend
├── app.py            FastAPI app: lifespan, CORS, the routers, the static UI (port 8000)
├── dashboard_state.py  DashboardState (channels, DeviceHub, runner) and BroadcastChannel
├── routers/          HTTP and WebSocket routes: pipeline, model, paths, device, camera
├── streamer.py       WaveTraceRunner — drives the pipeline in a worker thread
└── device_ctl.py     DeviceHub — serial monitor, firmware flash, Pi SSH, script runner

web/
└── ui/               React + Vite + TypeScript dashboard

tests/                pytest suite (631 tests; all offline on synthetic data — no hardware needed)
├── unit/             one module per test file, mirroring the source tree (unit/domain/dsp/ tests wavetrace/domain/dsp/)
├── integration/      several layers end to end: calibrate → collect → train → serve
├── contract/         adapters match their ports; no dumping-ground names (util(s), helper(s), common, misc, manager(s))
└── golden/           characterization tests with committed fixtures and expected output
data/                 git-ignored; created by the collect scripts
```

---

### Where to run the scripts from

Everything in `scripts/` must be run from the project root — the folder that contains `wavetrace/`, `web/`, `scripts/`, etc. The scripts import from the `wavetrace` package, and Python only finds it if you're in the right place.

```bash
# this is correct
cd WaveTrace/
python scripts/collect_baseline.py --root data/2g4_ht40

# this breaks
cd WaveTrace/scripts/
python collect_baseline.py    # can't find wavetrace
```

The same goes for the analysis scripts in `experiments/`. Run them from the project root too.

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

Run `source .venv/bin/activate` in every new terminal.

---

### B. Flash the ESP32 boards

Full steps are in [`firmware/README.md`](firmware/README.md). Summary:

1. Install ESP-IDF v5.3 (once, ~30–60 min).
2. Open `firmware/esp32_node/main/Config.h`. Set `ROUTER_SSID`, `ROUTER_PASS`, and `PC_IP` (your Mac's LAN IP — `ipconfig getifaddr en0`). `NODE_ID` is set per-board by `flash.sh` — no manual edit needed.
3. Flash board 1, then board 2. Plug in one at a time:

```bash
. ~/esp/esp-idf/export.sh
./flash.sh node 1 /dev/cu.usbmodem1101   # first board
./flash.sh node 2 /dev/cu.usbmodem1101   # second board
```

After flashing, boards only need a 5 V power supply. USB is only needed for flashing.

---

### C. Start the NTP server (keep this running throughout)

```bash
python scripts/ntp_server.py
```

Keep this running in a dedicated terminal whenever the boards are on.

---

### D. Verify hardware

```bash
python scripts/mesh_verify.py
```

With 2 boards, expect links `1->2` and `2->1` at a non-zero frame rate. With 6 boards: 30 links (6×5). Typical rate per link: 20–100 Hz depending on channel load.

If nothing appears: wrong `PC_IP` in `Config.h`, Mac firewall blocking UDP 9876, or boards haven't associated yet (wait ~10 s after power-on).

```bash
python scripts/health_monitor.py    # per-node uptime and free heap; Ctrl-C to stop
```

---

### E. Calibrate (empty room)

```bash
python scripts/collect_baseline.py --root data/2g4_ht40
```

The room must be completely empty and still — no people, no movement, no fans. This records what the empty room looks like and locks the AGC gain so later captures are comparable. Takes about 30 seconds.

Output: `data/2g4_ht40/cal/node{id}/` for every detected node.

Use the same `--root` for every step in a session. If you move hardware, change the band, or change the room layout, recalibrate.

---

### F. Collect presence data and train

```bash
python scripts/collect_presence.py --root data/2g4_ht40
```

The script walks you through it: records the empty room, then tells you to walk around. When finished, it trains a per-node model and saves it to `data/2g4_ht40/model/`.

---

### G. Run live presence detection

```bash
python scripts/run_live_mesh.py --root data/2g4_ht40
```

Expected output: `PRESENT` or `absent` printed every ~1.5 s, with a per-link breakdown and confidence bar. Votes from all active links are combined into one verdict weighted by each node's LOGO accuracy.

A link that delivers too slowly is left out of the fuse. `--rate-gate target` (default) drops a link below `--min-rate-fraction` (default 0.8) of its head's target rate; `--rate-gate peers` drops it below that fraction of the median delivered rate of the live links, so links that stay in step with each other survive even when all of them run slower than the target (nothing is dropped while fewer than 2 links are measured). `run_weapon.py` and `run_count.py` take the same two flags; the dashboard keeps the defaults. The single-node path (`wavetrace run --min-rate-fraction`) skips a window whose frames were delivered below that fraction of the head's target rate. Each drop is counted under `SampleRateMustBeWithinTolerance` or `SampleRateMustKeepPaceWithPeers` and logged at most once per 10 s.

---

### H. Weapon mode (only after presence works end to end)

```bash
python scripts/collect_weapon.py --root data/2g4_ht40 --subject p0 --carry chest
python scripts/run_weapon.py     --root data/2g4_ht40
```

`--subject p0` is a label for the person (used for leave-one-subject-out evaluation). `--carry chest` is the carry position. Collect multiple sessions — vary subjects and carry positions — before training. The script retrains on the full cumulative pool after each run.

Do not skip to weapon mode if presence is not working. The pipeline is identical. If presence fails, the problem is in calibration, gain lock, or the UDP stream — fix that first.

---

### I. Web dashboard (optional)

```bash
python -m wavetrace.entrypoints.api.app   # FastAPI backend on port 8000 — keep this running
```

In a separate terminal:

```bash
cd web/ui
npm install             # once
npm run dev             # Vite dev server at http://localhost:5173
```


Dashboard behaviour to know:
- The Alert tile fires only in weapon mode. Presence (and count) runs never raise a weapon alert.
- The run panel's `frame_average`, `use_baseline` and `gainLock` knobs are ignored for inference: the model's saved contract decides frame averaging, the baseline and the gain lock.
- Camera collection is `scripts/collect_camera.py` without training (see §11); a port that cannot be bound is logged as that error and ends the run.
- Calib and Collect always end with `pipeline_done` (the Running state clears), also after Stop or an error; an error is logged as `ERROR Calibration failed: ...` or `ERROR Collection failed: ...` with its traceback.
- Train trains what the matching collect script trains: Dataset Path is a capture-profile root (e.g. `data/2g4_ht40`), and the stage picks the script's pool under it: presence `ds_*`, count `count_ds_*` (classes up to the highest `c<k>` on disk), weapon the cumulative `weapon_ds` pool (Per-link: one head per tx->rx link), and with the camera toggle on, presence trains the camera pool `cam_ds/presence` (count has no camera pool: it logs an error and trains nothing). Heads go where the scripts put them (`<root>/model`, `model_count`, `model_weapon[/node<id>/link<tag>]`, `model_camera`), one per node, so mesh serving finds them; Output Model Dir (`train_out`) is ignored. The chosen backend is used (presence and count accept only mlp/svm and fall back to mlp). The `heatmap` backend trains `heatmap.joblib` on one camera session dir (`<root>/cam_ds/heatmap/<session>`) next to the camera heads; a dataset without masks is skipped. The panel gets one `train_init` for the whole pool, epochs from cnn and heatmap fits (Stop ends them at the next epoch), one `done` after every head, and the run always ends with `pipeline_done`, also after an error or when nothing trained (see `web/README.md`). Stop during an mlp/svm run ends it before the next head, so the model root is left mixed: the heads the log names as trained are new and every other node or link keeps its previous head.

---

## 7. RF configuration — which profile to use

Data from different radio configurations cannot be mixed. A model trained on HT20 data is invalid on HT40, and vice versa. Every `collect_*.py` and `run_*.py` script takes `--root` to pick a profile. All data (calibration, recordings, datasets, models) for that profile lives under that directory.

| Profile (`--root`) | Band | Bandwidth | Subcarriers | Hardware | Status |
|---|---|---|---|---|---|
| `data/2g4_ht20` | 2.4 GHz | 20 MHz | ~52 | ESP32-S3 | legacy only |
| `data/2g4_ht40` | 2.4 GHz | 40 MHz | ~114 | ESP32-S3 | **current active** |
| `data/5g_ht40` | 5 GHz | 40 MHz | ~114 | Pi / Nexmon | not yet used |
| `data/5g_ht80` | 5 GHz | 80 MHz | ~256 | Pi / Nexmon | target for weapon |

**Current setup:** The firmware has `WT_BW_HT40 1` enabled and the router is set to 40 MHz, so all ESP32 captures go into `data/2g4_ht40`. Use this profile for all collect and run commands until the Pi node is ready.

**For weapon detection:** 5 GHz HT80 (`data/5g_ht80`) is better suited because the shorter wavelength (~60 mm vs 125 mm at 2.4 GHz) is closer to the size of a knife or gun. The Pi is configured for channel 36 / HT80 / 256 subcarriers and emits wire format v3 (int16 I/Q, fixed scale). Pi 5 Nexmon support is a community patch — not yet validated on this hardware. See [`firmware/pi/README.md`](firmware/pi/README.md).

**Do not mix profiles.** If you switch band or bandwidth, recalibrate from scratch.

---

## 8. Data collection rules

Bad data collection produces a model that looks good on paper and fails in real use. Follow these rules.

### Room and hardware setup

- **Non-LOS geometry.** Do not place TX and RX facing each other with the subject walking between them. A strong direct path masks a small weapon's signal. Put TX and RX on the same side of the room so the measured signal comes from reflections off walls and the subject.
- **Fixed hardware.** Do not move any node, antenna, or the router between calibration and data collection. Even a few centimeters changes the multipath signature. If you move anything, recalibrate.
- **Channel lock.** The router is locked to channel 6, 40 MHz. Confirm no nearby AP is on channels 4–8 (they overlap in 2.4 GHz). Heavy interference degrades signal quality past what the Hampel filter can fix.

### Calibration rules

- Run `collect_baseline.py` at the start of every new session (new day, moved hardware, changed room, changed band).
- The room must be completely empty during calibration. No people, no movement visible through windows, no fans near antennas.
- The capture takes about 30 seconds. Do not touch or walk near the equipment.

### Presence collection

- Collect at least 3–5 separate empty/occupied session pairs. More is better.
- "Occupied" sessions: walk naturally, change direction, sit down, stand up. Cover the whole area the system will monitor.
- Label the person consistently with `--subject` across sessions to enable leave-one-subject-out evaluation.

### Weapon collection

Follow the tier order. Do not attempt tier 2 or 3 until tier 1 works.

- **Tier 1 (static, open weapon):** Subject stands still for ~30 seconds with the object held openly at chest level. Reproduces the lab setup from the literature. If this fails, fix hardware (antenna, geometry) before proceeding.
- **Tier 2 (moving, real weapon):** Subject walks through the sensing area carrying the weapon concealed in clothing or a bag.
- **Tier 3 (truly concealed, scripted):** The weapon is in place before the session starts. Labels come from the operator's script (`--label-spans "start:end"`), not from a camera.

Collect multiple subjects and multiple carry positions (`--carry chest`, `--carry waist`, etc.) for each tier.

---

## 9. Weapon detection — the three tiers

Weapon detection is feasibility-gated. Each tier is a checkpoint. If a tier fails, fix the hardware before moving to the next — adding a more complex model on top of a broken sensor geometry will not help.

### Tier 1 — flat metal plate, non-LOS, static

Reproduce the setup from Yousaf et al. (2025): directional TX, non-LOS geometry, subject stands still with a flat metal plate at chest level. Run the σ²[p] variance threshold baseline:

```bash
python scripts/collect_weapon.py --root data/2g4_ht40 --subject p0 --carry chest
wavetrace train data/2g4_ht40/weapon_ds --stage weapon --backend variance --out data/2g4_ht40/model_weapon
```

Target: held-out sessions should show clearly separated σ²[p] distributions for metal vs no-metal. If they overlap completely, the problem is hardware (antenna, geometry, gain lock) — not the model.

### Tier 2 — real weapon, static, non-LOS

Replace the metal plate with a real knife or gun. Use the CNN backend:

```bash
wavetrace train data/2g4_ht40/weapon_ds --stage weapon --backend cnn --feature-mode cnn --out data/2g4_ht40/model_weapon
```

Target: FP ≤ 10% and TPR ≥ 90% on held-out sessions.

### Tier 3 — moving subject, weapon concealed

The person walks while carrying a concealed weapon. Requires at least 2–3 RX nodes at different angles and multiple training sessions:

```bash
python scripts/run_weapon.py --root data/2g4_ht40
```

Voting is handled automatically by `run_weapon.py` — each link contributes weighted by its LOGO accuracy.

---

## 10. How to evaluate results

### Use leave-one-session-out (LOGO)

A random train/test split on CSI data is meaningless. Consecutive frames in the same session are nearly identical — a model can learn to recognize the session rather than the class. This produces fake 97–99% accuracy that collapses on new data.

The correct evaluation is **leave-one-session-out (LOGO)**: train on all sessions except one, test on the held-out session, repeat for every session, report the average. `evaluate.py` does this automatically.

For weapon mode, also do **leave-one-subject-out (LOSO)**: hold out all sessions from one person, train on everyone else.

### What numbers to report

| Metric | How to compute | Minimum bar |
|---|---|---|
| Presence accuracy | LOGO cross-validation, all sessions | Clearly above majority-class baseline |
| Weapon TPR | `tier_verdict` in `evaluate.py` | ≥ 90% |
| Weapon FP rate | `tier_verdict` | ≤ 10% |
| Per-tier verdict | Run separately for each tier | Beat chance on each tier before moving to the next |

### What "works" means at each stage

- **Presence**: LOGO accuracy is clearly above the majority-class baseline (e.g. 60% empty → baseline is 60%; your model should be ≥ 85%).
- **Weapon tier 1**: σ²[p] distributions for metal vs no-metal are visually separated on held-out data.
- **Weapon tier 2**: `tier_verdict` returns True (FP ≤ 10% ∧ TPR ≥ 90%) on held-out data.
- **Weapon tier 3**: Same verdict gate, evaluated on moving-subject sessions from held-out subjects.

Do not report accuracy from a random within-session split.

---

## 11. Project status

### What is built and working (on synthetic/recorded data)

- Full C++ signal processing pipeline: frame parsing, conjugate-multiply, Hampel filter, phase unwrap, EMA detrend, gain lock, NBVI subcarrier selection, FFT, 9-feature extractor, inter-subcarrier σ²[p], spectrogram builder.
- Full Python presence pipeline: calibration, dataset builder, MLP/SVM training, LOGO evaluation, live inference, per-node voting, multi-node fusion, result publishing.
- People-count pipeline: a standalone path (`collect_count.py` + `run_count.py`) with a four-class head per node and live voting across all links. On real hardware it reaches 53–61% accuracy against a 25% random baseline (see projectProgress.md §9.10). It runs today; more sessions raise the score.
- Occupancy-heatmap pipeline: a camera-supervised CNN (`HeatmapHead`, trained with BCE plus soft-Dice), a `SegmentationLabeler` that turns YOLO masks into a 16×16 occupancy grid, temporal smoothing in `occupancy.py` (a per-cell Bayesian grid plus a 2D Kalman tracker that follows the hottest cell without jumping), `collect_camera.py --train`, live serving in `wavetrace/entrypoints/api/streamer.py`, and the `OccupancyHeatmap` and `OccupancyGrid3D` panels in the dashboard. All parts are built and covered by unit tests across the vision-labeler, segmentation, webcam, and ground-truth (alignment and dataset) suites. The one thing left is a full webcam-to-model run on real hardware; until a model is trained the live path uses an energy-based estimate (see projectProgress.md §9.11).
- Automatic camera labeling: `collect_camera.py` captures every link's CSI and the MacBook webcam at the same time, runs a YOLO segmentation model live on each frame, and time-aligns the labels to each link's CSI windows. This produces labeled per-link datasets and a heatmap dataset in one pass, with no hand-labeling. The camera is used only for collection; the deployed detector needs no camera. The script and the dashboard's camera collection run the same use-case (`CollectCameraUseCase`, through `pipeline.collect_camera`) and print the same messages; the dashboard labels at 5 Hz, uses subject `cam`, writes under its `train_data` path and never trains. Each run is the next free session, `<subject>_cam_s<n>`, so runs accumulate. Every (tx->rx) link is resampled and written on its own, like every other collect path: presence datasets go to `<root>/cam_ds/presence/node<rx>/link_<tag>/<session>`, and `--train` trains one presence head per node on that node's whole camera pool into `<root>/model_camera/node<id>` (never `<root>/model`, so camera heads cannot overwrite the live presence heads). With `--stage weapon` the datasets join `collect_weapon.py`'s cumulative pool as `<root>/weapon_ds/node<rx>/<session>_link<tag>` (the session's `cam` is the pool's carry position), and `--train` trains the ic27 heads exactly as `collect_weapon.py` does, into `<root>/model_weapon`. The heatmap dataset (`<root>/cam_ds/heatmap/<session>`) is the one deliberate exception: like calibration it merges each node's links into one stream before resampling, because the heatmap head takes one channel per node; its head (`heatmap.joblib`) is trained next to the stage's heads. The dashboard loads `heatmap.joblib` from beside the model it serves, so a camera-trained heatmap is used only when it serves `<root>/model_camera` (presence) or `<root>/model_weapon` (weapon); serving `<root>/model` falls back to the variance-based occupancy estimate.
- Weapon pipeline: σ²[p] variance baseline, sklearn head, CNN head, `SegmentVoter`, `tier_verdict` gate.
- Extra recognition tools built and tested, used by the dashboard and the fusion path: model explainability (`explain.py` — CNN channel-ablation importance, permutation importance for MLP/SVM, confusion matrices), per-link decision fusion with automatic blockage recovery (`link.py` — a link that loses sight of the target drops its own weight so the others take over), learned late fusion across radio bands (`stack.py`), and label-free in-room adaptation (`adapt.py` — refresh input statistics and recalibrate without retraining weights).
- Spatial localization (`localize.py`, `tracking.py`): a joint 2D delay-and-angle MUSIC estimator (SpotFi-style) plus a MUSIC/Bartlett angle spectrum and a Kalman position tracker. It is fully implemented but **parked**, because it needs at least two phase-coherent receive antennas on one radio and the current boards have one receive chain (see §13).
- Ground-truth tools: YOLO vision and segmentation labelers, scripted labeler, timestamp alignment, dataset serializer.
- CLI: `wavetrace capture / calibrate / collect-data / train / localize / run`.
- Web dashboard: spectrograms, node health, live predictions, and Train, which trains the same pools into the same model dirs as the `collect_*.py` scripts.
- Pi 5 GHz node: `firmware/pi/` is implemented and tested against `wavetrace/adapters/source/wire_format.py` via `test_pi_publisher.py`. Not yet validated on real Pi hardware.
- All 631 pytest tests pass offline (`pytest tests/ -q` from repo root with venv active).

### What to Verify on Real Hardware Before Trusting Results

These must be verified on a real capture before trusting any results:

- **Subcarrier count and pattern** — with `WT_BW_HT40 1` the expected count is ~114. Verify on the capture.
- **AGC / PHY gain lock** — confirm the lock is stable across two back-to-back empty captures taken minutes apart.
- **You can't throw away bad-gain frames on the PC.** When a node's automatic gain control (AGC) saturates, that node's σ²[p] feature goes garbage. You might think to drop those frames during weapon training — but you can't from the host. The ESP only sends `mac | timestamp | length | CSI`, so the gain value never leaves the board. If we ever want this, the firmware has to either report the gain or skip those frames itself.
- **Actual CSI sample rate** — the firmware targets ~250 Hz per link. The pipeline estimates `fs` from timestamps; confirm it is within range of the 100 Hz resample target.
- **Antenna performance on weapon detection** — the 8dBi omnidirectional whip is not the directional horn used in the published weapon-detection papers. Tier 1 (flat metal plate, static) is the reality check.
- **Pi 5 Nexmon** — Pi 5 is not in the official Nexmon CSI repo. A community patch exists but is not validated on this hardware.
- **All weapon accuracy tiers** — Real recordings exist (3 datasets, 3 different environments) and have been evaluated. Honest result (group-aware LOGO, no leakage): **no above-chance body-worn weapon signal found in any dataset** (ic27 AUC: 0.290, 0.300, 0.455 — all inverted or near-chance). Desk-based static detection (no person) reached LOGO 0.782–0.800 at 2 of 3 nodes. Body-worn detection at 2.4 GHz with omnidirectional antennas in LOS geometry produces no usable signal. Next required step: NLOS geometry + σ²[p] litmus gate (AUC ≥ 0.65) before any ML attempt.

### What is not built and not planned

- AoA localization (`localize.py` exists but is parked — needs ≥ 2 phase-coherent antennas; the ESP32-S3 has one receive chain).
- Through-wall sensing, vitals detection (breathing/heartbeat).
- On-device model updates — the adaptation mechanism is recalibration plus retraining on the Mac.
- HomeKit / Matter / MQTT integrations.

---

## 12. Troubleshooting

### `mesh_verify.py` shows nothing

1. Check `PC_IP` in `firmware/esp32_node/main/Config.h`. It must match your Mac's current LAN IP (`ipconfig getifaddr en0`). If the IP changed, reflash with the correct value.
2. Check the Mac firewall: `System Settings → Network → Firewall → Options`. UDP 9876 must be allowed.
3. Check that the boards associated to the router. In the serial monitor after flashing, you should see a log line confirming the Wi-Fi connection. If not, `ROUTER_SSID`/`ROUTER_PASS` are wrong, or the router is on a different band.
4. If one specific node is missing, that board has stale NVS. Wipe and reflash: `idf.py erase-flash`, then `rm -rf build` and flash again.

### Boards won't associate to the router

The most common cause is stale NVS (saved Wi-Fi credentials from a previous network). Run `idf.py erase-flash` to wipe the entire chip, then reflash. See [`firmware/README.md`](firmware/README.md) for the full wipe procedure.

### Frame rate is very low (< 5 Hz per link)

Two known causes:
- ESP-NOW is sending at the legacy Wi-Fi rate instead of the HT rate. Check `rate_config` in the firmware.
- FreeRTOS tick rate is 100 Hz instead of 1000 Hz. Check `sdkconfig.defaults` — `CONFIG_FREERTOS_HZ` should be 1000.

### Presence model accuracy is poor

In order of likelihood:

1. Did not recalibrate after moving hardware. Recalibrate; collect fresh data.
2. Mixed capture profiles. Check that you used the same `--root` for baseline, collection, and training.
3. Room not empty during calibration. Re-run `scripts/collect_baseline.py` with the room actually empty.
4. Too few training sessions. Collect more; 3–5 session pairs is the minimum.
5. Wrong evaluation method. Use LOGO, not a random split.

### `pip install -e .` fails with a C++ build error

- Confirm CMake and a C++ compiler are installed: `cmake --version`, `g++ --version` (or `clang++ --version`).
- On macOS: `xcode-select --install` if the compiler is missing.
- Check that the venv is active before running `pip install -e .`.

### `idf.py build` fails with `cmake -E touch` error

Leftover stamp file from a partial `fullclean`. Fix: `rm -rf build`. If it persists, also delete `sdkconfig`, run `idf.py set-target esp32s3`, then build again.

### Pi `mesh_verify` entry at very low rate (< 10 Hz)

The illuminator is not running or the Mac is not associated to modem B. An idle AP only sends ~10 beacons/s. Run `bash firmware/pi/illuminate.sh <modem_B_ip>` from the Mac while associated to modem B's 5 GHz network.

### Web dashboard shows no data

1. Make sure `python -m wavetrace.entrypoints.api.app` is running (port 8000).
2. Check that port 8000 is not blocked. Both the backend and the browser must be on the same machine, or the browser must be able to reach port 8000 on the Mac.

---

## 13. Things that will break the system

**Do not use a random train/test split on CSI data.** Consecutive frames in one session are nearly identical. A random split leaks session-level identity into the test set and gives fake 97–99% accuracy. Use LOGO.

**Do not run presence and weapon on the same `--root` with different subcarrier counts.** Calibration files are tied to a specific subcarrier layout. If you change bandwidth or band, recalibrate.

**Do not mix 2.4 GHz and 5 GHz data in the same model.** They have different subcarrier counts (K). Fuse at the decision level (vote) if you want to combine them.

**Do not implement AoA localization with the ESP32 nodes.** Each ESP32-S3 has one receive chain. The "diversity switch" selects between two antennas on that one chain — it is not a 2-antenna MIMO receiver. `localize.py` exists but is parked.

**Do not use LOS-blocking ("doorway") geometry for weapon detection.** Placing TX and RX on opposite sides of a door so the person walks between them creates a strong direct path that masks the weapon's signal. Use non-LOS geometry.

**Do not attempt weapon tier 2 or 3 before tier 1 works.** If σ²[p] distributions for a flat metal plate are not separated from empty room on held-out data, no CNN will fix it. The problem is in the sensor. Fix the hardware first.

**Do not rename the identifiers listed in the "Skipped — crosses a boundary" section below.** A rename that touches only one side corrupts the wire format silently.

---

## 14. Development

### Running the tests

```bash
source .venv/bin/activate
pytest tests/ -q
```

All tests run on synthetic data. No hardware needed. Expected: 631 passed. `tests/README.md` maps each folder (`unit/`, `integration/`, `contract/`, `golden/`) and test file to what it covers. If a test fails after a code change, fix the test and the code in the same commit.

Tests import `fixtures.synthetic_csi` and the operator scripts as `scripts.<name>`; `pythonpath = ["."]` in `pyproject.toml` puts the repo root on the test path for both `pytest` and `python -m pytest`.

### Rebuilding the C++ extension after a change in `native/`

```bash
pip install -e . --no-build-isolation
```

### Adding a new ESP32 node

1. Flash the new board with the next available `NODE_ID` using `flash.sh`.
2. No Python code change needed — the host aggregator reads `node_id` from the UDP header and handles any count dynamically.

### Changing the model backend (MLP → SVM → CNN)

```bash
wavetrace train ./dataset --stage presence --backend mlp    # default
wavetrace train ./dataset --stage presence --backend svm
wavetrace train ./dataset --stage weapon   --backend cnn --feature-mode cnn
```

The saved `model.joblib` is backend-agnostic at load time — `infer.py` detects the type automatically.

### Switching modes (presence ↔ weapon)

```bash
wavetrace run --calibration ./calib --model ./model.joblib --head-mode presence
wavetrace run --calibration ./calib --model ./model.joblib --head-mode weapon --vote
```

`--vote` enables `SegmentVoter`, which accumulates per-window predictions over a motion segment and emits one stable verdict per segment. Recommended for weapon mode.

### Capture profiles for new hardware

If you add hardware with a different bandwidth (e.g. HT80 on the Pi), create a new directory under `data/` and pass it as `--root`. The pipeline reads K (subcarrier count) from the calibration file and adapts automatically.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| CSI | Channel State Information — complex amplitude + phase per subcarrier, measured from the preamble of every incoming Wi-Fi packet. Encodes how the room shaped the signal. |
| Subcarrier | OFDM splits the Wi-Fi channel into narrow frequency bins. HT20 ≈ 52 subcarriers; HT40 ≈ 114; HT80 ≈ 256. Each bin is an independent amplitude + phase measurement. |
| Link | A directed (TX node, RX node) pair. N mesh nodes → N·(N−1) links per cycle. Each link trains and runs its own model. |
| Node | One hardware unit (ESP32 or Pi) sending UDP CSI. Identified by `node_id` in the datagram header. |
| Gain lock | Fixing the ESP32's AGC amplification to the value measured during calibration. Without this, the receiver adjusts its own gain between sessions and the amplitude baseline shifts. Implemented in C++ `GainLock`; coefficient of variation (σ/μ) is the fallback if the firmware lock is unavailable. |
| HT20/40/80 | 802.11 channel-width modes: 20/40/80 MHz. Wider = more subcarriers = more frequency resolution, but uses more spectrum. |
| Calibration | One `collect_baseline.py` run: records a quiet empty room, stores the gain-lock scalar, NBVI mask, and quiet baseline. Redo it when the room layout, antenna position, or radio band changes. |
| LOGO | Leave-One-Session-Out — the correct cross-validation for CSI data. Train on all sessions except one, test on the held-out session, repeat for every session. Prevents session-identity leakage. |
| LOSO | Leave-One-Subject-Out — same idea but the held-out unit is one person's sessions. Tests generalization to new individuals. |
| σ²[p] | Per-packet inter-subcarrier variance — variance of CSI amplitude across subcarriers within one packet. Metal reflects evenly → lower σ²; human body scatters unevenly → higher σ². The primary weapon discriminator. |
| NBVI | Narrowband variance index — ranks subcarriers by how much their amplitude changes when a person enters. The top-K are kept; the rest are discarded to reduce noise and computation. Run offline during calibration. |
| ESP-NOW | Espressif's connectionless 802.11 protocol used for the TX burst. No association required; RX nodes capture CSI from the packet preamble without decoding the payload. |
| Non-LOS | Non-line-of-sight geometry: TX and RX are positioned so there is no direct unobstructed path between them. The measured signal comes from reflections. Recommended for weapon detection. |
| Wire format v2/v3 | Binary UDP datagram layout shared between `firmware/pi/publisher.py` and `wavetrace/adapters/source/wire_format.py`. v2 = int8 I/Q (ESP32 nodes); v3 = int16 I/Q with fixed scale (Pi node — preserves absolute amplitude for weapon detection). |
| ESP-IDF | Espressif IoT Development Framework — the build toolchain and SDK used to compile and flash the ESP32 firmware. |
| Nexmon CSI | A firmware patch for Broadcom/Cypress Wi-Fi chips (including the Pi's CYW43455) that exposes raw CSI over a local UDP socket. |

---

## 16. Where to look next

| Topic | File |
|---|---|
| ESP32 mesh flash + wipe steps | [`firmware/README.md`](firmware/README.md) |
| Pi 5 GHz Nexmon node setup | [`firmware/pi/README.md`](firmware/pi/README.md) |
| Python library — modules and subfolders | [`wavetrace/README.md`](wavetrace/README.md) |
| Web dashboard endpoints and startup | [`web/README.md`](web/README.md) |
| Data directory layout | [`data/README.md`](data/README.md) |
| CLI flags (all subcommands) | [`wavetrace/entrypoints/cli/main.py`](wavetrace/entrypoints/cli/main.py) |
| Development history and design decisions | `projectProgress.md` |
| Next radio front end (UWB) — hardware research | `docs/HARDWARE_SELECTION.md` (local only: `docs/` is gitignored) |

---

## Skipped — crosses a boundary (do not rename without updating both sides)

The following identifiers cross a hardware or wire-format boundary. Renaming one side without the other corrupts the data stream silently, and no test will catch it.

- UDP datagram struct fields and header layout — `wavetrace/adapters/source/wire_format.py` ↔ `firmware/pi/publisher.py` ↔ `firmware/esp32_node/main/Main.cpp`
- Wire format version bytes and JSON header keys: `v`, `node`, `tx`, `ntp_ms`
- `Config.h` macro names: `NODE_ID`, `PC_IP`, `ROUTER_SSID`, `ROUTER_PASS`, `CHANNEL`
- ESP-IDF sdkconfig keys in `sdkconfig.defaults`
- CLI flag names: `--root`, `--node`, `--carry`, `--subject`, `--head-mode`, `--stage`, `--vote`
- `data/` subdirectory names: `cal/`, `model/`, `model_weapon/`, `model_count/`, `2g4_ht20/`, `2g4_ht40/`, `5g_ht40/`, `5g_ht80/`
- pybind11 binding names in `native/Bindings.cpp` and `wavetrace/_wavetrace.pyi`
- UDP ports: 9876 (CSI mesh), 5566 (legacy 1-TX rig), 9877 (health monitor), 9878 (node discovery), 9879 (UWB CIR, wire v4), 8000 (web dashboard)
