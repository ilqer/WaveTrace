# WaveTrace firmware — bring-up runbook

## Full-mesh node firmware (`esp32_node/`) — current path

A 2.4 GHz radio is **half-duplex**, so true simultaneous TX+RX is impossible. The mesh is therefore
a **time-division round-robin**: in each turn exactly one node transmits an ESP-NOW broadcast burst
while every other node captures CSI from it; the TX role rotates through all nodes. N nodes →
**N·(N−1) directed (tx,rx) links per cycle** (vs. 6 for the legacy 1-TX/6-RX rig). Every board runs
the **same** firmware — `NODE_ID` is the only per-board difference; there is no TX/RX split.

- **Backhaul + channel lock:** each node is a **STA on RD-WIN1** (no SoftAP). STA association forces
  every node onto the router's channel, so they all hear each other's ESP-NOW automatically, and CSI
  is sent over the same link to the PC (`UDP 9876`, hardcoded `PC_IP` — static-lease the Mac or edit
  `esp32_node/main/config.h`).
- **Turn-taking = token passing; the burst IS the token:** a burst's last frame names the next node
  id. **Node 1 is the leader** — it kicks off the first burst and restarts the token if the air goes
  silent. No global clock drives the rotation.
- **SNTP** stamps CSI with a shared wall clock *only* for cross-node alignment (data retrieval), not
  for turn-taking.
- Each CSI datagram header carries both ends of the link: `{"v":1,"node":<rx>,"tx":<tx>,"ntp_ms":…}`.

Set `MESH_NODES` in `esp32_node/main/config.h` (start at **2** for the proof). Flash each board:
```bash
. ~/esp/esp-idf/export.sh
./flash.sh node 1 /dev/cu.usbmodem1101    # leader
./flash.sh node 2 /dev/cu.usbmodem1101    # second board (one at a time)
```
Verify on the PC (on RD-WIN1, boards powered — USB only needed for flashing):
```bash
.venv/bin/python mesh_verify.py           # expect both 1->2 and 2->1 at ~equal rates
```
Scale up by raising `MESH_NODES` and flashing nodes 3..N. Next integration step (after the proof):
thread `"tx"` through `parse_batch`/`CsiFrame` and key `NodeAggregator`/`LinkVoter` on `(tx,rx)`.

---

## Legacy 1-TX/6-RX rig (`esp32_rx/` + `esp32_tx/`)

Topology during **training**: 6× ESP32-S3 RX + 1 dedicated ESP32-S3 TX sense on **2.4 GHz**, all
sending UDP CSI to the **PC** (`UdpSource`, port 5566). The Raspberry Pi is a separate **5 GHz**
nexmon node (TODO, below) and uploads its camera for ground-truth labels. After training, the model
deploys to the Pi with **no camera**.

```
TX (ESP-NOW broadcast, ch 11) ──▶ 6× RX (ESP32-S3) ──UDP:5566──▶ PC 10.6.1.121 (UdpSource)
Pi camera ──HTTP MJPEG:8090──▶ PC (CameraLabeler, training only)
Pi nexmon 5 GHz CSI ──UDP:5566 node=100──▶ PC   [TODO]
```

## Hard RF constraint (read first)
Each RX has ONE 2.4 GHz radio. It associates to the AP to send UDP **and** must hear the TX's
ESP-NOW on the same channel. Therefore: **the dedicated 2.4 GHz AP, the TX `CHANNEL`, and the RX
`CHANNEL` must all be the same channel** (default 11). Lock the AP to channel 11 (disable auto).
The PC must be on that same AP/LAN as 10.6.1.121 (give it a static DHCP lease — if the PC IP
changes, update `DEST_IP` in `esp32_rx/main/config.h`).

ESP32-S3 is 2.4 GHz only — the 6-node mesh cannot be 5 GHz. 5 GHz is the Pi's job.

## 0. Install ESP-IDF v5.x (once, ~30–60 min)
```bash
mkdir -p ~/esp && cd ~/esp
git clone -b release/v5.3 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh esp32s3
. ~/esp/esp-idf/export.sh      # run this in every new shell before flashing
```

## 1. Configure once
Edit `esp32_rx/main/config.h`: set `WIFI_SSID`, `WIFI_PASS`, confirm `DEST_IP "10.6.1.121"` and
`CHANNEL 11`. Edit `esp32_tx/main/config.h` only if you change the channel (must match).

## 2. Flash the TX (1 board)
```bash
. ~/esp/esp-idf/export.sh
./flash.sh tx /dev/cu.usbmodem1101      # Ctrl-] to exit the monitor
```

## 3. Flash the 6 RX boards (NODE_ID is the only per-board change)
Plug in ONE board at a time; find its port with `ls /dev/cu.usbmodem*`.
```bash
./flash.sh rx 1 /dev/cu.usbmodem1101    # board #1 -> node 1
./flash.sh rx 2 /dev/cu.usbmodem1101    # board #2 -> node 2
# ... through node 6
```

## 4. Verify on the PC
```bash
.venv/bin/python -c "from wavetrace.Source import UdpSource; \
import collections; c=collections.Counter(); \
[c.update([f.node_id]) for f in UdpSource(5566, timeout_s=10).frames()]; print(c)"
```
Expect a Counter with all 6 node ids and roughly equal counts (~100/s each). If a node is missing:
that board isn't associated to the AP, is on the wrong channel, or `DEST_IP` is wrong.

## 5. Pi camera (training only)
On the Pi: `sudo apt install -y python3-picamera2 && python3 pi/camera_stream.py`
The PC reads `http://<pi-ip>:8090/stream.mjpg` and feeds it to the camera labeler during collection.

## Pi 5 GHz nexmon CSI node (`pi/` — IMPLEMENTED 2026-06-22)
Python node that captures 5 GHz CSI on the Pi's **onboard CYW43455** via **Nexmon CSI** (no
external NIC) and emits the current **binary v2** datagram to the PC as **node 5** on **UDP 9876**
(same format/port the ESP mesh uses → host auto-discovers it, zero host changes). See
[`pi/README.md`](pi/README.md) for firmware bring-up, the 2-modem topology (Pi RX-only over
wired eth, Mac illuminates modem B), and run/validate steps. Files: `nexmon_reader.py`,
`publisher.py`, `pi5_csi_node.py`, `config.py`. Tested via `tests/TestPiPublisher.py`
(round-trips the wire format through `wavetrace/Source.py`).
