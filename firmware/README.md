# WaveTrace firmware

## Current path — full-mesh node (`esp32_node/`)

Every board runs the same binary. `NODE_ID` is the only per-board difference; there is no TX-only or RX-only role. The mesh is time-division round-robin: one node transmits an ESP-NOW burst while the others capture CSI; the TX role rotates. N nodes → N·(N−1) directed (tx, rx) links per cycle.

Each node joins the router as a STA, which locks all boards to the same channel so they hear each other's ESP-NOW traffic automatically. CSI datagrams go to the Mac over UDP port 9876.

Turn-taking is token-passing: the last frame of a burst names the next node to transmit. Node 1 is the leader — it starts the first burst and restarts the token if the air goes quiet.

### Step 1 — Install ESP-IDF (once, ~30–60 min)

```bash
mkdir -p ~/esp && cd ~/esp
git clone -b release/v5.3 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh esp32s3
. ~/esp/esp-idf/export.sh    # you must run this in every new shell before flashing
```

### Step 2 — Configure `config.h` (once per deployment)

Open `firmware/esp32_node/main/config.h` and set:

```c
#define WIFI_SSID   "your-router-ssid"
#define WIFI_PASS   "your-router-password"
#define PC_IP       "192.168.x.x"    // your Mac's IP on that LAN
#define MESH_NODES  2                // number of boards you are flashing (start at 2)
```

To find your Mac's IP: **System Settings → Network → your Wi-Fi connection → IP address**. Or run `ipconfig getifaddr en0` in the Mac terminal. Give the Mac a static DHCP lease on the router so this IP does not change.

### Step 3 — Flash each board

Plug in one board at a time. Find its USB port:

```bash
ls /dev/cu.usbserial-* /dev/cu.usbmodem*
```

Then flash (run from the repo root):

```bash
. ~/esp/esp-idf/export.sh
./flash.sh node 1 /dev/cu.usbmodem1101    # leader (NODE_ID=1)
./flash.sh node 2 /dev/cu.usbmodem1101    # second board
```

Each `flash.sh` call builds, flashes, and opens a serial monitor (Ctrl+] to exit). You should see the board print its node ID and associate to the router.

### Step 4 — Verify

Run from the repo root with the Python venv active and both boards powered (USB only needed for flashing — they can run from any 5 V supply after that):

```bash
.venv/bin/python mesh_verify.py
```

Expected output: two lines, one for link `1->2` and one for `2->1`, both with a frame rate > 0. If nothing appears, check `PC_IP` in `config.h` and that the Mac firewall is not blocking UDP 9876 (`System Settings → Network → Firewall`).

### Wipe and flash from scratch

Do this when a board fails to associate (stale NVS), after changing `config.h`, or to guarantee a clean image.

```bash
. ~/esp/esp-idf/export.sh
cd firmware/esp32_node

idf.py -p /dev/cu.usbserial-120 erase-flash          # wipes app + NVS + Wi-Fi credentials
rm -rf build                                           # force a true from-source rebuild
idf.py -p /dev/cu.usbserial-120 -DNODE_ID=1 build flash monitor
```

Use `rm -rf build`, **not** `idf.py fullclean`. `fullclean` can leave stamp files half-removed and the next build fails with a `cmake -E touch` error. If that happens, also delete `sdkconfig`, then run `idf.py set-target esp32s3` before building.

### Scale up

Set `MESH_NODES` in `config.h` to the number of boards and flash each one with the matching `NODE_ID`. No Python code change is needed — the aggregator is node-count-agnostic.

---

## Legacy 1-TX/6-RX rig (`esp32_rx/` + `esp32_tx/`)

Six ESP32-S3 boards act as RX-only nodes; a seventh is the dedicated TX. All send CSI to the Mac on UDP port 5566. This was the original topology before the mesh firmware.

```
TX (ESP-NOW broadcast, ch 11) ──► 6× RX (ESP32-S3) ──UDP:5566──► PC (UdpSource)
```

**RF constraint:** each RX uses its one radio for both AP association (to send UDP) and CSI capture (to hear the TX's ESP-NOW). The AP, TX, and all RX boards must be on the **same channel** (default 11). Set `PC_IP` in `esp32_rx/main/config.h` to a static IP on that LAN; if the IP changes, the boards can't reach the Mac.

### Install ESP-IDF (once)

Same as the mesh path above — see Step 1.

### Flash the TX

```bash
. ~/esp/esp-idf/export.sh
./flash.sh tx /dev/cu.usbmodem1101
```

### Flash the six RX boards

```bash
./flash.sh rx 1 /dev/cu.usbmodem1101    # board 1 → node_id 1
./flash.sh rx 2 /dev/cu.usbmodem1101    # board 2 → node_id 2
# ... through node 6
```

### Verify

```bash
.venv/bin/python -c "
from wavetrace.Source import UdpSource
import collections
c = collections.Counter()
for f in UdpSource(5566, timeout_s=10).frames():
    c.update([f.node_id])
print(c)"
```

Expect all six node IDs with roughly equal counts (~100 frames/s each). A missing node means it failed to associate, is on the wrong channel, or has the wrong `DEST_IP`.

---

## Pi 5 GHz nexmon node (`pi/`)

The Raspberry Pi captures 5 GHz CSI on its onboard CYW43455 chip via Nexmon and streams it to the Mac as **node 5** on the same UDP port 9876 the ESP mesh uses. See [pi/README.md](pi/README.md) for setup steps.
