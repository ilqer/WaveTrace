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

Open `firmware/esp32_node/main/config.h` and set the four values that are specific to your network:

```c
#define ROUTER_SSID  "your-router-ssid"
#define ROUTER_PASS  "your-router-password"
#define PC_IP        "192.168.x.x"    // your Mac's LAN IP (run: ipconfig getifaddr en0)
#define MESH_NODES   2                // number of boards you are flashing right now
```

To find your Mac's IP: `System Settings → Network → your connection → IP address`, or run `ipconfig getifaddr en0`. Give the Mac a static DHCP lease on the router so this IP never changes between sessions.

The remaining settings in `config.h` are tuned for the current 6-node, HT40, channel-6 setup and do not need to change:

| Setting | Value | Meaning |
|---|---|---|
| `WT_BW_HT40` | `1` | HT40 bandwidth (router must also be set to 40 MHz) |
| `BURST_LEN` | `10` | Frames per ESP-NOW burst |
| `BURST_MS` | `2` | ms between burst frames (requires `CONFIG_FREERTOS_HZ=1000`) |
| `MAX_NODES` | `16` | Ring capacity; active count is discovered at runtime — no change when adding boards |
| `CSI_MAX_HZ` | `0` | Uncapped; the host resamples each link to 100 Hz |
| `SNTP_SERVER` | `PC_IP` | Nodes use the Mac as their NTP clock source (`ntp_server.py`) |
| `CSI_UDP_PORT` | `9876` | CSI datagrams to the Mac |
| `HEALTH_UDP_PORT` | `9877` | Per-node heartbeat |
| `DISCOVERY_PORT` | `9878` | Nodes ping this port to learn the current `PC_IP` |

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

### Step 4 — Start the NTP server and verify

The firmware uses the Mac as its SNTP clock source. Run this in a dedicated terminal before verifying or collecting any data:

```bash
python ntp_server.py
```

Then verify CSI is arriving (venv active, boards powered — USB only needed for flashing):

```bash
python mesh_verify.py
```

Expected output: one line per directed link. With 2 boards: `1->2` and `2->1`, both at a non-zero frame rate. With 6 boards: 30 links (6×5). If nothing appears: wrong `PC_IP` in `config.h`, Mac firewall blocking UDP 9876, or boards not yet associated (wait ~10 s after power-on). Check that the firewall also allows UDP 9877 (health) and 9878 (discovery).

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
