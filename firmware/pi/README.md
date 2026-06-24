# Pi 5 GHz CSI node (WaveTrace node 5)

The Raspberry Pi captures 5 GHz Wi-Fi CSI via Nexmon CSI and streams it to the Mac as **node 5** on UDP port 9876 — the same format and port the ESP mesh uses. No host code changes needed. Runs at HT80 (256 subcarriers), wire format v3 (int16 I/Q), which preserves absolute amplitude for the weapon feature.

**Pi 5 compatibility note:** The Pi being used here is a Raspberry Pi 5. Nexmon CSI officially supports Pi 3B+ and Pi 4 (CYW43455C0 chip). Pi 5 uses a different Wi-Fi chip and is not in the official Nexmon repo. Community patches exist but require manual kernel-version pinning and are not guaranteed to work with every Pi 5 OS image. If Nexmon installation fails on Pi 5, the options are: (a) use a Pi 3B+ or Pi 4 for this node, or (b) skip the 5 GHz arm and use only the 6-node ESP32 mesh at 2.4 GHz HT40 until a Pi 4 is available. The 2.4 GHz mesh alone is sufficient for presence detection and is the current active path.

## Topology

| Device | Modem A (backhaul LAN) | Modem B (5 GHz illuminator) |
|---|---|---|
| Pi | **eth0 (wired)** → CSI UDP 9876 to Mac | **wlan0 = monitor mode**, sniffs CSI (not associated) |
| Mac | on the LAN → receives UDP 9876 | **Wi-Fi associated** → pings modem B so it transmits |

The Pi's single Wi-Fi radio is in monitor mode, so the backhaul to the Mac must be wired Ethernet.

## Setup steps

**1. Modem B (once, in its admin UI):** set 5 GHz, fixed channel 36, 80 MHz (HT80). Disable auto-channel, DFS, and band-steering. Note its 5 GHz BSSID and LAN IP.

**2. Fill in `firmware/pi/config.py` (on the Mac, before copying to the Pi):** set `PC_IP` to your Mac's LAN IP (`ipconfig getifaddr en0`) and `AP_BSSID` to modem B's 5 GHz BSSID. The script refuses to start until both are set.

**3. Install Nexmon CSI on the Pi (SSH into the Pi, run once):**
```bash
# on the Pi (ssh pi@<pi-ip>)
cd /path/to/WaveTrace/firmware/pi
bash setup_nexmon.sh    # builds and installs Nexmon CSI; may reboot once for the kernel pin, then re-run
```

**4. Start capture (SSH into the Pi, run each boot):**
```bash
# on the Pi
bash start_capture.sh   # sets the HT80 CSI filter, puts wlan0 in monitor mode, checks 5 frames
```

**5. Illuminate (on the Mac — must be associated to modem B's 5 GHz network):**
```bash
# on the Mac
bash firmware/pi/illuminate.sh <modem_B_ip>    # ~300 pings/s so the AP keeps transmitting
```

An idle AP only sends ~10 beacons/s — not enough for useful CSI. The illuminator keeps the AP transmitting so the Pi can capture at a high rate.

**6. Stream CSI to the Mac (on the Pi):**
```bash
# on the Pi
pip install numpy          # only dependency
python3 pi5_csi_node.py   # prints frames/s to the terminal; warns if the rate drops below threshold
```

## Validate (on the Mac, from the repo root)

```bash
source .venv/bin/activate
python mesh_verify.py                    # expect a link at ≥150 Hz from node 5 (the Pi)
python collect_baseline.py --node 5     # writes data/cal/node5
```

If `mesh_verify` shows nothing from node 5: wrong `PC_IP` in `config.py`, or macOS firewall is blocking UDP 9876 (`System Settings → Network → Firewall`). Check the network path before assuming a Nexmon problem.

## Configuration reference (`config.py`)

The two values you must set before running:

| Setting | What to put |
|---|---|
| `PC_IP` | Your Mac's LAN IP on the wired backhaul network (`ipconfig getifaddr en0` on Mac) |
| `AP_BSSID` | Modem B's 5 GHz BSSID (from the router admin UI, e.g. `aa:bb:cc:dd:ee:ff`) |

The remaining settings are fixed for the current hardware:

| Setting | Value | Meaning |
|---|---|---|
| `NODE_ID` | `5` | Unique node ID; host auto-discovers alongside ESP nodes 1–4 |
| `CHANNEL_SPEC` | `"36/80"` | Channel 36, HT80 (256 subcarriers) |
| `EXPECT_S` | `256` | Frames with a different subcarrier count are dropped |
| `WIRE_VER` | `3` | int16 I/Q (preserves absolute amplitude for weapon detection) |
| `CSI_SCALE` | `1.0` | Fixed scale — never per-frame auto-scale for weapon mode |
| `UDP_PORT` | `9876` | Same port the ESP mesh uses; host auto-discovers |

## Files

| File | Purpose |
|---|---|
| `config.py` | All deployment settings; `validate()` fails loudly if `PC_IP` or `AP_BSSID` are still set to `"TODO_*"` |
| `setup_nexmon.sh` | one-time Nexmon CSI firmware build and install |
| `start_capture.sh` | per-boot: `makecsiparams` (HT80) + monitor mode + a 5-frame check |
| `illuminate.sh` | run on the Mac; pings modem B so it emits frames for the Pi to sniff |
| `nexmon_reader.py` | reads Nexmon's local UDP port 5500 → `(timestamp, mac, complex csi[S])` |
| `publisher.py` | packs the byte-exact v2/v3 record (mirrors `wavetrace/Source.py`) → UDP 9876 |
| `pi5_csi_node.py` | main loop: reader → quantize → batch → send, with a rate printout |

## Notes

- Weapon mode requires `WIRE_VER=3` and a fixed `CSI_SCALE` (both are the default). A per-frame auto-scale removes the absolute amplitude that the metal signature lives in.
- I/Q ordering in `parse_nexmon_csi` does not matter for amplitude-based presence detection; verify it on hardware before using phase-based features.
- For fusing this node with the ESP mesh, NTP-sync the Pi's clock to the same source the ESP nodes use (the Mac). Weapon or presence detection using the Pi alone does not need NTP.
