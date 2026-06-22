# Pi 5 GHz CSI node (WaveTrace node 5)

A Raspberry Pi 5 that captures 5 GHz Wi-Fi CSI on its **onboard** chip (Infineon/Cypress
CYW43455) via **Nexmon CSI** and streams it to the WaveTrace host in the existing binary
v2 UDP format. The host auto-discovers it as **node 5** — no host code changes.

No external NIC is needed: the CYW43455 is the same chip family as the Pi 3B+/4B and is
Nexmon CSI's flagship-supported chip (5 GHz up to 80 MHz, 1×1 → single stream).

## Topology (2 modems, 1 Pi, 1 Mac)

| Device | Modem A (backhaul LAN) | Modem B (5 GHz illuminator) |
|---|---|---|
| **Pi 5** | **eth0 (wired)** → CSI UDP 9876 to Mac | **wlan0 = monitor mode**, captures CSI (not associated) |
| **MacBook** | on the LAN → receives UDP 9876 | **Wi-Fi associated** → pings modem B to make it transmit |

**The Pi's single Wi-Fi radio is fully used for CSI capture — the backhaul to the Mac MUST
be wired Ethernet.** The Pi is never associated to modem B; it only listens on its channel.

## Part A — Pi firmware (once, over SSH)

1. Pi 5 only: add `kernel=kernel8.img` to `/boot/firmware/config.txt`, reboot.
2. Build + flash Nexmon CSI for the CYW43455 (`Makefile.rpi` flow; build `nexutil` with
   `USE_VENDOR_CMD=1`). Refs: nexmon_csi Discussion #395; `nexmonster/nexmon_csi` `pi-5.4.51-plus`.
3. Set the CSI filter to modem B's channel/width/BSSID and enable monitor mode, e.g.:
   ```bash
   PARAMS=$(makecsiparams -c 36/40 -C 1 -N 1 -m <MODEM_B_BSSID>)
   nexutil -Iwlan0 -s500 -b -l34 -v"$PARAMS"
   iw dev wlan0 interface add mon0 type monitor && ip link set mon0 up
   ```
   Verify: `tcpdump -i wlan0 dst port 5500` shows CSI frames arriving.

## Part B — Transmitter (modem B) + illuminator (Mac)

- In **modem B** admin (5 GHz): fixed **channel 36** (non-DFS), **40 MHz (HT40)**, disable
  auto-channel / DFS / band-steering, keep 802.11n/ac. Record its **5 GHz BSSID**.
- An idle AP only beacons (~10 Hz). To reach ≥150 Hz, the **Mac** (associated to modem B's
  5 GHz) generates traffic so the AP transmits:
  ```bash
  ping -i 0.003 <modem_B_ip>     # ~300 Hz   (or: iperf3 -u -b 5M -c <modem_B_ip>)
  ```
  The Pi sniffs the AP's frames; host link = `(AP_short → 5)`.

## Run (on the Pi)

```bash
pip install -r requirements.txt      # numpy
# edit config.py: PC_IP (Mac LAN IP), AP_BSSID (modem B BSSID), EXPECT_S (128 for HT40)
python3 pi5_csi_node.py
```

## Validate (on the Mac)

```bash
.venv/bin/python mesh_verify.py                  # expect an (AP_short → 5) link at ≥150 Hz
.venv/bin/python collect_baseline.py --node 5    # writes data/cal/node5
.venv/bin/python collect_presence.py             # then run_live_mesh.py — node 5 LOGO beats majority
```

If `mesh_verify` is empty: wrong `PC_IP` or a macOS firewall blocking UDP 9876 — not a
capture bug. Check the network path first.

## Files

- `config.py` — deployment settings (PC_IP, AP_BSSID, ports, width, scale).
- `nexmon_reader.py` — local Nexmon UDP 5500 → `(ts, mac, complex csi[S])`; pins width.
- `publisher.py` — packs the byte-exact v2 record (mirrors `wavetrace/Source.py`) → UDP 9876.
- `pi5_csi_node.py` — glue loop (reader → quantize → batch → send) with a per-second rate print.

## Notes / future

- **int8 / ver-2** wire format now (zero host change) — validate **presence** first. If weapon
  or people-count later need more fidelity, add a backward-compatible **ver-3 (int16)** branch
  in `wavetrace/Source.py` (`_parse_bin_header` accepts `ver in (2,3)`; `_iter_bin_records`
  reads `<i2` for ver 3). The ESP ver-2 path keeps working.
- The int16 I/Q ordering in `parse_nexmon_csi` is irrelevant for presence (amplitude); confirm
  on hardware only if you need phase.
