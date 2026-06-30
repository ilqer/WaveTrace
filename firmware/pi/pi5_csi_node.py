#!/usr/bin/env python3
"""Entry point: stream 5 GHz CSI from the Pi's Nexmon firmware to the WaveTrace host as node 5.

  Nexmon firmware --(UDP 5500, on-Pi)--> NexmonReader --> quantize --> BatchPublisher
                                                                  --(UDP 9876, eth0)--> Mac

Run on the Pi after the firmware is set up (setup_nexmon.sh) and capture is started
(start_capture.sh), with the Mac illuminating modem B (illuminate.sh):
    python3 pi5_csi_node.py
Fill in firmware/pi/config.py first (PC_IP, AP_BSSID).

NOTE: for cross-node fusion with the ESP mesh, keep this Pi's clock NTP-synced to the same source
the ESP nodes use (the Mac). On the Pi: `sudo timedatectl set-ntp true` pointing at the Mac, or
add the Mac as an NTP server. Single-node (Pi-only) weapon/presence does not need this.
"""
import subprocess
import time

import config
from nexmon_reader import NexmonReader
from publisher import BatchPublisher, mac_to_bytes, quantize_csi, quantize_csi_i16

LOW_RATE_HZ = 50.0   # warn below this: usually the Mac stopped illuminating modem B (AP idle ~10 Hz)
LOW_RATE_HOLD = 5    # consecutive low-rate seconds before escalating to a sustained-outage warning


def _ntp_synced() -> bool | None:
    """True/False if the Pi clock is NTP-disciplined, None if it can't be determined (no systemd).
    The publisher stamps ntp_ms from the OS wall clock, so cross-node fusion drifts if this is False."""
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() == "yes" if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def main() -> None:
    config.validate()
    synced = _ntp_synced()
    if synced is False:
        # Single-node Pi capture tolerates an unsynced clock; multi-node fusion does NOT (timestamps
        # from this Pi and the ESP mesh won't share an origin). Warn prominently rather than abort.
        print("[pi5-csi] WARNING: Pi clock is NOT NTP-synchronized. Cross-node fusion with the ESP "
              "mesh will drift. Fix: `sudo timedatectl set-ntp true` (point at the Mac). "
              "Single-node Pi-only capture can ignore this.", flush=True)
    elif synced is None:
        print("[pi5-csi] note: could not verify NTP sync (timedatectl unavailable); ensure the Pi "
              "clock is disciplined before multi-node fusion.", flush=True)
    ap_mac = mac_to_bytes(config.AP_BSSID)
    reader = NexmonReader(config.NEXMON_PORT, expect_s=config.EXPECT_S, ap_mac=ap_mac)
    pub = BatchPublisher(config.PC_IP, config.UDP_PORT, config.NODE_ID, config.AP_BSSID,
                         ver=config.WIRE_VER)
    # int16 (ver 3) keeps absolute amplitude for weapon; int8 (ver 2) is the lighter presence-only path.
    encode = quantize_csi_i16 if config.WIRE_VER == 3 else quantize_csi

    print(
        f"[pi5-csi] node={config.NODE_ID} -> {config.PC_IP}:{config.UDP_PORT} | "
        f"AP={config.AP_BSSID} | S={config.EXPECT_S} ver={config.WIRE_VER} | nexmon:{config.NEXMON_PORT}",
        flush=True,
    )

    sent = 0
    low_streak = 0   # consecutive low-rate seconds, for the sustained-outage watchdog (Item 15)
    t_report = time.monotonic()
    t_flush = t_report
    try:
        for _ts, _mac, csi in reader.frames():
            ts_us = time.monotonic_ns() // 1000  # host only needs consistent per-link spacing
            pub.add(encode(csi, config.CSI_SCALE), ts_us)
            sent += 1

            now = time.monotonic()
            # Bound latency at low rates: force a flush ~50 ms even if the MTU batch isn't full.
            if now - t_flush >= 0.05:
                pub.flush()
                t_flush = now
            if now - t_report >= 1.0:
                rate = sent / (now - t_report)
                low = rate < LOW_RATE_HZ
                low_streak = low_streak + 1 if low else 0
                if low_streak >= LOW_RATE_HOLD:
                    warn = f"  <-- SUSTAINED LOW ({low_streak}s): illuminator likely down — restart illuminate.sh"
                elif low:
                    warn = "  <-- LOW: is the Mac illuminating modem B?"
                else:
                    warn = ""
                print(f"[pi5-csi] {rate:.0f} frames/s (S={csi.size}){warn}", flush=True)
                sent = 0
                t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        pub.close()
        print("\n[pi5-csi] stopped", flush=True)


if __name__ == "__main__":
    main()
