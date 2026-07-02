#!/usr/bin/env python3
"""Entry point: stream 5 GHz CSI from Nexmon to WaveTrace host as node 5.
Pipeline: Nexmon (UDP 5500) -> NexmonReader -> quantize -> BatchPublisher (UDP 9876) -> Mac

Run after setup_nexmon.sh, start_capture.sh, and illuminate.sh: python3 pi5_csi_node.py
Requires config.py filled.

Note: Cross-node fusion requires Pi clock NTP-synced to Mac (`sudo timedatectl set-ntp true`).
Single-node capture does not.
"""
import subprocess
import time

import config
from nexmon_reader import NexmonReader
from publisher import BatchPublisher, mac_to_bytes, quantize_csi, quantize_csi_i16

LOW_RATE_HZ = 50.0   # Warn below this (usually Mac stopped illuminating).
LOW_RATE_HOLD = 5    # Consecutive low-rate seconds for sustained warning.


def _ntp_synced() -> bool | None:
    """Check if Pi clock is NTP-disciplined. Crucial for cross-node fusion sync."""
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
        # Single-node capture tolerates unsynced clock; multi-node fusion requires it. Warn.
        print("[pi5-csi] WARNING: Pi clock is not NTP-synchronized. Cross-node fusion with the ESP "
              "mesh will drift. Fix: `sudo timedatectl set-ntp true` (point at the Mac). "
              "Single-node Pi-only capture can ignore this.", flush=True)
    elif synced is None:
        print("[pi5-csi] note: could not verify NTP sync (timedatectl unavailable); make sure the "
              "Pi clock is disciplined before multi-node fusion.", flush=True)
    apMac = mac_to_bytes(config.AP_BSSID)
    reader = NexmonReader(config.NEXMON_PORT, expect_s=config.EXPECT_S, ap_mac=apMac)
    pub = BatchPublisher(config.PC_IP, config.UDP_PORT, config.NODE_ID, config.AP_BSSID,
                         ver=config.WIRE_VER)
    # int16 (ver 3) keeps absolute amplitude for weapon; int8 (ver 2) is presence-only.
    encode = quantize_csi_i16 if config.WIRE_VER == 3 else quantize_csi

    print(
        f"[pi5-csi] node={config.NODE_ID} -> {config.PC_IP}:{config.UDP_PORT} | "
        f"AP={config.AP_BSSID} | S={config.EXPECT_S} ver={config.WIRE_VER} | nexmon:{config.NEXMON_PORT}",
        flush=True,
    )

    sent = 0
    lowStreak = 0   # consecutive low-rate seconds, feeds the sustained-outage watchdog above
    tReport = time.monotonic()
    tFlush = tReport
    try:
        for _ts, _mac, csi in reader.frames():
            ts_us = time.monotonic_ns() // 1000  # host only needs consistent per-link spacing
            pub.add(encode(csi, config.CSI_SCALE), ts_us)
            sent += 1

            now = time.monotonic()
            # Force flush at ~50ms to bound latency at low rates.
            if now - tFlush >= 0.05:
                pub.flush()
                tFlush = now
            if now - tReport >= 1.0:
                rate = sent / (now - tReport)
                low = rate < LOW_RATE_HZ
                lowStreak = lowStreak + 1 if low else 0
                if lowStreak >= LOW_RATE_HOLD:
                    warn = f"  <-- SUSTAINED LOW ({lowStreak}s): illuminator likely down, restart illuminate.sh"
                elif low:
                    warn = "  <-- LOW: is the Mac illuminating modem B?"
                else:
                    warn = ""
                print(f"[pi5-csi] {rate:.0f} frames/s (S={csi.size}){warn}", flush=True)
                sent = 0
                tReport = now
    except KeyboardInterrupt:
        pass
    finally:
        pub.close()
        print("\n[pi5-csi] stopped", flush=True)


if __name__ == "__main__":
    main()
