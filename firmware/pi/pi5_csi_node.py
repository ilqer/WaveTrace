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


def _is_ntp_synced() -> bool | None:
    """Check if Pi clock is NTP-disciplined. Crucial for cross-node fusion sync.
    None when timedatectl is missing, fails or times out."""
    try:
        timedatectl_result = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                                            capture_output=True, text=True, timeout=3)
        return timedatectl_result.stdout.strip() == "yes" if timedatectl_result.returncode == 0 else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def main() -> None:
    config.validate()
    is_clock_synced = _is_ntp_synced()
    if is_clock_synced is False:
        print("[pi5-csi] WARNING: Pi clock is not NTP-synchronized. Cross-node fusion with the ESP "
              "mesh will drift. Fix: `sudo timedatectl set-ntp true` (point at the Mac). "
              "Single-node Pi-only capture can ignore this.", flush=True)
    elif is_clock_synced is None:
        print("[pi5-csi] note: could not verify NTP sync (timedatectl unavailable); make sure the "
              "Pi clock is disciplined before multi-node fusion.", flush=True)
    access_point_mac = mac_to_bytes(config.AP_BSSID)
    reader = NexmonReader(config.NEXMON_PORT, expected_subcarrier_count=config.EXPECT_S,
                          access_point_mac=access_point_mac)
    batch_publisher = BatchPublisher(config.PC_IP, config.UDP_PORT, config.NODE_ID, config.AP_BSSID,
                                     wire_version=config.WIRE_VER)
    # int16 (ver 3) keeps absolute amplitude for weapon; int8 (ver 2) is presence-only.
    encode = quantize_csi_i16 if config.WIRE_VER == 3 else quantize_csi

    print(
        f"[pi5-csi] node={config.NODE_ID} -> {config.PC_IP}:{config.UDP_PORT} | "
        f"AP={config.AP_BSSID} | S={config.EXPECT_S} ver={config.WIRE_VER} | nexmon:{config.NEXMON_PORT}",
        flush=True,
    )

    frames_since_report = 0
    low_rate_streak_s = 0
    last_report_time_s = time.monotonic()
    last_flush_time_s = last_report_time_s
    try:
        for _receive_timestamp_s, _source_mac, csi in reader.frames():
            timestamp_us = time.monotonic_ns() // 1000  # host only needs consistent per-link spacing
            batch_publisher.add(encode(csi, config.CSI_SCALE), timestamp_us)
            frames_since_report += 1

            now_s = time.monotonic()
            # Force flush at ~50ms to bound latency at low rates.
            if now_s - last_flush_time_s >= 0.05:
                batch_publisher.flush()
                last_flush_time_s = now_s
            if now_s - last_report_time_s >= 1.0:
                frame_rate_hz = frames_since_report / (now_s - last_report_time_s)
                is_rate_low = frame_rate_hz < LOW_RATE_HZ
                low_rate_streak_s = low_rate_streak_s + 1 if is_rate_low else 0
                if low_rate_streak_s >= LOW_RATE_HOLD:
                    warning = (f"  <-- SUSTAINED LOW ({low_rate_streak_s}s): illuminator likely down, "
                               "restart illuminate.sh")
                elif is_rate_low:
                    warning = "  <-- LOW: is the Mac illuminating modem B?"
                else:
                    warning = ""
                print(f"[pi5-csi] {frame_rate_hz:.0f} frames/s (S={csi.size}){warning}", flush=True)
                frames_since_report = 0
                last_report_time_s = now_s
    except KeyboardInterrupt:
        pass
    finally:
        batch_publisher.close()
        print("\n[pi5-csi] stopped", flush=True)


if __name__ == "__main__":
    main()
