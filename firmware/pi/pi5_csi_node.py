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
import time

import config
from nexmon_reader import NexmonReader
from publisher import BatchPublisher, mac_to_bytes, quantize_csi, quantize_csi_i16

LOW_RATE_HZ = 50.0  # warn below this: usually the Mac stopped illuminating modem B (AP idle ~10 Hz)


def main() -> None:
    config.validate()
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
                warn = "  <-- LOW: is the Mac illuminating modem B?" if rate < LOW_RATE_HZ else ""
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
