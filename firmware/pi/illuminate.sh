#!/usr/bin/env bash
# Run ON THE MAC (not the Pi). The Mac must be Wi-Fi associated to modem B's 5 GHz network.
# An idle AP only beacons (~10 Hz); this generates traffic so modem B transmits ~300 frames/s,
# which the Pi sniffs as CSI. Stop with Ctrl-C.
#
#   bash illuminate.sh <modem_B_ip>
# macOS needs root for sub-0.1 s ping intervals, hence sudo.
set -euo pipefail

IP="${1:-}"
if [ -z "$IP" ]; then echo "usage: bash illuminate.sh <modem_B_ip>   (modem B's 5 GHz LAN IP)"; exit 1; fi

echo "==> illuminating $IP at ~300 Hz (Ctrl-C to stop). Keep this running during capture."
echo "    alternative if ping is rate-limited:  iperf3 -u -b 5M -c $IP"
exec sudo ping -i 0.003 "$IP"
