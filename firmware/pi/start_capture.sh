#!/usr/bin/env bash
# Per-boot: point the Nexmon CSI filter at modem B and bring up monitor mode. Run ON THE PI:
#   bash start_capture.sh
# Reads AP_BSSID + CHANNEL_SPEC from config.py. Run setup_nexmon.sh once first.
set -euo pipefail
cd "$(dirname "$0")"

# Pull the deployment values straight from config.py so there's one source of truth.
read -r BSSID CHAN < <(python3 - <<'PY'
import config
print(config.AP_BSSID, config.CHANNEL_SPEC)
PY
)
if [[ "$BSSID" == TODO_* ]]; then
  echo "ERROR: set AP_BSSID in config.py (modem B 5 GHz BSSID) first."; exit 1
fi

echo "==> CSI params: chan ${CHAN}, BSSID ${BSSID}"
PARAMS="$(mcp -c "${CHAN}" -C 1 -N 1 -m "${BSSID}")"

echo "==> applying filter to wlan0"
sudo ifconfig wlan0 up
sudo nexutil -Iwlan0 -s500 -b -l34 -v"${PARAMS}"

echo "==> monitor interface mon0"
if ! iw dev | grep -q mon0; then
  sudo iw dev wlan0 interface add mon0 type monitor
fi
sudo ip link set mon0 up

echo "==> verify (5 s of CSI frames on udp/5500; Ctrl-C to stop early)"
echo "    if this is silent: the Mac must be illuminating modem B (run illuminate.sh on the Mac)."
sudo timeout 5 tcpdump -i wlan0 dst port 5500 -c 5 || true
echo "==> ready. Now run:  python3 pi5_csi_node.py"
