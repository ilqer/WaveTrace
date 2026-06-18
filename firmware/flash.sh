#!/usr/bin/env bash
# Flash one WaveTrace board. ESP-IDF v5.x must be sourced first (`. $HOME/esp/esp-idf/export.sh`).
#
#   NODE: ./flash.sh node <NODE_ID> <PORT> e.g. ./flash.sh node 2 /dev/cu.usbmodem1101  (full mesh)
#   RX:  ./flash.sh rx <NODE_ID> <PORT>   e.g.  ./flash.sh rx 3 /dev/cu.usbmodem1101    (legacy 1TX/6RX)
#   TX:  ./flash.sh tx <PORT>             e.g.  ./flash.sh tx /dev/cu.usbmodem1101       (legacy 1TX/6RX)
#
# NODE_ID is the ONLY per-board difference; it is injected as a compile definition.
# node = unified full-mesh firmware (every board identical). Edit PC_IP / ROUTER_* once in the
# matching main/config.h before the first flash (esp32_node for node, esp32_rx for rx).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROLE="${1:-}"

case "$ROLE" in
  node)
    NODE_ID="${2:?need NODE_ID 1..MESH_NODES}"; PORT="${3:?need serial port}"
    cd "$HERE/esp32_node"
    [ -f sdkconfig ] || idf.py set-target esp32s3
    # -D cache var (not env) so changing the id reconfigures + recompiles for each board
    idf.py -p "$PORT" -DNODE_ID="$NODE_ID" build flash monitor
    ;;
  rx)
    NODE_ID="${2:?need NODE_ID 1..6}"; PORT="${3:?need serial port}"
    cd "$HERE/esp32_rx"
    [ -f sdkconfig ] || idf.py set-target esp32s3
    NODE_ID="$NODE_ID" idf.py -p "$PORT" build flash monitor
    ;;
  tx)
    PORT="${2:?need serial port}"
    cd "$HERE/esp32_tx"
    [ -f sdkconfig ] || idf.py set-target esp32s3
    idf.py -p "$PORT" build flash monitor
    ;;
  *)
    echo "usage: flash.sh node <NODE_ID> <PORT> | rx <NODE_ID> <PORT> | tx <PORT>" >&2; exit 1 ;;
esac
