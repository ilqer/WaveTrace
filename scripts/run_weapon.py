"""Independent live WEAPON detection — every (tx->rx) link served through its RX node's cal + weapon
head (inter-carrier features), fused into one armed/clear verdict. Standalone from run_live_mesh /
run_count (imports only library code), reads the weapon models from data/model_weapon.

Each link's RX-node WeaponHead emits P(weapon); LinkVoter blends them weighted by static reliability
(per-node LOGO accuracy via accuracyWeights) x live decision margin. A node validated at/below chance
gets weight 0 and drops out.

    .venv/bin/python scripts/run_weapon.py
"""

import argparse
import collections
import socket
import time

from wavetrace.Source import parseBatchLinks, bindUdp
from wavetrace.recognition import LinkVoter, dwellProbaDetailed, linkHealth, loadWeaponLinks


def _entryFor(entries, buf_key):
    """Match a live buffer key (tx_short, rx_node) to a serving entry: the per-link (tag, nid) head
    first (tx_short '4f:9c' -> tag '4f9c'), then the per-node fallback (None, nid). None if neither."""
    txShort, nid = buf_key
    return entries.get((txShort.replace(":", ""), nid)) or entries.get((None, nid))


def main():
    parser = argparse.ArgumentParser(description="Live ALL-PAIRS weapon detection (per-link, per-RX-node cal+head).")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Weapon model root (default: <root>/model_weapon)")
    args = parser.parse_args()
    if args.cal is None:
        args.cal = f"{args.root}/cal"
    if args.model is None:
        args.model = f"{args.root}/model_weapon"

    CHUNK_S = 1.5          # fuse + print at this cadence
    LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
    BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing

    entries = loadWeaponLinks(args.cal, args.model)
    if not entries:
        print(f"[ERROR] no weapon heads under {args.model}/node*/[link*/]model.joblib with a matching "
              f"{args.cal}/node*/. Run collect_baseline.py then collect_weapon.py first.")
        return
    weaponI = next(iter(entries.values()))["weapon_i"]  # ordering validated equal in loadWeaponLinks
    # the artifact's own resample rate, not a re-declared constant -- every head is trained at the
    # same rate today, so any entry's contract stands in for the print banner.
    sample_rate_hz = next(iter(entries.values()))["session"].head.contract.target_sample_rate_hz

    buffers = collections.defaultdict(collections.deque)  # keyed by (tx_short, rx_node)
    lastSeen = {}
    linkIds = {}

    sock = bindUdp(args.port, timeout=0.5)
    perLink = any(tag is not None for tag, _ in entries)
    def _wlabel(key):
        tag, nid = key
        return f"{tag}->{nid}" if tag is not None else f"*->{nid}"
    wsummary = "  ".join(f"{_wlabel(k)}:w={entries[k]['weight']:.2f}" for k in sorted(entries))
    print(f"WEAPON detection on udp/{args.port} (fs={sample_rate_hz:g}Hz, "
          f"{'per-link' if perLink else 'per-node'} heads; vote weights {wsummary}). Ctrl+C to stop.\n")

    nextFuse = time.time() + CHUNK_S
    try:
        while True:
            now = time.time()
            try:
                payload, _ = sock.recvfrom(65535)
                for key, frames in parseBatchLinks(payload).items():
                    m = _entryFor(entries, key)  # key=(tx_short, rx_node) -> per-link, then per-node
                    if m is not None and frames[0].num_subcarriers >= m["min_width"]:
                        buffers[key].extend(frames)
                        lastSeen[key] = now
                        linkIds.setdefault(key, len(linkIds))
            except socket.timeout:
                pass

            if now < nextFuse:
                continue
            nextFuse = now + CHUNK_S

            for buf in buffers.values():
                if buf:
                    cutoff = buf[-1].timestamp - BUFFER_S
                    while buf and buf[0].timestamp < cutoff:
                        buf.popleft()

            # static per-link reliability x live margin (LinkVoter multiplies them); uniform fallback.
            linkStatic = {lid: _entryFor(entries, key)["weight"] for key, lid in linkIds.items()}
            static = linkStatic if any(w > 0 for w in linkStatic.values()) else None
            voter = LinkVoter(static)
            breakdown = []
            for key in sorted(buffers):
                if now - lastSeen.get(key, 0) > LINK_TIMEOUT_S or len(buffers[key]) < 2:
                    continue
                m = _entryFor(entries, key)
                proba, *_ = dwellProbaDetailed(
                    list(buffers[key]), m["session"].head.contract.target_sample_rate_hz, m
                )
                if proba is None:
                    continue
                wi = m["weapon_i"]
                pWeapon = float(proba[wi]) if wi >= 0 else 0.0
                quality = abs(pWeapon - 0.5) * 2.0  # decision margin -> 0 (unsure) .. 1 (confident)
                voter.add(linkIds[key], proba, quality=quality)
                hz, miss = linkHealth(buffers[key])  # delivered rate + missing-frame fraction (C9b)
                tail = f"@{hz:.0f}Hz" + (f"!{miss:.0%}drop" if miss > 0.1 else "")
                breakdown.append(f"{key[0]}->{key[1]}:{pWeapon:.2f}{tail}")

            if not breakdown:
                print("\r(no live links with a full window yet)            ", end="", flush=True)
                continue
            try:
                _cls, blended = voter.finalize()
            except ValueError:
                print("\r(live links present, but all from chance-level nodes)   ", end="", flush=True)
                continue
            pWeapon = float(blended[weaponI]) if weaponI >= 0 else 0.0
            label = "WEAPON" if pWeapon >= 0.5 else "clear "
            bar = "#" * int(pWeapon * 20)
            print(f"{label}  P {pWeapon:0.2f}  {bar:<20}  [{len(breakdown)} links] "
                  + " ".join(breakdown))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
