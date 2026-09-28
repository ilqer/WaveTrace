"""Live weapon detection: every (tx->rx) link is served through its rx node's cal and weapon head
(inter-carrier features), and the links are fused into one armed/clear verdict.

LinkVoter weights each link by static reliability (LOGO accuracy) times live decision margin, so a
head validated at or below chance gets weight 0 and drops out.
"""

import argparse
import collections
import socket
import time

from wavetrace.adapters.source import parse_batch_by_link, bind_udp_socket
from wavetrace.recognition import LinkVoter, dwell_proba, find_weapon_entry, link_health, load_weapon_links


CHUNK_S = 1.5          # fuse + print at this cadence
LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing


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

    entries = load_weapon_links(args.cal, args.model)
    if not entries:
        print(f"[ERROR] no weapon heads under {args.model}/node*/[link*/]model.joblib with a matching "
              f"{args.cal}/node*/. Run collect_baseline.py then collect_weapon.py first.")
        return
    # ordering validated equal in load_weapon_links
    weapon_class_index = next(iter(entries.values()))["weapon_i"]
    # read off a head's own contract rather than re-declared here; every head is trained at the same
    # rate today, so any one of them stands in for the banner
    sample_rate_hz = next(iter(entries.values()))["session"].head.contract.target_sample_rate_hz

    buffers = collections.defaultdict(collections.deque)  # keyed by (tx_short, rx_node)
    last_seen_time_s = {}
    link_ids = {}

    sock = bind_udp_socket(args.port, timeout_s=0.5)
    has_per_link_heads = any(tag is not None for tag, _ in entries)
    def _wlabel(key):
        tag, nid = key
        return f"{tag}->{nid}" if tag is not None else f"*->{nid}"
    wsummary = "  ".join(f"{_wlabel(k)}:w={entries[k]['weight']:.2f}" for k in sorted(entries))
    print(f"WEAPON detection on udp/{args.port} (fs={sample_rate_hz:g}Hz, "
          f"{'per-link' if has_per_link_heads else 'per-node'} heads; vote weights {wsummary}). "
          "Ctrl+C to stop.\n")

    next_fuse_time_s = time.time() + CHUNK_S
    try:
        while True:
            now = time.time()
            try:
                payload, _ = sock.recvfrom(65535)
                for key, frames in parse_batch_by_link(payload).items():
                    m = find_weapon_entry(entries, *key)  # key=(tx_short, rx_node) -> per-link, then per-node
                    if m is not None and frames[0].num_subcarriers >= m["min_width"]:
                        buffers[key].extend(frames)
                        last_seen_time_s[key] = now
                        link_ids.setdefault(key, len(link_ids))
            except socket.timeout:
                pass

            if now < next_fuse_time_s:
                continue
            next_fuse_time_s = now + CHUNK_S

            for buf in buffers.values():
                if buf:
                    cutoff = buf[-1].timestamp - BUFFER_S
                    while buf and buf[0].timestamp < cutoff:
                        buf.popleft()

            # LinkVoter multiplies static reliability by live margin; uniform when nothing is weighted
            static_link_weights = {lid: find_weapon_entry(entries, *key)["weight"] for key, lid in link_ids.items()}
            static = static_link_weights if any(w > 0 for w in static_link_weights.values()) else None
            voter = LinkVoter(static)
            breakdown = []
            for key in sorted(buffers):
                if now - last_seen_time_s.get(key, 0) > LINK_TIMEOUT_S or len(buffers[key]) < 2:
                    continue
                m = find_weapon_entry(entries, *key)
                proba, *_ = dwell_proba(
                    list(buffers[key]), m["session"].head.contract.target_sample_rate_hz, m
                )
                if proba is None:
                    continue
                wi = m["weapon_i"]
                weapon_proba = float(proba[wi]) if wi >= 0 else 0.0
                quality = abs(weapon_proba - 0.5) * 2.0  # decision margin -> 0 (unsure) .. 1 (confident)
                voter.add(link_ids[key], proba, quality=quality)
                hz, miss = link_health(buffers[key])  # delivered rate and missing-frame fraction
                tail = f"@{hz:.0f}Hz" + (f"!{miss:.0%}drop" if miss > 0.1 else "")
                breakdown.append(f"{key[0]}->{key[1]}:{weapon_proba:.2f}{tail}")

            if not breakdown:
                print("\r(no live links with a full window yet)            ", end="", flush=True)
                continue
            try:
                _cls, blended = voter.finalize()
            except ValueError:
                print("\r(live links present, but all from chance-level nodes)   ", end="", flush=True)
                continue
            weapon_proba = float(blended[weapon_class_index]) if weapon_class_index >= 0 else 0.0
            label = "WEAPON" if weapon_proba >= 0.5 else "clear "
            bar = "#" * int(weapon_proba * 20)
            print(f"{label}  P {weapon_proba:0.2f}  {bar:<20}  [{len(breakdown)} links] "
                  + " ".join(breakdown))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
