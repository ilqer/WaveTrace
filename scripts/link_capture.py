"""Interactive UDP capture shared by the collect_* scripts: find the live nodes, then record every
(tx->rx) link for one labeled condition."""

import collections
import socket
import time

from wavetrace.adapters.source import UdpSource, UdpSourceOptions, bind_udp_socket, parse_batch_by_link


def detect_nodes(port, timeout_seconds=3.0):
    """Node ids heard in one short listen, sorted."""
    print("listening for active nodes...")
    detected = collections.Counter()
    source = UdpSource(UdpSourceOptions(port=port, timeout_seconds=timeout_seconds, max_frames=150))
    for fr in source.frames():
        detected[fr.node_id] += 1
    return sorted(detected.keys())


def capture_links(prompt, max_frames_per_link, port, node_ids, countdown=0, max_capture_s=60.0,
                  capture_hint="hold the condition steady..."):
    """Collect up to n frames per (tx->rx) link in one pass. Returns {(tx_short, rx_node): [frames]},
    each link reduced to its dominant subcarrier width. Splitting per link keeps every directed link a
    single clean stream, which is how the run_* scripts serve it. The wall-clock deadline is needed
    because the receive timeout only fires on total silence, so one quiet link would stall the loop."""
    print(f"\n>> {prompt}\n   press Enter to start...", flush=True)
    input()
    if countdown:
        for d in range(countdown, 0, -1):
            print(f"   starting in {d}s...", end="\r")
            time.sleep(1)
        print()
    print(f"   [CAPTURING] {capture_hint}")

    links = collections.defaultdict(list)  # (tx_short, rx_node) -> [frames]
    want = set(node_ids)
    sock = bind_udp_socket(port, timeout_s=15.0)
    start = time.time()
    last_print_time_s = start
    try:
        while True:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                break  # total silence
            for key, frames in parse_batch_by_link(payload).items():
                buf = links[key]
                if len(buf) < max_frames_per_link:
                    buf.extend(frames[: max_frames_per_link - len(buf)])
            now = time.time()
            if links and want <= {k[1] for k in links} and all(len(b) >= max_frames_per_link for b in links.values()):
                break
            if now - start > max_capture_s:
                short = [k for k, v in sorted(links.items()) if len(v) < max_frames_per_link]
                print(f"\n[WARN] deadline {max_capture_s:g}s hit; links short of {max_frames_per_link}: {short}")
                break
            if now - last_print_time_s >= 1.0:
                link_count_by_node = dict(sorted(collections.Counter(k[1] for k in links).items()))
                mn = min((len(v) for v in links.values()), default=0)
                print(f"   links/node {link_count_by_node}  min {mn}/{max_frames_per_link}...", end="\r")
                last_print_time_s = now
    finally:
        sock.close()
    print()
    for key in links:  # dominant subcarrier width per link (widths can differ across boards/bands)
        if links[key]:
            S = collections.Counter(f.num_subcarriers for f in links[key]).most_common(1)[0][0]
            links[key] = [f for f in links[key] if f.num_subcarriers == S]
    print('\a', end='', flush=True)
    return dict(links)
