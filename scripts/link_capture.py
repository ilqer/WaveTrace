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
    for frame in source.frames():
        detected[frame.node_id] += 1
    return sorted(detected.keys())


def capture_links(prompt, max_frames_per_link, port, node_ids, countdown=0, max_capture_s=60.0,
                  capture_hint="hold the condition steady..."):
    """Returns {(tx_short, rx_node): [frames]}, each link reduced to its dominant subcarrier width;
    splitting per directed link matches how the run_* scripts serve it (train/serve parity).
    The wall-clock deadline is needed because the receive timeout only fires on total silence, so
    one quiet link would stall the loop."""
    print(f"\n>> {prompt}\n   press Enter to start...", flush=True)
    input()
    if countdown:
        for seconds_left in range(countdown, 0, -1):
            print(f"   starting in {seconds_left}s...", end="\r")
            time.sleep(1)
        print()
    print(f"   [CAPTURING] {capture_hint}")

    links = collections.defaultdict(list)  # (tx_short, rx_node) -> [frames]
    wanted_node_ids = set(node_ids)
    sock = bind_udp_socket(port, timeout_s=15.0)
    start_time_s = time.time()
    last_print_time_s = start_time_s
    try:
        while True:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                break  # total silence
            for key, frames in parse_batch_by_link(payload).items():
                link_frames = links[key]
                if len(link_frames) < max_frames_per_link:
                    link_frames.extend(frames[: max_frames_per_link - len(link_frames)])
            now = time.time()
            if (links and wanted_node_ids <= {link_key[1] for link_key in links}
                    and all(len(queued_frames) >= max_frames_per_link for queued_frames in links.values())):
                break
            if now - start_time_s > max_capture_s:
                short_links = [link_key for link_key, captured_frames in sorted(links.items())
                               if len(captured_frames) < max_frames_per_link]
                print(f"\n[WARN] deadline {max_capture_s:g}s hit; "
                      f"links short of {max_frames_per_link}: {short_links}")
                break
            if now - last_print_time_s >= 1.0:
                link_count_by_node = dict(sorted(
                    collections.Counter(link_key[1] for link_key in links).items()))
                min_frame_count = min((len(captured_frames) for captured_frames in links.values()), default=0)
                print(f"   links/node {link_count_by_node}  min {min_frame_count}/{max_frames_per_link}...",
                      end="\r")
                last_print_time_s = now
    finally:
        sock.close()
    print()
    for key in links:  # dominant subcarrier width per link (widths can differ across boards/bands)
        if links[key]:
            subcarrier_count = collections.Counter(
                frame.num_subcarriers for frame in links[key]).most_common(1)[0][0]
            links[key] = [frame for frame in links[key] if frame.num_subcarriers == subcarrier_count]
    print('\a', end='', flush=True)
    return dict(links)
