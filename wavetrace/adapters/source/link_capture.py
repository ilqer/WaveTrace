"""UDP capture of every (tx->rx) link for the collect scripts: find the live nodes, record one labeled
condition interactively, or record for a fixed time while a camera labels the scene."""

import collections
import socket
import time

from wavetrace.adapters.source.udp import UdpSource, UdpSourceOptions, bind_udp_socket
from wavetrace.adapters.source.wire_format import parse_batch_by_link
from wavetrace.application.ports import LinkCapture, TimedLinkCapture
from wavetrace.domain.link_key import LinkKey

_TIMED_RECEIVE_TIMEOUT_S = 1.0


def detect_node_ids(port, timeout_s=3.0):
    """Sorted ids of the nodes heard in the first 150 frames, or until `timeout_s` of silence."""
    print("listening for active nodes...")
    source = UdpSource(UdpSourceOptions(port=port, timeout_seconds=timeout_s, max_frames=150))
    return sorted({frame.node_id for frame in source.frames()})


class UdpLinkCapture(LinkCapture):
    """Binds 0.0.0.0:port per capture. A capture ends once every link of `node_ids` holds
    `max_frames_per_link`, after 15 s of total silence, or at the `max_capture_s` deadline."""

    def __init__(self, port, node_ids, *, max_frames_per_link, max_capture_s=60.0):
        self._port = port
        self._node_ids = node_ids
        self._max_frames_per_link = max_frames_per_link
        self._max_capture_s = max_capture_s

    def capture(self, prompt, *, countdown_s, hint):
        """The wall-clock deadline is needed because the receive timeout only fires on total silence,
        so one quiet link would stall the loop."""
        max_frames_per_link, max_capture_s = self._max_frames_per_link, self._max_capture_s
        print(f"\n>> {prompt}\n   press Enter to start...", flush=True)
        input()
        if countdown_s:
            for seconds_left in range(countdown_s, 0, -1):
                print(f"   starting in {seconds_left}s...", end="\r")
                time.sleep(1)
            print()
        print(f"   [CAPTURING] {hint}")

        links: collections.defaultdict[LinkKey, list] = collections.defaultdict(list)
        wanted_node_ids = set(self._node_ids)
        udp_socket = bind_udp_socket(self._port, timeout_s=15.0)
        start_time_s = time.time()
        last_print_time_s = start_time_s
        try:
            while True:
                frames_by_link = _receive_frames_by_link(udp_socket)
                if frames_by_link is None:
                    break
                for key, frames in frames_by_link.items():
                    link_frames = links[key]
                    if len(link_frames) < max_frames_per_link:
                        link_frames.extend(frames[: max_frames_per_link - len(link_frames)])
                now_s = time.time()
                if (links and wanted_node_ids <= {link_key[1] for link_key in links}
                        and all(len(queued_frames) >= max_frames_per_link
                                for queued_frames in links.values())):
                    break
                if now_s - start_time_s > max_capture_s:
                    short_links = [link_key for link_key, captured_frames in sorted(links.items())
                                   if len(captured_frames) < max_frames_per_link]
                    print(f"\n[WARN] deadline {max_capture_s:g}s hit; "
                          f"links short of {max_frames_per_link}: {short_links}")
                    break
                if now_s - last_print_time_s >= 1.0:
                    link_count_by_node = dict(sorted(
                        collections.Counter(link_key[1] for link_key in links).items()))
                    min_frame_count = min((len(captured_frames) for captured_frames in links.values()),
                                          default=0)
                    print(f"   links/node {link_count_by_node}  "
                          f"min {min_frame_count}/{max_frames_per_link}...", end="\r")
                    last_print_time_s = now_s
        finally:
            udp_socket.close()
        print()
        _keep_dominant_width(links)
        print('\a', end='', flush=True)
        return dict(links)


class UdpTimedLinkCapture(TimedLinkCapture):
    """Binds 0.0.0.0:port per capture and keeps only the links received at one of `node_ids`."""

    def __init__(self, port, node_ids):
        self._port = port
        self._node_ids = frozenset(node_ids)

    def capture_for(self, duration_s, *, stop):
        """Ends at the `duration_s` deadline, or within one 1 s receive timeout of `stop`. A bind error
        propagates as is."""
        links: collections.defaultdict[LinkKey, list] = collections.defaultdict(list)
        udp_socket = bind_udp_socket(self._port, timeout_s=_TIMED_RECEIVE_TIMEOUT_S)
        end_time_s = time.monotonic() + duration_s
        try:
            while time.monotonic() < end_time_s and not stop.is_set():
                for key, frames in (_receive_frames_by_link(udp_socket) or {}).items():
                    if key[1] in self._node_ids:
                        links[key].extend(frames)
        finally:
            udp_socket.close()
        _keep_dominant_width(links)
        return dict(links)


def _receive_frames_by_link(udp_socket) -> dict | None:
    """One datagram's CsiFrames per link; None when the receive timed out."""
    try:
        payload, _ = udp_socket.recvfrom(65535)
    except socket.timeout:
        return None
    return parse_batch_by_link(payload)


def _keep_dominant_width(links) -> None:
    """In place: each link keeps only its frames of its most common subcarrier width, since widths can
    differ across boards and bands and one link must stack into one array. O(frames)."""
    for key in links:
        if links[key]:
            dominant_num_subcarriers = collections.Counter(
                frame.num_subcarriers for frame in links[key]).most_common(1)[0][0]
            links[key] = [frame for frame in links[key] if frame.num_subcarriers == dominant_num_subcarriers]
