"""5 GHz CSI from a Raspberry Pi running nexmon_csi. Needs `csiread`."""

from dataclasses import dataclass

import numpy as np

from wavetrace import CsiFrame
from wavetrace.adapters.source.options import SourceOptions
from wavetrace.application.ports import CsiSource


@dataclass(frozen=True, slots=True, kw_only=True)
class NexmonSourceOptions(SourceOptions):
    """Exactly one capture mode must be selected: a `pcap_path` to replay, or `live=True` to spawn
    tcpdump on `iface`."""

    pcap_path: str | None = None
    iface: str = "wlan0"
    live: bool = False
    node_id: int = 100                # >=100 so it never collides with ESP32 ids 1..6
    bandwidth_mhz: int = 80

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.node_id < 0:
            raise ValueError("node_id must be >= 0")
        if self.bandwidth_mhz <= 0:
            raise ValueError("bandwidth_mhz must be positive")
        if not self.live and not self.pcap_path:
            raise ValueError("NexmonSource: set pcap_path or live=True")


class NexmonSource(CsiSource):
    """Decodes bcm43455c0 nexmon_csi (Pi 3B+/4B/5, fw 7_45_189). `csiread.Nexmon` does the decoding,
    so only `_decode_file` changes for another csiread version or chip.

    node_id defaults to 100: ESP32 nodes are 1..6 and everything downstream reads node_id >= 100
    as the 5 GHz band."""

    def __init__(self, options: NexmonSourceOptions):
        self._pcap_path = options.pcap_path
        self._interface = options.iface
        self._is_live = options.live
        self._node_id = options.node_id
        self._timeout_seconds = options.timeout_seconds
        self._max_frames = options.max_frames
        self._bandwidth_mhz = options.bandwidth_mhz

    def _csiread(self):
        try:
            import csiread
            return csiread
        except ImportError as error:
            raise ImportError("NexmonSource needs csiread: pip install csiread") from error

    def _decode_file(self):
        csiread = self._csiread()
        reader = csiread.Nexmon(self._pcap_path, chip="43455c0", bw=self._bandwidth_mhz)
        reader.read()
        csi = np.asarray(reader.csi)           # (F, S) complex
        raw_timestamps_s = getattr(reader, "sec", None)
        timestamps_s = (np.asarray(raw_timestamps_s, dtype=float)
                        if raw_timestamps_s is not None else np.arange(len(csi)) / 100.0)
        # csiread's `sec` holds whole seconds; `usec` carries the fraction
        microseconds = getattr(reader, "usec", None)
        if raw_timestamps_s is not None and microseconds is not None:
            timestamps_s = timestamps_s + np.asarray(microseconds, dtype=float) * 1e-6
        for i in range(len(csi)):
            frame = CsiFrame(1, csi.shape[1])
            frame.grid[0, :] = csi[i].astype(np.complex64)
            frame.timestamp = float(timestamps_s[i]) if i < timestamps_s.size else i / 100.0
            frame.node_id = self._node_id
            yield frame

    def _decode_live(self):
        import subprocess, tempfile, os, time as _time
        pcap_file_descriptor, live_pcap_path = tempfile.mkstemp(suffix=".pcap")
        os.close(pcap_file_descriptor)
        try:
            # leaving the `with` waits for the terminated tcpdump, so no zombie outlives the generator
            with subprocess.Popen(
                ["tcpdump", "-i", self._interface, "-w", live_pcap_path, "dst port 5500"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ) as tcpdump_process:
                count = 0
                try:
                    while self._max_frames is None or count < self._max_frames:
                        _time.sleep(0.3)
                        if os.path.getsize(live_pcap_path) < 64:
                            continue
                        self._pcap_path = live_pcap_path
                        for frame in self._decode_file():
                            yield frame
                            count += 1
                            if self._max_frames is not None and count >= self._max_frames:
                                return
                        open(live_pcap_path, "wb").close()    # truncate consumed chunk
                finally:
                    tcpdump_process.terminate()
        finally:
            try:
                os.unlink(live_pcap_path)
            except OSError:
                pass

    def frames(self):
        if self._is_live:
            yield from self._decode_live()
        else:
            yield from self._decode_file()
