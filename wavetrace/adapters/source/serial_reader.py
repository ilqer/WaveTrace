"""esp-csi CSI over USB serial. Needs `pyserial`."""

from dataclasses import dataclass
import time

from wavetrace import CsiFrame
from wavetrace.adapters.source.options import SourceOptions
from wavetrace.adapters.source.wire_format import parse_csi_line
from wavetrace.application.ports import CsiSource


@dataclass(frozen=True, slots=True, kw_only=True)
class SerialSourceOptions(SourceOptions):
    device: str                       # e.g. "/dev/tty.usbserial-…"
    baud: int = 921600
    node_id: int = 1
    tx_mac: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.device:
            raise ValueError("device must be non-empty")
        if self.baud <= 0:
            raise ValueError("baud must be positive")
        if self.node_id < 0:
            raise ValueError("node_id must be >= 0")


class SerialReader(CsiSource):
    """One ESP32 = one node = one antenna.

    Frame timestamp = PC wall-clock at read, NOT the ESP's `local_timestamp` counter: independent ESP
    clocks have no common origin, so the PC arrival time is the consistent basis for aligning against
    camera labels and other nodes (sub-ms serial latency ≪ the front-end window). `tx_mac` filters to
    the dedicated TX (drops beacons / foreign traffic). The pyserial import and port open are deferred
    to frames() so importing this module never requires pyserial."""

    def __init__(self, options: SerialSourceOptions):
        self._port = options.device
        self._baud = options.baud
        self._node_id = options.node_id
        self._tx_mac = options.tx_mac
        self._timeout_seconds = options.timeout_seconds
        self._max_frames = options.max_frames

    def frames(self):
        """Stops on read timeout or max_frames. Malformed/filtered lines are skipped silently.

        Subcarrier-count guard (same as parse_batch): even from the dedicated TX, an RX occasionally
        receives off-format frames (legacy/HT20/HT40 differ in S). The first yielded frame sets the
        expected S and any line with a different S is dropped, so downstream shape stays (1, S)."""
        try:
            import serial  # pyserial
        except ImportError as error:
            raise ImportError("SerialReader needs pyserial: pip install pyserial") from error
        serial_port = serial.Serial(self._port, self._baud, timeout=self._timeout_seconds)
        try:
            count = 0
            expected_num_subcarriers = None
            while self._max_frames is None or count < self._max_frames:
                raw_line = serial_port.readline()
                if not raw_line:
                    return
                result = parse_csi_line(raw_line.decode("utf-8", errors="replace"), tx_mac=self._tx_mac)
                if result is None:
                    continue
                csi, _local_timestamp_us, _mac = result
                num_subcarriers = int(csi.size)
                if expected_num_subcarriers is None:
                    expected_num_subcarriers = num_subcarriers
                elif num_subcarriers != expected_num_subcarriers:
                    continue  # off-format frame (different bandwidth/mode) -> drop to keep fixed shape
                frame = CsiFrame(1, expected_num_subcarriers)
                frame.grid[0, :] = csi
                frame.timestamp = time.time()
                frame.node_id = self._node_id
                yield frame
                count += 1
        finally:
            serial_port.close()
