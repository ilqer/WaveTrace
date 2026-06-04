"""Type stubs for the native `_wavetrace` extension (Phase 1 core types)."""

from typing import Optional

import numpy as np
import numpy.typing as npt

class WaveTraceError(Exception): ...
class FrameError(WaveTraceError): ...

class CsiFrame:
    def __init__(self, num_antennas: int, num_subcarriers: int) -> None: ...
    @property
    def num_antennas(self) -> int: ...
    @property
    def num_subcarriers(self) -> int: ...
    @property
    def size(self) -> int: ...
    timestamp: float
    node_id: int
    def reshape(self, num_antennas: int, num_subcarriers: int) -> None: ...
    @property
    def grid(self) -> npt.NDArray[np.complex64]:
        """Zero-copy writable view, shape (num_antennas, num_subcarriers)."""
        ...

class FrameParser:
    def __init__(self, num_antennas: int, num_subcarriers: int) -> None: ...
    @property
    def num_antennas(self) -> int: ...
    @property
    def num_subcarriers(self) -> int: ...
    def parse(
        self, raw: npt.NDArray[np.uint8], timestamp: float = 0.0, node_id: int = -1
    ) -> CsiFrame:
        """Decode one raw int8 [imag,real] frame into the reused CsiFrame (same object each call)."""
        ...

class NodeAggregator:
    def __init__(self) -> None: ...
    def submit(self, frame: CsiFrame) -> None: ...
    @property
    def num_nodes(self) -> int: ...
    def synced(self, tolerance: float) -> list[CsiFrame]: ...

class RecognitionResult:
    def __init__(self) -> None: ...
    class_id: int
    confidence: float
    timestamp: float
    bbox: Optional[list[float]]  # 4 elements (x, y, w, h); set from any 4-sequence
    keypoints: list[float]

class Label:
    def __init__(self) -> None: ...
    class_id: int
    name: str
    timestamp: float
    bbox: Optional[list[float]]  # 4 elements (x, y, w, h); set from any 4-sequence
    keypoints: list[float]
