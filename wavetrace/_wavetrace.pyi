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
