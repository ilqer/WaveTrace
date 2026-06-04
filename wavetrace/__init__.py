"""WaveTrace: WiFi-CSI sensing pipeline (posture & weapon recognition, CSI-only at deploy).

Phase 1 re-exports the native core types from the `_wavetrace` extension so callers use
`from wavetrace import CsiFrame, RecognitionResult, Label`.
"""

from ._wavetrace import (
    CsiFrame,
    Label,
    RecognitionResult,
    WaveTraceError,
    FrameError,
)

__all__ = [
    "CsiFrame",
    "Label",
    "RecognitionResult",
    "WaveTraceError",
    "FrameError",
]

__version__ = "0.1.0"
