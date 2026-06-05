"""WaveTrace: WiFi-CSI sensing pipeline (posture & weapon recognition, CSI-only at deploy).

Phase 1 re-exports the native core types from the `_wavetrace` extension so callers use
`from wavetrace import CsiFrame, RecognitionResult, Label`.
"""

from ._wavetrace import (
    CsiFrame,
    FrameParser,
    NodeAggregator,
    Preprocessor,
    GainLock,
    conjugate_multiply,
    hampel,
    unwrap_step,
    coefficient_of_variation,
    nbvi_scores,
    select_subcarriers_nbvi,
    fft,
    nine_features,
    power_spectrum,
    doppler_features,
    FeatureExtractor,
    SpectrogramBuilder,
    Label,
    RecognitionResult,
    WaveTraceError,
    FrameError,
)

__all__ = [
    "CsiFrame",
    "FrameParser",
    "NodeAggregator",
    "Preprocessor",
    "GainLock",
    "conjugate_multiply",
    "hampel",
    "unwrap_step",
    "coefficient_of_variation",
    "nbvi_scores",
    "select_subcarriers_nbvi",
    "fft",
    "nine_features",
    "power_spectrum",
    "doppler_features",
    "FeatureExtractor",
    "SpectrogramBuilder",
    "Label",
    "RecognitionResult",
    "WaveTraceError",
    "FrameError",
]

__version__ = "0.1.0"
