"""Project configuration. Phase 1 defines only capture geometry; later phases extend it.

No fabricated defaults: dimensions depend on the (still-undecided) sensor hardware and link
count (plan.md §3), so every field is required and supplied explicitly by the caller.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Geometry of one capture session, shared by the fixture and (later) the live reader."""

    num_antennas: int
    num_subcarriers: int
    sample_rate_hz: float

    def __post_init__(self) -> None:
        if self.num_antennas <= 0 or self.num_subcarriers <= 0:
            raise ValueError("num_antennas and num_subcarriers must be positive")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")


@dataclass(frozen=True, slots=True)
class SignalConfig:
    """DSP front-end toggles. A behavioral switch (not a hardware dimension), so a default is fine.

    gain_lock_enabled gates the amplitude-path AGC surrogate (GainLock). It is OPTIONAL: the phase
    path (Preprocessor) is scale-invariant and never needs it, and the material features (σ²[p],
    reflection_signature) must NOT see gain-locked frames — GainLock is per-frame mean normalization,
    which erases the bulk attenuation those features measure. Enable it only for the amplitude /
    presence feature path.
    """

    gain_lock_enabled: bool = True


@dataclass(frozen=True, slots=True)
class Config:
    """Top-level config. Grows as phases land (DSP, model, output)."""

    capture: CaptureConfig
    signal: SignalConfig = field(default_factory=SignalConfig)
