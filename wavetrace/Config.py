"""Domain configuration for CSI capture and recognition: the shape of a capture geometry, a DSP
toggle set, and a recognition head's hyperparameters. Explicit dimensions required.

Transport/adapter settings (`SourceOptions` and its subclasses) live in `wavetrace/Source.py`;
delivery settings (`WebServerOptions`) live in `web/`."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Capture geometry."""

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
    """DSP toggles. The gain lock applies to the amplitude path only."""

    gain_lock_enabled: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Recognition head hyperparameters.

    `backend` accepts whatever `wavetrace.adapters.recognition` registers; an unregistered name is
    rejected when the head is built or loaded, not here."""

    stage: str                      # "presence" | "weapon"
    k: int                          # NBVI subcarrier count -> feature dim = 9*k per node
    backend: str = "mlp"
    window: int = 128               # front-end window, in frames
    hop: int = 32                   # front-end hop, in frames
    fs_tol: float = 0.10            # fsOk: max relative live-fs deviation before a window is dropped
    hidden: int = 32                # MLP hidden width (single layer — tiny head, O(1) forward)
    seed: int = 0                   # backend rng seed (deterministic training/inference)
    frame_average: int = 1          # non-overlapping decimating mean; 1 = no averaging
    subtract_baseline: bool = False  # subtract the quiet-room baseline from the image path
    subtract_ic_baseline: bool = False  # subtract the raw quiet-room baseline from the weapon IC path

    def __post_init__(self) -> None:
        if self.stage not in ("presence", "weapon"):
            raise ValueError("stage must be 'presence' or 'weapon'")
        if self.k <= 0 or self.window <= 0 or self.hop <= 0 or self.hidden <= 0:
            raise ValueError("k, window, hop and hidden must be positive")
        if not 0.0 < self.fs_tol < 1.0:
            raise ValueError("fs_tol must be in (0, 1)")
