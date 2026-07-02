"""Project configuration. Explicit dimensions required."""

from dataclasses import dataclass, field


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
    """DSP toggles. Gain lock is optional (only for amplitude path)."""

    gain_lock_enabled: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Recognition head config.

    `stage`: Target (presence/weapon).
    `backend`: 'mlp' (default), 'svm', 'variance' (weapon baseline), 'cnn' (image-based).
    `k`: NBVI subcarrier count (from calibration)."""

    stage: str                      # "presence" | "weapon"
    k: int                          # NBVI subcarrier count -> feature dim = 9*k per node
    backend: str = "mlp"            # "mlp" (default) | "svm" | "variance" (P7) | "cnn" (P7)
    window: int = 128               # front-end window (frames), locked P4
    hop: int = 32                   # front-end hop (frames), locked P4
    fs_tol: float = 0.10            # fsOk: max relative live-fs deviation before a window is dropped
    hidden: int = 32                # MLP hidden width (single layer — tiny head, O(1) forward)
    seed: int = 0                   # backend rng seed (deterministic training/inference)
    frame_average: int = 1          # T2/P10: non-overlapping decimating mean (M=1 = no change)
    subtract_baseline: bool = False  # T3/P10: subtract quiet-room baseline from image path
    subtract_ic_baseline: bool = False  # Item 10/CAUSE 2B: subtract raw baseline from the weapon IC path

    def __post_init__(self) -> None:
        if self.stage not in ("presence", "weapon"):
            raise ValueError("stage must be 'presence' or 'weapon'")
        if self.backend not in ("mlp", "svm", "variance", "cnn"):
            raise ValueError("backend must be one of 'mlp', 'svm', 'variance', 'cnn'")
        if self.k <= 0 or self.window <= 0 or self.hop <= 0 or self.hidden <= 0:
            raise ValueError("k, window, hop and hidden must be positive")
        if not 0.0 < self.fs_tol < 1.0:
            raise ValueError("fs_tol must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class Config:
    """Top-level config. Grows as phases land (DSP, model, output)."""

    capture: CaptureConfig
    signal: SignalConfig = field(default_factory=SignalConfig)
    model: ModelConfig | None = None  # None until a head is configured (Phase 6+)
