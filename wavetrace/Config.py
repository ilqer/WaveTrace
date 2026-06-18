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
class ModelConfig:
    """Recognition-head config (Phase 6). `stage` picks the target (A presence / E weapon, Phase 7);
    `backend` picks the implementation behind the backend-agnostic head wrapper.

    `k` (NBVI subcarrier count) is dataset-dependent — required, no fabricated default (it comes from
    the calibration that built the training dataset). window/hop default to the locked P4 cadence.
    Backends: "mlp" = sklearn MLPClassifier (DEFAULT, locked: native predict_proba for the Phase-7
    soft vote + weights port to the future numpy-tiny ESP32 head), "svm" = calibrated SVC (the
    classic CSI-sensing literature head, kept selectable for A/B on real recordings). Weapon-head
    (P7) additions: "variance" = the σ²[p]-threshold baseline (Yousaf: metal → lower inter-carrier
    variance), "cnn" = torch 2D-CNN on the CSI image (lazy import; PresenceHead stays sklearn-only)."""

    stage: str                      # "presence" | "weapon"
    k: int                          # NBVI subcarrier count -> feature dim = 9*k per node
    backend: str = "mlp"            # "mlp" (default) | "svm" | "variance" (P7) | "cnn" (P7)
    window: int = 128               # front-end window (frames), locked P4
    hop: int = 32                   # front-end hop (frames), locked P4
    fs_tol: float = 0.10            # fs_ok: max relative live-fs deviation before a window is dropped
    hidden: int = 32                # MLP hidden width (single layer — tiny head, O(1) forward)
    seed: int = 0                   # backend rng seed (deterministic training/inference)
    frame_average: int = 1          # T2/P10: non-overlapping decimating mean (M=1 = no change)
    subtract_baseline: bool = False  # T3/P10: subtract quiet-room baseline from image path

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
