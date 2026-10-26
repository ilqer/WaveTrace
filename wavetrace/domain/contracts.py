from dataclasses import asdict, dataclass
from typing import Literal

from wavetrace.domain.model_config import ModelConfig

DEFAULT_TARGET_SAMPLE_RATE_HZ: float = 100.0
DEFAULT_MIN_SAMPLE_RATE_FRACTION: float = 0.8
DEFAULT_WINDOW_FRAMES: int = 128
DEFAULT_HOP_FRAMES: int = 32



def require_valid_min_sample_rate_fraction(min_sample_rate_fraction: float) -> None:
    if not 0.0 < min_sample_rate_fraction <= 1.0:
        raise ValueError("min_sample_rate_fraction must be in (0, 1]")


# Bump only when PipelineContract's field set or types change.
SCHEMA_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class PipelineContract:
    """`subcarrier_width` is the raw per-frame capture width (64/128/192 by band) — never the
    NBVI-selected subcarrier count (`ModelConfig.k`), a different quantity. `None` means no capture
    width was recorded (a head built outside `train.py`, or trained from a dataset saved before
    `meta["num_subcarriers"]` existed); `train.py` is the only writer of a real value."""

    target_sample_rate_hz: float
    window_frames: int
    hop_frames: int
    subcarrier_width: int | None = None

    def __post_init__(self) -> None:
        if self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive")
        if self.window_frames <= 0:
            raise ValueError("window_frames must be positive")
        if self.hop_frames <= 0:
            raise ValueError("hop_frames must be positive")
        if self.hop_frames > self.window_frames:
            raise ValueError("hop_frames must not exceed window_frames")
        if self.subcarrier_width is not None and self.subcarrier_width <= 0:
            raise ValueError("subcarrier_width must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineContract":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class LibraryVersions:
    python: str
    numpy: str
    scikit_learn: str
    wavetrace: str | None  # None when wavetrace runs from a source tree that was never installed
    torch: str | None = None  # None when PyTorch is not installed


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Where one model run came from, saved beside its model. `origin` is "trained" or "uploaded"; nothing is
    known of an uploaded run's making, so every other field but `created_timestamp_s` is None for it.
    `git_commit` and `has_uncommitted_changes` are also None outside a git checkout, and the latter counts
    tracked files only. Both hashes are hex sha256: `config_hash` of the ModelConfig's canonical JSON,
    `dataset_hash` streamed over every dataset file the run read, each with its relative path (None when
    those files could not be read). `created_timestamp_s` is UTC Unix seconds."""

    origin: Literal["trained", "uploaded"]
    git_commit: str | None
    has_uncommitted_changes: bool | None
    config_hash: str | None
    dataset_hash: str | None
    seed: int | None
    library_versions: LibraryVersions | None
    contract: PipelineContract | None
    created_timestamp_s: int
    schema_version: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def derive_pipeline_contract(
    config: ModelConfig, target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ
) -> PipelineContract:
    return PipelineContract(
        target_sample_rate_hz=target_sample_rate_hz,
        window_frames=config.window,
        hop_frames=config.hop,
    )
