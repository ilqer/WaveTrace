"""A trained model's resample rate, window/hop framing, and subcarrier width, declared once and
persisted with the model (`PresenceHead.save`/`WeaponHead.save`) so serving loads it instead of
re-declaring the numbers."""

from dataclasses import asdict, dataclass

DEFAULT_TARGET_SAMPLE_RATE_HZ: float = 100.0
DEFAULT_WINDOW_FRAMES: int = 128
DEFAULT_HOP_FRAMES: int = 32

# Bump only when PipelineContract's field set or types change.
SCHEMA_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class PipelineContract:
    """`subcarrier_width` is the raw per-frame capture width (64/128/192 by band) — never the
    NBVI-selected subcarrier count (`ModelConfig.k`), a different quantity. `None` means no capture
    width was recorded (a head built outside `Train.py`, or trained from a dataset saved before
    `meta["num_subcarriers"]` existed); `Train.py` is the only writer of a real value."""

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
        """For joblib/json artifact storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineContract":
        return cls(**data)


def derive_pipeline_contract(config, target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ):
    """`config` is a `wavetrace.Config.ModelConfig` (duck-typed on `.window`/`.hop`)."""
    return PipelineContract(
        target_sample_rate_hz=target_sample_rate_hz,
        window_frames=config.window,
        hop_frames=config.hop,
    )
