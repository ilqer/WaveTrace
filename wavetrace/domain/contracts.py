"""Pipeline contract: the one declaration site for the numbers a train/serve mismatch would
corrupt silently — resample rate, window/hop framing, and subcarrier width. Every trained model
persists one (`PresenceHead.save`/`WeaponHead.save`); serving loads it back instead of
re-declaring the numbers."""

from dataclasses import asdict, dataclass

# "today's values": the pipeline's default resample rate, window and hop.
DEFAULT_TARGET_SAMPLE_RATE_HZ: float = 100.0
DEFAULT_WINDOW_FRAMES: int = 128
DEFAULT_HOP_FRAMES: int = 32

# Bumped only if PipelineContract's shape changes. Enforcement is deliberately deferred; recorded
# now so a future enforcement pass can tell an artifact's contract format apart from its absence.
SCHEMA_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class PipelineContract:
    """What a trained model was fit on: resample rate, window/hop framing, subcarrier width.

    `subcarrier_width` is the raw per-frame capture width (64/128/192 by band) — never the
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
        """Serialize for joblib/json artifact storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineContract":
        """Round-trip a dict produced by `to_dict`."""
        return cls(**data)


def derive_pipeline_contract(config, target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ):
    """Build a `PipelineContract` from a `ModelConfig` — the one shared derivation function.

    `config.window` / `config.hop` are already correctly plumbed per-model from dataset meta
    (`Train.py`); this just packages them with the pipeline's resample rate so
    `PresenceHead.save`/`WeaponHead.save` and their `load` fallback never re-type the numbers.
    Leaves `subcarrier_width` at its `None` default — this function has no dataset to read a real
    capture width from; only `Train.py` (which does) ever sets it. `config` is a
    `wavetrace.Config.ModelConfig`; left untyped here (duck-typed on `.window`/`.hop`) so this
    module keeps no import of `wavetrace.Config`.
    """
    return PipelineContract(
        target_sample_rate_hz=target_sample_rate_hz,
        window_frames=config.window,
        hop_frames=config.hop,
    )
