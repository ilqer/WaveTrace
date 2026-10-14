from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceOptions:
    """`timeout_seconds` is the idle wait for the next packet before the stream ends."""

    timeout_seconds: float = 5.0
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when given")
