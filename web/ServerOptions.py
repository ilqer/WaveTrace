"""Delivery-layer configuration for the FastAPI/uvicorn dashboard server."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WebServerOptions:
    """Settings for the FastAPI/uvicorn dashboard server (`web/app.py`), including the
    reload-watcher's own paths/globs."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    reload_dirs: tuple[str, ...] = ("web", "wavetrace")
    reload_includes: tuple[str, ...] = ("*.py",)

    def __post_init__(self) -> None:
        if not 0 < self.port < 65536:
            raise ValueError("port must be in (0, 65536)")
