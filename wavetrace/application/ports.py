import threading
from abc import abstractmethod
from collections.abc import Callable
from typing import Protocol

import numpy as np

from wavetrace import Label
from wavetrace.domain.link_key import LinkKey


class CsiSource(Protocol):
    @abstractmethod
    def frames(self):
        """Yield CsiFrame objects in capture order."""


class LinkBatchReceiver(Protocol):
    @abstractmethod
    def receive(self) -> dict[LinkKey, list]:
        """One mesh datagram's CsiFrames per link; {} when the read times out."""

    @abstractmethod
    def close(self) -> None: ...


class LinkCapture(Protocol):
    @abstractmethod
    def capture(self, prompt: str, *, countdown_s: int, hint: str) -> dict[LinkKey, list]:
        """Prompts, waits for the operator, then records every link it hears; each link's CsiFrames at
        its dominant subcarrier width."""


class TimedLinkCapture(Protocol):
    @abstractmethod
    def capture_for(self, duration_s: float, *, stop: threading.Event) -> dict[LinkKey, list]:
        """Records every link it hears for `duration_s`, or until `stop` is set; each link's CsiFrames at
        its dominant subcarrier width."""


class CameraLabelCapture(Protocol):
    @abstractmethod
    def record_labels(self, duration_s: float, *, stop: threading.Event,
                      on_label: Callable[[Label], None]) -> list[Label]:
        """Labels camera frames live for `duration_s`, or until `stop` is set, calling `on_label` per frame;
        returns the Labels sorted by timestamp, on the CSI clock. Raises when the camera cannot open. Must
        return once `stop` is set: the runner joins its thread without a timeout."""


class RecognitionBackend(Protocol):
    """`feature_mode` is the default input shape: 'ic27' (inter-carrier block), 'fusion' (inter-carrier
    block + amplitude features) or 'cnn' (CSI image); serving falls back to it for a head saved before
    it recorded its own. `save`/`load` round-trip only the backend's own state; the head owns `config`,
    `contract` and `schema_version` in the outer joblib blob."""

    feature_mode: str

    def __init__(self, config, **backend_options: object) -> None:
        """Every implementation takes the head's `ModelConfig` plus backend-specific keywords."""

    @property
    def classes_(self) -> np.ndarray:
        """Class ids in the order `predict_proba` columns are ordered. Valid only once fitted."""
        ...

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit in place on (n, d) features (or images) and (n,) integer labels."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, d) -> (n, C) class probabilities, columns ordered by `classes_`. O(1) per row."""

    def save(self) -> dict:
        """This backend's own state, as a dict fragment merged into the head's saved blob."""

    @classmethod
    def load(cls, blob: dict, config, **backend_options) -> "RecognitionBackend":
        """Reconstruct a fitted backend from a blob fragment produced by `save`."""


class CirSource(Protocol):
    @abstractmethod
    def records(self):
        """Yield CirRecord objects in arrival order."""
