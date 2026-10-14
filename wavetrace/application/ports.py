from abc import abstractmethod
from typing import Protocol

import numpy as np


class CsiSource(Protocol):
    @abstractmethod
    def frames(self):
        """Yield CsiFrame objects in capture order."""


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
