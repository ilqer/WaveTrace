"""Shared `StandardScaler` + sklearn-classifier plumbing for the 'mlp' and 'svm'
`RecognitionBackend` implementations — they differ only in which classifier the scaler feeds.
Not itself registered as a backend; `MlpBackend` and `SvmBackend` subclass it.
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SklearnPipelineBackend(ABC):
    """A `StandardScaler` + sklearn classifier pipeline.

    StandardScaler is required: the input features live on wildly different scales (mean |H| ~1 vs
    lag-1 autocorrelation in [-1, 1]); every sklearn-based backend assumes standardized inputs.

    `feature_mode` defaults to 'ic27' — the fallback `planInferenceInput` uses for any head saved
    before `feature_mode` was recorded. 'fusion' is also a legal training-time choice for this
    backend and is then recorded explicitly on the head, overriding this default."""

    feature_mode = "ic27"

    def __init__(self, config, **backend_options):
        self.config = config
        self._pipeline = Pipeline([("scale", StandardScaler()), ("clf", self._build_classifier(config))])

    @abstractmethod
    def _build_classifier(self, config):
        """The one step `MlpBackend`/`SvmBackend` differ on."""

    @property
    def classes_(self) -> np.ndarray:
        return self._pipeline.classes_

    def fit(self, X, y, **kwargs) -> None:
        self._pipeline.fit(np.asarray(X, dtype=np.float32), y)

    def predict_proba(self, X) -> np.ndarray:
        return self._pipeline.predict_proba(np.asarray(X, dtype=np.float32))

    def save(self) -> dict:
        return {"pipeline": self._pipeline}

    @classmethod
    def load(cls, blob: dict, config, **backend_options) -> "SklearnPipelineBackend":
        backend = cls(config, **backend_options)
        backend._pipeline = blob["pipeline"]
        return backend
