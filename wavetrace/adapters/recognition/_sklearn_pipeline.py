from abc import ABC, abstractmethod

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SklearnPipelineBackend(ABC):
    """StandardScaler is required: the features live on wildly different scales (mean |H| ~1 vs
    lag-1 autocorrelation in [-1, 1]).

    `feature_mode` 'ic27' is the fallback `plan_input` uses for any head saved before
    `feature_mode` was recorded."""

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
