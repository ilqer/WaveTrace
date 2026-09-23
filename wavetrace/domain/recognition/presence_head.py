"""Is a person in the room? One verdict per window of CSI features."""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from wavetrace.Config import ModelConfig
from wavetrace.domain.contracts import SCHEMA_VERSION, PipelineContract, derive_pipeline_contract


class PresenceHead:
    """Classifies one (9*K,) gain-locked feature row through an injected backend.

    Fitting is offline; a forward pass is O(1) in a fixed-length row."""

    def __init__(self, config: ModelConfig, backend):
        self.config = config
        self._backend = backend
        self._fitted = False
        self.contract = derive_pipeline_contract(config)

    @classmethod
    def restore(cls, config: ModelConfig, backend, contract: PipelineContract) -> "PresenceHead":
        """Rebuild a fitted head around an already-loaded backend and contract."""
        head = cls(config, backend)
        head.contract = contract
        head._fitted = True
        return head

    @property
    def classes_(self) -> np.ndarray:
        self._require_fitted()
        return self._backend.classes_

    def fit(self, X, y) -> "PresenceHead":
        """Fit on (n, d) float32 features, (n,) int labels. Offline. Returns self."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(f"fit expects X (n, d) and y (n,), got {X.shape} / {y.shape}")
        classes = np.unique(y)
        if classes.size < 2:
            # a 1-class dataset fits a model that can only ever answer that class
            raise ValueError(
                f"PresenceHead.fit: training data has a single class {classes.tolist()}; need both "
                "present and absent windows (check collect-data label spans / presence turbulence)"
            )
        self._backend.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """(n, d) -> (n,) class ids. O(1) per row."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        """(n, d) -> (n, C) class probabilities, columns ordered by classes_. O(1) per row."""
        self._require_fitted()
        return self._backend.predict_proba(X)

    def save(self, path) -> Path:
        """Persist (joblib) — config stored as a plain dict so loads survive dataclass evolution."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "config": asdict(self.config),
            "contract": self.contract.to_dict(),
            "schema_version": SCHEMA_VERSION,
        }
        blob.update(self._backend.save())
        joblib.dump(blob, path)
        return path

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("PresenceHead: not fitted (call fit() or load())")
