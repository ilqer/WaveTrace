"""Is the person carrying metal? One verdict per window of inter-carrier features."""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from wavetrace.Config import ModelConfig
from wavetrace.domain.contracts import SCHEMA_VERSION, PipelineContract, derive_pipeline_contract

# Column 9 of the 27-wide inter-carrier block: the window mean of the per-packet σ²[p] series
# (the block is µ | σ² | CV, 9 stats each, stat 0 = mean). Metal flattens amplitude across
# subcarriers, so this column carries the signature.
VARIANCE_FEATURE = 9


class WeaponHead:
    """Classifies one window as weapon / no-weapon through an injected backend.

    The 'variance', 'mlp' and 'svm' backends read the inter-carrier block built from RAW
    magnitudes: gain-locked amplitudes cancel the cross-subcarrier flatness metal produces.
    The 'cnn' backend reads the spectrogram image instead."""
    def __init__(self, config: ModelConfig, backend, *, variance_feature: int = VARIANCE_FEATURE):
        self.config = config
        self._backend = backend
        self._variance_feature_column = int(variance_feature)
        # which input the backend expects: "ic27" | "fusion" | "cnn"; None until training records it
        self.feature_mode: str | None = None
        self._fitted = False
        self.contract = derive_pipeline_contract(config)

    @classmethod
    def restore(
        cls, config: ModelConfig, backend, contract: PipelineContract, *,
        feature_mode: str | None, variance_feature: int,
    ) -> "WeaponHead":
        """Rebuild a fitted head around an already-loaded backend and contract."""
        head = cls(config, backend, variance_feature=variance_feature)
        head.feature_mode = feature_mode
        head.contract = contract
        head._fitted = True
        return head

    @property
    def classes_(self) -> np.ndarray:
        self._require_fitted()
        return self._backend.classes_

    @property
    def default_feature_mode(self) -> str:
        """What the backend expects when the artifact recorded no `feature_mode` of its own."""
        return self._backend.feature_mode

    def fit(self, X, y, *, epochs: int = 30, lr: float = 1e-3, batch_size: int = 32,
            report=None) -> "WeaponHead":
        """Fit on inter-carrier blocks or images. Returns self."""
        y = np.asarray(y, dtype=np.int64)
        classes = np.unique(y)
        if classes.size < 2:
            # a 1-class dataset fits a model that can only ever answer that class
            raise ValueError(
                f"WeaponHead.fit: training data has a single class {classes.tolist()}; need both "
                "weapon and no-weapon windows (check weapon label spans / --weapon-depth)"
            )
        self._backend.fit(X, y, epochs=epochs, lr=lr, batch_size=batch_size, report=report)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        self._require_fitted()
        return self._backend.predict_proba(X)

    def save(self, path) -> Path:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "config": asdict(self.config),
            "variance_feature": self._variance_feature_column,
            "feature_mode": self.feature_mode,
            "contract": self.contract.to_dict(),
            "schema_version": SCHEMA_VERSION,
        }
        blob.update(self._backend.save())
        joblib.dump(blob, path)
        return path

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("WeaponHead: not fitted (call fit() or load())")
