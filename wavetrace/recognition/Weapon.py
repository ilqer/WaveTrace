"""Phase 7p-b/c: WeaponHead — the backend-agnostic weapon-detection wrapper.

Input by backend:
* 'variance', 'mlp', 'svm': X_intercarrier (built from RAW magnitudes). Do NOT use gain-locked features.
* 'cnn': X_image.

The head takes an already-resolved backend (constructor injection): resolving `config.backend`
against the registry, and choosing which backend-specific keywords to pass, is
`wavetrace.adapters.recognition.heads`'s job, not this class's.
"""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from wavetrace.Config import ModelConfig
from wavetrace.domain.contracts import SCHEMA_VERSION, PipelineContract, derive_pipeline_contract

# Column of the 27-block holding the WINDOW MEAN of the per-packet σ²[p] series (µ|σ²|CV, 9 stats
# each, stat 0 = mean) — a domain fact about the inter-carrier block, not a backend detail.
# `variance_backend.py` imports this constant from here.
VARIANCE_FEATURE = 9


class WeaponHead:
    """Stage-E weapon head (binary weapon / no-weapon; same API as PresenceHead)."""

    def __init__(self, config: ModelConfig, backend, *, variance_feature: int = VARIANCE_FEATURE):
        self.config = config
        self._backend = backend
        self._variance_feature_column = int(variance_feature)
        # how serving (Infer.planInferenceInput) must assemble X: "ic27" | "fusion" | "cnn"; None
        # for directly-constructed heads.
        self.feature_mode: str | None = None
        self._fitted = False
        self.contract = derive_pipeline_contract(config)  # what this head trains/serves against

    @classmethod
    def restore(
        cls, config: ModelConfig, backend, contract: PipelineContract, *,
        feature_mode: str | None, variance_feature: int,
    ) -> "WeaponHead":
        """Rebuild a fitted head around an already-loaded backend and contract. Resolves nothing."""
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
        """The feature_mode this head's backend expects when none was explicitly recorded — the
        serving fallback `planInferenceInput` uses for artifacts saved before `feature_mode` was
        tracked on the head itself."""
        return self._backend.feature_mode

    # ----- fit ------------------------------------------------------------------------------------

    def fit(self, X, y, *, epochs: int = 30, lr: float = 1e-3, batch_size: int = 32,
            report=None) -> "WeaponHead":
        """Fit on inter-carrier blocks or images. Returns self."""
        y = np.asarray(y, dtype=np.int64)
        classes = np.unique(y)
        if classes.size < 2:
            # 1-class data -> a model that only ever predicts that class (silent failure); refuse.
            raise ValueError(
                f"WeaponHead.fit: training data has a single class {classes.tolist()}; need both "
                "weapon and no-weapon windows (check weapon label spans / --weapon-depth)"
            )
        self._backend.fit(X, y, epochs=epochs, lr=lr, batch_size=batch_size, report=report)
        self._fitted = True
        return self

    # ----- predict --------------------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """Predict class ids."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        self._require_fitted()
        return self._backend.predict_proba(X)

    # ----- persist --------------------------------------------------------------------------------

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
