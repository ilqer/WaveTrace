"""Phase 6b — PresenceHead: the first CSI-only model (Stage A), behind a backend-agnostic wrapper.

The head consumes the Phase-5 `X_features` (n, 9·K) — §2.9 nine features per NBVI subcarrier on
gain-locked amplitudes; a present human's dynamic multipath raises std/MAD/waveform-length, which is
what separates "present" from a quiet room. The wrapper API (fit/predict/predict_proba/save/load) is
deliberately backend-agnostic so the future numpy-only tiny head (ESP32 deployment) and the torch CNN
(Phase-7 weapon heatmap) drop in unchanged.

Backends (ModelConfig.backend, user-locked: MLP default, SVM selectable):
  * "mlp" — StandardScaler + MLPClassifier(one small hidden layer). Native predict_proba (the Phase-7
    soft vote needs calibrated probabilities) and the weight matrices port directly to a numpy-only
    forward pass on the ESP32.
  * "svm" — StandardScaler + SVC(probability=True). The classic CSI-sensing literature head
    (WiFiSenseSurvey CSUR'19), kept for A/B on real recordings; predict_proba = Platt scaling.

Training is OFFLINE; the forward pass is O(1) (fixed-length feature vector, tiny model) and is the
real-time path Infer.py wraps (<8 ms gate).
"""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from wavetrace.Config import ModelConfig


def sklearn_pipeline(config: ModelConfig) -> Pipeline:
    """Shared sklearn backend builder ('mlp' | 'svm') — used by PresenceHead and WeaponHead (P7).

    StandardScaler is required: the input features live on wildly different scales (mean |H| ~1 vs
    lag-1 autocorr in [-1, 1]); both backends assume standardized inputs."""
    if config.backend == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(config.hidden,),
            max_iter=3000,
            random_state=config.seed,
        )
    elif config.backend == "svm":
        # sklearn 1.9 deprecated SVC(probability=True); this is its documented replacement
        # (sigmoid calibration on the decision function, single model with ensemble=False)
        clf = CalibratedClassifierCV(SVC(random_state=config.seed), ensemble=False)
    else:
        raise ValueError(f"sklearn_pipeline supports 'mlp'/'svm', not {config.backend!r}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


class PresenceHead:
    """Backend-agnostic recognition head (Stage A presence now; same wrapper serves later stages)."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._pipe = sklearn_pipeline(config)  # presence backends are sklearn-only (P6 lock)
        self._fitted = False

    @property
    def classes_(self) -> np.ndarray:
        self._require_fitted()
        return self._pipe.classes_

    def fit(self, X, y) -> "PresenceHead":
        """Fit on (n, d) float32 features, (n,) int labels. Offline. Returns self."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(f"fit expects X (n, d) and y (n,), got {X.shape} / {y.shape}")
        classes = np.unique(y)
        if classes.size < 2:
            # a 1-class dataset fits a model that can only ever predict that class (the silent failure
            # behind the all-one-verdict bug) — refuse instead of reporting a meaningless acc 1.0
            raise ValueError(
                f"PresenceHead.fit: training data has a single class {classes.tolist()}; need both "
                "present and absent windows (check collect-data label spans / presence turbulence)"
            )
        self._pipe.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """(n, d) -> (n,) class ids. O(1) per row."""
        self._require_fitted()
        return self._pipe.predict(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X) -> np.ndarray:
        """(n, d) -> (n, C) class probabilities, columns ordered by classes_. O(1) per row."""
        self._require_fitted()
        return self._pipe.predict_proba(np.asarray(X, dtype=np.float32))

    def save(self, path) -> Path:
        """Persist (joblib) — config stored as a plain dict so loads survive dataclass evolution."""
        self._require_fitted()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": asdict(self.config), "pipeline": self._pipe}, p)
        return p

    @classmethod
    def load(cls, path) -> "PresenceHead":
        """Round-trip a saved head."""
        blob = joblib.load(path)
        head = cls(ModelConfig(**blob["config"]))
        head._pipe = blob["pipeline"]
        head._fitted = True
        return head

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("PresenceHead: not fitted (call fit() or load())")
