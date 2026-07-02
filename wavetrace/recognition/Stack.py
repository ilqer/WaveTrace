"""Learned late-fusion of per-band models.

Logistic regression maps band probabilities to final probability. Stacking learns weights from validation split.

Band split: node_id < 100 is 2.4 GHz, node_id >= 100 is 5 GHz."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class BandFusion:
    """Combine per-band heads with learned weights. Fit on held-out validation split."""

    def __init__(self, bands: dict):
        self.bands = dict(bands)
        self.band_order = sorted(self.bands.keys())
        self._combiner: LogisticRegression | None = None
        self._classes: np.ndarray | None = None

    def _stack_probs(self, X_by_band: dict) -> np.ndarray:
        """Concatenate each band's positive-class probability into (n, n_bands)."""
        cols = []
        for b in self.band_order:
            p = self.bands[b].predict_proba(np.asarray(X_by_band[b]))
            cols.append(p[:, 1] if p.shape[1] == 2 else p.max(axis=1))
        return np.column_stack(cols)

    def fit(self, X_by_band_val: dict, y_val) -> "BandFusion":
        """Fit combiner on validation split."""
        Z = self._stack_probs(X_by_band_val)
        y = np.asarray(y_val, dtype=np.int64)
        self._combiner = LogisticRegression(max_iter=1000).fit(Z, y)
        self._classes = self._combiner.classes_
        return self

    def predict_proba(self, X_by_band: dict) -> np.ndarray:
        Z = self._stack_probs(X_by_band)
        return self._combiner.predict_proba(Z)

    def predict(self, X_by_band: dict) -> np.ndarray:
        proba = self.predict_proba(X_by_band)
        return self._classes[np.argmax(proba, axis=1)]

    @property
    def weights_(self) -> dict:
        """Learned per-band trust: softmax of coefficients. For display."""
        if self._combiner is None:
            return {}
        coef = self._combiner.coef_.ravel()
        ex = np.exp(coef - coef.max())
        w = ex / ex.sum()
        return {b: round(float(w[i]), 3) for i, b in enumerate(self.band_order)}

    def contribution(self, X_by_band: dict) -> dict:
        """Per-band positive probability for one window."""
        single = {b: (X_by_band[b][:1] if np.asarray(X_by_band[b]).ndim > 1
                      else np.asarray(X_by_band[b]).reshape(1, -1))
                  for b in self.band_order}
        Z = self._stack_probs(single)[0]
        fused = float(self.predict_proba(single)[0][1])
        return {**{b: round(float(Z[i]), 3) for i, b in enumerate(self.band_order)},
                "fused": round(fused, 3), "weights": self.weights_}

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"combiner": self._combiner, "band_order": self.band_order,
                     "classes": self._classes}, p)
        return p

    @classmethod
    def load(cls, path, bands: dict) -> "BandFusion":
        blob = joblib.load(path)
        obj = cls(bands)
        obj._combiner = blob["combiner"]
        obj.band_order = blob["band_order"]
        obj._classes = blob["classes"]
        return obj
