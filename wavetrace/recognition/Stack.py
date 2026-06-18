"""Learned late-fusion of per-band models (2.4 GHz ESP32s vs 5 GHz Pi).

Each band has its own trained head; a logistic-regression combiner maps their positive-class
probabilities -> the final probability. The combiner's coefficients ARE the band weights so the UI
can show 'this room learned to trust 5 GHz 0.71 / 2.4 GHz 0.29'.

Why stacking and not a hand-weighted average: stacking learns the weights from a validation split
with no magic numbers, stays interpretable, and lets you add/remove a band by retraining only the
(cheap) combiner. A single end-to-end two-branch net is the later option if this plateaus.

Band split rule (matches NodeHealthMeter): node_id < 100 = 2.4 GHz, node_id >= 100 = 5 GHz."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class BandFusion:
    """Combine N per-band heads into one verdict with learned weights.

    bands: dict[band_name -> fitted head] (each exposes predict_proba(X) -> (n, C)).
    Fit the combiner on a HELD-OUT VALIDATION split — not the same data used to train each band head,
    or the combiner just echoes the strongest single-band head."""

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
        """Fit the combiner on VALIDATION split band probabilities + true labels. Offline."""
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
        """Learned per-band trust: softmax of combiner coefficients, sums to 1. For display.
        Positive coef = this band pushes toward the weapon/present class."""
        if self._combiner is None:
            return {}
        coef = self._combiner.coef_.ravel()
        ex = np.exp(coef - coef.max())
        w = ex / ex.sum()
        return {b: round(float(w[i]), 3) for i, b in enumerate(self.band_order)}

    def contribution(self, X_by_band: dict) -> dict:
        """Per-band positive probability for ONE window (for the DecisionContribution widget).
        Returns {band: prob, ..., 'fused': final_prob, 'weights': learned_weights}."""
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
