"""Learned late fusion: a logistic regression maps each band's probability to a final one.

The weights are learned on a held-out validation split. node_id < 100 is 2.4 GHz, >= 100 is 5 GHz.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class BandFusion:
    def __init__(self, bands: dict):
        self.bands = dict(bands)
        self.band_order = sorted(self.bands.keys())
        self._combiner: LogisticRegression | None = None
        self._classes: np.ndarray | None = None

    def _stack_probas(self, X_by_band: dict) -> np.ndarray:
        """Concatenate each band's positive-class probability into (n, n_bands)."""
        columns = []
        for band in self.band_order:
            probabilities = self.bands[band].predict_proba(np.asarray(X_by_band[band]))
            columns.append(probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities.max(axis=1))
        return np.column_stack(columns)

    def fit(self, X_by_band_validation: dict, y_validation) -> "BandFusion":
        band_probabilities = self._stack_probas(X_by_band_validation)
        y = np.asarray(y_validation, dtype=np.int64)
        self._combiner = LogisticRegression(max_iter=1000).fit(band_probabilities, y)
        self._classes = self._combiner.classes_
        return self

    def predict_proba(self, X_by_band: dict) -> np.ndarray:
        band_probabilities = self._stack_probas(X_by_band)
        return self._combiner.predict_proba(band_probabilities)

    def predict(self, X_by_band: dict) -> np.ndarray:
        proba = self.predict_proba(X_by_band)
        return self._classes[np.argmax(proba, axis=1)]

    @property
    def weights_(self) -> dict:
        """Learned per-band trust: softmax of coefficients. For display."""
        if self._combiner is None:
            return {}
        coefficients = self._combiner.coef_.ravel()
        exponentials = np.exp(coefficients - coefficients.max())
        band_weights = exponentials / exponentials.sum()
        return {band: round(float(band_weights[i]), 3) for i, band in enumerate(self.band_order)}

    def contribution(self, X_by_band: dict) -> dict:
        """Per-band positive probability for one window."""
        single = {band: (X_by_band[band][:1] if np.asarray(X_by_band[band]).ndim > 1
                         else np.asarray(X_by_band[band]).reshape(1, -1))
                  for band in self.band_order}
        band_probabilities = self._stack_probas(single)[0]
        fused = float(self.predict_proba(single)[0][1])
        return {**{band: round(float(band_probabilities[i]), 3) for i, band in enumerate(self.band_order)},
                "fused": round(fused, 3), "weights": self.weights_}

    def save(self, path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"combiner": self._combiner, "band_order": self.band_order,
                     "classes": self._classes}, output_path)
        return output_path

    @classmethod
    def load(cls, path, bands: dict) -> "BandFusion":
        blob = joblib.load(path)
        fusion = cls(bands)
        fusion._combiner = blob["combiner"]
        fusion.band_order = blob["band_order"]
        fusion._classes = blob["classes"]
        return fusion
