"""The weapon baseline recognition backend: a single threshold learned over one inter-carrier
feature column. Physics: metal lowers CSI variance, so the window-mean of the per-packet σ²[p]
series (the `variance_feature`-th column of the 27-block) separates weapon from no-weapon.
Fit is one O(n log n) sort over the training window; predict is a hand-rolled logistic in the
threshold margin.
"""

import numpy as np


class VarianceBackend:
    """A learned threshold `_threshold`, scale `_scale` and direction `_positive_below` over one
    feature column. `predict_proba`'s logistic sign follows the learned direction."""

    feature_mode = "ic27"

    def __init__(self, config, **backend_options):
        # deferred: wavetrace.recognition.Weapon pulls in the recognition package, which pulls in
        # this package's heads.py — importing VARIANCE_FEATURE at module load time would cycle back
        # here before get_backend_class exists whenever adapters.recognition loads first.
        from wavetrace.recognition.Weapon import VARIANCE_FEATURE
        self.config = config
        self._variance_feature_column = int(backend_options.get("variance_feature", VARIANCE_FEATURE))
        self._threshold = None
        self._scale = None
        self._positive_below = True
        self._classes = None

    @property
    def classes_(self) -> np.ndarray:
        return self._classes

    def fit(self, X, y, **kwargs) -> None:
        X = np.asarray(X, dtype=np.float32)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(f"variance backend is binary, got classes {classes.tolist()}")
        feature_values = X[:, self._variance_feature_column]
        order = np.argsort(feature_values, kind="stable")
        sorted_feature = feature_values[order]
        is_positive = (y[order] == classes[1]).astype(np.int64)
        positive_count = int(is_positive.sum())
        negative_count = int(is_positive.size - positive_count)
        if positive_count == 0 or negative_count == 0:
            raise ValueError("variance backend needs both classes in the training data")
        cumulative_positive = np.cumsum(is_positive)  # positives among sorted_feature[:i+1]
        count_at_or_below = np.arange(1, sorted_feature.size + 1)
        # balanced accuracy of "positive when sorted_feature <= _threshold" at every split point;
        # one O(n) pass over the sorted feature
        true_positive_rate_low = cumulative_positive / positive_count
        true_negative_rate_low = (negative_count - (count_at_or_below - cumulative_positive)) / negative_count
        balanced_accuracy_low = (true_positive_rate_low + true_negative_rate_low) / 2.0
        balanced_accuracy_high = 1.0 - balanced_accuracy_low  # flipping the direction flips both rates
        is_valid_split = np.empty(sorted_feature.size, dtype=bool)  # no threshold between equal values
        is_valid_split[:-1] = sorted_feature[:-1] < sorted_feature[1:]
        is_valid_split[-1] = False
        if not is_valid_split.any():
            raise ValueError("variance backend: feature is constant, nothing to threshold")
        best_low_index = int(
            np.flatnonzero(is_valid_split)[np.argmax(balanced_accuracy_low[is_valid_split])]
        )
        best_high_index = int(
            np.flatnonzero(is_valid_split)[np.argmax(balanced_accuracy_high[is_valid_split])]
        )
        self._positive_below = bool(
            balanced_accuracy_low[best_low_index] >= balanced_accuracy_high[best_high_index]
        )
        chosen_index = best_low_index if self._positive_below else best_high_index
        self._threshold = float((sorted_feature[chosen_index] + sorted_feature[chosen_index + 1]) / 2.0)
        mad = float(np.median(np.abs(feature_values - np.median(feature_values))))
        # robust σ for the logistic
        self._scale = 1.4826 * mad if mad > 0 else (float(feature_values.std()) or 1.0)
        self._classes = classes

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        # logistic in the threshold margin; sign flips with the learned direction
        margin = (self._threshold - X[:, self._variance_feature_column]) / self._scale
        positive_probability = 1.0 / (1.0 + np.exp(-(margin if self._positive_below else -margin)))
        return np.stack([1.0 - positive_probability, positive_probability], axis=1)

    def save(self) -> dict:
        return {
            "thr": self._threshold,
            "scale": self._scale,
            "positive_below": self._positive_below,
            "classes": self._classes,
        }

    @classmethod
    def load(cls, blob: dict, config, **backend_options) -> "VarianceBackend":
        backend = cls(config, **backend_options)
        backend._threshold = blob["thr"]
        backend._scale = blob["scale"]
        backend._positive_below = blob["positive_below"]
        backend._classes = blob["classes"]
        return backend
