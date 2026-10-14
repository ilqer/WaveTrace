import numpy as np


def decide_class(proba: np.ndarray, classes: np.ndarray) -> tuple[int, float]:
    """One window's (C,) probabilities, columns ordered by `classes` -> (class_id, its probability). O(C)."""
    class_index = int(np.argmax(proba))
    return int(classes[class_index]), float(proba[class_index])
