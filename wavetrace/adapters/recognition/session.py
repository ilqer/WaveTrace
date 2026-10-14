import numpy as np

from wavetrace.adapters.recognition.heads import load_presence_head, load_weapon_head
from wavetrace.domain.recognition import PresenceHead, WeaponHead


class InferenceSession:
    def __init__(self, model_path=None, *, head=None, loader=None):
        if head is not None:
            self._head = head
        else:
            self._head = (loader or load_presence_head)(model_path)
        self._row = None

    @property
    def head(self) -> PresenceHead | WeaponHead:
        return self._head

    def predict_window_proba(self, feature_vector) -> np.ndarray:
        """Copies the input into a reused (1, d) row buffer. O(1)."""
        flattened = np.asarray(feature_vector, dtype=np.float32).ravel()
        if self._row is None or self._row.shape[1] != flattened.size:
            self._row = np.empty((1, flattened.size), dtype=np.float32)
        self._row[0, :] = flattened
        return self._head.predict_proba(self._row)[0]

    def predict_window(self, feature_vector) -> tuple[int, float]:
        """One emitted window's feature vector (d,) -> (class_id, probability). O(1)."""
        proba = self.predict_window_proba(feature_vector)
        i = int(np.argmax(proba))
        return int(self._head.classes_[i]), float(proba[i])


def open_session(mode: str, model_path) -> InferenceSession:
    """Mode switch: 'presence' or 'weapon'. O(1)."""
    if mode == "presence":
        loader = load_presence_head
    elif mode == "weapon":
        loader = load_weapon_head
    else:
        raise ValueError(f"mode must be 'presence' or 'weapon', got {mode!r}")
    return InferenceSession(model_path, loader=loader)
