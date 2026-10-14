"""The recognition-backend registry.

Maps a `ModelConfig.backend` key to its `RecognitionBackend` implementation. `PresenceHead` and
`WeaponHead` resolve a backend once — at construction or `load()` — through `get_backend_class`.
"""

from wavetrace.adapters.recognition._sklearn_pipeline import SklearnPipelineBackend
from wavetrace.adapters.recognition.cnn_backend import CnnBackend
from wavetrace.adapters.recognition.mlp_backend import MlpBackend
from wavetrace.adapters.recognition.svm_backend import SvmBackend
from wavetrace.adapters.recognition.variance_backend import VarianceBackend
from wavetrace.application.ports import RecognitionBackend

_REGISTRY: dict[str, type[RecognitionBackend]] = {
    "mlp": MlpBackend,
    "svm": SvmBackend,
    "variance": VarianceBackend,
    "cnn": CnnBackend,
}

# The backend `train_weapon` picks for each feature_mode when the caller supplies no `ModelConfig`.
DEFAULT_BACKEND_BY_FEATURE_MODE: dict[str, str] = {
    "ic27": "variance",
    "fusion": "mlp",
    "cnn": "cnn",
}


def get_backend_class(name: str) -> type[RecognitionBackend]:
    """Raises ValueError naming the registered backends for an unknown `name`: `ModelConfig.backend`
    accepts any string, so this is the only place a bad name is rejected."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown recognition backend {name!r}; registered backends: {sorted(_REGISTRY)}"
        ) from None


def registered_backend_names() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def is_sklearn_backend(name: str) -> bool:
    """False for an unregistered `name`."""
    try:
        backend_class = get_backend_class(name)
    except ValueError:
        return False
    return issubclass(backend_class, SklearnPipelineBackend)


__all__ = [
    "get_backend_class",
    "registered_backend_names",
    "is_sklearn_backend",
    "DEFAULT_BACKEND_BY_FEATURE_MODE",
]
