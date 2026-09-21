"""The recognition-backend registry.

Maps a `ModelConfig.backend` key to its `RecognitionBackend` implementation. `PresenceHead` and
`WeaponHead` resolve a backend once — at construction or `load()` — through `get_backend_class`.

Adding a backend means one new module plus one `_REGISTRY` line below.
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

# The backend `trainWeapon` picks by default for each feature_mode when the caller supplies no
# explicit `ModelConfig` — a training-time convenience, not a per-backend property (several
# backends can consume the same feature_mode; this just names the canonical one).
DEFAULT_BACKEND_BY_FEATURE_MODE: dict[str, str] = {
    "ic27": "variance",
    "fusion": "mlp",
    "cnn": "cnn",
}


def get_backend_class(name: str) -> type[RecognitionBackend]:
    """Look up the `RecognitionBackend` implementation registered for `name`.

    This is where an unregistered backend is rejected: `ModelConfig.backend` itself accepts any
    string, so head construction/`load()` calling this is the only place a bad name raises,
    naming the registered backends in the message."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown recognition backend {name!r}; registered backends: {sorted(_REGISTRY)}"
        ) from None


def registered_backend_names() -> tuple[str, ...]:
    """Every backend name currently registered, for enumerating the registry's contents."""
    return tuple(_REGISTRY)


def is_sklearn_backend(name: str) -> bool:
    """True when `name` is registered and its implementation is a `SklearnPipelineBackend`;
    False otherwise, including for an unregistered name."""
    try:
        backend_cls = get_backend_class(name)
    except ValueError:
        return False
    return issubclass(backend_cls, SklearnPipelineBackend)


__all__ = [
    "get_backend_class",
    "registered_backend_names",
    "is_sklearn_backend",
    "DEFAULT_BACKEND_BY_FEATURE_MODE",
]
