"""Backend resolution for `PresenceHead`/`WeaponHead` — the only place recognition heads are built
or loaded from the registry. The heads themselves take an already-resolved backend (constructor
injection); this module is the composition point that resolves one.

`wavetrace.recognition.Model`/`.Weapon` are imported inside each function, not at module load: the
`wavetrace.recognition` package's own `__init__.py` imports `Train.py`/`Infer.py`, which import this
module — a module-level import here of `PresenceHead`/`WeaponHead` would re-enter this
still-loading module whenever `wavetrace.adapters.recognition` is the first thing touched.
"""

from typing import TYPE_CHECKING

import joblib

from wavetrace.adapters.recognition import get_backend_class, is_sklearn_backend
from wavetrace.adapters.recognition.variance_backend import VarianceBackend
from wavetrace.Config import ModelConfig
from wavetrace.domain.contracts import PipelineContract, derive_pipeline_contract

if TYPE_CHECKING:
    from wavetrace.recognition.Model import PresenceHead
    from wavetrace.recognition.Weapon import WeaponHead


def _variance_backend_options(backend_cls: type, variance_feature: int) -> dict:
    """Only `VarianceBackend` accepts `variance_feature`."""
    if backend_cls is VarianceBackend:
        return {"variance_feature": variance_feature}
    return {}


def build_presence_head(config: ModelConfig) -> "PresenceHead":
    """A fresh, unfitted `PresenceHead` for training. Sklearn-only (P6 lock)."""
    from wavetrace.recognition.Model import PresenceHead
    if not is_sklearn_backend(config.backend):
        raise ValueError(f"PresenceHead supports 'mlp'/'svm', not {config.backend!r}")
    backend = get_backend_class(config.backend)(config)
    return PresenceHead(config, backend)


def build_weapon_head(config: ModelConfig, *, variance_feature: int | None = None) -> "WeaponHead":
    """A fresh, unfitted `WeaponHead` for training. `variance_feature` defaults to
    `wavetrace.recognition.Weapon.VARIANCE_FEATURE`."""
    from wavetrace.recognition.Weapon import VARIANCE_FEATURE, WeaponHead
    if variance_feature is None:
        variance_feature = VARIANCE_FEATURE
    backend_cls = get_backend_class(config.backend)
    backend = backend_cls(config, **_variance_backend_options(backend_cls, variance_feature))
    return WeaponHead(config, backend, variance_feature=variance_feature)


def load_presence_head(path) -> "PresenceHead":
    """Load a persisted `PresenceHead`, resolving its backend from the joblib blob's config."""
    from wavetrace.recognition.Model import PresenceHead
    blob = joblib.load(path)
    config = ModelConfig(**blob["config"])
    if not is_sklearn_backend(config.backend):
        raise ValueError(f"PresenceHead supports 'mlp'/'svm', not {config.backend!r}")
    backend = get_backend_class(config.backend).load(blob, config)
    # absent in artifacts saved before PipelineContract existed -> fall back to the same
    # derivation `save` uses, so every existing model keeps loading unchanged.
    contract = (PipelineContract.from_dict(blob["contract"]) if "contract" in blob
                else derive_pipeline_contract(config))
    return PresenceHead.restore(config, backend, contract)


def load_weapon_head(path) -> "WeaponHead":
    """Load a persisted `WeaponHead`, resolving its backend from the joblib blob's config."""
    from wavetrace.recognition.Weapon import WeaponHead
    blob = joblib.load(path)
    config = ModelConfig(**blob["config"])
    variance_feature = blob["variance_feature"]
    backend_cls = get_backend_class(config.backend)
    options = _variance_backend_options(backend_cls, variance_feature)
    backend = backend_cls.load(blob, config, **options)
    feature_mode = blob.get("feature_mode")  # absent in pre-Phase-8 models -> None
    # absent in artifacts saved before PipelineContract existed -> fall back to the same
    # derivation `save` uses, so every existing model keeps loading unchanged.
    contract = (PipelineContract.from_dict(blob["contract"]) if "contract" in blob
                else derive_pipeline_contract(config))
    return WeaponHead.restore(
        config, backend, contract, feature_mode=feature_mode, variance_feature=variance_feature
    )
