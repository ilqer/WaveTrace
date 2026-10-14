"""Backend resolution: the only place a recognition head is built or loaded from the registry."""

import joblib

from wavetrace.adapters.recognition import get_backend_class, is_sklearn_backend
from wavetrace.adapters.recognition.variance_backend import VarianceBackend
from wavetrace.config import ModelConfig
from wavetrace.domain.contracts import PipelineContract, derive_pipeline_contract
from wavetrace.domain.recognition import VARIANCE_FEATURE, PresenceHead, WeaponHead


def _variance_backend_options(backend_class: type, variance_feature: int) -> dict:
    """Only `VarianceBackend` accepts `variance_feature`."""
    if backend_class is VarianceBackend:
        return {"variance_feature": variance_feature}
    return {}


def build_presence_head(config: ModelConfig) -> PresenceHead:
    if not is_sklearn_backend(config.backend):
        raise ValueError(f"PresenceHead supports 'mlp'/'svm', not {config.backend!r}")
    backend = get_backend_class(config.backend)(config)
    return PresenceHead(config, backend)


def build_weapon_head(config: ModelConfig, *, variance_feature: int = VARIANCE_FEATURE) -> WeaponHead:
    backend_class = get_backend_class(config.backend)
    backend = backend_class(config, **_variance_backend_options(backend_class, variance_feature))
    return WeaponHead(config, backend, variance_feature=variance_feature)


def load_presence_head(path) -> PresenceHead:
    blob = joblib.load(path)
    config = ModelConfig(**blob["config"])
    if not is_sklearn_backend(config.backend):
        raise ValueError(f"PresenceHead supports 'mlp'/'svm', not {config.backend!r}")
    backend = get_backend_class(config.backend).load(blob, config)
    # older artifacts predate PipelineContract: derive it the same way `save` does, so every
    # model already on disk keeps loading unchanged
    contract = (PipelineContract.from_dict(blob["contract"]) if "contract" in blob
                else derive_pipeline_contract(config))
    return PresenceHead.restore(config, backend, contract)


def load_weapon_head(path) -> WeaponHead:
    blob = joblib.load(path)
    config = ModelConfig(**blob["config"])
    variance_feature = blob["variance_feature"]
    backend_class = get_backend_class(config.backend)
    options = _variance_backend_options(backend_class, variance_feature)
    backend = backend_class.load(blob, config, **options)
    feature_mode = blob.get("feature_mode")  # absent in older artifacts
    # older artifacts predate PipelineContract: derive it the same way `save` does, so every
    # model already on disk keeps loading unchanged
    contract = (PipelineContract.from_dict(blob["contract"]) if "contract" in blob
                else derive_pipeline_contract(config))
    return WeaponHead.restore(
        config, backend, contract, feature_mode=feature_mode, variance_feature=variance_feature
    )
