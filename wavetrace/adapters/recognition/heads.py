"""Backend resolution: the only place a recognition head is built or loaded from the registry."""

import logging
from pathlib import Path

import joblib

from wavetrace.adapters.recognition import get_backend_class, is_sklearn_backend
from wavetrace.adapters.recognition.variance_backend import VarianceBackend
from wavetrace.adapters.store.model_runs import find_model_dir, resolve_model_path
from wavetrace.domain.contracts import SCHEMA_VERSION, PipelineContract, derive_pipeline_contract
from wavetrace.domain.model_config import ModelConfig
from wavetrace.domain.recognition import VARIANCE_FEATURE, PresenceHead, WeaponHead

# what a head saved before `schema_version` existed is read as
LEGACY_SCHEMA_VERSION = 1


_warned_legacy_paths: set[Path] = set()


class SchemaVersionMismatchError(Exception):
    """A saved head's `schema_version` is not the SCHEMA_VERSION this code reads."""


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
    """`path` may name `<model_dir>/model.joblib` flat; it loads the current run's copy. Raises
    SchemaVersionMismatchError for another schema_version."""
    path = resolve_model_path(path)
    blob = joblib.load(path)
    check_schema_version(blob, path, _format_train_command("presence", path))
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
    """`path` may name `<model_dir>/model.joblib` flat; it loads the current run's copy. Raises
    SchemaVersionMismatchError for another schema_version."""
    path = resolve_model_path(path)
    blob = joblib.load(path)
    check_schema_version(blob, path, _format_train_command("weapon", path))
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


def check_schema_version(blob: dict, path: Path, retrain_command: str) -> None:
    """A blob without `schema_version` reads as LEGACY_SCHEMA_VERSION, warned once per path per process; any
    version but SCHEMA_VERSION raises SchemaVersionMismatchError naming `path`, both versions and
    `retrain_command`."""
    schema_version = blob.get("schema_version")
    if schema_version is None:
        _warn_legacy_once(path)
        schema_version = LEGACY_SCHEMA_VERSION
    if schema_version != SCHEMA_VERSION:
        raise SchemaVersionMismatchError(
            f"{path} has schema_version {schema_version}, but this wavetrace expects {SCHEMA_VERSION}; "
            f"retrain it: {retrain_command}")


def _warn_legacy_once(path: Path) -> None:
    warning_key = path.resolve()  # one key however the path was spelled or linked
    if warning_key in _warned_legacy_paths:
        return
    _warned_legacy_paths.add(warning_key)
    logging.getLogger(__name__).warning("%s has no schema_version; loading it as legacy version %d",
                                        path, LEGACY_SCHEMA_VERSION)


def _format_train_command(stage: str, path: Path) -> str:
    return f"wavetrace train <dataset dirs> --stage {stage} --out {find_model_dir(path.parent)}"
