"""PresenceHead/WeaponHead persist a PipelineContract + schema_version on save, load() falls back
to the same derivation for artifacts saved before those keys existed, and Train.py stamps the
dataset's real capture width onto the contract rather than the NBVI-selected subcarrier count."""

from dataclasses import replace

import joblib
import numpy as np

from wavetrace import Label
from wavetrace.Config import ModelConfig
from wavetrace.domain.contracts import SCHEMA_VERSION, derive_pipeline_contract
from wavetrace.groundtruth import Dataset, saveDataset
from wavetrace.recognition import trainPresence, trainWeapon
from wavetrace.recognition.Model import PresenceHead
from wavetrace.recognition.Weapon import WeaponHead


def _presence_blobs(n=120, k=2, seed=0):
    """Two trivially separable feature clusters (per-class mean shift) — mirrors TestRecognition._blobs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 9 * k)).astype(np.float32)
    y = (np.arange(n) % 2).astype(np.int64)
    X[y == 1] += 3.0
    return X, y


def _weapon_variance_blobs(n=120, seed=0):
    """27-wide inter-carrier blocks with the VARIANCE_FEATURE column (index 9) separable by class."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 27)).astype(np.float32)
    y = (np.arange(n) % 2).astype(np.int64)
    X[y == 1, 9] -= 3.0  # metal lowers variance -> weapon class sits below the no-weapon class
    return X, y


def _labels(names, y):
    labels = []
    for i, (name, class_id) in enumerate(zip(names, y)):
        lab = Label()
        lab.class_id = int(class_id)
        lab.name = name
        lab.timestamp = float(i)
        labels.append(lab)
    return labels


def _presence_dataset(num_subcarriers, k=2, n=120, seed=0):
    """A saveable presence Dataset whose meta['num_subcarriers'] differs from k, so a test can
    tell "real capture width" apart from "NBVI-selected count"."""
    X, y = _presence_blobs(n=n, k=k, seed=seed)
    names = ["present" if c else "absent" for c in y]
    meta = {"K": k, "window": 64, "hop": 16, "num_subcarriers": num_subcarriers}
    return Dataset(
        X_features=X, X_image=np.zeros((n, 1, 1), np.float32), y=y,
        t=np.arange(n, dtype=np.float64), labels=_labels(names, y), meta=meta,
        session_ids=np.full(n, "s0", dtype=object), subject_ids=np.full(n, "u0", dtype=object),
    )


def _weapon_dataset(num_subcarriers, k=12, n=120, seed=0):
    """A saveable weapon Dataset (ic27 feature mode) with meta['num_subcarriers'] != k."""
    X, y = _weapon_variance_blobs(n=n, seed=seed)
    names = ["weapon" if c else "no_weapon" for c in y]
    meta = {"K": k, "window": 64, "hop": 8, "num_subcarriers": num_subcarriers}
    return Dataset(
        X_features=np.zeros((n, 9 * k), np.float32), X_image=np.zeros((n, 1, 1), np.float32), y=y,
        t=np.arange(n, dtype=np.float64), labels=_labels(names, y), meta=meta,
        session_ids=np.full(n, "s0", dtype=object), subject_ids=np.full(n, "u0", dtype=object),
        X_intercarrier=X,
    )


def _strip_contract_keys(path):
    """Rewrite a saved artifact as if it predated PipelineContract (no contract/schema_version keys)."""
    blob = joblib.load(path)
    del blob["contract"]
    del blob["schema_version"]
    joblib.dump(blob, path)
    return path


# ----- PresenceHead ------------------------------------------------------------------------------

def test_presence_head_save_writes_contract_and_schema_version(tmp_path):
    config = ModelConfig(stage="presence", k=2, window=64, hop=16)
    X, y = _presence_blobs(k=2)
    path = PresenceHead(config).fit(X, y).save(tmp_path / "model.joblib")

    blob = joblib.load(path)
    assert blob["schema_version"] == SCHEMA_VERSION
    assert blob["contract"] == derive_pipeline_contract(config).to_dict()
    assert blob["contract"]["window_frames"] == 64
    assert blob["contract"]["hop_frames"] == 16
    # built directly (not via trainPresence/Train.py), so no dataset supplied a real capture width
    assert blob["contract"]["subcarrier_width"] is None


def test_presence_head_load_falls_back_when_contract_absent(tmp_path):
    config = ModelConfig(stage="presence", k=2)
    X, y = _presence_blobs(k=2)
    head = PresenceHead(config).fit(X, y)
    path = head.save(tmp_path / "model.joblib")
    _strip_contract_keys(path)  # simulate a pre-PipelineContract artifact on disk in data/

    loaded = PresenceHead.load(path)
    assert loaded.contract == derive_pipeline_contract(loaded.config)
    assert np.array_equal(loaded.predict(X), head.predict(X))  # backward-compatible: still predicts


# ----- WeaponHead ---------------------------------------------------------------------------------

def test_weapon_head_save_writes_contract_and_schema_version(tmp_path):
    config = ModelConfig(stage="weapon", k=12, backend="variance", window=64, hop=8)
    X, y = _weapon_variance_blobs()
    path = WeaponHead(config).fit(X, y).save(tmp_path / "weapon.joblib")

    blob = joblib.load(path)
    assert blob["schema_version"] == SCHEMA_VERSION
    assert blob["contract"] == derive_pipeline_contract(config).to_dict()
    assert blob["contract"]["window_frames"] == 64
    assert blob["contract"]["hop_frames"] == 8
    # built directly (not via trainWeapon/Train.py), so no dataset supplied a real capture width
    assert blob["contract"]["subcarrier_width"] is None


def test_weapon_head_load_falls_back_when_contract_absent(tmp_path):
    config = ModelConfig(stage="weapon", k=12, backend="variance")
    X, y = _weapon_variance_blobs()
    head = WeaponHead(config).fit(X, y)
    path = head.save(tmp_path / "weapon.joblib")
    _strip_contract_keys(path)  # simulate a pre-PipelineContract artifact on disk in data/

    loaded = WeaponHead.load(path)
    assert loaded.contract == derive_pipeline_contract(loaded.config)
    assert np.array_equal(loaded.predict(X), head.predict(X))  # backward-compatible: still predicts


# ----- save() serializes self.contract, not a fresh re-derivation ---------------------------------

def test_presence_head_resave_preserves_a_stored_contract_that_diverged_from_derivation(tmp_path):
    """Load an artifact whose stored contract differs from derive_pipeline_contract(config) (as
    Train.py's capture-width override now produces), re-save it, and confirm the divergent value
    survives — save() used to call derive_pipeline_contract(self.config) again and silently
    discard it."""
    config = ModelConfig(stage="presence", k=2)
    X, y = _presence_blobs(k=2)
    head = PresenceHead(config).fit(X, y)
    head.contract = replace(head.contract, subcarrier_width=32)  # diverges from config.k == 2
    path = head.save(tmp_path / "model.joblib")
    assert joblib.load(path)["contract"]["subcarrier_width"] == 32

    reloaded = PresenceHead.load(path)
    assert reloaded.contract.subcarrier_width == 32
    resaved = reloaded.save(tmp_path / "resaved.joblib")
    assert joblib.load(resaved)["contract"]["subcarrier_width"] == 32


def test_weapon_head_resave_preserves_a_stored_contract_that_diverged_from_derivation(tmp_path):
    config = ModelConfig(stage="weapon", k=12, backend="variance")
    X, y = _weapon_variance_blobs()
    head = WeaponHead(config).fit(X, y)
    head.contract = replace(head.contract, subcarrier_width=48)  # diverges from config.k == 12
    path = head.save(tmp_path / "weapon.joblib")

    reloaded = WeaponHead.load(path)
    assert reloaded.contract.subcarrier_width == 48
    resaved = reloaded.save(tmp_path / "resaved.joblib")
    assert joblib.load(resaved)["contract"]["subcarrier_width"] == 48


# ----- subcarrier_width is the real capture width, not the NBVI-selected k -------------------------

def test_train_presence_stamps_dataset_capture_width_not_k(tmp_path):
    ds = _presence_dataset(num_subcarriers=32, k=2)
    out = saveDataset(ds, tmp_path / "ds")
    head, _ = trainPresence([out], tmp_path / "models")

    assert head.config.k == 2
    assert head.contract.subcarrier_width == 32
    reloaded = PresenceHead.load(tmp_path / "models" / "model.joblib")
    assert reloaded.contract.subcarrier_width == 32


def test_train_presence_leaves_capture_width_none_when_dataset_predates_it(tmp_path):
    """A dataset with no meta['num_subcarriers'] must report subcarrier_width=None, never silently
    substitute k (the NBVI-selected count) -- the two are different quantities and a reader must be
    able to tell "unknown" apart from either one."""
    ds = _presence_dataset(num_subcarriers=32, k=2)
    del ds.meta["num_subcarriers"]  # simulate a dataset built before this field existed
    out = saveDataset(ds, tmp_path / "ds")
    head, _ = trainPresence([out], tmp_path / "models")

    assert head.contract.subcarrier_width is None
    reloaded = PresenceHead.load(tmp_path / "models" / "model.joblib")
    assert reloaded.contract.subcarrier_width is None


def test_train_weapon_stamps_dataset_capture_width_not_k(tmp_path):
    ds = _weapon_dataset(num_subcarriers=48, k=12)
    out = saveDataset(ds, tmp_path / "ds")
    head, _ = trainWeapon([out], tmp_path / "models")

    assert head.config.k == 12
    assert head.contract.subcarrier_width == 48
    reloaded = WeaponHead.load(tmp_path / "models" / "model.joblib")
    assert reloaded.contract.subcarrier_width == 48
