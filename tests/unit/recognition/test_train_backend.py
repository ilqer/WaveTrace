import json

import numpy as np

from wavetrace import Label
from wavetrace.adapters.store.dataset import save_dataset
from wavetrace.adapters.store.model_runs import resolve_run_dir
from wavetrace.domain.groundtruth.dataset import Dataset
from wavetrace.recognition.train import train_presence, train_weapon

NUM_SUBCARRIERS = 2


def _save_two_class_dataset(dataset_dir, seed):
    rng = np.random.default_rng(seed)
    class_ids = np.asarray([0, 1] * 6)
    labels = []
    for class_id in class_ids:
        label = Label()
        label.class_id = int(class_id)
        labels.append(label)
    num_windows = class_ids.size
    dataset = Dataset(
        X_features=(rng.normal(size=(num_windows, 9 * NUM_SUBCARRIERS)) + class_ids[:, None]).astype(np.float32),
        X_image=np.zeros((num_windows, 1, 4), np.float32), y=class_ids, t=np.arange(float(num_windows)),
        labels=labels, meta={"K": NUM_SUBCARRIERS, "window": 48, "hop": 24},
        session_ids=np.asarray(["s0"] * num_windows, dtype=object),
        subject_ids=np.asarray(["u0"] * num_windows, dtype=object),
        X_intercarrier=(rng.normal(size=(num_windows, 27)) + class_ids[:, None]).astype(np.float32))
    return save_dataset(dataset, dataset_dir)


def test_presence_trains_the_chosen_backend_at_the_datasets_cadence(tmp_path):
    head, metrics = train_presence([_save_two_class_dataset(tmp_path / "ds", 1)], out_dir=tmp_path / "m",
                                   backend="svm")
    assert metrics["backend"] == json.loads((resolve_run_dir(tmp_path / "m") / "metrics.json").read_text())["backend"] == "svm"
    assert (head.config.backend, head.config.window, head.config.hop, head.config.k) == ("svm", 48, 24, 2)


def test_presence_without_a_backend_trains_the_default_mlp(tmp_path):
    _, metrics = train_presence([_save_two_class_dataset(tmp_path / "ds", 2)], out_dir=tmp_path / "m")
    assert metrics["backend"] == "mlp"


def test_weapon_trains_the_chosen_backend_instead_of_the_feature_modes_default(tmp_path):
    dataset_dirs = [_save_two_class_dataset(tmp_path / "ds", 3)]
    _, svm_metrics = train_weapon(dataset_dirs, out_dir=tmp_path / "svm", feature_mode="ic27", backend="svm")
    _, default_metrics = train_weapon(dataset_dirs, out_dir=tmp_path / "default", feature_mode="ic27")
    assert (svm_metrics["backend"], default_metrics["backend"]) == ("svm", "variance")
