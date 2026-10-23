import json

import joblib
import numpy as np
import pytest

from wavetrace import Label
from wavetrace.adapters.store.dataset import save_dataset
from wavetrace.cli import main
from wavetrace.domain.groundtruth.dataset import Dataset

NUM_WINDOWS_PER_CLASS = 6  # the calibrated svm needs 5 per class in every LOGO fold


def _save_weapon_dataset(dataset_dir):
    rng = np.random.default_rng(0)
    class_ids = np.asarray([0, 1] * NUM_WINDOWS_PER_CLASS)
    labels = []
    for class_id in class_ids:
        label = Label()
        label.class_id = int(class_id)
        labels.append(label)
    num_windows = class_ids.size
    save_dataset(Dataset(
        X_features=(rng.normal(size=(num_windows, 18)) + class_ids[:, None]).astype(np.float32),
        X_image=(rng.normal(size=(num_windows, 4, 8)) + class_ids[:, None, None]).astype(np.float32),
        y=class_ids, t=np.arange(float(num_windows)), labels=labels,
        meta={"K": 2, "window": 32, "hop": 16, "frame_average": 2, "subtract_ic_baseline": True},
        session_ids=np.asarray(["s0"] * num_windows), subject_ids=np.asarray(["u0"] * num_windows),
        X_intercarrier=(rng.normal(size=(num_windows, 27)) + class_ids[:, None]).astype(np.float32)), dataset_dir)


@pytest.mark.parametrize(("backend_args", "expected_backend"), [
    (["--backend", "svm"], "svm"), ([], "variance"), (["--backend", ""], "variance")])
def test_weapon_training_takes_the_backend_flag_or_the_feature_modes_default(tmp_path, backend_args,
                                                                             expected_backend):
    _save_weapon_dataset(tmp_path / "ds")
    model_dir = tmp_path / "model"
    assert main(["train", str(tmp_path / "ds"), "--out", str(model_dir), "--stage", "weapon", *backend_args]) == 0
    assert joblib.load(model_dir / "model.joblib")["config"] == {
        "stage": "weapon", "k": 2, "backend": expected_backend, "window": 32, "hop": 16, "fs_tol": 0.10,
        "hidden": 32, "seed": 0, "frame_average": 2, "subtract_baseline": False, "subtract_ic_baseline": True}
    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert (metrics["backend"], metrics["feature_mode"], metrics["k"]) == (expected_backend, "ic27", 2)
