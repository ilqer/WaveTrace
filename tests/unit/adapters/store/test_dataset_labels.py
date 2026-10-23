import numpy as np

from wavetrace import Label
from wavetrace.adapters.store.dataset import load_dataset_labels, save_dataset
from wavetrace.domain.groundtruth.dataset import Dataset


def _label(class_id):
    label = Label()
    label.class_id = class_id
    label.name = str(class_id)
    return label


def test_labels_load_from_the_manifest_and_meta_alone(tmp_path):
    class_ids = [0, 1, 1]
    dataset = Dataset(X_features=np.zeros((3, 2), np.float32), X_image=np.zeros((3, 1, 2), np.float32),
                      y=np.asarray(class_ids), t=np.arange(3.0), labels=[_label(i) for i in class_ids],
                      meta={"K": 7}, session_ids=np.asarray(["s0", "s0", "s1"], dtype=object),
                      subject_ids=np.asarray(["u0", "u0", "u1"], dtype=object))
    save_dataset(dataset, tmp_path / "ds")
    (tmp_path / "ds" / "images.npy").unlink()  # never read
    labels = load_dataset_labels(tmp_path / "ds")
    assert labels.y.tolist() == [0, 1, 1] and labels.y.dtype == np.int64
    assert labels.session_ids.tolist() == ["s0", "s0", "s1"] and labels.subject_ids.tolist() == ["u0", "u0", "u1"]
    assert labels.meta == {"K": 7}
