from types import SimpleNamespace

import numpy as np

from wavetrace.diagnostics import dataset_report


def _labels(class_ids, session_id, meta):
    return SimpleNamespace(y=np.asarray(class_ids), session_ids=np.asarray([session_id] * len(class_ids), dtype=object),
                           subject_ids=np.asarray(["u0"] * len(class_ids), dtype=object), meta=meta)


def test_the_report_pools_every_dataset_and_takes_the_shape_from_the_first():
    report = dataset_report([_labels([0, 0, 1], "s0", {"K": 12, "K_img": 28, "num_nodes": 1}),
                             _labels([1, 2], "s1", {"K": 99})])
    assert report["n_samples"] == 5
    assert report["distribution"] == report["class_counts"] == {"0": 2, "1": 2, "2": 1}
    assert report["balance"] == 0.5
    assert (report["sessions"], report["subjects"]) == (["s0", "s1"], ["u0"])
    assert (report["K"], report["K_img"], report["num_nodes"], report["sync_error"]) == (12, 28, 1, {})
