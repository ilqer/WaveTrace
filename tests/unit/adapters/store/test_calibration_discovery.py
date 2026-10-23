import os

from wavetrace.adapters.store.calibration import discover_calibrated_node_ids, find_node_calibration_dirs


def test_discovers_numbered_node_entries_sorted_including_files(tmp_path):
    for name in ("node10", "node2", "node02x", "nodes", "cal_node3"):
        (tmp_path / name).mkdir()
    (tmp_path / "node7").write_text("stray file")
    assert discover_calibrated_node_ids(tmp_path) == [2, 7, 10]


def test_a_missing_root_has_no_calibrated_nodes(tmp_path):
    assert discover_calibrated_node_ids(tmp_path / "missing") == []


def test_finds_each_calibrated_nodes_dir_by_node_id(tmp_path):
    for name in ("node2", "node10", "nodes"):
        (tmp_path / name).mkdir()
    assert find_node_calibration_dirs(str(tmp_path)) == {2: os.path.join(str(tmp_path), "node2"),
                                                         10: os.path.join(str(tmp_path), "node10")}
