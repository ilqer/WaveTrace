from pathlib import Path
from types import SimpleNamespace

import pytest

from wavetrace.recognition import collect_modes
from wavetrace.recognition.collect_modes import (
    CountCollectMode, PresenceCollectMode, WeaponCollectMode, WeaponTrainingPolicy, default_model_root,
    find_collected_max_count,
)


def _frames(*timestamps):
    return [SimpleNamespace(timestamp=timestamp) for timestamp in timestamps]


def _metrics(logo=None):
    return {"n_samples": 10, "class_counts": {"0": 4, "1": 6}, "train_accuracy": 0.9, "logo": logo or {}}


class _RecordedTraining:
    """Stands in for train_presence / train_weapon; raises `error` for the out_dirs it lists."""

    def __init__(self, metrics, failing_out_dirs=()):
        self._metrics = metrics
        self._failing_out_dirs = failing_out_dirs
        self.calls = []

    def __call__(self, dataset_dirs, *, out_dir, run_provenance, **options):
        self.calls.append((list(dataset_dirs), out_dir, options))
        if out_dir in self._failing_out_dirs:
            raise ValueError("needs both classes")
        return None, self._metrics


def test_presence_session_captures_empty_then_present_as_one_dataset():
    [[conditions]] = PresenceCollectMode(root="r", model_root="m").plan_session(1, 3)
    assert [(condition.prompt, condition.countdown_s, condition.hint, condition.num_beeps_after_capture)
            for condition in conditions] == [
        ("session 2/3 — part A: keep the zone empty and still.", 5, "hold the condition steady...", 0),
        ("session 2/3 — part B: stand and move in the zone.", 0, "hold the condition steady...", 0)]


def test_presence_joins_both_parts_into_one_recording_labelled_present_over_part_b():
    mode = PresenceCollectMode(root="r", model_root="m")
    [[conditions]] = mode.plan_session(0, 1)
    empty, present = _frames(0.0, 0.5), _frames(10.0, 10.5)
    request = mode.describe_link_dataset(4, conditions, ("4f:9c", 2), [empty, present])
    assert request.frames == empty + present
    assert (request.rx_node_id, request.recording_dir, request.dataset_dir) == (
        2, "r/sess_4/node2/link_4f9c", "r/ds_4/node2/link_4f9c")
    assert request.spans == [(10.0, 11.5)]
    assert (request.stage, request.session_id, request.subject_id) == ("presence", "sess4", "u0")
    assert request.labeler is None and not request.should_subtract_ic_baseline


def test_presence_reports_a_trained_node_with_its_session_logo(monkeypatch):
    training = _RecordedTraining(_metrics({"session": {"accuracy": 0.8, "majority_accuracy": 0.6,
                                                       "tpr": 0.7, "fp_rate": 0.1}}))
    monkeypatch.setattr(collect_modes, "train_presence", training)
    [report] = PresenceCollectMode(root="r", model_root="m").train_node(3, ["d1", "d2"])
    assert training.calls == [(["d1", "d2"], "m/node3", {"backend": None})]
    assert report.line == ("   [OK]   node 3: samples=10 class_counts={'0': 4, '1': 6} train_acc=0.900"
                           "  LOGO=0.800 (majority 0.600, TPR 0.700, FP 0.100)")
    assert report.trained_head_key == 3


def test_presence_skips_a_node_without_datasets():
    [report] = PresenceCollectMode(root="r", model_root="m").train_node(3, [])
    assert (report.line, report.trained_head_key) == ("   [SKIP] node 3: no usable sessions.", None)


def test_count_captures_one_level_per_group_counting_down_only_before_the_empty_zone():
    mode = CountCollectMode(root="r", model_root="m", max_count=2)
    groups = mode.plan_session(0, 2)
    assert [(condition.prompt, condition.countdown_s, condition.num_beeps_after_capture)
            for [[condition]] in groups] == [
        ("session 1/2 — put 0 people in the zone and have them move.", 5, 3),
        ("session 1/2 — put 1 people in the zone and have them move.", 0, 3),
        ("session 1/2 — put 2+ people in the zone and have them move.", 0, 3)]
    assert mode.format_banner([1, 2]) == "training nodes: [1, 2]; count classes: ['0', '1', '2+']"


def test_count_labels_every_window_of_a_level_with_that_level():
    mode = CountCollectMode(root="r", model_root="m", max_count=2)
    [conditions] = mode.plan_session(1, 2)[2]
    request = mode.describe_link_dataset(1, conditions, ("64:b8", 1), [_frames(3.0, 4.0)])
    assert (request.recording_dir, request.dataset_dir) == ("r/count_sess_1/c2/node1/link_64b8",
                                                            "r/count_ds_1/c2/node1/link_64b8")
    assert request.spans == [(3.0, 5.0)]
    label = request.labeler.label_at(3.5)
    assert (label.class_id, request.session_id, request.stage) == (2, "sess1", "presence")


def test_count_reports_per_class_recall_under_the_logo_line(monkeypatch):
    confusion = [[2, 0, 0], [1, 1, 0], [0, 0, 0]]
    logo = {"session": {"accuracy": 0.5, "majority_accuracy": 0.4, "confusion": confusion}}
    monkeypatch.setattr(collect_modes, "train_presence", _RecordedTraining(_metrics(logo)))
    [report] = CountCollectMode(root="r", model_root="m", max_count=2).train_node(1, ["d"])
    assert report.line == ("   [OK]   node 1: samples=10 class_counts={'0': 4, '1': 6} train_acc=0.900"
                           "  LOGO=0.500 (majority 0.400)\n"
                           "          per-class: {'0': '100%', '1': '50%', '2+': 'n/a'}")


def _weapon_mode(**options):
    return WeaponCollectMode(**{"root": "r", "model_root": "m", "subject_id": "p1", "carry": "chest",
                                "should_subtract_ic_baseline": True, "is_per_link": False, **options})


def test_weapon_captures_clear_then_armed_as_one_group_of_two_datasets():
    [group] = _weapon_mode().plan_session(2, 3)
    assert [(condition.prompt, condition.countdown_s, condition.hint) for [condition] in group] == [
        ("session 3/3 — stand still, no weapon on you.", 5, "hold the condition steady (stand STILL)..."),
        ("session 3/3 — stand still, weapon concealed on you.", 5,
         "hold the condition steady (stand STILL)...")]


def test_weapon_marks_only_the_armed_condition_and_names_datasets_by_subject_carry_and_session():
    mode = _weapon_mode()
    [(clear_conditions, armed_conditions)] = mode.plan_session(2, 3)
    clear = mode.describe_link_dataset(2, clear_conditions, ("4f:9c", 1), [_frames(0.0, 1.0)])
    armed = mode.describe_link_dataset(2, armed_conditions, ("4f:9c", 1), [_frames(5.0, 6.0)])
    assert (clear.spans, armed.spans) == ([], [(5.0, 7.0)])
    assert (armed.recording_dir, armed.dataset_dir) == ("r/weapon_rec/p1_chest_s2/weapon/node1/link_4f9c",
                                                        "r/weapon_ds/node1/p1_chest_s2_weapon_link4f9c")
    assert (armed.stage, armed.session_id, armed.subject_id) == ("weapon", "p1_chest_s2", "p1")
    assert armed.should_subtract_ic_baseline and armed.labeler is None


def test_weapon_per_link_trains_one_head_per_tag_over_the_whole_pool(monkeypatch, tmp_path):
    for name in ("p0_na_s0_clear_link64b8", "p0_na_s0_weapon_link4f9c", "p1_na_s0_clear_link4f9c"):
        (tmp_path / "weapon_ds" / "node1" / name).mkdir(parents=True)
    root = str(tmp_path)
    logo = {"subject": {"accuracy": 0.7, "majority_accuracy": 0.5, "tpr": 0.6, "fp_rate": 0.2},
            "carry": {"accuracy": 0.55, "majority_accuracy": 0.5}}
    training = _RecordedTraining(_metrics(logo), failing_out_dirs=("m/node1/link64b8",))
    monkeypatch.setattr(collect_modes, "train_weapon", training)
    reports = list(_weapon_mode(root=root, is_per_link=True).train_node(1, ["ignored: the pool is globbed"]))
    pool = f"{root}/weapon_ds/node1"
    assert training.calls == [
        ([f"{pool}/p0_na_s0_weapon_link4f9c", f"{pool}/p1_na_s0_clear_link4f9c"], "m/node1/link4f9c",
         {"feature_mode": "ic27", "backend": None, "report": None}),
        ([f"{pool}/p0_na_s0_clear_link64b8"], "m/node1/link64b8",
         {"feature_mode": "ic27", "backend": None, "report": None})]
    assert [(report.line, report.trained_head_key) for report in reports] == [
        ("   [OK]   Node 1 link4f9c: samples=10 class_counts={'0': 4, '1': 6} train_acc=0.900"
         "  LOGO=0.700 (majority 0.500, TPR 0.600, FP 0.200)  carry-LOGO=0.550 (maj 0.500)", (1, "4f9c")),
        ("   [SKIP] Node 1 link64b8: needs both classes", None)]


def test_weapon_per_link_skips_a_node_whose_pool_has_only_per_node_datasets(tmp_path):
    (tmp_path / "weapon_ds" / "node1" / "p0_na_s0_clear").mkdir(parents=True)
    [report] = _weapon_mode(root=str(tmp_path), is_per_link=True).train_node(1, [])
    assert report.line == "   [SKIP] node 1: no per-link datasets, re-capture with collect_weapon."


def test_weapon_per_node_skips_an_empty_pool(tmp_path):
    [report] = _weapon_mode(root=str(tmp_path)).train_node(2, [])
    assert (report.line, report.trained_head_key) == ("   [SKIP] Node 2: no datasets in pool.", None)


def test_weapon_summary_names_the_head_unit_and_its_destination():
    assert _weapon_mode(is_per_link=True).format_trained_summary([(1, "4f9c"), (2, "64b8")]) == (
        "\nweapon models saved (2 per-link (tx->rx) heads) -> m/node*/link*/  "
        "(LOGO must clearly beat majority; expect this to need many subjects/positions/objects.)")
    assert _weapon_mode().format_training_banner() == (
        "\ntraining per-node weapon models (ic27) on the cumulative pool...")


def _make_dirs(directories):
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


_LINK_KEYS = [("4f:9c", 1), ("64:b8", 1), ("4f:9c", 2), ("4f:9c", 10)]


def _write_decoys(root):
    """One dataset dir of every other pool, and a malformed node dir."""
    _make_dirs([f"{root}/count_ds_0/c1/node1/link_4f9c", f"{root}/weapon_ds/node1/p0_na_s0_clear_link4f9c",
                f"{root}/cam_ds/presence/node1/link_4f9c/cam_cam_s0", f"{root}/ds_0/nodeX/link_4f9c"])


def test_presence_finds_exactly_the_datasets_it_writes_grouped_by_node(tmp_path):
    root = str(tmp_path)
    mode = PresenceCollectMode(root=root, model_root="m")
    for session_index in range(2):
        [[conditions]] = mode.plan_session(session_index, 2)
        for link_key in _LINK_KEYS:
            request = mode.describe_link_dataset(session_index, conditions, link_key,
                                                 [_frames(0.0, 1.0), _frames(5.0, 6.0)])
            _make_dirs([request.dataset_dir, request.recording_dir])
    _write_decoys(root)
    assert mode.find_dataset_dirs_by_node() == {
        1: [f"{root}/ds_0/node1/link_4f9c", f"{root}/ds_0/node1/link_64b8", f"{root}/ds_1/node1/link_4f9c",
            f"{root}/ds_1/node1/link_64b8"],
        2: [f"{root}/ds_0/node2/link_4f9c", f"{root}/ds_1/node2/link_4f9c"],
        10: [f"{root}/ds_0/node10/link_4f9c", f"{root}/ds_1/node10/link_4f9c"]}


def test_count_finds_exactly_the_datasets_it_writes_and_the_top_level_collected(tmp_path):
    root = str(tmp_path)
    mode = CountCollectMode(root=root, model_root="m", max_count=3)
    for count, [conditions] in enumerate(mode.plan_session(0, 1)):
        for link_key in _LINK_KEYS[:3]:
            request = mode.describe_link_dataset(0, conditions, link_key, [_frames(0.0, 1.0)])
            _make_dirs([request.dataset_dir, request.recording_dir])
    _write_decoys(root)
    dataset_dirs_by_node = mode.find_dataset_dirs_by_node()
    assert sorted(dataset_dirs_by_node) == [1, 2]
    assert dataset_dirs_by_node[2] == [f"{root}/count_ds_0/c{count}/node2/link_4f9c" for count in range(4)]
    assert len(dataset_dirs_by_node[1]) == 8
    assert find_collected_max_count(root) == 3


def test_no_count_level_collected_is_max_count_zero(tmp_path):
    assert find_collected_max_count(str(tmp_path)) == 0


def test_the_weapon_pool_finds_every_dataset_written_into_it(tmp_path):
    root = str(tmp_path)
    mode = _weapon_mode(root=root)
    [group] = mode.plan_session(0, 1)
    written = []
    for [condition] in group:
        for link_key in _LINK_KEYS[:3]:
            written.append(mode.describe_link_dataset(0, (condition,), link_key, [_frames(0.0, 1.0)]).dataset_dir)
    _make_dirs(written + [f"{root}/weapon_ds/node2/p0_cam_s0_link64b8", f"{root}/ds_0/node1/link_4f9c"])
    assert WeaponTrainingPolicy(root=root, model_root="m", is_per_link=True).find_dataset_dirs_by_node() == {
        1: [f"{root}/weapon_ds/node1/p1_chest_s0_clear_link4f9c", f"{root}/weapon_ds/node1/p1_chest_s0_clear_link64b8",
            f"{root}/weapon_ds/node1/p1_chest_s0_weapon_link4f9c", f"{root}/weapon_ds/node1/p1_chest_s0_weapon_link64b8"],
        2: [f"{root}/weapon_ds/node2/p0_cam_s0_link64b8", f"{root}/weapon_ds/node2/p1_chest_s0_clear_link4f9c",
            f"{root}/weapon_ds/node2/p1_chest_s0_weapon_link4f9c"]}


@pytest.mark.parametrize(("backend", "feature_mode"), [(None, "ic27"), ("variance", "ic27"), ("svm", "ic27"),
                                                        ("cnn", "cnn")])
def test_the_weapon_backend_picks_the_feature_mode_and_reaches_training(monkeypatch, tmp_path, backend, feature_mode):
    (tmp_path / "weapon_ds" / "node1" / "p0_na_s0_clear").mkdir(parents=True)
    training = _RecordedTraining(_metrics())
    monkeypatch.setattr(collect_modes, "train_weapon", training)
    policy = WeaponTrainingPolicy(root=str(tmp_path), model_root="m", is_per_link=False, backend=backend,
                                  report=print)
    [report] = policy.train_node(1, policy.find_dataset_dirs_by_node()[1])
    assert training.calls == [([f"{tmp_path}/weapon_ds/node1/p0_na_s0_clear"], "m/node1",
                               {"feature_mode": feature_mode, "backend": backend, "report": print})]
    assert (report.trained_head_key, report.metrics) == (1, _metrics())
    assert policy.format_training_banner() == (
        f"\ntraining per-node weapon models ({feature_mode}) on the cumulative pool...")


def test_an_unregistered_weapon_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown recognition backend 'ic27'"):
        WeaponTrainingPolicy(root="r", model_root="m", is_per_link=True, backend="ic27")


def test_presence_and_count_train_with_the_chosen_backend_and_report_the_metrics(monkeypatch):
    training = _RecordedTraining(_metrics())
    monkeypatch.setattr(collect_modes, "train_presence", training)
    [presence_report] = PresenceCollectMode(root="r", model_root="m", backend="svm").train_node(1, ["d"])
    [count_report] = CountCollectMode(root="r", model_root="mc", max_count=2, backend="svm").train_node(2, ["e"])
    assert training.calls == [(["d"], "m/node1", {"backend": "svm"}), (["e"], "mc/node2", {"backend": "svm"})]
    assert presence_report.metrics == count_report.metrics == _metrics()


def test_each_stage_trains_into_its_scripts_default_model_root():
    assert [default_model_root("r", stage) for stage in ("presence", "count", "weapon")] == [
        "r/model", "r/model_count", "r/model_weapon"]


def test_the_weapon_policy_trains_the_datasets_it_is_given_one_link_head_at_a_time(monkeypatch, tmp_path):
    training = _RecordedTraining(_metrics())
    monkeypatch.setattr(collect_modes, "train_weapon", training)
    policy = WeaponTrainingPolicy(root=str(tmp_path), model_root="m", is_per_link=True)
    heads = policy.train_node(1, ["x/p0_na_s0_clear_link4f9c", "x/p0_na_s0_weapon_link4f9c", "x/p0_na_s0_clear_link64b8"])
    next(heads)
    assert [(dataset_dirs, out_dir) for dataset_dirs, out_dir, _ in training.calls] == [
        (["x/p0_na_s0_clear_link4f9c", "x/p0_na_s0_weapon_link4f9c"], "m/node1/link4f9c")]
    next(heads)
    assert training.calls[1][:2] == (["x/p0_na_s0_clear_link64b8"], "m/node1/link64b8")
