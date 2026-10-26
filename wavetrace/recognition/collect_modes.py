"""The collect modes for `CollectMeshUseCase`, each both its `CollectPlan` and its `TrainingPolicy`:
what a session captures, how each link's dataset is labelled and laid out, how the heads are
trained and reported, and where each pool's datasets are found again."""

import collections
import glob
import re
from pathlib import Path

import numpy as np

from wavetrace.adapters.recognition import get_backend_class
from wavetrace.adapters.store.run_manifest import RunProvenance
from wavetrace.application.collect_mesh import CaptureCondition, HeadTrainingReport, LinkDatasetRequest
from wavetrace.domain.groundtruth.labeler import ScriptedLabeler
from wavetrace.recognition.count import count_name
from wavetrace.recognition.train import train_presence, train_weapon
from wavetrace.recognition.weapon_serving import make_link_tag, parse_link_tag

SUBJECT_ID = "u0"
COUNTDOWN_S = 5
WEAPON_CLASS_ID = 1
_HOLD_STEADY_HINT = "hold the condition steady..."
_STAND_STILL_HINT = "hold the condition steady (stand STILL)..."
_NO_NODE_TRAINED_ERROR = "\n[ERROR] no node trained — check the boards or run mesh_verify.py."
_MODEL_DIR_NAME_BY_STAGE = {"presence": "model", "count": "model_count", "weapon": "model_weapon"}
_NODE_DIR_PATTERN = re.compile(r"node(\d+)")
_COUNT_LEVEL_DIR_PATTERN = re.compile(r"c(\d+)")


def default_model_root(root, stage) -> str:
    """Where `stage`'s ('presence', 'count' or 'weapon') heads train unless a model root is given."""
    return f"{root}/{_MODEL_DIR_NAME_BY_STAGE[stage]}"


def _link_dir(rx_node_id, tag):
    return f"node{rx_node_id}/link_{tag}"


def _presence_dataset_dir(root, session_index, link_dir):
    return f"{root}/ds_{session_index}/{link_dir}"


def _count_dataset_dir(root, session_index, class_id, link_dir):
    return f"{root}/count_ds_{session_index}/c{class_id}/{link_dir}"


def weapon_pool_dataset_dir(root, rx_node_id, dataset_name) -> str:
    """A dataset's dir in the cumulative weapon pool that collect_weapon and weapon camera sessions both join."""
    return f"{root}/weapon_ds/node{rx_node_id}/{dataset_name}"


def _find_number(relative_dir: Path, pattern: re.Pattern) -> int | None:
    """The number of the first path component `pattern` matches whole."""
    for part in relative_dir.parts:
        match = pattern.fullmatch(part)
        if match:
            return int(match[1])
    return None


def group_dirs_by_node(root, pattern) -> dict[int, list[str]]:
    """Every dir the glob `pattern` matches, grouped by the rx node id of its first node<id> component under
    `root`, each group in path order; a match with no node id is dropped. O(P log P) for P matches."""
    dirs_by_node = collections.defaultdict(list)
    for directory in sorted(glob.glob(pattern)):
        node_id = _find_number(Path(directory).relative_to(root), _NODE_DIR_PATTERN)
        if node_id is not None:
            dirs_by_node[node_id].append(directory)
    return dict(dirs_by_node)


def find_collected_max_count(root) -> int:
    """The highest count level with a dataset under `root`, 0 when there is none. O(P) for P count datasets."""
    pattern = _count_dataset_dir(glob.escape(root), "*", "*", _link_dir("*", "*"))
    levels = (_find_number(Path(directory).relative_to(root), _COUNT_LEVEL_DIR_PATTERN)
              for directory in glob.glob(pattern))
    return max((level for level in levels if level is not None), default=0)


def _compute_label_span(frames):
    return (frames[0].timestamp, frames[-1].timestamp + 1.0)


def _format_trained_line(label, metrics):
    return (f"   [OK]   {label}: samples={metrics['n_samples']} class_counts={metrics['class_counts']} "
            f"train_acc={metrics['train_accuracy']:.3f}")


def _format_binary_logo(logo):
    return (f"  LOGO={logo['accuracy']:.3f} (majority {logo['majority_accuracy']:.3f}, "
            f"TPR {logo.get('tpr', 0):.3f}, FP {logo.get('fp_rate', 0):.3f})")


class PresenceCollectMode:
    """A session's part A and part B join into one recording per link, labelled present over part B.
    backend None trains `train_presence`'s default; run_provenance None starts this policy's own."""

    def __init__(self, *, root, model_root, backend=None, run_provenance=None):
        self._root = root
        self.model_root = model_root
        self._backend = backend
        self._run_provenance = RunProvenance() if run_provenance is None else run_provenance

    def format_banner(self, node_ids):
        return f"training nodes: {node_ids} (calibrated)"

    def plan_session(self, session_index, num_sessions):
        session = f"session {session_index + 1}/{num_sessions}"
        return [((
            CaptureCondition(prompt=f"{session} — part A: keep the zone empty and still.",
                             countdown_s=COUNTDOWN_S, hint=_HOLD_STEADY_HINT, class_id=0, class_name="empty",
                             num_beeps_after_capture=0),
            CaptureCondition(prompt=f"{session} — part B: stand and move in the zone.", countdown_s=0,
                             hint=_HOLD_STEADY_HINT, class_id=1, class_name="present",
                             num_beeps_after_capture=0),
        ),)]

    def describe_link_dataset(self, session_index, conditions, link_key, segments):
        empty_frames, present_frames = segments
        tx_short, rx_node_id = link_key
        link_dir = _link_dir(rx_node_id, make_link_tag(tx_short))
        return LinkDatasetRequest(
            frames=empty_frames + present_frames, rx_node_id=rx_node_id,
            recording_dir=f"{self._root}/sess_{session_index}/{link_dir}",
            dataset_dir=_presence_dataset_dir(self._root, session_index, link_dir),
            spans=[_compute_label_span(present_frames)], stage="presence", session_id=f"sess{session_index}",
            subject_id=SUBJECT_ID, labeler=None, should_subtract_ic_baseline=False)

    def find_dataset_dirs_by_node(self) -> dict[int, list[str]]:
        return group_dirs_by_node(self._root,
                                  _presence_dataset_dir(glob.escape(self._root), "*", _link_dir("*", "*")))

    def format_training_banner(self):
        return "\ntraining per-node presence models..."

    def train_node(self, node_id, dataset_dirs):
        if not dataset_dirs:
            return [HeadTrainingReport(f"   [SKIP] node {node_id}: no usable sessions.", None)]
        _, metrics = train_presence(dataset_dirs, out_dir=f"{self.model_root}/node{node_id}", backend=self._backend,
                                    run_provenance=self._run_provenance)
        line = _format_trained_line(f"node {node_id}", metrics)
        logo = metrics.get("logo", {}).get("session")
        if logo:
            line += _format_binary_logo(logo)
        return [HeadTrainingReport(line, node_id, metrics)]

    def format_trained_summary(self, trained_head_keys):
        return (f"\nmodels saved for nodes {trained_head_keys} -> {self.model_root}/node*/  "
                "(LOGO is the honest number: it must clearly beat the majority baseline.)")

    def format_untrained_error(self):
        return _NO_NODE_TRAINED_ERROR


class CountCollectMode:
    """One group per count level 0..max_count, each recording labelled with its level throughout; the
    top level means 'max_count or more'. backend None trains `train_presence`'s default."""

    def __init__(self, *, root, model_root, max_count, backend=None):
        self._root = root
        self.model_root = model_root
        self._backend = backend
        self._labels = [count_name(count, max_count) for count in range(max_count + 1)]  # index == class_id
        self._run_provenance = RunProvenance()

    def format_banner(self, node_ids):
        return f"training nodes: {node_ids}; count classes: {self._labels}"

    def plan_session(self, session_index, num_sessions):
        session = f"session {session_index + 1}/{num_sessions}"
        return [((CaptureCondition(prompt=f"{session} — put {label} people in the zone and have them move.",
                                   countdown_s=COUNTDOWN_S if count == 0 else 0, hint=_HOLD_STEADY_HINT,
                                   class_id=count, class_name=label,
                                   num_beeps_after_capture=3),),)  # 3 beeps = done, stop moving
                for count, label in enumerate(self._labels)]

    def describe_link_dataset(self, session_index, conditions, link_key, segments):
        [condition], [frames] = conditions, segments
        tx_short, rx_node_id = link_key
        link_dir = _link_dir(rx_node_id, make_link_tag(tx_short))
        span = _compute_label_span(frames)
        class_label = (condition.class_id, condition.class_name)
        labeler = ScriptedLabeler([(span[0], span[1], True)],
                                  label_function=lambda raw, timestamp_s: class_label)
        return LinkDatasetRequest(
            frames=frames, rx_node_id=rx_node_id,
            recording_dir=f"{self._root}/count_sess_{session_index}/c{condition.class_id}/{link_dir}",
            dataset_dir=_count_dataset_dir(self._root, session_index, condition.class_id, link_dir), spans=[span],
            stage="presence",
            session_id=f"sess{session_index}", subject_id=SUBJECT_ID, labeler=labeler,
            should_subtract_ic_baseline=False)

    def find_dataset_dirs_by_node(self) -> dict[int, list[str]]:
        return group_dirs_by_node(self._root,
                                  _count_dataset_dir(glob.escape(self._root), "*", "*", _link_dir("*", "*")))

    def format_training_banner(self):
        return "\ntraining per-node count models..."

    def train_node(self, node_id, dataset_dirs):
        if not dataset_dirs:
            return [HeadTrainingReport(f"   [SKIP] node {node_id}: no usable segments.", None)]
        _, metrics = train_presence(dataset_dirs, out_dir=f"{self.model_root}/node{node_id}", backend=self._backend,
                                    run_provenance=self._run_provenance)
        line = _format_trained_line(f"node {node_id}", metrics)
        logo = metrics.get("logo", {}).get("session")
        if logo:
            line += f"  LOGO={logo['accuracy']:.3f} (majority {logo['majority_accuracy']:.3f})"
        if logo and "confusion" in logo:
            confusion = np.asarray(logo["confusion"])
            row_sums = confusion.sum(axis=1)
            per_class_recall = {self._labels[i]: (f"{confusion[i, i] / row_sums[i]:.0%}" if row_sums[i] > 0
                                                  else "n/a")
                                for i in range(min(len(self._labels), confusion.shape[0]))}
            line += f"\n          per-class: {per_class_recall}"
        return [HeadTrainingReport(line, node_id, metrics)]

    def format_trained_summary(self, trained_head_keys):
        return (f"\ncount models saved for nodes {trained_head_keys} -> {self.model_root}/node*/  "
                "(LOGO must clearly beat the majority baseline, or it's not real.)")

    def format_untrained_error(self):
        return _NO_NODE_TRAINED_ERROR


class WeaponTrainingPolicy:
    """Trains a node's weapon heads on the given datasets of the pool <root>/weapon_ds/node<id>/*: one head per
    node, or with is_per_link one per (tx->rx) direction, each trained only when its report is consumed.
    backend None trains the variance head on the inter-carrier (ic27) block; report is the per-epoch
    callback, which only the cnn backend calls."""

    def __init__(self, *, root, model_root, is_per_link, backend=None, report=None):
        self._root = root
        self.model_root = model_root
        self._is_per_link = is_per_link
        self._backend = backend
        self._feature_mode = "ic27" if backend is None else get_backend_class(backend).feature_mode
        self._report = report
        self._head_unit_label = "per-link (tx->rx)" if is_per_link else "per-node"
        self._run_provenance = RunProvenance()

    def find_dataset_dirs_by_node(self) -> dict[int, list[str]]:
        return group_dirs_by_node(self._root, weapon_pool_dataset_dir(glob.escape(self._root), "*", "*"))

    def find_node_dataset_dirs(self, node_id) -> list[str]:
        return sorted(glob.glob(weapon_pool_dataset_dir(glob.escape(self._root), node_id, "*")))

    def format_training_banner(self):
        return f"\ntraining {self._head_unit_label} weapon models ({self._feature_mode}) on the cumulative pool..."

    def train_node(self, node_id, dataset_dirs):
        if not self._is_per_link:
            yield self._train_head(dataset_dirs, f"{self.model_root}/node{node_id}", f"Node {node_id}", node_id)
            return
        dataset_dirs_by_tag = collections.defaultdict(list)
        for directory in dataset_dirs:
            tag = parse_link_tag(directory)
            if tag is not None:
                dataset_dirs_by_tag[tag].append(directory)
        if not dataset_dirs_by_tag:
            yield HeadTrainingReport(
                f"   [SKIP] node {node_id}: no per-link datasets, re-capture with collect_weapon.", None)
        for tag in sorted(dataset_dirs_by_tag):
            yield self._train_head(dataset_dirs_by_tag[tag], f"{self.model_root}/node{node_id}/link{tag}",
                                   f"Node {node_id} link{tag}", (node_id, tag))

    def _train_head(self, dataset_dirs, out_dir, label, head_key):
        if not dataset_dirs:
            return HeadTrainingReport(f"   [SKIP] {label}: no datasets in pool.", None)
        try:
            _, metrics = train_weapon(dataset_dirs, out_dir=out_dir, feature_mode=self._feature_mode,
                                      backend=self._backend, report=self._report,
                                      run_provenance=self._run_provenance)
        except ValueError as error:  # WeaponHead.fit needs BOTH classes in the pool
            return HeadTrainingReport(f"   [SKIP] {label}: {error}", None)
        logo = metrics.get("logo", {})
        line = _format_trained_line(label, metrics)
        session_logo = logo.get("session") or logo.get("subject")
        if session_logo:
            line += _format_binary_logo(session_logo)
        carry = logo.get("carry")  # does it generalize across carry pose, or ride that confound
        if carry:
            line += f"  carry-LOGO={carry['accuracy']:.3f} (maj {carry['majority_accuracy']:.3f})"
        return HeadTrainingReport(line, head_key, metrics)

    def format_trained_summary(self, trained_head_keys):
        destination = f"{self.model_root}/node*/link*/" if self._is_per_link else f"{self.model_root}/node*/"
        return (f"\nweapon models saved ({len(trained_head_keys)} {self._head_unit_label} heads) -> "
                f"{destination}  (LOGO must clearly beat majority; "
                "expect this to need many subjects/positions/objects.)")

    def format_untrained_error(self):
        return "\n[ERROR] nothing trained — need both no-weapon and weapon captures in the pool."


class WeaponCollectMode(WeaponTrainingPolicy):
    """Each session is a no-weapon then a weapon capture back to back, the subject standing still,
    written as one dataset per condition and link into the cumulative pool. Every node retrains on its
    whole pool, not just this run's datasets, letting subject and carry diversity build up over many
    runs. is_per_link: one head per (tx->rx) direction instead of one per node."""

    def __init__(self, *, root, model_root, subject_id, carry, should_subtract_ic_baseline, is_per_link):
        super().__init__(root=root, model_root=model_root, is_per_link=is_per_link)
        self._subject_id = subject_id
        self._carry = carry
        self._should_subtract_ic_baseline = should_subtract_ic_baseline

    def format_banner(self, node_ids):
        return f"training nodes: {node_ids}  (subject={self._subject_id}, carry={self._carry})"

    def train_node(self, node_id, dataset_dirs):
        return super().train_node(node_id, self.find_node_dataset_dirs(node_id))

    def plan_session(self, session_index, num_sessions):
        session = f"session {session_index + 1}/{num_sessions}"
        conditions = ((0, "clear", "no weapon on you"),
                      (WEAPON_CLASS_ID, "weapon", "weapon concealed on you"))
        return [tuple((CaptureCondition(prompt=f"{session} — stand still, {situation}.",
                                        countdown_s=COUNTDOWN_S, hint=_STAND_STILL_HINT, class_id=class_id,
                                        class_name=class_name, num_beeps_after_capture=0),)
                      for class_id, class_name, situation in conditions)]

    def describe_link_dataset(self, session_index, conditions, link_key, segments):
        [condition], [frames] = conditions, segments
        tx_short, rx_node_id = link_key
        tag = make_link_tag(tx_short)
        session_id = f"{self._subject_id}_{self._carry}_s{session_index}"
        condition_name = condition.class_name
        # subtracting the ic baseline takes the quiet-room level out of σ²[p]
        return LinkDatasetRequest(
            frames=frames, rx_node_id=rx_node_id,
            recording_dir=f"{self._root}/weapon_rec/{session_id}/{condition_name}/{_link_dir(rx_node_id, tag)}",
            dataset_dir=weapon_pool_dataset_dir(self._root, rx_node_id, f"{session_id}_{condition_name}_link{tag}"),
            spans=[_compute_label_span(frames)] if condition.class_id == WEAPON_CLASS_ID else [],
            stage="weapon", session_id=session_id, subject_id=self._subject_id, labeler=None,
            should_subtract_ic_baseline=self._should_subtract_ic_baseline)
