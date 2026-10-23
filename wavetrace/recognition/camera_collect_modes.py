"""The camera collect modes for `CollectCameraUseCase`, each its `CameraCollectPlan` and, with --train,
its `CameraTrainingPolicy`: where each link's camera-labelled dataset goes, and which heads train on it.
Every run is one session, `<subject>_cam_s<n>`; its recordings go under <root>/cam_rec/<session> and
its stacked heatmap dataset under <root>/cam_ds/heatmap/<session>."""

import glob
from pathlib import Path

from wavetrace.adapters.store.dataset import load_dataset
from wavetrace.application.collect_mesh import HeadTrainingReport, LinkDatasetRequest
from wavetrace.recognition.collect_modes import (
    PresenceCollectMode, WeaponTrainingPolicy, default_model_root, group_dirs_by_node, weapon_pool_dataset_dir,
)
from wavetrace.recognition.heatmap import train_heatmap
from wavetrace.recognition.weapon_serving import make_link_tag

_CAMERA_CARRY = "cam"  # the carry position camera sessions stand for in the weapon pool's carry LOGO


def default_camera_model_root(root, stage) -> str:
    """Where a camera session's `stage` heads and its heatmap head train unless a model root is given: with
    collect_weapon's heads for 'weapon' (its datasets join that pool), <root>/model_camera otherwise."""
    return default_model_root(root, "weapon") if stage == "weapon" else f"{root}/model_camera"


def _format_session_id(subject_id, session_index):
    return f"{subject_id}_{_CAMERA_CARRY}_s{session_index}"


def _recording_dir(root, session_id):
    return f"{root}/cam_rec/{session_id}"


def _heatmap_dataset_dir(root, session_id):
    return f"{root}/cam_ds/heatmap/{session_id}"


def _presence_dataset_dir(root, rx_node_id, tag, session_id):
    return f"{root}/cam_ds/presence/node{rx_node_id}/link_{tag}/{session_id}"


def _weapon_pool_dataset_name(session_id, tag):
    return f"{session_id}_link{tag}"


def find_heatmap_capture_root(dataset_dir) -> str | None:
    """The capture-profile root of a camera session's heatmap dataset, <root>/cam_ds/heatmap/<session>; None
    for any other dir."""
    session_dir = Path(dataset_dir)
    if len(session_dir.parents) < 3:
        return None
    root = session_dir.parents[2]
    return str(root) if Path(_heatmap_dataset_dir(root, session_dir.name)) == session_dir else None


def next_camera_session_id(root, subject_id) -> str:
    """The lowest-index session of `subject_id` with nothing of it left under `root` (recordings, heatmap,
    presence or weapon-pool datasets), so runs accumulate and never overwrite. O(sessions·links) globbing."""
    escaped_root = glob.escape(root)
    session_index = 0
    while True:
        session_id = _format_session_id(subject_id, session_index)
        escaped_session_id = glob.escape(session_id)
        patterns = (_recording_dir(escaped_root, escaped_session_id),
                    _heatmap_dataset_dir(escaped_root, escaped_session_id),
                    _presence_dataset_dir(escaped_root, "*", "*", escaped_session_id),
                    weapon_pool_dataset_dir(escaped_root, "*", _weapon_pool_dataset_name(escaped_session_id, "*")))
        if not any(glob.glob(pattern) for pattern in patterns):
            return session_id
        session_index += 1


def train_heatmap_head(dataset_dir, model_root, *, report=None) -> HeadTrainingReport:
    """Trains the heatmap head on one stacked camera dataset into <model_root>/heatmap.joblib, or skips a
    dataset without camera masks. report: the per-epoch callback."""
    head_path = f"{model_root}/heatmap.joblib"
    dataset = load_dataset(dataset_dir)
    try:
        head, num_masked_windows = train_heatmap(dataset, report=report)
    except ValueError as error:  # no window has a camera mask
        return HeadTrainingReport(f"   [SKIP] heatmap: {error}", None)
    head.save(head_path)
    return HeadTrainingReport(f"   [OK]   heatmap ({head.grid}x{head.grid}, {num_masked_windows} masks) -> "
                              f"{head_path}", "heatmap", {"grid": head.grid, "n": num_masked_windows})


class CameraPresenceTrainingPolicy:
    """Trains one presence head per node on the given datasets of its camera pool (links and sessions under
    <root>/cam_ds/presence/node<id>) into `model_root`. backend None trains `train_presence`'s default."""

    def __init__(self, *, root, model_root, backend=None):
        self._root = root
        self.model_root = model_root
        self._presence_training = PresenceCollectMode(root=root, model_root=model_root, backend=backend)

    def find_dataset_dirs_by_node(self) -> dict[int, list[str]]:
        return group_dirs_by_node(self._root, _presence_dataset_dir(glob.escape(self._root), "*", "*", "*"))

    def find_node_dataset_dirs(self, node_id) -> list[str]:
        return sorted(glob.glob(_presence_dataset_dir(glob.escape(self._root), node_id, "*", "*")))

    def format_training_banner(self):
        return self._presence_training.format_training_banner()

    def train_node(self, node_id, dataset_dirs):
        return self._presence_training.train_node(node_id, dataset_dirs)

    def format_trained_summary(self, trained_head_keys):
        return self._presence_training.format_trained_summary(trained_head_keys)

    def format_untrained_error(self):
        return self._presence_training.format_untrained_error()


class CameraPresenceMode(CameraPresenceTrainingPolicy):
    """One presence dataset per link at <root>/cam_ds/presence/node<rx>/link_<tag>/<session>; each node
    retrains on its whole camera pool (every link and session), not just this session's datasets."""

    positive_class_name = "present"

    def __init__(self, *, root, session_id, subject_id, model_root):
        super().__init__(root=root, model_root=model_root)
        self._session_id = session_id
        self._subject_id = subject_id
        self.heatmap_dataset_dir = _heatmap_dataset_dir(root, session_id)

    def describe_link_dataset(self, link_key, frames, labels):
        tx_short, rx_node_id = link_key
        tag = make_link_tag(tx_short)
        return LinkDatasetRequest(
            frames=frames, rx_node_id=rx_node_id,
            recording_dir=f"{_recording_dir(self._root, self._session_id)}/node{rx_node_id}/link_{tag}",
            dataset_dir=_presence_dataset_dir(self._root, rx_node_id, tag, self._session_id), spans=[],
            stage="presence", session_id=self._session_id, subject_id=self._subject_id, labeler=labels,
            should_subtract_ic_baseline=False)

    def train_node(self, node_id, dataset_dirs):
        return super().train_node(node_id, self.find_node_dataset_dirs(node_id))

    def train_heatmap(self, dataset_dir):
        return train_heatmap_head(dataset_dir, self.model_root)


class CameraWeaponMode(WeaponTrainingPolicy):
    """One weapon dataset per link into collect_weapon.py's cumulative pool, as
    <root>/weapon_ds/node<rx>/<session>_link<tag>; each node retrains exactly as collect_weapon does (ic27, one
    head per node on its whole pool) into `model_root`."""

    positive_class_name = "weapon"

    def __init__(self, *, root, session_id, subject_id, model_root):
        super().__init__(root=root, model_root=model_root, is_per_link=False)
        self._session_id = session_id
        self._subject_id = subject_id
        self.heatmap_dataset_dir = _heatmap_dataset_dir(root, session_id)

    def describe_link_dataset(self, link_key, frames, labels):
        tx_short, rx_node_id = link_key
        tag = make_link_tag(tx_short)
        # subtracting the ic baseline takes the quiet-room level out of σ²[p]
        return LinkDatasetRequest(
            frames=frames, rx_node_id=rx_node_id,
            recording_dir=f"{_recording_dir(self._root, self._session_id)}/node{rx_node_id}/link_{tag}",
            dataset_dir=weapon_pool_dataset_dir(self._root, rx_node_id,
                                                _weapon_pool_dataset_name(self._session_id, tag)),
            spans=[], stage="weapon", session_id=self._session_id, subject_id=self._subject_id, labeler=labels,
            should_subtract_ic_baseline=True)

    def train_node(self, node_id, dataset_dirs):
        return super().train_node(node_id, self.find_node_dataset_dirs(node_id))

    def train_heatmap(self, dataset_dir):
        return train_heatmap_head(dataset_dir, self.model_root)
