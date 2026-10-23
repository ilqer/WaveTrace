"""Directory-based entry points: load from disk, run a `wavetrace.application` use-case, save to disk."""

import os
import sys
import time

from wavetrace.adapters.recognition.session import open_session
from wavetrace.adapters.source import (
    RecordingSource, UdpLinkBatchReceiver, UdpLinkCapture, UdpTimedLinkCapture, save_recording,
)
from wavetrace.adapters.store.calibration import (
    discover_calibrated_node_ids, load_calibration, save_calibration,
)
from wavetrace.adapters.store.dataset import save_dataset
from wavetrace.adapters.store.localization import save_localization
from wavetrace.application.behaviours import (
    DriftMonitoredSource, SegmentVoteBehaviour, build_alert_behaviours,
)
from wavetrace.application.calibrate import calibrate
from wavetrace.application.collect import collect_dataset
from wavetrace.application.collect_camera import CollectCameraUseCase, HeatmapDatasetReport
from wavetrace.application.collect_mesh import CollectMeshUseCase, CollectMode, LinkDatasetRequest
from wavetrace.application.ports import CameraLabelCapture
from wavetrace.application.localize import localize
from wavetrace.application.serve import infer_windows
from wavetrace.application.serve_mesh import ServeMeshUseCase, ServeMode
from wavetrace.domain.calibration import Calibration
from wavetrace.domain.contracts import (
    DEFAULT_HOP_FRAMES, DEFAULT_TARGET_SAMPLE_RATE_HZ, DEFAULT_WINDOW_FRAMES,
)
from wavetrace.domain.groundtruth.dataset import build_stacked_dataset
from wavetrace.domain.guard import DriftMonitor
from wavetrace.domain.localize import Localizer
from wavetrace.domain.recognition.vote import SegmentVoter
from wavetrace.domain.tracking import Tracker
from wavetrace.recognition.camera_collect_modes import CameraPresenceMode, CameraWeaponMode, next_camera_session_id

CHUNK_S = 1.5          # fuse + print at this cadence
LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing


def calibrate_source(source, out_dir, *, baseline_packets=300, use_gain_lock=True, nbvi_max=12):
    """Returns (calibration dir, CalibrationResult)."""
    calibration_result = calibrate(source, Calibration(baseline_packets=baseline_packets, nbvi_max=nbvi_max,
                                                       use_gain_lock=use_gain_lock))
    return save_calibration(calibration_result, out_dir), calibration_result


def collect_source(source, calibration_dir, out_dir, spans, *, stage="presence",
                    window=DEFAULT_WINDOW_FRAMES, hop=DEFAULT_HOP_FRAMES,
                    session_id="", subject_id="", frame_average=1, subtract_baseline=False,
                    subtract_ic_baseline=False, labeler=None, tier=""):
    """Returns (dataset dir, Dataset); the options are `collect_dataset`'s."""
    calibration_result, gain_lock = load_calibration(calibration_dir)
    dataset = collect_dataset(source, calibration_result, gain_lock, spans, stage=stage, window=window,
                              hop=hop, session_id=session_id, subject_id=subject_id,
                              frame_average=frame_average, subtract_baseline=subtract_baseline,
                              subtract_ic_baseline=subtract_ic_baseline, labeler=labeler, tier=tier)
    return save_dataset(dataset, out_dir), dataset


def run_inference(source, calibration_dir, model_path, mode, publisher, *, vote=False, guard=False):
    """mode: 'presence' or 'weapon', picks the head loader. guard adds recalibration advisories
    (O(S) extra per frame) and, in weapon mode only, debounced weapon alerts; vote appends one soft-vote verdict over the
    stream. Returns the published verdicts."""
    calibration_result, gain_lock = load_calibration(calibration_dir)
    session = open_session(mode, model_path)
    behaviours = []
    if guard:
        source = DriftMonitoredSource(source, DriftMonitor(calibration_result.baseline_mag), publisher)
        behaviours += build_alert_behaviours(mode)
    if vote:
        behaviours.append(SegmentVoteBehaviour(SegmentVoter()))
    return infer_windows(source, calibration_result, gain_lock, session, mode, publisher,
                         behaviours=behaviours)


def serve_mesh(mode: ServeMode, port):
    """Prints `mode`'s banner, then one fused verdict per CHUNK_S from the mesh on udp/`port` until
    Ctrl-C."""
    banner = mode.format_banner(port)
    receiver = UdpLinkBatchReceiver(port, timeout_s=0.5)
    print(banner)
    ServeMeshUseCase(mode, clock=time.time, chunk_s=CHUNK_S, link_timeout_s=LINK_TIMEOUT_S,
                     buffer_s=BUFFER_S).run(receiver)


def collect_mesh(mode: CollectMode, *, port, calibration_root, node_id, num_sessions, max_frames_per_link):
    """Captures `num_sessions` sessions of `mode` from every calibrated node (only `node_id` when set)
    on udp/`port`, writes one dataset per usable link, then trains `mode`'s heads. A missing
    calibration is reported on stderr and nothing is captured."""
    node_ids = discover_calibrated_node_ids(calibration_root)
    if not node_ids:
        print(f"\n[ERROR] no per-node calibrations in {calibration_root}/node*. "
              "Run collect_baseline.py first.", file=sys.stderr)
        return
    if node_id is not None:
        if node_id not in node_ids:
            print(f"\n[ERROR] node {node_id} has no calibration in {calibration_root}.", file=sys.stderr)
            return
        node_ids = [node_id]
    os.makedirs(mode.model_root, exist_ok=True)
    capture = UdpLinkCapture(port, node_ids, max_frames_per_link=max_frames_per_link)
    CollectMeshUseCase(
        mode, mode, capture=capture,
        write_link_dataset=lambda request: write_link_dataset(request, f"{calibration_root}/node{request.rx_node_id}"),
        target_sample_rate_hz=DEFAULT_TARGET_SAMPLE_RATE_HZ,
        min_window_frames=DEFAULT_WINDOW_FRAMES).run(node_ids, num_sessions)


def write_link_dataset(request: LinkDatasetRequest, calibration_dir) -> str:
    """Saves the request's recording, then its dataset; returns the dataset dir."""
    save_recording(request.frames, request.recording_dir)
    # built from the saved complex64 recording, so re-collecting it offline reproduces the dataset
    collect_source(RecordingSource(request.recording_dir), calibration_dir, request.dataset_dir, request.spans,
                   stage=request.stage, session_id=request.session_id, subject_id=request.subject_id,
                   labeler=request.labeler, subtract_ic_baseline=request.should_subtract_ic_baseline)
    return request.dataset_dir


def collect_camera(camera: CameraLabelCapture, *, stage, port, root, calibration_dir_by_node_id, subject_id,
                   duration_s, stop, model_root, log, on_capture_end):
    """Records `duration_s` of mesh CSI on udp/`port` from the nodes of `calibration_dir_by_node_id` while
    `camera` labels the scene, as the next free `<subject_id>_cam_s<n>` session under `root`: one
    camera-labelled `stage` dataset per usable link (weapon datasets join <root>/weapon_ds) and one
    all-node heatmap dataset. A `model_root` trains `stage`'s heads and the heatmap head there; None trains
    nothing. `stop` ends the capture early; every message goes to `log`; `on_capture_end` is called once the
    capture ends."""
    node_ids = sorted(calibration_dir_by_node_id)
    session_id = next_camera_session_id(root, subject_id)
    mode_type = CameraWeaponMode if stage == "weapon" else CameraPresenceMode
    mode = mode_type(root=root, session_id=session_id, subject_id=subject_id, model_root=model_root)
    calibrations = {node_id: load_calibration(calibration_dir)
                    for node_id, calibration_dir in calibration_dir_by_node_id.items()}

    def build_heatmap(frames_by_node, labels):
        dataset = build_stacked_dataset([frame for frames in frames_by_node.values() for frame in frames],
                                        {node_id: calibrations[node_id] for node_id in frames_by_node}, labels,
                                        window=DEFAULT_WINDOW_FRAMES,
                                        hop=DEFAULT_HOP_FRAMES, session_id=session_id, subject_id=subject_id)
        save_dataset(dataset, mode.heatmap_dataset_dir)
        return HeatmapDatasetReport(dataset_dir=mode.heatmap_dataset_dir, num_windows=int(dataset.X_image.shape[0]),
                                    num_masked_windows=sum(1 for label in dataset.labels if label.mask))

    CollectCameraUseCase(
        mode, mode if model_root is not None else None, capture=UdpTimedLinkCapture(port, node_ids),
        camera=camera,
        write_link_dataset=lambda request: write_link_dataset(request,
                                                              calibration_dir_by_node_id[request.rx_node_id]),
        build_heatmap=build_heatmap, log=log, on_capture_end=on_capture_end,
        target_sample_rate_hz=DEFAULT_TARGET_SAMPLE_RATE_HZ,
        min_window_frames=DEFAULT_WINDOW_FRAMES).run(node_ids, duration_s, stop)


def localize_source(source, out_dir, *, num_antennas, antenna_spacing_wavelengths=0.5,
                     method="music", num_sources=1, num_angles=181,
                     subcarrier_spacing_hz=312.5e3, max_range_m=12.0, num_ranges=64,
                     range_enabled=True, filter_track=True, publisher=None):
    """Returns (localization dir, aggregate Localization). Needs >= 2 RX antennas (2-antenna ESP32 /
    Pi NIC). filter_track=False publishes the raw per-frame fixes instead of the Kalman track."""
    localizer = Localizer(num_antennas, spacing=antenna_spacing_wavelengths, method=method,
                          num_sources=num_sources, num_angles=num_angles,
                          subcarrier_spacing_hz=subcarrier_spacing_hz, max_range_m=max_range_m,
                          num_ranges=num_ranges, range_enabled=range_enabled)
    tracker = Tracker(range_enabled=range_enabled) if filter_track else None
    room_map = localize(source, localizer, tracker, publisher)
    return save_localization(room_map, out_dir), room_map
