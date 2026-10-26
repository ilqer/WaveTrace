"""Webcam frame source for camera-supervised CSI labeling, with timestamps on the CSI wall clock so
`build_dataset`'s align step can match camera Labels to CSI windows.

Capture is split from inference on purpose: `record_frames` grabs (timestamp, RGB) cheaply into a
buffer during the live CSI capture, then `stream_labels` runs YOLO OFFLINE over that buffer. That
keeps the capture loop real-time (no per-frame model latency).

  Input:  webcam index (or an injected grab fn) + a Labeler.
  Output: list[Label] timestamped on the CSI wall clock → pass as collect_source(labeler=...).

`WebcamLabelCapture` is the camera-collect adapter: YOLO-seg labels each webcam frame live.
"""

import select
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from wavetrace.application.ports import CameraLabelCapture
from wavetrace.domain.groundtruth.labeler import presence_label_function, weapon_label_function
from wavetrace.groundtruth.yolo_labeler import YoloLabelerOptions, YoloSegLabeler


@dataclass(frozen=True, slots=True)
class WebcamOptions:
    index: int = 0
    width: int = 1280
    height: int = 720

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be >= 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")


_PIPE_POLL_S = 0.1
_TERMINATE_GRACE_S = 2.0


class WebcamCapture:
    """ffmpeg-based webcam wrapper yielding (timestamp_s, RGB ndarray).
    Uses ffmpeg subprocess for capture (avoids macOS AVFoundation run-loop
    segfault when cv2.VideoCapture is called from a background thread).
    On macOS, ffmpeg triggers the system permission dialog on first run—no
    manual Terminal camera grant needed.
    `clock` is injectable (defaults to wall-clock `time.time`, matching the CSI
    ntp_ms stamp so labels and CSI windows align). Once `stop` is set a read returns None within
    _PIPE_POLL_S, even from a stalled ffmpeg. Use as a context manager: it owns the ffmpeg process."""

    def __init__(self, options: WebcamOptions = WebcamOptions(), *, clock=time.time,
                 stop: threading.Event | None = None):
        self._index = options.index
        self._width, self._height = options.width, options.height
        self._clock = clock
        self._stop = stop
        self._ffmpeg_process = None

    def open(self) -> "WebcamCapture":
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg not found — install it: brew install ffmpeg"
            )
        ffmpeg_command = [
            ffmpeg_path, "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation",
            "-framerate", "30",
            "-video_size", f"{self._width}x{self._height}",
            "-i", f"{self._index}:none",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]
        # unbuffered, so select() on the pipe sees every byte ffmpeg wrote
        self._ffmpeg_process = subprocess.Popen(
            ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        try:
            first_frame = self._frame_bytes()
        except BaseException:
            self.close()
            raise
        if first_frame is None:
            self.close()
            raise RuntimeError(
                f"cannot open webcam index {self._index} via ffmpeg "
                "(grant camera permission in System Settings → Privacy → Camera)"
            )
        self._first_frame = first_frame  # buffer the first frame so read() can return it
        return self

    def _frame_bytes(self) -> bytearray | None:
        """Exactly one raw RGB frame from the ffmpeg pipe, or None on EOF or once `stop` is set. Fills one
        new frame buffer in place, O(frame bytes). select() on a pipe is POSIX-only, as is avfoundation."""
        frame = bytearray(self._width * self._height * 3)
        frame_view = memoryview(frame)
        pipe = self._ffmpeg_process.stdout
        num_received_bytes = 0
        while num_received_bytes < len(frame):
            if self._stop is not None and self._stop.is_set():
                return None
            is_readable, _, _ = select.select([pipe], [], [], _PIPE_POLL_S)
            if not is_readable:
                continue
            num_read_bytes = pipe.readinto(frame_view[num_received_bytes:])
            if not num_read_bytes:
                return None
            num_received_bytes += num_read_bytes
        return frame

    def read(self):
        """Grab one frame → (timestamp_s, RGB ndarray) or None on failure."""
        import numpy as np
        if self._ffmpeg_process is None:
            return None
        raw = getattr(self, "_first_frame", None)
        if raw is not None:
            self._first_frame = None
        else:
            raw = self._frame_bytes()
        if raw is None:
            return None
        rgb_frame = np.frombuffer(raw, dtype=np.uint8).reshape(self._height, self._width, 3)
        return self._clock(), rgb_frame

    def close(self) -> None:
        """Terminates ffmpeg, kills it if it outlives _TERMINATE_GRACE_S, and reaps it either way."""
        ffmpeg_process, self._ffmpeg_process = self._ffmpeg_process, None
        if ffmpeg_process is None:
            return
        try:
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=_TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                ffmpeg_process.kill()
                ffmpeg_process.wait()
        finally:
            ffmpeg_process.stdout.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exception_info):
        self.close()


def _paced(grab, duration_s, *, frame_rate_hz, stop, sleep, clock):
    """Yield non-None items from `grab` for `duration_s`, throttled to ~`frame_rate_hz`.
    `grab` is callable() -> item|None; `stop` an optional Event for early exit."""
    period_s = 1.0 / frame_rate_hz if frame_rate_hz > 0 else 0.0
    end_time_s = clock() + duration_s
    next_grab_time_s = clock()
    while clock() < end_time_s and (stop is None or not stop.is_set()):
        now_s = clock()
        if now_s < next_grab_time_s:
            sleep(min(next_grab_time_s - now_s, end_time_s - now_s))
            continue
        next_grab_time_s = now_s + period_s
        item = grab()
        if item is not None:
            yield item


def record_frames(grab, duration_s: float, *, frame_rate_hz: float = 10.0, stop=None,
                  sleep=time.sleep, clock=time.monotonic) -> list:
    """Buffer (timestamp_s, frame) from `grab` for `duration_s` at ~`frame_rate_hz`; label OFFLINE later
    via `stream_labels`. O(frames)."""
    return list(_paced(grab, duration_s, frame_rate_hz=frame_rate_hz, stop=stop, sleep=sleep, clock=clock))


def record_labels_online(grab, labeler, duration_s: float, *, frame_rate_hz: float = 15.0, on_label=None,
                         stop=None, sleep=time.sleep, clock=time.monotonic) -> list:
    """Grab a frame and run `labeler` LIVE per frame for `duration_s` → sorted list[Label].
    `on_label(label)` is an optional per-frame callback. Runs the model inside the capture loop,
    unlike `record_frames`. O(frames·model)."""
    labels = []
    for timestamp_s, image in _paced(grab, duration_s, frame_rate_hz=frame_rate_hz, stop=stop, sleep=sleep,
                                     clock=clock):
        label = labeler.label(image, timestamp_s)
        labels.append(label)
        if on_label is not None:
            on_label(label)
    labels.sort(key=lambda label: label.timestamp)
    return labels


def stream_labels(labeler, frames, *, max_frames=None) -> list:
    """Run `labeler` (a stream labeler — YoloLabeler / YoloSegLabeler / VisionLabeler) over an
    iterable of (timestamp, image) → list[Label] sorted by time, the stream `collect_source(...,
    labeler=...)` / `build_dataset` aligns to CSI window timestamps. O(frames · model)."""
    labels = []
    for i, (timestamp_s, image) in enumerate(frames):
        if max_frames is not None and i >= max_frames:
            break
        labels.append(labeler.label(image, timestamp_s))
    labels.sort(key=lambda label: label.timestamp)
    return labels


# COCO classes a stock YOLO can flag as a *visible* weapon (open-carry tier only — COCO has NO
# firearm class; a real concealed gun needs a custom-trained model and/or scripted labels).
COCO_WEAPON_CLASSES = (43,)  # 43 = knife (add 76=scissors / 34=baseball bat if you want them)


class WebcamLabelCapture(CameraLabelCapture):
    """Labels webcam frames live with YOLO-seg at ~`frame_rate_hz`: class 1 is a person for the presence
    stage, a visible `weapon_classes` object held by the person for the weapon stage; every label carries
    the `grid`x`grid` occupancy mask. Building one loads the net, so a missing ultralytics or weights file
    raises here."""

    def __init__(self, *, stage, camera_index, weights_path, weapon_classes, confidence_threshold, grid,
                 frame_rate_hz):
        label_function = weapon_label_function if stage == "weapon" else presence_label_function
        self._labeler = YoloSegLabeler(
            options=YoloLabelerOptions(weights_path=weights_path, weapon_classes=weapon_classes),
            confidence_threshold=confidence_threshold, grid=grid, label_function=label_function)
        self._camera_index = camera_index
        self._frame_rate_hz = frame_rate_hz

    def record_labels(self, duration_s, *, stop, on_label):
        with WebcamCapture(WebcamOptions(index=self._camera_index), stop=stop) as webcam:
            return record_labels_online(webcam.read, self._labeler, duration_s, frame_rate_hz=self._frame_rate_hz,
                                        stop=stop, on_label=on_label)
