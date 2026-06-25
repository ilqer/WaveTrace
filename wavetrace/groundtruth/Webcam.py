"""MacBook (or any OpenCV) webcam frame source for camera-supervised CSI labeling.

The labelers already exist (`VisionLabeler`/`YoloLabeler` → presence+weapon boxes, `YoloSegLabeler`
→ occupancy-mask "where" target). The only missing piece for a laptop is a frame source whose
timestamps share the CSI wall clock so `build_dataset`'s align step can match camera Labels to CSI
windows. This module provides that.

Capture is split from inference on purpose: `record_frames` grabs (timestamp, RGB) cheaply into a
buffer during the live CSI capture, then `stream_labels` runs YOLO OFFLINE over that buffer. That
keeps the capture loop real-time (no per-frame model latency) and makes the model step testable with
an injected detector. cv2 is imported lazily so importing this module never requires OpenCV.

  Input:  webcam index (or an injected grab fn) + a Labeler.
  Output: list[Label] timestamped on the CSI wall clock → pass as collect_source(labeler=...).
"""

import time

from wavetrace import Label


class WebcamCapture:
    """ffmpeg-based webcam wrapper yielding (timestamp_s, RGB ndarray).
    Uses ffmpeg subprocess for capture (avoids macOS AVFoundation run-loop
    segfault when cv2.VideoCapture is called from a background thread).
    On macOS, ffmpeg triggers the system permission dialog on first run—no
    manual Terminal camera grant needed.
    `clock` is injectable (defaults to wall-clock `time.time`, matching the CSI
    ntp_ms stamp so labels and CSI windows align). Use as a context manager."""

    def __init__(self, index: int = 0, size=(1280, 720), clock=time.time):
        self._index = int(index)
        self._width, self._height = size
        self._clock = clock
        self._proc = None

    def open(self) -> "WebcamCapture":
        import shutil, subprocess
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found — install it: brew install ffmpeg"
            )
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation",
            "-framerate", "30",
            "-video_size", f"{self._width}x{self._height}",
            "-i", f"{self._index}:none",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # Verify we can read at least one frame
        frame_bytes = self._frame_bytes()
        if frame_bytes is None:
            self._proc.terminate()
            self._proc = None
            raise RuntimeError(
                f"cannot open webcam index {self._index} via ffmpeg "
                "(grant camera permission in System Settings → Privacy → Camera)"
            )
        self._first_frame = frame_bytes  # buffer the first frame so read() can return it
        return self

    def _frame_bytes(self) -> bytes | None:
        """Read exactly one raw RGB frame from the ffmpeg pipe, or None on EOF/error."""
        n = self._width * self._height * 3
        buf = b""
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def read(self):
        """Grab one frame → (timestamp_s, RGB ndarray) or None on failure."""
        import numpy as np
        if self._proc is None:
            return None
        # Return buffered first frame if present
        raw = getattr(self, "_first_frame", None)
        if raw is not None:
            self._first_frame = None
        else:
            raw = self._frame_bytes()
        if raw is None:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self._height, self._width, 3)
        return self._clock(), arr

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
            self._proc = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


def _paced(grab, duration_s, *, fps, stop, sleep, clock):
    """Yield non-None items from `grab` for `duration_s`, throttled to ~`fps`. Shared by the buffered
    and online paths. `grab` is callable() -> item|None; `stop` an optional Event for early exit."""
    period = 1.0 / fps if fps > 0 else 0.0
    t_end = clock() + duration_s
    next_t = clock()
    while clock() < t_end and (stop is None or not stop.is_set()):
        now = clock()
        if now < next_t:
            sleep(min(next_t - now, t_end - now))
            continue
        next_t = now + period
        item = grab()
        if item is not None:
            yield item


def record_frames(grab, duration_s: float, *, fps: float = 10.0, stop=None,
                  sleep=time.sleep, clock=time.monotonic) -> list:
    """Buffer (ts, frame) from `grab` for `duration_s` at ~`fps` (label OFFLINE later via
    `stream_labels`). Keeps capture real-time when you don't want per-frame model latency. O(frames)."""
    return list(_paced(grab, duration_s, fps=fps, stop=stop, sleep=sleep, clock=clock))


def record_labels_online(grab, labeler, duration_s: float, *, fps: float = 15.0, on_label=None,
                         stop=None, sleep=time.sleep, clock=time.monotonic) -> list:
    """ONLINE path: grab a frame and run `labeler` LIVE per frame for `duration_s` → sorted
    list[Label]. `on_label(label)` is an optional per-frame callback for live feedback (e.g. a rolling
    present/weapon count). Labels carry the CSI wall-clock timestamp so `build_dataset` aligns them to
    CSI windows. Heavier than buffering (YOLO runs in the loop) but gives live detections. O(frames·model)."""
    labels = []
    for ts, img in _paced(grab, duration_s, fps=fps, stop=stop, sleep=sleep, clock=clock):
        lab = labeler.label(img, ts)
        labels.append(lab)
        if on_label is not None:
            on_label(lab)
    labels.sort(key=lambda l: l.timestamp)
    return labels


def stream_labels(labeler, frames, *, max_frames=None) -> list:
    """Run `labeler` (a CameraLabeler — YoloLabeler / YoloSegLabeler / VisionLabeler) over an
    iterable of (timestamp, image) → list[Label] sorted by time. This is the camera label stream
    `collect_source(..., labeler=...)` / `build_dataset` aligns to CSI window timestamps. Pure and
    detector-agnostic, so it unit-tests with a stub detector. O(frames · model)."""
    labels = []
    for i, (ts, image) in enumerate(frames):
        if max_frames is not None and i >= max_frames:
            break
        labels.append(labeler.label(image, ts))
    labels.sort(key=lambda l: l.timestamp)
    return labels


# COCO classes a stock YOLO can flag as a *visible* weapon (open-carry tier only — COCO has NO
# firearm class; a real concealed gun needs a custom-trained model and/or scripted labels).
COCO_WEAPON_CLASSES = (43,)  # 43 = knife (add 76=scissors / 34=baseball bat if you want them)
