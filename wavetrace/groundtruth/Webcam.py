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
    """OpenCV webcam wrapper yielding (timestamp_s, RGB frame). On macOS index 0 is the built-in
    FaceTime camera. `clock` is injectable (defaults to wall-clock `time.time`, matching the CSI
    ntp_ms stamp so labels and CSI windows align). Use as a context manager."""

    def __init__(self, index: int = 0, size=(1280, 720), clock=time.time):
        self._index = int(index)
        self._size = size
        self._clock = clock
        self._cap = None
        self._cv2 = None

    def open(self) -> "WebcamCapture":
        import cv2  # lazy: importing this module must not require OpenCV
        cap = cv2.VideoCapture(self._index)
        if self._size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        if not cap.isOpened():
            raise RuntimeError(f"cannot open webcam index {self._index} (grant camera permission / "
                               "close other apps using it)")
        self._cap, self._cv2 = cap, cv2
        return self

    def read(self):
        """Grab one frame -> (timestamp_s, RGB ndarray) or None on failure. BGR→RGB so the frame
        matches what ultralytics/most detectors expect."""
        ok, bgr = self._cap.read()
        if not ok:
            return None
        return self._clock(), self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

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
