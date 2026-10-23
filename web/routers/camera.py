"""Frames come from an ffmpeg subprocess, not cv2.VideoCapture: on macOS, AVFoundation's run loop
segfaults on a background thread. cv2 is used only for YOLO annotation."""

import asyncio
import shutil
import subprocess
import threading
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from ultralytics import YOLO

router = APIRouter()

JPEG_START_OF_IMAGE = b"\xff\xd8"
JPEG_END_OF_IMAGE = b"\xff\xd9"

_yolo_cache: dict[str, "YOLO | None"] = {}
_yolo_lock = threading.Lock()
_camera_active = False


def _load_yolo(weights: str = "yolov8n-seg.pt"):
    with _yolo_lock:
        if weights not in _yolo_cache:
            try:
                from ultralytics import YOLO
                _yolo_cache[weights] = YOLO(weights)
            except Exception as error:
                _yolo_cache[weights] = None   # cache the failure so we don't retry on every frame
                print(f"[YOLO] load failed: {error}")
    return _yolo_cache.get(weights)


def _annotate_frame(model, frame, weapon_class_ids=(43,)):
    import cv2, numpy as np
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    for result in results:
        boxes = result.boxes
        masks = result.masks
        for i, detection_box in enumerate(boxes):
            class_id = int(detection_box.cls[0])
            confidence = float(detection_box.conf[0])
            is_weapon = class_id in weapon_class_ids
            color = (30, 120, 255) if is_weapon else (50, 220, 80)   # BGR: orange / green
            label = f"{'WEAPON' if is_weapon else model.names.get(class_id, str(class_id))} {confidence:.0%}"
            if masks is not None and i < len(masks.xy):
                mask_polygon_px = masks.xy[i].astype(np.int32)
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [mask_polygon_px], color)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.55, overlay, 0.45, 0)
                cv2.polylines(annotated_frame, [mask_polygon_px], True, color, 2)
            left_px, top_px, right_px, bottom_px = map(int, detection_box.xyxy[0])
            cv2.rectangle(annotated_frame, (left_px, top_px), (right_px, bottom_px), color, 1)
            (text_width_px, text_height_px), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_frame, (left_px, top_px - text_height_px - 4),
                          (left_px + text_width_px + 2, top_px), color, -1)
            cv2.putText(annotated_frame, label, (left_px + 1, top_px - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated_frame


def _find_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg not found — install it: brew install ffmpeg"
        )
    return ffmpeg_path


@router.get("/api/camera/check")
async def camera_check(cam_index: int = 0):
    def _probe():
        try:
            ffmpeg_path = _find_ffmpeg()
        except RuntimeError as error:
            return {"ok": False, "error": str(error)}
        ffmpeg_command = [
            ffmpeg_path, "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation", "-framerate", "30", "-video_size", "1280x720",
            "-i", f"{cam_index}:none",
            "-vframes", "1", "-f", "rawvideo", "-vcodec", "rawvideo", "-",
        ]
        try:
            result = subprocess.run(ffmpeg_command, capture_output=True, timeout=10)
            if result.returncode != 0 or not result.stdout:
                stderr_tail = result.stderr.decode(errors="replace")[-400:]
                if "permission" in stderr_tail.lower() or "authorization" in stderr_tail.lower():
                    return {"ok": False,
                            "error": "Camera permission denied — allow Terminal in System Settings → Privacy → Camera"}
                return {"ok": False, "error": f"ffmpeg exit {result.returncode}: {stderr_tail}"}
            return {"ok": True, "width": 1280, "height": 720, "cam_index": cam_index}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Camera probe timed out"}
        except Exception as error:
            return {"ok": False, "error": str(error)}
    return await asyncio.to_thread(_probe)


@router.post("/api/camera/stop")
def stop_camera():
    global _camera_active
    _camera_active = False
    return {"ok": True}


@router.get("/api/camera/stream")
async def camera_stream(request: Request, index: int = 0, annotate: bool = False,
                        weights: str = "yolov8n-seg.pt"):
    """ffmpeg's mjpeg output is one continuous byte stream, so frames are cut at the JPEG
    SOI/EOI markers and sent as multipart/x-mixed-replace chunks."""
    global _camera_active
    _camera_active = True

    yolo_model = await asyncio.to_thread(_load_yolo, weights) if annotate else None

    async def _generate():
        try:
            ffmpeg_path = _find_ffmpeg()
        except RuntimeError:
            return  # ffmpeg missing: an empty body makes the browser's <img> fire onError

        ffmpeg_command = [
            ffmpeg_path, "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation",
            "-framerate", "30",
            "-video_size", "1280x720",
            "-i", f"{index}:none",
            "-f", "mjpeg",
            "-q:v", "5",
            "-",
        ]

        ffmpeg_process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        unparsed_bytes = b""
        frame_index = 0

        try:
            while _camera_active:
                if await request.is_disconnected():
                    break

                chunk = await ffmpeg_process.stdout.read(65536)
                if not chunk:
                    break
                unparsed_bytes += chunk
                while True:
                    frame_start_byte = unparsed_bytes.find(JPEG_START_OF_IMAGE)
                    if frame_start_byte == -1:
                        unparsed_bytes = b""
                        break
                    frame_end_byte = unparsed_bytes.find(JPEG_END_OF_IMAGE, frame_start_byte + 2)
                    if frame_end_byte == -1:
                        unparsed_bytes = unparsed_bytes[frame_start_byte:]
                        break
                    jpeg_frame = unparsed_bytes[frame_start_byte: frame_end_byte + 2]
                    unparsed_bytes = unparsed_bytes[frame_end_byte + 2:]

                    frame_index += 1

                    if yolo_model is not None:
                        # throttle YOLO to 5 Hz: 1 frame in every 6 of the 30 Hz source
                        if frame_index % 6 != 0:
                            continue

                        import cv2, numpy as np
                        encoded_jpeg = np.frombuffer(jpeg_frame, dtype=np.uint8)
                        frame = cv2.imdecode(encoded_jpeg, cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame = await asyncio.to_thread(_annotate_frame, yolo_model, frame)
                            _, annotated_jpeg = cv2.imencode(".jpg", frame,
                                                             [cv2.IMWRITE_JPEG_QUALITY, 75])
                            jpeg_frame = annotated_jpeg.tobytes()

                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg_frame)).encode()
                           + b"\r\n\r\n" + jpeg_frame + b"\r\n")
        finally:
            try:
                ffmpeg_process.kill()
            except ProcessLookupError:
                pass

    return StreamingResponse(_generate(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
