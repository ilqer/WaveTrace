from dataclasses import dataclass

from wavetrace.domain.groundtruth.labeler import presence_label_function
from wavetrace.groundtruth.segmentation_labeler import Segment, SegmentationLabeler
from wavetrace.groundtruth.vision_labeler import Detection, LabelerOptions, VisionLabeler


@dataclass(frozen=True, slots=True)
class YoloLabelerOptions(LabelerOptions):
    """Adds the Ultralytics network-selection knobs only `YoloLabeler`/`YoloSegLabeler` read:
    `device` (a torch device string, e.g. "cuda"; `None` = ultralytics' own default) and
    `weights_path` (a weights file for that class to load; `None` = its own `DEFAULT_WEIGHTS`)."""

    device: str | None = None
    weights_path: str | None = None


class YoloLabeler(VisionLabeler):
    """Ultralytics-YOLO adapter (person detection + optional weapon classes; pose model -> keypoints).
    The dependency is OPTIONAL and imported lazily (`pip install ultralytics`); pass `model=` an
    already-built model object to bypass the import (what the tests do).

    `model`: a ready model object exposing `__call__(image) -> results` — bypasses
    `options.weights_path` entirely. `options.weights_path`: a weights path/name (e.g.
    "yolov8n.pt", "yolov8n-pose.pt") for THIS class to load via ultralytics when `model` isn't
    given; `None` loads `DEFAULT_WEIGHTS`. Results are converted to `Detection`s with boxes in
    normalized xywh; pose keypoints (normalized) ride along when present."""

    DEFAULT_WEIGHTS = "yolov8n.pt"

    def __init__(self, *, options: YoloLabelerOptions = YoloLabelerOptions(), model=None,
                 confidence_threshold=0.35, image_size_px=640, label_function=presence_label_function):
        net = model if model is not None else self._load(
            options.weights_path or self.DEFAULT_WEIGHTS, options.device)

        def detector(image):
            results = net(image, imgsz=image_size_px, verbose=False) if _accepts_kwargs(net) else net(image)
            return _yolo_to_detections(results)

        super().__init__(detector, options=options, confidence_threshold=confidence_threshold,
                         label_function=label_function)

    @staticmethod
    def _load(weights_path, device):
        try:
            from ultralytics import YOLO
        except ImportError as import_error:  # pragma: no cover - exercised only without the optional dep
            raise ImportError(
                "YoloLabeler needs ultralytics: pip install ultralytics (or pass model= a prebuilt model)"
            ) from import_error
        net = YOLO(weights_path)
        if device is not None:
            net.to(device)
        return net


def _accepts_kwargs(net) -> bool:
    """ultralytics models take imgsz/verbose kwargs; a bare test stub may not."""
    return hasattr(net, "predict") or hasattr(net, "model")


def _yolo_to_detections(results) -> list[Detection]:
    detections: list[Detection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        boxes_xywhn = boxes.xywhn.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        keypoints = getattr(result, "keypoints", None)
        keypoints_xyn = (keypoints.xyn.cpu().numpy()
                         if keypoints is not None and keypoints.xyn is not None else None)
        for i in range(len(class_ids)):
            flat_keypoints = keypoints_xyn[i].reshape(-1).tolist() if keypoints_xyn is not None else []
            detections.append(Detection(int(class_ids[i]), float(confidences[i]),
                                        tuple(boxes_xywhn[i].tolist()), flat_keypoints))
    return detections


class YoloSegLabeler(SegmentationLabeler):
    """Ultralytics YOLO-seg adapter (e.g. "yolov8n-seg.pt"). Optional dep imported lazily; pass `model=`
    to bypass the import. See `YoloLabeler` for the `model` vs. `options.weights_path` split. Per-instance
    masks come from `result.masks.data`, (N,H,W)."""

    DEFAULT_WEIGHTS = "yolov8n-seg.pt"

    def __init__(self, *, options: YoloLabelerOptions = YoloLabelerOptions(), model=None,
                 confidence_threshold=0.5, grid=16, overlap_min=0.5, image_size_px=640,
                 label_function=presence_label_function):
        net = model if model is not None else YoloLabeler._load(
            options.weights_path or self.DEFAULT_WEIGHTS, options.device)

        def segmenter(image):
            results = net(image, imgsz=image_size_px, verbose=False) if _accepts_kwargs(net) else net(image)
            return _yolo_to_segments(results)

        super().__init__(segmenter, options=options, confidence_threshold=confidence_threshold, grid=grid,
                         overlap_min=overlap_min, label_function=label_function)


def _yolo_to_segments(results) -> list[Segment]:
    segments: list[Segment] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None:
            continue
        boxes_xywhn = boxes.xywhn.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        soft_masks = masks.data.cpu().numpy()  # (N, H, W) soft masks in [0, 1]
        for i in range(len(class_ids)):
            segments.append(Segment(int(class_ids[i]), float(confidences[i]), soft_masks[i] > 0.5,
                                    tuple(boxes_xywhn[i].tolist())))
    return segments
