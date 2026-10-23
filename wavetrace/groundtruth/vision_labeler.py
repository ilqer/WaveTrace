from dataclasses import dataclass, field

from wavetrace.domain.groundtruth.labeler import FrameDetection, Labeler, presence_label_function


@dataclass
class Detection:
    """One object-detector hit, geometry normalized to the frame ([0,1]). `cls` is the model's class
    id; `keypoints` is a flat [x0,y0,x1,y1,...] list (empty unless a pose model)."""

    cls: int
    confidence: float
    bbox_xywhn: tuple  # (x, y, w, h) normalized to [0, 1]
    keypoints: list = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LabelerOptions:
    """Detector-selection knobs every `Labeler` subclass reads: which detector class id counts as
    the person, and which class ids count as a weapon."""

    person_class: int = 0
    weapon_classes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "weapon_classes", tuple(int(class_id) for class_id in self.weapon_classes))
        if self.person_class < 0:
            raise ValueError("person_class must be >= 0")


class VisionLabeler(Labeler):
    """Camera-supervised labeler driven by a real object detector. `detector(image) -> list[Detection]`
    runs the model on one RGB frame; this picks the highest-confidence person (-> present + bbox +
    keypoints) and flags a weapon if any detection's class is in `weapon_classes`.

    observation = the RGB frame (an (H, W, 3) array or a path the detector accepts); timestamp comes
    from the stream. With `label_function=presence_label_function` it teaches Stage A;
    `weapon_label_function` + `weapon_classes` teaches Stage E from a camera that CAN see the weapon
    (open-carry tier)."""

    def __init__(self, detector, *, options: LabelerOptions = LabelerOptions(), confidence_threshold=0.35,
                 label_function=presence_label_function):
        super().__init__(label_function)
        if not callable(detector):
            raise ValueError("VisionLabeler: detector must be callable(image) -> list[Detection]")
        self._detector = detector
        self._person = options.person_class
        self._weapon = set(options.weapon_classes)
        self._confidence_threshold = float(confidence_threshold)

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        detections = [detection for detection in self._detector(observation)
                      if detection.confidence >= self._confidence_threshold]
        persons = [detection for detection in detections if detection.cls == self._person]
        best_person = max(persons, key=lambda detection: detection.confidence) if persons else None
        weapon = any(detection.cls in self._weapon for detection in detections)
        return FrameDetection(
            present=best_person is not None,
            weapon=weapon,
            bbox=list(best_person.bbox_xywhn) if best_person is not None else None,
            keypoints=list(best_person.keypoints) if best_person is not None else [],
        )
