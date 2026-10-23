from wavetrace.domain.groundtruth.labeler import FrameDetection, Labeler


class ReplayLabeler(Labeler):
    """Replays pre-computed detections (synthetic fixture or a recorded vision-model output stream).
    observation = {"t": ts, "raw": {detection}}."""

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        raw = observation.get("raw", observation)
        return FrameDetection(
            present=bool(raw.get("present", False)),
            weapon=bool(raw.get("weapon", False)),
            bbox=raw.get("bbox"),
            keypoints=list(raw.get("keypoints") or []),
            position=raw.get("position"),
            mask=raw.get("mask"),
            mask_grid=int(raw.get("mask_grid") or 0),
        )
