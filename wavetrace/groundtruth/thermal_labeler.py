from wavetrace.domain.groundtruth.labeler import FrameDetection, Labeler


class ThermalLabeler(Labeler):
    """A thermal camera can see a concealed weapon's cold-metal shadow against body heat, labeling
    data an RGB camera cannot. Accepting a detection only inside a detected person box keeps the
    false-positive rate down. Needs a thermal model and recordings that are not here: wire it by
    overriding `_detect`."""

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        raise NotImplementedError(
            "ThermalLabeler is a hardware seam: provide a thermal detector and gate weapon "
            "detections on a person bbox."
        )
