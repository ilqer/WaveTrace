import bisect

from wavetrace import Label
from wavetrace.domain.groundtruth.labeler import FrameDetection, Labeler, weapon_label_function


class LocationChipLabeler(Labeler):
    """A BLE/UWB tag on the weapon gives precise time and position ground truth, which beats a
    coarse scripted span. `track` is (t, present, position) on the CSI clock, looked up by nearest
    sample. The position is stashed on the Label as a hint for separating the weapon's reflection
    from the body's - a hint only, because WiFi delay resolution is coarse and leaks."""

    def __init__(self, track, label_function=weapon_label_function):
        samples = sorted(track, key=lambda sample: sample[0])
        self._timestamps_s = [float(sample[0]) for sample in samples]
        self._present = [bool(sample[1]) for sample in samples]
        self._positions = [sample[2] for sample in samples]
        super().__init__(label_function)

    def _nearest(self, timestamp: float) -> int:
        """Index of the track sample nearest in time. O(log n)."""
        if not self._timestamps_s:
            raise ValueError("LocationChipLabeler: empty track")
        i = bisect.bisect_left(self._timestamps_s, timestamp)
        if i == 0:
            return 0
        if i >= len(self._timestamps_s):
            return len(self._timestamps_s) - 1
        return i if (self._timestamps_s[i] - timestamp) < (timestamp - self._timestamps_s[i - 1]) else i - 1

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        i = self._nearest(timestamp)
        present = self._present[i]
        return FrameDetection(
            present=present,
            weapon=present,
            position=list(self._positions[i]) if present and self._positions[i] is not None else None,
        )

    def label_at(self, timestamp: float) -> Label:
        return self.label(None, timestamp)

    __call__ = label_at
