"""In-memory frame replay for dev and CI."""

from wavetrace.application.ports import CsiSource


class SyntheticSource(CsiSource):
    def __init__(self, frames):
        self._frames = list(frames)

    def frames(self):
        return iter(self._frames)
