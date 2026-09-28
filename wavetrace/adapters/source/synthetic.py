"""In-memory frame replay for dev and CI."""

from wavetrace.application.ports import CsiSource


class SyntheticSource(CsiSource):
    """Replay an in-memory frame list (e.g. from wavetrace.synthetic)."""

    def __init__(self, frames):
        self._frames = list(frames)

    def frames(self):
        return iter(self._frames)
