import numpy as np

from wavetrace.application.ports import CsiSource
from wavetrace.domain.guard import DriftMonitor


class DriftMonitoredSource:
    """A `CsiSource` that feeds each frame's raw, pre-lock |H| to a `DriftMonitor` and publishes its
    recalibration advisories, yielding the frames unchanged. O(S) extra per frame."""

    def __init__(self, source: CsiSource, drift_monitor: DriftMonitor, publisher):
        self._source = source
        self._drift_monitor = drift_monitor
        self._publisher = publisher

    def frames(self):
        for frame in self._source.frames():
            event = self._drift_monitor.update(
                float(frame.timestamp), np.abs(np.asarray(frame.grid)).mean(axis=0).astype(np.float32))
            if event:
                self._publisher.publish_event(event)
            yield frame
