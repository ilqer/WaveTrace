"""Observability layer — per-node link health, antenna/band weights, feature separation, drift.

Nothing here is on the <8 ms hot path: telemetry is computed on a slower cadence (once per emit
window or once per N frames), separate from the inference call, so it never slows detection."""

from wavetrace.diagnostics.Telemetry import (
    NodeHealthMeter,
    clusterSync,
    baselineDrift,
    featureSeparation,
    datasetReport,
)

__all__ = [
    "NodeHealthMeter",
    "clusterSync",
    "baselineDrift",
    "featureSeparation",
    "datasetReport",
]
