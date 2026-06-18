"""Observability layer — per-node link health, antenna/band weights, feature separation, drift.

Nothing here is on the <8 ms hot path: telemetry is computed on a slower cadence (once per emit
window or once per N frames), separate from the inference call, so it never slows detection."""

from wavetrace.diagnostics.Telemetry import (
    NodeHealthMeter,
    cluster_sync,
    baseline_drift,
    feature_separation,
    dataset_report,
)

__all__ = [
    "NodeHealthMeter",
    "cluster_sync",
    "baseline_drift",
    "feature_separation",
    "dataset_report",
]
