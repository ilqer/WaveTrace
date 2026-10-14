"""Observability layer — per-node link health, antenna/band weights, drift.

Nothing here is on the <8 ms hot path: telemetry runs on a slower cadence, separate from inference."""

from wavetrace.diagnostics.telemetry import (
    HealthMeterOptions,
    NodeHealthMeter,
    cluster_sync,
    baseline_drift,
    dataset_report,
)
from wavetrace.diagnostics.weapon_separation import (
    gather_sigma2,
    json_histogram,
    key_label,
    separation,
    sigma2_per_frame,
    verdict,
)

__all__ = [
    "HealthMeterOptions",
    "NodeHealthMeter",
    "cluster_sync",
    "baseline_drift",
    "dataset_report",
    "gather_sigma2",
    "separation",
    "json_histogram",
    "verdict",
    "key_label",
    "sigma2_per_frame",
]
