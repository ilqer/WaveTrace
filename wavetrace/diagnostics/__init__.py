"""Observability layer — per-node link health, antenna/band weights, feature separation, drift.

Nothing here is on the <8 ms hot path: telemetry is computed on a slower cadence (once per emit
window or once per N frames), separate from the inference call, so it never slows detection."""

from wavetrace.diagnostics.telemetry import (
    HealthMeterOptions,
    NodeHealthMeter,
    cluster_sync,
    baseline_drift,
    feature_separation,
    dataset_report,
)
from wavetrace.diagnostics.weapon_separation import (
    gather_sigma2,
    json_hist,
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
    "feature_separation",
    "dataset_report",
    "gather_sigma2",
    "separation",
    "json_hist",
    "verdict",
    "key_label",
    "sigma2_per_frame",
]
