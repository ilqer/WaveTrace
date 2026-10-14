"""Time the hot C++ functions through the pybind11 binding, over fixed synthetic input.

`--compare` re-runs the same benchmarks and diffs them against a committed baseline:

    python benchmarks/bench_native.py --compare benchmarks/baseline.json

The budget is 5%, measured on the minimum: scheduler noise, GC and throttling only slow a batch
down, so the minimum is the most noise-robust estimate. Input is deterministic, nothing is timed
that does I/O, and each function gets a warmup pass so first-call cache and allocator effects stay
out of the numbers.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from wavetrace import (
    CsiFrame,
    Preprocessor,
    coefficient_of_variation,
    combined_channel_difference,
    conjugate_multiply,
    fft,
    hampel,
    inter_carrier_phase_stats,
    inter_carrier_stats,
    nine_features,
    unwrap_step,
)

SEED = 1234
NUM_ANTENNAS = 3
NUM_SUBCARRIERS = 64  # power-of-two so the same array also feeds fft()
WINDOW_FRAMES = 128
BATCH_SIZE = 500  # calls timed together per sample
NUM_BATCHES = 50  # sample points -> 25_000 total calls per function
WARMUP_BATCHES = 5
REGRESSION_BUDGET = 0.05


def _time_calls(benchmark_call, batch_size=BATCH_SIZE, num_batches=NUM_BATCHES,
                warmup_batches=WARMUP_BATCHES):
    """Seconds per call for each batch. Timing `batch_size` calls together amortizes perf_counter()
    overhead and dispatch jitter, which dominate at sub-microsecond scale. O(num_batches * batch_size)."""
    for _ in range(warmup_batches):
        for _ in range(batch_size):
            benchmark_call()
    samples = np.empty(num_batches, dtype=np.float64)
    for i in range(num_batches):
        start_time_s = time.perf_counter()
        for _ in range(batch_size):
            benchmark_call()
        samples[i] = (time.perf_counter() - start_time_s) / batch_size
    return samples


def _summarize(name: str, samples) -> dict:
    sorted_samples = np.sort(np.asarray(samples))
    return {
        "name": name,
        "batches": int(sorted_samples.size),
        "calls_per_batch": BATCH_SIZE,
        # min is the comparison metric: scheduler noise, GC and throttling only slow a batch down,
        # so the minimum is the most noise-robust estimate of the true per-call cost
        "min_us": float(sorted_samples.min() * 1e6),
        "mean_us": float(sorted_samples.mean() * 1e6),
        "median_us": float(np.median(sorted_samples) * 1e6),
        "p95_us": float(np.percentile(sorted_samples, 95) * 1e6),
    }


def run_benchmarks() -> list[dict]:
    rng = np.random.default_rng(SEED)
    antenna_count, subcarrier_count = NUM_ANTENNAS, NUM_SUBCARRIERS

    grid = (rng.uniform(0.5, 1.5, (antenna_count, subcarrier_count))
            * np.exp(1j * rng.uniform(-np.pi, np.pi, (antenna_count, subcarrier_count)))).astype(
        np.complex64
    )
    frame = CsiFrame(antenna_count, subcarrier_count)
    frame.grid[:, :] = grid
    scratch_frame = CsiFrame(1, 1)  # reshaped in-place by conjugate_multiply/combined_channel_difference

    amplitudes = rng.uniform(0.5, 1.5, subcarrier_count).astype(np.float32)
    hampel_window = rng.uniform(0.5, 1.5, 7).astype(np.float32)
    fft_input = (rng.normal(size=subcarrier_count)
                 + 1j * rng.normal(size=subcarrier_count)).astype(np.complex64)
    feature_window = rng.normal(size=WINDOW_FRAMES).astype(np.float32)
    ic_magnitudes = rng.uniform(0.5, 1.5, subcarrier_count).astype(np.float32)
    ic_phase = rng.uniform(-np.pi, np.pi, subcarrier_count).astype(np.float32)
    preprocessor = Preprocessor(antenna_count, subcarrier_count)

    benchmarks = [
        ("conjugate_multiply", lambda: conjugate_multiply(frame, scratch_frame)),
        ("combined_channel_difference", lambda: combined_channel_difference(frame, scratch_frame)),
        ("coefficient_of_variation", lambda: coefficient_of_variation(amplitudes)),
        ("hampel", lambda: hampel(hampel_window, 1.0, 5.0)),
        ("unwrap_step", lambda: unwrap_step(0.1, 0.0, 0.0)),
        ("preprocessor_process", lambda: preprocessor.process(frame)),
        ("fft", lambda: fft(fft_input)),
        ("nine_features", lambda: nine_features(feature_window)),
        ("inter_carrier_stats", lambda: inter_carrier_stats(ic_magnitudes)),
        ("inter_carrier_phase_stats", lambda: inter_carrier_phase_stats(ic_phase)),
    ]
    return [_summarize(name, _time_calls(benchmark_call)) for name, benchmark_call in benchmarks]


def _compare(current: list[dict], baseline_path: Path) -> bool:
    """Print a per-function regression report on min_us; True if every function is within budget."""
    baseline = {saved_entry["name"]: saved_entry
                for saved_entry in json.loads(baseline_path.read_text())["benchmarks"]}
    is_within_budget = True
    print(f"{'function':<32}{'baseline (us)':>16}{'current (us)':>16}{'delta':>10}")
    for entry in current:
        name = entry["name"]
        if name not in baseline:
            print(f"{name:<32}  (no baseline entry — skipped)")
            continue
        baseline_us = baseline[name]["min_us"]
        current_us = entry["min_us"]
        delta = (current_us - baseline_us) / baseline_us if baseline_us > 0 else 0.0
        flag = "" if delta <= REGRESSION_BUDGET else "  REGRESSION"
        is_within_budget = is_within_budget and delta <= REGRESSION_BUDGET
        print(f"{name:<32}{baseline_us:>16.3f}{current_us:>16.3f}{delta:>+9.1%}{flag}")
    return is_within_budget


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/baseline.json",
                     help="where to write the benchmark results (default: benchmarks/baseline.json)")
    parser.add_argument("--compare", default=None,
                     help="diff against this committed baseline instead of writing a new one; "
                          "exits 1 if any function's minimum regresses beyond the 5%% budget")
    args = parser.parse_args()

    results = run_benchmarks()

    if args.compare:
        return 0 if _compare(results, Path(args.compare)) else 1

    payload = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "seed": SEED,
        "batches_per_benchmark": NUM_BATCHES,
        "calls_per_batch": BATCH_SIZE,
        "regression_budget": REGRESSION_BUDGET,
        "benchmarks": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path} ({len(results)} benchmarks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
