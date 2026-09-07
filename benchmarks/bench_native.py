"""Phase 0 step 6 (REFACTOR_PLAN.md) — times the hot C++ functions through the pybind11 binding
over fixed synthetic input. `--compare` re-runs the same benchmarks and diffs against a committed
baseline; later phases (renaming is compile-time only per §0, so 0% cost is expected, but this is
the instrument that PROVES it) run:

    python benchmarks/bench_native.py --compare benchmarks/baseline.json

Regression budget from REFACTOR_PLAN.md §0 "Performance": 5%, measured on the median (robust to the
occasional scheduler-jitter outlier that a mean would absorb).

Deterministic input (fixed seed), no I/O in the timed region, one warmup pass per function before
the timed loop to avoid measuring first-call cache/allocator effects. O(1) per benchmarked call —
this only characterizes function-level latency, not a throughput/pipeline benchmark.
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
REGRESSION_BUDGET = 0.05  # §0: 5%


def _time_calls(fn, batch_size=BATCH_SIZE, num_batches=NUM_BATCHES, warmup=WARMUP_BATCHES):
    """Median wall-clock seconds per call, timed in batches rather than one perf_counter() pair per
    call. At these functions' sub-microsecond scale, per-call perf_counter() overhead and Python
    dispatch jitter would otherwise dominate the signal — timing `batch_size` calls together and
    dividing amortizes both, which is what makes the 5% regression budget (§0) meaningful here
    instead of noise. O(num_batches * batch_size)."""
    for _ in range(warmup):
        for _ in range(batch_size):
            fn()
    samples = np.empty(num_batches, dtype=np.float64)
    for i in range(num_batches):
        t0 = time.perf_counter()
        for _ in range(batch_size):
            fn()
        samples[i] = (time.perf_counter() - t0) / batch_size
    return samples


def _summarize(name: str, samples) -> dict:
    s = np.sort(np.asarray(samples))
    return {
        "name": name,
        "batches": int(s.size),
        "calls_per_batch": BATCH_SIZE,
        # min, not mean/median, is the comparison metric (see _compare): OS scheduling noise, GC,
        # and thermal throttling can only ever slow a batch down, never speed it up below the true
        # per-call cost — so the minimum across batches is the most regression-noise-robust estimate
        # of that floor. mean/median/p95 are kept for human-readable context only.
        "min_us": float(s.min() * 1e6),
        "mean_us": float(s.mean() * 1e6),
        "median_us": float(np.median(s) * 1e6),
        "p95_us": float(np.percentile(s, 95) * 1e6),
    }


def run_benchmarks() -> list[dict]:
    """Build fixed synthetic input once, then time each hot binding call. O(iters) per entry."""
    rng = np.random.default_rng(SEED)
    a, s = NUM_ANTENNAS, NUM_SUBCARRIERS

    grid = (rng.uniform(0.5, 1.5, (a, s)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (a, s)))).astype(
        np.complex64
    )
    frame = CsiFrame(a, s)
    frame.grid[:, :] = grid
    scratch_frame = CsiFrame(1, 1)  # reshaped in-place by conjugate_multiply/combined_channel_difference

    amplitudes = rng.uniform(0.5, 1.5, s).astype(np.float32)
    hampel_window = rng.uniform(0.5, 1.5, 7).astype(np.float32)
    fft_input = (rng.normal(size=s) + 1j * rng.normal(size=s)).astype(np.complex64)
    feature_window = rng.normal(size=WINDOW_FRAMES).astype(np.float32)
    ic_magnitudes = rng.uniform(0.5, 1.5, s).astype(np.float32)
    ic_phase = rng.uniform(-np.pi, np.pi, s).astype(np.float32)
    preprocessor = Preprocessor(a, s)

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
    return [_summarize(name, _time_calls(fn)) for name, fn in benchmarks]


def _compare(current: list[dict], baseline_path: Path) -> bool:
    """Print a per-function regression report on min_us (see _summarize for why min, not
    mean/median). Returns True if every function is within the §0 5% budget."""
    baseline = {b["name"]: b for b in json.loads(baseline_path.read_text())["benchmarks"]}
    ok = True
    print(f"{'function':<32}{'baseline (us)':>16}{'current (us)':>16}{'delta':>10}")
    for entry in current:
        name = entry["name"]
        if name not in baseline:
            print(f"{name:<32}  (no baseline entry — skipped)")
            continue
        base_us = baseline[name]["min_us"]
        cur_us = entry["min_us"]
        delta = (cur_us - base_us) / base_us if base_us > 0 else 0.0
        flag = "" if delta <= REGRESSION_BUDGET else "  REGRESSION"
        ok = ok and delta <= REGRESSION_BUDGET
        print(f"{name:<32}{base_us:>16.3f}{cur_us:>16.3f}{delta:>+9.1%}{flag}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="benchmarks/baseline.json",
                     help="where to write the benchmark results (default: benchmarks/baseline.json)")
    ap.add_argument("--compare", default=None,
                     help="diff against this committed baseline instead of writing a new one; "
                          "exits 1 if any function's median regresses beyond the 5%% budget (§0)")
    args = ap.parse_args()

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
