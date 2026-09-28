"""Offline delay-domain super-resolution (Channel Impulse Response) via L1 sparse recovery.

The wideband channel is sparse in the delay domain (a few physical paths), so the K measured
subcarriers H(f_k) can be inverted into a fine-grid CIR `x` whose taps are the multipath arrivals,
including a concealed metal object's reflection.

Why ISTA on a DFT dictionary, not a plain IFFT:
  * A zero-padded IFFT smears each tap with -13 dB sidelobes, so adjacent taps (body vs object) merge.
  * Φ is a *sub*-DFT, one column per fine delay bin; solving `min ‖H-Φx‖² + λ‖x‖₁` super-resolves the
    delay axis roughly `oversample`× past the 1/BW Nyquist limit. κ(Φ)≈1 by construction.
  * ISTA over OMP: when the specular and body reflections fall in one Nyquist bin, OMP's greedy
    commit is irreversible, while ISTA's continuous shrinkage degrades gracefully at low SNR.

Φ atoms are built from the MEASURED subcarrier indices, so non-contiguous layouts (HT40's central
null gap, masked pilots) are handled natively — no separate subcarrier-infill step is needed.

Pass phase-sanitized H (conj-multiplied / CFO-removed, e.g. via reconstruct_complex_csi). Raw
STO/CFO ramps fit as ghost taps near τ=0.

Offline only (never the hot path): dictionary build O(K·G) once; each estimate O(max_iterations·K·G).
"""

from dataclasses import dataclass

import numpy as np

SUBCARRIER_SPACING_HZ = 312_500.0  # 802.11n HT20/HT40 OFDM tone spacing
_NOISE_FLOOR_DB = -25.0            # tap power this far below the peak counts as noise


class CirError(ValueError):
    """Raised on malformed CIR inputs (bad shapes, empty bands, non-finite CSI)."""


@dataclass(frozen=True, slots=True)
class Cir:
    """A recovered delay-domain channel impulse response and its summary statistics."""

    taps: np.ndarray            # complex (G,) — fine-grid tap amplitudes; index 0 = zero delay
    tap_delays_s: np.ndarray    # float (G,) — delay of each bin in seconds
    subcarrier_spacing_hz: float
    dominant_bin: int           # argmax |tap|^2
    dominant_ratio: float       # peak power / total power (high => coherent reflector / strong LOS)
    rms_delay_spread_s: float   # power-weighted RMS spread (diffuse body => large; flat metal => small)
    active_tap_count: int       # taps above the noise floor

    @property
    def dominant_delay_s(self) -> float:
        return float(self.tap_delays_s[self.dominant_bin])

    @property
    def dominant_tap(self) -> complex:
        return complex(self.taps[self.dominant_bin])


def delay_dictionary(subcarrier_indices: np.ndarray, num_delay_bins: int) -> np.ndarray:
    """Sub-DFT sensing matrix Φ ∈ ℂ^{K×G}: column g is the delay atom for fine bin g, sampled at the
    MEASURED subcarrier offsets `subcarrier_indices` (integer tone indices; supplies the band
    geometry / gaps). Δf cancels in the matrix (it only scales bin→seconds), so atoms are
    exp(-j2π·subcarrier_indices·g/G). Normalised by 1/√K so ΦΦᴴ ≈ (G/K)·I (well-conditioned). O(K·G)."""
    subcarrier_indices = np.asarray(subcarrier_indices, dtype=np.float64).ravel()
    num_subcarriers = subcarrier_indices.size
    if num_subcarriers == 0 or num_delay_bins <= 0:
        raise CirError("delay_dictionary: need >=1 subcarrier and num_delay_bins>0")
    bin_indices = np.arange(num_delay_bins, dtype=np.float64)
    phase = -2.0 * np.pi * np.outer(subcarrier_indices, bin_indices) / float(num_delay_bins)  # (K, G)
    return (np.exp(1j * phase) / np.sqrt(num_subcarriers)).astype(np.complex64)


def _soft_threshold(taps: np.ndarray, threshold: float) -> np.ndarray:
    """Complex soft-threshold: shrink each magnitude by `threshold`, keep phase. The ISTA prox of λ‖·‖₁."""
    magnitude = np.abs(taps)
    shrink_factor = np.maximum(0.0, 1.0 - threshold / np.maximum(magnitude, 1e-12))
    return (taps * shrink_factor).astype(taps.dtype)


def estimate_cir_taps(H: np.ndarray, phi: np.ndarray, *, lam: float = 0.05,
                      max_iterations: int = 40, convergence_tolerance: float = 1e-4) -> np.ndarray:
    """ISTA solve of `min ‖H − Φx‖₂² + λ·max|ΦᴴH|·‖x‖₁` → sparse taps x (G,).
    λ is scale-INVARIANT (relative to the matched-filter peak ‖ΦᴴH‖∞), so it transfers across
    captures/gains. Step = 1/L with L = ‖Φ‖₂² (largest singular value²). O(max_iterations·K·G)."""
    H = np.asarray(H, dtype=np.complex64).ravel()
    if H.shape[0] != phi.shape[0]:
        raise CirError(f"estimate_cir_taps: H has {H.shape[0]} subcarriers, Φ expects {phi.shape[0]}")
    if not np.all(np.isfinite(H.view(np.float32))):
        raise CirError("estimate_cir_taps: H contains non-finite values (sanitize first)")
    phi_hermitian = phi.conj().T
    matched = phi_hermitian @ H                 # ΦᴴH — the IFFT/matched-filter seed
    shrink_threshold = float(lam) * float(np.max(np.abs(matched)))  # absolute shrink, scale-invariant in lam
    lipschitz = float(np.linalg.norm(phi, 2)) ** 2 or 1.0
    step = 1.0 / lipschitz
    taps = np.zeros(phi.shape[1], dtype=np.complex64)
    for _ in range(int(max_iterations)):
        gradient = phi_hermitian @ (phi @ taps - H)   # ∇ of the 0.5‖·‖² data term
        next_taps = _soft_threshold(taps - step * gradient, step * shrink_threshold)
        if float(np.linalg.norm(next_taps - taps)) < convergence_tolerance:
            taps = next_taps
            break
        taps = next_taps
    return taps


def cir_from_csi(H: np.ndarray, *, subcarrier_indices: np.ndarray | None = None, oversample: int = 3,
                 subcarrier_spacing_hz: float = SUBCARRIER_SPACING_HZ, lam: float = 0.05,
                 max_iterations: int = 40, convergence_tolerance: float = 1e-4) -> Cir:
    """Recover a super-resolved CIR from one frame's complex CSI H (K,). `subcarrier_indices` = the
    integer tone offsets of the measured subcarriers (default: contiguous 0..K-1; pass real offsets
    for gapped HT40/pilot-masked bands). Fine grid G = oversample·K. Delays τ_g = g/(G·Δf)."""
    H = np.asarray(H, dtype=np.complex64).ravel()
    num_subcarriers = H.shape[0]
    if num_subcarriers == 0:
        raise CirError("cir_from_csi: empty CSI")
    if subcarrier_indices is None:
        subcarrier_indices = np.arange(num_subcarriers)
    subcarrier_indices = np.asarray(subcarrier_indices).ravel()
    if subcarrier_indices.size != num_subcarriers:
        raise CirError(f"cir_from_csi: subcarrier_indices has {subcarrier_indices.size} entries, "
                       f"H has {num_subcarriers} subcarriers")
    num_bins = int(oversample) * num_subcarriers
    phi = delay_dictionary(subcarrier_indices, num_bins)
    taps = estimate_cir_taps(H, phi, lam=lam, max_iterations=max_iterations,
                             convergence_tolerance=convergence_tolerance)
    tap_delays_s = np.arange(num_bins, dtype=np.float64) / (num_bins * subcarrier_spacing_hz)

    power = (taps.real.astype(np.float64) ** 2 + taps.imag.astype(np.float64) ** 2)
    total_power = float(power.sum())
    mean_delay_s = float((tap_delays_s * power).sum() / total_power) if total_power > 0 else 0.0
    rms_delay_spread_s = (float(np.sqrt(((tap_delays_s - mean_delay_s) ** 2 * power).sum() / total_power))
                          if total_power > 0 else 0.0)

    # a tap is a local maximum above the noise floor; dominant_ratio sums each bin into its
    # nearest peak's basin, so leakage split across bins is counted once. O(G).
    noise_floor = power.max() * (10.0 ** (_NOISE_FLOOR_DB / 10.0))
    left_power = np.empty_like(power); left_power[0] = -np.inf; left_power[1:] = power[:-1]
    right_power = np.empty_like(power); right_power[-1] = -np.inf; right_power[:-1] = power[1:]
    peaks = np.flatnonzero((power > noise_floor) & (power >= left_power) & (power >= right_power))
    active_tap_count = int(peaks.size)
    if active_tap_count == 0:
        dominant_bin, dominant_ratio = int(np.argmax(power)), 0.0
    else:
        bins_above_floor = np.flatnonzero(power > noise_floor)
        nearest_peaks = peaks[np.argmin(np.abs(bins_above_floor[:, None] - peaks[None, :]), axis=1)]
        basin_power = np.zeros(active_tap_count)
        for peak_index, bin_index in zip(np.searchsorted(peaks, nearest_peaks), bins_above_floor):
            basin_power[peak_index] += power[bin_index]
        best = int(np.argmax(basin_power))
        dominant_bin = int(peaks[best])
        dominant_ratio = float(basin_power[best] / total_power) if total_power > 0 else 0.0
    return Cir(taps=taps, tap_delays_s=tap_delays_s, subcarrier_spacing_hz=float(subcarrier_spacing_hz),
               dominant_bin=dominant_bin, dominant_ratio=dominant_ratio, rms_delay_spread_s=rms_delay_spread_s,
               active_tap_count=active_tap_count)


def cir_features(cir: Cir) -> np.ndarray:
    """Compact per-frame delay-domain feature vector for a recognition head (float32, len 5):
      0 dominant_tap_ratio · 1 rms_delay_spread_s · 2 active_tap_count ·
      3 dominant_delay_s · 4 |dominant_tap|
    Metal (flat coherent reflector) → high ratio, low spread, few taps; diffuse body → the reverse."""
    return np.array([cir.dominant_ratio, cir.rms_delay_spread_s, float(cir.active_tap_count),
                     cir.dominant_delay_s, abs(cir.dominant_tap)], dtype=np.float32)
