"""Offline delay-domain super-resolution (Channel Impulse Response) via L1 sparse recovery.

Stage-E weapon tool (plan §1, REFERENCE_DIGEST §0C, RuView ADR-134). The wideband channel is sparse
in the delay domain (~few physical paths), so the K measured subcarriers H(f_k) can be inverted to a
fine-grid CIR `x` whose taps are the multipath arrivals — incl. a concealed metal object's reflection.

Why ISTA on a DFT dictionary, not a plain IFFT:
  * A zero-padded IFFT smears each tap with -13 dB sidelobes — adjacent taps (body vs. object) merge.
  * Φ is a *sub*-DFT (one column per fine delay bin); solving `min ‖H-Φx‖² + λ‖x‖₁` super-resolves the
    delay axis ~`oversample`× past the 1/BW Nyquist limit (ADR-134 §2.1). κ(Φ)≈1 by construction.
  * ISTA over OMP: when specular + body reflections fall in one Nyquist bin, OMP's greedy commit is
    irreversible; ISTA's continuous shrinkage degrades gracefully at low SNR (ADR-134 §2.2).

Φ atoms are built from the MEASURED subcarrier indices, so non-contiguous layouts (HT40's central
null gap, masked pilots) are handled natively — no separate subcarrier-infill step is needed.

PHASE PRECONDITION (ADR-134 §2.5): pass phase-sanitized H (conj-mult / CFO-removed, e.g. via
reconstruct_complex_csi). Raw STO/CFO ramps fit as ghost taps near τ=0.

Offline only (never the hot path): dictionary build O(K·G) once; each estimate O(n_iter·K·G).
"""

from dataclasses import dataclass

import numpy as np

SUBCARRIER_SPACING_HZ = 312_500.0  # 802.11n HT20/HT40 OFDM tone spacing
_NOISE_FLOOR_DB = -25.0            # tap power below this (rel. to peak) = noise (ADR-134 §2.9)


class CirError(ValueError):
    """Raised on malformed CIR inputs (bad shapes, empty bands, non-finite CSI)."""


@dataclass(frozen=True, slots=True)
class Cir:
    """A recovered delay-domain channel impulse response and its summary statistics."""

    taps: np.ndarray            # complex (G,) — fine-grid tap amplitudes; index 0 = zero delay
    tap_delays_s: np.ndarray    # float (G,) — delay of each bin in seconds
    df_hz: float                # subcarrier spacing used
    dominant_idx: int           # argmax |tap|^2
    dominant_ratio: float       # peak power / total power (high => coherent reflector / strong LOS)
    rms_delay_spread_s: float   # power-weighted RMS spread (diffuse body => large; flat metal => small)
    active_tap_count: int       # taps above the noise floor

    @property
    def dominant_delay_s(self) -> float:
        return float(self.tap_delays_s[self.dominant_idx])

    @property
    def dominant_tap(self) -> complex:
        return complex(self.taps[self.dominant_idx])


def delay_dictionary(freq_idx: np.ndarray, n_bins: int) -> np.ndarray:
    """Sub-DFT sensing matrix Φ ∈ ℂ^{K×G}: column g is the delay atom for fine bin g, sampled at the
    MEASURED subcarrier offsets `freq_idx` (integer tone indices; supplies the band geometry / gaps).
    Δf cancels in the matrix (it only scales bin→seconds), so atoms are exp(-j2π·freq_idx·g/G).
    Normalised by 1/√K so ΦΦᴴ ≈ (G/K)·I (well-conditioned). O(K·G)."""
    freq_idx = np.asarray(freq_idx, dtype=np.float64).ravel()
    k = freq_idx.size
    if k == 0 or n_bins <= 0:
        raise CirError("delay_dictionary: need >=1 subcarrier and n_bins>0")
    g = np.arange(n_bins, dtype=np.float64)
    phase = -2.0 * np.pi * np.outer(freq_idx, g) / float(n_bins)  # (K, G)
    return (np.exp(1j * phase) / np.sqrt(k)).astype(np.complex64)


def _soft_threshold(z: np.ndarray, thr: float) -> np.ndarray:
    """Complex soft-threshold: shrink each magnitude by `thr`, keep phase. The ISTA prox of λ‖·‖₁."""
    mag = np.abs(z)
    scale = np.maximum(0.0, 1.0 - thr / np.maximum(mag, 1e-12))
    return (z * scale).astype(z.dtype)


def estimate_cir_taps(H: np.ndarray, phi: np.ndarray, *, lam: float = 0.05,
                      n_iter: int = 40, tol: float = 1e-4) -> np.ndarray:
    """ISTA solve of `min ‖H − Φx‖₂² + λ·max|ΦᴴH|·‖x‖₁` → sparse taps x (G,).
    λ is scale-INVARIANT (relative to the matched-filter peak ‖ΦᴴH‖∞), so it transfers across
    captures/gains. Step = 1/L with L = ‖Φ‖₂² (largest singular value²). O(n_iter·K·G)."""
    H = np.asarray(H, dtype=np.complex64).ravel()
    if H.shape[0] != phi.shape[0]:
        raise CirError(f"estimate_cir_taps: H has {H.shape[0]} subcarriers, Φ expects {phi.shape[0]}")
    if not np.all(np.isfinite(H.view(np.float32))):
        raise CirError("estimate_cir_taps: H contains non-finite values (sanitize first)")
    phi_h = phi.conj().T
    matched = phi_h @ H                          # ΦᴴH — the IFFT/matched-filter seed
    thr = float(lam) * float(np.max(np.abs(matched)))  # absolute shrink, scale-invariant in lam
    L = float(np.linalg.norm(phi, 2)) ** 2 or 1.0
    step = 1.0 / L
    x = np.zeros(phi.shape[1], dtype=np.complex64)
    for _ in range(int(n_iter)):
        grad = phi_h @ (phi @ x - H)             # ∇ of the 0.5‖·‖² data term
        x_new = _soft_threshold(x - step * grad, step * thr)
        if float(np.linalg.norm(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    return x


def cir_from_csi(H: np.ndarray, *, freq_idx: np.ndarray | None = None, oversample: int = 3,
                 df_hz: float = SUBCARRIER_SPACING_HZ, lam: float = 0.05, n_iter: int = 40,
                 tol: float = 1e-4) -> Cir:
    """Recover a super-resolved CIR from one frame's complex CSI H (K,). `freq_idx` = the integer tone
    offsets of the measured subcarriers (default: contiguous 0..K-1; pass real offsets for gapped
    HT40/pilot-masked bands). Fine grid G = oversample·K. Delays τ_g = g/(G·Δf)."""
    H = np.asarray(H, dtype=np.complex64).ravel()
    k = H.shape[0]
    if k == 0:
        raise CirError("cir_from_csi: empty CSI")
    if freq_idx is None:
        freq_idx = np.arange(k)
    freq_idx = np.asarray(freq_idx).ravel()
    if freq_idx.size != k:
        raise CirError(f"cir_from_csi: freq_idx has {freq_idx.size} entries, H has {k} subcarriers")
    g = int(oversample) * k
    phi = delay_dictionary(freq_idx, g)
    taps = estimate_cir_taps(H, phi, lam=lam, n_iter=n_iter, tol=tol)
    delays = np.arange(g, dtype=np.float64) / (g * df_hz)

    power = (taps.real.astype(np.float64) ** 2 + taps.imag.astype(np.float64) ** 2)
    total = float(power.sum())
    mean_tau = float((delays * power).sum() / total) if total > 0 else 0.0
    rms = float(np.sqrt(((delays - mean_tau) ** 2 * power).sum() / total)) if total > 0 else 0.0

    # An off-grid tap leaks across adjacent fine bins, so a TAP = a LOCAL MAXIMUM above the floor
    # (the "tolerance-aware tap-peak detector" ADR-134 §2.9 requires). dominant_ratio sums each bin
    # into its nearest peak's basin, so split leakage is credited to one tap. O(G).
    floor = power.max() * (10.0 ** (_NOISE_FLOOR_DB / 10.0))
    left = np.empty_like(power); left[0] = -np.inf; left[1:] = power[:-1]
    right = np.empty_like(power); right[-1] = -np.inf; right[:-1] = power[1:]
    peaks = np.flatnonzero((power > floor) & (power >= left) & (power >= right))
    active = int(peaks.size)
    if active == 0:
        dom, dom_ratio = int(np.argmax(power)), 0.0
    else:
        above = np.flatnonzero(power > floor)
        nearest = peaks[np.argmin(np.abs(above[:, None] - peaks[None, :]), axis=1)]
        basin = np.zeros(active)
        for p_i, b in zip(np.searchsorted(peaks, nearest), above):
            basin[p_i] += power[b]
        best = int(np.argmax(basin))
        dom = int(peaks[best])
        dom_ratio = float(basin[best] / total) if total > 0 else 0.0
    return Cir(taps=taps, tap_delays_s=delays, df_hz=float(df_hz), dominant_idx=dom,
               dominant_ratio=dom_ratio, rms_delay_spread_s=rms, active_tap_count=active)


def cir_features(cir: Cir) -> np.ndarray:
    """Compact per-frame delay-domain feature vector for a recognition head (float32, len 5):
      0 dominant_tap_ratio · 1 rms_delay_spread_s · 2 active_tap_count ·
      3 dominant_delay_s · 4 |dominant_tap|
    Metal (flat coherent reflector) → high ratio, low spread, few taps; diffuse body → the reverse."""
    return np.array([cir.dominant_ratio, cir.rms_delay_spread_s, float(cir.active_tap_count),
                     cir.dominant_delay_s, abs(cir.dominant_tap)], dtype=np.float32)
