"""Tap alignment for UWB CIR windows (R RX chains on one chip, N taps each).

The chip moves its window start between packets, so the same path lands on different taps. Two
alignments: an integer re-anchor on each RX's first path for per-tap features, and a cross-RX
cross-spectrum in which any common window shift cancels.
"""

import numpy as np


def reanchor_taps(taps: np.ndarray, first_path_columns: np.ndarray, anchor_column: int,
                  out: np.ndarray | None = None) -> np.ndarray:
    """(R, N) taps -> (R, N) taps with row r shifted so its column first_path_columns[r]
    (= first_path_index − window_start_index) lands on anchor_column; taps shifted past an edge are
    dropped, vacated columns are 0. Written into `out` (allocated only if None). O(R·N), no FFT;
    residual jitter is the sub-tap part of the true first-path delay (≤ ½ tap)."""
    num_rx, num_taps = taps.shape
    if not np.issubdtype(np.asarray(first_path_columns).dtype, np.integer):
        raise TypeError("reanchor_taps: first_path_columns must be integer tap columns")
    if out is None:
        out = np.empty_like(taps)
    elif not np.iscomplexobj(out):
        raise TypeError("reanchor_taps: out must be complex")
    elif np.may_share_memory(out, taps):
        # out is zeroed before taps are read, so an aliased out would return zeros
        raise ValueError("reanchor_taps: out must not share memory with taps")
    out.fill(0)
    for rx in range(num_rx):
        shift = anchor_column - int(first_path_columns[rx])
        if shift >= 0:
            if shift < num_taps:
                out[rx, shift:] = taps[rx, :num_taps - shift]
        elif -shift < num_taps:
            out[rx, :num_taps + shift] = taps[rx, -shift:]
    return out


def cross_spectrum(taps: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """(R, N) taps, R ≥ 2, as received (not re-anchored) -> (R−1, N) complex
    X_r[k] = H_r[k]·conj(H_0[k]), H_r = fft(taps_r), for r = 1..R−1. Written into `out` (allocated
    only if None). Exact while no path leaves the window; per-path delay is lost. O(R·N log N)."""
    if taps.shape[0] < 2:
        raise ValueError(f"cross_spectrum: needs >= 2 RX rows, got {taps.shape[0]}")
    # the RX chains share one LO, so CFO and timing phase are common to every row and cancel in
    # H_r·conj(H_0); a common window shift Δ is the common ramp e^{−j2πkΔ/N}, which cancels too
    spectra = np.fft.fft(taps, axis=-1)
    if out is None:
        out = np.empty((taps.shape[0] - 1, taps.shape[1]), dtype=spectra.dtype)
    np.multiply(spectra[1:], np.conj(spectra[0]), out=out)
    return out
