from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class CirRecord:
    """One UWB packet's CIR from one node: R RX chains of one chip, N taps each. eq=False because the
    array fields have no scalar equality. Tap column c is chip CIR index `window_start_index + c`."""

    node_id: int
    ntp_ms: int
    peer_short_addr: int
    timestamp_us: int
    tap_spacing_s: float
    window_start_index: int
    first_path_indices: np.ndarray    # int16 (R,), in the chip's whole CIR, not the window
    distance_m: float
    aoa_azimuth_deg: float
    aoa_elevation_deg: float
    pdoa_deg: float
    pdoa_index: int
    first_path_snr_db: np.ndarray     # uint8 (R,)
    taps: np.ndarray                  # complex64 (R, N)

    @property
    def first_path_columns(self) -> np.ndarray:
        """int32 (R,): each RX's first-path column inside `taps`; int32 so the subtraction cannot wrap int16."""
        return self.first_path_indices.astype(np.int32) - self.window_start_index
