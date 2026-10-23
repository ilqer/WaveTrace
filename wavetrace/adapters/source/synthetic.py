"""In-memory frame replay for dev and CI."""

from wavetrace.application.ports import CsiSource


class SyntheticSource(CsiSource):
    def __init__(self, frames):
        self._frames = list(frames)

    def frames(self):
        return iter(self._frames)


def build_synthetic_source(*, num_antennas: int, num_subcarriers: int, sample_rate_hz: float, duration_s: float,
                           presence_spans: list[tuple[float, float]], weapon_spans: list[tuple[float, float]],
                           weapon_signature_depth: float, seed: int) -> SyntheticSource:
    """Seeded paired recording from `wavetrace.synthetic`; spans are (start_s, end_s). An empty
    `presence_spans` puts a person in the room for the whole recording."""
    from wavetrace.synthetic import generate_paired_recording
    frames, _, _ = generate_paired_recording(
        num_antennas=num_antennas, num_subcarriers=num_subcarriers, sample_rate_hz=sample_rate_hz,
        duration_s=duration_s, camera_frame_rate_hz=30.0,
        presence_spans=presence_spans or [(0.0, duration_s)],
        presence_turbulence_std=0.10, weapon_spans=weapon_spans,
        weapon_signature_depth=weapon_signature_depth, seed=seed,
    )
    return SyntheticSource(frames)
