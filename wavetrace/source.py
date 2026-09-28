"""Compatibility path: the golden test imports `wavetrace.source`, so the old module keeps its names.
The CSI sources live in `wavetrace.adapters.source`, `resample_uniform` in `wavetrace.domain.dsp`."""

from wavetrace.adapters.source import (ESP_CSI_COLUMNS, NexmonSource, NexmonSourceOptions, RecordingSource,
                                       SerialReader, SerialSourceOptions, SourceOptions, SyntheticSource,
                                       UdpSource, UdpSourceOptions, bind_udp_socket, build_csi_source,
                                       load_recording, parse_batch, parse_batch_by_link, parse_csi_line,
                                       parse_time_spans, save_recording, short_mac)
from wavetrace.domain.dsp.resample import resample_uniform

__all__ = [
    "ESP_CSI_COLUMNS", "parse_csi_line", "parse_batch", "parse_batch_by_link", "short_mac",
    "resample_uniform", "bind_udp_socket", "SourceOptions", "SyntheticSource", "RecordingSource",
    "UdpSourceOptions", "UdpSource", "SerialSourceOptions", "SerialReader", "NexmonSourceOptions",
    "NexmonSource", "save_recording", "load_recording", "parse_time_spans", "build_csi_source",
]
