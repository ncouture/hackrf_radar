"""Passive 0 dBm RX-only 20 MSPS 802.11 packet, L-LTF CSI, and simulator capture layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .packet_event import (
    PILOT_INDICES,
    SUBCARRIER_INDICES,
    VALID_FRAME_TYPES,
    WIFI_CHANNELS_HZ,
    PacketEvent,
    channel_to_freq_hz,
    freq_hz_to_channel,
    sanitize_csi,
)
from .soapy_source import (
    HACKRF_BASEBAND_BWS_HZ,
    HackRFIQSource,
    IQChunk,
    compute_adc_clip_ratio_cf32,
    compute_adc_clip_ratio_int8,
    quantize_lna_gain,
    quantize_vga_gain,
    select_baseband_bw,
)

if TYPE_CHECKING:
    from .gr_wifi_bridge import HackRFPacketCapture
    from .simulator import SyntheticRadarStream

__all__ = [
    "HACKRF_BASEBAND_BWS_HZ",
    "PILOT_INDICES",
    "SUBCARRIER_INDICES",
    "VALID_FRAME_TYPES",
    "WIFI_CHANNELS_HZ",
    "HackRFIQSource",
    "HackRFPacketCapture",
    "IQChunk",
    "PacketEvent",
    "SyntheticRadarStream",
    "channel_to_freq_hz",
    "compute_adc_clip_ratio_cf32",
    "compute_adc_clip_ratio_int8",
    "freq_hz_to_channel",
    "quantize_lna_gain",
    "quantize_vga_gain",
    "sanitize_csi",
    "select_baseband_bw",
]


def __getattr__(name: str) -> Any:
    if name == "HackRFPacketCapture":
        from .gr_wifi_bridge import HackRFPacketCapture

        return HackRFPacketCapture
    if name == "SyntheticRadarStream":
        from .simulator import SyntheticRadarStream

        return SyntheticRadarStream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
