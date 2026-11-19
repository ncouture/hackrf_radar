"""Realtime web interface (stdlib HTTP/SSE server and frame serializer) for HackRF Radar."""

from __future__ import annotations

from .serializer import serialize_frame, serialize_frame_json
from .server import RealtimeWebServer, TomographyStateHub, WebRadarDashboard

__all__ = [
    "RealtimeWebServer",
    "TomographyStateHub",
    "WebRadarDashboard",
    "serialize_frame",
    "serialize_frame_json",
]
