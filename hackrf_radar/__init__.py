"""Passive 0 dBm RX-only Wi-Fi Radio Tomographic Imaging (RTI) & CSI radar for HackRF One."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import (
    AutoDiscoverySettings,
    RadarConfig,
    RoomAnchor,
    RxSettings,
    SolverSettings,
    load_config,
    save_config,
)

__version__ = "0.1.0"

__all__ = [
    "PILOT_INDICES",
    "SUBCARRIER_INDICES",
    "WIFI_CHANNELS_HZ",
    "AutoDiscoverySettings",
    "HackRFPacketCapture",
    "LinkDiagnostic",
    "MatplotlibRadarDashboard",
    "PacketEvent",
    "PyQtGraphRadarDashboard",
    "RadarConfig",
    "RichTuiRadarDashboard",
    "RoomAnchor",
    "RxSettings",
    "SolverSettings",
    "SyntheticRadarStream",
    "TomographicRadarEngine",
    "TomographyFrame",
    "TrackedHuman",
    "__version__",
    "create_plotter",
    "load_config",
    "save_config",
    "select_ui_backend",
]

if TYPE_CHECKING:
    from .capture import (
        PILOT_INDICES,
        SUBCARRIER_INDICES,
        WIFI_CHANNELS_HZ,
        HackRFPacketCapture,
        PacketEvent,
        SyntheticRadarStream,
    )
    from .plotter import (
        MatplotlibRadarDashboard,
        PyQtGraphRadarDashboard,
        RichTuiRadarDashboard,
        create_plotter,
        select_ui_backend,
    )
    from .tomography import (
        LinkDiagnostic,
        TomographicRadarEngine,
        TomographyFrame,
        TrackedHuman,
    )


_CAPTURE_EXPORTS = {
    "PacketEvent",
    "SUBCARRIER_INDICES",
    "PILOT_INDICES",
    "WIFI_CHANNELS_HZ",
    "SyntheticRadarStream",
    "HackRFPacketCapture",
}

_TOMO_EXPORTS = {
    "TrackedHuman",
    "LinkDiagnostic",
    "TomographyFrame",
    "TomographicRadarEngine",
}

_PLOTTER_EXPORTS = {
    "MatplotlibRadarDashboard",
    "PyQtGraphRadarDashboard",
    "RichTuiRadarDashboard",
    "create_plotter",
    "select_ui_backend",
}


def __getattr__(name: str) -> Any:
    if name in _CAPTURE_EXPORTS:
        from . import capture

        return getattr(capture, name)
    if name in _TOMO_EXPORTS:
        from . import tomography

        return getattr(tomography, name)
    if name in _PLOTTER_EXPORTS:
        from . import plotter

        return getattr(plotter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
