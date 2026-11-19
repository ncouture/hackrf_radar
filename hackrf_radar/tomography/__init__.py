"""First Fresnel Zone geometry, link CSI/PSD feature extraction, & RTI solvers."""

from __future__ import annotations

from .features import LinkFeatureSummary, LinkWindowExtractor
from .geometry import (
    SPEED_OF_LIGHT,
    SUBCARRIER_INDICES,
    SUBCARRIER_SPACING_HZ,
    LinkChord,
    RoomGeometry,
    canonicalize_mac,
    compute_bistatic_csi_phase,
    compute_bistatic_doppler,
    compute_bistatic_shell_weights,
    compute_excess_path,
    compute_ffz_weights,
    is_unicast_station_mac,
    subcarrier_frequencies_hz,
    wavelength_m,
)
from .solver import (
    LinkDiagnostic,
    MultiTargetKalmanTracker,
    TomographicRadarEngine,
    TomographyFrame,
    TrackedHuman,
    solve_fista_tv,
    solve_nnls_regularized,
    solve_sart,
)

__all__ = [
    "SPEED_OF_LIGHT",
    "SUBCARRIER_INDICES",
    "SUBCARRIER_SPACING_HZ",
    "LinkChord",
    "LinkDiagnostic",
    "LinkFeatureSummary",
    "LinkWindowExtractor",
    "MultiTargetKalmanTracker",
    "RoomGeometry",
    "TomographicRadarEngine",
    "TomographyFrame",
    "TrackedHuman",
    "canonicalize_mac",
    "compute_bistatic_csi_phase",
    "compute_bistatic_doppler",
    "compute_bistatic_shell_weights",
    "compute_excess_path",
    "compute_ffz_weights",
    "is_unicast_station_mac",
    "solve_fista_tv",
    "solve_nnls_regularized",
    "solve_sart",
    "subcarrier_frequencies_hz",
    "wavelength_m",
]
