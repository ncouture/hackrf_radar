"""2D room grid, anchor placement, First Fresnel Zone (FFZ), and bistatic shell geometry."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

SPEED_OF_LIGHT: float = 299_792_458.0
SUBCARRIER_INDICES: tuple[int, ...] = tuple(list(range(-26, 0)) + list(range(1, 27)))
SUBCARRIER_SPACING_HZ: float = 312_500.0  # 20 MHz / 64 FFT bins

_SUBCARRIER_K_F64: np.ndarray = np.array(SUBCARRIER_INDICES, dtype=np.float64)


_HEX_CHARS: frozenset[str] = frozenset("0123456789ABCDEF")


def canonicalize_mac(mac: str) -> str:
    """Normalize a 48-bit IEEE 802.11 MAC/BSSID string to uppercase colon-hex."""
    cleaned = str(mac).strip().upper()
    if "-" in cleaned and ":" not in cleaned:
        cleaned = cleaned.replace("-", ":")
    parts = cleaned.split(":")
    if len(parts) == 6 and all(
        1 <= len(p) <= 2 and all(ch in _HEX_CHARS for ch in p) for p in parts
    ):
        return ":".join(f"{int(p, 16):02X}" for p in parts)
    return cleaned


def is_unicast_station_mac(mac: str | None) -> bool:
    """Return True if ``mac`` represents a valid non-broadcast/non-multicast station or AP."""
    if not mac:
        return False
    canon = canonicalize_mac(mac)
    if canon in ("FF:FF:FF:FF:FF:FF", "00:00:00:00:00:00", "NONE", "UNKNOWN"):
        return False
    parts = canon.split(":")
    if len(parts) != 6 or not all(
        len(p) == 2 and all(ch in _HEX_CHARS for ch in p) for p in parts
    ):
        return False
    first_octet = int(parts[0], 16)
    # Reject all IEEE 802 group/multicast addresses (I/G bit 0 == 1); locally administered
    # unicast addresses (U/L bit 1 == 1, I/G bit 0 == 0) remain valid.
    return (first_octet & 0x01) == 0


def wavelength_m(freq_hz: float) -> float:
    """Return electromagnetic free-space wavelength in meters for ``freq_hz``."""
    f = max(float(freq_hz), 1e6)
    return SPEED_OF_LIGHT / f


def subcarrier_frequencies_hz(freq_hz: float) -> np.ndarray:
    """Return exact 52 active 802.11 L-LTF subcarrier frequencies in Hz."""
    return float(freq_hz) + _SUBCARRIER_K_F64 * SUBCARRIER_SPACING_HZ


def _band_key(freq_hz: float) -> str:
    """Group frequencies into 2.4 GHz vs 5 GHz sub-bands for wavelength-accurate FFZ caching."""
    f = float(freq_hz)
    if f >= 5.45e9:
        return "5G_HI"
    if f >= 4.9e9:
        return "5G_LO"
    return "2.4G"


def compute_excess_path(
    voxel_coords: np.ndarray,
    tx_pos: tuple[float, float] | np.ndarray,
    rx_pos: tuple[float, float] | np.ndarray,
) -> np.ndarray:
    """Compute bistatic excess path length ``Delta_j = ||x_j - p_tx|| + ||x_j - p_rx|| - d >= 0``.

    Parameters
    ----------
    voxel_coords:
        Coordinates of shape ``(2,)``, ``(N, 2)``, or ``(..., 2)``.
    tx_pos, rx_pos:
        2D endpoint coordinates ``(x_m, y_m)``.

    Returns
    -------
    np.ndarray
        Non-negative excess path lengths in meters with shape ``voxel_coords.shape[:-1]``
        (0-D float64 array when a single ``(2,)`` point is passed). Guaranteed finite.
    """
    pts = np.asarray(voxel_coords, dtype=np.float64)
    if pts.shape[-1] != 2:
        raise ValueError(
            f"voxel_coords last dimension must be 2, got shape {pts.shape}"
        )

    p_tx = np.asarray(tx_pos, dtype=np.float64).reshape(2)
    p_rx = np.asarray(rx_pos, dtype=np.float64).reshape(2)

    d_baseline = float(np.hypot(p_tx[0] - p_rx[0], p_tx[1] - p_rx[1]))
    dx_tx = pts[..., 0] - p_tx[0]
    dy_tx = pts[..., 1] - p_tx[1]
    dx_rx = pts[..., 0] - p_rx[0]
    dy_rx = pts[..., 1] - p_rx[1]

    r_tx = np.hypot(dx_tx, dy_tx)
    r_rx = np.hypot(dx_rx, dy_rx)
    delta = np.maximum(0.0, (r_tx + r_rx) - d_baseline)
    return delta


def compute_ffz_weights(
    voxel_coords: np.ndarray,
    tx_pos: tuple[float, float] | np.ndarray,
    rx_pos: tuple[float, float] | np.ndarray,
    freq_hz: float,
    clearance_scale: float = 1.0,
) -> np.ndarray:
    """Compute singularity-free First Fresnel Zone (FFZ) elliptical tomographic weights.

    Uses exact First Fresnel Zone support ``Delta_ij < lambda / 2`` (for default
    ``clearance_scale=1.0``), C1-smooth raised-cosine decay ``cos^2(pi * Delta / lambda)``,
    and ``d_i^{-1/2}`` link-distance normalization. Guaranteed ``W_ij >= 0`` and finite
    everywhere including directly on ``tx_pos`` and ``rx_pos``.
    """
    p_tx = np.asarray(tx_pos, dtype=np.float64).reshape(2)
    p_rx = np.asarray(rx_pos, dtype=np.float64).reshape(2)
    d_baseline = float(np.hypot(p_tx[0] - p_rx[0], p_tx[1] - p_rx[1]))
    d_safe = max(d_baseline, 0.05)

    lam = wavelength_m(freq_hz)
    delta_max = max(0.5 * float(clearance_scale) * lam, 1e-6)
    delta = compute_excess_path(voxel_coords, p_tx, p_rx)

    weights = np.zeros_like(delta, dtype=np.float64)
    inside = delta < delta_max
    if np.any(inside):
        phase_arg = (0.5 * np.pi / delta_max) * delta[inside]
        weights[inside] = (1.0 / np.sqrt(d_safe)) * (np.cos(phase_arg) ** 2)
    return np.maximum(0.0, weights)


def compute_bistatic_shell_weights(
    voxel_coords: np.ndarray,
    tx_pos: tuple[float, float] | np.ndarray,
    rx_pos: tuple[float, float] | np.ndarray,
    freq_hz: float,
    shell_decay_m: float = 0.45,
    endpoint_softening_m: float = 0.35,
) -> np.ndarray:
    """Compute singularity-free off-LOS bistatic scattering shell weights for CSI multipath.

    Models scattered CSI amplitude/phase sensitivity across iso-range ellipses ``Delta_ij >= 0``
    with wavelength-scaled exponential excess-path decay and softened ``(r_tx * r_rx)^{-1}``
    bistatic radar range weighting. Strictly non-negative and finite at endpoints.
    """
    pts = np.asarray(voxel_coords, dtype=np.float64)
    p_tx = np.asarray(tx_pos, dtype=np.float64).reshape(2)
    p_rx = np.asarray(rx_pos, dtype=np.float64).reshape(2)

    d_baseline = float(np.hypot(p_tx[0] - p_rx[0], p_tx[1] - p_rx[1]))
    d_safe = max(d_baseline, 0.05)

    lam = wavelength_m(freq_hz)
    lam_ref = SPEED_OF_LIGHT / 2437e6  # 2.4 GHz reference wavelength (~0.123 m)
    eff_decay = max(float(shell_decay_m) * np.sqrt(lam / lam_ref), 0.05)
    r0_sq = max(float(endpoint_softening_m), 0.05) ** 2

    dx_tx = pts[..., 0] - p_tx[0]
    dy_tx = pts[..., 1] - p_tx[1]
    dx_rx = pts[..., 0] - p_rx[0]
    dy_rx = pts[..., 1] - p_rx[1]

    r_tx_sq = dx_tx * dx_tx + dy_tx * dy_tx
    r_rx_sq = dx_rx * dx_rx + dy_rx * dy_rx
    delta = np.maximum(0.0, np.sqrt(r_tx_sq) + np.sqrt(r_rx_sq) - d_baseline)

    bistatic_factor = max(d_baseline, 0.25) / (
        np.sqrt(r_tx_sq + r0_sq) * np.sqrt(r_rx_sq + r0_sq)
    )
    shell = (1.0 / np.sqrt(d_safe)) * np.exp(-delta / eff_decay) * bistatic_factor
    return np.maximum(0.0, shell)


def compute_bistatic_csi_phase(
    voxel_coords: np.ndarray,
    tx_pos: tuple[float, float] | np.ndarray,
    rx_pos: tuple[float, float] | np.ndarray,
    freq_hz: float,
) -> np.ndarray:
    """Compute 52-subcarrier bistatic phase progression ``Delta_phi_k = -2*pi*f_k*Delta_ij / c``.

    Constant along iso-range ellipses (``Delta_ij = const``) and linear in subcarrier frequency.
    Returns shape ``(..., 52)`` matching ``voxel_coords.shape[:-1] + (52,)``.
    """
    delta = compute_excess_path(voxel_coords, tx_pos, rx_pos)
    sub_freqs = subcarrier_frequencies_hz(freq_hz)
    return (-2.0 * np.pi / SPEED_OF_LIGHT) * (delta[..., np.newaxis] * sub_freqs)


def compute_bistatic_doppler(
    points: np.ndarray | tuple[float, float],
    velocity: np.ndarray | tuple[float, float],
    tx_pos: tuple[float, float] | np.ndarray,
    rx_pos: tuple[float, float] | np.ndarray,
    freq_hz: float,
) -> np.ndarray:
    """Compute bistatic Doppler shift ``f_D = -(1/lambda) * (u_tx + u_rx) . v`` in Hz.

    Here ``u_tx = (x - p_tx)/||x - p_tx||`` and ``u_rx = (x - p_rx)/||x - p_rx||`` are unit
    propagation vectors from endpoints to the scatterer ``x``. On the forward-scatter baseline
    strictly between TX and RX, ``u_tx + u_rx = 0`` so ``f_D = 0``; broadside motion yields
    maximal ``|f_D| <= 2 ||v|| / lambda``. Singularity-free at endpoints.
    """
    pts = np.asarray(points, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64).reshape(2)
    p_tx = np.asarray(tx_pos, dtype=np.float64).reshape(2)
    p_rx = np.asarray(rx_pos, dtype=np.float64).reshape(2)

    lam = wavelength_m(freq_hz)

    d_tx = pts - p_tx
    d_rx = pts - p_rx
    norm_tx = np.hypot(d_tx[..., 0], d_tx[..., 1])
    norm_rx = np.hypot(d_rx[..., 0], d_rx[..., 1])

    u_tx = np.zeros_like(d_tx, dtype=np.float64)
    u_rx = np.zeros_like(d_rx, dtype=np.float64)

    valid_tx = norm_tx > 1e-9
    valid_rx = norm_rx > 1e-9
    if np.any(valid_tx):
        u_tx[valid_tx] = d_tx[valid_tx] / norm_tx[valid_tx][..., np.newaxis]
    if np.any(valid_rx):
        u_rx[valid_rx] = d_rx[valid_rx] / norm_rx[valid_rx][..., np.newaxis]

    u_sum = u_tx + u_rx
    proj = u_sum[..., 0] * vel[0] + u_sum[..., 1] * vel[1]
    return -(1.0 / lam) * proj


def _perimeter_point_from_unit(
    u: float,
    width: float,
    height: float,
    margin: float,
) -> tuple[float, float]:
    """Map ``u in [0, 1)`` continuously around the inset room rectangle perimeter."""
    m_x = min(max(0.05, margin), 0.35 * width)
    m_y = min(max(0.05, margin), 0.35 * height)
    w_eff = max(width - 2.0 * m_x, 0.1)
    h_eff = max(height - 2.0 * m_y, 0.1)
    perim = 2.0 * (w_eff + h_eff)
    s = (u % 1.0) * perim

    # Traverse North -> East -> West -> South so auto-placed APs favor upper/side walls first
    if s < w_eff:
        return (m_x + s, height - m_y)
    s -= w_eff
    if s < h_eff:
        return (width - m_x, height - m_y - s)
    s -= h_eff
    if s < h_eff:
        return (m_x, m_y + s)
    s -= h_eff
    return (width - m_x - s, m_y)


@dataclass(slots=True)
class LinkChord:
    """Precomputed geometric and sensitivity representation for a single TX -> RX chord."""

    link_id: str
    tx_bssid: str
    rx_id: str
    tx_pos: tuple[float, float]
    rx_pos: tuple[float, float]
    freq_hz: float
    is_peer: bool
    distance_m: float
    ffz_weights: np.ndarray  # shape (Ny * Nx,), strict Delta < lambda/2
    body_weights: np.ndarray  # shape (Ny * Nx,), finite human torso FFZ sensitivity
    shell_weights: (
        np.ndarray
    )  # shape (Ny * Nx,), off-LOS bistatic CSI shell sensitivity


@dataclass
class _AutoNodeState:
    """Internal RSSI accumulator for deterministic ambient BSSID auto-placement."""

    bssid: str
    rssi_samples: list[float] = field(default_factory=list)
    freq_hz: float = 2437e6
    packet_count: int = 0
    position: tuple[float, float] | None = None
    locked: bool = False


class RoomGeometry:
    """Manages the 2D tomographic grid, anchor coordinates, auto-placement, and link chords."""

    def __init__(
        self,
        room_width: float = 6.0,
        room_height: float = 5.0,
        grid_shape: tuple[int, int] = (40, 34),
        rx_pos: tuple[float, float] = (3.0, 0.35),
        known_tx_positions: Mapping[str, tuple[float, float]] | None = None,
        *,
        enable_auto_discovery: bool = True,
        enable_peer_chords: bool = True,
        min_auto_packets: int = 3,
        perimeter_margin_m: float = 0.25,
        use_rssi_log_distance: bool = True,
        ref_rssi_1m_dbm: float = -40.0,
        path_loss_exponent: float = 2.4,
    ) -> None:
        self.room_width = float(room_width)
        self.room_height = float(room_height)
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        if nx < 4 or ny < 4:
            raise ValueError(f"grid_shape must be >= (4, 4), got {grid_shape}")
        self.grid_shape: tuple[int, int] = (nx, ny)
        self.nx = nx
        self.ny = ny
        self.rx_pos: tuple[float, float] = (float(rx_pos[0]), float(rx_pos[1]))

        self.dx = self.room_width / self.nx
        self.dy = self.room_height / self.ny
        self.grid_x: np.ndarray = (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dx
        self.grid_y: np.ndarray = (np.arange(self.ny, dtype=np.float64) + 0.5) * self.dy

        xx, yy = np.meshgrid(self.grid_x, self.grid_y, indexing="xy")
        self.voxel_coords: np.ndarray = np.column_stack([xx.ravel(), yy.ravel()])

        self.enable_auto_discovery = bool(enable_auto_discovery)
        self.enable_peer_chords = bool(enable_peer_chords)
        self.min_auto_packets = max(1, int(min_auto_packets))
        self.perimeter_margin_m = float(perimeter_margin_m)
        self.use_rssi_log_distance = bool(use_rssi_log_distance)
        self.ref_rssi_1m_dbm = float(ref_rssi_1m_dbm)
        self.path_loss_exponent = max(1.5, float(path_loss_exponent))

        self.known_positions: dict[str, tuple[float, float]] = {}
        if known_tx_positions:
            for mac, pos in known_tx_positions.items():
                canon = canonicalize_mac(mac)
                self.known_positions[canon] = (float(pos[0]), float(pos[1]))

        self._auto_nodes: dict[str, _AutoNodeState] = {}
        self._auto_order: list[str] = []
        self._chords: dict[str, LinkChord] = {}

    @property
    def num_voxels(self) -> int:
        return self.nx * self.ny

    def all_node_positions(self) -> dict[str, tuple[float, float]]:
        """Return all known + auto-placed stationary node coordinates."""
        out = dict(self.known_positions)
        for mac, state in self._auto_nodes.items():
            if state.position is not None and mac not in out:
                out[mac] = state.position
        return out

    def get_node_position(self, mac: str | None) -> tuple[float, float] | None:
        """Return ``(x, y)`` if ``mac`` is a known or resolved auto-discovered node."""
        if not is_unicast_station_mac(mac):
            return None
        canon = canonicalize_mac(mac)  # type: ignore[arg-type]
        if canon in self.known_positions:
            return self.known_positions[canon]
        state = self._auto_nodes.get(canon)
        return state.position if state is not None else None

    def observe_node(
        self,
        mac: str | None,
        rssi_dbm: float | None = None,
        freq_hz: float = 2437e6,
    ) -> tuple[float, float] | None:
        """Record an observation of ``mac`` and return its known or auto-placed coordinate."""
        if not is_unicast_station_mac(mac):
            return None
        canon = canonicalize_mac(mac)  # type: ignore[arg-type]
        if canon in self.known_positions:
            return self.known_positions[canon]
        if not self.enable_auto_discovery:
            return None

        state = self._auto_nodes.get(canon)
        if state is None:
            state = _AutoNodeState(bssid=canon, freq_hz=float(freq_hz))
            self._auto_nodes[canon] = state
            self._auto_order.append(canon)

        state.packet_count += 1
        if freq_hz > 0:
            state.freq_hz = float(freq_hz)
        if rssi_dbm is not None and np.isfinite(rssi_dbm) and not state.locked:
            state.rssi_samples.append(float(rssi_dbm))
            if len(state.rssi_samples) >= 20:
                state.locked = True

        if state.packet_count < self.min_auto_packets:
            return None

        new_pos = self._compute_auto_position(state)
        if state.position is None:
            state.position = new_pos
        else:
            shift = float(
                np.hypot(new_pos[0] - state.position[0], new_pos[1] - state.position[1])
            )
            if shift > 0.20 and not state.locked:
                state.position = new_pos
                # Invalidate cached chords involving this node so weights stay aligned
                stale_keys = [
                    k
                    for k, c in self._chords.items()
                    if c.tx_bssid == canon or c.rx_id == canon
                ]
                for k in stale_keys:
                    self._chords.pop(k, None)
        return state.position

    def _compute_auto_position(self, state: _AutoNodeState) -> tuple[float, float]:
        """Deterministically place an unknown BSSID via SHA-256 azimuth + RSSI path-loss."""
        digest = hashlib.sha256(state.bssid.encode("ascii", errors="ignore")).digest()
        hash_u = int.from_bytes(digest[:4], "big") / 4294967296.0
        # Blend deterministic hash with golden-ratio sequence slot to avoid accidental collisions
        try:
            slot_idx = self._auto_order.index(state.bssid)
        except ValueError:
            slot_idx = 0
        golden_u = (0.6180339887498949 * slot_idx + 0.35 * hash_u) % 1.0

        p_perim = _perimeter_point_from_unit(
            golden_u, self.room_width, self.room_height, self.perimeter_margin_m
        )
        # Ensure perimeter point is not right on top of HackRF RX
        dist_rx_perim = float(
            np.hypot(p_perim[0] - self.rx_pos[0], p_perim[1] - self.rx_pos[1])
        )
        if dist_rx_perim < 0.9:
            p_perim = _perimeter_point_from_unit(
                (golden_u + 0.5) % 1.0,
                self.room_width,
                self.room_height,
                self.perimeter_margin_m,
            )
            dist_rx_perim = float(
                np.hypot(p_perim[0] - self.rx_pos[0], p_perim[1] - self.rx_pos[1])
            )

        if not self.use_rssi_log_distance or not state.rssi_samples:
            return p_perim

        med_rssi = float(np.median(np.asarray(state.rssi_samples, dtype=np.float64)))
        # Account for additional 5 GHz free-space 1m reference loss (~6.5 dB vs 2.4 GHz)
        band_offset_db = 6.5 if state.freq_hz >= 4.9e9 else 0.0
        p0_dbm = self.ref_rssi_1m_dbm - band_offset_db
        d_est = float(10.0 ** ((p0_dbm - med_rssi) / (10.0 * self.path_loss_exponent)))
        d_est = min(max(d_est, 0.75), 1.3 * max(self.room_width, self.room_height))

        if dist_rx_perim > 1e-3 and d_est < 0.82 * dist_rx_perim:
            ratio = d_est / dist_rx_perim
            px = self.rx_pos[0] + ratio * (p_perim[0] - self.rx_pos[0])
            py = self.rx_pos[1] + ratio * (p_perim[1] - self.rx_pos[1])
            px = min(
                max(px, self.perimeter_margin_m),
                self.room_width - self.perimeter_margin_m,
            )
            py = min(
                max(py, self.perimeter_margin_m),
                self.room_height - self.perimeter_margin_m,
            )
            return (float(px), float(py))

        return p_perim

    @staticmethod
    def make_link_id(
        tx_bssid: str, rx_id: str = "RX", freq_hz: float | None = None
    ) -> str:
        """Build a canonical link identifier string."""
        tx = canonicalize_mac(tx_bssid)
        rx = "RX" if rx_id == "RX" else canonicalize_mac(rx_id)
        if freq_hz is not None and float(freq_hz) >= 4.9e9:
            ch_mhz = round(float(freq_hz) / 1e6)
            return f"{tx}->{rx}@{ch_mhz}"
        return f"{tx}->{rx}"

    def get_or_create_chord(
        self,
        tx_bssid: str,
        rx_id: str = "RX",
        freq_hz: float = 2437e6,
    ) -> LinkChord | None:
        """Retrieve or construct cached FFZ & bistatic shell weights for ``tx_bssid -> rx_id``."""
        tx_canon = canonicalize_mac(tx_bssid)
        tx_pos = self.get_node_position(tx_canon)
        if tx_pos is None:
            return None

        is_peer = rx_id != "RX"
        if is_peer:
            if not self.enable_peer_chords:
                return None
            rx_canon = canonicalize_mac(rx_id)
            if rx_canon == tx_canon:
                return None
            rx_pos = self.get_node_position(rx_canon)
            if rx_pos is None:
                return None
        else:
            rx_canon = "RX"
            rx_pos = self.rx_pos

        dist = float(np.hypot(tx_pos[0] - rx_pos[0], tx_pos[1] - rx_pos[1]))
        if dist < 0.25:
            return None

        band = _band_key(freq_hz)
        cache_key = f"{tx_canon}->{rx_canon}:{band}"
        cached = self._chords.get(cache_key)
        if cached is not None:
            return cached

        link_id = self.make_link_id(tx_canon, rx_canon, freq_hz)
        ffz_w = compute_ffz_weights(
            self.voxel_coords, tx_pos, rx_pos, freq_hz, clearance_scale=1.0
        )
        # Human torso cross-section (~0.30 m diameter) slightly widens high-frequency 5 GHz chords
        lam = wavelength_m(freq_hz)
        torso_clearance = max(1.0, (0.5 * lam + 0.035) / (0.5 * lam))
        body_w = compute_ffz_weights(
            self.voxel_coords, tx_pos, rx_pos, freq_hz, clearance_scale=torso_clearance
        )
        shell_w = compute_bistatic_shell_weights(
            self.voxel_coords, tx_pos, rx_pos, freq_hz, shell_decay_m=0.45
        )

        chord = LinkChord(
            link_id=link_id,
            tx_bssid=tx_canon,
            rx_id=rx_canon,
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            freq_hz=float(freq_hz),
            is_peer=is_peer,
            distance_m=dist,
            ffz_weights=ffz_w,
            body_weights=body_w,
            shell_weights=shell_w,
        )
        self._chords[cache_key] = chord
        return chord
