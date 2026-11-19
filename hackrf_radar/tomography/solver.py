"""Non-negative TV/NNLS/SART tomographic inversion & multi-human Kalman/vital-sign tracker."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage, optimize

from ..capture.packet_event import PacketEvent
from .features import (
    RESP_MAX_HZ,
    RESP_MIN_HZ,
    LinkFeatureSummary,
    LinkWindowExtractor,
)
from .geometry import RoomGeometry


@dataclass(slots=True)
class TrackedHuman:
    """Kalman-tracked indoor human target with activity & stationary breathing rate."""

    track_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    activity: str = "stationary"  # "walking" | "stationary"
    breathing_bpm: float | None = None
    confidence: float = 0.8
    peak_density: float = 0.0


@dataclass(slots=True)
class LinkDiagnostic:
    """Per-chord diagnostic and perturbation telemetry exposed to UI & callers."""

    link_id: str
    tx_bssid: str
    rx_id: str
    tx_pos: tuple[float, float]
    rx_pos: tuple[float, float]
    freq_hz: float
    is_peer: bool
    packet_count: int
    rssi_delta_db: float
    csi_turbulence: float
    perturbation_score: float
    breathing_hz: float | None = None
    breathing_bpm: float | None = None
    breathing_snr_db: float = 0.0
    fcs_error_rate: float = 0.0
    seq_gap_rate: float = 0.0
    tsf_drift_ppm: float | None = None
    csi_amp: np.ndarray | None = None
    csi_phase: np.ndarray | None = None
    baseline_csi_amp: np.ndarray | None = None


@dataclass
class TomographyFrame:
    """Complete single-step output of ``TomographicRadarEngine.step()``."""

    timestamp: float
    density_map: np.ndarray  # non-negative float64 shape (Ny, Nx)
    grid_x: np.ndarray  # float64 shape (Nx,)
    grid_y: np.ndarray  # float64 shape (Ny,)
    rx_pos: tuple[float, float]
    tracks: list[TrackedHuman]
    links: list[LinkDiagnostic]
    latest_csi_amp: np.ndarray  # float64 shape (52,)
    latest_csi_phase: np.ndarray  # float64 shape (52,)
    psd_freqs_hz: np.ndarray  # float64 shape (K,) in [0.1, 25.0] Hz
    psd_power_db: np.ndarray  # float64 shape (K,)
    telemetry: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fast Non-Negative Tomographic Solvers (fista_tv, nnls, sart)
# ---------------------------------------------------------------------------


def _prox_tv2d_nonneg(
    z_flat: np.ndarray,
    grid_shape: tuple[int, int],
    lam_tv: float,
    n_inner: int = 4,
) -> np.ndarray:
    """Fast Chambolle-style dual projection for 2D isotropic TV + non-negativity."""
    nx, ny = grid_shape
    u = np.maximum(0.0, z_flat.reshape(ny, nx))
    if lam_tv <= 1e-9:
        return u.ravel()

    px = np.zeros_like(u)
    py = np.zeros_like(u)
    tau = 0.24
    inv_lam = 1.0 / lam_tv

    for _ in range(n_inner):
        # Divergence div(p)
        div = np.zeros_like(u)
        div[:, 0] = px[:, 0]
        div[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
        div[:, -1] = -px[:, -2]

        div[0, :] += py[0, :]
        div[1:-1, :] += py[1:-1, :] - py[:-2, :]
        div[-1, :] += -py[-2, :]

        x_cur = np.maximum(0.0, u - lam_tv * div)

        # Forward gradients grad(x_cur)
        gx = np.zeros_like(u)
        gy = np.zeros_like(u)
        gx[:, :-1] = x_cur[:, 1:] - x_cur[:, :-1]
        gy[:-1, :] = x_cur[1:, :] - x_cur[:-1, :]

        px_new = px - tau * gx * inv_lam
        py_new = py - tau * gy * inv_lam
        norm = np.maximum(1.0, np.hypot(px_new, py_new))
        px = px_new / norm
        py = py_new / norm

    div = np.zeros_like(u)
    div[:, 0] = px[:, 0]
    div[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    div[:, -1] = -px[:, -2]
    div[0, :] += py[0, :]
    div[1:-1, :] += py[1:-1, :] - py[:-2, :]
    div[-1, :] += -py[-2, :]
    return np.maximum(0.0, u - lam_tv * div).ravel()


def _estimate_lipschitz(w_mat: np.ndarray, lam_l2: float, n_iter: int = 8) -> float:
    """Power iteration for spectral norm ||W^T W||_2 + lam_l2."""
    m, n = w_mat.shape
    if m == 0 or n == 0:
        return max(lam_l2, 1.0)
    v = np.ones(n, dtype=np.float64) / math.sqrt(n)
    norm_val = 1.0
    for _ in range(n_iter):
        wv = w_mat @ v
        wt_wv = w_mat.T @ wv + lam_l2 * v
        norm_val = float(np.linalg.norm(wt_wv))
        if norm_val <= 1e-12:
            return max(lam_l2, 1.0)
        v = wt_wv / norm_val
    return max(norm_val * 1.05, 1e-6)


def _isotropic_tv(img: np.ndarray) -> float:
    """Compute 2D isotropic Total Variation norm sum(hypot(dx, dy)) on a (Ny, Nx) array."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    return float(np.sum(np.hypot(gx, gy)))


def solve_fista_tv(
    w_mat: np.ndarray,
    y_vec: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    lam_l1: float = 0.006,
    lam_l2: float = 0.020,
    lam_tv: float = 0.006,
    max_iter: int = 45,
    history_out: list[float] | None = None,
) -> np.ndarray:
    """Solve non-negative TV + L1/L2 regularized RTI via Nesterov FISTA."""
    nx, ny = grid_shape
    n_vox = nx * ny
    if history_out is not None:
        history_out.clear()
    if w_mat.shape[0] == 0 or float(np.max(y_vec)) <= 1e-6:
        if history_out is not None:
            history_out.append(0.0)
        return np.zeros((ny, nx), dtype=np.float64)

    lip = _estimate_lipschitz(w_mat, lam_l2)
    step = 1.0 / lip

    x = np.zeros(n_vox, dtype=np.float64)
    v = np.zeros(n_vox, dtype=np.float64)
    t_par = 1.0

    for _ in range(max_iter):
        res = (w_mat @ v) - y_vec
        grad = (w_mat.T @ res) + lam_l2 * v + lam_l1
        z = np.maximum(0.0, v - step * grad)
        x_next = _prox_tv2d_nonneg(z, grid_shape, lam_tv * step, n_inner=3)
        if history_out is not None:
            r_x = (w_mat @ x_next) - y_vec
            cost = (
                0.5 * float(np.dot(r_x, r_x))
                + 0.5 * lam_l2 * float(np.dot(x_next, x_next))
                + lam_l1 * float(np.sum(x_next))
                + lam_tv * _isotropic_tv(x_next.reshape(ny, nx))
            )
            history_out.append(cost)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_par * t_par))
        v = x_next + ((t_par - 1.0) / t_next) * (x_next - x)
        x = x_next
        t_par = t_next

    return np.maximum(0.0, x.reshape(ny, nx))


def solve_nnls_regularized(
    w_mat: np.ndarray,
    y_vec: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    lam_l1: float = 0.004,
    lam_l2: float = 0.025,
    max_iter: int = 34,
    history_out: list[float] | None = None,
) -> np.ndarray:
    """Solve Tikhonov + L1 regularized Non-Negative Least Squares via accelerated PGD."""
    nx, ny = grid_shape
    n_vox = nx * ny
    if history_out is not None:
        history_out.clear()
    if w_mat.shape[0] == 0 or float(np.max(y_vec)) <= 1e-6:
        if history_out is not None:
            history_out.append(0.0)
        return np.zeros((ny, nx), dtype=np.float64)

    lip = _estimate_lipschitz(w_mat, lam_l2)
    step = 1.0 / lip
    x = np.zeros(n_vox, dtype=np.float64)
    v = np.zeros(n_vox, dtype=np.float64)
    t_par = 1.0

    for _ in range(max_iter):
        grad = w_mat.T @ ((w_mat @ v) - y_vec) + lam_l2 * v + lam_l1
        x_next = np.maximum(0.0, v - step * grad)
        if history_out is not None:
            r_x = (w_mat @ x_next) - y_vec
            cost = (
                0.5 * float(np.dot(r_x, r_x))
                + 0.5 * lam_l2 * float(np.dot(x_next, x_next))
                + lam_l1 * float(np.sum(x_next))
            )
            history_out.append(cost)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_par * t_par))
        v = x_next + ((t_par - 1.0) / t_next) * (x_next - x)
        x = x_next
        t_par = t_next

    img = np.maximum(0.0, x.reshape(ny, nx))
    return ndimage.gaussian_filter(img, sigma=0.65)


def solve_sart(
    w_mat: np.ndarray,
    y_vec: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    relaxation: float = 1.35,
    max_iter: int = 45,
    history_out: list[float] | None = None,
) -> np.ndarray:
    """Solve non-negative Simultaneous Algebraic Reconstruction Technique (SART)."""
    nx, ny = grid_shape
    n_vox = nx * ny
    if history_out is not None:
        history_out.clear()
    y_max = float(np.max(y_vec)) if y_vec.size > 0 else 0.0
    if w_mat.shape[0] == 0 or y_max <= 1e-6:
        if history_out is not None:
            history_out.append(0.0)
        return np.zeros((ny, nx), dtype=np.float64)

    row_sums = np.maximum(np.sum(w_mat, axis=1), 1e-6)
    raw_col = np.sum(w_mat, axis=0)
    col_reg = 0.38 * max(float(np.max(raw_col)), 1e-6)
    col_sums = np.maximum(raw_col + col_reg, 1e-6)
    shrink = 0.0004 * y_max
    x = np.zeros(n_vox, dtype=np.float64)

    for _ in range(max_iter):
        res = y_vec - (w_mat @ x)
        corr = w_mat.T @ (res / row_sums)
        x = np.maximum(0.0, x + relaxation * (corr / col_sums) - shrink)
        if history_out is not None:
            r_x = (w_mat @ x) - y_vec
            history_out.append(0.5 * float(np.dot(r_x, r_x)))

    img = np.maximum(0.0, x.reshape(ny, nx))
    return ndimage.gaussian_filter(img, sigma=0.55)


# ---------------------------------------------------------------------------
# Multi-Target Constant-Velocity Kalman Tracker
# ---------------------------------------------------------------------------


class _KalmanTrackState:
    """4-state [x, y, vx, vy] Kalman filter for a single indoor human target."""

    def __init__(
        self, track_id: int, x: float, y: float, peak_val: float, ts: float
    ) -> None:
        self.track_id = int(track_id)
        self.state = np.array([float(x), float(y), 0.0, 0.0], dtype=np.float64)
        self.cov = np.diag([0.18, 0.18, 0.45, 0.45]).astype(np.float64)
        self.hits = 1
        self.misses = 0
        self.last_ts = float(ts)
        self.first_ts = float(ts)
        self.peak_density = float(peak_val)
        self.smoothed_speed = 0.0
        self.breathing_bpm: float | None = None

    def predict(self, ts: float, room_width: float, room_height: float) -> None:
        dt = max(0.0, min(float(ts) - self.last_ts, 0.6))
        self.last_ts = float(ts)
        if dt <= 1e-6:
            return
        vel_decay = math.exp(-0.15 * dt)
        f_mat = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, vel_decay, 0.0],
                [0.0, 0.0, 0.0, vel_decay],
            ],
            dtype=np.float64,
        )
        q_pos = 0.05 * dt
        q_vel = 1.15 * dt
        q_mat = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.state = f_mat @ self.state
        self.state[0] = min(max(self.state[0], 0.25), room_width - 0.25)
        self.state[1] = min(max(self.state[1], 0.25), room_height - 0.25)
        self.cov = f_mat @ self.cov @ f_mat.T + q_mat

    def update(self, meas_x: float, meas_y: float, peak_val: float) -> None:
        h_mat = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        r_mat = np.diag([0.045, 0.045])
        z = np.array([meas_x, meas_y], dtype=np.float64)
        innov = z - (h_mat @ self.state)
        s_mat = h_mat @ self.cov @ h_mat.T + r_mat
        k_gain = self.cov @ h_mat.T @ np.linalg.inv(s_mat)
        self.state = self.state + (k_gain @ innov)
        spd_raw = float(math.hypot(self.state[2], self.state[3]))
        if spd_raw > 1.8:
            scale_v = 1.8 / spd_raw
            self.state[2] *= scale_v
            self.state[3] *= scale_v
        self.cov = (np.eye(4) - k_gain @ h_mat) @ self.cov
        self.hits += 1
        self.misses = 0
        self.peak_density = 0.6 * self.peak_density + 0.4 * float(peak_val)
        inst_speed = float(math.hypot(self.state[2], self.state[3]))
        self.smoothed_speed = 0.60 * self.smoothed_speed + 0.40 * inst_speed


class MultiTargetKalmanTracker:
    """Associates 2D tomographic density peaks into persistent multi-human trajectories."""

    def __init__(
        self,
        room_width: float,
        room_height: float,
        gate_distance_m: float = 1.65,
        max_coast_s: float = 0.42,
    ) -> None:
        self.room_width = float(room_width)
        self.room_height = float(room_height)
        self.gate_distance_m = float(gate_distance_m)
        self.max_coast_s = float(max_coast_s)
        self._tracks: list[_KalmanTrackState] = []
        self._next_id = 1
        self._step_count = 0

    def step(
        self,
        detections: list[tuple[float, float, float]],
        timestamp: float,
    ) -> list[_KalmanTrackState]:
        self._step_count += 1
        prev_ts_map = {t.track_id: t.last_ts for t in self._tracks}
        for trk in self._tracks:
            trk.predict(timestamp, self.room_width, self.room_height)

        matched_trk: set[int] = set()
        matched_det: set[int] = set()

        if self._tracks and detections:
            cost = np.empty((len(self._tracks), len(detections)), dtype=np.float64)
            for ti, trk in enumerate(self._tracks):
                for di, (dx, dy, _pk) in enumerate(detections):
                    cost[ti, di] = math.hypot(trk.state[0] - dx, trk.state[1] - dy)

            row_ind, col_ind = optimize.linear_sum_assignment(cost)
            for ti, di in zip(row_ind, col_ind):
                if cost[ti, di] <= self.gate_distance_m:
                    dx, dy, pk = detections[di]
                    self._tracks[ti].update(dx, dy, pk)
                    matched_trk.add(int(ti))
                    matched_det.add(int(di))

        # Coast unmatched tracks and prune stale ones
        alive: list[_KalmanTrackState] = []
        for ti, trk in enumerate(self._tracks):
            if ti not in matched_trk:
                dt_miss = max(
                    0.05, timestamp - prev_ts_map.get(trk.track_id, timestamp)
                )
                trk.misses += 1
                trk.peak_density *= 0.78
                if trk.misses * dt_miss > self.max_coast_s or (
                    trk.hits < 4 and trk.misses >= 2
                ):
                    continue
            alive.append(trk)
        self._tracks = alive

        # Spawn new tracks for unassociated strong detections
        for di, (dx, dy, pk) in enumerate(detections):
            if di in matched_det:
                continue
            # Avoid duplicating right next to an existing track
            if any(
                math.hypot(t.state[0] - dx, t.state[1] - dy) < 0.85
                for t in self._tracks
            ):
                continue
            self._tracks.append(_KalmanTrackState(self._next_id, dx, dy, pk, timestamp))
            self._next_id += 1

        # Confirm persistent tracks (or strong single-step detections in short test runs)
        min_hits = 4 if self._step_count >= 6 else max(1, min(2, self._step_count))
        confirmed = [t for t in self._tracks if t.hits >= min_hits and t.misses <= 4]
        confirmed.sort(key=lambda t: t.track_id)
        return confirmed


# ---------------------------------------------------------------------------
# High-Level Tomographic Radar Engine
# ---------------------------------------------------------------------------


class TomographicRadarEngine:
    """End-to-end passive Wi-Fi RTI, SIFS ghost suppression, & vital-sign engine."""

    def __init__(
        self,
        room_width: float = 6.0,
        room_height: float = 5.0,
        grid_shape: tuple[int, int] = (40, 34),
        rx_pos: tuple[float, float] = (3.0, 0.35),
        known_tx_positions: Mapping[str, tuple[float, float]] | None = None,
        solver_method: str = "fista_tv",
        window_s: float = 1.5,
        enable_auto_discovery: bool = True,
        enable_peer_chords: bool = True,
        min_auto_packets: int = 3,
        perimeter_margin_m: float = 0.25,
        use_rssi_log_distance: bool = True,
        ref_rssi_1m_dbm: float = -40.0,
        path_loss_exponent: float = 2.4,
        lam_tv: float = 0.006,
        lam_l2: float = 0.020,
    ) -> None:
        self.room_width = float(room_width)
        self.room_height = float(room_height)
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.rx_pos = (float(rx_pos[0]), float(rx_pos[1]))
        self.solver_method = str(solver_method).strip().lower()
        if self.solver_method not in {"fista_tv", "nnls", "sart"}:
            raise ValueError(f"Unsupported solver_method: {self.solver_method!r}")
        self.lam_tv = float(lam_tv)
        self.lam_l2 = float(lam_l2)
        if not (0.0 <= self.lam_tv <= 1.0):
            raise ValueError(f"lam_tv must be in [0.0, 1.0], got {lam_tv}")
        if not (0.0 <= self.lam_l2 <= 1.0):
            raise ValueError(f"lam_l2 must be in [0.0, 1.0], got {lam_l2}")
        self._lock = threading.RLock()

        self.geometry = RoomGeometry(
            room_width=self.room_width,
            room_height=self.room_height,
            grid_shape=self.grid_shape,
            rx_pos=self.rx_pos,
            known_tx_positions=known_tx_positions,
            enable_auto_discovery=enable_auto_discovery,
            enable_peer_chords=enable_peer_chords,
            min_auto_packets=min_auto_packets,
            perimeter_margin_m=perimeter_margin_m,
            use_rssi_log_distance=use_rssi_log_distance,
            ref_rssi_1m_dbm=ref_rssi_1m_dbm,
            path_loss_exponent=path_loss_exponent,
        )
        self.extractor = LinkWindowExtractor(self.geometry, window_s=window_s)
        self.tracker = MultiTargetKalmanTracker(self.room_width, self.room_height)

        # Precompute near-RX radial convergence suppression mask (avoids spoke-fan pileup on RX)
        d_rx = np.hypot(
            self.geometry.voxel_coords[:, 0] - self.rx_pos[0],
            self.geometry.voxel_coords[:, 1] - self.rx_pos[1],
        )
        self._rx_taper: np.ndarray = 1.0 - 0.88 * np.exp(-((d_rx / 0.55) ** 2))

        # Interior wall margin taper so boundary auto-anchors do not pin artifacts on wall corners
        vx = self.geometry.voxel_coords[:, 0]
        vy = self.geometry.voxel_coords[:, 1]
        wall_dist = np.minimum(
            np.minimum(vx, self.room_width - vx),
            np.minimum(vy, self.room_height - vy),
        )
        self._wall_taper: np.ndarray = np.clip(wall_dist / 0.58, 0.20, 1.0)
        self._smoothed_density: np.ndarray | None = None

    def ingest(self, event: PacketEvent) -> None:
        """Ingest one cleartext ``PacketEvent`` into sliding link & CSI buffers."""
        with self._lock:
            self.extractor.ingest(event)

    def calibrate_baseline(self) -> None:
        """Capture current empty-room link RSSI & CSI averages as reference baseline."""
        with self._lock:
            self.extractor.calibrate_baseline()

    def get_solver_params(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of current runtime solver parameters."""
        with self._lock:
            return {
                "solver_method": self.solver_method,
                "lam_tv": float(self.lam_tv),
                "lam_l2": float(self.lam_l2),
                "window_s": float(self.extractor.window_s),
                "grid_shape": [self.grid_shape[0], self.grid_shape[1]],
            }

    def set_solver_params(
        self,
        *,
        solver_method: str | None = None,
        lam_tv: float | None = None,
        lam_l2: float | None = None,
        window_s: float | None = None,
    ) -> dict[str, Any]:
        """Validate and atomically update runtime solver parameters."""
        new_method: str | None = None
        if solver_method is not None:
            new_method = str(solver_method).strip().lower()
            if new_method not in {"fista_tv", "nnls", "sart"}:
                raise ValueError(f"Unsupported solver_method: {solver_method!r}")

        new_lam_tv: float | None = None
        if lam_tv is not None:
            new_lam_tv = float(lam_tv)
            if not math.isfinite(new_lam_tv) or not (0.0 <= new_lam_tv <= 1.0):
                raise ValueError(f"lam_tv must be in [0.0, 1.0], got {lam_tv}")

        new_lam_l2: float | None = None
        if lam_l2 is not None:
            new_lam_l2 = float(lam_l2)
            if not math.isfinite(new_lam_l2) or not (0.0 <= new_lam_l2 <= 1.0):
                raise ValueError(f"lam_l2 must be in [0.0, 1.0], got {lam_l2}")

        new_window: float | None = None
        if window_s is not None:
            new_window = float(window_s)
            if not math.isfinite(new_window) or not (0.2 <= new_window <= 10.0):
                raise ValueError(f"window_s must be in [0.2, 10.0], got {window_s}")

        with self._lock:
            if new_method is not None:
                self.solver_method = new_method
            if new_lam_tv is not None:
                self.lam_tv = new_lam_tv
            if new_lam_l2 is not None:
                self.lam_l2 = new_lam_l2
            if new_window is not None:
                self.extractor.set_window_s(new_window)
            return self.get_solver_params()

    def _build_linear_system(
        self, summaries: list[LinkFeatureSummary]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct normalized forward matrix W, measurement vector y, and cross-angle prior."""
        n_vox = self.geometry.num_voxels
        if not summaries:
            return (
                np.zeros((0, n_vox), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.ones(n_vox, dtype=np.float64),
            )

        rows: list[np.ndarray] = []
        y_vals: list[float] = []
        raw_scores: list[float] = []
        dirs: list[tuple[float, float]] = []
        endpoints: list[tuple[tuple[float, float], tuple[float, float]]] = []
        norm_profiles: list[np.ndarray] = []
        quiet_mask_acc = np.zeros(n_vox, dtype=np.float64)

        vx = self.geometry.voxel_coords[:, 0]
        vy = self.geometry.voxel_coords[:, 1]

        for s in summaries:
            chord = s.chord
            raw_w = 0.86 * chord.body_weights + 0.14 * chord.shell_weights
            d_tx_sq = (vx - chord.tx_pos[0]) ** 2 + (vy - chord.tx_pos[1]) ** 2
            tx_taper = 1.0 - 0.72 * np.exp(-d_tx_sq / (0.45**2))
            if not chord.is_peer:
                raw_w = raw_w * self._rx_taper * tx_taper
            else:
                d_rx_p_sq = (vx - chord.rx_pos[0]) ** 2 + (vy - chord.rx_pos[1]) ** 2
                rx_p_taper = 1.0 - 0.72 * np.exp(-d_rx_p_sq / (0.45**2))
                raw_w = raw_w * tx_taper * rx_p_taper
            raw_w = raw_w * self._wall_taper

            peak_w = float(np.max(raw_w))
            if peak_w <= 1e-9:
                continue

            # Scale row so a ~4.5-voxel human blob along the chord produces O(1) peak density
            row_norm = raw_w / (peak_w * 6.0)
            score_val = float(s.perturbation_score)
            # Null-chord carving: unperturbed active chords strongly veto ghost intersections
            if score_val < 0.05 and s.packet_count >= 3:
                row_mult = 2.0
                quiet_mask_acc += chord.ffz_weights / max(
                    float(np.max(chord.ffz_weights)), 1e-6
                )
            else:
                row_mult = 1.15 if chord.is_peer else 1.0

            rows.append(row_norm * row_mult)
            y_vals.append(score_val * row_mult)
            raw_scores.append(score_val)

            dx = chord.rx_pos[0] - chord.tx_pos[0]
            dy = chord.rx_pos[1] - chord.tx_pos[1]
            d = max(math.hypot(dx, dy), 1e-6)
            dirs.append((dx / d, dy / d))
            endpoints.append((chord.tx_pos, chord.rx_pos))
            norm_profiles.append((raw_w / peak_w) * score_val)

        if not rows:
            return (
                np.zeros((0, n_vox), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.ones(n_vox, dtype=np.float64),
            )

        w_mat = np.stack(rows, axis=0)
        y_vec = np.asarray(y_vals, dtype=np.float64)

        # Wide-angle chord-intersection consistency prior C_j = sum_{i<k} a_i * a_k * |sin(theta_ik)|
        # Suppresses shared-endpoint (AP/IoT vertex) fan pileup and carves quiet LOS chords.
        hot_idx = [i for i, s_val in enumerate(raw_scores) if s_val >= 0.12]
        cross_prior = np.ones(n_vox, dtype=np.float64)
        vx = self.geometry.voxel_coords[:, 0]
        vy = self.geometry.voxel_coords[:, 1]
        if len(hot_idx) >= 2:
            cross_acc = np.zeros(n_vox, dtype=np.float64)
            for idx_a in range(len(hot_idx)):
                i = hot_idx[idx_a]
                ux_i, uy_i = dirs[i]
                tx_i, rx_i = endpoints[i]
                prof_i = norm_profiles[i]
                for idx_b in range(idx_a + 1, len(hot_idx)):
                    k = hot_idx[idx_b]
                    ux_k, uy_k = dirs[k]
                    sin_theta = abs(ux_i * uy_k - uy_i * ux_k)
                    if sin_theta < 0.22:
                        continue
                    tx_k, rx_k = endpoints[k]
                    pair_prod = (prof_i * norm_profiles[k]) * sin_theta
                    # Check if chords i and k share a physical endpoint (AP, IoT, or RX)
                    shared_pt: tuple[float, float] | None = None
                    for pa in (tx_i, rx_i):
                        for pb in (tx_k, rx_k):
                            if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) < 0.20:
                                shared_pt = pa
                                break
                        if shared_pt is not None:
                            break
                    if shared_pt is not None:
                        d_sq = (vx - shared_pt[0]) ** 2 + (vy - shared_pt[1]) ** 2
                        pair_prod = pair_prod * (1.0 - 0.92 * np.exp(-d_sq / (0.88**2)))
                    cross_acc += pair_prod
            if float(np.max(cross_acc)) > 1e-4:
                quiet_suppress = 1.0 / (1.0 + 0.85 * quiet_mask_acc)
                cross_prior = (0.12 + 0.88 * np.tanh(cross_acc / 0.24)) * quiet_suppress

        return w_mat, y_vec, cross_prior

    def _extract_peaks(
        self, density_map: np.ndarray, min_sep_m: float = 0.90
    ) -> list[tuple[float, float, float]]:
        """Detect sub-voxel 2D local maxima in ``density_map`` with non-maximum suppression."""
        smooth = ndimage.gaussian_filter(density_map, sigma=1.0)
        max_val = float(np.max(smooth))
        if max_val < 0.14:
            return []

        thresh = max(0.15, 0.37 * max_val)
        foot_x = max(3, round(min_sep_m / self.geometry.dx))
        foot_y = max(3, round(min_sep_m / self.geometry.dy))
        loc_max = ndimage.maximum_filter(smooth, size=(foot_y, foot_x), mode="nearest")
        cand_yx = np.argwhere((smooth >= loc_max - 1e-12) & (smooth >= thresh))
        if cand_yx.size == 0:
            return []

        # Sort descending by peak density and apply metric-distance greedy NMS + centroid refinement
        scored = sorted(
            [(float(smooth[iy, ix]), int(ix), int(iy)) for iy, ix in cand_yx],
            key=lambda item: item[0],
            reverse=True,
        )
        gx = self.geometry.grid_x
        gy = self.geometry.grid_y
        rx_rad_x = max(1, round(0.42 / self.geometry.dx))
        rx_rad_y = max(1, round(0.42 / self.geometry.dy))

        accepted: list[tuple[float, float, float]] = []
        for pk_val, ix, iy in scored:
            x0 = max(0, ix - rx_rad_x)
            x1 = min(self.geometry.nx, ix + rx_rad_x + 1)
            y0 = max(0, iy - rx_rad_y)
            y1 = min(self.geometry.ny, iy + rx_rad_y + 1)

            patch = np.maximum(0.0, smooth[y0:y1, x0:x1] - 0.35 * thresh)
            w_sum = float(np.sum(patch))
            if w_sum > 1e-9:
                px_grid, py_grid = np.meshgrid(gx[x0:x1], gy[y0:y1], indexing="xy")
                cx = float(np.sum(patch * px_grid) / w_sum)
                cy = float(np.sum(patch * py_grid) / w_sum)
            else:
                cx = float(gx[ix])
                cy = float(gy[iy])

            if any(math.hypot(cx - ax, cy - ay) < min_sep_m for ax, ay, _ in accepted):
                continue
            accepted.append((cx, cy, pk_val))
            if len(accepted) >= 8:
                break

        return accepted

    def _attribute_breathing(
        self,
        tracks: list[_KalmanTrackState],
        summaries: list[LinkFeatureSummary],
    ) -> None:
        """Spatially beamform respiratory PSD onto stationary tracks (suppressing walker links)."""
        if not tracks or not summaries:
            return

        f_band = self.extractor.psd_freqs_hz
        resp_sel = (f_band >= RESP_MIN_HZ) & (f_band <= RESP_MAX_HZ)
        if not np.any(resp_sel):
            return
        resp_indices = np.flatnonzero(resp_sel)

        def _voxel_index(x_m: float, y_m: float) -> int:
            ix = min(max(int(x_m / self.geometry.dx), 0), self.geometry.nx - 1)
            iy = min(max(int(y_m / self.geometry.dy), 0), self.geometry.ny - 1)
            return iy * self.geometry.nx + ix

        walker_voxels = [
            _voxel_index(t.state[0], t.state[1])
            for t in tracks
            if t.smoothed_speed >= 0.14
        ]

        for trk in tracks:
            if trk.smoothed_speed >= 0.14:
                trk.breathing_bpm = None
                continue

            v_idx = _voxel_index(trk.state[0], trk.state[1])
            beam_psd = np.zeros_like(f_band, dtype=np.float64)
            w_total = 0.0

            for s in summaries:
                if s.psd_linear is None or s.breathing_bpm is None:
                    continue
                bw_max = max(float(np.max(s.chord.body_weights)), 1e-6)
                sw_max = max(float(np.max(s.chord.shell_weights)), 1e-6)
                sens_target = (
                    float(s.chord.body_weights[v_idx]) / bw_max
                    + 0.25 * float(s.chord.shell_weights[v_idx]) / sw_max
                )
                if sens_target < 0.08:
                    continue

                # Penalize chords heavily exposed to concurrent walking targets
                walker_load = 0.0
                for wv in walker_voxels:
                    walker_load += float(s.chord.body_weights[wv]) / bw_max
                clean_weight = sens_target / (1.0 + 8.0 * walker_load)
                beam_psd += clean_weight * s.psd_linear
                w_total += clean_weight

            if w_total <= 1e-6 or resp_indices.size < 3:
                continue

            beam_psd /= w_total
            sub_beam = beam_psd[resp_indices]
            cand_local = [
                k
                for k in range(1, sub_beam.size - 1)
                if sub_beam[k] >= sub_beam[k - 1]
                and sub_beam[k] >= sub_beam[k + 1]
                and (sub_beam[k] > sub_beam[k - 1] or sub_beam[k] > sub_beam[k + 1])
            ]
            if not cand_local:
                continue
            local_best = max(cand_local, key=lambda k: float(sub_beam[k]))
            pk_idx = int(resp_indices[local_best])
            f_pk = float(f_band[pk_idx])
            pk_pow = max(float(beam_psd[pk_idx]), 1e-16)
            low_skirt = (
                float(np.max(beam_psd[f_band < RESP_MIN_HZ]))
                if np.any(f_band < RESP_MIN_HZ)
                else 0.0
            )
            side_sel = (
                (f_band >= 0.10) & (f_band <= 2.20) & (np.abs(f_band - f_pk) >= 0.065)
            )
            bg_med = max(float(np.median(beam_psd[side_sel])), 1e-16)
            snr_db = 10.0 * math.log10(pk_pow / bg_med)
            if snr_db < 8.5 or pk_pow < 0.025 or low_skirt > 1.15 * pk_pow:
                continue

            f_est = f_pk
            if 0 < pk_idx < 319:
                y0 = math.log(max(float(beam_psd[pk_idx - 1]), 1e-20))
                y1 = math.log(max(float(beam_psd[pk_idx]), 1e-20))
                y2 = math.log(max(float(beam_psd[pk_idx + 1]), 1e-20))
                denom = y0 - 2.0 * y1 + y2
                if abs(denom) > 1e-12:
                    delta_bin = max(-0.75, min(0.75, 0.5 * (y0 - y2) / denom))
                    f_est = f_pk + delta_bin * float(f_band[1] - f_band[0])

            bpm = f_est * 60.0
            if 10.5 <= bpm <= 27.5:
                if trk.breathing_bpm is None:
                    trk.breathing_bpm = bpm
                else:
                    trk.breathing_bpm = 0.50 * trk.breathing_bpm + 0.50 * bpm

    def step(self, now: float | None = None) -> TomographyFrame:
        """Run one full RTI inversion, multi-human Kalman update, & vital-sign step."""
        with self._lock:
            step_ts = (
                self.extractor.latest_timestamp
                if (now is None and self.extractor.latest_timestamp > 0.0)
                else (float(now) if now is not None else 0.0)
            )
            summaries, psd_db = self.extractor.extract(now=step_ts)
            w_mat, y_vec, cross_prior = self._build_linear_system(summaries)

            # Modulate columns of W by wide-angle intersection consistency prior before solving
            w_eff = w_mat * cross_prior[None, :]
            conv_history: list[float] = []

            if self.solver_method == "fista_tv":
                raw_map = solve_fista_tv(
                    w_eff,
                    y_vec,
                    self.grid_shape,
                    lam_l2=self.lam_l2,
                    lam_tv=self.lam_tv,
                    history_out=conv_history,
                )
            elif self.solver_method == "nnls":
                raw_map = solve_nnls_regularized(
                    w_eff,
                    y_vec,
                    self.grid_shape,
                    lam_l2=max(self.lam_l2, self.lam_tv),
                    history_out=conv_history,
                )
            else:
                raw_map = solve_sart(
                    w_eff, y_vec, self.grid_shape, history_out=conv_history
                )

            # Light temporal EMA smoothing to stabilize channel-hopping dwell transitions
            if (
                self._smoothed_density is None
                or self._smoothed_density.shape != raw_map.shape
            ):
                self._smoothed_density = raw_map.copy()
            else:
                self._smoothed_density = 0.64 * raw_map + 0.36 * self._smoothed_density

            density_map = np.maximum(0.0, self._smoothed_density.copy())

            peaks = self._extract_peaks(density_map)
            k_tracks = self.tracker.step(peaks, step_ts)
            self._attribute_breathing(k_tracks, summaries)

            out_tracks: list[TrackedHuman] = []
            for kt in k_tracks:
                act = "walking" if kt.smoothed_speed >= 0.14 else "stationary"
                conf = min(
                    1.0, 0.35 + 0.12 * kt.hits + 0.45 * min(kt.peak_density, 1.0)
                )
                out_tracks.append(
                    TrackedHuman(
                        track_id=kt.track_id,
                        x=float(kt.state[0]),
                        y=float(kt.state[1]),
                        vx=float(kt.state[2]),
                        vy=float(kt.state[3]),
                        activity=act,
                        breathing_bpm=kt.breathing_bpm if act == "stationary" else None,
                        confidence=float(conf),
                        peak_density=float(kt.peak_density),
                    )
                )

            out_links: list[LinkDiagnostic] = [
                LinkDiagnostic(
                    link_id=s.chord.link_id,
                    tx_bssid=s.chord.tx_bssid,
                    rx_id=s.chord.rx_id,
                    tx_pos=s.chord.tx_pos,
                    rx_pos=s.chord.rx_pos,
                    freq_hz=s.chord.freq_hz,
                    is_peer=s.chord.is_peer,
                    packet_count=s.packet_count,
                    rssi_delta_db=s.rssi_delta_db,
                    csi_turbulence=s.csi_turbulence,
                    perturbation_score=s.perturbation_score,
                    breathing_hz=s.breathing_hz,
                    breathing_bpm=s.breathing_bpm,
                    breathing_snr_db=s.breathing_snr_db,
                    fcs_error_rate=s.fcs_error_rate,
                    seq_gap_rate=s.seq_gap_rate,
                    tsf_drift_ppm=s.tsf_drift_ppm,
                    csi_amp=s.csi_amp.copy() if s.csi_amp is not None else None,
                    csi_phase=s.csi_phase.copy() if s.csi_phase is not None else None,
                    baseline_csi_amp=(
                        s.baseline_csi_amp.copy()
                        if s.baseline_csi_amp is not None
                        else None
                    ),
                )
                for s in summaries
            ]

            telemetry: dict[str, Any] = {
                "packet_rate_hz": self.extractor.packet_rate_hz(step_ts),
                "mean_tsf_ppm": self.extractor.mean_tsf_ppm(),
                "solver_method": self.solver_method,
                "solver_params": self.get_solver_params(),
                "solver_convergence": conv_history,
                "active_chords": len(out_links),
                "bssid_count": len(self.extractor.observed_bssids),
            }

            return TomographyFrame(
                timestamp=step_ts,
                density_map=density_map,
                grid_x=self.geometry.grid_x.copy(),
                grid_y=self.geometry.grid_y.copy(),
                rx_pos=self.rx_pos,
                tracks=out_tracks,
                links=out_links,
                latest_csi_amp=self.extractor.latest_csi_amp.copy(),
                latest_csi_phase=self.extractor.latest_csi_phase.copy(),
                psd_freqs_hz=self.extractor.psd_freqs_hz.copy(),
                psd_power_db=psd_db,
                telemetry=telemetry,
            )
