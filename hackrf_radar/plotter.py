"""Multi-Human Live Tomographic Radar Dashboard (PyQtGraph, Matplotlib, and Rich TUI).

Supports:
- ``pyqtgraph``: High-FPS Qt radar display (supports X11 and ``QT_QPA_PLATFORM=offscreen``).
- ``matplotlib``: 4-panel scientific dashboard (supports interactive X11 and headless ``Agg``).
- ``tui``: Rich live ANSI Truecolor Unicode half-block 2D heatmap + telemetry HUD for SSH/TTY.
- ``auto``: Picks ``pyqtgraph`` when installed & GUI active, else ``matplotlib``, or ``tui`` when ``--no-gui``.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import shutil
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .tomography.geometry import canonicalize_mac, is_unicast_station_mac

SPEED_OF_LIGHT_MPS: float = 299_792_458.0
DEFAULT_SUBCARRIER_INDICES: np.ndarray = np.array(
    list(range(-26, 0)) + list(range(1, 27)), dtype=np.float64
)
PILOT_SUBCARRIERS: tuple[int, ...] = (-21, -7, 7, 21)
SPARK_BLOCKS: str = " ▁▂▃▄▅▆▇█"


def _build_gui_backend_errors() -> tuple[type[Exception], ...]:
    errs: list[type[Exception]] = [RuntimeError, OSError, ValueError, AttributeError]
    with contextlib.suppress(ImportError):
        import _tkinter  # type: ignore[import-not-found]

        if isinstance(getattr(_tkinter, "TclError", None), type) and issubclass(
            _tkinter.TclError, Exception
        ):
            errs.append(_tkinter.TclError)
    return tuple(errs)


_GUI_BACKEND_ERRORS: tuple[type[Exception], ...] = _build_gui_backend_errors()


def has_gui_display() -> bool:
    """Return True when an X11/Wayland display or explicit offscreen Qt platform is configured."""
    if os.environ.get("QT_QPA_PLATFORM", "").lower() in {"offscreen", "minimal"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def pyqtgraph_available() -> bool:
    """Return True if pyqtgraph and a working Qt binding can be imported."""
    try:
        import pyqtgraph  # noqa: F401
    except (ImportError, RuntimeError, OSError):
        return False
    return True


def select_ui_backend(requested: str = "auto", no_gui: bool = False) -> str:
    """Resolve ``auto|pyqtgraph|matplotlib|tui|none`` to a concrete UI backend name."""
    req = (requested or "auto").strip().lower()
    if req in {"none", "headless"}:
        return "matplotlib"
    if req in {"pyqtgraph", "matplotlib", "tui"}:
        if req == "pyqtgraph" and not pyqtgraph_available():
            raise RuntimeError(
                "Requested --ui pyqtgraph, but pyqtgraph/Qt is not installed in this Python "
                "interpreter. Use /usr/bin/python3 or pass --ui matplotlib / --ui tui."
            )
        return req
    if req != "auto":
        raise ValueError(
            f"Unknown UI backend {requested!r}; expected auto, pyqtgraph, matplotlib, or tui."
        )

    if no_gui:
        return "tui"
    if has_gui_display() and pyqtgraph_available():
        return "pyqtgraph"
    if has_gui_display() or os.environ.get("MPLBACKEND", "").lower() == "agg":
        return "matplotlib"
    return "tui"


def compute_fresnel_ellipse(
    tx_pos: tuple[float, float],
    rx_pos: tuple[float, float],
    freq_hz: float = 2437e6,
    n_pts: int = 29,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute First Fresnel Zone boundary (excess path Delta = lambda / 2) in 2D meters."""
    dx = float(rx_pos[0]) - float(tx_pos[0])
    dy = float(rx_pos[1]) - float(tx_pos[1])
    d = math.hypot(dx, dy)
    if d < 1e-3 or freq_hz <= 0.0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    wavelength = SPEED_OF_LIGHT_MPS / float(freq_hz)
    # 2*a = d + lambda/2 => a = d/2 + lambda/4, c_f = d/2
    a = 0.5 * d + 0.25 * wavelength
    cf = 0.5 * d
    b_sq = max(a * a - cf * cf, 1e-6)
    b = math.sqrt(b_sq)

    ux = dx / d
    uy = dy / d
    nx = -uy
    ny = ux
    mx = 0.5 * (float(tx_pos[0]) + float(rx_pos[0]))
    my = 0.5 * (float(tx_pos[1]) + float(rx_pos[1]))

    theta = np.linspace(0.0, 2.0 * math.pi, n_pts, dtype=np.float64)
    ct = np.cos(theta)
    st = np.sin(theta)
    ex = mx + a * ct * ux + b * st * nx
    ey = my + a * ct * uy + b * st * ny
    return ex, ey


def _concat_segments_with_nan(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate 2D polylines separated by NaN for single-artist fast rendering."""
    if not segments:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    nan1 = np.array([np.nan], dtype=np.float64)
    for sx, sy in segments:
        if sx.size == 0:
            continue
        xs.append(sx)
        xs.append(nan1)
        ys.append(sy)
        ys.append(nan1)
    if not xs:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(xs), np.concatenate(ys)


@dataclass
class _HudMetrics:
    packet_rate_hz: float
    bssid_count: int
    chord_count: int
    peer_chord_count: int
    beacons: int
    cts: int
    ack: int
    block_ack: int
    probes: int
    data_frames: int
    fcs_drop_pct: float
    seq_gaps: int
    seq_gap_rate: float
    mean_tsf_ppm: float | None
    adc_clip_max: float
    adc_status: str
    backend_label: str
    peak_resp_hz: float | None
    peak_resp_bpm: float | None


def _extract_hud_metrics(
    frame: Any, default_backend: str = "passive-rx"
) -> _HudMetrics:
    tel: Mapping[str, Any] = getattr(frame, "telemetry", None) or {}
    links: Sequence[Any] = getattr(frame, "links", None) or []
    rx_pos = tuple(getattr(frame, "rx_pos", (3.0, 0.35)))

    fc_raw = tel.get("frame_counts") or {}
    fc = {str(k).upper(): int(v) for k, v in fc_raw.items()}

    beacons = fc.get("BEACON", 0)
    cts = fc.get("CTS", 0)
    ack = fc.get("ACK", 0)
    block_ack = fc.get("BLOCK_ACK", 0) + fc.get("BA", 0)
    probes = fc.get("PROBE_REQ", 0) + fc.get("PROBE_RESP", 0) + fc.get("PROBE", 0)
    data_frames = (
        fc.get("DATA_PREAMBLE", 0)
        + fc.get("DATA", 0)
        + fc.get("QOS_NULL", 0)
        + fc.get("QOS_DATA", 0)
    )

    bssid_set = {
        canonicalize_mac(str(getattr(lnk, "tx_bssid", "")))
        for lnk in links
        if is_unicast_station_mac(getattr(lnk, "tx_bssid", None))
    }
    raw_bssid_cnt = tel.get("bssid_count")
    bssid_count = int(raw_bssid_cnt) if raw_bssid_cnt is not None else len(bssid_set)
    chord_count = len(links)
    peer_chord_count = sum(
        1
        for lnk in links
        if math.hypot(
            float(lnk.rx_pos[0]) - float(rx_pos[0]),
            float(lnk.rx_pos[1]) - float(rx_pos[1]),
        )
        > 0.15
    )

    if "fcs_drop_pct" in tel and tel["fcs_drop_pct"] is not None:
        fcs_drop_pct = float(tel["fcs_drop_pct"])
    elif links:
        fcs_drop_pct = 100.0 * float(
            np.mean([float(getattr(lnk, "fcs_error_rate", 0.0)) for lnk in links])
        )
    else:
        fcs_drop_pct = 0.0

    seq_gaps_raw = tel.get(
        "seq_gaps", tel.get("total_seq_gaps", tel.get("seq_gap_count"))
    )
    seq_gap_rate = (
        float(np.mean([float(getattr(lnk, "seq_gap_rate", 0.0)) for lnk in links]))
        if links
        else 0.0
    )
    if seq_gaps_raw is not None:
        seq_gaps = int(seq_gaps_raw)
    else:
        total_pkts = sum(int(getattr(lnk, "packet_count", 0)) for lnk in links)
        seq_gaps = round(seq_gap_rate * total_pkts)

    tsf_raw = tel.get("mean_tsf_ppm")
    mean_tsf_ppm = (
        float(tsf_raw) if tsf_raw is not None and np.isfinite(float(tsf_raw)) else None
    )

    adc_clip_max = float(tel.get("adc_clip_max", 0.0) or 0.0)
    if adc_clip_max >= 0.02:
        adc_status = f"CLIP WARN ({adc_clip_max * 100.0:.1f}%)"
    else:
        adc_status = f"OK ({adc_clip_max * 100.0:.2f}%)"

    backend_label = str(tel.get("backend", default_backend))

    # Extract validated respiratory rate:
    #   1) Stationary tracked human breathing BPM (highest confidence)
    #   2) Validated per-link respiratory detection when not exclusively tracking walkers
    #   3) Prominent (>= 6 dB interior local maximum in [0.18, 0.45] Hz) raw PSD peak
    #      only for standalone frames without LinkDiagnostic entries (never on flat PSD).
    peak_resp_hz: float | None = None
    peak_resp_bpm: float | None = None

    tracks: Sequence[Any] = getattr(frame, "tracks", None) or []
    best_trk_conf = -1.0
    for trk in tracks:
        bpm = getattr(trk, "breathing_bpm", None)
        if bpm is not None and np.isfinite(float(bpm)) and float(bpm) > 0.0:
            conf = float(getattr(trk, "confidence", 0.0) or 0.0)
            if conf > best_trk_conf:
                best_trk_conf = conf
                peak_resp_bpm = float(bpm)
                peak_resp_hz = peak_resp_bpm / 60.0

    all_tracks_walking = bool(tracks) and all(
        str(getattr(trk, "activity", "")).lower() == "walking"
        and getattr(trk, "breathing_bpm", None) is None
        for trk in tracks
    )

    if peak_resp_bpm is None and not all_tracks_walking:
        if links:
            best_snr = -1.0
            for lnk in links:
                lbpm = getattr(lnk, "breathing_bpm", None)
                if lbpm is not None and np.isfinite(float(lbpm)) and float(lbpm) > 0.0:
                    lsnr = float(getattr(lnk, "breathing_snr_db", 0.0) or 0.0)
                    if lsnr > best_snr:
                        best_snr = lsnr
                        peak_resp_bpm = float(lbpm)
                        lhz = getattr(lnk, "breathing_hz", None)
                        peak_resp_hz = (
                            float(lhz)
                            if lhz is not None
                            and np.isfinite(float(lhz))
                            and float(lhz) > 0.0
                            else peak_resp_bpm / 60.0
                        )
        else:
            freqs = getattr(frame, "psd_freqs_hz", None)
            psd_db = getattr(frame, "psd_power_db", None)
            if freqs is not None and psd_db is not None:
                f_arr = np.asarray(freqs, dtype=np.float64)
                p_arr = np.asarray(psd_db, dtype=np.float64)
                if f_arr.size == p_arr.size and f_arr.size >= 3:
                    mask = (f_arr >= 0.18) & (f_arr <= 0.45) & np.isfinite(p_arr)
                    if np.count_nonzero(mask) >= 3:
                        sub_f = f_arr[mask]
                        sub_p = p_arr[mask]
                        cand_local = [
                            k
                            for k in range(1, sub_p.size - 1)
                            if sub_p[k] >= sub_p[k - 1]
                            and sub_p[k] >= sub_p[k + 1]
                            and (sub_p[k] > sub_p[k - 1] or sub_p[k] > sub_p[k + 1])
                        ]
                        if cand_local:
                            idx = max(cand_local, key=lambda k: float(sub_p[k]))
                            pk_db = float(sub_p[idx])
                            f_pk = float(sub_f[idx])
                            edge_ceil = max(float(sub_p[0]), float(sub_p[-1]))
                            bg_mask = (
                                np.isfinite(p_arr)
                                & (f_arr >= 0.10)
                                & (f_arr <= 1.20)
                                & (np.abs(f_arr - f_pk) >= 0.06)
                            )
                            bg_vals = p_arr[bg_mask] if np.any(bg_mask) else sub_p
                            prom_med_db = pk_db - float(np.median(bg_vals))
                            prom_p85_db = pk_db - float(np.percentile(bg_vals, 85))
                            if (
                                pk_db >= edge_ceil + 2.5
                                and prom_med_db >= 6.0
                                and prom_p85_db >= 3.5
                            ):
                                peak_resp_hz = f_pk
                                peak_resp_bpm = peak_resp_hz * 60.0

    return _HudMetrics(
        packet_rate_hz=float(tel.get("packet_rate_hz", 0.0) or 0.0),
        bssid_count=bssid_count,
        chord_count=chord_count,
        peer_chord_count=peer_chord_count,
        beacons=beacons,
        cts=cts,
        ack=ack,
        block_ack=block_ack,
        probes=probes,
        data_frames=data_frames,
        fcs_drop_pct=fcs_drop_pct,
        seq_gaps=seq_gaps,
        seq_gap_rate=seq_gap_rate,
        mean_tsf_ppm=mean_tsf_ppm,
        adc_clip_max=adc_clip_max,
        adc_status=adc_status,
        backend_label=backend_label,
        peak_resp_hz=peak_resp_hz,
        peak_resp_bpm=peak_resp_bpm,
    )


def format_track_badge(track: Any) -> str:
    """Format a compact target callout including ID, activity, and '17 BPM' breathing badge."""
    tid = getattr(track, "track_id", 0)
    act = str(getattr(track, "activity", "stationary"))
    vx = float(getattr(track, "vx", 0.0))
    vy = float(getattr(track, "vy", 0.0))
    speed = math.hypot(vx, vy)
    bpm = getattr(track, "breathing_bpm", None)
    parts = [f"T{tid} [{act}]"]
    if bpm is not None and np.isfinite(float(bpm)) and float(bpm) > 0.0:
        parts.append(f"{round(float(bpm)):.0f} BPM")
    elif speed >= 0.08:
        parts.append(f"{speed:.2f} m/s")
    return "\n".join(parts)


class TrackHistoryBuffer:
    """Bounded O(1)-memory trajectory history per track_id with stale-track pruning."""

    def __init__(
        self, maxlen: int = 36, stale_timeout_s: float = 10.0, max_tracks: int = 24
    ) -> None:
        self.maxlen = maxlen
        self.stale_timeout_s = stale_timeout_s
        self.max_tracks = max_tracks
        self._trails: dict[int, deque[tuple[float, float]]] = {}
        self._last_seen: dict[int, float] = {}

    def update(
        self, tracks: Sequence[Any], timestamp: float
    ) -> dict[int, deque[tuple[float, float]]]:
        now = float(timestamp)
        for trk in tracks:
            tid = int(getattr(trk, "track_id", 0))
            x = float(getattr(trk, "x", 0.0))
            y = float(getattr(trk, "y", 0.0))
            buf = self._trails.get(tid)
            if buf is None:
                buf = deque(maxlen=self.maxlen)
                self._trails[tid] = buf
            buf.append((x, y))
            self._last_seen[tid] = now

        stale_ids = [
            tid
            for tid, seen_t in self._last_seen.items()
            if (now - seen_t) > self.stale_timeout_s
        ]
        for tid in stale_ids:
            self._trails.pop(tid, None)
            self._last_seen.pop(tid, None)

        if len(self._trails) > self.max_tracks:
            oldest = sorted(self._last_seen.items(), key=lambda kv: kv[1])[
                : len(self._trails) - self.max_tracks
            ]
            for tid, _ in oldest:
                self._trails.pop(tid, None)
                self._last_seen.pop(tid, None)

        return self._trails


def _normalize_density_map(
    frame: Any,
    room_width: float,
    room_height: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return non-negative (Ny, Nx) density map and (x0, x1, y0, y1) meter extent."""
    dm = np.asarray(getattr(frame, "density_map", np.zeros((34, 40))), dtype=np.float64)
    dm = np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)
    dm = np.maximum(dm, 0.0)
    gx = getattr(frame, "grid_x", None)
    gy = getattr(frame, "grid_y", None)
    if gx is not None and gy is not None and len(gx) >= 2 and len(gy) >= 2:
        nx, ny = len(gx), len(gy)
        if dm.shape == (nx, ny) and nx != ny:
            dm = dm.T
        dx = float(gx[1] - gx[0])
        dy = float(gy[1] - gy[0])
        extent = (
            max(0.0, float(gx[0]) - 0.5 * dx),
            min(room_width, float(gx[-1]) + 0.5 * dx)
            if float(gx[-1]) <= room_width
            else float(gx[-1]),
            max(0.0, float(gy[0]) - 0.5 * dy),
            min(room_height, float(gy[-1]) + 0.5 * dy)
            if float(gy[-1]) <= room_height
            else float(gy[-1]),
        )
        extent = (0.0, max(room_width, extent[1]), 0.0, max(room_height, extent[3]))
    else:
        extent = (0.0, room_width, 0.0, room_height)
    return dm, extent


def _build_chord_geometry(
    links: Sequence[Any],
    anchor_positions: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    list[tuple[str, float, float]],
]:
    """Partition links into quiet/mid/hot chord segments, FFZ ellipses, and unique TX nodes."""
    quiet_segs: list[tuple[np.ndarray, np.ndarray]] = []
    mid_segs: list[tuple[np.ndarray, np.ndarray]] = []
    hot_segs: list[tuple[np.ndarray, np.ndarray]] = []
    ffz_segs: list[tuple[np.ndarray, np.ndarray]] = []

    tx_nodes: dict[str, tuple[float, float]] = {}
    if anchor_positions:
        for bssid, pos in anchor_positions.items():
            tx_nodes[bssid] = (float(pos[0]), float(pos[1]))

    ranked_hot: list[tuple[float, Any]] = []
    for lnk in links:
        tx = tuple(getattr(lnk, "tx_pos", (0.0, 0.0)))
        rx = tuple(getattr(lnk, "rx_pos", (0.0, 0.0)))
        bssid = str(getattr(lnk, "tx_bssid", ""))
        if bssid:
            tx_nodes[bssid] = (float(tx[0]), float(tx[1]))
        seg = (
            np.array([float(tx[0]), float(rx[0])], dtype=np.float64),
            np.array([float(tx[1]), float(rx[1])], dtype=np.float64),
        )
        score = float(getattr(lnk, "perturbation_score", 0.0) or 0.0)
        if score >= 0.45:
            hot_segs.append(seg)
            ranked_hot.append((score, lnk))
        elif score >= 0.18:
            mid_segs.append(seg)
            ranked_hot.append((score, lnk))
        else:
            quiet_segs.append(seg)

    # Render First Fresnel Zone (lambda/2) ellipses on up to 6 most-perturbed chords
    ranked_hot.sort(key=lambda item: item[0], reverse=True)
    for _, lnk in ranked_hot[:6]:
        ex, ey = compute_fresnel_ellipse(
            tuple(lnk.tx_pos),
            tuple(lnk.rx_pos),
            float(getattr(lnk, "freq_hz", 2437e6) or 2437e6),
        )
        if ex.size > 0:
            ffz_segs.append((ex, ey))

    node_list = [(bssid, pos[0], pos[1]) for bssid, pos in sorted(tx_nodes.items())]
    return (
        _concat_segments_with_nan(quiet_segs),
        _concat_segments_with_nan(mid_segs),
        _concat_segments_with_nan(hot_segs),
        _concat_segments_with_nan(ffz_segs),
        node_list,
    )


class MatplotlibRadarDashboard:
    """High-speed zero-allocation 4-panel Matplotlib radar dashboard (X11 + headless Agg)."""

    MAX_ANCHOR_LABELS = 14
    MAX_TRACK_LABELS = 10
    MAX_GT_LABELS = 8

    def __init__(
        self,
        room_width: float = 6.0,
        room_height: float = 5.0,
        anchor_positions: Mapping[str, tuple[float, float]] | None = None,
        anchor_labels: Mapping[str, str] | None = None,
        headless: bool = False,
        show_ground_truth: bool = True,
        title: str = "HackRF Passive Wi-Fi Tomographic Radar (0 dBm RX-Only)",
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.room_width = float(room_width)
        self.room_height = float(room_height)
        self.anchor_positions = dict(anchor_positions or {})
        self.anchor_labels = dict(anchor_labels or {})
        self.show_ground_truth = bool(show_ground_truth)
        self.title = title

        mpl_env = os.environ.get("MPLBACKEND", "").lower()
        can_interact = (not headless) and (mpl_env != "agg") and has_gui_display()
        self._interactive = False
        self._plt: Any = None

        if can_interact:
            try:
                import matplotlib.pyplot as plt

                backend_name = str(plt.get_backend()).strip().lower()
                if "agg" not in backend_name or backend_name in {
                    "tkagg",
                    "qtagg",
                    "qt5agg",
                    "qt6agg",
                    "gtk3agg",
                    "gtk4agg",
                    "wxagg",
                    "macosx",
                    "webagg",
                }:
                    self._plt = plt
                    self.fig = plt.figure(
                        figsize=(11.5, 7.0), dpi=80, facecolor="#0b1020"
                    )
                    self.canvas = self.fig.canvas
                    if str(plt.get_backend()).strip().lower() != "agg":
                        plt.show(block=False)
                        self._interactive = True
                    else:
                        plt.close(self.fig)
                        self._plt = None
                        self.fig = Figure(
                            figsize=(11.5, 7.0), dpi=80, facecolor="#0b1020"
                        )
                        self.canvas = FigureCanvasAgg(self.fig)
                else:
                    self.fig = Figure(figsize=(11.5, 7.0), dpi=80, facecolor="#0b1020")
                    self.canvas = FigureCanvasAgg(self.fig)
            except (ImportError, *_GUI_BACKEND_ERRORS):
                self._plt = None
                self.fig = Figure(figsize=(11.5, 7.0), dpi=80, facecolor="#0b1020")
                self.canvas = FigureCanvasAgg(self.fig)
        else:
            self.fig = Figure(figsize=(11.5, 7.0), dpi=80, facecolor="#0b1020")
            self.canvas = FigureCanvasAgg(self.fig)

        self._history = TrackHistoryBuffer()
        self._last_render_ms: float = 0.0
        self._frame_count: int = 0
        self._closed: bool = False

        gs = self.fig.add_gridspec(
            2,
            2,
            width_ratios=[1.22, 1.0],
            height_ratios=[1.12, 0.88],
            left=0.055,
            right=0.965,
            bottom=0.075,
            top=0.915,
            wspace=0.22,
            hspace=0.28,
        )
        self.fig.suptitle(
            f"{self.title}  |  0 dBm RX-ONLY (RF AMP OFF)",
            color="#e2e8f0",
            fontsize=12,
            fontweight="bold",
            y=0.968,
        )

        # Panel 1: 2D Tomographic Presence Heatmap + Chords + RX + BSSIDs + Tracks
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self._style_axis(
            self.ax_map, "2D Radio Tomographic Presence & First Fresnel Links"
        )
        self.ax_map.set_xlim(-0.15, self.room_width + 0.15)
        self.ax_map.set_ylim(-0.15, self.room_height + 0.15)
        self.ax_map.set_aspect("equal", adjustable="box")
        self.ax_map.set_xlabel("Room X (m)", color="#cbd5e1", fontsize=9)
        self.ax_map.set_ylabel("Room Y (m)", color="#cbd5e1", fontsize=9)

        # Room perimeter outline
        self.ax_map.plot(
            [0.0, self.room_width, self.room_width, 0.0, 0.0],
            [0.0, 0.0, self.room_height, self.room_height, 0.0],
            color="#475569",
            linewidth=1.2,
            linestyle="-",
            zorder=2,
        )

        init_dm = np.zeros((34, 40), dtype=np.float64)
        self._im = self.ax_map.imshow(
            init_dm,
            origin="lower",
            extent=(0.0, self.room_width, 0.0, self.room_height),
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            alpha=0.90,
            zorder=1,
        )

        (self._chord_quiet,) = self.ax_map.plot(
            [], [], color="#38bdf8", alpha=0.24, linewidth=0.9, zorder=3
        )
        (self._chord_mid,) = self.ax_map.plot(
            [], [], color="#fbbf24", alpha=0.62, linewidth=1.5, zorder=4
        )
        (self._chord_hot,) = self.ax_map.plot(
            [], [], color="#fb7185", alpha=0.88, linewidth=2.2, zorder=5
        )
        (self._ffz_ellipses,) = self.ax_map.plot(
            [], [], color="#22d3ee", alpha=0.45, linewidth=0.9, linestyle="--", zorder=4
        )

        (self._tx_markers,) = self.ax_map.plot(
            [],
            [],
            linestyle="None",
            marker="s",
            markersize=6.5,
            markerfacecolor="#38bdf8",
            markeredgecolor="#0f172a",
            markeredgewidth=0.9,
            zorder=6,
            label="Ambient Wi-Fi BSSID",
        )
        (self._rx_marker,) = self.ax_map.plot(
            [3.0],
            [0.35],
            linestyle="None",
            marker="^",
            markersize=10.0,
            markerfacecolor="#22c55e",
            markeredgecolor="#ffffff",
            markeredgewidth=1.2,
            zorder=8,
            label="HackRF One RX (0 dBm)",
        )
        self._rx_text = self.ax_map.text(
            3.0,
            0.12,
            "HackRF RX (0 dBm)",
            color="#86efac",
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=9,
        )

        (self._gt_markers,) = self.ax_map.plot(
            [],
            [],
            linestyle="None",
            marker="P",
            markersize=9.0,
            markerfacecolor="none",
            markeredgecolor="#4ade80",
            markeredgewidth=1.6,
            zorder=8,
            label="Ground Truth",
        )
        (self._trail_lines,) = self.ax_map.plot(
            [], [], color="#67e8f9", alpha=0.75, linewidth=1.7, zorder=7
        )
        (self._vel_lines,) = self.ax_map.plot(
            [], [], color="#fde047", alpha=0.95, linewidth=2.0, zorder=9
        )
        (self._track_markers,) = self.ax_map.plot(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=9.5,
            markerfacecolor="#22d3ee",
            markeredgecolor="#ffffff",
            markeredgewidth=1.4,
            zorder=9,
            label="Tracked Human",
        )

        self._anchor_texts = [
            self.ax_map.text(
                0.0,
                0.0,
                "",
                color="#bae6fd",
                fontsize=6.8,
                ha="center",
                va="bottom",
                visible=False,
                zorder=7,
            )
            for _ in range(self.MAX_ANCHOR_LABELS)
        ]
        self._track_texts = [
            self.ax_map.text(
                0.0,
                0.0,
                "",
                color="#fef08a",
                fontsize=8.0,
                fontweight="bold",
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "#0f172a",
                    "edgecolor": "#22d3ee",
                    "alpha": 0.82,
                },
                visible=False,
                zorder=10,
            )
            for _ in range(self.MAX_TRACK_LABELS)
        ]
        self._gt_texts = [
            self.ax_map.text(
                0.0,
                0.0,
                "",
                color="#86efac",
                fontsize=7.0,
                ha="right",
                va="top",
                visible=False,
                zorder=9,
            )
            for _ in range(self.MAX_GT_LABELS)
        ]
        leg = self.ax_map.legend(
            loc="lower right",
            fontsize=7.2,
            facecolor="#0f172a",
            edgecolor="#334155",
            labelcolor="#e2e8f0",
        )
        leg.set_zorder(11)

        # Panel 2: Live 52-Subcarrier L-LTF CSI |H[k]| & Sanitized Phase
        self.ax_csi = self.fig.add_subplot(gs[0, 1])
        self._style_axis(
            self.ax_csi, "52-Subcarrier L-LTF CSI |H[k]| & Sanitized Phase"
        )
        self.ax_csi.set_xlim(-27.5, 27.5)
        self.ax_csi.set_ylim(0.0, 2.0)
        self.ax_csi.set_xlabel(
            "OFDM Subcarrier Index k (Pilots: ±7, ±21)", color="#cbd5e1", fontsize=8
        )
        self.ax_csi.set_ylabel("|H[k]| Amplitude", color="#38bdf8", fontsize=8)
        for pk in PILOT_SUBCARRIERS:
            self.ax_csi.axvline(pk, color="#475569", linestyle=":", linewidth=0.8)
        self.ax_csi_phase = self.ax_csi.twinx()
        self.ax_csi_phase.set_ylim(-math.pi - 0.25, math.pi + 0.25)
        self.ax_csi_phase.set_ylabel(
            "Sanitized Phase (rad)", color="#f472b6", fontsize=8
        )
        self.ax_csi_phase.tick_params(colors="#f472b6", labelsize=7.5)
        for spine in self.ax_csi_phase.spines.values():
            spine.set_color("#334155")

        (self._csi_amp_line,) = self.ax_csi.plot(
            DEFAULT_SUBCARRIER_INDICES,
            np.zeros(52, dtype=np.float64),
            color="#38bdf8",
            linewidth=1.6,
            marker=".",
            markersize=3.5,
            label="|H[k]|",
        )
        (self._csi_phase_line,) = self.ax_csi_phase.plot(
            DEFAULT_SUBCARRIER_INDICES,
            np.zeros(52, dtype=np.float64),
            color="#f472b6",
            linewidth=1.2,
            alpha=0.88,
            label="∠H_san[k]",
        )

        # Panel 3: Micro-Doppler & Vital-Sign 0.1-25 Hz PSD + Passive Wi-Fi HUD Split
        sub_gs = gs[1, 1].subgridspec(2, 1, height_ratios=[1.05, 0.95], hspace=0.34)
        self.ax_psd = self.fig.add_subplot(sub_gs[0, 0])
        self._style_axis(self.ax_psd, "Micro-Doppler & Vital-Sign PSD (0.1 - 25 Hz)")
        self.ax_psd.set_xlim(0.1, 25.0)
        self.ax_psd.set_ylim(-45.0, 5.0)
        self.ax_psd.set_xlabel(
            "Frequency (Hz)  [Green Band = 0.15-0.50 Hz Breathing]",
            color="#cbd5e1",
            fontsize=7.8,
        )
        self.ax_psd.set_ylabel("PSD (dB)", color="#cbd5e1", fontsize=8)
        self.ax_psd.axvspan(0.15, 0.50, color="#22c55e", alpha=0.22)
        (self._psd_line,) = self.ax_psd.plot(
            np.linspace(0.1, 25.0, 64),
            np.full(64, -40.0),
            color="#a78bfa",
            linewidth=1.5,
        )
        self._psd_peak_line = self.ax_psd.axvline(
            0.28, color="#4ade80", linestyle="--", linewidth=1.1, visible=False
        )
        self._psd_peak_text = self.ax_psd.text(
            0.65,
            0.84,
            "",
            transform=self.ax_psd.transAxes,
            color="#86efac",
            fontsize=8.0,
            fontweight="bold",
        )

        # Panel 4: Non-Authenticated Passive Wi-Fi & Interference HUD
        self.ax_hud = self.fig.add_subplot(sub_gs[1, 0])
        self.ax_hud.set_facecolor("#0f172a")
        self.ax_hud.set_xticks([])
        self.ax_hud.set_yticks([])
        for spine in self.ax_hud.spines.values():
            spine.set_color("#334155")
        self._hud_text = self.ax_hud.text(
            0.025,
            0.93,
            "PASSIVE 0 dBm RX-ONLY HUD — Initializing...",
            transform=self.ax_hud.transAxes,
            color="#e2e8f0",
            fontsize=7.6,
            family="monospace",
            va="top",
            ha="left",
        )

        # Configure fast Agg/GUI blitting so per-frame updates run in < 25 ms (>= 40 FPS)
        self._map_artists: list[Any] = [
            self._im,
            self._chord_quiet,
            self._chord_mid,
            self._chord_hot,
            self._ffz_ellipses,
            self._tx_markers,
            self._gt_markers,
            self._trail_lines,
            self._vel_lines,
            self._track_markers,
            self._rx_marker,
            self._rx_text,
            *self._anchor_texts,
            *self._gt_texts,
            *self._track_texts,
        ]
        self._csi_artists: list[Any] = [self._csi_amp_line]
        self._csi_phase_artists: list[Any] = [self._csi_phase_line]
        self._psd_artists: list[Any] = [
            self._psd_line,
            self._psd_peak_line,
            self._psd_peak_text,
        ]
        self._hud_artists: list[Any] = [self._hud_text]
        self._all_animated: list[Any] = (
            self._map_artists
            + self._csi_artists
            + self._csi_phase_artists
            + self._psd_artists
            + self._hud_artists
        )
        for art in self._all_animated:
            art.set_animated(True)

        self._bg: Any = None
        self._bg_dirty: bool = True
        self._rebuild_blit_background()

    def _rebuild_blit_background(self) -> None:
        self.canvas.draw()
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._bg_dirty = False

    @staticmethod
    def _style_axis(ax: Any, title: str) -> None:
        ax.set_facecolor("#0f172a")
        ax.set_title(title, color="#e2e8f0", fontsize=9.5, fontweight="bold", pad=6)
        ax.tick_params(colors="#94a3b8", labelsize=7.8)
        ax.grid(True, color="#1e293b", linestyle="--", linewidth=0.6, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    @property
    def last_render_ms(self) -> float:
        return self._last_render_ms

    def update(
        self,
        frame: Any,
        ground_truth: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        """Update all 4 dashboard panels in-place and render; returns elapsed time in ms."""
        if self._closed:
            return 0.0
        t0 = time.perf_counter()
        self._frame_count += 1

        dm, extent = _normalize_density_map(frame, self.room_width, self.room_height)
        vmax = float(np.max(dm)) if dm.size > 0 else 0.0
        self._im.set_data(dm)
        self._im.set_extent(extent)
        self._im.set_clim(0.0, max(vmax, 1e-3))

        rx_x, rx_y = float(frame.rx_pos[0]), float(frame.rx_pos[1])
        self._rx_marker.set_data([rx_x], [rx_y])
        rx_lbl_pos = (rx_x, max(-0.12, rx_y - 0.16))
        if self._rx_text.get_position() != rx_lbl_pos:
            self._rx_text.set_position(rx_lbl_pos)

        links: Sequence[Any] = getattr(frame, "links", None) or []
        (q_seg, m_seg, h_seg, ffz_seg, tx_nodes) = _build_chord_geometry(
            links, self.anchor_positions
        )
        self._chord_quiet.set_data(*q_seg)
        self._chord_mid.set_data(*m_seg)
        self._chord_hot.set_data(*h_seg)
        self._ffz_ellipses.set_data(*ffz_seg)

        if tx_nodes:
            self._tx_markers.set_data(
                [n[1] for n in tx_nodes], [n[2] for n in tx_nodes]
            )
        else:
            self._tx_markers.set_data([], [])

        for idx, txt in enumerate(self._anchor_texts):
            if idx < len(tx_nodes):
                bssid, nx, ny = tx_nodes[idx]
                lbl = self.anchor_labels.get(bssid, bssid[-5:])
                pos = (nx, ny + 0.08)
                if txt.get_position() != pos:
                    txt.set_position(pos)
                if txt.get_text() != lbl:
                    txt.set_text(lbl)
                if not txt.get_visible():
                    txt.set_visible(True)
            elif txt.get_visible():
                txt.set_visible(False)

        # Tracked humans, trails, velocity vectors, and "17 BPM" badges
        tracks: Sequence[Any] = getattr(frame, "tracks", None) or []
        trails = self._history.update(tracks, float(getattr(frame, "timestamp", 0.0)))

        trail_segs: list[tuple[np.ndarray, np.ndarray]] = []
        for trk in tracks:
            tid = int(getattr(trk, "track_id", 0))
            pts = trails.get(tid)
            if pts and len(pts) >= 2:
                arr = np.asarray(pts, dtype=np.float64)
                trail_segs.append((arr[:, 0], arr[:, 1]))
        self._trail_lines.set_data(*_concat_segments_with_nan(trail_segs))

        vel_segs: list[tuple[np.ndarray, np.ndarray]] = []
        tx_list: list[float] = []
        ty_list: list[float] = []
        for trk in tracks:
            x = float(trk.x)
            y = float(trk.y)
            vx = float(getattr(trk, "vx", 0.0))
            vy = float(getattr(trk, "vy", 0.0))
            tx_list.append(x)
            ty_list.append(y)
            if math.hypot(vx, vy) >= 0.08:
                vel_segs.append(
                    (
                        np.array([x, x + 0.55 * vx], dtype=np.float64),
                        np.array([y, y + 0.55 * vy], dtype=np.float64),
                    )
                )
        self._track_markers.set_data(tx_list, ty_list)
        self._vel_lines.set_data(*_concat_segments_with_nan(vel_segs))

        for idx, txt in enumerate(self._track_texts):
            if idx < len(tracks):
                trk = tracks[idx]
                bx = min(self.room_width - 0.55, float(trk.x) + 0.12)
                by = min(self.room_height - 0.35, float(trk.y) + 0.10)
                badge = format_track_badge(trk)
                txt.set_position((bx, by))
                if txt.get_text() != badge:
                    txt.set_text(badge)
                if not txt.get_visible():
                    txt.set_visible(True)
            elif txt.get_visible():
                txt.set_visible(False)

        # Ground truth overlay in simulation mode
        if self.show_ground_truth and ground_truth:
            gx_pts = [float(g["x"]) for g in ground_truth]
            gy_pts = [float(g["y"]) for g in ground_truth]
            self._gt_markers.set_data(gx_pts, gy_pts)
            for idx, txt in enumerate(self._gt_texts):
                if idx < len(ground_truth):
                    g = ground_truth[idx]
                    gid = g.get("id", idx + 1)
                    gbpm = g.get("breathing_bpm")
                    glbl = f"GT#{gid}" + (
                        f" ({round(float(gbpm)):.0f} BPM)" if gbpm else ""
                    )
                    txt.set_position((float(g["x"]) - 0.10, float(g["y"]) - 0.10))
                    if txt.get_text() != glbl:
                        txt.set_text(glbl)
                    if not txt.get_visible():
                        txt.set_visible(True)
                elif txt.get_visible():
                    txt.set_visible(False)
        else:
            self._gt_markers.set_data([], [])
            for txt in self._gt_texts:
                if txt.get_visible():
                    txt.set_visible(False)

        # Panel 2: 52-subcarrier L-LTF CSI |H[k]| & sanitized phase
        csi_amp = getattr(frame, "latest_csi_amp", None)
        csi_phase = getattr(frame, "latest_csi_phase", None)
        if csi_amp is not None and len(csi_amp) == 52:
            amp_arr = np.nan_to_num(np.asarray(csi_amp, dtype=np.float64), nan=0.0)
            self._csi_amp_line.set_ydata(amp_arr)
            amp_top = max(float(np.max(amp_arr)) * 1.25, 0.5)
            _, cur_hi = self.ax_csi.get_ylim()
            if amp_top > cur_hi or amp_top < 0.30 * cur_hi:
                self.ax_csi.set_ylim(0.0, amp_top * 1.15)
                self._bg_dirty = True
        if csi_phase is not None and len(csi_phase) == 52:
            ph_arr = np.nan_to_num(np.asarray(csi_phase, dtype=np.float64), nan=0.0)
            ph_wrapped = (ph_arr + math.pi) % (2.0 * math.pi) - math.pi
            self._csi_phase_line.set_ydata(ph_wrapped)

        # Panel 3: Micro-Doppler & Vital-Sign PSD (0.1 - 25 Hz)
        psd_f = getattr(frame, "psd_freqs_hz", None)
        psd_p = getattr(frame, "psd_power_db", None)
        if (
            psd_f is not None
            and psd_p is not None
            and len(psd_f) == len(psd_p)
            and len(psd_f) >= 2
        ):
            f_arr = np.asarray(psd_f, dtype=np.float64)
            p_arr = np.nan_to_num(np.asarray(psd_p, dtype=np.float64), nan=-60.0)
            valid = (f_arr >= 0.08) & (f_arr <= 25.0)
            if np.any(valid):
                fv = f_arr[valid]
                pv = p_arr[valid]
                self._psd_line.set_data(fv, pv)
                p_min = float(np.min(pv))
                p_max = float(np.max(pv))
                if p_max - p_min < 10.0:
                    p_min = p_max - 30.0
                cur_plo, cur_phi = self.ax_psd.get_ylim()
                if (
                    (p_min - 2.0) < cur_plo
                    or (p_max + 2.0) > cur_phi
                    or (cur_phi - cur_plo) > 2.5 * (p_max - p_min + 12.0)
                ):
                    self.ax_psd.set_ylim(p_min - 6.0, p_max + 6.0)
                    self._bg_dirty = True

        hud = _extract_hud_metrics(frame)
        vital_str = (
            f"{round(hud.peak_resp_bpm):.0f} BPM ({hud.peak_resp_hz:.2f} Hz)"
            if hud.peak_resp_bpm is not None and hud.peak_resp_hz is not None
            else "none"
        )
        if hud.peak_resp_hz is not None and hud.peak_resp_bpm is not None:
            self._psd_peak_line.set_xdata([hud.peak_resp_hz, hud.peak_resp_hz])
            self._psd_peak_line.set_visible(True)
            pk_txt = f"Vital Peak: {vital_str}"
            if self._psd_peak_text.get_text() != pk_txt:
                self._psd_peak_text.set_text(pk_txt)
        else:
            self._psd_peak_line.set_visible(False)
            if self._psd_peak_text.get_text() != "Vital Peak: none":
                self._psd_peak_text.set_text("Vital Peak: none")

        # Panel 4: Passive Wi-Fi & Interference HUD text
        tsf_str = (
            f"{hud.mean_tsf_ppm:+.2f} ppm"
            if hud.mean_tsf_ppm is not None
            else "calibrating"
        )
        hud_lines = [
            f"[0 dBm RX-ONLY | AMP: OFF | CLEARTEXT 802.11]  Backend: {hud.backend_label}",
            (
                f"Rate: {hud.packet_rate_hz:6.1f} pkt/s | BSSIDs: {hud.bssid_count:2d} | "
                f"Chords: {hud.chord_count:2d} (SIFS Peer: {hud.peer_chord_count:2d})"
            ),
            (
                f"Frames: Beacon={hud.beacons} CTS={hud.cts} ACK={hud.ack} "
                f"BA={hud.block_ack} Probe={hud.probes} Data={hud.data_frames}"
            ),
            (
                f"FCS CRC Drop: {hud.fcs_drop_pct:5.1f}% | Seq Gaps: {hud.seq_gaps:4d} "
                f"({hud.seq_gap_rate * 100.0:4.1f}%) | TSF Drift: {tsf_str}"
            ),
            (
                f"ADC Clip: {hud.adc_status} | Targets: {len(tracks)} | "
                f"Vital Peak: {vital_str} | Render: {self._last_render_ms:4.1f} ms"
            ),
        ]
        self._hud_text.set_text("\n".join(hud_lines))

        try:
            if self._bg_dirty or self._bg is None:
                self._rebuild_blit_background()

            self.canvas.restore_region(self._bg)
            for art in self._map_artists:
                if art.get_visible():
                    self.ax_map.draw_artist(art)
            for art in self._csi_artists:
                if art.get_visible():
                    self.ax_csi.draw_artist(art)
            for art in self._csi_phase_artists:
                if art.get_visible():
                    self.ax_csi_phase.draw_artist(art)
            for art in self._psd_artists:
                if art.get_visible():
                    self.ax_psd.draw_artist(art)
            for art in self._hud_artists:
                if art.get_visible():
                    self.ax_hud.draw_artist(art)

            self.canvas.blit(self.fig.bbox)
            if self._interactive:
                self.canvas.flush_events()
        except _GUI_BACKEND_ERRORS:
            if self._interactive:
                self._closed = True
                return 0.0
            raise

        self._last_render_ms = (time.perf_counter() - t0) * 1000.0
        return self._last_render_ms

    def save_snapshot(self, path: str | Path, dpi: int = 120) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._im.set_interpolation("bilinear")
        for art in self._all_animated:
            art.set_animated(False)
        try:
            self.fig.savefig(out, dpi=dpi, facecolor=self.fig.get_facecolor())
        finally:
            self._im.set_interpolation("nearest")
            for art in self._all_animated:
                art.set_animated(True)
            self._bg_dirty = True
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._plt is not None:
            with contextlib.suppress(*_GUI_BACKEND_ERRORS):
                self._plt.close(self.fig)
        else:
            with contextlib.suppress(*_GUI_BACKEND_ERRORS):
                self.fig.clear()


class PyQtGraphRadarDashboard:
    """High-FPS PyQtGraph multi-human radar dashboard (supports X11 and QT_QPA_PLATFORM=offscreen)."""

    MAX_TRACK_LABELS = 10
    MAX_ANCHOR_LABELS = 14

    def __init__(
        self,
        room_width: float = 6.0,
        room_height: float = 5.0,
        anchor_positions: Mapping[str, tuple[float, float]] | None = None,
        anchor_labels: Mapping[str, str] | None = None,
        headless: bool = False,
        show_ground_truth: bool = True,
        title: str = "HackRF Passive Wi-Fi Tomographic Radar (0 dBm RX-Only)",
    ) -> None:
        if headless and not os.environ.get("QT_QPA_PLATFORM") and not has_gui_display():
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore

        self._pg = pg
        self._QtCore = QtCore
        self.room_width = float(room_width)
        self.room_height = float(room_height)
        self.anchor_positions = dict(anchor_positions or {})
        self.anchor_labels = dict(anchor_labels or {})
        self.show_ground_truth = bool(show_ground_truth)
        self._headless = bool(headless)
        self._history = TrackHistoryBuffer()
        self._last_render_ms = 0.0
        self._closed = False
        self._last_frame: Any = None
        self._last_gt: Sequence[Mapping[str, Any]] | None = None

        self.app = pg.mkQApp("HackRF Passive Wi-Fi Radar")
        self.win = pg.GraphicsLayoutWidget(show=not headless, title=title)
        self.win.resize(1280, 800)
        self.win.setBackground("#0b1020")

        # Title banner
        self._header = self.win.addLabel(
            f"<b>{title}</b> &nbsp;|&nbsp; <span style='color:#4ade80'>0 dBm RX-ONLY (RF AMP OFF)</span>",
            row=0,
            col=0,
            colspan=2,
            color="#e2e8f0",
            size="12pt",
        )

        # Panel 1: 2D Tomographic Heatmap + Chords + Targets
        self.p_map = self.win.addPlot(
            row=1,
            col=0,
            rowspan=2,
            title="2D Radio Tomographic Presence & First Fresnel Links",
        )
        self.p_map.setAspectLocked(True, ratio=1.0)
        self.p_map.setXRange(-0.15, self.room_width + 0.15, padding=0)
        self.p_map.setYRange(-0.15, self.room_height + 0.15, padding=0)
        self.p_map.setLabel("bottom", "Room X", units="m")
        self.p_map.setLabel("left", "Room Y", units="m")
        self.p_map.showGrid(x=True, y=True, alpha=0.25)

        self._img = pg.ImageItem(
            np.zeros((34, 40), dtype=np.float32), axisOrder="row-major"
        )
        lut = self._build_inferno_lut()
        self._img.setLookupTable(lut)
        self._img.setRect(QtCore.QRectF(0.0, 0.0, self.room_width, self.room_height))
        self.p_map.addItem(self._img)

        self._room_box = self.p_map.plot(
            [0.0, self.room_width, self.room_width, 0.0, 0.0],
            [0.0, 0.0, self.room_height, self.room_height, 0.0],
            pen=pg.mkPen("#64748b", width=1.4),
        )
        self._chord_quiet = self.p_map.plot(
            [], [], connect="finite", pen=pg.mkPen((56, 189, 248, 70), width=1.0)
        )
        self._chord_mid = self.p_map.plot(
            [], [], connect="finite", pen=pg.mkPen((251, 191, 36, 165), width=1.8)
        )
        self._chord_hot = self.p_map.plot(
            [], [], connect="finite", pen=pg.mkPen((251, 113, 133, 225), width=2.4)
        )
        self._ffz_item = self.p_map.plot(
            [],
            [],
            connect="finite",
            pen=pg.mkPen(
                (34, 211, 238, 120), width=1.0, style=QtCore.Qt.PenStyle.DashLine
            ),
        )

        self._tx_scatter = pg.ScatterPlotItem(
            size=9,
            brush=pg.mkBrush("#38bdf8"),
            pen=pg.mkPen("#0f172a", width=1.0),
            symbol="s",
        )
        self._rx_scatter = pg.ScatterPlotItem(
            size=14,
            brush=pg.mkBrush("#22c55e"),
            pen=pg.mkPen("#ffffff", width=1.5),
            symbol="t1",
        )
        self._gt_scatter = pg.ScatterPlotItem(
            size=12,
            brush=pg.mkBrush(None),
            pen=pg.mkPen("#4ade80", width=1.8),
            symbol="+",
        )
        self._trail_item = self.p_map.plot(
            [], [], connect="finite", pen=pg.mkPen((103, 232, 249, 195), width=2.0)
        )
        self._vel_item = self.p_map.plot(
            [], [], connect="finite", pen=pg.mkPen((253, 224, 71, 240), width=2.3)
        )
        self._trk_scatter = pg.ScatterPlotItem(
            size=13,
            brush=pg.mkBrush("#22d3ee"),
            pen=pg.mkPen("#ffffff", width=1.6),
            symbol="o",
        )
        self.p_map.addItem(self._tx_scatter)
        self.p_map.addItem(self._rx_scatter)
        self.p_map.addItem(self._gt_scatter)
        self.p_map.addItem(self._trk_scatter)

        self._rx_label = pg.TextItem(
            "HackRF RX (0 dBm)", color="#86efac", anchor=(0.5, 0.0)
        )
        self.p_map.addItem(self._rx_label)

        self._anchor_items: list[Any] = []
        for _ in range(self.MAX_ANCHOR_LABELS):
            ti = pg.TextItem("", color="#bae6fd", anchor=(0.5, 1.0))
            ti.setVisible(False)
            self.p_map.addItem(ti)
            self._anchor_items.append(ti)

        self._track_items: list[Any] = []
        for _ in range(self.MAX_TRACK_LABELS):
            ti = pg.TextItem(
                "",
                color="#fef08a",
                anchor=(0.0, 1.0),
                fill=pg.mkBrush(15, 23, 42, 210),
                border=pg.mkPen("#22d3ee", width=1.0),
            )
            ti.setVisible(False)
            self.p_map.addItem(ti)
            self._track_items.append(ti)

        # Panel 2: 52-Subcarrier L-LTF CSI |H[k]| & Sanitized Phase
        self.p_csi = self.win.addPlot(
            row=1, col=1, title="52-Subcarrier L-LTF CSI |H[k]| & Sanitized Phase (rad)"
        )
        self.p_csi.setXRange(-27.0, 27.0, padding=0)
        self.p_csi.setLabel("bottom", "Subcarrier k")
        self.p_csi.showGrid(x=True, y=True, alpha=0.25)
        for pk in PILOT_SUBCARRIERS:
            vline = pg.InfiniteLine(
                pos=pk,
                angle=90,
                pen=pg.mkPen("#475569", width=0.9, style=QtCore.Qt.PenStyle.DotLine),
            )
            self.p_csi.addItem(vline)
        self._csi_amp_curve = self.p_csi.plot(
            DEFAULT_SUBCARRIER_INDICES,
            np.zeros(52),
            pen=pg.mkPen("#38bdf8", width=2.0),
            name="|H[k]|",
        )
        self._csi_phase_curve = self.p_csi.plot(
            DEFAULT_SUBCARRIER_INDICES,
            np.zeros(52),
            pen=pg.mkPen("#f472b6", width=1.4),
            name="∠H_san[k]",
        )

        # Panel 3 & 4: Right-bottom sub-layout for PSD + Passive Wi-Fi HUD
        right_sub = self.win.addLayout(row=2, col=1)
        self.p_psd = right_sub.addPlot(
            row=0, col=0, title="Micro-Doppler & Vital-Sign PSD (0.1 - 25 Hz)"
        )
        self.p_psd.setXRange(0.1, 25.0, padding=0)
        self.p_psd.setLabel("bottom", "Frequency", units="Hz")
        self.p_psd.setLabel("left", "PSD", units="dB")
        self.p_psd.showGrid(x=True, y=True, alpha=0.25)
        resp_band = pg.LinearRegionItem(
            values=(0.15, 0.50),
            movable=False,
            brush=pg.mkBrush(34, 197, 94, 55),
            pen=pg.mkPen((34, 197, 94, 120), width=1.0),
        )
        self.p_psd.addItem(resp_band)
        self._psd_curve = self.p_psd.plot(
            np.linspace(0.1, 25.0, 64),
            np.full(64, -40.0),
            pen=pg.mkPen("#a78bfa", width=1.8),
        )
        self._psd_peak_line = pg.InfiniteLine(
            pos=0.28,
            angle=90,
            pen=pg.mkPen("#4ade80", width=1.3, style=QtCore.Qt.PenStyle.DashLine),
        )
        self._psd_peak_line.setVisible(False)
        self.p_psd.addItem(self._psd_peak_line)

        self._hud_label = right_sub.addLabel(
            "Initializing Passive 0 dBm RX HUD...",
            row=1,
            col=0,
            justify="left",
            color="#e2e8f0",
            size="9pt",
        )
        self.app.processEvents()

    @staticmethod
    def _build_inferno_lut() -> np.ndarray:
        """256-entry RGB lookup table matching dark navy -> purple -> orange -> white-yellow."""
        stops = np.array([0.0, 0.25, 0.55, 0.80, 1.0], dtype=np.float64)
        colors = np.array(
            [
                [11, 16, 32],
                [72, 21, 108],
                [187, 55, 84],
                [249, 142, 9],
                [252, 255, 164],
            ],
            dtype=np.float64,
        )
        x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
        lut = np.column_stack([np.interp(x, stops, colors[:, c]) for c in range(3)])
        return np.clip(lut, 0, 255).astype(np.ubyte)

    @property
    def last_render_ms(self) -> float:
        return self._last_render_ms

    def update(
        self,
        frame: Any,
        ground_truth: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        if self._closed:
            return 0.0
        if not self._headless and not self.win.isVisible():
            self._closed = True
            return 0.0
        t0 = time.perf_counter()
        self._last_frame = frame
        self._last_gt = ground_truth

        dm, extent = _normalize_density_map(frame, self.room_width, self.room_height)
        vmax = max(float(np.max(dm)) if dm.size > 0 else 0.0, 1e-3)
        self._img.setImage(dm.astype(np.float32), levels=(0.0, vmax), autoLevels=False)
        self._img.setRect(
            self._QtCore.QRectF(
                extent[0], extent[2], extent[1] - extent[0], extent[3] - extent[2]
            )
        )

        rx_x, rx_y = float(frame.rx_pos[0]), float(frame.rx_pos[1])
        self._rx_scatter.setData([rx_x], [rx_y])
        self._rx_label.setPos(rx_x, max(-0.10, rx_y - 0.08))

        links: Sequence[Any] = getattr(frame, "links", None) or []
        (q_seg, m_seg, h_seg, ffz_seg, tx_nodes) = _build_chord_geometry(
            links, self.anchor_positions
        )
        self._chord_quiet.setData(q_seg[0], q_seg[1])
        self._chord_mid.setData(m_seg[0], m_seg[1])
        self._chord_hot.setData(h_seg[0], h_seg[1])
        self._ffz_item.setData(ffz_seg[0], ffz_seg[1])

        if tx_nodes:
            self._tx_scatter.setData([n[1] for n in tx_nodes], [n[2] for n in tx_nodes])
        else:
            self._tx_scatter.setData([], [])

        for idx, ti in enumerate(self._anchor_items):
            if idx < len(tx_nodes):
                bssid, nx, ny = tx_nodes[idx]
                ti.setText(self.anchor_labels.get(bssid, bssid[-5:]))
                ti.setPos(nx, ny + 0.07)
                ti.setVisible(True)
            else:
                ti.setVisible(False)

        tracks: Sequence[Any] = getattr(frame, "tracks", None) or []
        trails = self._history.update(tracks, float(getattr(frame, "timestamp", 0.0)))
        trail_segs: list[tuple[np.ndarray, np.ndarray]] = []
        vel_segs: list[tuple[np.ndarray, np.ndarray]] = []
        tx_pts: list[float] = []
        ty_pts: list[float] = []
        for trk in tracks:
            tid = int(getattr(trk, "track_id", 0))
            pts = trails.get(tid)
            if pts and len(pts) >= 2:
                arr = np.asarray(pts, dtype=np.float64)
                trail_segs.append((arr[:, 0], arr[:, 1]))
            x, y = float(trk.x), float(trk.y)
            vx, vy = float(getattr(trk, "vx", 0.0)), float(getattr(trk, "vy", 0.0))
            tx_pts.append(x)
            ty_pts.append(y)
            if math.hypot(vx, vy) >= 0.08:
                vel_segs.append(
                    (
                        np.array([x, x + 0.55 * vx], dtype=np.float64),
                        np.array([y, y + 0.55 * vy], dtype=np.float64),
                    )
                )
        self._trail_item.setData(*_concat_segments_with_nan(trail_segs))
        self._vel_item.setData(*_concat_segments_with_nan(vel_segs))
        self._trk_scatter.setData(tx_pts, ty_pts)

        for idx, ti in enumerate(self._track_items):
            if idx < len(tracks):
                trk = tracks[idx]
                ti.setText(format_track_badge(trk))
                ti.setPos(float(trk.x) + 0.10, float(trk.y) + 0.08)
                ti.setVisible(True)
            else:
                ti.setVisible(False)

        if self.show_ground_truth and ground_truth:
            self._gt_scatter.setData(
                [float(g["x"]) for g in ground_truth],
                [float(g["y"]) for g in ground_truth],
            )
        else:
            self._gt_scatter.setData([], [])

        csi_amp = getattr(frame, "latest_csi_amp", None)
        csi_phase = getattr(frame, "latest_csi_phase", None)
        if csi_amp is not None and len(csi_amp) == 52:
            self._csi_amp_curve.setData(
                DEFAULT_SUBCARRIER_INDICES,
                np.nan_to_num(np.asarray(csi_amp, dtype=np.float64), nan=0.0),
            )
        if csi_phase is not None and len(csi_phase) == 52:
            ph = np.nan_to_num(np.asarray(csi_phase, dtype=np.float64), nan=0.0)
            ph_wrapped = (ph + math.pi) % (2.0 * math.pi) - math.pi
            self._csi_phase_curve.setData(DEFAULT_SUBCARRIER_INDICES, ph_wrapped)

        psd_f = getattr(frame, "psd_freqs_hz", None)
        psd_p = getattr(frame, "psd_power_db", None)
        if (
            psd_f is not None
            and psd_p is not None
            and len(psd_f) == len(psd_p)
            and len(psd_f) >= 2
        ):
            f_arr = np.asarray(psd_f, dtype=np.float64)
            p_arr = np.nan_to_num(np.asarray(psd_p, dtype=np.float64), nan=-60.0)
            valid = (f_arr >= 0.08) & (f_arr <= 25.0)
            if np.any(valid):
                self._psd_curve.setData(f_arr[valid], p_arr[valid])

        hud = _extract_hud_metrics(frame)
        if hud.peak_resp_hz is not None:
            self._psd_peak_line.setPos(hud.peak_resp_hz)
            self._psd_peak_line.setVisible(True)
        else:
            self._psd_peak_line.setVisible(False)

        tsf_str = (
            f"{hud.mean_tsf_ppm:+.2f} ppm"
            if hud.mean_tsf_ppm is not None
            else "calibrating"
        )
        vital_str = (
            f"{round(hud.peak_resp_bpm):.0f} BPM ({hud.peak_resp_hz:.2f} Hz)"
            if hud.peak_resp_bpm is not None and hud.peak_resp_hz is not None
            else "none"
        )
        hud_html = (
            "<div style='font-family:monospace; color:#e2e8f0'>"
            f"<b>[0 dBm RX-ONLY | AMP: OFF]</b> Backend: {hud.backend_label} | "
            f"Rate: {hud.packet_rate_hz:.1f} pkt/s | BSSIDs: {hud.bssid_count} | "
            f"Chords: {hud.chord_count} (Peer: {hud.peer_chord_count})<br/>"
            f"Beacon={hud.beacons} CTS={hud.cts} ACK={hud.ack} BA={hud.block_ack} "
            f"Probe={hud.probes} Data={hud.data_frames}<br/>"
            f"FCS CRC Drop: {hud.fcs_drop_pct:.1f}% | Seq Gaps: {hud.seq_gaps} | "
            f"TSF Drift: {tsf_str} | ADC: {hud.adc_status}<br/>"
            f"Tracked Humans: {len(tracks)} | Vital Peak: {vital_str} | "
            f"Render: {self._last_render_ms:.1f} ms"
            "</div>"
        )
        try:
            self._hud_label.setText(hud_html)
            self.app.processEvents()
        except _GUI_BACKEND_ERRORS:
            if not self._headless:
                self._closed = True
                return 0.0
            raise
        self._last_render_ms = (time.perf_counter() - t0) * 1000.0
        return self._last_render_ms

    def save_snapshot(self, path: str | Path, dpi: int = 120) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.app.processEvents()
        pix = self.win.grab()
        saved = pix.save(str(out))
        if (
            not saved or not out.is_file() or out.stat().st_size == 0
        ) and self._last_frame is not None:
            # Fallback to Agg Matplotlib snapshot if offscreen QPixmap export failed
            fallback = MatplotlibRadarDashboard(
                room_width=self.room_width,
                room_height=self.room_height,
                anchor_positions=self.anchor_positions,
                anchor_labels=self.anchor_labels,
                headless=True,
                show_ground_truth=self.show_ground_truth,
            )
            try:
                fallback.update(self._last_frame, self._last_gt)
                fallback.save_snapshot(out, dpi=dpi)
            finally:
                fallback.close()
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(*_GUI_BACKEND_ERRORS):
            self.win.close()
            self.win.deleteLater()
            self.app.processEvents()


def _sparkline(values: np.ndarray, width: int = 44) -> str:
    """Downsample/interpolate 1D array into a compact Unicode block sparkline."""
    if values.size == 0:
        return " " * width
    arr = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    if arr.size != width:
        xs_old = np.linspace(0.0, 1.0, arr.size)
        xs_new = np.linspace(0.0, 1.0, width)
        arr = np.interp(xs_new, xs_old, arr)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    if span < 1e-9:
        return SPARK_BLOCKS[1] * width
    norm = np.clip((arr - lo) / span, 0.0, 1.0)
    idxs = np.round(norm * (len(SPARK_BLOCKS) - 1)).astype(int)
    return "".join(SPARK_BLOCKS[int(i)] for i in idxs)


def _inferno_rgb(val: float) -> tuple[int, int, int]:
    """Map normalized scalar in [0, 1] to dark-radar RGB tuple."""
    v = max(0.0, min(1.0, float(val)))
    if v < 0.30:
        u = v / 0.30
        return (int(12 + 75 * u), int(18 + 10 * u), int(38 + 90 * u))
    if v < 0.68:
        u = (v - 0.30) / 0.38
        return (int(87 + 145 * u), int(28 + 75 * u), int(128 - 65 * u))
    u = (v - 0.68) / 0.32
    return (int(232 + 23 * u), int(103 + 145 * u), int(63 + 95 * u))


class RichTuiRadarDashboard:
    """Rich ANSI Truecolor Unicode half-block 2D tomographic heatmap + telemetry TUI."""

    def __init__(
        self,
        room_width: float = 6.0,
        room_height: float = 5.0,
        anchor_positions: Mapping[str, tuple[float, float]] | None = None,
        anchor_labels: Mapping[str, str] | None = None,
        headless: bool = False,
        show_ground_truth: bool = True,
        map_cols: int = 48,
        map_char_rows: int = 18,
    ) -> None:
        from rich.console import Console
        from rich.live import Live

        self.room_width = float(room_width)
        self.room_height = float(room_height)
        self.anchor_positions = dict(anchor_positions or {})
        self.anchor_labels = dict(anchor_labels or {})
        self.show_ground_truth = bool(show_ground_truth)
        self.map_cols = max(24, int(map_cols))
        self.map_char_rows = max(10, int(map_char_rows))
        self.headless = bool(headless)
        tsize = shutil.get_terminal_size((116, 30))
        term_cols = 116 if self.headless else max(116, tsize.columns)
        term_rows = 30 if self.headless else max(28, tsize.lines)
        self._console = Console(
            width=term_cols,
            height=term_rows,
            force_terminal=not headless,
        )
        self._live: Live | None = None
        if not self.headless:
            self._live = Live(
                console=self._console,
                refresh_per_second=20,
                transient=False,
            )
            self._live.start()

        self._history = TrackHistoryBuffer()
        self._last_render_ms = 0.0
        self._last_renderable: Any = None
        self._last_frame: Any = None
        self._last_gt: Sequence[Mapping[str, Any]] | None = None
        self._closed = False

    @property
    def last_render_ms(self) -> float:
        return self._last_render_ms

    def build_renderable(
        self,
        frame: Any,
        ground_truth: Sequence[Mapping[str, Any]] | None = None,
    ) -> Any:
        """Build a complete Rich dashboard renderable for the current TomographyFrame."""
        from rich.panel import Panel
        from rich.style import Style
        from rich.table import Table
        from rich.text import Text

        dm, _ = _normalize_density_map(frame, self.room_width, self.room_height)
        sub_h = self.map_char_rows * 2
        sub_w = self.map_cols

        # Resample density_map (Ny, Nx) onto (sub_h, sub_w), flipping Y so top row = room_height
        ny, nx = dm.shape
        y_idx = np.clip(
            np.round(np.linspace(ny - 1, 0, sub_h)).astype(int), 0, max(0, ny - 1)
        )
        x_idx = np.clip(
            np.round(np.linspace(0, nx - 1, sub_w)).astype(int), 0, max(0, nx - 1)
        )
        grid = dm[np.ix_(y_idx, x_idx)]
        vmax = float(np.max(grid)) if grid.size > 0 else 0.0
        norm_grid = grid / vmax if vmax > 1e-9 else np.zeros_like(grid)

        # Character overlay matrix on (map_char_rows, map_cols)
        overlay_ch: dict[tuple[int, int], tuple[str, str]] = {}

        def _xy_to_cell(x_m: float, y_m: float) -> tuple[int, int]:
            c = round((x_m / max(self.room_width, 1e-6)) * (self.map_cols - 1))
            r = round(
                ((self.room_height - y_m) / max(self.room_height, 1e-6))
                * (self.map_char_rows - 1)
            )
            return (
                max(0, min(self.map_char_rows - 1, r)),
                max(0, min(self.map_cols - 1, c)),
            )

        links: Sequence[Any] = getattr(frame, "links", None) or []
        for lnk in links:
            score = float(getattr(lnk, "perturbation_score", 0.0) or 0.0)
            if score >= 0.22:
                tx0, ty0 = float(lnk.tx_pos[0]), float(lnk.tx_pos[1])
                rx0, ry0 = float(lnk.rx_pos[0]), float(lnk.rx_pos[1])
                for s in np.linspace(0.12, 0.88, 12):
                    rc = _xy_to_cell(tx0 + s * (rx0 - tx0), ty0 + s * (ry0 - ty0))
                    overlay_ch.setdefault(
                        rc, ("·", "#fbbf24" if score < 0.45 else "#fb7185")
                    )
            bssid = str(getattr(lnk, "tx_bssid", ""))
            if bssid:
                overlay_ch[_xy_to_cell(float(lnk.tx_pos[0]), float(lnk.tx_pos[1]))] = (
                    "A",
                    "#38bdf8",
                )

        for pos in self.anchor_positions.values():
            overlay_ch[_xy_to_cell(float(pos[0]), float(pos[1]))] = ("A", "#38bdf8")

        if self.show_ground_truth and ground_truth:
            for g in ground_truth:
                overlay_ch[_xy_to_cell(float(g["x"]), float(g["y"]))] = ("+", "#4ade80")

        rx_x, rx_y = float(frame.rx_pos[0]), float(frame.rx_pos[1])
        overlay_ch[_xy_to_cell(rx_x, rx_y)] = ("R", "#22c55e")

        tracks: Sequence[Any] = getattr(frame, "tracks", None) or []
        self._history.update(tracks, float(getattr(frame, "timestamp", 0.0)))
        for trk in tracks:
            tid_ch = str(int(getattr(trk, "track_id", 1)) % 10)
            overlay_ch[_xy_to_cell(float(trk.x), float(trk.y))] = (tid_ch, "#fef08a")

        heatmap_text = Text(no_wrap=True)
        for r in range(self.map_char_rows):
            for c in range(self.map_cols):
                top_rgb = _inferno_rgb(float(norm_grid[2 * r, c]))
                bot_rgb = _inferno_rgb(float(norm_grid[2 * r + 1, c]))
                ov = overlay_ch.get((r, c))
                if ov is not None:
                    ch, fg_hex = ov
                    avg_bg = (
                        (top_rgb[0] + bot_rgb[0]) // 2,
                        (top_rgb[1] + bot_rgb[1]) // 2,
                        (top_rgb[2] + bot_rgb[2]) // 2,
                    )
                    heatmap_text.append(
                        ch,
                        style=Style(
                            color=fg_hex,
                            bgcolor=f"rgb({avg_bg[0]},{avg_bg[1]},{avg_bg[2]})",
                            bold=True,
                        ),
                    )
                else:
                    heatmap_text.append(
                        "▀",
                        style=Style(
                            color=f"rgb({top_rgb[0]},{top_rgb[1]},{top_rgb[2]})",
                            bgcolor=f"rgb({bot_rgb[0]},{bot_rgb[1]},{bot_rgb[2]})",
                        ),
                    )
            if r < self.map_char_rows - 1:
                heatmap_text.append("\n")

        map_panel = Panel(
            heatmap_text,
            title=f"2D Tomography ({self.room_width:.1f}x{self.room_height:.1f}m) [R=RX A=AP 1-9=Human]",
            border_style="cyan",
        )

        # Right side: Tracked Humans + CSI + PSD + Passive Wi-Fi HUD
        trk_table = Table(expand=False, box=None, pad_edge=False)
        trk_table.add_column("ID", style="bold cyan", width=4)
        trk_table.add_column("(x, y) m", width=12)
        trk_table.add_column("|v|", width=8)
        trk_table.add_column("State", width=11)
        trk_table.add_column("Breathing", style="bold yellow", width=10)
        if tracks:
            for trk in tracks[:6]:
                spd = math.hypot(
                    float(getattr(trk, "vx", 0.0)), float(getattr(trk, "vy", 0.0))
                )
                bpm = getattr(trk, "breathing_bpm", None)
                bpm_badge = (
                    f"{round(float(bpm)):.0f} BPM"
                    if bpm is not None and np.isfinite(float(bpm)) and float(bpm) > 0.0
                    else "-"
                )
                trk_table.add_row(
                    f"T{trk.track_id}",
                    f"({trk.x:4.2f},{trk.y:4.2f})",
                    f"{spd:4.2f}m/s",
                    str(trk.activity),
                    bpm_badge,
                )
        else:
            trk_table.add_row("-", "no active", "targets", "-", "-")

        csi_amp = getattr(frame, "latest_csi_amp", None)
        csi_phase = getattr(frame, "latest_csi_phase", None)
        amp_spark = _sparkline(
            np.asarray(csi_amp) if csi_amp is not None else np.zeros(16), 38
        )
        ph_spark = _sparkline(
            np.asarray(csi_phase) if csi_phase is not None else np.zeros(16), 38
        )

        psd_f = getattr(frame, "psd_freqs_hz", None)
        psd_p = getattr(frame, "psd_power_db", None)
        if (
            psd_f is not None
            and psd_p is not None
            and len(psd_f) == len(psd_p)
            and len(psd_f) > 0
        ):
            f_arr = np.asarray(psd_f, dtype=np.float64)
            p_arr = np.asarray(psd_p, dtype=np.float64)
            mask = (f_arr >= 0.1) & (f_arr <= 25.0)
            psd_spark = _sparkline(p_arr[mask] if np.any(mask) else p_arr, 38)
        else:
            psd_spark = _sparkline(np.zeros(16), 38)

        hud = _extract_hud_metrics(frame)
        tsf_str = (
            f"{hud.mean_tsf_ppm:+.2f} ppm"
            if hud.mean_tsf_ppm is not None
            else "calibrating"
        )
        vital_str = (
            f"{round(hud.peak_resp_bpm):.0f} BPM ({hud.peak_resp_hz:.2f} Hz)"
            if hud.peak_resp_bpm is not None and hud.peak_resp_hz is not None
            else "none"
        )

        hud_body = Text()
        hud_body.append(
            "0 dBm RX-ONLY (AMP OFF) | CLEARTEXT 802.11\n", style="bold green"
        )
        hud_body.append(
            f"Rate: {hud.packet_rate_hz:5.1f} pkt/s | BSSIDs: {hud.bssid_count} | "
            f"Chords: {hud.chord_count} (Peer: {hud.peer_chord_count})\n"
        )
        hud_body.append(
            f"Beacon:{hud.beacons} CTS:{hud.cts} ACK:{hud.ack} "
            f"BA:{hud.block_ack} Probe:{hud.probes} Data:{hud.data_frames}\n"
        )
        hud_body.append(
            f"FCS Drop: {hud.fcs_drop_pct:4.1f}% | SeqGaps: {hud.seq_gaps} | "
            f"TSF: {tsf_str} | ADC: {hud.adc_status}\n"
        )
        hud_body.append(f"L-LTF |H[k]|: {amp_spark}\n", style="bright_cyan")
        hud_body.append(f"L-LTF ∠H[k] : {ph_spark}\n", style="bright_magenta")
        hud_body.append(f"PSD 0.1-25Hz: {psd_spark}\n", style="bright_yellow")
        hud_body.append(
            f"Vital Peak: {vital_str} | Render: {self._last_render_ms:4.1f} ms",
            style="bold bright_green",
        )

        right_grid = Table.grid(padding=(0, 0))
        right_grid.add_row(
            Panel(trk_table, title="Tracked Humans & Breathing", border_style="yellow")
        )
        right_grid.add_row(
            Panel(
                hud_body, title="Passive Wi-Fi, CSI & Doppler HUD", border_style="green"
            )
        )

        root = Table.grid(padding=(0, 1))
        root.add_row(map_panel, right_grid)
        return root

    def render_ansi(
        self,
        frame: Any,
        ground_truth: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        """Render the TUI frame into an ANSI-formatted string (useful for SSH/tests)."""
        from rich.console import Console

        buf_console = Console(
            file=io.StringIO(),
            width=116,
            height=30,
            force_terminal=True,
            color_system="truecolor",
            record=True,
        )
        renderable = self.build_renderable(frame, ground_truth)
        buf_console.print(renderable)
        return buf_console.export_text(styles=True)

    def update(
        self,
        frame: Any,
        ground_truth: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        if self._closed:
            return 0.0
        t0 = time.perf_counter()
        self._last_frame = frame
        self._last_gt = ground_truth
        renderable = self.build_renderable(frame, ground_truth)
        self._last_renderable = renderable
        if self._live is not None:
            self._live.update(renderable, refresh=True)
        self._last_render_ms = (time.perf_counter() - t0) * 1000.0
        return self._last_render_ms

    def save_snapshot(self, path: str | Path, dpi: int = 120) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        suffix = out.suffix.lower()
        if suffix in {".txt", ".ansi"} and self._last_frame is not None:
            out.write_text(
                self.render_ansi(self._last_frame, self._last_gt), encoding="utf-8"
            )
            return out
        if self._last_frame is not None:
            # Export a crisp PNG snapshot via headless Agg dashboard when .png requested
            snap = MatplotlibRadarDashboard(
                room_width=self.room_width,
                room_height=self.room_height,
                anchor_positions=self.anchor_positions,
                anchor_labels=self.anchor_labels,
                headless=True,
                show_ground_truth=self.show_ground_truth,
            )
            try:
                snap.update(self._last_frame, self._last_gt)
                snap.save_snapshot(out, dpi=dpi)
            finally:
                snap.close()
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._live is not None:
            with contextlib.suppress(RuntimeError, OSError, AttributeError):
                self._live.stop()
            self._live = None


def create_plotter(
    backend: str = "auto",
    *,
    no_gui: bool = False,
    headless: bool = False,
    room_width: float = 6.0,
    room_height: float = 5.0,
    anchor_positions: Mapping[str, tuple[float, float]] | None = None,
    anchor_labels: Mapping[str, str] | None = None,
    show_ground_truth: bool = True,
) -> MatplotlibRadarDashboard | PyQtGraphRadarDashboard | RichTuiRadarDashboard:
    """Instantiate the requested radar dashboard backend."""
    resolved = select_ui_backend(backend, no_gui=no_gui)
    if resolved == "pyqtgraph":
        return PyQtGraphRadarDashboard(
            room_width=room_width,
            room_height=room_height,
            anchor_positions=anchor_positions,
            anchor_labels=anchor_labels,
            headless=headless or no_gui,
            show_ground_truth=show_ground_truth,
        )
    if resolved == "matplotlib":
        return MatplotlibRadarDashboard(
            room_width=room_width,
            room_height=room_height,
            anchor_positions=anchor_positions,
            anchor_labels=anchor_labels,
            headless=headless or no_gui,
            show_ground_truth=show_ground_truth,
        )
    return RichTuiRadarDashboard(
        room_width=room_width,
        room_height=room_height,
        anchor_positions=anchor_positions,
        anchor_labels=anchor_labels,
        headless=headless,
        show_ground_truth=show_ground_truth,
    )
