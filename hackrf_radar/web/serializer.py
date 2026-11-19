"""Compact, JSON-safe serialization of TomographyFrame data for realtime web streaming."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..capture.packet_event import PILOT_INDICES, SUBCARRIER_INDICES
from ..plotter import (
    TrackHistoryBuffer,
    _extract_hud_metrics,
    _normalize_density_map,
)
from ..tomography.geometry import canonicalize_mac


def _safe_float(val: Any, digits: int = 4, default: float = 0.0) -> float:
    """Convert scalar to a finite rounded float safe for strict RFC 8259 JSON."""
    if val is None:
        return default
    try:
        num = float(val)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return round(num, digits)


def _safe_opt_float(val: Any, digits: int = 3) -> float | None:
    """Convert optional scalar to a finite rounded float or None."""
    if val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return round(num, digits)


def _safe_float_list(arr: Any, digits: int = 4) -> list[float]:
    """Convert a 1D or ND numeric array to a flat list of finite rounded floats."""
    if arr is None:
        return []
    raw = np.asarray(arr, dtype=np.float64).ravel()
    if raw.size == 0:
        return []
    clean = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return [round(float(x), digits) for x in clean.tolist()]


def serialize_frame(
    frame: Any,
    *,
    room_width: float = 6.0,
    room_height: float = 5.0,
    anchor_positions: Mapping[str, tuple[float, float]] | None = None,
    anchor_labels: Mapping[str, str] | None = None,
    ground_truth: Sequence[Mapping[str, Any]] | None = None,
    show_ground_truth: bool = True,
    track_history: TrackHistoryBuffer | None = None,
    frame_seq: int = 0,
    server_fps: float = 0.0,
) -> dict[str, Any]:
    """Serialize a ``TomographyFrame`` into a compact JSON-safe dictionary."""
    rw = max(0.5, _safe_float(room_width, digits=3, default=6.0))
    rh = max(0.5, _safe_float(room_height, digits=3, default=5.0))
    ts = _safe_float(getattr(frame, "timestamp", 0.0), digits=4)
    rx_raw = getattr(frame, "rx_pos", (rw * 0.5, 0.35))
    rx_pos = [
        _safe_float(rx_raw[0], digits=3),
        _safe_float(rx_raw[1], digits=3),
    ]

    dm, extent = _normalize_density_map(frame, rw, rh)
    ny, nx = int(dm.shape[0]), int(dm.shape[1])
    max_density = _safe_float(float(np.max(dm)) if dm.size else 0.0, digits=4)

    labels_map: dict[str, str] = {}
    if anchor_labels:
        for bssid, lbl in anchor_labels.items():
            labels_map[canonicalize_mac(bssid)] = str(lbl)

    nodes_map: dict[str, dict[str, Any]] = {}
    if anchor_positions:
        for bssid, pos in anchor_positions.items():
            canon = canonicalize_mac(bssid)
            nodes_map[canon] = {
                "bssid": canon,
                "label": labels_map.get(canon, canon[-5:]),
                "x": _safe_float(pos[0], digits=3),
                "y": _safe_float(pos[1], digits=3),
                "known": True,
            }

    links_raw: Sequence[Any] = getattr(frame, "links", None) or []
    serialized_links: list[dict[str, Any]] = []
    for lnk in links_raw:
        tx_bssid = canonicalize_mac(str(getattr(lnk, "tx_bssid", "")))
        rx_id_raw = str(getattr(lnk, "rx_id", "RX"))
        rx_id = "RX" if rx_id_raw == "RX" else canonicalize_mac(rx_id_raw)
        tx_p = getattr(lnk, "tx_pos", (0.0, 0.0))
        rx_p = getattr(lnk, "rx_pos", rx_pos)

        if tx_bssid and tx_bssid not in nodes_map:
            nodes_map[tx_bssid] = {
                "bssid": tx_bssid,
                "label": labels_map.get(tx_bssid, tx_bssid[-5:]),
                "x": _safe_float(tx_p[0], digits=3),
                "y": _safe_float(tx_p[1], digits=3),
                "known": False,
            }

        link_entry: dict[str, Any] = {
            "link_id": str(getattr(lnk, "link_id", f"{tx_bssid}->{rx_id}")),
            "tx_bssid": tx_bssid,
            "rx_id": rx_id,
            "tx_label": labels_map.get(tx_bssid, tx_bssid[-5:] if tx_bssid else "TX"),
            "rx_label": (
                "HackRF RX"
                if rx_id == "RX"
                else labels_map.get(rx_id, rx_id[-5:] if rx_id else "RX")
            ),
            "tx_pos": [_safe_float(tx_p[0], digits=3), _safe_float(tx_p[1], digits=3)],
            "rx_pos": [_safe_float(rx_p[0], digits=3), _safe_float(rx_p[1], digits=3)],
            "freq_hz": _safe_float(getattr(lnk, "freq_hz", 2437e6), digits=1),
            "is_peer": bool(getattr(lnk, "is_peer", False)),
            "packet_count": int(getattr(lnk, "packet_count", 0) or 0),
            "rssi_delta_db": _safe_float(getattr(lnk, "rssi_delta_db", 0.0), digits=2),
            "csi_turbulence": _safe_float(
                getattr(lnk, "csi_turbulence", 0.0), digits=4
            ),
            "perturbation_score": _safe_float(
                getattr(lnk, "perturbation_score", 0.0), digits=4
            ),
            "breathing_hz": _safe_opt_float(getattr(lnk, "breathing_hz", None), 4),
            "breathing_bpm": _safe_opt_float(getattr(lnk, "breathing_bpm", None), 1),
            "breathing_snr_db": _safe_float(
                getattr(lnk, "breathing_snr_db", 0.0), digits=2
            ),
            "fcs_error_rate": _safe_float(
                getattr(lnk, "fcs_error_rate", 0.0), digits=4
            ),
            "seq_gap_rate": _safe_float(getattr(lnk, "seq_gap_rate", 0.0), digits=4),
            "tsf_drift_ppm": _safe_opt_float(getattr(lnk, "tsf_drift_ppm", None), 2),
        }

        lnk_csi_amp = getattr(lnk, "csi_amp", None)
        if lnk_csi_amp is not None:
            link_entry["csi_amp"] = _safe_float_list(lnk_csi_amp, digits=4)
        lnk_csi_phase = getattr(lnk, "csi_phase", None)
        if lnk_csi_phase is not None:
            link_entry["csi_phase"] = _safe_float_list(lnk_csi_phase, digits=4)
        lnk_base_amp = getattr(lnk, "baseline_csi_amp", None)
        if lnk_base_amp is not None:
            link_entry["baseline_csi_amp"] = _safe_float_list(lnk_base_amp, digits=4)

        serialized_links.append(link_entry)

    tracks_raw: Sequence[Any] = getattr(frame, "tracks", None) or []
    trails_map = (
        track_history.update(tracks_raw, ts) if track_history is not None else {}
    )
    serialized_tracks: list[dict[str, Any]] = []
    for trk in tracks_raw:
        tid = int(getattr(trk, "track_id", 0))
        vx = _safe_float(getattr(trk, "vx", 0.0), digits=3)
        vy = _safe_float(getattr(trk, "vy", 0.0), digits=3)
        trail_pts = [
            [_safe_float(px, digits=3), _safe_float(py, digits=3)]
            for px, py in trails_map.get(tid, ())
        ]
        serialized_tracks.append(
            {
                "track_id": tid,
                "x": _safe_float(getattr(trk, "x", 0.0), digits=3),
                "y": _safe_float(getattr(trk, "y", 0.0), digits=3),
                "vx": vx,
                "vy": vy,
                "speed_mps": round(math.hypot(vx, vy), 3),
                "activity": str(getattr(trk, "activity", "stationary")),
                "breathing_bpm": _safe_opt_float(
                    getattr(trk, "breathing_bpm", None), 1
                ),
                "confidence": _safe_float(getattr(trk, "confidence", 0.0), digits=3),
                "peak_density": _safe_float(
                    getattr(trk, "peak_density", 0.0), digits=3
                ),
                "trail": trail_pts,
            }
        )

    serialized_gt: list[dict[str, Any]] = []
    if show_ground_truth and ground_truth:
        for gt in ground_truth:
            serialized_gt.append(
                {
                    "id": int(gt.get("id", 0)),
                    "x": _safe_float(gt.get("x", 0.0), digits=3),
                    "y": _safe_float(gt.get("y", 0.0), digits=3),
                    "vx": _safe_float(gt.get("vx", 0.0), digits=3),
                    "vy": _safe_float(gt.get("vy", 0.0), digits=3),
                    "activity": str(gt.get("activity", "stationary")),
                    "breathing_bpm": _safe_opt_float(gt.get("breathing_bpm"), 1),
                }
            )

    hud = _extract_hud_metrics(frame)
    tel: Mapping[str, Any] = getattr(frame, "telemetry", None) or {}
    solver_params = dict(tel.get("solver_params") or {})
    solver_method = str(
        solver_params.get("solver_method") or tel.get("solver_method") or "fista_tv"
    )
    conv_raw = tel.get("solver_convergence") or []
    conv_list = _safe_float_list(conv_raw, digits=4)

    fc_raw = tel.get("frame_counts") or {}
    frame_counts = {str(k).upper(): int(v) for k, v in fc_raw.items()}

    return {
        "seq": int(frame_seq),
        "timestamp": ts,
        "server_fps": _safe_float(server_fps, digits=2),
        "room": {
            "width": rw,
            "height": rh,
            "extent": [_safe_float(e, digits=3) for e in extent],
            "grid_nx": nx,
            "grid_ny": ny,
            "rx_pos": rx_pos,
            "anchors": [nodes_map[k] for k in sorted(nodes_map)],
        },
        "density_map": {
            "nx": nx,
            "ny": ny,
            "max": max_density,
            "values": _safe_float_list(dm, digits=3),
        },
        "links": serialized_links,
        "tracks": serialized_tracks,
        "ground_truth": serialized_gt,
        "csi": {
            "subcarriers": list(SUBCARRIER_INDICES),
            "pilots": list(PILOT_INDICES),
            "amp": _safe_float_list(getattr(frame, "latest_csi_amp", None), digits=4),
            "phase": _safe_float_list(
                getattr(frame, "latest_csi_phase", None), digits=4
            ),
        },
        "psd": {
            "freqs_hz": _safe_float_list(
                getattr(frame, "psd_freqs_hz", None), digits=3
            ),
            "power_db": _safe_float_list(
                getattr(frame, "psd_power_db", None), digits=2
            ),
            "peak_resp_hz": _safe_opt_float(hud.peak_resp_hz, 4),
            "peak_resp_bpm": _safe_opt_float(hud.peak_resp_bpm, 1),
        },
        "solver": {
            "method": solver_method,
            "lam_tv": _safe_float(solver_params.get("lam_tv", 0.006), digits=4),
            "lam_l2": _safe_float(solver_params.get("lam_l2", 0.020), digits=4),
            "window_s": _safe_float(solver_params.get("window_s", 1.5), digits=2),
            "convergence": conv_list,
        },
        "hud": {
            "packet_rate_hz": _safe_float(hud.packet_rate_hz, digits=1),
            "bssid_count": int(hud.bssid_count),
            "chord_count": int(hud.chord_count),
            "peer_chord_count": int(hud.peer_chord_count),
            "beacons": int(hud.beacons),
            "cts": int(hud.cts),
            "ack": int(hud.ack),
            "block_ack": int(hud.block_ack),
            "probes": int(hud.probes),
            "data_frames": int(hud.data_frames),
            "fcs_drop_pct": _safe_float(hud.fcs_drop_pct, digits=2),
            "seq_gaps": int(hud.seq_gaps),
            "seq_gap_rate": _safe_float(hud.seq_gap_rate, digits=4),
            "mean_tsf_ppm": _safe_opt_float(hud.mean_tsf_ppm, 2),
            "adc_clip_max": _safe_float(hud.adc_clip_max, digits=5),
            "adc_status": hud.adc_status,
            "backend_label": hud.backend_label,
            "peak_resp_hz": _safe_opt_float(hud.peak_resp_hz, 4),
            "peak_resp_bpm": _safe_opt_float(hud.peak_resp_bpm, 1),
            "frame_counts": frame_counts,
            "passive_rx_only": True,
            "amp_enable": False,
            "bias_tee_enable": False,
        },
    }


def serialize_frame_json(frame: Any, **kwargs: Any) -> str:
    """Serialize a ``TomographyFrame`` directly into a compact strict JSON string."""
    payload = serialize_frame(frame, **kwargs)
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)
