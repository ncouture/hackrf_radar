"""Physics-based multi-human passive 802.11 RTI, 52-subcarrier CSI, & breathing simulator."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import canonicalize_bssid
from ..tomography.geometry import (
    SPEED_OF_LIGHT,
    compute_excess_path,
    subcarrier_frequencies_hz,
    wavelength_m,
)
from .packet_event import (
    SUBCARRIER_INDICES,
    PacketEvent,
    channel_to_freq_hz,
    sanitize_csi,
)


@dataclass(slots=True)
class _SimAnchor:
    bssid: str
    x: float
    y: float
    channel: int
    role: str  # "ap" | "iot"
    drift_ppm: float
    tx_power_offset_db: float = 0.0


@dataclass(slots=True)
class _SimHumanSpec:
    target_id: int
    mode: str  # "walk_line" | "stationary" | "orbit"
    x0: float
    y0: float
    x1: float
    y1: float
    speed_mps: float
    breathing_bpm: float | None
    phase0: float


class SyntheticRadarStream:
    """Deterministic multi-human passive Wi-Fi packet, L-LTF CSI, & SIFS chord simulator.

    Models:
      - 8 perimeter ambient Wi-Fi AP + IoT nodes across 2.4 GHz (Ch 1/6/11) and 5 GHz UNII
      - First Fresnel Zone (Delta < lambda/2) human body shadowing attenuation
      - 52-subcarrier L-LTF multipath bistatic scattering & excess-path phase progression
      - Stationary chest-wall micro-displacement (12..24 BPM, e.g. 17 BPM = 0.283 Hz)
      - SIFS control exchanges (RTS/CTS, Data/ACK, BlockAck) adding cross-room peer chords
      - Per-packet STO/PDD linear phase slope + random CFO/LO phase offset + AWGN
    """

    def __init__(
        self,
        room_width: float = 6.0,
        room_height: float = 5.0,
        rx_pos: tuple[float, float] = (3.0, 0.35),
        num_humans: int = 2,
        scenario: str = "walk_and_breathe",
        channels: Sequence[int] = (1, 6, 11),
        seed: int = 42,
        packet_rate_hz: float = 340.0,
        dwell_ms: float = 75.0,
        tx_positions: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.room_width = float(room_width)
        self.room_height = float(room_height)
        self.rx_pos: tuple[float, float] = (float(rx_pos[0]), float(rx_pos[1]))
        self.scenario = str(scenario or "walk_and_breathe").strip().lower()
        self.channels: tuple[int, ...] = tuple(int(c) for c in (channels or (1, 6, 11)))
        for ch in self.channels:
            channel_to_freq_hz(ch)
        self.seed = int(seed)
        self.packet_rate_hz = max(40.0, float(packet_rate_hz))
        self.dwell_ms = max(15.0, float(dwell_ms))

        if self.scenario == "empty":
            self.num_humans = 0
        elif self.scenario in {"single_walk", "single_breathe"} and num_humans == 2:
            self.num_humans = 1
        elif self.scenario == "three_humans" and num_humans < 3:
            self.num_humans = 3
        else:
            self.num_humans = max(0, int(num_humans))

        self._rng = np.random.default_rng(self.seed)
        self._anchors: list[_SimAnchor] = self._build_anchors(tx_positions)
        self._humans: list[_SimHumanSpec] = self._build_humans()

        # Static multipath clutter profile per anchor (deterministic from seed)
        k_arr = np.asarray(SUBCARRIER_INDICES, dtype=np.float64)
        self._static_clutter: dict[str, np.ndarray] = {}
        for anc in self._anchors:
            tau1 = float(self._rng.uniform(18e-9, 55e-9))
            tau2 = float(self._rng.uniform(60e-9, 115e-9))
            a1 = float(self._rng.uniform(0.08, 0.17)) * np.exp(
                1j * float(self._rng.uniform(0, 2 * math.pi))
            )
            a2 = float(self._rng.uniform(0.04, 0.10)) * np.exp(
                1j * float(self._rng.uniform(0, 2 * math.pi))
            )
            df = 312_500.0 * k_arr
            self._static_clutter[anc.bssid] = a1 * np.exp(
                -1j * 2.0 * math.pi * df * tau1
            ) + a2 * np.exp(-1j * 2.0 * math.pi * df * tau2)

        self._t0: float = 1_700_000_000.0
        self._last_ts: float = self._t0
        self._override_lock = threading.Lock()
        self._target_overrides: dict[int, tuple[float, float]] = {}

    def _build_anchors(
        self, tx_positions: Mapping[str, tuple[float, float]] | None
    ) -> list[_SimAnchor]:
        sx = self.room_width / 6.0
        sy = self.room_height / 5.0
        defaults = [
            ("AA:BB:CC:00:00:01", 0.40 * sx, 4.70 * sy, 1, "ap", +6.4),
            ("AA:BB:CC:00:00:02", 3.00 * sx, 4.75 * sy, 6, "ap", -4.8),
            ("AA:BB:CC:00:00:03", 5.60 * sx, 4.70 * sy, 11, "ap", +9.1),
            ("AA:BB:CC:00:00:04", 0.25 * sx, 2.60 * sy, 6, "iot", -7.2),
            ("AA:BB:CC:00:00:05", 5.75 * sx, 2.60 * sy, 1, "iot", +3.5),
            ("AA:BB:CC:00:00:06", 0.35 * sx, 1.15 * sy, 11, "iot", -2.9),
            ("AA:BB:CC:00:00:07", 5.65 * sx, 1.15 * sy, 6, "iot", +5.2),
            ("AA:BB:CC:00:00:08", 1.70 * sx, 4.75 * sy, 11, "iot", -6.1),
        ]
        ch_list = list(self.channels)
        out: list[_SimAnchor] = []
        if tx_positions:
            for idx, (mac, pos) in enumerate(tx_positions.items()):
                canon = canonicalize_bssid(mac)
                ch = ch_list[idx % len(ch_list)]
                role = "ap" if idx < max(1, len(tx_positions) // 2) else "iot"
                drift = float((idx - 3.5) * 2.3)
                out.append(
                    _SimAnchor(canon, float(pos[0]), float(pos[1]), ch, role, drift)
                )
            return out

        for idx, (mac, x, y, pref_ch, role, drift) in enumerate(defaults):
            ch = pref_ch if pref_ch in self.channels else ch_list[idx % len(ch_list)]
            out.append(_SimAnchor(mac, x, y, ch, role, drift))
        return out

    def _build_humans(self) -> list[_SimHumanSpec]:
        n = self.num_humans
        if n <= 0 or self.scenario == "empty":
            return []

        sx = self.room_width / 6.0
        sy = self.room_height / 5.0

        if self.scenario == "single_breathe":
            return [
                _SimHumanSpec(
                    target_id=1,
                    mode="stationary",
                    x0=1.35 * sx,
                    y0=3.42 * sy,
                    x1=1.35 * sx,
                    y1=3.42 * sy,
                    speed_mps=0.0,
                    breathing_bpm=17.0,
                    phase0=0.35,
                )
            ]

        if self.scenario == "single_walk" or (
            n == 1 and self.scenario not in {"single_breathe"}
        ):
            return [
                _SimHumanSpec(
                    target_id=1,
                    mode="walk_line",
                    x0=1.55 * sx,
                    y0=1.80 * sy,
                    x1=4.55 * sx,
                    y1=3.20 * sy,
                    speed_mps=0.46,
                    breathing_bpm=None,
                    phase0=0.0,
                )
            ]

        if self.scenario == "two_walkers":
            specs = [
                _SimHumanSpec(
                    target_id=1,
                    mode="walk_line",
                    x0=1.35 * sx,
                    y0=1.75 * sy,
                    x1=1.85 * sx,
                    y1=3.95 * sy,
                    speed_mps=0.38,
                    breathing_bpm=None,
                    phase0=0.1,
                ),
                _SimHumanSpec(
                    target_id=2,
                    mode="walk_line",
                    x0=4.65 * sx,
                    y0=3.15 * sy,
                    x1=3.95 * sx,
                    y1=1.60 * sy,
                    speed_mps=0.36,
                    breathing_bpm=None,
                    phase0=1.2,
                ),
            ]
        elif self.scenario == "walk_and_breathe" and n == 2:
            specs = [
                _SimHumanSpec(
                    target_id=1,
                    mode="stationary",
                    x0=1.35 * sx,
                    y0=3.42 * sy,
                    x1=1.35 * sx,
                    y1=3.42 * sy,
                    speed_mps=0.0,
                    breathing_bpm=17.0,
                    phase0=0.40,
                ),
                _SimHumanSpec(
                    target_id=2,
                    mode="walk_line",
                    x0=3.85 * sx,
                    y0=1.55 * sy,
                    x1=4.75 * sx,
                    y1=3.65 * sy,
                    speed_mps=0.36,
                    breathing_bpm=None,
                    phase0=1.10,
                ),
            ]
        else:
            # 3+ humans ("three_humans" or arbitrary N >= 3)
            specs = [
                _SimHumanSpec(
                    target_id=1,
                    mode="stationary",
                    x0=1.35 * sx,
                    y0=3.42 * sy,
                    x1=1.35 * sx,
                    y1=3.42 * sy,
                    speed_mps=0.0,
                    breathing_bpm=17.0,
                    phase0=0.30,
                ),
                _SimHumanSpec(
                    target_id=2,
                    mode="walk_line",
                    x0=4.05 * sx,
                    y0=1.70 * sy,
                    x1=4.65 * sx,
                    y1=3.05 * sy,
                    speed_mps=0.34,
                    breathing_bpm=None,
                    phase0=1.40,
                ),
                _SimHumanSpec(
                    target_id=3,
                    mode="stationary",
                    x0=3.00 * sx,
                    y0=2.90 * sy,
                    x1=3.00 * sx,
                    y1=2.90 * sy,
                    speed_mps=0.0,
                    breathing_bpm=14.5,
                    phase0=2.20,
                ),
            ]

        while len(specs) < n:
            idx = len(specs) + 1
            ang = 2.0 * math.pi * (idx - 1) / max(n, 4)
            cx = (3.0 + 1.45 * math.cos(ang)) * sx
            cy = (2.65 + 1.15 * math.sin(ang)) * sy
            specs.append(
                _SimHumanSpec(
                    target_id=idx,
                    mode="orbit",
                    x0=cx,
                    y0=cy,
                    x1=0.45 * sx,
                    y1=0.45 * sy,
                    speed_mps=0.34,
                    breathing_bpm=None,
                    phase0=ang,
                )
            )
        return specs[:n]

    def get_tx_positions(self) -> dict[str, tuple[float, float]]:
        """Return canonical BSSID -> (x_m, y_m) map of simulated room transmitters."""
        return {a.bssid: (a.x, a.y) for a in self._anchors}

    def set_interactive_target(
        self, target_id: int, x: float, y: float
    ) -> dict[str, Any]:
        """Override a simulated human position in realtime (e.g. from the web UI)."""
        tid = int(target_id)
        if not self._humans:
            raise ValueError("Cannot position target in an empty-room simulation.")
        valid_ids = {s.target_id for s in self._humans}
        if tid not in valid_ids:
            tid = self._humans[0].target_id
        margin = 0.35
        cx = min(max(float(x), margin), max(margin, self.room_width - margin))
        cy = min(max(float(y), margin), max(margin, self.room_height - margin))
        with self._override_lock:
            self._target_overrides[tid] = (cx, cy)
        return {"id": tid, "x": round(cx, 3), "y": round(cy, 3)}

    def clear_interactive_targets(self) -> None:
        """Clear interactive position overrides and resume scripted human trajectories."""
        with self._override_lock:
            self._target_overrides.clear()

    def _eval_human(self, spec: _SimHumanSpec, rel_t: float) -> dict[str, Any]:
        with self._override_lock:
            override = self._target_overrides.get(spec.target_id)
        if override is not None:
            return {
                "id": spec.target_id,
                "x": override[0],
                "y": override[1],
                "vx": 0.0,
                "vy": 0.0,
                "activity": "stationary",
                "breathing_bpm": spec.breathing_bpm,
                "phase0": spec.phase0,
            }
        if spec.mode == "stationary" or spec.speed_mps <= 1e-4:
            return {
                "id": spec.target_id,
                "x": spec.x0,
                "y": spec.y0,
                "vx": 0.0,
                "vy": 0.0,
                "activity": "stationary",
                "breathing_bpm": spec.breathing_bpm,
                "phase0": spec.phase0,
            }

        if spec.mode == "orbit":
            r = max(spec.x1, 0.25)
            omega = spec.speed_mps / r
            theta = spec.phase0 + omega * rel_t
            x = min(max(spec.x0 + r * math.cos(theta), 0.45), self.room_width - 0.45)
            y = min(max(spec.y0 + r * math.sin(theta), 0.45), self.room_height - 0.45)
            vx = -spec.speed_mps * math.sin(theta)
            vy = spec.speed_mps * math.cos(theta)
            return {
                "id": spec.target_id,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "activity": "walking",
                "breathing_bpm": None,
                "phase0": spec.phase0,
            }

        dx = spec.x1 - spec.x0
        dy = spec.y1 - spec.y0
        seg_len = max(math.hypot(dx, dy), 1e-3)
        period_half = seg_len / spec.speed_mps
        cycle = (rel_t / period_half) % 2.0
        if cycle <= 1.0:
            u = cycle
            sign = 1.0
        else:
            u = 2.0 - cycle
            sign = -1.0
        # Smooth turnaround near endpoints using mild cosine blend on velocity
        x = spec.x0 + u * dx
        y = spec.y0 + u * dy
        vx = sign * spec.speed_mps * (dx / seg_len)
        vy = sign * spec.speed_mps * (dy / seg_len)
        return {
            "id": spec.target_id,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "activity": "walking",
            "breathing_bpm": None,
            "phase0": spec.phase0,
        }

    def get_ground_truth(self, timestamp: float | None = None) -> list[dict[str, Any]]:
        """Return ground-truth target states at ``timestamp`` (or latest emitted timestamp)."""
        ts = self._last_ts if timestamp is None else float(timestamp)
        rel_t = max(0.0, ts - self._t0)
        out: list[dict[str, Any]] = []
        for spec in self._humans:
            st = self._eval_human(spec, rel_t)
            out.append(
                {
                    "id": st["id"],
                    "x": st["x"],
                    "y": st["y"],
                    "vx": st["vx"],
                    "vy": st["vy"],
                    "activity": st["activity"],
                    "breathing_bpm": st["breathing_bpm"],
                }
            )
        return out

    def _simulate_link_rf(
        self,
        tx_pos: tuple[float, float],
        rx_pos: tuple[float, float],
        bssid: str,
        freq_hz: float,
        rel_t: float,
        targets: list[dict[str, Any]],
        peer_chord_endpoints: tuple[tuple[float, float], tuple[float, float]]
        | None = None,
    ) -> tuple[float, np.ndarray, float]:
        """Compute (rssi_dbm, raw_csi_52, peer_drop_prob) under multi-human FFZ + bistatic physics."""
        d_los = max(math.hypot(tx_pos[0] - rx_pos[0], tx_pos[1] - rx_pos[1]), 0.30)
        lam = wavelength_m(freq_hz)
        sub_freqs = subcarrier_frequencies_hz(freq_hz)
        ffz_decay_m = (
            0.5 * lam + 0.045
        )  # ~0.106 m at 2.4 GHz (human torso effective FFZ width)

        band_loss = 6.5 if freq_hz >= 4.9e9 else 0.0
        base_rssi_dbm = -40.0 - band_loss - 22.0 * math.log10(d_los)

        shadow_db = 0.0
        breath_rssi_mod_db = 0.0
        # LOS component + static clutter
        los_amp = 10.0 ** ((base_rssi_dbm + 48.0) / 20.0)
        los_amp = min(max(los_amp, 0.18), 1.45)

        h_scatter = np.zeros(52, dtype=np.complex128)

        for tgt in targets:
            pt = np.array([tgt["x"], tgt["y"]], dtype=np.float64)
            delta = float(compute_excess_path(pt, tx_pos, rx_pos))
            # Torso FFZ shadowing
            w_ffz = math.exp(-delta / ffz_decay_m)
            shadow_db += 6.2 * w_ffz

            # Chest micro-displacement for stationary breathing target
            bpm = tgt["breathing_bpm"]
            chest_disp_m = 0.0
            if bpm is not None and bpm > 0.0:
                f_resp = float(bpm) / 60.0
                breath_wave = math.sin(
                    2.0 * math.pi * f_resp * rel_t + float(tgt["phase0"])
                )
                chest_disp_m = 0.0062 * breath_wave  # 6.2 mm chest displacement
                breath_rssi_mod_db += 0.95 * math.exp(-delta / 0.22) * breath_wave
            elif tgt["activity"] == "walking":
                # Walking limb/torso micro-Doppler modulation
                spd = math.hypot(float(tgt["vx"]), float(tgt["vy"]))
                limb_wave = math.sin(
                    2.0 * math.pi * (2.2 * spd + 1.6) * rel_t + float(tgt["id"])
                )
                shadow_db += 0.55 * w_ffz * limb_wave

            # Bistatic scattered field across 52 L-LTF subcarriers
            r_tx = max(math.hypot(pt[0] - tx_pos[0], pt[1] - tx_pos[1]), 0.30)
            r_rx = max(math.hypot(pt[0] - rx_pos[0], pt[1] - rx_pos[1]), 0.30)
            bistatic_path = (r_tx + r_rx) + 2.0 * chest_disp_m
            scat_gain = (
                0.22
                * los_amp
                * math.exp(-delta / 0.24)
                * (d_los / (r_tx * r_rx + 0.35))
            )
            phase_vec = (
                (-2.0 * math.pi / SPEED_OF_LIGHT) * sub_freqs * (bistatic_path - d_los)
            )
            h_scatter += scat_gain * np.exp(1j * phase_vec)

        # Peer-chord coupling when simulating a SIFS control reply (AP <-> IoT client)
        peer_drop_prob = 0.0
        if peer_chord_endpoints is not None:
            pa, pb = peer_chord_endpoints
            peer_shadow_db = 0.0
            for tgt in targets:
                pt = np.array([tgt["x"], tgt["y"]], dtype=np.float64)
                delta_p = float(compute_excess_path(pt, pa, pb))
                w_p = math.exp(-delta_p / ffz_decay_m)
                peer_shadow_db += 6.0 * w_p
                bpm = tgt["breathing_bpm"]
                if bpm is not None and bpm > 0.0:
                    f_resp = float(bpm) / 60.0
                    bw = math.sin(2.0 * math.pi * f_resp * rel_t + float(tgt["phase0"]))
                    breath_rssi_mod_db += 0.70 * math.exp(-delta_p / 0.20) * bw
                    h_scatter += (0.09 * los_amp * math.exp(-delta_p / 0.22)) * np.exp(
                        1j
                        * (
                            1.8 * bw
                            + 0.08 * np.asarray(SUBCARRIER_INDICES, dtype=np.float64)
                        )
                    )
            if peer_shadow_db > 0.25:
                # Closed-loop SIFS link perturbation & frame loss probability on obstructed peer chord
                shadow_db += 0.85 * peer_shadow_db
                peer_drop_prob = min(0.55, 0.09 * peer_shadow_db)
                k_vec = np.asarray(SUBCARRIER_INDICES, dtype=np.float64)
                jitter_ph = float(self._rng.normal(0.0, 0.14 * peer_shadow_db))
                h_scatter += (0.075 * peer_shadow_db * los_amp) * np.exp(
                    1j * (jitter_ph + 0.11 * k_vec)
                )

        atten_lin = 10.0 ** (-shadow_db / 20.0)
        clutter = self._static_clutter.get(bssid, 0.0) * los_amp
        h_clean = (los_amp * atten_lin) + clutter + h_scatter

        # Per-packet hardware STO/PDD phase slope + random LO phase + thermal noise
        k_arr = np.asarray(SUBCARRIER_INDICES, dtype=np.float64)
        sto_slope = float(self._rng.uniform(-0.22, 0.22))
        lo_offset = float(self._rng.uniform(-math.pi, math.pi))
        awgn = (
            0.012
            * los_amp
            * (self._rng.standard_normal(52) + 1j * self._rng.standard_normal(52))
        )
        h_raw = (h_clean + awgn) * np.exp(1j * (lo_offset + sto_slope * k_arr))

        rssi_noise = float(self._rng.normal(0.0, 0.28))
        rssi_dbm = base_rssi_dbm - shadow_db + breath_rssi_mod_db + rssi_noise
        return float(rssi_dbm), h_raw.astype(np.complex64), peer_drop_prob

    def iter_events(
        self,
        duration_s: float | None = 6.0,
        realtime: bool = False,
    ) -> Iterator[PacketEvent]:
        """Yield simulated passive 802.11 ``PacketEvent`` observations."""
        dt_mean = 1.0 / self.packet_rate_hz
        dwell_s = self.dwell_ms * 1e-3
        seq_counters: dict[str, int] = {
            a.bssid: int(self._rng.integers(0, 500)) for a in self._anchors
        }
        tsf_history: dict[str, list[tuple[float, int]]] = {}

        # Group anchors by channel; also allow cross-room peer pairing across anchors
        by_channel: dict[int, list[_SimAnchor]] = {}
        for anc in self._anchors:
            by_channel.setdefault(anc.channel, []).append(anc)

        sim_rel_t = 0.0
        wall_t0 = time.monotonic()

        while duration_s is None or sim_rel_t < duration_s:
            dwell_idx = int(sim_rel_t / dwell_s)
            ch = self.channels[dwell_idx % len(self.channels)]
            freq_hz = channel_to_freq_hz(ch)

            ch_anchors = by_channel.get(ch) or self._anchors
            anc = ch_anchors[int(self._rng.integers(0, len(ch_anchors)))]

            ts = self._t0 + sim_rel_t
            self._last_ts = ts
            targets = [self._eval_human(s, sim_rel_t) for s in self._humans]

            # Decide packet exchange type:
            #  - 26%: AP Beacon (with 64-bit TSF + drift)
            #  - 38%: Directed Data/RTS followed by SIFS CTS/ACK/BlockAck from peer anchor
            #  - 36%: Standalone QOS_NULL / PROBE_RESP / DATA_PREAMBLE
            roll = float(self._rng.random())

            if roll < 0.26 and anc.role == "ap":
                rssi_dbm, csi_raw, _ = self._simulate_link_rf(
                    (anc.x, anc.y), self.rx_pos, anc.bssid, freq_hz, sim_rel_t, targets
                )
                seq_num = seq_counters[anc.bssid]
                seq_counters[anc.bssid] = (seq_num + 1) % 4096

                tsf_us = (
                    round((sim_rel_t * 1e6) * (1.0 + anc.drift_ppm * 1e-6)) + 50_000_000
                )
                hist = tsf_history.setdefault(anc.bssid, [])
                hist.append((ts, tsf_us))
                if len(hist) > 20:
                    del hist[:-20]
                drift_est: float | None = None
                if len(hist) >= 2 and (hist[-1][0] - hist[0][0]) >= 0.05:
                    dt_h = (hist[-1][0] - hist[0][0]) * 1e6
                    dt_u = float(hist[-1][1] - hist[0][1])
                    drift_est = ((dt_u / dt_h) - 1.0) * 1e6

                yield PacketEvent(
                    timestamp=ts,
                    freq_hz=freq_hz,
                    channel=ch,
                    bssid=anc.bssid,
                    receiver_addr="FF:FF:FF:FF:FF:FF",
                    frame_type="BEACON",
                    rssi_dbm=rssi_dbm,
                    noise_floor_dbm=-92.0,
                    snr_db=rssi_dbm + 92.0,
                    csi=csi_raw,
                    csi_sanitized=sanitize_csi(csi_raw),
                    cfo_hz=float(
                        anc.drift_ppm * (freq_hz / 1e6) + self._rng.normal(0.0, 180.0)
                    ),
                    tsf_us=tsf_us,
                    tsf_drift_ppm=drift_est,
                    seq_num=seq_num,
                    seq_gap=0,
                    fcs_ok=True,
                    adc_clip_ratio=0.001,
                    attributed_via_sifs=False,
                )

            elif roll < 0.70 and len(self._anchors) >= 2:
                # Pick a cross-room peer (prefer same channel first, else opposite-wall mesh peer)
                same_ch_peers = [p for p in ch_anchors if p.bssid != anc.bssid]
                pool = (
                    same_ch_peers
                    if same_ch_peers
                    else [p for p in self._anchors if p.bssid != anc.bssid]
                )
                peer = pool[int(self._rng.integers(0, len(pool)))]

                rssi_1, csi_1, peer_drop = self._simulate_link_rf(
                    (anc.x, anc.y),
                    self.rx_pos,
                    anc.bssid,
                    freq_hz,
                    sim_rel_t,
                    targets,
                )
                # Occasional missed sequence number when peer chord is heavily shadowed (retry/drop)
                gap_1 = (
                    1
                    if (peer_drop > 0.05 and float(self._rng.random()) < peer_drop)
                    else 0
                )
                seq_1 = (seq_counters[anc.bssid] + gap_1) % 4096
                seq_counters[anc.bssid] = (seq_1 + 1) % 4096
                fcs_ok_1 = float(self._rng.random()) >= 0.015

                yield PacketEvent(
                    timestamp=ts,
                    freq_hz=freq_hz,
                    channel=ch,
                    bssid=anc.bssid,
                    receiver_addr=peer.bssid,
                    frame_type="DATA_PREAMBLE" if fcs_ok_1 else "FCS_ERROR",
                    rssi_dbm=rssi_1,
                    noise_floor_dbm=-92.0,
                    snr_db=rssi_1 + 92.0,
                    csi=csi_1,
                    csi_sanitized=sanitize_csi(csi_1),
                    cfo_hz=float(
                        anc.drift_ppm * (freq_hz / 1e6) + self._rng.normal(0.0, 210.0)
                    ),
                    tsf_us=None,
                    tsf_drift_ppm=None,
                    seq_num=seq_1 if fcs_ok_1 else None,
                    seq_gap=gap_1 if fcs_ok_1 else 0,
                    fcs_ok=fcs_ok_1,
                    adc_clip_ratio=0.001,
                    attributed_via_sifs=False,
                )

                # Emit SIFS control response (10 us at 2.4 GHz, 16 us at 5 GHz + frame duration)
                if fcs_ok_1 and float(self._rng.random()) >= (0.35 * peer_drop):
                    sifs_dt = 190e-6 if freq_hz < 4.9e9 else 196e-6
                    ts_reply = ts + sifs_dt
                    rssi_2, csi_2, _ = self._simulate_link_rf(
                        (peer.x, peer.y),
                        self.rx_pos,
                        peer.bssid,
                        freq_hz,
                        sim_rel_t + sifs_dt,
                        targets,
                        peer_chord_endpoints=((anc.x, anc.y), (peer.x, peer.y)),
                    )
                    ctrl_roll = float(self._rng.random())
                    if ctrl_roll < 0.50:
                        ctrl_type = "ACK"
                    elif ctrl_roll < 0.80:
                        ctrl_type = "CTS"
                    else:
                        ctrl_type = "BLOCK_ACK"

                    yield PacketEvent(
                        timestamp=ts_reply,
                        freq_hz=freq_hz,
                        channel=ch,
                        bssid=peer.bssid,
                        receiver_addr=anc.bssid,
                        frame_type=ctrl_type,
                        rssi_dbm=rssi_2,
                        noise_floor_dbm=-92.0,
                        snr_db=rssi_2 + 92.0,
                        csi=csi_2,
                        csi_sanitized=sanitize_csi(csi_2),
                        cfo_hz=float(
                            peer.drift_ppm * (freq_hz / 1e6)
                            + self._rng.normal(0.0, 210.0)
                        ),
                        tsf_us=None,
                        tsf_drift_ppm=None,
                        seq_num=None,
                        seq_gap=0,
                        fcs_ok=True,
                        adc_clip_ratio=0.001,
                        attributed_via_sifs=(ctrl_type in {"ACK", "CTS"}),
                    )
                    sim_rel_t += sifs_dt

            else:
                rssi_dbm, csi_raw, _ = self._simulate_link_rf(
                    (anc.x, anc.y), self.rx_pos, anc.bssid, freq_hz, sim_rel_t, targets
                )
                seq_num = seq_counters[anc.bssid]
                seq_counters[anc.bssid] = (seq_num + 1) % 4096
                ft_choices = ("QOS_NULL", "PROBE_RESP", "PROBE_REQ", "DATA_PREAMBLE")
                ft = ft_choices[int(self._rng.integers(0, len(ft_choices)))]
                yield PacketEvent(
                    timestamp=ts,
                    freq_hz=freq_hz,
                    channel=ch,
                    bssid=anc.bssid,
                    receiver_addr="FF:FF:FF:FF:FF:FF"
                    if ft == "PROBE_REQ"
                    else anc.bssid,
                    frame_type=ft,
                    rssi_dbm=rssi_dbm,
                    noise_floor_dbm=-92.0,
                    snr_db=rssi_dbm + 92.0,
                    csi=csi_raw,
                    csi_sanitized=sanitize_csi(csi_raw),
                    cfo_hz=float(
                        anc.drift_ppm * (freq_hz / 1e6) + self._rng.normal(0.0, 200.0)
                    ),
                    tsf_us=None,
                    tsf_drift_ppm=None,
                    seq_num=seq_num,
                    seq_gap=0,
                    fcs_ok=True,
                    adc_clip_ratio=0.001,
                    attributed_via_sifs=False,
                )

            step_dt = float(self._rng.exponential(dt_mean))
            step_dt = min(max(step_dt, 0.25 * dt_mean), 3.0 * dt_mean)
            sim_rel_t += step_dt

            if realtime:
                target_wall = wall_t0 + sim_rel_t
                sleep_s = target_wall - time.monotonic()
                if sleep_s > 0.002:
                    time.sleep(min(sleep_s, 0.025))
