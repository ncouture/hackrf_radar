"""Sliding-window link perturbation, 52-subcarrier CSI turbulence, & vital-sign PSD extraction."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..capture.packet_event import PacketEvent
from .geometry import LinkChord, RoomGeometry, canonicalize_mac, is_unicast_station_mac

PSD_SAMPLE_RATE_HZ: float = 50.0
PSD_NFFT: int = 4096
RESP_MIN_HZ: float = 0.18  # 10.8 BPM
RESP_MAX_HZ: float = 0.45  # 27.0 BPM


@dataclass(slots=True)
class _PacketObs:
    timestamp: float
    rssi_dbm: float
    csi_san: np.ndarray
    fcs_ok: bool
    seq_gap: int


@dataclass(slots=True)
class LinkFeatureSummary:
    """Extracted RTI & vital-sign features for a single geometric chord at step ``now``."""

    chord: LinkChord
    packet_count: int
    rssi_delta_db: float
    csi_turbulence: float
    multipath_spread: float
    perturbation_score: float
    breathing_hz: float | None
    breathing_bpm: float | None
    breathing_snr_db: float
    fcs_error_rate: float
    seq_gap_rate: float
    tsf_drift_ppm: float | None
    psd_linear: np.ndarray | None = None
    csi_amp: np.ndarray | None = None
    csi_phase: np.ndarray | None = None
    baseline_csi_amp: np.ndarray | None = None


class _SingleStreamStats:
    """Bounded time-indexed buffer for one packet stream (direct or SIFS peer)."""

    def __init__(
        self, max_history_s: float = 12.5, maxlen: int = 1600, freq_hz: float = 2437e6
    ) -> None:
        self.max_history_s = float(max_history_s)
        self.obs: deque[_PacketObs] = deque(maxlen=maxlen)
        self.baseline_rssi: float | None = None
        self.baseline_csi: np.ndarray | None = None
        self.last_freq_hz: float = float(freq_hz)

    def append(self, item: _PacketObs, freq_hz: float | None = None) -> None:
        if freq_hz is not None and freq_hz > 0.0:
            self.last_freq_hz = float(freq_hz)
        self.obs.append(item)
        cutoff = item.timestamp - self.max_history_s
        while len(self.obs) > 2 and self.obs[0].timestamp < cutoff:
            self.obs.popleft()

    def calibrate_from_current(self) -> None:
        valid = [o for o in self.obs if o.fcs_ok]
        if not valid:
            return
        self.baseline_rssi = float(np.median([o.rssi_dbm for o in valid]))
        csi_mat = np.stack([o.csi_san for o in valid], axis=0)
        self.baseline_csi = np.mean(csi_mat, axis=0).astype(np.complex64)


class LinkWindowExtractor:
    """Extracts per-chord RTI attenuation, CSI turbulence, SIFS peer drops, & 0.1-25 Hz PSD."""

    def __init__(
        self,
        geometry: RoomGeometry,
        window_s: float = 1.5,
        vital_window_s: float = 10.0,
    ) -> None:
        self.geometry = geometry
        self.window_s = max(0.2, float(window_s))
        self.vital_window_s = max(3.0, float(vital_window_s))

        self._direct_streams: dict[str, _SingleStreamStats] = {}
        self._peer_streams: dict[str, _SingleStreamStats] = {}
        self._peer_seq_gaps: dict[str, deque[tuple[float, int]]] = {}
        self._bssid_tsf_ppm: dict[str, float] = {}
        self._observed_bssids: set[str] = set()
        self._all_timestamps: deque[float] = deque(maxlen=4000)

        self.latest_csi_amp: np.ndarray = np.zeros(52, dtype=np.float64)
        self.latest_csi_phase: np.ndarray = np.zeros(52, dtype=np.float64)
        self.latest_timestamp: float = 0.0
        self._calibrated_explicitly: bool = False

        f_fine = np.linspace(0.10, 1.20, 320, dtype=np.float64)
        f_coarse = np.linspace(1.25, 25.0, 192, dtype=np.float64)
        self.psd_freqs_hz: np.ndarray = np.concatenate([f_fine, f_coarse])
        self._psd_omegas: np.ndarray = 2.0 * np.pi * self.psd_freqs_hz

    def set_window_s(self, window_s: float) -> float:
        """Update the sliding feature extraction window duration in seconds."""
        val = float(window_s)
        if not math.isfinite(val) or not (0.2 <= val <= 10.0):
            raise ValueError(f"window_s must be in [0.2, 10.0], got {window_s}")
        self.window_s = val
        return self.window_s

    @property
    def observed_bssids(self) -> set[str]:
        """Return canonical unicast BSSIDs observed on FCS-verified frames."""
        return set(self._observed_bssids)

    def ingest(self, ev: PacketEvent) -> None:
        """Ingest a single cleartext ``PacketEvent`` into direct and SIFS-peer link buffers."""
        ts = float(ev.timestamp)
        self.latest_timestamp = max(self.latest_timestamp, ts)
        self._all_timestamps.append(ts)
        cutoff_rate = ts - max(self.window_s, 2.0)
        while len(self._all_timestamps) > 2 and self._all_timestamps[0] < cutoff_rate:
            self._all_timestamps.popleft()

        if ev.fcs_ok:
            self.latest_csi_amp = np.abs(ev.csi_sanitized).astype(np.float64)
            self.latest_csi_phase = np.angle(ev.csi_sanitized).astype(np.float64)

        bssid = canonicalize_mac(ev.bssid)
        if not is_unicast_station_mac(bssid):
            return

        if ev.fcs_ok:
            self._observed_bssids.add(bssid)
            if ev.tsf_drift_ppm is not None and math.isfinite(float(ev.tsf_drift_ppm)):
                self._bssid_tsf_ppm[bssid] = float(ev.tsf_drift_ppm)
            tx_pos = self.geometry.observe_node(
                bssid,
                rssi_dbm=ev.rssi_dbm,
                freq_hz=ev.freq_hz,
            )
        else:
            tx_pos = self.geometry.get_node_position(bssid)
        rx_peer = canonicalize_mac(ev.receiver_addr) if ev.receiver_addr else None
        is_peer_node = (
            self.geometry.enable_peer_chords
            and rx_peer is not None
            and rx_peer != bssid
            and is_unicast_station_mac(rx_peer)
            and self.geometry.get_node_position(rx_peer) is not None
        )

        is_sifs_reply = bool(
            is_peer_node
            and (ev.attributed_via_sifs or ev.frame_type in {"ACK", "CTS", "BLOCK_ACK"})
        )

        obs = _PacketObs(
            timestamp=ts,
            rssi_dbm=float(ev.rssi_dbm),
            csi_san=ev.csi_sanitized,
            fcs_ok=bool(ev.fcs_ok),
            seq_gap=max(0, int(ev.seq_gap)),
        )

        # Record standalone frames on direct chord TX -> RX (and all frames when peer chords disabled)
        if tx_pos is not None:
            direct_chord = self.geometry.get_or_create_chord(bssid, "RX", ev.freq_hz)
            if direct_chord is not None and not is_sifs_reply:
                d_stream = self._direct_streams.get(direct_chord.link_id)
                if d_stream is None:
                    d_stream = _SingleStreamStats(
                        max_history_s=self.vital_window_s + 2.0, freq_hz=ev.freq_hz
                    )
                    self._direct_streams[direct_chord.link_id] = d_stream
                d_stream.append(obs, freq_hz=ev.freq_hz)

        # Record directed peer traffic & SIFS replies on directed link (bssid -> rx_peer)
        if is_peer_node and rx_peer is not None and self.geometry.enable_peer_chords:
            peer_chord = self.geometry.get_or_create_chord(bssid, rx_peer, ev.freq_hz)
            if peer_chord is not None:
                if is_sifs_reply:
                    p_stream = self._peer_streams.get(peer_chord.link_id)
                    if p_stream is None:
                        p_stream = _SingleStreamStats(
                            max_history_s=self.vital_window_s + 2.0, freq_hz=ev.freq_hz
                        )
                        self._peer_streams[peer_chord.link_id] = p_stream
                    p_stream.append(obs, freq_hz=ev.freq_hz)
                else:
                    # Sequence gaps on DATA from bssid -> rx_peer reflect obstruction on bssid<->rx_peer;
                    # attach to reverse SIFS chord (rx_peer -> bssid) as well as forward chord
                    for cid in (
                        peer_chord.link_id,
                        self.geometry.make_link_id(rx_peer, bssid, ev.freq_hz),
                    ):
                        gap_q = self._peer_seq_gaps.setdefault(cid, deque(maxlen=400))
                        gap_q.append((ts, max(0, int(ev.seq_gap))))

    def calibrate_baseline(self) -> None:
        """Lock current observed median RSSI & mean sanitized CSI as empty-room baseline."""
        for st in self._direct_streams.values():
            st.calibrate_from_current()
        for st in self._peer_streams.values():
            st.calibrate_from_current()
        self._calibrated_explicitly = True

    def packet_rate_hz(self, now: float) -> float:
        if len(self._all_timestamps) < 2:
            return 0.0
        t_min = max(self._all_timestamps[0], now - self.window_s)
        recent = sum(1 for t in self._all_timestamps if t >= t_min)
        span = max(now - t_min, 0.08)
        return float(recent / span)

    def mean_tsf_ppm(self) -> float | None:
        if not self._bssid_tsf_ppm:
            return None
        vals = [v for v in self._bssid_tsf_ppm.values() if math.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    def _fit_uncalibrated_path_loss(
        self, direct_medians: list[tuple[LinkChord, float]]
    ) -> Mapping[str, float]:
        """Estimate expected unobstructed RSSI per direct chord via robust upper-envelope fit."""
        if not direct_medians:
            return {}
        dists = np.array(
            [max(c.distance_m, 0.35) for c, _ in direct_medians], dtype=np.float64
        )
        rssis = np.array([r for _, r in direct_medians], dtype=np.float64)
        band_adj = np.array(
            [6.5 if c.freq_hz >= 4.9e9 else 0.0 for c, _ in direct_medians],
            dtype=np.float64,
        )
        # Normalize to 2.4 GHz 1m equivalent RSSI assuming indoor n ~ 2.2
        p0_equiv = rssis + band_adj + 22.0 * np.log10(dists)
        # Upper 90th percentile represents unobstructed LOS links even with 3+ humans in room
        p0_ref = (
            float(np.percentile(p0_equiv, 90))
            if p0_equiv.size >= 3
            else float(np.max(p0_equiv))
        )
        expected: dict[str, float] = {}
        for idx, (chord, _) in enumerate(direct_medians):
            expected[chord.link_id] = float(
                p0_ref - band_adj[idx] - 22.0 * math.log10(dists[idx])
            )
        return expected

    def _compute_vital_psd(
        self, obs_list: list[_PacketObs], now: float
    ) -> tuple[np.ndarray | None, float | None, float | None, float]:
        """Compute alias-free Lomb-Scargle 0.1-25 Hz PSD & respiratory rate (12-24 BPM)."""
        valid = [
            o
            for o in obs_list
            if o.fcs_ok and o.timestamp >= (now - self.vital_window_s)
        ]
        if len(valid) < 18:
            return None, None, None, 0.0

        t_arr = np.array([o.timestamp for o in valid], dtype=np.float64)
        span = float(t_arr[-1] - t_arr[0])
        if span < 2.4:
            return None, None, None, 0.0

        uniq_mask = np.concatenate([[True], np.diff(t_arr) > 1e-6])
        if int(np.sum(uniq_mask)) < 16:
            return None, None, None, 0.0
        sel_idx = np.flatnonzero(uniq_mask)
        t_rel = t_arr[sel_idx] - t_arr[sel_idx[0]]
        eff_span = float(t_rel[-1])

        rssi_arr = np.array([valid[i].rssi_dbm for i in sel_idx], dtype=np.float64)
        csi_mat = np.stack([valid[i].csi_san for i in sel_idx], axis=0).astype(
            np.complex128
        )

        # Reject macroscopic walking chord-crossing shadow dips (>2.8 dB swing or half-window step)
        # from being misclassified as stationary chest breathing
        mid_n = rssi_arr.size // 2
        rssi_span_90 = float(np.percentile(rssi_arr, 95) - np.percentile(rssi_arr, 5))
        half_step = abs(
            float(np.median(rssi_arr[:mid_n])) - float(np.median(rssi_arr[mid_n:]))
        )
        is_macro_transit = (rssi_span_90 > 2.85) or (half_step > 1.25)

        # True timestamp-linear detrending (immune to channel-hopping dwell gaps)
        t_c = t_rel - float(np.mean(t_rel))
        t_var = max(float(np.dot(t_c, t_c)), 1e-12)

        def _detrend_time(vec: np.ndarray) -> np.ndarray:
            v0 = vec - float(np.mean(vec))
            slope = float(np.dot(t_c, v0)) / t_var
            return v0 - slope * t_c

        rssi_dt = _detrend_time(rssi_arr)

        # Quadrature CSI phase perturbation Im(H(t) * conj(H_mean)) / |H_mean|^2 (zero blind spots)
        csi_mean = np.mean(csi_mat, axis=0)
        denom_h = np.maximum(np.abs(csi_mean) ** 2, 1e-8)
        phase_pert = np.mean(
            np.imag(csi_mat * np.conj(csi_mean)[None, :]) / denom_h[None, :], axis=1
        )
        phase_dt = _detrend_time(phase_pert) * 8.0

        var_rssi = float(np.var(rssi_dt))
        var_phase = float(np.var(phase_dt))
        if (var_rssi + var_phase) < 1e-8:
            return None, None, None, 0.0

        ls_rssi = signal.lombscargle(t_rel, rssi_dt, self._psd_omegas, normalize=True)
        ls_phase = (
            signal.lombscargle(t_rel, phase_dt, self._psd_omegas, normalize=True)
            if var_phase > 1e-8
            else np.zeros_like(ls_rssi)
        )
        # Variance-weighted power spectrum + normalized coherence in [0, 1]
        psd_band = var_rssi * ls_rssi + var_phase * ls_phase
        coherence = (var_rssi * ls_rssi + var_phase * ls_phase) / (var_rssi + var_phase)

        f_band = self.psd_freqs_hz
        resp_sel = (f_band >= RESP_MIN_HZ) & (f_band <= RESP_MAX_HZ)
        resp_indices = np.flatnonzero(resp_sel)
        if resp_indices.size < 3:
            return psd_band, None, None, 0.0

        # Select strongest strictly interior local maximum inside [RESP_MIN_HZ, RESP_MAX_HZ]
        # so monotonic low-frequency walking transit skirts never clamp onto band edges.
        sub_p = psd_band[resp_indices]
        cand_local = [
            k
            for k in range(1, sub_p.size - 1)
            if sub_p[k] >= sub_p[k - 1]
            and sub_p[k] >= sub_p[k + 1]
            and (sub_p[k] > sub_p[k - 1] or sub_p[k] > sub_p[k + 1])
        ]
        if not cand_local:
            return psd_band, None, None, 0.0

        local_best = max(cand_local, key=lambda k: float(sub_p[k]))
        pk_idx = int(resp_indices[local_best])
        f_pk = float(f_band[pk_idx])
        pk_pow = max(float(psd_band[pk_idx]), 1e-16)
        pk_coh = float(coherence[pk_idx])

        # Compare peak against same-passband [0.10, 2.20] Hz sidebands (outside +-0.06 Hz of f_pk)
        side_sel = (
            (f_band >= 0.10) & (f_band <= 2.20) & (np.abs(f_band - f_pk) >= 0.065)
        )
        bg_med = max(float(np.median(psd_band[side_sel])), 1e-16)
        bg_p85 = max(float(np.percentile(psd_band[side_sel], 85)), 1e-16)

        snr_med_db = 10.0 * math.log10(pk_pow / bg_med)
        snr_p85_db = 10.0 * math.log10(pk_pow / bg_p85)
        low_skirt = (
            float(np.max(psd_band[f_band < RESP_MIN_HZ]))
            if np.any(f_band < RESP_MIN_HZ)
            else 0.0
        )

        # Require genuine periodic coherence (pk_coh >= 0.30), physical variance above AWGN floor,
        # absence of macroscopic walking shadow transit steps, and dominance over <0.18 Hz skirts
        if (
            is_macro_transit
            or pk_coh < 0.30
            or snr_med_db < 9.0
            or snr_p85_db < 4.5
            or pk_pow < 0.032
            or low_skirt > 1.15 * pk_pow
        ):
            return psd_band, None, None, float(max(0.0, snr_med_db))

        # 3-point parabolic log-power interpolation on fine 0.0034 Hz grid
        f_est = f_pk
        if 0 < pk_idx < 319:
            y0 = math.log(max(float(psd_band[pk_idx - 1]), 1e-20))
            y1 = math.log(max(float(psd_band[pk_idx]), 1e-20))
            y2 = math.log(max(float(psd_band[pk_idx + 1]), 1e-20))
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta_bin = max(-0.75, min(0.75, 0.5 * (y0 - y2) / denom))
                df = float(f_band[1] - f_band[0])
                f_est = f_pk + delta_bin * df

        # Require >= 1.25 observed respiratory cycles so a 1-cycle detrended transit dip at 1/T is rejected
        if eff_span * f_est < 1.25:
            return psd_band, None, None, float(max(0.0, snr_med_db))

        bpm_est = f_est * 60.0
        if not (10.5 <= bpm_est <= 27.5):
            return psd_band, None, None, float(snr_med_db)
        return psd_band, f_est, bpm_est, float(snr_med_db)

    def extract(
        self, now: float | None = None
    ) -> tuple[list[LinkFeatureSummary], np.ndarray]:
        """Compute per-link ``LinkFeatureSummary`` list and aggregated ``psd_power_db``."""
        step_t = self.latest_timestamp if now is None else float(now)
        win_lo = step_t - self.window_s
        fast_lo = step_t - min(self.window_s, 0.55)

        # Pass 1: gather active direct chords and instantaneous (0.55s) + full-window medians
        direct_active: list[
            tuple[
                LinkChord,
                _SingleStreamStats,
                list[_PacketObs],
                list[_PacketObs],
                list[_PacketObs],
                float,
            ]
        ] = []
        direct_medians: list[tuple[LinkChord, float]] = []
        direct_rssi_by_bssid: dict[str, float] = {}
        direct_turb_by_bssid: dict[str, float] = {}

        for link_id, st in list(self._direct_streams.items()):
            win_obs = [o for o in st.obs if o.timestamp >= win_lo]
            valid_win = [o for o in win_obs if o.fcs_ok]
            if not valid_win:
                continue
            tx_mac = link_id.split("->")[0]
            chord = self.geometry.get_or_create_chord(tx_mac, "RX", st.last_freq_hz)
            if chord is None:
                continue
            fast_valid = [o for o in valid_win if o.timestamp >= fast_lo]
            inst_obs = fast_valid if len(fast_valid) >= 2 else valid_win
            inst_rssi = float(np.median([o.rssi_dbm for o in inst_obs]))
            full_rssi = float(np.percentile([o.rssi_dbm for o in valid_win], 75))
            direct_active.append((chord, st, win_obs, valid_win, inst_obs, inst_rssi))
            direct_medians.append((chord, full_rssi))
            direct_rssi_by_bssid[chord.tx_bssid] = inst_rssi

        expected_uncal = (
            {}
            if self._calibrated_explicitly
            else self._fit_uncalibrated_path_loss(direct_medians)
        )

        summaries: list[LinkFeatureSummary] = []
        psd_accum = np.zeros_like(self.psd_freqs_hz, dtype=np.float64)
        psd_weight_sum = 0.0

        # Pass 2: compute features for direct TX -> RX chords
        for chord, st, win_obs, valid_win, inst_obs, inst_rssi in direct_active:
            dyn_obs = inst_obs if len(inst_obs) >= 3 else valid_win
            rssis = np.array([o.rssi_dbm for o in dyn_obs], dtype=np.float64)
            rssi_iqr = float(np.percentile(rssis, 85) - np.percentile(rssis, 15))

            if st.baseline_rssi is not None:
                rssi_drop = max(0.0, float(st.baseline_rssi) - inst_rssi)
            else:
                exp_r = expected_uncal.get(chord.link_id, inst_rssi)
                rssi_drop = max(0.0, float(exp_r) - inst_rssi - 0.55)

            csi_mat = np.stack([o.csi_san for o in dyn_obs], axis=0).astype(
                np.complex128
            )
            csi_mean = np.mean(csi_mat, axis=0)
            mean_norm = max(float(np.linalg.norm(csi_mean)), 1e-6)
            # Instantaneous complex CSI turbulence normalized by link amplitude
            csi_std = float(
                np.sqrt(np.mean(np.abs(csi_mat - csi_mean[None, :]) ** 2))
            ) / (mean_norm / math.sqrt(52.0))
            csi_dev_base = 0.0
            if st.baseline_csi is not None:
                csi_dev_base = (
                    float(np.linalg.norm(csi_mean - st.baseline_csi)) / mean_norm
                )

            # Subcarrier frequency-selectivity (excess-path multipath spread)
            amp_norm = np.abs(csi_mean) / max(float(np.mean(np.abs(csi_mean))), 1e-6)
            multipath_spread = float(np.std(np.diff(amp_norm)))

            direct_turb_by_bssid[chord.tx_bssid] = csi_std

            psd_lin, b_hz, b_bpm, b_snr = self._compute_vital_psd(list(st.obs), step_t)
            if psd_lin is not None:
                w_psd = (
                    1.0
                    + (2.5 if b_bpm is not None else 0.0)
                    + min(rssi_drop, 4.0) * 0.4
                )
                psd_accum += w_psd * psd_lin
                psd_weight_sum += w_psd

            # Noise-gated non-negative perturbation score in [0, 1.5]
            drop_term = max(0.0, rssi_drop - 0.45) / 4.8
            dyn_term = (
                max(0.0, csi_std - 0.045) * 2.4 + max(0.0, rssi_iqr - 0.85) * 0.16
            )
            base_csi_term = max(0.0, csi_dev_base - 0.06) * 1.4
            vital_term = 0.22 if (b_bpm is not None and b_snr >= 10.0) else 0.0
            score = min(1.5, drop_term + dyn_term + base_csi_term + vital_term)
            if score < 0.07:
                score = 0.0

            n_tot = len(win_obs)
            fcs_err_rate = float(sum(1 for o in win_obs if not o.fcs_ok)) / max(
                n_tot, 1
            )
            seq_gap_rate = float(sum(o.seq_gap for o in valid_win)) / max(
                len(valid_win), 1
            )

            summaries.append(
                LinkFeatureSummary(
                    chord=chord,
                    packet_count=n_tot,
                    rssi_delta_db=float(rssi_drop),
                    csi_turbulence=float(csi_std),
                    multipath_spread=multipath_spread,
                    perturbation_score=float(score),
                    breathing_hz=b_hz,
                    breathing_bpm=b_bpm,
                    breathing_snr_db=b_snr,
                    fcs_error_rate=fcs_err_rate,
                    seq_gap_rate=seq_gap_rate,
                    tsf_drift_ppm=self._bssid_tsf_ppm.get(chord.tx_bssid),
                    psd_linear=psd_lin,
                    csi_amp=np.abs(csi_mean).astype(np.float64),
                    csi_phase=np.angle(csi_mean).astype(np.float64),
                    baseline_csi_amp=(
                        np.abs(st.baseline_csi).astype(np.float64)
                        if st.baseline_csi is not None
                        else None
                    ),
                )
            )

        # Pass 3: compute differential SIFS peer-chord features (Peer_A <-> Peer_B)
        for link_id, pst in list(self._peer_streams.items()):
            win_obs = [o for o in pst.obs if o.timestamp >= win_lo]
            valid_win = [o for o in win_obs if o.fcs_ok]
            if len(valid_win) < 2:
                continue

            parts = link_id.split("@")[0].split("->")
            if len(parts) != 2:
                continue
            tx_a, rx_b = parts[0], parts[1]
            chord = self.geometry.get_or_create_chord(tx_a, rx_b, pst.last_freq_hz)
            if chord is None:
                continue

            # Every SIFS reply in pst.obs was transmitted by tx_a in response to rx_b,
            # so direct_rssi_by_bssid[tx_a] is its exact standalone radial reference!
            fast_peer = [o for o in valid_win if o.timestamp >= fast_lo]
            peer_inst = fast_peer if len(fast_peer) >= 2 else valid_win
            sifs_med_rssi = float(np.median([o.rssi_dbm for o in peer_inst]))
            tx_direct_drop = next(
                (
                    s.rssi_delta_db
                    for s in summaries
                    if not s.chord.is_peer and s.chord.tx_bssid == tx_a
                ),
                0.0,
            )
            if pst.baseline_rssi is not None:
                raw_peer_drop = max(0.0, float(pst.baseline_rssi) - sifs_med_rssi)
                peer_drop_db = max(0.0, raw_peer_drop - tx_direct_drop)
            elif tx_a in direct_rssi_by_bssid:
                peer_drop_db = max(
                    0.0, direct_rssi_by_bssid[tx_a] - sifs_med_rssi - 0.50
                )
            else:
                hist_valid = [o.rssi_dbm for o in pst.obs if o.fcs_ok]
                sifs_ceil = (
                    float(np.percentile(hist_valid, 80))
                    if len(hist_valid) >= 6
                    else sifs_med_rssi
                )
                peer_drop_db = max(0.0, sifs_ceil - sifs_med_rssi - 0.55)

            peer_dyn = fast_peer if len(fast_peer) >= 3 else valid_win
            csi_mat = np.stack([o.csi_san for o in peer_dyn], axis=0).astype(
                np.complex128
            )
            csi_mean = np.mean(csi_mat, axis=0)
            mean_norm = max(float(np.linalg.norm(csi_mean)), 1e-6)
            sifs_turb = float(
                np.sqrt(np.mean(np.abs(csi_mat - csi_mean[None, :]) ** 2))
            ) / (mean_norm / math.sqrt(52.0))
            ref_turb = direct_turb_by_bssid.get(tx_a, 0.035)
            excess_turb = max(0.0, sifs_turb - ref_turb)

            gap_q = self._peer_seq_gaps.get(chord.link_id)
            fast_gaps = [g for t_g, g in (gap_q or ()) if t_g >= fast_lo]
            recent_gaps = (
                fast_gaps
                if len(fast_gaps) >= 2
                else [g for t_g, g in (gap_q or ()) if t_g >= win_lo]
            )
            peer_gap_rate = (
                float(sum(recent_gaps)) / max(len(recent_gaps), 1)
                if recent_gaps
                else 0.0
            )

            psd_lin, b_hz, b_bpm, b_snr = self._compute_vital_psd(list(pst.obs), step_t)
            if psd_lin is not None and b_bpm is not None:
                psd_accum += 1.2 * psd_lin
                psd_weight_sum += 1.2

            p_score = min(
                1.5,
                max(0.0, peer_drop_db - 0.35) / 4.2
                + max(0.0, excess_turb - 0.035) * 2.6
                + min(0.55, peer_gap_rate * 2.2),
            )
            if p_score < 0.08:
                p_score = 0.0

            summaries.append(
                LinkFeatureSummary(
                    chord=chord,
                    packet_count=len(win_obs),
                    rssi_delta_db=float(peer_drop_db),
                    csi_turbulence=float(excess_turb),
                    multipath_spread=0.0,
                    perturbation_score=float(p_score),
                    breathing_hz=b_hz,
                    breathing_bpm=b_bpm,
                    breathing_snr_db=b_snr,
                    fcs_error_rate=0.0,
                    seq_gap_rate=peer_gap_rate,
                    tsf_drift_ppm=None,
                    psd_linear=psd_lin,
                    csi_amp=np.abs(csi_mean).astype(np.float64),
                    csi_phase=np.angle(csi_mean).astype(np.float64),
                    baseline_csi_amp=(
                        np.abs(pst.baseline_csi).astype(np.float64)
                        if pst.baseline_csi is not None
                        else None
                    ),
                )
            )

        if psd_weight_sum > 0.0:
            avg_psd = psd_accum / psd_weight_sum
            peak_ref = max(float(np.max(avg_psd)), 1e-6)
            psd_db = 10.0 * np.log10(np.maximum(avg_psd / peak_ref, 1e-5))
        else:
            psd_db = np.full_like(self.psd_freqs_hz, -42.0, dtype=np.float64)

        return summaries, psd_db
