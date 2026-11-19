"""Tests for First Fresnel Zone geometry, RTI solvers (fista_tv/nnls/sart), & multi-human tracking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hackrf_radar.capture import SyntheticRadarStream
from hackrf_radar.tomography import (
    RoomGeometry,
    TomographicRadarEngine,
    compute_bistatic_csi_phase,
    compute_bistatic_doppler,
    compute_bistatic_shell_weights,
    compute_excess_path,
    compute_ffz_weights,
    is_unicast_station_mac,
    solve_fista_tv,
    solve_nnls_regularized,
    solve_sart,
    wavelength_m,
)


def test_first_fresnel_zone_and_bistatic_geometry_invariants() -> None:
    tx = (0.5, 4.5)
    rx = (3.0, 0.5)
    freq = 2437e6
    lam = wavelength_m(freq)

    # Midpoint on LOS has excess path == 0 and strictly positive FFZ weight
    mid = np.array([(tx[0] + rx[0]) * 0.5, (tx[1] + rx[1]) * 0.5])
    assert float(compute_excess_path(mid, tx, rx)) == pytest.approx(0.0, abs=1e-9)

    # Singularity-free directly on endpoints
    pts = np.array([tx, rx, mid, (5.8, 0.4)], dtype=np.float64)
    ffz = compute_ffz_weights(pts, tx, rx, freq)
    shell = compute_bistatic_shell_weights(pts, tx, rx, freq)
    phases = compute_bistatic_csi_phase(pts, tx, rx, freq)

    assert np.all(np.isfinite(ffz)) and np.all(ffz >= 0.0)
    assert np.all(np.isfinite(shell)) and np.all(shell >= 0.0)
    assert phases.shape == (4, 52)
    # Far point outside Delta >= lambda/2 must have strict zero FFZ weight
    assert float(compute_excess_path(pts[3], tx, rx)) > 0.5 * lam
    assert ffz[3] == 0.0
    assert ffz[2] > 0.0

    # Forward-scatter baseline midpoint has zero bistatic Doppler for any velocity
    dop_mid = float(compute_bistatic_doppler(mid, (0.8, -0.5), tx, rx, freq))
    assert dop_mid == pytest.approx(0.0, abs=1e-9)

    # Broadside motion produces bounded |f_D| <= 2 * |v| / lambda
    v = (0.6, 0.8)  # |v| = 1.0 m/s
    dop_off = float(compute_bistatic_doppler((4.5, 3.5), v, tx, rx, freq))
    assert 0.5 < abs(dop_off) <= (2.0 * 1.0 / lam + 1e-6)


def test_room_geometry_auto_discovery_and_mac_filtering() -> None:
    assert not is_unicast_station_mac("FF:FF:FF:FF:FF:FF")
    assert not is_unicast_station_mac("00:00:00:00:00:00")
    assert not is_unicast_station_mac("01:00:5E:00:00:FB")
    assert not is_unicast_station_mac("33:33:00:00:00:01")
    # Non-01/33/FF IEEE 802 group/multicast addresses (I/G bit 0 == 1) must be rejected
    assert not is_unicast_station_mac("09:00:07:FF:FF:FF")
    assert not is_unicast_station_mac("03:80:C2:00:00:01")
    assert not is_unicast_station_mac("CF:00:00:00:00:00")
    # Out-of-range and signed hex segments must be rejected
    assert not is_unicast_station_mac("100:02:03:04:05:06")
    assert not is_unicast_station_mac("AA:BB:CC:DD:EE:FFF")
    assert not is_unicast_station_mac("+2:02:03:04:05:06")
    assert not is_unicast_station_mac("-2:02:03:04:05:06")
    # Locally administered & universal unicast MACs (I/G bit 0 == 0) remain valid
    assert is_unicast_station_mac("AA:BB:CC:00:00:01")
    assert is_unicast_station_mac("02:11:22:33:44:55")
    assert is_unicast_station_mac("B2:00:5E:00:53:30")

    geom = RoomGeometry(
        room_width=6.0, room_height=5.0, grid_shape=(24, 20), min_auto_packets=3
    )
    assert geom.observe_node("DE:AD:BE:EF:00:11", -52.0, 2437e6) is None
    assert geom.observe_node("DE:AD:BE:EF:00:11", -52.5, 2437e6) is None
    pos = geom.observe_node("DE:AD:BE:EF:00:11", -51.8, 2437e6)
    assert pos is not None
    assert 0.0 < pos[0] < 6.0 and 0.0 < pos[1] < 5.0
    # Must not be placed right on top of HackRF RX (3.0, 0.35)
    assert math.hypot(pos[0] - 3.0, pos[1] - 0.35) >= 0.7


@pytest.mark.parametrize(
    "solver_fn", [solve_fista_tv, solve_nnls_regularized, solve_sart]
)
def test_all_three_nonnegative_solvers_localize_intersection(solver_fn) -> None:
    geom = RoomGeometry(
        room_width=6.0,
        room_height=5.0,
        grid_shape=(30, 25),
        rx_pos=(3.0, 0.35),
        known_tx_positions={
            "AA:BB:CC:00:00:02": (3.0, 4.75),
            "AA:BB:CC:00:00:03": (5.6, 4.70),
            "AA:BB:CC:00:00:06": (0.35, 1.15),
        },
    )
    c1 = geom.get_or_create_chord("AA:BB:CC:00:00:02", "RX", 2437e6)
    c2 = geom.get_or_create_chord("AA:BB:CC:00:00:06", "AA:BB:CC:00:00:03", 2462e6)
    c3 = geom.get_or_create_chord("AA:BB:CC:00:00:03", "RX", 2462e6)
    assert c1 is not None and c2 is not None and c3 is not None

    w_mat = np.stack([c1.body_weights, c2.body_weights, c3.body_weights], axis=0)
    # Zero input produces exact zero output
    zero_map = solver_fn(w_mat, np.zeros(3), geom.grid_shape)
    assert np.all(zero_map == 0.0)

    # c1 (x=3.0) and c2 (diagonal through (3.0, 2.94)) active, c3 quiet
    img = solver_fn(w_mat, np.array([1.0, 1.0, 0.0]), geom.grid_shape)
    assert img.shape == (25, 30)
    assert np.all(img >= 0.0) and np.all(np.isfinite(img))
    iy, ix = np.unravel_index(int(np.argmax(img)), img.shape)
    peak_x = float(geom.grid_x[ix])
    peak_y = float(geom.grid_y[iy])
    assert math.hypot(peak_x - 3.0, peak_y - 2.94) < 0.65


def test_empty_room_produces_zero_false_tracks() -> None:
    sim = SyntheticRadarStream(num_humans=0, scenario="empty", seed=42)
    eng = TomographicRadarEngine(known_tx_positions=sim.get_tx_positions())
    for ev in sim.iter_events(duration_s=3.0):
        eng.ingest(ev)
    frame = eng.step()
    assert len(frame.tracks) == 0
    assert float(np.max(frame.density_map)) < 0.05


def test_walk_and_breathe_and_three_humans_localization() -> None:
    # 2 humans: 1 stationary breather (~17 BPM) + 1 walker
    sim2 = SyntheticRadarStream(num_humans=2, scenario="walk_and_breathe", seed=42)
    eng2 = TomographicRadarEngine(
        known_tx_positions=sim2.get_tx_positions(), solver_method="fista_tv"
    )
    t_last = None
    for ev in sim2.iter_events(duration_s=5.5):
        eng2.ingest(ev)
        if t_last is None or (ev.timestamp - t_last) >= 0.05:
            eng2.step(now=ev.timestamp)
            t_last = ev.timestamp
    frame2 = eng2.step()
    assert len(frame2.tracks) == 2

    stationary = [t for t in frame2.tracks if t.activity == "stationary"]
    walkers = [t for t in frame2.tracks if t.activity == "walking"]
    assert len(stationary) == 1
    assert len(walkers) == 1
    assert stationary[0].breathing_bpm is not None
    assert 14.5 <= stationary[0].breathing_bpm <= 19.5
    assert walkers[0].breathing_bpm is None

    # 3 humans: verify 3 distinct localized targets
    sim3 = SyntheticRadarStream(num_humans=3, scenario="three_humans", seed=42)
    eng3 = TomographicRadarEngine(
        known_tx_positions=sim3.get_tx_positions(), solver_method="fista_tv"
    )
    t_last3 = None
    for ev in sim3.iter_events(duration_s=5.5):
        eng3.ingest(ev)
        if t_last3 is None or (ev.timestamp - t_last3) >= 0.05:
            eng3.step(now=ev.timestamp)
            t_last3 = ev.timestamp
    frame3 = eng3.step()
    assert len(frame3.tracks) == 3


def test_two_walkers_and_sart_and_peer_chord_toggle() -> None:
    from hackrf_radar.capture import PacketEvent, sanitize_csi
    from hackrf_radar.tomography import LinkWindowExtractor

    # 1. two_walkers must localize 2 targets without ghost proliferation
    sim_tw = SyntheticRadarStream(num_humans=2, scenario="two_walkers", seed=42)
    eng_tw = TomographicRadarEngine(
        known_tx_positions=sim_tw.get_tx_positions(), solver_method="fista_tv"
    )
    t_last = None
    for ev in sim_tw.iter_events(duration_s=5.5):
        eng_tw.ingest(ev)
        if t_last is None or (ev.timestamp - t_last) >= 0.05:
            eng_tw.step(now=ev.timestamp)
            t_last = ev.timestamp
    fr_tw = eng_tw.step()
    assert len(fr_tw.tracks) == 2

    # 2. SART solver must detect BOTH stationary and walking targets in walk_and_breathe
    sim_wb = SyntheticRadarStream(num_humans=2, scenario="walk_and_breathe", seed=42)
    eng_sart = TomographicRadarEngine(
        known_tx_positions=sim_wb.get_tx_positions(), solver_method="sart"
    )
    t_last_s = None
    for ev in sim_wb.iter_events(duration_s=5.5):
        eng_sart.ingest(ev)
        if t_last_s is None or (ev.timestamp - t_last_s) >= 0.05:
            eng_sart.step(now=ev.timestamp)
            t_last_s = ev.timestamp
    fr_sart = eng_sart.step()
    assert len(fr_sart.tracks) == 2

    # 3. enable_peer_chords=False must retain SIFS ACK frames on direct TX -> RX chords
    geom = RoomGeometry(
        known_tx_positions={
            "AA:BB:CC:00:00:01": (0.5, 4.5),
            "AA:BB:CC:00:00:02": (5.5, 4.5),
        },
        enable_peer_chords=False,
    )
    ext = LinkWindowExtractor(geom)
    csi = np.ones(52, dtype=np.complex64)
    for i in range(5):
        ext.ingest(
            PacketEvent(
                timestamp=10.0 + 0.1 * i,
                freq_hz=2437e6,
                channel=6,
                bssid="AA:BB:CC:00:00:02",
                receiver_addr="AA:BB:CC:00:00:01",
                frame_type="ACK",
                rssi_dbm=-52.0,
                noise_floor_dbm=-92.0,
                snr_db=40.0,
                csi=csi,
                csi_sanitized=sanitize_csi(csi),
                cfo_hz=0.0,
                tsf_us=None,
                tsf_drift_ppm=None,
                seq_num=None,
                seq_gap=0,
                fcs_ok=True,
                adc_clip_ratio=0.0,
                attributed_via_sifs=True,
            )
        )
    sums, _ = ext.extract(now=10.5)
    assert len(sums) == 1
    assert sums[0].chord.is_peer is False
