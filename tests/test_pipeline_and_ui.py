"""End-to-end CLI pipeline, JSONL/PCAP/IQ replay, and Matplotlib/TUI/PyQtGraph UI tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hackrf_radar
from hackrf_radar.capture.gr_wifi_bridge import (
    build_80211_mac_frame,
    synthesize_80211_ofdm_burst,
    write_radiotap_pcap,
)
from hackrf_radar.cli import build_arg_parser, main, run_pipeline
from hackrf_radar.plotter import pyqtgraph_available


def test_top_level_package_lazy_exports() -> None:
    for symbol in hackrf_radar.__all__:
        val = getattr(hackrf_radar, symbol)
        assert val is not None


def test_cli_sim_record_and_jsonl_replay_with_matplotlib_and_tui(
    tmp_path: Path,
) -> None:
    snap_png = tmp_path / "sim_snapshot.png"
    rec_jsonl = tmp_path / "session.jsonl"

    rc = main(
        [
            "--mode",
            "sim",
            "--humans",
            "2",
            "--scenario",
            "walk_and_breathe",
            "--duration",
            "4.0",
            "--fast",
            "--no-gui",
            "--ui",
            "matplotlib",
            "--record",
            str(rec_jsonl),
            "--save-snapshot",
            str(snap_png),
        ]
    )
    assert rc == 0
    assert snap_png.is_file()
    png_bytes = snap_png.read_bytes()
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png_bytes) > 8000
    assert rec_jsonl.is_file() and rec_jsonl.stat().st_size > 1000

    # Replay recorded JSONL into Rich TUI and export ANSI snapshot
    ansi_txt = tmp_path / "tui_snapshot.txt"
    args_replay = build_arg_parser().parse_args(
        [
            "--mode",
            "replay",
            "--input",
            str(rec_jsonl),
            "--duration",
            "3.5",
            "--fast",
            "--no-gui",
            "--ui",
            "tui",
            "--solver",
            "nnls",
            "--save-snapshot",
            str(ansi_txt),
        ]
    )
    res_replay = run_pipeline(args_replay)
    assert res_replay.packets_processed > 50
    assert res_replay.frames_rendered >= 1
    assert ansi_txt.is_file()
    ansi_content = ansi_txt.read_text(encoding="utf-8")
    assert "0 dBm RX-ONLY" in ansi_content
    assert "2D Tomography" in ansi_content


def test_cli_pcap_and_cs8_replay_and_calibration(tmp_path: Path) -> None:
    bcn = build_80211_mac_frame(
        "BEACON", addr2="AA:BB:CC:00:00:01", seq_num=1, tsf_us=500_000
    )
    pcap_file = tmp_path / "capture.pcap"
    write_radiotap_pcap(
        pcap_file,
        [(10.0 + 0.02 * i, bcn, 2412e6, -49.0, False) for i in range(15)],
    )
    args_pcap = build_arg_parser().parse_args(
        [
            "--mode",
            "replay",
            "--input",
            str(pcap_file),
            "--fast",
            "--ui",
            "none",
            "--solver",
            "sart",
        ]
    )
    res_pcap = run_pipeline(args_pcap)
    assert res_pcap.packets_processed == 15
    assert res_pcap.last_frame is not None

    # Replay .cs8 I/Q file via CLI
    wf = np.concatenate(
        [
            synthesize_80211_ofdm_burst(
                bcn,
                rate_mbps=6,
                snr_db=28.0,
                amplitude=0.35,
                pad_before=1500,
                pad_after=1500,
            ),
            np.zeros(25_000, dtype=np.complex64),
        ]
    )
    cs8 = np.empty(wf.size * 2, dtype=np.int8)
    cs8[0::2] = np.clip(np.round(wf.real * 127.0), -127, 127).astype(np.int8)
    cs8[1::2] = np.clip(np.round(wf.imag * 127.0), -127, 127).astype(np.int8)
    cs8_file = tmp_path / "capture.cs8"
    cs8.tofile(cs8_file)

    args_iq = build_arg_parser().parse_args(
        [
            "--mode",
            "replay",
            "--input",
            str(cs8_file),
            "--channels",
            "1",
            "--fast",
            "--ui",
            "none",
        ]
    )
    res_iq = run_pipeline(args_iq)
    assert res_iq.packets_processed == 1

    # Calibration workflow in sim mode
    args_cal = build_arg_parser().parse_args(
        [
            "--mode",
            "sim",
            "--humans",
            "0",
            "--duration",
            "1.2",
            "--calibrate-s",
            "0.5",
            "--fast",
            "--ui",
            "none",
        ]
    )
    res_cal = run_pipeline(args_cal)
    assert len(res_cal.last_frame.tracks) == 0

    # Error handling edge cases
    with pytest.raises(ValueError, match="--input"):
        run_pipeline(
            build_arg_parser().parse_args(["--mode", "replay", "--ui", "none"])
        )
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            build_arg_parser().parse_args(
                [
                    "--mode",
                    "replay",
                    "--input",
                    str(tmp_path / "missing.jsonl"),
                    "--ui",
                    "none",
                ]
            )
        )


@pytest.mark.skipif(
    not pyqtgraph_available(), reason="pyqtgraph/Qt not installed in this interpreter"
)
def test_pyqtgraph_offscreen_dashboard_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    snap_qt = tmp_path / "qt_snapshot.png"
    args = build_arg_parser().parse_args(
        [
            "--mode",
            "sim",
            "--humans",
            "2",
            "--duration",
            "2.0",
            "--fast",
            "--no-gui",
            "--ui",
            "pyqtgraph",
            "--save-snapshot",
            str(snap_qt),
        ]
    )
    res = run_pipeline(args)
    assert res.ui_backend == "pyqtgraph"
    assert res.frames_rendered >= 1
    assert snap_qt.is_file() and snap_qt.stat().st_size > 4000


def test_cli_auto_discovery_disabled_and_stop_check(tmp_path: Path) -> None:
    import json

    from hackrf_radar.capture import HackRFPacketCapture, PacketEvent, sanitize_csi
    from hackrf_radar.config import RadarConfig, save_config

    cfg = RadarConfig.default_example_room()
    cfg.anchors = []
    cfg.auto_discovery.enabled = False
    cfg_path = tmp_path / "no_auto.json"
    save_config(cfg, cfg_path)

    csi = np.ones(52, dtype=np.complex64)
    ev = PacketEvent(
        timestamp=100.0,
        freq_hz=2437e6,
        channel=6,
        bssid="DE:AD:BE:EF:00:42",
        receiver_addr="FF:FF:FF:FF:FF:FF",
        frame_type="BEACON",
        rssi_dbm=-50.0,
        noise_floor_dbm=-92.0,
        snr_db=42.0,
        csi=csi,
        csi_sanitized=sanitize_csi(csi),
        cfo_hz=0.0,
        tsf_us=1_000_000,
        tsf_drift_ppm=0.0,
        seq_num=1,
        seq_gap=0,
        fcs_ok=True,
        adc_clip_ratio=0.0,
    )
    jsonl = tmp_path / "unknown.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(ev.to_dict()) for _ in range(8)) + "\n", encoding="utf-8"
    )

    res = run_pipeline(
        build_arg_parser().parse_args(
            [
                "--mode",
                "replay",
                "--config",
                str(cfg_path),
                "--input",
                str(jsonl),
                "--fast",
                "--ui",
                "none",
            ]
        )
    )
    assert len(res.last_frame.links) == 0

    # Verify stop_check terminates continuous I/Q loop immediately
    dummy_iq = tmp_path / "zeros.cs8"
    dummy_iq.write_bytes(b"\x00" * 100_000)
    with HackRFPacketCapture(
        channels=[6], backend="iq_file", iq_path=dummy_iq, loop_file=True
    ) as cap:
        evs = list(cap.iter_events(duration_s=None, stop_check=lambda: True))
    assert evs == []


def test_bssid_count_filters_unattributed_ht_preamble_and_fcs_errors(
    tmp_path: Path,
) -> None:
    import json

    from hackrf_radar.capture import PacketEvent, sanitize_csi
    from hackrf_radar.plotter import RichTuiRadarDashboard, _extract_hud_metrics

    csi = np.ones(52, dtype=np.complex64)
    cs_san = sanitize_csi(csi)

    def _mk_ev(
        ts: float, bssid: str, frame_type: str = "DATA_PREAMBLE", fcs_ok: bool = True
    ) -> PacketEvent:
        return PacketEvent(
            timestamp=ts,
            freq_hz=2437e6,
            channel=6,
            bssid=bssid,
            receiver_addr="FF:FF:FF:FF:FF:FF",
            frame_type=frame_type,
            rssi_dbm=-55.0,
            noise_floor_dbm=-92.0,
            snr_db=37.0,
            csi=csi,
            csi_sanitized=cs_san,
            cfo_hz=0.0,
            tsf_us=None,
            tsf_drift_ppm=None,
            seq_num=1,
            seq_gap=0,
            fcs_ok=fcs_ok,
            adc_clip_ratio=0.0,
        )

    events = [
        _mk_ev(1.00, "UNATTRIBUTED", "CTS", True),
        _mk_ev(1.02, "HT:OFDM:PREAMBLE", "DATA_PREAMBLE", True),
        _mk_ev(1.04, "DE:AD:BE:EF:99:99", "FCS_ERROR", False),
        _mk_ev(1.06, "FF:FF:FF:FF:FF:FF", "DATA_PREAMBLE", True),
        _mk_ev(1.08, "01:00:5E:00:00:FB", "DATA_PREAMBLE", True),
        _mk_ev(1.09, "09:00:07:FF:FF:FF", "DATA_PREAMBLE", True),
        _mk_ev(1.095, "100:02:03:04:05:06", "DATA_PREAMBLE", True),
        _mk_ev(1.10, "aa:bb:cc:00:00:01", "BEACON", True),
        _mk_ev(1.12, "AA:BB:CC:00:00:01", "BEACON", True),
        _mk_ev(1.14, "AA:BB:CC:00:00:02", "BEACON", True),
    ]
    jsonl = tmp_path / "bssid_filter.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(e.to_dict()) for e in events) + "\n", encoding="utf-8"
    )

    res = run_pipeline(
        build_arg_parser().parse_args(
            ["--mode", "replay", "--input", str(jsonl), "--fast", "--ui", "none"]
        )
    )
    assert res.packets_processed == 10
    assert res.last_frame.telemetry["bssid_count"] == 2
    hud = _extract_hud_metrics(res.last_frame)
    assert hud.bssid_count == 2
    # Corrupted fcs_ok=False MAC must not auto-discover a chord or node
    assert all(lnk.tx_bssid != "DE:AD:BE:EF:99:99" for lnk in res.last_frame.links)

    import re

    tui = RichTuiRadarDashboard(headless=True)
    try:
        ansi = tui.render_ansi(res.last_frame)
    finally:
        tui.close()
    plain_ansi = re.sub(r"\x1b\[[0-9;]*m", "", ansi)
    assert "BSSIDs: 2" in plain_ansi
    # Full 48-column heatmap and sparklines must render without ellipsis truncation
    assert "…" not in plain_ansi


def test_vital_peak_hud_reports_none_on_flat_or_empty_room_and_detects_real_breathing() -> (
    None
):
    from hackrf_radar.plotter import (
        MatplotlibRadarDashboard,
        PyQtGraphRadarDashboard,
        RichTuiRadarDashboard,
        _extract_hud_metrics,
    )
    from hackrf_radar.tomography import TomographyFrame

    # 1. Short empty-room simulation (flat psd_power_db == -42 dB) must report Vital Peak: none
    res_short = run_pipeline(
        build_arg_parser().parse_args(
            [
                "--mode",
                "sim",
                "--humans",
                "0",
                "--duration",
                "0.8",
                "--fast",
                "--ui",
                "none",
            ]
        )
    )
    hud_short = _extract_hud_metrics(res_short.last_frame)
    assert hud_short.peak_resp_bpm is None
    assert hud_short.peak_resp_hz is None

    tui = RichTuiRadarDashboard(headless=True)
    mpl = MatplotlibRadarDashboard(headless=True)
    try:
        ansi_short = tui.render_ansi(res_short.last_frame)
        mpl.update(res_short.last_frame)
        assert "Vital Peak: none" in ansi_short
        assert "9 BPM" not in ansi_short
        assert mpl._psd_peak_text.get_text() == "Vital Peak: none"
        assert "Vital Peak: none" in mpl._hud_text.get_text()
    finally:
        tui.close()
        mpl.close()

    if pyqtgraph_available():
        qt_dash = PyQtGraphRadarDashboard(headless=True)
        try:
            qt_dash.update(res_short.last_frame)
            assert "Vital Peak: none" in qt_dash._hud_label.text
        finally:
            qt_dash.close()

    # 2. Multi-second empty room (AWGN Lomb-Scargle floor, zero breathers) must also report none
    res_empty = run_pipeline(
        build_arg_parser().parse_args(
            [
                "--mode",
                "sim",
                "--humans",
                "0",
                "--duration",
                "4.0",
                "--fast",
                "--ui",
                "none",
            ]
        )
    )
    hud_empty = _extract_hud_metrics(res_empty.last_frame)
    assert hud_empty.peak_resp_bpm is None
    assert hud_empty.peak_resp_hz is None

    # 3. Standalone TomographyFrame: flat PSD and monotonic 1/f slope -> None; +20 dB peak -> ~17 BPM
    freqs = np.linspace(0.10, 5.0, 250, dtype=np.float64)
    flat_frame = TomographyFrame(
        timestamp=1.0,
        density_map=np.zeros((34, 40)),
        grid_x=np.linspace(0.1, 5.9, 40),
        grid_y=np.linspace(0.1, 4.9, 34),
        rx_pos=(3.0, 0.35),
        tracks=[],
        links=[],
        latest_csi_amp=np.zeros(52),
        latest_csi_phase=np.zeros(52),
        psd_freqs_hz=freqs,
        psd_power_db=np.full_like(freqs, -42.0),
    )
    assert _extract_hud_metrics(flat_frame).peak_resp_bpm is None

    slope_frame = TomographyFrame(
        timestamp=1.0,
        density_map=np.zeros((34, 40)),
        grid_x=np.linspace(0.1, 5.9, 40),
        grid_y=np.linspace(0.1, 4.9, 34),
        rx_pos=(3.0, 0.35),
        tracks=[],
        links=[],
        latest_csi_amp=np.zeros(52),
        latest_csi_phase=np.zeros(52),
        psd_freqs_hz=freqs,
        psd_power_db=-10.0 * np.log10(freqs),
    )
    assert _extract_hud_metrics(slope_frame).peak_resp_bpm is None

    peak_psd = -40.0 + 22.0 * np.exp(-(((freqs - 0.28) / 0.018) ** 2))
    peak_frame = TomographyFrame(
        timestamp=1.0,
        density_map=np.zeros((34, 40)),
        grid_x=np.linspace(0.1, 5.9, 40),
        grid_y=np.linspace(0.1, 4.9, 34),
        rx_pos=(3.0, 0.35),
        tracks=[],
        links=[],
        latest_csi_amp=np.zeros(52),
        latest_csi_phase=np.zeros(52),
        psd_freqs_hz=freqs,
        psd_power_db=peak_psd,
    )
    hud_peak = _extract_hud_metrics(peak_frame)
    assert hud_peak.peak_resp_bpm is not None
    assert 16.0 <= hud_peak.peak_resp_bpm <= 17.6

    # 4. Walking-only simulation (single_walk default 6.0s) must report Vital Peak: none
    res_walk = run_pipeline(
        build_arg_parser().parse_args(
            [
                "--mode",
                "sim",
                "--humans",
                "1",
                "--scenario",
                "single_walk",
                "--duration",
                "6.0",
                "--seed",
                "42",
                "--fast",
                "--ui",
                "none",
            ]
        )
    )
    assert _extract_hud_metrics(res_walk.last_frame).peak_resp_bpm is None
    assert all(lnk.breathing_bpm is None for lnk in res_walk.last_frame.links)

    # 5. Noisy 1/f slope on actual 512-bin engine PSD grid must never trigger false Vital Peak
    f_eng = np.concatenate(
        [
            np.linspace(0.10, 1.20, 320, dtype=np.float64),
            np.linspace(1.25, 25.0, 192, dtype=np.float64),
        ]
    )
    for seed_idx in range(25):
        rng_1f = np.random.default_rng(seed_idx)
        noisy_1f = -15.0 * np.log10(f_eng) + rng_1f.normal(0.0, 0.35, f_eng.size)
        noisy_frame = TomographyFrame(
            timestamp=1.0,
            density_map=np.zeros((34, 40)),
            grid_x=np.linspace(0.1, 5.9, 40),
            grid_y=np.linspace(0.1, 4.9, 34),
            rx_pos=(3.0, 0.35),
            tracks=[],
            links=[],
            latest_csi_amp=np.zeros(52),
            latest_csi_phase=np.zeros(52),
            psd_freqs_hz=f_eng,
            psd_power_db=noisy_1f,
        )
        assert _extract_hud_metrics(noisy_frame).peak_resp_bpm is None

    # 6. Stationary breathing simulation still reports accurate ~16-17 BPM Vital Peak
    res_breathe = run_pipeline(
        build_arg_parser().parse_args(
            [
                "--mode",
                "sim",
                "--humans",
                "1",
                "--scenario",
                "single_breathe",
                "--duration",
                "5.0",
                "--fast",
                "--ui",
                "none",
            ]
        )
    )
    hud_breathe = _extract_hud_metrics(res_breathe.last_frame)
    assert hud_breathe.peak_resp_bpm is not None
    assert 14.5 <= hud_breathe.peak_resp_bpm <= 19.5


def test_matplotlib_gui_fallback_and_tclerror_window_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warnings

    import matplotlib.pyplot as plt

    from hackrf_radar.plotter import _GUI_BACKEND_ERRORS, MatplotlibRadarDashboard
    from hackrf_radar.tomography import TomographyFrame

    # Broken/unreachable DISPLAY where pyplot falls back to Agg must not warn or claim _interactive=True
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.setattr(plt, "get_backend", lambda: "agg")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dash = MatplotlibRadarDashboard(headless=False)
    try:
        assert dash._interactive is False
        assert not any(
            "FigureCanvasAgg is non-interactive" in str(w.message) for w in caught
        )

        # Simulate interactive window abrupt destruction raising TclError (or GUI backend error)
        exc_cls = _GUI_BACKEND_ERRORS[-1]
        dash._interactive = True

        def _broken_blit(_bbox: object) -> None:
            raise exc_cls("window destroyed")

        monkeypatch.setattr(dash.canvas, "blit", _broken_blit)
        frame = TomographyFrame(
            timestamp=1.0,
            density_map=np.zeros((34, 40)),
            grid_x=np.linspace(0.1, 5.9, 40),
            grid_y=np.linspace(0.1, 4.9, 34),
            rx_pos=(3.0, 0.35),
            tracks=[],
            links=[],
            latest_csi_amp=np.zeros(52),
            latest_csi_phase=np.zeros(52),
            psd_freqs_hz=np.linspace(0.1, 25.0, 64),
            psd_power_db=np.full(64, -42.0),
        )
        assert dash.update(frame) == 0.0
        assert dash._closed is True
    finally:
        dash.close()
