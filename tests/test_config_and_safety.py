"""Tests for configuration validation and strictly passive 0 dBm RX-only safety locks."""

from __future__ import annotations

from pathlib import Path

import pytest

from hackrf_radar.capture import HackRFIQSource, HackRFPacketCapture
from hackrf_radar.config import (
    RoomAnchor,
    RxSettings,
    SolverSettings,
    canonicalize_bssid,
    load_config,
    save_config,
)


def test_rx_settings_enforces_passive_zero_dbm_locks() -> None:
    rx = RxSettings()
    assert rx.amp_enable is False
    assert rx.bias_tee_enable is False

    with pytest.raises(ValueError, match="AMP"):
        RxSettings(amp_enable=True)

    with pytest.raises(ValueError, match="bias-tee"):
        RxSettings(bias_tee_enable=True)

    with pytest.raises(ValueError, match="AMP"):
        rx.amp_enable = True

    with pytest.raises(ValueError, match="bias-tee"):
        rx.bias_tee_enable = True

    with pytest.raises(ValueError, match="TX"):
        rx.tx_enable = True  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="tx_enable"):
        RxSettings.from_dict({"tx_enable": True})


def test_capture_drivers_reject_amp_and_bias_tee(tmp_path: Path) -> None:
    dummy_iq = tmp_path / "zeros.cs8"
    dummy_iq.write_bytes(b"\x00" * 4096)

    with pytest.raises(ValueError, match="AMP"):
        HackRFIQSource(amp_enable=True, backend="iq_file", iq_path=dummy_iq)

    with pytest.raises(ValueError, match="bias-tee"):
        HackRFIQSource(bias_tee_enable=True, backend="iq_file", iq_path=dummy_iq)

    with pytest.raises(ValueError, match="AMP"):
        HackRFPacketCapture(amp_enable=True, backend="iq_file", iq_path=dummy_iq)

    with pytest.raises(ValueError, match="bias-tee"):
        HackRFPacketCapture(bias_tee_enable=True, backend="iq_file", iq_path=dummy_iq)


def test_codebase_contains_zero_transmit_apis() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    forbidden = (
        "hackrf_start_tx",
        "soapy_hackrf_sink",
        "hackrf_transfer -t",
        "SOAPY_SDR_TX",
    )
    checked = 0
    for folder in (repo_root / "hackrf_radar", repo_root / "grc"):
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.suffix in {".py", ".grc"} and path.is_file():
                text = path.read_text(encoding="utf-8")
                checked += 1
                for token in forbidden:
                    assert token not in text, (
                        f"Forbidden TX symbol {token!r} found in {path}"
                    )
    assert checked >= 8


def test_gain_quantization_and_channel_validation() -> None:
    rx = RxSettings(lna_db=21, vga_db=19, channels=[1, 6, 11, 36, 149])
    assert rx.lna_db == 24
    assert rx.vga_db == 18
    assert rx.channels == [1, 6, 11, 36, 149]

    with pytest.raises(ValueError):
        RxSettings(channels=[])
    with pytest.raises(ValueError):
        RxSettings(channels=[999])
    with pytest.raises(ValueError):
        RxSettings(sample_rate_hz=50e6)
    with pytest.raises(ValueError):
        SolverSettings(solver_method="unknown_solver")
    with pytest.raises(ValueError):
        SolverSettings(grid_shape=(4, 4))


def test_config_roundtrip_and_example_room(tmp_path: Path) -> None:
    assert canonicalize_bssid("aa-bb-cc-1-2-3") == "AA:BB:CC:01:02:03"
    cfg = load_config()
    assert cfg.room_width == 6.0
    assert cfg.room_height == 5.0
    assert len(cfg.anchors) == 8
    assert cfg.rx.amp_enable is False

    out_json = tmp_path / "room_copy.json"
    save_config(cfg, out_json)
    reloaded = load_config(out_json)
    assert reloaded.to_dict() == cfg.to_dict()

    anc = RoomAnchor.from_dict(
        {"bssid": "aa:bb:cc:00:00:99", "pos": [1.2, 3.4], "channel": 36}
    )
    assert (anc.x, anc.y, anc.channel) == (1.2, 3.4, 36)


def test_hackrf_transfer_backend_rejects_usage_banner_and_streams_cs8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import stat

    from hackrf_radar.capture.soapy_source import _HackRFTransferBackend

    fake_bin = tmp_path / "hackrf_transfer"
    fake_bin.write_text(
        "#!/bin/sh\n"
        'echo "Usage:\n\t-h # this help\n\t[-r filename]"\n'
        'echo "hackrf_open() failed: Resource busy (-1000)" >&2\n'
        "exit 1\n",
        encoding="utf-8",
    )
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")

    with pytest.raises(RuntimeError, match="Resource busy"):
        _HackRFTransferBackend(20e6, 2437e6, 24, 20)

    fake_ok = tmp_path / "hackrf_transfer"
    fake_ok.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "assert '-a' in sys.argv and sys.argv[sys.argv.index('-a') + 1] == '0'\n"
        "assert '-p' in sys.argv and sys.argv[sys.argv.index('-p') + 1] == '0'\n"
        "sys.stdout.buffer.write(bytes([16, 240] * 8192))\n"
        "sys.stdout.buffer.flush()\n"
        "time.sleep(0.25)\n",
        encoding="utf-8",
    )
    fake_ok.chmod(fake_ok.stat().st_mode | stat.S_IXUSR)

    drv = _HackRFTransferBackend(20e6, 2437e6, 24, 20)
    try:
        iq, clip = drv.read_samples(4096)
        assert iq.shape == (4096,)
        assert clip == 0.0
        assert iq[0] == pytest.approx(0.125 - 0.125j, abs=1e-4)
    finally:
        drv.close()


def test_grc_flowgraph_passive_rx_and_sync_topology() -> None:
    grc_path = (
        Path(__file__).resolve().parent.parent / "grc" / "hackrf_wifi_passive_radar.grc"
    )
    text = grc_path.read_text(encoding="utf-8")
    assert "hackrf=0,bias=0" in text
    assert "gain0: '0'" in text
    assert "blocks_stream_to_vector_0" in text
    # Verify all 3 Schmidl-Cox inputs of ieee802_11_sync_short_0 and both inputs of sync_long_0 are wired
    assert "[blocks_delay_0, '0', ieee802_11_sync_short_0, '0']" in text
    assert "[fir_filter_xxx_0, '0', ieee802_11_sync_short_0, '1']" in text
    assert "[blocks_divide_xx_0, '0', ieee802_11_sync_short_0, '2']" in text
    assert "[ieee802_11_sync_short_0, '0', ieee802_11_sync_long_0, '0']" in text
    assert "[blocks_delay_1, '0', ieee802_11_sync_long_0, '1']" in text
    assert "[ieee802_11_sync_long_0, '0', blocks_stream_to_vector_0, '0']" in text
    assert "[blocks_stream_to_vector_0, '0', fft_vxx_0, '0']" in text
