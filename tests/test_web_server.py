"""Tests for realtime tomography JSON serialization, SSE hub, and stdlib HTTP server."""

from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pytest

from hackrf_radar.capture import SyntheticRadarStream
from hackrf_radar.plotter import TrackHistoryBuffer
from hackrf_radar.tomography import (
    LinkDiagnostic,
    TomographicRadarEngine,
    TomographyFrame,
    TrackedHuman,
)
from hackrf_radar.web import (
    RealtimeWebServer,
    TomographyStateHub,
    WebRadarDashboard,
    serialize_frame,
    serialize_frame_json,
)


def _http_request(
    url: str,
    method: str = "GET",
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    parsed = urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5.0)
    try:
        conn.request(method, parsed.path or "/", body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, resp_headers, data
    finally:
        conn.close()


def test_serialize_frame_real_engine_and_nan_sanitization() -> None:
    sim = SyntheticRadarStream(
        num_humans=2, scenario="walk_and_breathe", seed=42, packet_rate_hz=200.0
    )
    eng = TomographicRadarEngine(
        known_tx_positions=sim.get_tx_positions(), solver_method="fista_tv"
    )
    for ev in sim.iter_events(duration_s=1.0):
        eng.ingest(ev)
    frame = eng.step()
    hist = TrackHistoryBuffer()

    payload = serialize_frame(
        frame,
        room_width=6.0,
        room_height=5.0,
        anchor_positions=sim.get_tx_positions(),
        ground_truth=sim.get_ground_truth(),
        track_history=hist,
        frame_seq=7,
        server_fps=19.8,
    )
    assert payload["seq"] == 7
    assert payload["room"]["width"] == 6.0
    assert payload["room"]["height"] == 5.0
    assert payload["density_map"]["nx"] == 40
    assert payload["density_map"]["ny"] == 34
    assert len(payload["density_map"]["values"]) == 40 * 34
    assert len(payload["links"]) > 0
    assert len(payload["csi"]["subcarriers"]) == 52
    assert len(payload["csi"]["amp"]) == 52
    assert len(payload["psd"]["freqs_hz"]) == len(payload["psd"]["power_db"])
    assert payload["solver"]["method"] == "fista_tv"
    assert len(payload["solver"]["convergence"]) >= 1
    assert payload["hud"]["passive_rx_only"] is True
    assert payload["hud"]["amp_enable"] is False
    assert payload["hud"]["bias_tee_enable"] is False
    assert len(payload["ground_truth"]) == 2
    assert any(
        "csi_amp" in lnk and len(lnk["csi_amp"]) == 52 for lnk in payload["links"]
    )

    # Adversarial NaN/Inf frame must still serialize as strict RFC 8259 JSON
    bad_dm = np.full((10, 12), np.nan, dtype=np.float64)
    bad_dm[0, 0] = np.inf
    bad_frame = TomographyFrame(
        timestamp=float("nan"),
        density_map=bad_dm,
        grid_x=np.linspace(0.25, 5.75, 12),
        grid_y=np.linspace(0.25, 4.75, 10),
        rx_pos=(3.0, 0.35),
        tracks=[
            TrackedHuman(
                track_id=1,
                x=2.0,
                y=3.0,
                vx=float("nan"),
                vy=0.2,
                breathing_bpm=float("inf"),
                confidence=0.9,
            )
        ],
        links=[
            LinkDiagnostic(
                link_id="AA:BB:CC:00:00:01->RX",
                tx_bssid="AA:BB:CC:00:00:01",
                rx_id="RX",
                tx_pos=(0.4, 4.7),
                rx_pos=(3.0, 0.35),
                freq_hz=2437e6,
                is_peer=False,
                packet_count=5,
                rssi_delta_db=float("nan"),
                csi_turbulence=0.1,
                perturbation_score=0.5,
            )
        ],
        latest_csi_amp=np.full(52, np.nan),
        latest_csi_phase=np.full(52, np.inf),
        psd_freqs_hz=np.linspace(0.1, 25.0, 16),
        psd_power_db=np.full(16, np.nan),
        telemetry={"solver_convergence": [1.0, float("nan"), 0.2]},
    )
    raw_json = serialize_frame_json(bad_frame)
    parsed = json.loads(raw_json)
    assert parsed["tracks"][0]["breathing_bpm"] is None
    assert parsed["tracks"][0]["vx"] == 0.0
    assert parsed["links"][0]["rssi_delta_db"] == 0.0
    assert all(v == 0.0 for v in parsed["density_map"]["values"])


def test_hub_backpressure_and_clean_close() -> None:
    hub = TomographyStateHub(subscriber_queue_size=2)
    q_slow = hub.subscribe()
    assert hub.subscriber_count == 1

    for idx in range(25):
        hub.publish_frame_json(json.dumps({"seq": idx + 1}))

    # Drop-oldest keeps the 2 freshest frames without blocking
    first = q_slow.get_nowait()
    second = q_slow.get_nowait()
    assert first is not None and second is not None
    assert first[0] == 24
    assert second[0] == 25

    # Refill queue and confirm close() delivers None sentinel without deadlock
    hub.publish_frame_json(json.dumps({"seq": 26}))
    hub.publish_frame_json(json.dumps({"seq": 27}))
    hub.close()
    drained: list[Any] = []
    while not q_slow.empty():
        drained.append(q_slow.get_nowait())
    assert None in drained


def test_web_server_routes_security_headers_control_and_sse(tmp_path: Path) -> None:
    received_cmds: list[dict[str, Any]] = []

    def _control_cb(cmd: dict[str, Any]) -> dict[str, Any]:
        received_cmds.append(cmd)
        return {"echo": cmd}

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text(
        "<!DOCTYPE html><html lang='en'><body>OK</body></html>", encoding="utf-8"
    )
    (static_dir / "app.js").write_text("console.log('ok');", encoding="utf-8")

    dash = WebRadarDashboard(
        room_width=6.0,
        room_height=5.0,
        host="127.0.0.1",
        port=0,
        control_handler=_control_cb,
        sim_interactive=True,
        static_dir=static_dir,
    )
    try:
        base = dash.url.rstrip("/")

        # 1. Index HTML + strict security headers
        st, hdrs, body = _http_request(f"{base}/")
        assert st == 200
        assert "text/html" in hdrs["content-type"]
        assert hdrs["x-content-type-options"] == "nosniff"
        assert hdrs["x-frame-options"] == "SAMEORIGIN"
        assert hdrs["cross-origin-resource-policy"] == "same-origin"
        assert hdrs["referrer-policy"] == "strict-origin-when-cross-origin"
        csp = hdrs["content-security-policy"]
        assert "script-src 'self'" in csp
        assert "object-src 'none'" in csp
        assert "base-uri 'none'" in csp
        assert b"OK" in body

        # 2. Static asset allowlist + traversal defense
        st_js, hdrs_js, body_js = _http_request(f"{base}/static/app.js")
        assert st_js == 200
        assert "application/javascript" in hdrs_js["content-type"]
        assert b"console.log" in body_js

        st_trav, _, _ = _http_request(f"{base}/static/../server.py")
        assert st_trav == 404
        st_missing, _, _ = _http_request(f"{base}/static/nonexistent.js")
        assert st_missing == 404

        # 3. Health, config, and initial waiting frame
        st_h, _, b_h = _http_request(f"{base}/api/health")
        assert st_h == 200
        health = json.loads(b_h)
        assert health["status"] == "ok"
        assert health["passive_rx_only"] is True
        assert health["amp_enable"] is False

        st_c, _, b_c = _http_request(f"{base}/api/config")
        assert st_c == 200
        cfg = json.loads(b_c)
        assert cfg["room_width"] == 6.0
        assert cfg["sim_interactive"] is True
        assert cfg["control_enabled"] is True

        st_f0, _, b_f0 = _http_request(f"{base}/api/frame")
        assert st_f0 == 200
        assert json.loads(b_f0)["status"] == "waiting"

        # 4. Control endpoint validation, safety locks, and Fetch Metadata
        st_ok, _, b_ok = _http_request(
            f"{base}/api/control",
            "POST",
            body=json.dumps(
                {
                    "solver_method": "nnls",
                    "lam_tv": 0.03,
                    "window_s": 2.0,
                    "sim_target": {"id": 1, "x": 99.0, "y": -5.0},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_ok == 200
        resp_ok = json.loads(b_ok)
        assert resp_ok["ok"] is True
        # Target coordinates clamped cleanly inside room margins
        assert received_cmds[-1]["sim_target"]["x"] == pytest.approx(5.75)
        assert received_cmds[-1]["sim_target"]["y"] == pytest.approx(0.25)

        # Forbidden passive safety keys rejected
        for bad_key in ("amp_enable", "bias_tee_enable", "tx_enable"):
            st_bad, _, b_bad = _http_request(
                f"{base}/api/control",
                "POST",
                body=json.dumps({bad_key: True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            assert st_bad == 400
            assert "safety lock" in json.loads(b_bad)["error"].lower()

        # Unknown key, wrong media type, oversized payload, cross-site Fetch Metadata
        st_unk, _, _ = _http_request(
            f"{base}/api/control",
            "POST",
            body=json.dumps({"unknown_field": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_unk == 400

        st_ct, _, _ = _http_request(
            f"{base}/api/control",
            "POST",
            body=b'{"solver_method":"sart"}',
            headers={"Content-Type": "text/plain"},
        )
        assert st_ct == 415

        st_big, _, _ = _http_request(
            f"{base}/api/control",
            "POST",
            body=b" " * (20 * 1024),
            headers={"Content-Type": "application/json"},
        )
        assert st_big == 413

        st_cs, _, _ = _http_request(
            f"{base}/api/control",
            "POST",
            body=b'{"action":"calibrate_baseline"}',
            headers={
                "Content-Type": "application/json",
                "Sec-Fetch-Site": "cross-site",
            },
        )
        assert st_cs == 403

        # 5. Live SSE streaming delivers config + realtime TomographyFrame updates
        sse_events: list[tuple[str, dict[str, Any]]] = []
        sse_ready = threading.Event()

        def _sse_reader() -> None:
            parsed = urlparse(f"{base}/api/stream")
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5.0)
            try:
                conn.request("GET", "/api/stream")
                resp = conn.getresponse()
                assert resp.status == 200
                assert "text/event-stream" in (resp.getheader("Content-Type") or "")
                cur_event = "message"
                while len(sse_events) < 3:
                    line_b = resp.readline()
                    if not line_b:
                        break
                    line = line_b.decode("utf-8").strip()
                    if line.startswith("event:"):
                        cur_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line.split(":", 1)[1].strip())
                        sse_events.append((cur_event, data))
                        if cur_event == "config":
                            sse_ready.set()
            finally:
                conn.close()

        t_sse = threading.Thread(target=_sse_reader, daemon=True)
        t_sse.start()
        assert sse_ready.wait(timeout=3.0)

        # Publish two real frames
        dummy_frame = TomographyFrame(
            timestamp=100.0,
            density_map=np.ones((12, 16), dtype=np.float64) * 0.4,
            grid_x=np.linspace(0.2, 5.8, 16),
            grid_y=np.linspace(0.2, 4.8, 12),
            rx_pos=(3.0, 0.35),
            tracks=[],
            links=[],
            latest_csi_amp=np.ones(52),
            latest_csi_phase=np.zeros(52),
            psd_freqs_hz=np.linspace(0.1, 25.0, 32),
            psd_power_db=np.full(32, -40.0),
            telemetry={"solver_method": "fista_tv", "solver_convergence": [1.5, 0.4]},
        )
        ms1 = dash.update(dummy_frame)
        assert ms1 >= 0.0
        time.sleep(0.03)
        dummy_frame.timestamp = 100.05
        dash.update(dummy_frame)

        t_sse.join(timeout=3.0)
        assert not t_sse.is_alive()
        assert sse_events[0][0] == "config"
        frame_msgs = [ev for ev in sse_events if ev[0] == "frame"]
        assert len(frame_msgs) >= 2
        assert frame_msgs[-1][1]["timestamp"] == pytest.approx(100.05)

        # 6. Snapshot export (.json and .png)
        snap_json = tmp_path / "snap.json"
        snap_png = tmp_path / "snap.png"
        dash.save_snapshot(snap_json)
        dash.save_snapshot(snap_png)
        assert json.loads(snap_json.read_text(encoding="utf-8"))[
            "timestamp"
        ] == pytest.approx(100.05)
        assert snap_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        dash.close()


def test_realtime_web_server_context_manager_fallback_html(tmp_path: Path) -> None:
    hub = TomographyStateHub()
    with RealtimeWebServer(
        hub, host="127.0.0.1", port=0, static_dir=tmp_path / "missing_static"
    ) as srv:
        st, _, body = _http_request(srv.url)
        assert st == 200
        assert b"HackRF Passive Wi-Fi Tomographic Imaging" in body
    assert hub.is_closed
