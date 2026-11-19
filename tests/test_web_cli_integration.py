"""End-to-end CLI, live SSE streaming, runtime controls, and dual-UI integration tests."""

from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from hackrf_radar.cli import build_arg_parser, run_pipeline


def _req(
    url: str,
    method: str = "GET",
    *,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5.0)
    try:
        headers: dict[str, str] = {}
        raw_body: bytes | None = None
        if body is not None:
            raw_body = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        conn.request(method, parsed.path or "/", body=raw_body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, json.loads(data.decode("utf-8"))
    finally:
        conn.close()


def test_cli_realtime_web_streaming_and_live_control_mutations() -> None:
    web_url_box: list[str] = []
    ready_ev = threading.Event()
    stop_ev = threading.Event()
    result_box: list[Any] = []
    err_box: list[Exception] = []

    def _on_ready(url: str) -> None:
        web_url_box.append(url)
        ready_ev.set()

    def _runner() -> None:
        try:
            args = build_arg_parser().parse_args(
                [
                    "--mode",
                    "sim",
                    "--humans",
                    "2",
                    "--scenario",
                    "walk_and_breathe",
                    "--ui",
                    "web",
                    "--web-host",
                    "127.0.0.1",
                    "--web-port",
                    "0",
                    "--fps",
                    "20",
                    "--realtime",
                ]
            )
            res = run_pipeline(args, on_web_ready=_on_ready, stop_event=stop_ev)
            result_box.append(res)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            err_box.append(exc)
        finally:
            ready_ev.set()

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()

    try:
        assert ready_ev.wait(timeout=5.0), "Web server did not start in time"
        assert not err_box, f"Pipeline raised: {err_box[0]!r}"
        base = web_url_box[0].rstrip("/")

        # Verify config endpoint reports interactive sim + passive safety locks
        st_cfg, cfg = _req(f"{base}/api/config")
        assert st_cfg == 200
        assert cfg["passive_rx_only"] is True
        assert cfg["amp_enable"] is False
        assert cfg["sim_interactive"] is True
        assert cfg["control_enabled"] is True

        # Stream at least 3 live SSE frames from the running radar pipeline
        parsed = urlparse(f"{base}/api/stream")
        sse_frames: list[dict[str, Any]] = []
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=6.0)
        try:
            conn.request("GET", "/api/stream")
            resp = conn.getresponse()
            assert resp.status == 200
            cur_event = ""
            while len(sse_frames) < 3:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8").strip()
                if line.startswith("event:"):
                    cur_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and cur_event == "frame":
                    sse_frames.append(json.loads(line.split(":", 1)[1].strip()))
        finally:
            conn.close()

        assert len(sse_frames) >= 3
        assert sse_frames[-1]["seq"] > sse_frames[0]["seq"]
        assert sse_frames[-1]["hud"]["passive_rx_only"] is True
        assert len(sse_frames[-1]["csi"]["amp"]) == 52
        assert len(sse_frames[-1]["links"]) > 0

        # Mutate solver method, lambda, window, calibrate baseline, and move sim target live
        st_ctrl, ctrl_resp = _req(
            f"{base}/api/control",
            "POST",
            body={
                "solver_method": "nnls",
                "lam_tv": 0.04,
                "window_s": 2.0,
                "action": "calibrate_baseline",
                "sim_target": {"id": 1, "x": 4.2, "y": 2.4},
            },
        )
        assert st_ctrl == 200
        assert ctrl_resp["ok"] is True

        # Passive RX safety lock must reject amp_enable even while running
        st_bad, _ = _req(
            f"{base}/api/control",
            "POST",
            body={"amp_enable": True},
        )
        assert st_bad == 400

        # Wait briefly for next frame reflecting updated solver, calibrated baseline, & moved target
        updated_frame: dict[str, Any] | None = None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            st_f, fr = _req(f"{base}/api/frame")
            if (
                st_f == 200
                and fr.get("solver", {}).get("method") == "nnls"
                and any(
                    gt.get("id") == 1 and abs(gt.get("x", 0.0) - 4.2) < 0.05
                    for gt in fr.get("ground_truth", [])
                )
            ):
                updated_frame = fr
                break
            time.sleep(0.05)

        assert updated_frame is not None
        assert updated_frame["solver"]["lam_tv"] == pytest.approx(0.04)
        assert updated_frame["solver"]["window_s"] == pytest.approx(2.0)
        assert any("baseline_csi_amp" in lnk for lnk in updated_frame["links"])

        # Reset interactive sim targets cleanly
        st_rst, rst_resp = _req(
            f"{base}/api/control",
            "POST",
            body={"action": "reset_sim_targets"},
        )
        assert st_rst == 200
        assert rst_resp["ok"] is True
    finally:
        stop_ev.set()
        worker.join(timeout=5.0)

    assert not worker.is_alive()
    assert not err_box
    assert len(result_box) == 1
    res = result_box[0]
    assert res.ui_backend == "web"
    assert res.web_url is not None and res.web_url.startswith("http://127.0.0.1:")
    assert res.frames_rendered >= 3


def test_cli_dual_tui_plus_web_and_replay_snapshot(tmp_path: Path) -> None:
    rec_path = tmp_path / "web_replay.jsonl"
    args_dual = build_arg_parser().parse_args(
        [
            "--mode",
            "sim",
            "--humans",
            "1",
            "--scenario",
            "single_walk",
            "--duration",
            "1.0",
            "--fast",
            "--no-gui",
            "--ui",
            "tui",
            "--web",
            "--web-port",
            "0",
            "--record",
            str(rec_path),
        ]
    )
    res_dual = run_pipeline(args_dual)
    assert res_dual.ui_backend == "tui"
    assert res_dual.web_url is not None
    assert res_dual.frames_rendered >= 1
    assert rec_path.is_file() and rec_path.stat().st_size > 500

    # Replay with --ui web and PNG snapshot export
    snap_png = tmp_path / "web_snap.png"
    args_rep = build_arg_parser().parse_args(
        [
            "--mode",
            "replay",
            "--input",
            str(rec_path),
            "--fast",
            "--ui",
            "web",
            "--web-port",
            "0",
            "--save-snapshot",
            str(snap_png),
        ]
    )
    res_rep = run_pipeline(args_rep)
    assert res_rep.ui_backend == "web"
    assert res_rep.web_url is not None
    assert snap_png.is_file()
    assert snap_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
