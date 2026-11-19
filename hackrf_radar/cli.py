"""Command-line interface and end-to-end execution loop for HackRF Passive Wi-Fi Radar."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RadarConfig, load_config
from .plotter import (
    _GUI_BACKEND_ERRORS,
    CompositeRadarDashboard,
    create_plotter,
    has_gui_display,
    select_ui_backend,
)
from .tomography.geometry import canonicalize_mac, is_unicast_station_mac


def _default_scenario_for_humans(num_humans: int) -> str:
    if num_humans <= 0:
        return "empty"
    if num_humans == 1:
        return "single_walk"
    if num_humans == 2:
        return "walk_and_breathe"
    return "three_humans"


def _parse_channels(
    raw_channels: Sequence[str] | None, fallback: list[int]
) -> list[int]:
    if not raw_channels:
        return list(fallback)
    out: list[int] = []
    for token in raw_channels:
        for part in str(token).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out or list(fallback)


@dataclass
class PipelineRunResult:
    """Summary metrics returned by a completed radar pipeline run."""

    mode: str
    ui_backend: str
    capture_backend: str
    packets_processed: int
    frames_rendered: int
    mean_render_ms: float
    last_frame: Any = None
    snapshot_path: Path | None = None
    recorded_path: Path | None = None
    web_url: str | None = None
    render_times_ms: list[float] = field(default_factory=list)


def _iter_jsonl_events(path: Path, duration_s: float | None = None) -> Iterator[Any]:
    from .capture import PacketEvent

    t0: float | None = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            ev = PacketEvent.from_dict(payload)
            if t0 is None:
                t0 = float(ev.timestamp)
            if duration_s is not None and (float(ev.timestamp) - t0) > duration_s:
                break
            yield ev


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hackrf-radar",
        description=(
            "Passive 0 dBm RX-only multi-human Wi-Fi Radio Tomographic Imaging (RTI), "
            "52-subcarrier L-LTF CSI, and vital-sign radar for HackRF One."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("sim", "live", "replay"),
        default="sim",
        help="Execution source: synthetic simulator ('sim'), physical HackRF ('live'), or file ('replay').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to room geometry & anchor JSON config (default: configs/example_room.json).",
    )
    parser.add_argument(
        "--humans",
        type=int,
        default=2,
        help="Number of simulated humans in --mode sim (0, 1, 2, 3+).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help=(
            "Simulation scenario in --mode sim: empty | single_walk | single_breathe | "
            "walk_and_breathe | two_walkers | three_humans."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for --mode sim.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file path for --mode replay (.jsonl, .pcap/.pcapng, or .cs8/.cf32 I/Q).",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Optional output JSONL path to record cleartext PacketEvent telemetry.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Capture/simulation duration in seconds (default: 6.0s in sim, continuous until Ctrl+C in live).",
    )
    parser.add_argument(
        "--save-snapshot",
        type=str,
        default=None,
        help="Save final rendered radar dashboard snapshot to PNG (or .txt for ANSI TUI).",
    )
    parser.add_argument(
        "--ui",
        choices=("auto", "pyqtgraph", "matplotlib", "tui", "web", "none"),
        default="auto",
        help="Live radar UI backend (default: auto; use 'web' for the realtime browser UI).",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Expose the realtime tomography web interface alongside the active --ui backend.",
    )
    parser.add_argument(
        "--web-host",
        type=str,
        default="127.0.0.1",
        help="Bind host for the realtime web interface (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8765,
        help="TCP port for the realtime web interface (default: 8765; 0 for ephemeral port).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable X11/Qt GUI windows (uses TUI when --ui auto, or headless renderer).",
    )
    parser.add_argument(
        "--backend",
        choices=(
            "auto",
            "soapy",
            "libhackrf",
            "osmosdr",
            "hackrf_transfer",
            "gr_zmq",
            "pcap",
            "iq_file",
        ),
        default=None,
        help="Live HackRF RX backend override (default: from config / auto fallback).",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="802.11 channel hop list, e.g. --channels 1 6 11 or --channels 1,6,11,36,149.",
    )
    parser.add_argument(
        "--dwell-ms",
        type=float,
        default=None,
        help="Channel hop dwell time in milliseconds (default: 80 ms).",
    )
    parser.add_argument(
        "--lna-db",
        type=int,
        default=None,
        help="HackRF IF LNA gain in dB, 0..40 in 8 dB steps (RF AMP is locked OFF).",
    )
    parser.add_argument(
        "--vga-db",
        type=int,
        default=None,
        help="HackRF baseband VGA gain in dB, 0..62 in 2 dB steps.",
    )
    parser.add_argument(
        "--zmq-endpoint",
        type=str,
        default=None,
        help="Optional GNU Radio gr-ieee802-11 ZMQ SUB endpoint (e.g. tcp://127.0.0.1:5555).",
    )
    parser.add_argument(
        "--solver",
        choices=("fista_tv", "nnls", "sart"),
        default=None,
        help="Non-negative tomographic inversion solver (default: fista_tv).",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=None,
        help="Sliding observation window in seconds (default: 1.5s).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target tomographic inversion & UI frame rate in Hz (default: 20 Hz).",
    )
    parser.add_argument(
        "--calibrate-s",
        type=float,
        default=None,
        help="Initial empty-room baseline calibration duration in seconds.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace --mode sim or --mode replay in wall-clock real time.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run --mode sim or --mode replay as fast as CPU allows without sleep pacing.",
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Hide ground-truth target overlay in --mode sim.",
    )
    return parser


def run_pipeline(
    args: argparse.Namespace,
    *,
    on_web_ready: Callable[[str], None] | None = None,
    stop_event: threading.Event | None = None,
) -> PipelineRunResult:
    """Execute the passive Wi-Fi tomographic radar pipeline from parsed CLI arguments."""
    from .capture import HackRFPacketCapture, SyntheticRadarStream
    from .tomography import TomographicRadarEngine

    cfg: RadarConfig = load_config(args.config)

    # Apply CLI overrides while preserving strict passive 0 dBm RX safety locks
    if args.channels is not None:
        cfg.rx.channels = _parse_channels(args.channels, cfg.rx.channels)
    if args.dwell_ms is not None:
        cfg.rx.dwell_ms = float(args.dwell_ms)
    if args.lna_db is not None:
        cfg.rx.lna_db = int(args.lna_db)
    if args.vga_db is not None:
        cfg.rx.vga_db = int(args.vga_db)
    if args.backend is not None:
        cfg.rx.backend = str(args.backend)
    if args.zmq_endpoint is not None:
        cfg.rx.zmq_endpoint = str(args.zmq_endpoint)
    if args.solver is not None:
        cfg.solver.solver_method = str(args.solver)
    if args.window_s is not None:
        cfg.solver.window_s = float(args.window_s)
    if args.fps is not None:
        cfg.solver.target_fps = max(1.0, float(args.fps))
    if args.calibrate_s is not None:
        cfg.solver.calibration_s = max(0.0, float(args.calibrate_s))

    # Re-validate RxSettings invariants (AMP=False, bias_tee=False, valid gains/channels)
    cfg.rx.__post_init__()
    cfg.solver.__post_init__()
    if cfg.rx.amp_enable or cfg.rx.bias_tee_enable:
        raise RuntimeError(
            "Safety violation: HackRF AMP and bias-tee must remain disabled."
        )

    use_web = (args.ui == "web") or bool(getattr(args, "web", False))
    web_host = str(getattr(args, "web_host", "127.0.0.1"))
    web_port = int(getattr(args, "web_port", 8765))

    headless_env = (
        bool(args.no_gui)
        or args.ui in {"none", "web"}
        or os.environ.get("MPLBACKEND", "").lower() == "agg"
        or os.environ.get("QT_QPA_PLATFORM", "").lower() in {"offscreen", "minimal"}
        or not has_gui_display()
    )
    headless_plot = headless_env or (
        args.save_snapshot is not None and not args.realtime and args.mode != "live"
    )
    resolved_ui = select_ui_backend(args.ui, no_gui=bool(args.no_gui))

    duration_s: float | None = args.duration
    if duration_s is None and args.mode == "sim":
        if use_web and not args.fast and args.save_snapshot is None:
            # Continuous realtime streaming until Ctrl+C / stop_event in interactive web mode
            duration_s = None
        else:
            duration_s = 6.0

    pace_realtime = bool(args.realtime) or (
        not args.fast
        and args.save_snapshot is None
        and (
            (not headless_plot and args.mode == "sim")
            or (use_web and args.mode in {"sim", "replay"})
        )
    )

    known_tx = cfg.known_tx_positions()
    anchor_labels = cfg.anchor_labels()
    capture_obj: Any = None
    sim_stream: Any = None
    capture_backend = cfg.rx.backend

    stop_requested = False
    is_main_thread = threading.current_thread() is threading.main_thread()
    prev_sigint = signal.getsignal(signal.SIGINT) if is_main_thread else None

    def _should_stop() -> bool:
        return stop_requested or (stop_event is not None and stop_event.is_set())

    def _on_sigint(_sig: int, _frm: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    record_fh = None
    recorded_path: Path | None = None
    dashboard: Any = None
    web_url: str | None = None

    try:
        if is_main_thread:
            signal.signal(signal.SIGINT, _on_sigint)

        if args.record:
            recorded_path = Path(args.record)
            recorded_path.parent.mkdir(parents=True, exist_ok=True)
            record_fh = recorded_path.open("w", encoding="utf-8")

        # Set up event iterator according to --mode
        if args.mode == "sim":
            scenario = args.scenario or _default_scenario_for_humans(int(args.humans))
            sim_stream = SyntheticRadarStream(
                room_width=cfg.room_width,
                room_height=cfg.room_height,
                rx_pos=cfg.rx.rx_pos,
                num_humans=max(0, int(args.humans)),
                scenario=scenario,
                channels=cfg.rx.channels,
                seed=int(args.seed),
            )
            sim_tx = sim_stream.get_tx_positions()
            known_tx = {**known_tx, **sim_tx}
            for bssid in sim_tx:
                anchor_labels.setdefault(bssid, bssid[-5:])
            capture_backend = f"sim:{scenario}"
            event_iter = sim_stream.iter_events(
                duration_s=duration_s, realtime=pace_realtime
            )

        elif args.mode == "live":
            cap_kwargs: dict[str, Any] = {
                "channels": cfg.rx.channels,
                "dwell_ms": cfg.rx.dwell_ms,
                "settle_ms": cfg.rx.pll_settle_ms,
                "sample_rate": cfg.rx.sample_rate_hz,
                "lna_db": cfg.rx.lna_db,
                "vga_db": cfg.rx.vga_db,
                "dc_offset_removal": cfg.rx.dc_offset_removal,
                "backend": cfg.rx.backend,
                "zmq_endpoint": cfg.rx.zmq_endpoint,
            }
            sig_params = inspect.signature(HackRFPacketCapture).parameters
            filtered_kwargs = {k: v for k, v in cap_kwargs.items() if k in sig_params}
            capture_obj = HackRFPacketCapture(**filtered_kwargs)
            capture_backend = str(
                getattr(capture_obj, "active_backend", cfg.rx.backend)
            )
            iter_params = inspect.signature(capture_obj.iter_events).parameters
            if "stop_check" in iter_params:
                event_iter = capture_obj.iter_events(
                    duration_s=duration_s, stop_check=_should_stop
                )
            else:
                event_iter = capture_obj.iter_events(duration_s=duration_s)

        elif args.mode == "replay":
            if not args.input:
                raise ValueError(
                    "--mode replay requires --input <path> (.jsonl, .pcap, or .cs8/.cf32)."
                )
            in_path = Path(args.input)
            if not in_path.is_file():
                raise FileNotFoundError(f"Replay input file not found: {in_path}")
            suffix = in_path.suffix.lower()
            if suffix in {".jsonl", ".ndjson", ".json"}:
                capture_backend = f"replay:jsonl({in_path.name})"
                event_iter = _iter_jsonl_events(in_path, duration_s=duration_s)
            elif suffix in {".pcap", ".pcapng", ".cap"}:
                capture_obj = HackRFPacketCapture(
                    channels=cfg.rx.channels,
                    backend="pcap",
                    pcap_path=str(in_path),
                )
                capture_backend = f"replay:pcap({in_path.name})"
                event_iter = capture_obj.iter_events(
                    duration_s=duration_s, stop_check=_should_stop
                )
            else:
                capture_obj = HackRFPacketCapture(
                    channels=cfg.rx.channels,
                    sample_rate=cfg.rx.sample_rate_hz,
                    dc_offset_removal=cfg.rx.dc_offset_removal,
                    backend="iq_file",
                    iq_path=str(in_path),
                )
                capture_backend = f"replay:iq({in_path.name})"
                event_iter = capture_obj.iter_events(
                    duration_s=duration_s, stop_check=_should_stop
                )
        else:
            raise ValueError(f"Unsupported mode: {args.mode!r}")

        engine = TomographicRadarEngine(
            room_width=cfg.room_width,
            room_height=cfg.room_height,
            grid_shape=cfg.solver.grid_shape,
            rx_pos=cfg.rx.rx_pos,
            known_tx_positions=known_tx or None,
            solver_method=cfg.solver.solver_method,
            window_s=cfg.solver.window_s,
            enable_auto_discovery=cfg.auto_discovery.enabled,
            enable_peer_chords=cfg.auto_discovery.enable_sifs_peer_chords,
            min_auto_packets=cfg.auto_discovery.min_packets,
            perimeter_margin_m=cfg.auto_discovery.perimeter_margin_m,
            use_rssi_log_distance=cfg.auto_discovery.use_rssi_log_distance,
            ref_rssi_1m_dbm=cfg.auto_discovery.ref_rssi_1m_dbm,
            path_loss_exponent=cfg.auto_discovery.path_loss_exponent,
        )

        show_gt = (args.mode == "sim") and (not args.no_gt)

        def _web_control_handler(cmd: dict[str, Any]) -> dict[str, Any]:
            applied: dict[str, Any] = {}
            solver_updates = {
                k: cmd[k]
                for k in ("solver_method", "lam_tv", "lam_l2", "window_s")
                if k in cmd
            }
            if solver_updates:
                applied["solver"] = engine.set_solver_params(**solver_updates)
            action = cmd.get("action")
            if action == "calibrate_baseline":
                engine.calibrate_baseline()
                applied["calibrated"] = True
            elif action == "reset_sim_targets":
                if sim_stream is None:
                    raise ValueError("Target reset is only available in --mode sim.")
                sim_stream.clear_interactive_targets()
                applied["sim_targets_reset"] = True
            if "sim_target" in cmd:
                if sim_stream is None:
                    raise ValueError(
                        "Interactive target dragging is only available in --mode sim."
                    )
                tgt = cmd["sim_target"]
                applied["sim_target"] = sim_stream.set_interactive_target(
                    tgt["id"], tgt["x"], tgt["y"]
                )
            return applied

        web_config_summary: dict[str, Any] = {
            "room_name": cfg.room_name,
            "room_width": cfg.room_width,
            "room_height": cfg.room_height,
            "mode": args.mode,
            "capture_backend": capture_backend,
            "channels": list(cfg.rx.channels),
            "lna_db": cfg.rx.lna_db,
            "vga_db": cfg.rx.vga_db,
            "grid_shape": list(cfg.solver.grid_shape),
            "solver_method": cfg.solver.solver_method,
            "passive_rx_only": True,
        }
        sim_interactive = (
            (args.mode == "sim")
            and (sim_stream is not None)
            and (getattr(sim_stream, "num_humans", 0) > 0)
        )

        primary_dash: Any = None
        web_dash: Any = None
        if args.ui == "web":
            web_dash = create_plotter(
                "web",
                room_width=cfg.room_width,
                room_height=cfg.room_height,
                anchor_positions=known_tx,
                anchor_labels=anchor_labels,
                show_ground_truth=show_gt,
                web_host=web_host,
                web_port=web_port,
                config_summary=web_config_summary,
                control_handler=_web_control_handler,
                sim_interactive=sim_interactive,
            )
            dashboard = web_dash
        else:
            if args.ui != "none":
                primary_dash = create_plotter(
                    args.ui,
                    no_gui=bool(args.no_gui),
                    headless=headless_plot,
                    room_width=cfg.room_width,
                    room_height=cfg.room_height,
                    anchor_positions=known_tx,
                    anchor_labels=anchor_labels,
                    show_ground_truth=show_gt,
                )
            if use_web:
                web_dash = create_plotter(
                    "web",
                    room_width=cfg.room_width,
                    room_height=cfg.room_height,
                    anchor_positions=known_tx,
                    anchor_labels=anchor_labels,
                    show_ground_truth=show_gt,
                    web_host=web_host,
                    web_port=web_port,
                    config_summary=web_config_summary,
                    control_handler=_web_control_handler,
                    sim_interactive=sim_interactive,
                )
            if primary_dash is not None and web_dash is not None:
                dashboard = CompositeRadarDashboard([primary_dash, web_dash])
            else:
                dashboard = primary_dash or web_dash

        if web_dash is not None:
            web_url = web_dash.url
            print(
                f"[hackrf-radar] web UI listening on {web_url} (0 dBm RX-only)",
                file=sys.stderr,
                flush=True,
            )
            if on_web_ready is not None:
                on_web_ready(web_url)

        frame_period = 1.0 / max(cfg.solver.target_fps, 1.0)
        packets_processed = 0
        seq_gaps_total = 0
        fcs_bad_total = 0
        adc_clip_max = 0.0
        frame_counts: dict[str, int] = {}
        seen_bssids: set[str] = set()

        t_first_ev: float | None = None
        last_step_sim_t: float | None = None
        last_step_wall_t = time.monotonic()
        calibrated = cfg.solver.calibration_s <= 0.0
        last_ev_ts: float | None = None
        last_frame: Any = None
        render_times_ms: list[float] = []

        def _enrich_and_render(step_ts: float | None) -> None:
            nonlocal last_frame, adc_clip_max
            frame = engine.step(now=step_ts)
            if capture_obj is not None:
                active_b = getattr(capture_obj, "active_backend", None)
                if active_b:
                    frame.telemetry["backend"] = str(active_b)
                adc_clip_max = max(
                    adc_clip_max,
                    float(getattr(capture_obj, "max_adc_clip_ratio", 0.0) or 0.0),
                )
            frame.telemetry.setdefault("backend", capture_backend)
            frame.telemetry.setdefault("frame_counts", dict(frame_counts))
            frame.telemetry.setdefault("seq_gaps", seq_gaps_total)
            frame.telemetry["bssid_count"] = len(seen_bssids)
            frame.telemetry.setdefault(
                "fcs_drop_pct",
                (100.0 * fcs_bad_total / max(packets_processed, 1)),
            )
            frame.telemetry.setdefault("adc_clip_max", adc_clip_max)
            last_frame = frame

            if dashboard is not None:
                gt = (
                    sim_stream.get_ground_truth(step_ts)
                    if (show_gt and sim_stream is not None)
                    else None
                )
                ms = dashboard.update(frame, ground_truth=gt)
                render_times_ms.append(ms)

        for ev in event_iter:
            if _should_stop():
                break

            packets_processed += 1
            ev_ts = float(ev.timestamp)
            last_ev_ts = ev_ts
            if t_first_ev is None:
                t_first_ev = ev_ts

            ft = str(getattr(ev, "frame_type", "UNKNOWN")).upper()
            frame_counts[ft] = frame_counts.get(ft, 0) + 1
            seq_gaps_total += max(0, int(getattr(ev, "seq_gap", 0) or 0))
            fcs_ok = bool(getattr(ev, "fcs_ok", True))
            if not fcs_ok:
                fcs_bad_total += 1
            adc_clip_max = max(
                adc_clip_max, float(getattr(ev, "adc_clip_ratio", 0.0) or 0.0)
            )
            bssid = str(getattr(ev, "bssid", ""))
            if fcs_ok and is_unicast_station_mac(bssid):
                seen_bssids.add(canonicalize_mac(bssid))

            if record_fh is not None:
                record_fh.write(json.dumps(ev.to_dict()) + "\n")

            engine.ingest(ev)

            if (
                not calibrated
                and t_first_ev is not None
                and (ev_ts - t_first_ev) >= cfg.solver.calibration_s
            ):
                engine.calibrate_baseline()
                calibrated = True

            if args.mode == "live":
                now_wall = time.monotonic()
                if (now_wall - last_step_wall_t) >= frame_period:
                    _enrich_and_render(ev_ts)
                    last_step_wall_t = now_wall
                    last_step_sim_t = ev_ts
            else:
                if last_step_sim_t is None or (ev_ts - last_step_sim_t) >= frame_period:
                    _enrich_and_render(ev_ts)
                    last_step_sim_t = ev_ts
                    if args.mode == "replay" and pace_realtime:
                        time.sleep(min(frame_period * 0.5, 0.03))

        # Ensure at least one final tomographic step & render (crucial for short 0.5s live runs)
        if last_frame is None or (
            last_ev_ts is not None and last_ev_ts != last_step_sim_t
        ):
            _enrich_and_render(last_ev_ts)

        snapshot_out: Path | None = None
        if args.save_snapshot:
            if dashboard is not None:
                snapshot_out = dashboard.save_snapshot(args.save_snapshot)
            elif last_frame is not None:
                snap_dash = create_plotter(
                    "matplotlib",
                    headless=True,
                    room_width=cfg.room_width,
                    room_height=cfg.room_height,
                    anchor_positions=known_tx,
                    anchor_labels=anchor_labels,
                    show_ground_truth=show_gt,
                )
                try:
                    gt = (
                        sim_stream.get_ground_truth(last_ev_ts)
                        if (show_gt and sim_stream is not None)
                        else None
                    )
                    snap_dash.update(last_frame, ground_truth=gt)
                    snapshot_out = snap_dash.save_snapshot(args.save_snapshot)
                finally:
                    snap_dash.close()

        if capture_obj is not None:
            active_b = getattr(capture_obj, "active_backend", None)
            if active_b:
                capture_backend = str(active_b)

        mean_ms = (
            float(sum(render_times_ms) / len(render_times_ms))
            if render_times_ms
            else 0.0
        )
        return PipelineRunResult(
            mode=args.mode,
            ui_backend=resolved_ui
            if args.ui != "none"
            else ("web" if use_web else "none"),
            capture_backend=capture_backend,
            packets_processed=packets_processed,
            frames_rendered=len(render_times_ms)
            if dashboard is not None
            else (1 if last_frame is not None else 0),
            mean_render_ms=mean_ms,
            last_frame=last_frame,
            snapshot_path=snapshot_out,
            recorded_path=recorded_path,
            web_url=web_url,
            render_times_ms=render_times_ms,
        )

    finally:
        if is_main_thread and prev_sigint is not None:
            signal.signal(signal.SIGINT, prev_sigint)
        if record_fh is not None:
            with contextlib.suppress(OSError, ValueError):
                record_fh.flush()
                record_fh.close()
        if capture_obj is not None:
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                capture_obj.close()
        if dashboard is not None:
            with contextlib.suppress(*_GUI_BACKEND_ERRORS):
                dashboard.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_pipeline(args)
    n_tracks = len(getattr(result.last_frame, "tracks", []) or [])
    snap_msg = f" | snapshot={result.snapshot_path}" if result.snapshot_path else ""
    rec_msg = f" | recorded={result.recorded_path}" if result.recorded_path else ""
    print(
        f"[hackrf-radar] mode={result.mode} backend={result.capture_backend} "
        f"ui={result.ui_backend} packets={result.packets_processed} "
        f"frames={result.frames_rendered} targets={n_tracks} "
        f"mean_render={result.mean_render_ms:.1f}ms{snap_msg}{rec_msg}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
