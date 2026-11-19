"""Zero-dependency stdlib HTTP + Server-Sent Events (SSE) realtime tomography web server."""

from __future__ import annotations

import contextlib
import json
import math
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

from ..plotter import MatplotlibRadarDashboard, TrackHistoryBuffer
from .serializer import serialize_frame_json

STATIC_DIR: Path = Path(__file__).resolve().parent / "static"

ALLOWED_STATIC_ASSETS: dict[str, str] = {
    "luminous.css": "text/css; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "app.js": "application/javascript; charset=utf-8",
    "theme-init.js": "application/javascript; charset=utf-8",
}

MAX_CONTROL_BODY_BYTES: int = 16 * 1024

FORBIDDEN_SAFETY_KEYS: frozenset[str] = frozenset(
    {
        "amp_enable",
        "bias_tee_enable",
        "tx_enable",
        "transmit",
        "tx_power",
        "tx_gain",
    }
)

ALLOWED_CONTROL_KEYS: frozenset[str] = frozenset(
    {
        "solver_method",
        "lam_tv",
        "lam_l2",
        "window_s",
        "action",
        "sim_target",
    }
)

ALLOWED_SOLVER_METHODS: frozenset[str] = frozenset({"fista_tv", "nnls", "sart"})
ALLOWED_ACTIONS: frozenset[str] = frozenset({"calibrate_baseline", "reset_sim_targets"})

_FALLBACK_INDEX_HTML: bytes = (
    b'<!DOCTYPE html><html lang="en"><head>'
    b'<meta charset="UTF-8">'
    b'<meta name="color-scheme" content="light dark">'
    b"<title>HackRF Passive Wi-Fi Tomographic Imaging</title>"
    b"</head><body><main><h1>HackRF Passive Wi-Fi Tomographic Imaging</h1>"
    b"<p>Realtime tomography stream active at <code>/api/stream</code>.</p>"
    b"</main></body></html>"
)

_HTML_CSP: str = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "frame-ancestors 'self'; "
    "form-action 'self'"
)


def _validate_control_payload(
    raw: Any,
    *,
    room_width: float,
    room_height: float,
) -> dict[str, Any]:
    """Validate and sanitize an incoming ``POST /api/control`` JSON command."""
    if not isinstance(raw, Mapping):
        raise TypeError("Control payload must be a JSON object.")
    if not raw:
        raise ValueError("Control payload must not be empty.")

    for key in raw:
        key_s = str(key)
        if key_s in FORBIDDEN_SAFETY_KEYS:
            raise ValueError(
                f"Passive 0 dBm RX-only safety lock: parameter {key_s!r} is forbidden."
            )
        if key_s not in ALLOWED_CONTROL_KEYS:
            raise ValueError(f"Unsupported control parameter: {key_s!r}")

    out: dict[str, Any] = {}
    if "solver_method" in raw:
        method = str(raw["solver_method"]).strip().lower()
        if method not in ALLOWED_SOLVER_METHODS:
            raise ValueError(
                f"Invalid solver_method {raw['solver_method']!r}; "
                f"expected one of {sorted(ALLOWED_SOLVER_METHODS)}."
            )
        out["solver_method"] = method

    if "lam_tv" in raw:
        val = float(raw["lam_tv"])
        if not math.isfinite(val) or not (0.0 <= val <= 1.0):
            raise ValueError(f"lam_tv must be in [0.0, 1.0], got {raw['lam_tv']!r}")
        out["lam_tv"] = val

    if "lam_l2" in raw:
        val = float(raw["lam_l2"])
        if not math.isfinite(val) or not (0.0 <= val <= 1.0):
            raise ValueError(f"lam_l2 must be in [0.0, 1.0], got {raw['lam_l2']!r}")
        out["lam_l2"] = val

    if "window_s" in raw:
        val = float(raw["window_s"])
        if not math.isfinite(val) or not (0.2 <= val <= 10.0):
            raise ValueError(
                f"window_s must be in [0.2, 10.0], got {raw['window_s']!r}"
            )
        out["window_s"] = val

    if "action" in raw:
        act = str(raw["action"]).strip().lower()
        if act not in ALLOWED_ACTIONS:
            raise ValueError(
                f"Unsupported control action {raw['action']!r}; "
                f"expected one of {sorted(ALLOWED_ACTIONS)}."
            )
        out["action"] = act

    if "sim_target" in raw:
        tgt = raw["sim_target"]
        if not isinstance(tgt, Mapping):
            raise ValueError("sim_target must be an object with id, x, and y.")
        tid = int(tgt.get("id", 1))
        tx = float(tgt["x"])
        ty = float(tgt["y"])
        if tid < 1 or tid > 32:
            raise ValueError(f"sim_target.id must be in 1..32, got {tid}")
        if not (math.isfinite(tx) and math.isfinite(ty)):
            raise ValueError("sim_target coordinates must be finite.")
        margin = 0.25
        tx = min(max(tx, margin), max(margin, room_width - margin))
        ty = min(max(ty, margin), max(margin, room_height - margin))
        out["sim_target"] = {"id": tid, "x": round(tx, 4), "y": round(ty, 4)}

    return out


class TomographyStateHub:
    """Thread-safe realtime pub/sub broker for serialized tomography frames."""

    def __init__(
        self,
        *,
        room_width: float = 6.0,
        room_height: float = 5.0,
        config_summary: Mapping[str, Any] | None = None,
        control_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        sim_interactive: bool = False,
        subscriber_queue_size: int = 2,
    ) -> None:
        self.room_width = float(room_width)
        self.room_height = float(room_height)
        self._subscriber_queue_size = max(1, int(subscriber_queue_size))
        self._control_handler = control_handler
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue[tuple[int, str] | None]] = set()
        self._latest_frame_json: str | None = None
        self._frame_seq: int = 0
        self._closed: bool = False
        self._started_at: float = time.monotonic()

        base_cfg: dict[str, Any] = {
            "room_name": "Passive Wi-Fi Tomographic Room",
            "room_width": self.room_width,
            "room_height": self.room_height,
            "passive_rx_only": True,
            "amp_enable": False,
            "bias_tee_enable": False,
            "sim_interactive": bool(sim_interactive),
            "control_enabled": control_handler is not None,
        }
        if config_summary:
            for k, v in config_summary.items():
                if k not in FORBIDDEN_SAFETY_KEYS:
                    base_cfg[k] = v
            base_cfg["passive_rx_only"] = True
            base_cfg["amp_enable"] = False
            base_cfg["bias_tee_enable"] = False
        self._config_dict: dict[str, Any] = base_cfg
        self._config_json: str = json.dumps(
            base_cfg, separators=(",", ":"), allow_nan=False
        )

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    @property
    def frame_seq(self) -> int:
        with self._lock:
            return self._frame_seq

    def set_control_handler(
        self,
        handler: Callable[[dict[str, Any]], dict[str, Any]] | None,
        *,
        sim_interactive: bool | None = None,
    ) -> None:
        """Attach or replace the runtime control callback."""
        with self._lock:
            self._control_handler = handler
            self._config_dict["control_enabled"] = handler is not None
            if sim_interactive is not None:
                self._config_dict["sim_interactive"] = bool(sim_interactive)
            self._config_json = json.dumps(
                self._config_dict, separators=(",", ":"), allow_nan=False
            )

    def get_config_json(self) -> str:
        with self._lock:
            return self._config_json

    def get_latest_frame_json(self) -> tuple[int, str | None]:
        with self._lock:
            return self._frame_seq, self._latest_frame_json

    def get_health_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "closed" if self._closed else "ok",
                "uptime_s": round(max(0.0, time.monotonic() - self._started_at), 2),
                "frame_seq": self._frame_seq,
                "subscribers": len(self._subscribers),
                "passive_rx_only": True,
                "amp_enable": False,
                "bias_tee_enable": False,
            }

    def publish_frame_json(self, frame_json: str) -> int:
        """Store and fan-out a serialized frame with drop-oldest backpressure."""
        with self._lock:
            if self._closed:
                return self._frame_seq
            self._frame_seq += 1
            seq = self._frame_seq
            self._latest_frame_json = frame_json
            subs = list(self._subscribers)

        item = (seq, frame_json)
        for sub_q in subs:
            try:
                sub_q.put_nowait(item)
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    sub_q.get_nowait()
                with contextlib.suppress(queue.Full):
                    sub_q.put_nowait(item)
        return seq

    def subscribe(self) -> queue.Queue[tuple[int, str] | None]:
        """Register a bounded queue for an SSE client."""
        sub_q: queue.Queue[tuple[int, str] | None] = queue.Queue(
            maxsize=self._subscriber_queue_size
        )
        with self._lock:
            if self._closed:
                sub_q.put_nowait(None)
            else:
                self._subscribers.add(sub_q)
        return sub_q

    def unsubscribe(self, sub_q: queue.Queue[tuple[int, str] | None]) -> None:
        with self._lock:
            self._subscribers.discard(sub_q)

    def handle_control(self, raw_payload: Any) -> dict[str, Any]:
        """Validate and dispatch a live control command."""
        cmd = _validate_control_payload(
            raw_payload,
            room_width=self.room_width,
            room_height=self.room_height,
        )
        with self._lock:
            handler = self._control_handler
        if handler is None:
            raise ValueError("Live control handler is not enabled on this server.")
        applied = handler(cmd)
        return dict(applied) if isinstance(applied, Mapping) else {"status": "applied"}

    def close(self) -> None:
        """Mark hub closed and wake all SSE subscribers immediately."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            subs = list(self._subscribers)
            self._subscribers.clear()

        for sub_q in subs:
            while True:
                try:
                    sub_q.put_nowait(None)
                    break
                except queue.Full:
                    with contextlib.suppress(queue.Empty):
                        sub_q.get_nowait()


class _RadarHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        hub: TomographyStateHub,
        static_dir: Path,
        keepalive_s: float = 10.0,
    ) -> None:
        self.hub = hub
        self.static_dir = static_dir
        self.keepalive_s = max(0.2, float(keepalive_s))
        super().__init__(server_address, handler_cls)


class _RadarRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _RadarHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_security_headers(self, *, is_html: bool = False) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header(
            "Permissions-Policy",
            "camera=(), geolocation=(), microphone=(), focus-without-user-activation=()",
        )
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin-allow-popups")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Vary", "Sec-Fetch-Dest, Sec-Fetch-Mode, Sec-Fetch-Site")
        if is_html:
            self.send_header("Content-Security-Policy", _HTML_CSP)

    def _check_fetch_metadata(self, *, allow_navigation_get: bool = False) -> bool:
        """Enforce Fetch Metadata resource isolation (reject cross-site API requests)."""
        site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if not site or site in {"same-origin", "same-site", "none"}:
            return True
        if allow_navigation_get and self.command == "GET":
            mode = (self.headers.get("Sec-Fetch-Mode") or "").strip().lower()
            dest = (self.headers.get("Sec-Fetch-Dest") or "").strip().lower()
            if (
                site == "cross-site"
                and mode == "navigate"
                and dest
                not in {
                    "object",
                    "embed",
                }
            ):
                return True
        self._send_json(
            HTTPStatus.FORBIDDEN,
            {"ok": False, "error": "Cross-site request rejected by resource policy."},
        )
        return False

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        is_html: bool = False,
        cache_control: str = "no-cache",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self._send_security_headers(is_html=is_html)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            if not self._check_fetch_metadata(allow_navigation_get=True):
                return
            index_path = self.server.static_dir / "index.html"
            try:
                body = index_path.read_bytes()
            except OSError:
                body = _FALLBACK_INDEX_HTML
            self._send_bytes(
                HTTPStatus.OK,
                body,
                "text/html; charset=utf-8",
                is_html=True,
                cache_control="no-cache",
            )
            return

        if path.startswith("/static/"):
            if not self._check_fetch_metadata(allow_navigation_get=False):
                return
            asset_name = path[len("/static/") :]
            if (
                "/" in asset_name
                or "\\" in asset_name
                or ".." in asset_name
                or asset_name not in ALLOWED_STATIC_ASSETS
            ):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Asset not found."})
                return
            asset_path = self.server.static_dir / asset_name
            try:
                asset_bytes = asset_path.read_bytes()
            except OSError:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Asset not found."})
                return
            self._send_bytes(
                HTTPStatus.OK,
                asset_bytes,
                ALLOWED_STATIC_ASSETS[asset_name],
                cache_control="no-cache",
            )
            return

        if path == "/api/health":
            if not self._check_fetch_metadata(allow_navigation_get=False):
                return
            self._send_json(HTTPStatus.OK, self.server.hub.get_health_dict())
            return

        if path == "/api/config":
            if not self._check_fetch_metadata(allow_navigation_get=False):
                return
            cfg_bytes = self.server.hub.get_config_json().encode("utf-8")
            self._send_bytes(
                HTTPStatus.OK, cfg_bytes, "application/json; charset=utf-8"
            )
            return

        if path == "/api/frame":
            if not self._check_fetch_metadata(allow_navigation_get=False):
                return
            _seq, latest = self.server.hub.get_latest_frame_json()
            if latest is None:
                self._send_json(
                    HTTPStatus.OK,
                    {"status": "waiting", "passive_rx_only": True},
                )
            else:
                self._send_bytes(
                    HTTPStatus.OK,
                    latest.encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            return

        if path == "/api/stream":
            if not self._check_fetch_metadata(allow_navigation_get=False):
                return
            self._handle_sse_stream()
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/control":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return

        if not self._check_fetch_metadata(allow_navigation_get=False):
            return

        ctype = (
            (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        )
        if ctype != "application/json":
            self._send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"ok": False, "error": "Content-Type must be application/json."},
            )
            return

        raw_len = self.headers.get("Content-Length")
        try:
            content_length = int(raw_len) if raw_len is not None else -1
        except ValueError:
            content_length = -1

        if content_length <= 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Request body is required."},
            )
            return
        if content_length > MAX_CONTROL_BODY_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"ok": False, "error": "Control payload exceeds 16 KB limit."},
            )
            return

        try:
            raw_bytes = self.rfile.read(content_length)
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Malformed JSON payload."},
            )
            return

        try:
            applied = self.server.hub.handle_control(payload)
        except (TypeError, ValueError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": str(exc)},
            )
            return

        self._send_json(HTTPStatus.OK, {"ok": True, "applied": applied})

    def _handle_sse_stream(self) -> None:
        hub = self.server.hub
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._send_security_headers(is_html=False)
        self.end_headers()

        if self.command == "HEAD":
            return

        sub_q = hub.subscribe()
        try:
            # Initial config event + latest frame snapshot (if available)
            cfg_msg = f"event: config\ndata: {hub.get_config_json()}\n\n".encode()
            self.wfile.write(cfg_msg)
            last_seq, latest = hub.get_latest_frame_json()
            if latest is not None:
                init_msg = f"id: {last_seq}\nevent: frame\ndata: {latest}\n\n".encode()
                self.wfile.write(init_msg)
            self.wfile.flush()

            while not hub.is_closed:
                try:
                    item = sub_q.get(timeout=self.server.keepalive_s)
                except queue.Empty:
                    if hub.is_closed:
                        break
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue

                if item is None:
                    break
                seq, frame_json = item
                if seq <= last_seq:
                    continue
                last_seq = seq
                chunk = f"id: {seq}\nevent: frame\ndata: {frame_json}\n\n".encode()
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            hub.unsubscribe(sub_q)
            self.close_connection = True


class RealtimeWebServer:
    """Background threaded HTTP + SSE server serving live tomography data."""

    def __init__(
        self,
        hub: TomographyStateHub,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        static_dir: Path | None = None,
        keepalive_s: float = 10.0,
    ) -> None:
        self.hub = hub
        self.host = str(host)
        self.requested_port = int(port)
        self.static_dir = Path(static_dir) if static_dir is not None else STATIC_DIR
        self._httpd = _RadarHTTPServer(
            (self.host, self.requested_port),
            _RadarRequestHandler,
            hub=self.hub,
            static_dir=self.static_dir,
            keepalive_s=keepalive_s,
        )
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        host_disp = "127.0.0.1" if self.host in {"0.0.0.0", ""} else self.host
        return f"http://{host_disp}:{self.port}/"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="hackrf-radar-web",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.hub.close()
        with contextlib.suppress(OSError, RuntimeError):
            self._httpd.shutdown()
        with contextlib.suppress(OSError, RuntimeError):
            self._httpd.server_close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class WebRadarDashboard:
    """Realtime browser dashboard implementing the ``hackrf-radar`` plotter protocol."""

    def __init__(
        self,
        *,
        room_width: float = 6.0,
        room_height: float = 5.0,
        anchor_positions: Mapping[str, tuple[float, float]] | None = None,
        anchor_labels: Mapping[str, str] | None = None,
        show_ground_truth: bool = True,
        host: str = "127.0.0.1",
        port: int = 8765,
        config_summary: Mapping[str, Any] | None = None,
        control_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        sim_interactive: bool = False,
        static_dir: Path | None = None,
    ) -> None:
        self.room_width = float(room_width)
        self.room_height = float(room_height)
        self.anchor_positions = dict(anchor_positions or {})
        self.anchor_labels = dict(anchor_labels or {})
        self.show_ground_truth = bool(show_ground_truth)
        self._history = TrackHistoryBuffer()
        self._last_render_ms: float = 0.0
        self._last_publish_t: float | None = None
        self._ema_fps: float = 0.0
        self._last_frame: Any = None
        self._last_gt: Sequence[Mapping[str, Any]] | None = None
        self._closed = False

        self.hub = TomographyStateHub(
            room_width=self.room_width,
            room_height=self.room_height,
            config_summary=config_summary,
            control_handler=control_handler,
            sim_interactive=sim_interactive,
        )
        self.server = RealtimeWebServer(
            self.hub,
            host=host,
            port=port,
            static_dir=static_dir,
        )
        self.server.start()

    @property
    def url(self) -> str:
        return self.server.url

    @property
    def port(self) -> int:
        return self.server.port

    @property
    def last_render_ms(self) -> float:
        return self._last_render_ms

    def set_control_handler(
        self,
        handler: Callable[[dict[str, Any]], dict[str, Any]] | None,
        *,
        sim_interactive: bool | None = None,
    ) -> None:
        self.hub.set_control_handler(handler, sim_interactive=sim_interactive)

    def update(
        self,
        frame: Any,
        ground_truth: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        if self._closed:
            return 0.0
        t0 = time.perf_counter()
        now_mono = time.monotonic()
        if self._last_publish_t is not None:
            dt = max(now_mono - self._last_publish_t, 1e-4)
            inst_fps = min(120.0, 1.0 / dt)
            self._ema_fps = (
                inst_fps
                if self._ema_fps <= 0.0
                else (0.75 * self._ema_fps + 0.25 * inst_fps)
            )
        self._last_publish_t = now_mono
        self._last_frame = frame
        self._last_gt = ground_truth

        frame_json = serialize_frame_json(
            frame,
            room_width=self.room_width,
            room_height=self.room_height,
            anchor_positions=self.anchor_positions,
            anchor_labels=self.anchor_labels,
            ground_truth=ground_truth,
            show_ground_truth=self.show_ground_truth,
            track_history=self._history,
            frame_seq=self.hub.frame_seq + 1,
            server_fps=self._ema_fps,
        )
        self.hub.publish_frame_json(frame_json)
        self._last_render_ms = (time.perf_counter() - t0) * 1000.0
        return self._last_render_ms

    def save_snapshot(self, path: str | Path, dpi: int = 120) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".json":
            _seq, latest = self.hub.get_latest_frame_json()
            out.write_text(latest or "{}", encoding="utf-8")
            return out
        if self._last_frame is not None:
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
        self.server.close()
