"""Configuration models and passive 0 dBm RX-only safety locks for HackRF Wi-Fi Radar."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_24GHZ_CHANNELS: frozenset[int] = frozenset(range(1, 15))
VALID_5GHZ_CHANNELS: frozenset[int] = frozenset(
    {
        36,
        40,
        44,
        48,
        52,
        56,
        60,
        64,
        100,
        104,
        108,
        112,
        116,
        120,
        124,
        128,
        132,
        136,
        140,
        144,
        149,
        153,
        157,
        161,
        165,
    }
)
VALID_WIFI_CHANNELS: frozenset[int] = VALID_24GHZ_CHANNELS | VALID_5GHZ_CHANNELS

VALID_LNA_STEPS_DB: tuple[int, ...] = (0, 8, 16, 24, 32, 40)
VALID_SOLVER_METHODS: frozenset[str] = frozenset({"fista_tv", "nnls", "sart"})
VALID_RX_BACKENDS: frozenset[str] = frozenset(
    {
        "auto",
        "soapy",
        "libhackrf",
        "osmosdr",
        "hackrf_transfer",
        "gr_zmq",
        "pcap",
        "iq_file",
    }
)


_HEX_CHARS: frozenset[str] = frozenset("0123456789ABCDEF")


def canonicalize_bssid(mac: str) -> str:
    """Normalize a 48-bit IEEE 802.11 MAC/BSSID string to uppercase colon-hex."""
    cleaned = str(mac).strip().upper()
    if "-" in cleaned and ":" not in cleaned:
        cleaned = cleaned.replace("-", ":")
    parts = cleaned.split(":")
    if len(parts) == 6 and all(
        1 <= len(p) <= 2 and all(ch in _HEX_CHARS for ch in p) for p in parts
    ):
        return ":".join(f"{int(p, 16):02X}" for p in parts)
    return cleaned


@dataclass(slots=True)
class RoomAnchor:
    """Known ambient Wi-Fi AP or IoT transmitter anchor coordinate in room meters."""

    bssid: str
    x: float
    y: float
    label: str = ""
    channel: int | None = None
    role: str = "ap"  # "ap" | "iot" | "client"

    def __post_init__(self) -> None:
        self.bssid = canonicalize_bssid(self.bssid)
        self.x = float(self.x)
        self.y = float(self.y)
        if self.channel is not None:
            self.channel = int(self.channel)
            if self.channel not in VALID_WIFI_CHANNELS:
                raise ValueError(
                    f"Unsupported 802.11 channel {self.channel} for anchor {self.bssid}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bssid": self.bssid,
            "x": self.x,
            "y": self.y,
            "label": self.label,
            "channel": self.channel,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoomAnchor:
        pos = data.get("pos")
        x = data["x"] if "x" in data else pos[0]  # type: ignore[index]
        y = data["y"] if "y" in data else pos[1]  # type: ignore[index]
        return cls(
            bssid=str(data["bssid"]),
            x=float(x),
            y=float(y),
            label=str(data.get("label", "")),
            channel=int(data["channel"]) if data.get("channel") is not None else None,
            role=str(data.get("role", "ap")),
        )


@dataclass
class RxSettings:
    """Strictly passive (0 dBm RX-only) HackRF One front-end and channel-hopping parameters.

    Hardware safety locks:
    - ``amp_enable`` is permanently locked to ``False`` (MGA-81563 +14 dB RF amp off).
    - ``bias_tee_enable`` is permanently locked to ``False`` (antenna port DC power off).
    - ``lna_db`` must be in 0..40 dB (8 dB steps) and ``vga_db`` in 0..62 dB (2 dB steps).
    """

    rx_pos: tuple[float, float] = (3.0, 0.35)
    channels: list[int] = field(default_factory=lambda: [1, 6, 11])
    dwell_ms: float = 80.0
    sample_rate_hz: float = 20e6
    lna_db: int = 24
    vga_db: int = 20
    amp_enable: bool = False
    bias_tee_enable: bool = False
    dc_offset_removal: bool = True
    pll_settle_ms: float = 3.0
    backend: str = "auto"
    zmq_endpoint: str | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "amp_enable" and bool(value) is not False:
            raise ValueError(
                "Passive 0 dBm RX-only safety lock: HackRF RF AMP (+14 dB) must remain False."
            )
        if name == "bias_tee_enable" and bool(value) is not False:
            raise ValueError(
                "Passive 0 dBm RX-only safety lock: HackRF antenna bias-tee must remain False."
            )
        if name == "tx_enable" and bool(value):
            raise ValueError(
                "Passive 0 dBm RX-only safety lock: TX mode is strictly forbidden."
            )
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        if self.amp_enable is not False:
            raise ValueError(
                "Passive 0 dBm RX-only safety lock: HackRF RF AMP (+14 dB) must remain False."
            )
        if self.bias_tee_enable is not False:
            raise ValueError(
                "Passive 0 dBm RX-only safety lock: HackRF antenna bias-tee must remain False."
            )
        self.rx_pos = (float(self.rx_pos[0]), float(self.rx_pos[1]))
        self.channels = [int(ch) for ch in self.channels]
        if not self.channels:
            raise ValueError("At least one Wi-Fi channel must be configured.")
        for ch in self.channels:
            if ch not in VALID_WIFI_CHANNELS:
                raise ValueError(f"Invalid 802.11 channel: {ch}")

        self.dwell_ms = float(self.dwell_ms)
        if self.dwell_ms <= 0.0:
            raise ValueError(f"dwell_ms must be positive, got {self.dwell_ms}")

        self.sample_rate_hz = float(self.sample_rate_hz)
        if not (2e6 <= self.sample_rate_hz <= 20e6):
            raise ValueError(
                f"HackRF sample_rate_hz must be in [2e6, 20e6], got {self.sample_rate_hz}"
            )

        self.lna_db = int(self.lna_db)
        if self.lna_db < 0 or self.lna_db > 40:
            raise ValueError(f"HackRF LNA gain must be in 0..40 dB, got {self.lna_db}")
        # Snap to nearest 8 dB MAX2837 IF LNA step if intermediate value supplied
        if self.lna_db not in VALID_LNA_STEPS_DB:
            self.lna_db = min(VALID_LNA_STEPS_DB, key=lambda g: abs(g - self.lna_db))

        self.vga_db = int(self.vga_db)
        if self.vga_db < 0 or self.vga_db > 62:
            raise ValueError(
                f"HackRF baseband VGA gain must be in 0..62 dB, got {self.vga_db}"
            )
        if self.vga_db % 2 != 0:
            self.vga_db = (self.vga_db // 2) * 2

        self.pll_settle_ms = float(self.pll_settle_ms)
        if not (0.5 <= self.pll_settle_ms <= 25.0):
            raise ValueError(
                f"pll_settle_ms must be in [0.5, 25.0], got {self.pll_settle_ms}"
            )

        self.backend = str(self.backend).lower()
        if self.backend not in VALID_RX_BACKENDS:
            raise ValueError(
                f"Unsupported RX backend {self.backend!r}; expected one of {sorted(VALID_RX_BACKENDS)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rx_pos": [self.rx_pos[0], self.rx_pos[1]],
            "channels": list(self.channels),
            "dwell_ms": self.dwell_ms,
            "sample_rate_hz": self.sample_rate_hz,
            "lna_db": self.lna_db,
            "vga_db": self.vga_db,
            "amp_enable": False,
            "bias_tee_enable": False,
            "dc_offset_removal": self.dc_offset_removal,
            "pll_settle_ms": self.pll_settle_ms,
            "backend": self.backend,
            "zmq_endpoint": self.zmq_endpoint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RxSettings:
        if data.get("tx_enable"):
            raise ValueError(
                "Passive 0 dBm RX-only safety lock: tx_enable is forbidden."
            )
        rx_pos_raw = data.get("rx_pos", (3.0, 0.35))
        return cls(
            rx_pos=(float(rx_pos_raw[0]), float(rx_pos_raw[1])),
            channels=[int(c) for c in data.get("channels", [1, 6, 11])],
            dwell_ms=float(data.get("dwell_ms", 80.0)),
            sample_rate_hz=float(data.get("sample_rate_hz", 20e6)),
            lna_db=int(data.get("lna_db", 24)),
            vga_db=int(data.get("vga_db", 20)),
            amp_enable=bool(data.get("amp_enable", False)),
            bias_tee_enable=bool(data.get("bias_tee_enable", False)),
            dc_offset_removal=bool(data.get("dc_offset_removal", True)),
            pll_settle_ms=float(data.get("pll_settle_ms", 3.0)),
            backend=str(data.get("backend", "auto")),
            zmq_endpoint=data.get("zmq_endpoint"),
        )


@dataclass(slots=True)
class AutoDiscoverySettings:
    """Automatic room-perimeter placement settings for uncalibrated ambient BSSIDs."""

    enabled: bool = True
    min_packets: int = 3
    perimeter_margin_m: float = 0.25
    use_rssi_log_distance: bool = True
    ref_rssi_1m_dbm: float = -40.0
    path_loss_exponent: float = 2.4
    enable_sifs_peer_chords: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_packets": self.min_packets,
            "perimeter_margin_m": self.perimeter_margin_m,
            "use_rssi_log_distance": self.use_rssi_log_distance,
            "ref_rssi_1m_dbm": self.ref_rssi_1m_dbm,
            "path_loss_exponent": self.path_loss_exponent,
            "enable_sifs_peer_chords": self.enable_sifs_peer_chords,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AutoDiscoverySettings:
        return cls(
            enabled=bool(data.get("enabled", True)),
            min_packets=int(data.get("min_packets", 3)),
            perimeter_margin_m=float(data.get("perimeter_margin_m", 0.25)),
            use_rssi_log_distance=bool(data.get("use_rssi_log_distance", True)),
            ref_rssi_1m_dbm=float(data.get("ref_rssi_1m_dbm", -40.0)),
            path_loss_exponent=float(data.get("path_loss_exponent", 2.4)),
            enable_sifs_peer_chords=bool(data.get("enable_sifs_peer_chords", True)),
        )


@dataclass(slots=True)
class SolverSettings:
    """Tomographic inversion grid, regularization method, and frame rate settings."""

    grid_shape: tuple[int, int] = (40, 34)
    solver_method: str = "fista_tv"
    window_s: float = 1.5
    calibration_s: float = 0.0
    target_fps: float = 20.0

    def __post_init__(self) -> None:
        nx, ny = int(self.grid_shape[0]), int(self.grid_shape[1])
        if nx < 8 or ny < 8:
            raise ValueError(f"grid_shape must be at least (8, 8), got {(nx, ny)}")
        self.grid_shape = (nx, ny)
        self.solver_method = str(self.solver_method).lower()
        if self.solver_method not in VALID_SOLVER_METHODS:
            raise ValueError(
                f"Invalid solver_method {self.solver_method!r}; expected one of {sorted(VALID_SOLVER_METHODS)}"
            )
        self.window_s = float(self.window_s)
        if self.window_s <= 0.1:
            raise ValueError(f"window_s must be > 0.1s, got {self.window_s}")
        self.calibration_s = max(0.0, float(self.calibration_s))
        self.target_fps = max(1.0, float(self.target_fps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_shape": [self.grid_shape[0], self.grid_shape[1]],
            "solver_method": self.solver_method,
            "window_s": self.window_s,
            "calibration_s": self.calibration_s,
            "target_fps": self.target_fps,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SolverSettings:
        gs = data.get("grid_shape", (40, 34))
        return cls(
            grid_shape=(int(gs[0]), int(gs[1])),
            solver_method=str(data.get("solver_method", "fista_tv")),
            window_s=float(data.get("window_s", 1.5)),
            calibration_s=float(data.get("calibration_s", 0.0)),
            target_fps=float(data.get("target_fps", 20.0)),
        )


@dataclass
class RadarConfig:
    """Complete room geometry, HackRF RX, anchor layout, and RTI solver configuration."""

    room_name: str = "6m x 5m Multi-AP Smart Office"
    room_width: float = 6.0
    room_height: float = 5.0
    rx: RxSettings = field(default_factory=RxSettings)
    anchors: list[RoomAnchor] = field(default_factory=list)
    auto_discovery: AutoDiscoverySettings = field(default_factory=AutoDiscoverySettings)
    solver: SolverSettings = field(default_factory=SolverSettings)

    def __post_init__(self) -> None:
        self.room_width = float(self.room_width)
        self.room_height = float(self.room_height)
        if self.room_width <= 0.5 or self.room_height <= 0.5:
            raise ValueError(
                f"Room dimensions must exceed 0.5m x 0.5m, got {self.room_width} x {self.room_height}"
            )

    def known_tx_positions(self) -> dict[str, tuple[float, float]]:
        """Return mapping of canonical BSSID -> (x_m, y_m) for configured anchors."""
        return {a.bssid: (a.x, a.y) for a in self.anchors}

    def anchor_labels(self) -> dict[str, str]:
        """Return mapping of canonical BSSID -> human-readable anchor label."""
        return {a.bssid: (a.label or a.bssid[-5:]) for a in self.anchors}

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_name": self.room_name,
            "room_width": self.room_width,
            "room_height": self.room_height,
            "rx": self.rx.to_dict(),
            "anchors": [a.to_dict() for a in self.anchors],
            "auto_discovery": self.auto_discovery.to_dict(),
            "solver": self.solver.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RadarConfig:
        rx_dict = dict(data.get("rx", {}))
        if "rx_pos" in data and "rx_pos" not in rx_dict:
            rx_dict["rx_pos"] = data["rx_pos"]
        if "channels" in data and "channels" not in rx_dict:
            rx_dict["channels"] = data["channels"]
        anchors_raw: Sequence[Mapping[str, Any]] = data.get("anchors", [])
        return cls(
            room_name=str(data.get("room_name", "6m x 5m Multi-AP Smart Office")),
            room_width=float(data.get("room_width", 6.0)),
            room_height=float(data.get("room_height", 5.0)),
            rx=RxSettings.from_dict(rx_dict),
            anchors=[RoomAnchor.from_dict(item) for item in anchors_raw],
            auto_discovery=AutoDiscoverySettings.from_dict(
                data.get("auto_discovery", {})
            ),
            solver=SolverSettings.from_dict(data.get("solver", {})),
        )

    @classmethod
    def default_example_room(cls) -> RadarConfig:
        """Build the default 6m x 5m multi-AP + IoT perimeter room configuration."""
        return cls(
            room_name="6m x 5m Multi-AP + IoT Lab",
            room_width=6.0,
            room_height=5.0,
            rx=RxSettings(
                rx_pos=(3.0, 0.35),
                channels=[1, 6, 11],
                dwell_ms=80.0,
                sample_rate_hz=20e6,
                lna_db=24,
                vga_db=20,
                amp_enable=False,
                bias_tee_enable=False,
                backend="auto",
            ),
            anchors=[
                RoomAnchor(
                    "AA:BB:CC:00:00:01",
                    0.40,
                    4.70,
                    "NW Ceiling AP-1",
                    channel=1,
                    role="ap",
                ),
                RoomAnchor(
                    "AA:BB:CC:00:00:02",
                    3.00,
                    4.75,
                    "North Mesh AP-2",
                    channel=6,
                    role="ap",
                ),
                RoomAnchor(
                    "AA:BB:CC:00:00:03",
                    5.60,
                    4.70,
                    "NE Ceiling AP-3",
                    channel=11,
                    role="ap",
                ),
                RoomAnchor(
                    "AA:BB:CC:00:00:04",
                    0.25,
                    2.60,
                    "West Wall Thermostat",
                    channel=6,
                    role="iot",
                ),
                RoomAnchor(
                    "AA:BB:CC:00:00:05",
                    5.75,
                    2.60,
                    "East Shelf Speaker",
                    channel=1,
                    role="iot",
                ),
                RoomAnchor(
                    "AA:BB:CC:00:00:06",
                    0.35,
                    1.15,
                    "SW Desk Printer",
                    channel=11,
                    role="iot",
                ),
                RoomAnchor(
                    "AA:BB:CC:00:00:07",
                    5.65,
                    1.15,
                    "SE Bench Camera",
                    channel=6,
                    role="iot",
                ),
                RoomAnchor(
                    "AA:BB:CC:00:00:08",
                    1.70,
                    4.75,
                    "North Smart Plug",
                    channel=11,
                    role="iot",
                ),
            ],
            auto_discovery=AutoDiscoverySettings(
                enabled=True,
                min_packets=3,
                perimeter_margin_m=0.25,
                use_rssi_log_distance=True,
                ref_rssi_1m_dbm=-40.0,
                path_loss_exponent=2.4,
                enable_sifs_peer_chords=True,
            ),
            solver=SolverSettings(
                grid_shape=(40, 34),
                solver_method="fista_tv",
                window_s=1.5,
                calibration_s=0.0,
                target_fps=20.0,
            ),
        )


def load_config(path: str | Path | None = None) -> RadarConfig:
    """Load a RadarConfig from a JSON file, or return the default 6m x 5m room config."""
    if path is None:
        default_path = (
            Path(__file__).resolve().parent.parent / "configs" / "example_room.json"
        )
        if default_path.is_file():
            data = json.loads(default_path.read_text(encoding="utf-8"))
            return RadarConfig.from_dict(data)
        return RadarConfig.default_example_room()

    cfg_path = Path(path)
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return RadarConfig.from_dict(data)


def save_config(cfg: RadarConfig, path: str | Path) -> None:
    """Serialize a RadarConfig to formatted UTF-8 JSON."""
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
