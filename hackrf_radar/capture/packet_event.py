"""Canonical passive 802.11 packet & CSI event data model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# 52 active IEEE 802.11a/g/n (20 MHz non-HT L-LTF) subcarriers: [-26..-1, +1..+26]
SUBCARRIER_INDICES: tuple[int, ...] = tuple(list(range(-26, 0)) + list(range(1, 27)))

# 4 BPSK pilot subcarriers within the 52 active subcarriers
PILOT_INDICES: tuple[int, ...] = (-21, -7, 7, 21)

# Canonical 20 MHz Wi-Fi channel center frequencies (Hz): 2.4 GHz (1..14) + 5 GHz UNII (36..165)
WIFI_CHANNELS_HZ: dict[int, float] = {
    # 2.4 GHz ISM band
    1: 2412e6,
    2: 2417e6,
    3: 2422e6,
    4: 2427e6,
    5: 2432e6,
    6: 2437e6,
    7: 2442e6,
    8: 2447e6,
    9: 2452e6,
    10: 2457e6,
    11: 2462e6,
    12: 2467e6,
    13: 2472e6,
    14: 2484e6,
    # 5 GHz UNII-1
    36: 5180e6,
    40: 5200e6,
    44: 5220e6,
    48: 5240e6,
    # 5 GHz UNII-2A
    52: 5260e6,
    56: 5280e6,
    60: 5300e6,
    64: 5320e6,
    # 5 GHz UNII-2C (DFS)
    100: 5500e6,
    104: 5520e6,
    108: 5540e6,
    112: 5560e6,
    116: 5580e6,
    120: 5600e6,
    124: 5620e6,
    128: 5640e6,
    132: 5660e6,
    136: 5680e6,
    140: 5700e6,
    144: 5720e6,
    # 5 GHz UNII-3 / ISM
    149: 5745e6,
    153: 5765e6,
    157: 5785e6,
    161: 5805e6,
    165: 5825e6,
}

_SUBCARRIER_K_F64: np.ndarray = np.array(SUBCARRIER_INDICES, dtype=np.float64)

VALID_FRAME_TYPES: frozenset[str] = frozenset(
    {
        "BEACON",
        "PROBE_REQ",
        "PROBE_RESP",
        "CTS",
        "ACK",
        "BLOCK_ACK",
        "QOS_NULL",
        "DATA_PREAMBLE",
        "FCS_ERROR",
    }
)


def channel_to_freq_hz(channel: int) -> float:
    """Return center frequency in Hz for a 2.4 GHz or 5 GHz Wi-Fi channel number."""
    ch = int(channel)
    if ch in WIFI_CHANNELS_HZ:
        return WIFI_CHANNELS_HZ[ch]
    if 1 <= ch <= 13:
        return 2407e6 + 5e6 * ch
    if 32 <= ch <= 177:
        return 5000e6 + 5e6 * ch
    raise ValueError(f"Unsupported Wi-Fi channel number: {channel}")


def freq_hz_to_channel(freq_hz: float) -> int:
    """Return nearest canonical Wi-Fi channel number for a center frequency in Hz."""
    f = float(freq_hz)
    best_ch = min(WIFI_CHANNELS_HZ.keys(), key=lambda c: abs(WIFI_CHANNELS_HZ[c] - f))
    if abs(WIFI_CHANNELS_HZ[best_ch] - f) <= 12e6:
        return best_ch
    raise ValueError(
        f"Frequency {freq_hz} Hz does not match a known 2.4/5 GHz Wi-Fi channel"
    )


def sanitize_csi(csi: np.ndarray) -> np.ndarray:
    """Remove per-packet STO/PDD linear phase slope and random CFO/LO phase offset.

    Unwraps the 52 active subcarriers across the DC null (k = -1 -> +1) and
    subtracts a magnitude-weighted linear phase regression ``alpha + beta * k``,
    preserving multipath/bistatic amplitude and non-linear subcarrier phase
    structure while eliminating per-packet timing and oscillator phase nuisance.
    """
    h = np.asarray(csi, dtype=np.complex128).reshape(-1)
    if h.shape[0] != 52:
        raise ValueError(f"Expected 52 active subcarriers, got shape {csi.shape}")

    mag = np.abs(h)
    max_mag = float(np.max(mag))
    if max_mag < 1e-12 or not np.all(np.isfinite(h)):
        return np.zeros(52, dtype=np.complex64)

    raw_phase = np.angle(h)
    # Unwrap lower [-26..-1] and upper [+1..+26] half-bands independently first
    lower = np.unwrap(raw_phase[:26])
    upper = np.unwrap(raw_phase[26:])

    # Estimate single-bin slope (dk = 1) to bridge the 2-bin DC gap cleanly
    diffs = np.concatenate([np.diff(lower), np.diff(upper)])
    edge_weights = np.concatenate([mag[:25] * mag[1:26], mag[26:51] * mag[27:52]])
    valid_edges = edge_weights > (1e-4 * max_mag * max_mag)
    if np.any(valid_edges):
        est_slope = float(np.median(diffs[valid_edges]))
    else:
        est_slope = float(np.median(diffs))

    predicted_upper0 = lower[-1] + 2.0 * est_slope
    two_pi = 2.0 * np.pi
    wrap_shift = two_pi * np.round((upper[0] - predicted_upper0) / two_pi)
    upper = upper - wrap_shift
    unwrapped = np.concatenate([lower, upper])

    # Magnitude-squared weighted linear regression: phase ~ alpha + beta * k
    w = (mag / max_mag) ** 2 + 1e-6
    w_sum = float(np.sum(w))
    k = _SUBCARRIER_K_F64
    k_mean = float(np.sum(w * k) / w_sum)
    p_mean = float(np.sum(w * unwrapped) / w_sum)
    dk = k - k_mean
    denom = float(np.sum(w * dk * dk))
    beta = (
        float(np.sum(w * dk * (unwrapped - p_mean)) / denom) if denom > 1e-12 else 0.0
    )
    alpha = p_mean - beta * k_mean

    sanitized_phase = unwrapped - (alpha + beta * k)
    out = mag * np.exp(1j * sanitized_phase)
    return out.astype(np.complex64)


def _decode_complex_array(val: Any) -> np.ndarray:
    """Convert serialized CSI representations back into complex64 shape (52,)."""
    if isinstance(val, np.ndarray):
        arr = val.astype(np.complex64, copy=False).reshape(-1)
    elif isinstance(val, dict) and "real" in val and "imag" in val:
        real = np.asarray(val["real"], dtype=np.float32)
        imag = np.asarray(val["imag"], dtype=np.float32)
        arr = (real + 1j * imag).astype(np.complex64).reshape(-1)
    elif isinstance(val, (list, tuple)):
        if (
            len(val) == 52
            and len(val) > 0
            and isinstance(val[0], (list, tuple))
            and len(val[0]) == 2
        ):
            pairs = np.asarray(val, dtype=np.float32)
            arr = (pairs[:, 0] + 1j * pairs[:, 1]).astype(np.complex64)
        else:
            arr = np.asarray([complex(x) for x in val], dtype=np.complex64).reshape(-1)
    else:
        raise TypeError(f"Unsupported serialized CSI format: {type(val)!r}")

    if arr.shape != (52,):
        raise ValueError(f"CSI array must have shape (52,), got {arr.shape}")
    return arr


@dataclass(slots=True)
class PacketEvent:
    """Single passively captured 802.11 PHY/MAC packet observation."""

    timestamp: float  # seconds
    freq_hz: float  # e.g. 2437e6
    channel: int  # e.g. 6
    bssid: str  # canonical TX MAC ("AA:BB:CC:00:00:01"), SIFS-attributed for CTS/ACK
    receiver_addr: str | None  # Addr1 (RA) when decoded
    frame_type: str  # "BEACON"|"PROBE_REQ"|"PROBE_RESP"|"CTS"|"ACK"|"BLOCK_ACK"|"QOS_NULL"|"DATA_PREAMBLE"|"FCS_ERROR"
    rssi_dbm: float
    noise_floor_dbm: float
    snr_db: float
    csi: np.ndarray  # complex64 shape (52,) raw L-LTF H[k]
    csi_sanitized: (
        np.ndarray
    )  # complex64 shape (52,) STO/PDD slope & random phase sanitized
    cfo_hz: float
    tsf_us: int | None  # 64-bit AP TSF microseconds (Beacon/ProbeResp)
    tsf_drift_ppm: float | None  # per-BSSID TSF clock drift estimate (ppm)
    seq_num: int | None  # 12-bit 802.11 sequence number (0..4095)
    seq_gap: int  # missed frames >= 0
    fcs_ok: bool  # IEEE 802.11 CRC-32 check
    adc_clip_ratio: float  # fraction of int8 samples with |I|>=120 or |Q|>=120
    attributed_via_sifs: bool = False

    def __post_init__(self) -> None:
        self.csi = np.asarray(self.csi, dtype=np.complex64).reshape(-1)
        self.csi_sanitized = np.asarray(self.csi_sanitized, dtype=np.complex64).reshape(
            -1
        )
        if self.csi.shape != (52,):
            raise ValueError(
                f"PacketEvent.csi must have shape (52,), got {self.csi.shape}"
            )
        if self.csi_sanitized.shape != (52,):
            raise ValueError(
                f"PacketEvent.csi_sanitized must have shape (52,), got {self.csi_sanitized.shape}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize PacketEvent into a JSON-compatible dictionary."""
        return {
            "timestamp": float(self.timestamp),
            "freq_hz": float(self.freq_hz),
            "channel": int(self.channel),
            "bssid": str(self.bssid),
            "receiver_addr": str(self.receiver_addr)
            if self.receiver_addr is not None
            else None,
            "frame_type": str(self.frame_type),
            "rssi_dbm": float(self.rssi_dbm),
            "noise_floor_dbm": float(self.noise_floor_dbm),
            "snr_db": float(self.snr_db),
            "csi": {
                "real": [float(x) for x in self.csi.real],
                "imag": [float(x) for x in self.csi.imag],
            },
            "csi_sanitized": {
                "real": [float(x) for x in self.csi_sanitized.real],
                "imag": [float(x) for x in self.csi_sanitized.imag],
            },
            "cfo_hz": float(self.cfo_hz),
            "tsf_us": int(self.tsf_us) if self.tsf_us is not None else None,
            "tsf_drift_ppm": float(self.tsf_drift_ppm)
            if self.tsf_drift_ppm is not None
            else None,
            "seq_num": int(self.seq_num) if self.seq_num is not None else None,
            "seq_gap": int(self.seq_gap),
            "fcs_ok": bool(self.fcs_ok),
            "adc_clip_ratio": float(self.adc_clip_ratio),
            "attributed_via_sifs": bool(self.attributed_via_sifs),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PacketEvent:
        """Reconstruct PacketEvent from a dictionary produced by ``to_dict``."""
        if "csi" in d:
            csi = _decode_complex_array(d["csi"])
        elif "csi_real" in d and "csi_imag" in d:
            csi = _decode_complex_array({"real": d["csi_real"], "imag": d["csi_imag"]})
        else:
            raise KeyError("PacketEvent dict missing 'csi'")

        if "csi_sanitized" in d and d["csi_sanitized"] is not None:
            csi_san = _decode_complex_array(d["csi_sanitized"])
        else:
            csi_san = sanitize_csi(csi)

        freq_hz = float(d["freq_hz"])
        channel = (
            int(d["channel"])
            if "channel" in d and d["channel"] is not None
            else freq_hz_to_channel(freq_hz)
        )

        return cls(
            timestamp=float(d["timestamp"]),
            freq_hz=freq_hz,
            channel=channel,
            bssid=str(d["bssid"]),
            receiver_addr=str(d["receiver_addr"])
            if d.get("receiver_addr") is not None
            else None,
            frame_type=str(d["frame_type"]),
            rssi_dbm=float(d["rssi_dbm"]),
            noise_floor_dbm=float(d.get("noise_floor_dbm", -92.0)),
            snr_db=float(d.get("snr_db", float(d["rssi_dbm"]) + 92.0)),
            csi=csi,
            csi_sanitized=csi_san,
            cfo_hz=float(d.get("cfo_hz", 0.0)),
            tsf_us=int(d["tsf_us"]) if d.get("tsf_us") is not None else None,
            tsf_drift_ppm=float(d["tsf_drift_ppm"])
            if d.get("tsf_drift_ppm") is not None
            else None,
            seq_num=int(d["seq_num"]) if d.get("seq_num") is not None else None,
            seq_gap=int(d.get("seq_gap", 0)),
            fcs_ok=bool(d.get("fcs_ok", True)),
            adc_clip_ratio=float(d.get("adc_clip_ratio", 0.0)),
            attributed_via_sifs=bool(d.get("attributed_via_sifs", False)),
        )
