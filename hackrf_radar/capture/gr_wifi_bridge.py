"""GNU Radio 802.11 bridge, Radiotap PCAP reader, and native 20 MSPS OFDM CSI/MAC decoder.

Provides passive (0 dBm RX-only) IEEE 802.11a/g/n/ac/ax cleartext preamble & MAC ingestion:
  1. Live/file 20 MSPS I/Q via ``HackRFIQSource`` -> native NumPy/SciPy OFDM burst detector:
     - Schmidl-Cox L-STF coarse timing & CFO estimation
     - L-LTF dual-symbol fine timing & CFO refinement
     - 64-point FFT 52-subcarrier CSI extraction & STO/PDD phase-slope sanitization
     - Rate-1/2, 2/3, 3/4 K=7 (133_8, 171_8) Viterbi L-SIG & MAC demodulation + CRC-32 FCS
     - SIFS (10/16 us) CTS/ACK peer attribution, 64-bit Beacon TSF drift (ppm), & SeqCtrl gaps
  2. Radiotap / raw 802.11 ``.pcap`` & ``.pcapng`` replay parsed via Python ``struct``
  3. GNU Radio ``gr-ieee802-11`` ZMQ SUB PDU ingestion (``--backend gr_zmq``)
"""

from __future__ import annotations

import contextlib
import json
import math
import struct
import time
import zlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np

from ..config import canonicalize_bssid
from .packet_event import (
    PILOT_INDICES,
    SUBCARRIER_INDICES,
    PacketEvent,
    channel_to_freq_hz,
    freq_hz_to_channel,
    sanitize_csi,
)
from .soapy_source import HackRFIQSource, IQChunk

# ---------------------------------------------------------------------------
# IEEE Std 802.11-2020 Clause 17 20 MHz Non-HT OFDM PHY Constants
# ---------------------------------------------------------------------------

FFT_SIZE: int = 64
CP_LEN: int = 16
SYM_LEN: int = 80  # 16 GI + 64 FFT at 20 MSPS (4.0 us)

ACTIVE_FFT_BINS: np.ndarray = np.array(
    [(k + FFT_SIZE) % FFT_SIZE for k in SUBCARRIER_INDICES], dtype=np.intp
)
PILOT_POS_IN_52: np.ndarray = np.array(
    [SUBCARRIER_INDICES.index(p) for p in PILOT_INDICES], dtype=np.intp
)
DATA_POS_IN_52: np.ndarray = np.array(
    [i for i, k in enumerate(SUBCARRIER_INDICES) if k not in PILOT_INDICES],
    dtype=np.intp,
)

# IEEE 802.11-2020 Eq. 17-6 L-LTF active subcarrier BPSK sequence k in [-26..-1, +1..+26]
L_LTF_SEQ_52: np.ndarray = np.array(
    [
        # k = -26 .. -1
        1,
        1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        1,
        # k = +1 .. +26
        1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        1,
    ],
    dtype=np.float64,
)

# IEEE 802.11-2020 Eq. 17-5 L-STF non-zero subcarriers (scaled by sqrt(13/6))
_LSTF_SPARSE: dict[int, complex] = {
    -24: 1.0 + 1.0j,
    -20: -1.0 - 1.0j,
    -16: 1.0 + 1.0j,
    -12: -1.0 - 1.0j,
    -8: -1.0 - 1.0j,
    -4: 1.0 + 1.0j,
    4: -1.0 - 1.0j,
    8: -1.0 - 1.0j,
    12: 1.0 + 1.0j,
    16: 1.0 + 1.0j,
    20: 1.0 + 1.0j,
    24: 1.0 + 1.0j,
}

BASE_PILOT_SIGNS: np.ndarray = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float64)


def _build_pilot_polarity_table() -> np.ndarray:
    """Generate the 127-element IEEE 802.11 Clause 17.3.5.9 pilot polarity sequence p_v."""
    reg = [1, 1, 1, 1, 1, 1, 1]
    out = np.empty(127, dtype=np.float64)
    for idx in range(127):
        z = reg[3] ^ reg[6]
        out[idx] = 1.0 if z == 0 else -1.0
        reg = [z] + reg[:6]
    return out


PILOT_POLARITY_127: np.ndarray = _build_pilot_polarity_table()


def _build_reference_preamble() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct unitary-scaled 64-sample L-LTF symbol and 160-sample L-STF + L-LTF."""
    x_stf = np.zeros(FFT_SIZE, dtype=np.complex128)
    scale_stf = math.sqrt(13.0 / 6.0)
    for k, val in _LSTF_SPARSE.items():
        x_stf[(k + FFT_SIZE) % FFT_SIZE] = val * scale_stf
    # Unitary IFFT: ifft(X) * sqrt(64) = ifft(X) * 8.0
    stf_64 = np.fft.ifft(x_stf) * 8.0
    stf_160 = np.tile(stf_64[:16], 10)

    x_ltf = np.zeros(FFT_SIZE, dtype=np.complex128)
    x_ltf[ACTIVE_FFT_BINS] = L_LTF_SEQ_52
    ltf_64 = np.fft.ifft(x_ltf) * 8.0
    ltf_160 = np.concatenate([ltf_64[32:], ltf_64, ltf_64])
    return (
        stf_160.astype(np.complex64),
        ltf_160.astype(np.complex64),
        ltf_64.astype(np.complex64),
    )


REF_LSTF_160, REF_LLTF_160, REF_LLTF_64 = _build_reference_preamble()
_REF_LLTF_NORM: float = float(np.sqrt(np.sum(np.abs(REF_LLTF_64) ** 2)))

# L-SIG 4-bit RATE field (R1..R4) -> (mbps, n_bpsc, n_cbps, n_dbps, code_rate)
LSIG_RATE_TABLE: dict[tuple[int, int, int, int], tuple[int, int, int, int, str]] = {
    (1, 1, 0, 1): (6, 1, 48, 24, "1/2"),
    (1, 1, 1, 1): (9, 1, 48, 36, "3/4"),
    (0, 1, 0, 1): (12, 2, 96, 48, "1/2"),
    (0, 1, 1, 1): (18, 2, 96, 72, "3/4"),
    (1, 0, 0, 1): (24, 4, 192, 96, "1/2"),
    (1, 0, 1, 1): (36, 4, 192, 144, "3/4"),
    (0, 0, 0, 1): (48, 6, 288, 192, "2/3"),
    (0, 0, 1, 1): (54, 6, 288, 216, "3/4"),
}
MBPS_TO_RATE_BITS: dict[int, tuple[int, int, int, int]] = {
    v[0]: k for k, v in LSIG_RATE_TABLE.items()
}


def _build_interleaver_tables() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Precompute forward & inverse Clause 17.3.5.7 bit-interleaver permutations."""
    tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n_cbps in (48, 96, 192, 288):
        n_bpsc = n_cbps // 48
        s = max(n_bpsc // 2, 1)
        fwd = np.empty(n_cbps, dtype=np.intp)
        inv = np.empty(n_cbps, dtype=np.intp)
        for k in range(n_cbps):
            i = (n_cbps // 16) * (k % 16) + (k // 16)
            j = s * (i // s) + ((i + n_cbps - (16 * i // n_cbps)) % s)
            fwd[j] = k  # interleaved[j] = coded[k]
            inv[k] = j  # coded[k] = interleaved[j]
        tables[n_cbps] = (fwd, inv)
    return tables


_INTERLEAVER_TABLES = _build_interleaver_tables()


def _popcount7(x: int) -> int:
    return (x & 0x7F).bit_count() & 1


def _build_viterbi_tables() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Precompute 64-state K=7 (133_8, 171_8) Viterbi branch tables."""
    s_next = np.arange(64, dtype=np.intp)
    u_bit = (s_next >> 5) & 1
    prev0 = (s_next & 0x1F) << 1
    prev1 = prev0 | 1

    out0_a = np.empty(64, dtype=np.float64)
    out0_b = np.empty(64, dtype=np.float64)
    out1_a = np.empty(64, dtype=np.float64)
    out1_b = np.empty(64, dtype=np.float64)

    for sn in range(64):
        u = int(u_bit[sn])
        for prev, oa, ob in (
            (int(prev0[sn]), out0_a, out0_b),
            (int(prev1[sn]), out1_a, out1_b),
        ):
            reg7 = (u << 6) | prev
            a = _popcount7(reg7 & 0o133)
            b = _popcount7(reg7 & 0o171)
            oa[sn] = 1.0 if a else -1.0
            ob[sn] = 1.0 if b else -1.0

    return prev0, prev1, out0_a, out0_b, out1_a, out1_b


_VIT_PREV0, _VIT_PREV1, _VIT_O0A, _VIT_O0B, _VIT_O1A, _VIT_O1B = _build_viterbi_tables()


def _conv_encode(bits: np.ndarray, code_rate: str = "1/2") -> np.ndarray:
    """Encode 0/1 bit array with IEEE 802.11 K=7 (133_8, 171_8) convolutional code."""
    n = int(bits.size)
    out_a = np.empty(n, dtype=np.uint8)
    out_b = np.empty(n, dtype=np.uint8)
    state = 0
    for i in range(n):
        u = int(bits[i]) & 1
        reg7 = (u << 6) | state
        out_a[i] = _popcount7(reg7 & 0o133)
        out_b[i] = _popcount7(reg7 & 0o171)
        state = ((u << 5) | (state >> 1)) & 0x3F

    if code_rate == "1/2":
        coded = np.empty(2 * n, dtype=np.uint8)
        coded[0::2] = out_a
        coded[1::2] = out_b
        return coded
    if code_rate == "3/4":
        # Keep [A0, B0, A1, B2] per 3 input bits
        n3 = (n // 3) * 3
        coded = np.empty((n3 // 3) * 4, dtype=np.uint8)
        coded[0::4] = out_a[0:n3:3]
        coded[1::4] = out_b[0:n3:3]
        coded[2::4] = out_a[1:n3:3]
        coded[3::4] = out_b[2:n3:3]
        return coded
    if code_rate == "2/3":
        # Keep [A0, B0, A1] per 2 input bits
        n2 = (n // 2) * 2
        coded = np.empty((n2 // 2) * 3, dtype=np.uint8)
        coded[0::3] = out_a[0:n2:2]
        coded[1::3] = out_b[0:n2:2]
        coded[2::3] = out_a[1:n2:2]
        return coded
    raise ValueError(f"Unsupported convolutional code rate: {code_rate!r}")


def _depuncture_llrs(llrs: np.ndarray, code_rate: str) -> tuple[np.ndarray, np.ndarray]:
    """Expand punctured soft LLRs (positive -> bit 1) into (llr_a, llr_b) pairs."""
    if code_rate == "1/2":
        m = llrs.size // 2
        return llrs[0 : 2 * m : 2], llrs[1 : 2 * m : 2]
    if code_rate == "3/4":
        blocks = llrs.size // 4
        n_out = blocks * 3
        la = np.zeros(n_out, dtype=np.float64)
        lb = np.zeros(n_out, dtype=np.float64)
        src = llrs[: blocks * 4]
        la[0::3] = src[0::4]
        lb[0::3] = src[1::4]
        la[1::3] = src[2::4]
        lb[2::3] = src[3::4]
        return la, lb
    if code_rate == "2/3":
        blocks = llrs.size // 3
        n_out = blocks * 2
        la = np.zeros(n_out, dtype=np.float64)
        lb = np.zeros(n_out, dtype=np.float64)
        src = llrs[: blocks * 3]
        la[0::2] = src[0::3]
        lb[0::2] = src[1::3]
        la[1::2] = src[2::3]
        return la, lb
    raise ValueError(f"Unsupported code_rate: {code_rate!r}")


def _viterbi_decode_serial(
    llr_a: np.ndarray, llr_b: np.ndarray, zero_terminated: bool = True
) -> np.ndarray:
    """Exact serial 64-state K=7 (133_8, 171_8) Viterbi decoder for short blocks."""
    n_steps = int(min(llr_a.size, llr_b.size))
    if n_steps == 0:
        return np.empty(0, dtype=np.uint8)

    pm = np.full(64, -1e12, dtype=np.float64)
    pm[0] = 0.0
    tb = np.empty((n_steps, 64), dtype=np.uint8)

    prev0 = _VIT_PREV0
    prev1 = _VIT_PREV1
    o0a = _VIT_O0A
    o0b = _VIT_O0B
    o1a = _VIT_O1A
    o1b = _VIT_O1B

    for t in range(n_steps):
        la = float(llr_a[t])
        lb = float(llr_b[t])
        m0 = pm[prev0] + la * o0a + lb * o0b
        m1 = pm[prev1] + la * o1a + lb * o1b
        pick1 = m1 > m0
        pm = np.where(pick1, m1, m0)
        tb[t] = pick1

    state = 0 if zero_terminated else int(np.argmax(pm))
    decoded = np.empty(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = (state >> 5) & 1
        low = int(tb[t, state])
        state = ((state & 0x1F) << 1) | low
    return decoded


def _viterbi_decode(
    llr_a: np.ndarray, llr_b: np.ndarray, zero_terminated: bool = True
) -> np.ndarray:
    """Decode K=7 (133_8, 171_8) soft LLRs into uint8 bits using parallel overlap-save."""
    n_steps = int(min(llr_a.size, llr_b.size))
    if n_steps <= 640:
        return _viterbi_decode_serial(llr_a, llr_b, zero_terminated=zero_terminated)

    warmup = 64
    tail = 64
    stride = 512
    n_chunks = (n_steps + stride - 1) // stride
    win_len = warmup + stride + tail
    last_len = n_steps - (n_chunks - 1) * stride
    last_end_t = warmup + last_len - 1

    starts = np.arange(n_chunks, dtype=np.intp) * stride - warmup
    idx_grid = starts[None, :] + np.arange(win_len, dtype=np.intp)[:, None]
    valid = (idx_grid >= 0) & (idx_grid < n_steps)
    clamped = np.clip(idx_grid, 0, n_steps - 1)
    la_mat = np.where(valid, llr_a[clamped], 0.0)
    lb_mat = np.where(valid, llr_b[clamped], 0.0)

    pm = np.zeros((n_chunks, 64), dtype=np.float64)
    pm[0, 1:] = -1e12
    tb = np.empty((win_len, n_chunks, 64), dtype=np.uint8)

    prev0 = _VIT_PREV0
    prev1 = _VIT_PREV1
    o0a = _VIT_O0A[None, :]
    o0b = _VIT_O0B[None, :]
    o1a = _VIT_O1A[None, :]
    o1b = _VIT_O1B[None, :]
    end_state_last = 0

    for t in range(win_len):
        la = la_mat[t, :, None]
        lb = lb_mat[t, :, None]
        m0 = pm[:, prev0] + la * o0a + lb * o0b
        m1 = pm[:, prev1] + la * o1a + lb * o1b
        pick1 = m1 > m0
        pm = np.where(pick1, m1, m0)
        tb[t] = pick1
        if t == last_end_t:
            end_state_last = 0 if zero_terminated else int(np.argmax(pm[-1]))

    states = np.argmax(pm, axis=1).astype(np.intp)
    dec_win = np.empty((win_len, n_chunks), dtype=np.uint8)
    c_idx = np.arange(n_chunks, dtype=np.intp)
    for t in range(win_len - 1, -1, -1):
        if t == last_end_t:
            states[-1] = end_state_last
        dec_win[t] = (states >> 5) & 1
        low = tb[t, c_idx, states]
        states = ((states & 0x1F) << 1) | low

    return dec_win[warmup : warmup + stride, :].T.reshape(-1)[:n_steps].copy()


def _build_descramble_lut() -> np.ndarray:
    """Precompute all 128 periodic 127-bit sequences of IEEE 802.11 x^7 + x^4 + 1 LFSR."""
    lut = np.empty((128, 127), dtype=np.uint8)
    for p7 in range(128):
        z = np.empty(127, dtype=np.uint8)
        for b in range(7):
            z[b] = (p7 >> b) & 1
        for i in range(7, 127):
            z[i] = z[i - 4] ^ z[i - 7]
        lut[p7] = z
    return lut


_DESCRAMBLE_LUT: np.ndarray = _build_descramble_lut()


def _scramble_ieee80211(bits: np.ndarray, seed: int = 0x5D) -> np.ndarray:
    """Scramble DATA bits with IEEE 802.11 x^7 + x^4 + 1 LFSR (non-zero 7-bit seed)."""
    s = int(seed) & 0x7F
    if s == 0:
        s = 0x5D
    reg = [(s >> (6 - i)) & 1 for i in range(7)]
    out = np.empty(bits.size, dtype=np.uint8)
    for idx in range(bits.size):
        z = reg[3] ^ reg[6]
        out[idx] = (int(bits[idx]) & 1) ^ z
        reg = [z] + reg[:6]
    return out


def _descramble_ieee80211(scrambled_bits: np.ndarray) -> np.ndarray:
    """Self-synchronizing IEEE 802.11 descrambler using the 7-zero SERVICE header prefix."""
    n = int(scrambled_bits.size)
    if n < 16:
        return scrambled_bits.copy()
    b = (scrambled_bits & 1).astype(np.uint8, copy=False)
    # First 7 SERVICE bits are guaranteed 0 before scrambling -> z[0..6] == scrambled[0..6]
    p7 = int(
        int(b[0])
        | (int(b[1]) << 1)
        | (int(b[2]) << 2)
        | (int(b[3]) << 3)
        | (int(b[4]) << 4)
        | (int(b[5]) << 5)
        | (int(b[6]) << 6)
    )
    seq = _DESCRAMBLE_LUT[p7, np.arange(n, dtype=np.intp) % 127]
    return b ^ seq


def _bytes_to_bits_lsb(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return np.empty(0, dtype=np.uint8)
    shifts = np.arange(8, dtype=np.uint8)
    return ((arr[:, None] >> shifts[None, :]) & 1).reshape(-1).astype(np.uint8)


def _bits_to_bytes_lsb(bits: np.ndarray) -> bytes:
    n_bytes = int(bits.size) // 8
    if n_bytes == 0:
        return b""
    b = bits[: n_bytes * 8].reshape(n_bytes, 8).astype(np.uint8)
    weights = (1 << np.arange(8, dtype=np.uint8))[None, :]
    packed = np.sum(b * weights, axis=1, dtype=np.uint8)
    return packed.tobytes()


def _modulate_subcarriers(interleaved_bits: np.ndarray, n_bpsc: int) -> np.ndarray:
    """Gray-map interleaved coded bits (multiple of 48*n_bpsc) onto 48-subcarrier symbols."""
    if n_bpsc == 1:
        return (2.0 * interleaved_bits.astype(np.float64) - 1.0).astype(np.complex128)
    if n_bpsc == 2:
        pairs = interleaved_bits.reshape(-1, 2).astype(np.float64)
        return (
            ((2.0 * pairs[:, 0] - 1.0) + 1j * (2.0 * pairs[:, 1] - 1.0))
            / math.sqrt(2.0)
        ).astype(np.complex128)
    if n_bpsc == 4:
        quads = interleaved_bits.reshape(-1, 4)
        lut = np.array([-3.0, -1.0, 3.0, 1.0], dtype=np.float64)
        i_idx = (quads[:, 0] << 1) | quads[:, 1]
        q_idx = (quads[:, 2] << 1) | quads[:, 3]
        return ((lut[i_idx] + 1j * lut[q_idx]) / math.sqrt(10.0)).astype(np.complex128)
    if n_bpsc == 6:
        sext = interleaved_bits.reshape(-1, 6)
        lut6 = np.array([-7.0, -5.0, -1.0, -3.0, 7.0, 5.0, 1.0, 3.0], dtype=np.float64)
        i_idx = (sext[:, 0] << 2) | (sext[:, 1] << 1) | sext[:, 2]
        q_idx = (sext[:, 3] << 2) | (sext[:, 4] << 1) | sext[:, 5]
        return ((lut6[i_idx] + 1j * lut6[q_idx]) / math.sqrt(42.0)).astype(
            np.complex128
        )
    raise ValueError(f"Unsupported n_bpsc: {n_bpsc}")


def _demap_llrs(
    eq_syms: np.ndarray, chan_weights: np.ndarray, n_bpsc: int
) -> np.ndarray:
    """Compute soft LLRs (positive -> bit 1) for 48 data subcarriers with CSI reliability."""
    re = eq_syms.real
    im = eq_syms.imag
    w = chan_weights
    if n_bpsc == 1:
        return (re * w).reshape(-1)
    if n_bpsc == 2:
        s = math.sqrt(2.0)
        out = np.empty((eq_syms.size, 2), dtype=np.float64)
        out[:, 0] = re * s * w
        out[:, 1] = im * s * w
        return out.reshape(-1)
    if n_bpsc == 4:
        s = math.sqrt(10.0)
        ri = re * s
        rq = im * s
        out = np.empty((eq_syms.size, 4), dtype=np.float64)
        out[:, 0] = ri * w
        out[:, 1] = (2.0 - np.abs(ri)) * w
        out[:, 2] = rq * w
        out[:, 3] = (2.0 - np.abs(rq)) * w
        return out.reshape(-1)
    if n_bpsc == 6:
        s = math.sqrt(42.0)
        ri = re * s
        rq = im * s
        out = np.empty((eq_syms.size, 6), dtype=np.float64)
        out[:, 0] = ri * w
        out[:, 1] = (4.0 - np.abs(ri)) * w
        out[:, 2] = (2.0 - np.abs(np.abs(ri) - 4.0)) * w
        out[:, 3] = rq * w
        out[:, 4] = (4.0 - np.abs(rq)) * w
        out[:, 5] = (2.0 - np.abs(np.abs(rq) - 4.0)) * w
        return out.reshape(-1)
    raise ValueError(f"Unsupported n_bpsc: {n_bpsc}")


# ---------------------------------------------------------------------------
# Waveform & MAC Frame Synthesis Helpers (for tests & offline replay checks)
# ---------------------------------------------------------------------------


def mac_str_to_bytes(mac: str) -> bytes:
    canon = canonicalize_bssid(mac)
    parts = canon.split(":")
    if len(parts) != 6 or not all(len(p) == 2 for p in parts):
        raise ValueError(f"Invalid MAC address string: {mac!r}")
    octets = [int(p, 16) for p in parts]
    if not all(0 <= b <= 0xFF for b in octets):
        raise ValueError(f"MAC octet out of range in {mac!r}")
    return bytes(octets)


def mac_bytes_to_str(raw: bytes) -> str:
    return ":".join(f"{b:02X}" for b in raw[:6])


def append_fcs(frame_without_fcs: bytes) -> bytes:
    """Append IEEE 802.11 32-bit CRC FCS (little-endian) to an MPDU header/body."""
    crc = zlib.crc32(frame_without_fcs) & 0xFFFFFFFF
    return frame_without_fcs + struct.pack("<I", crc)


def build_80211_mac_frame(
    frame_kind: str = "BEACON",
    *,
    addr1: str = "FF:FF:FF:FF:FF:FF",
    addr2: str | None = "AA:BB:CC:00:00:01",
    addr3: str | None = None,
    seq_num: int = 0,
    tsf_us: int = 1_000_000,
    ssid: str = "HackRFRadarLab",
    corrupt_fcs: bool = False,
) -> bytes:
    """Construct a standards-compliant IEEE 802.11 MPDU with CRC-32 FCS."""
    kind = frame_kind.upper()
    ra = mac_str_to_bytes(addr1)

    if kind in {"CTS", "ACK"}:
        subtype = 12 if kind == "CTS" else 13
        fc = (1 << 2) | (subtype << 4)
        body = struct.pack("<HH", fc, 44) + ra
    elif kind in {"RTS", "BLOCK_ACK"}:
        if addr2 is None:
            raise ValueError(f"{kind} requires addr2 (TA)")
        ta = mac_str_to_bytes(addr2)
        subtype = 11 if kind == "RTS" else 9
        fc = (1 << 2) | (subtype << 4)
        extra = b"\x00" * 12 if kind == "BLOCK_ACK" else b""
        body = struct.pack("<HH", fc, 120) + ra + ta + extra
    else:
        if addr2 is None:
            raise ValueError(f"{kind} requires addr2 (TA)")
        ta = mac_str_to_bytes(addr2)
        bssid_bytes = mac_str_to_bytes(addr3) if addr3 is not None else ta
        seq_ctrl = (int(seq_num) & 0x0FFF) << 4

        if kind in {"BEACON", "PROBE_RESP"}:
            subtype = 8 if kind == "BEACON" else 5
            fc = (0 << 2) | (subtype << 4)
            hdr = (
                struct.pack("<HH", fc, 0)
                + ra
                + ta
                + bssid_bytes
                + struct.pack("<H", seq_ctrl)
            )
            ssid_bytes = ssid.encode("ascii", errors="ignore")[:32]
            fixed = struct.pack("<QHH", int(tsf_us) & 0xFFFFFFFFFFFFFFFF, 100, 0x0401)
            ie_ssid = bytes([0, len(ssid_bytes)]) + ssid_bytes
            body = hdr + fixed + ie_ssid
        elif kind == "PROBE_REQ":
            fc = (0 << 2) | (4 << 4)
            hdr = (
                struct.pack("<HH", fc, 0)
                + mac_str_to_bytes("FF:FF:FF:FF:FF:FF")
                + ta
                + mac_str_to_bytes("FF:FF:FF:FF:FF:FF")
                + struct.pack("<H", seq_ctrl)
            )
            body = hdr + b"\x00\x00"
        elif kind == "QOS_NULL":
            fc = (2 << 2) | (12 << 4) | (1 << 8)
            hdr = (
                struct.pack("<HH", fc, 44)
                + ra
                + ta
                + bssid_bytes
                + struct.pack("<HH", seq_ctrl, 0)
            )
            body = hdr
        else:
            # DATA_PREAMBLE / generic cleartext-header Data MPDU
            fc = (2 << 2) | (0 << 4) | (1 << 9)
            hdr = (
                struct.pack("<HH", fc, 44)
                + ra
                + ta
                + bssid_bytes
                + struct.pack("<H", seq_ctrl)
            )
            body = hdr + b"\xaa\xaa\x03\x00\x00\x00\x08\x00"

    mpdu = append_fcs(body)
    if corrupt_fcs:
        mpdu = mpdu[:-1] + bytes([mpdu[-1] ^ 0xFF])
    return mpdu


def synthesize_80211_ofdm_burst(
    mpdu: bytes,
    *,
    rate_mbps: int = 6,
    channel_csi: np.ndarray | None = None,
    cfo_hz: float = 0.0,
    sample_rate: float = 20e6,
    snr_db: float | None = 30.0,
    amplitude: float = 0.25,
    pad_before: int = 120,
    pad_after: int = 120,
    scrambler_seed: int = 0x5D,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesize a 20 MSPS IEEE 802.11a/g Non-HT OFDM baseband burst with channel & CFO."""
    if rate_mbps not in MBPS_TO_RATE_BITS:
        raise ValueError(f"Unsupported Non-HT OFDM rate: {rate_mbps} Mbps")
    rate_bits = MBPS_TO_RATE_BITS[rate_mbps]
    _, n_bpsc, n_cbps, n_dbps, code_rate = LSIG_RATE_TABLE[rate_bits]

    psdu_len = len(mpdu)
    if not (1 <= psdu_len <= 4095):
        raise ValueError(f"MPDU length must be in 1..4095 bytes, got {psdu_len}")

    # Build 24-bit L-SIG
    lsig_bits = np.zeros(24, dtype=np.uint8)
    lsig_bits[0:4] = rate_bits
    lsig_bits[4] = 0
    for b in range(12):
        lsig_bits[5 + b] = (psdu_len >> b) & 1
    lsig_bits[17] = int(np.sum(lsig_bits[:17]) & 1)
    lsig_coded = _conv_encode(lsig_bits, "1/2")
    fwd_48, _ = _INTERLEAVER_TABLES[48]
    lsig_inter = lsig_coded[fwd_48]
    lsig_data_syms = _modulate_subcarriers(lsig_inter, 1)

    # Build DATA = 16 zero SERVICE bits + PSDU + 6 tail zeros + pad
    psdu_bits = _bytes_to_bits_lsb(mpdu)
    n_sym = math.ceil((16 + psdu_bits.size + 6) / n_dbps)
    total_data_bits = n_sym * n_dbps
    raw_data = np.zeros(total_data_bits, dtype=np.uint8)
    raw_data[16 : 16 + psdu_bits.size] = psdu_bits
    scrambled = _scramble_ieee80211(raw_data, seed=scrambler_seed)
    # Zero the 6 tail bits immediately after PSDU so encoder terminates in state 0
    scrambled[16 + psdu_bits.size : 16 + psdu_bits.size + 6] = 0

    coded_data = _conv_encode(scrambled, code_rate)
    fwd_perm, _ = _INTERLEAVER_TABLES[n_cbps]
    inter_data = np.empty_like(coded_data)
    for sym_idx in range(n_sym):
        seg = coded_data[sym_idx * n_cbps : (sym_idx + 1) * n_cbps]
        inter_data[sym_idx * n_cbps : (sym_idx + 1) * n_cbps] = seg[fwd_perm]
    mod_data = _modulate_subcarriers(inter_data, n_bpsc).reshape(n_sym, 48)

    # Assemble all OFDM symbols in frequency domain (applying 52-subcarrier channel_csi if given)
    if channel_csi is not None:
        h52 = np.asarray(channel_csi, dtype=np.complex128).reshape(52)
    else:
        h52 = np.ones(52, dtype=np.complex128)

    # Reconstruct L-STF and L-LTF through the same 52-subcarrier channel response
    x_stf = np.zeros(FFT_SIZE, dtype=np.complex128)
    scale_stf = math.sqrt(13.0 / 6.0)
    for k, val in _LSTF_SPARSE.items():
        pos52 = SUBCARRIER_INDICES.index(k)
        x_stf[(k + FFT_SIZE) % FFT_SIZE] = val * scale_stf * h52[pos52]
    stf_64 = np.fft.ifft(x_stf) * 8.0
    stf_160 = np.tile(stf_64[:16], 10)

    x_ltf = np.zeros(FFT_SIZE, dtype=np.complex128)
    x_ltf[ACTIVE_FFT_BINS] = L_LTF_SEQ_52 * h52
    ltf_64 = np.fft.ifft(x_ltf) * 8.0
    ltf_160 = np.concatenate([ltf_64[32:], ltf_64, ltf_64])

    ofdm_blocks: list[np.ndarray] = [stf_160, ltf_160]
    for v_idx in range(1 + n_sym):
        active_52 = np.zeros(52, dtype=np.complex128)
        if v_idx == 0:
            active_52[DATA_POS_IN_52] = lsig_data_syms
        else:
            active_52[DATA_POS_IN_52] = mod_data[v_idx - 1]
        active_52[PILOT_POS_IN_52] = BASE_PILOT_SIGNS * PILOT_POLARITY_127[v_idx % 127]
        x64 = np.zeros(FFT_SIZE, dtype=np.complex128)
        x64[ACTIVE_FFT_BINS] = active_52 * h52
        sym64 = np.fft.ifft(x64) * 8.0
        ofdm_blocks.append(np.concatenate([sym64[48:], sym64]))

    burst = np.concatenate(ofdm_blocks) * float(amplitude)
    if abs(cfo_hz) > 1e-6:
        n_idx = np.arange(burst.size, dtype=np.float64)
        burst = burst * np.exp(
            1j * (2.0 * math.pi * float(cfo_hz) / float(sample_rate)) * n_idx
        )

    total_len = int(pad_before) + burst.size + int(pad_after)
    waveform = np.zeros(total_len, dtype=np.complex128)
    waveform[int(pad_before) : int(pad_before) + burst.size] = burst

    if snr_db is not None:
        gen = rng if rng is not None else np.random.default_rng(0)
        sig_pow = float(np.mean(np.abs(burst) ** 2))
        noise_var = max(sig_pow * (10.0 ** (-float(snr_db) / 10.0)), 1e-12)
        sigma = math.sqrt(0.5 * noise_var)
        noise = sigma * (
            gen.standard_normal(total_len) + 1j * gen.standard_normal(total_len)
        )
        waveform = waveform + noise

    return waveform.astype(np.complex64)


# ---------------------------------------------------------------------------
# Cleartext 802.11 MAC Header Parser & Per-BSSID SIFS / TSF / Seq Tracker
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParsedMACInfo:
    frame_type: str
    tx_mac: str | None
    rx_mac: str | None
    seq_num: int | None
    tsf_us: int | None
    fcs_ok: bool
    duration_us: int = 0
    ap_bssid: str | None = None


def parse_80211_mac_frame(
    mpdu: bytes, *, fcs_present: bool = True, fcs_known_bad: bool = False
) -> ParsedMACInfo:
    """Parse cleartext IEEE 802.11 MAC header + optional CRC-32 FCS without payload decryption."""
    min_len = 14 if fcs_present else 10
    if len(mpdu) < min_len:
        return ParsedMACInfo("FCS_ERROR", None, None, None, None, False)

    if fcs_present:
        expected_crc = zlib.crc32(mpdu[:-4]) & 0xFFFFFFFF
        actual_crc = struct.unpack_from("<I", mpdu, len(mpdu) - 4)[0]
        fcs_ok = (expected_crc == actual_crc) and (not fcs_known_bad)
        payload = mpdu[:-4]
    else:
        fcs_ok = not fcs_known_bad
        payload = mpdu

    fc, dur = struct.unpack_from("<HH", payload, 0)
    proto_ver = fc & 0x03
    if proto_ver != 0:
        return ParsedMACInfo("FCS_ERROR", None, None, None, None, False)

    ftype = (fc >> 2) & 0x03
    subtype = (fc >> 4) & 0x0F
    addr1 = mac_bytes_to_str(payload[4:10]) if len(payload) >= 10 else None

    if not fcs_ok:
        tx_guess = (
            mac_bytes_to_str(payload[10:16])
            if len(payload) >= 16 and ftype in (0, 2)
            else None
        )
        return ParsedMACInfo(
            "FCS_ERROR", tx_guess, addr1, None, None, False, dur & 0x7FFF
        )

    # Control frames (ftype == 1)
    if ftype == 1:
        if subtype == 12:
            return ParsedMACInfo("CTS", None, addr1, None, None, True, dur & 0x7FFF)
        if subtype == 13:
            return ParsedMACInfo("ACK", None, addr1, None, None, True, dur & 0x7FFF)
        tx_mac = mac_bytes_to_str(payload[10:16]) if len(payload) >= 16 else None
        if subtype == 9:
            return ParsedMACInfo(
                "BLOCK_ACK", tx_mac, addr1, None, None, True, dur & 0x7FFF
            )
        return ParsedMACInfo(
            "DATA_PREAMBLE", tx_mac, addr1, None, None, True, dur & 0x7FFF
        )

    # Management (ftype == 0) & Data (ftype == 2) require at least 24-byte header
    if len(payload) < 24:
        return ParsedMACInfo("FCS_ERROR", None, addr1, None, None, False)

    addr2 = mac_bytes_to_str(payload[10:16])
    addr3 = mac_bytes_to_str(payload[16:22])
    seq_ctrl = struct.unpack_from("<H", payload, 22)[0]
    seq_num = (seq_ctrl >> 4) & 0x0FFF

    if ftype == 0:
        ap_bssid = addr3 if _is_valid_unicast_mac(addr3) else None
        if subtype == 8:
            tsf = (
                struct.unpack_from("<Q", payload, 24)[0] if len(payload) >= 32 else None
            )
            return ParsedMACInfo(
                "BEACON",
                addr2,
                addr1,
                seq_num,
                tsf,
                True,
                dur & 0x7FFF,
                ap_bssid or addr2,
            )
        if subtype == 4:
            return ParsedMACInfo(
                "PROBE_REQ", addr2, addr1, seq_num, None, True, dur & 0x7FFF, ap_bssid
            )
        if subtype == 5:
            tsf = (
                struct.unpack_from("<Q", payload, 24)[0] if len(payload) >= 32 else None
            )
            return ParsedMACInfo(
                "PROBE_RESP",
                addr2,
                addr1,
                seq_num,
                tsf,
                True,
                dur & 0x7FFF,
                ap_bssid or addr2,
            )
        return ParsedMACInfo(
            "DATA_PREAMBLE", addr2, addr1, seq_num, None, True, dur & 0x7FFF, ap_bssid
        )

    if ftype == 2:
        to_ds = (fc >> 8) & 0x01
        from_ds = (fc >> 9) & 0x01
        if to_ds == 1 and from_ds == 0:
            raw_bssid = addr1
        elif to_ds == 0 and from_ds == 1:
            raw_bssid = addr2
        elif to_ds == 0 and from_ds == 0:
            raw_bssid = addr3
        else:
            raw_bssid = None
        ap_bssid = raw_bssid if _is_valid_unicast_mac(raw_bssid) else None
        if subtype in (4, 12):
            return ParsedMACInfo(
                "QOS_NULL", addr2, addr1, seq_num, None, True, dur & 0x7FFF, ap_bssid
            )
        return ParsedMACInfo(
            "DATA_PREAMBLE", addr2, addr1, seq_num, None, True, dur & 0x7FFF, ap_bssid
        )

    return ParsedMACInfo("FCS_ERROR", None, addr1, None, None, False)


_HEX_CHARS_MAC: frozenset[str] = frozenset("0123456789ABCDEF")


def _is_valid_unicast_mac(mac: str | None) -> bool:
    if not mac:
        return False
    canon = canonicalize_bssid(mac)
    if canon in {"FF:FF:FF:FF:FF:FF", "00:00:00:00:00:00", "UNKNOWN", "NONE"}:
        return False
    parts = canon.split(":")
    if len(parts) != 6 or not all(
        len(p) == 2 and all(ch in _HEX_CHARS_MAC for ch in p) for p in parts
    ):
        return False
    first = int(parts[0], 16)
    return (first & 0x01) == 0


class _LinkTelemetryTracker:
    """Stateful SIFS 1-address CTS/ACK attributor, TSF drift regressor, & SeqCtrl tracker."""

    def __init__(self, sifs_window_s: float = 0.0012) -> None:
        self.sifs_window_s = float(sifs_window_s)
        self._last_seq: dict[str, int] = {}
        self._tsf_history: dict[str, list[tuple[float, int]]] = {}
        self._known_aps_by_ch: dict[int, set[str]] = {}
        # Per-channel last directed frame: (end_ts, start_ts, tx_mac, rx_mac)
        self._last_directed: dict[int, tuple[float, float, str, str]] = {}

    def attribute_and_track(
        self,
        info: ParsedMACInfo,
        *,
        timestamp: float,
        burst_end_ts: float,
        channel: int,
    ) -> tuple[str, str | None, bool, int, float | None]:
        """Return (bssid, receiver_addr, attributed_via_sifs, seq_gap, tsf_drift_ppm)."""
        tx_mac = canonicalize_bssid(info.tx_mac) if info.tx_mac else None
        rx_mac = canonicalize_bssid(info.rx_mac) if info.rx_mac else None
        via_sifs = False

        if info.fcs_ok:
            if (
                info.frame_type in {"BEACON", "PROBE_RESP"}
                and tx_mac
                and _is_valid_unicast_mac(tx_mac)
            ):
                self._known_aps_by_ch.setdefault(channel, set()).add(tx_mac)
            if info.ap_bssid:
                ap_canon = canonicalize_bssid(info.ap_bssid)
                if _is_valid_unicast_mac(ap_canon):
                    self._known_aps_by_ch.setdefault(channel, set()).add(ap_canon)

        # SIFS attribution for 1-address control frames (CTS, ACK)
        if (
            info.fcs_ok
            and info.frame_type in {"CTS", "ACK"}
            and tx_mac is None
            and rx_mac
        ):
            prev = self._last_directed.get(channel)
            if prev is not None:
                prev_end, prev_start, prev_tx, prev_rx = prev
                dt_from_end = timestamp - prev_end
                dt_from_start = timestamp - prev_start
                if (
                    rx_mac == prev_tx
                    and _is_valid_unicast_mac(prev_rx)
                    and (
                        -25e-6 <= dt_from_end <= self.sifs_window_s
                        or 0.0 <= dt_from_start <= 0.004
                    )
                ):
                    tx_mac = prev_rx
                    via_sifs = True
            # Fallback for 802.11g/n CTS-to-self protection where Addr1 == AP's own BSSID
            if tx_mac is None and info.frame_type == "CTS":
                ch_aps = self._known_aps_by_ch.get(channel, set())
                if rx_mac in ch_aps:
                    tx_mac = rx_mac

        # Record directed unicast frame for subsequent SIFS CTS/ACK matching
        if (
            info.fcs_ok
            and tx_mac is not None
            and rx_mac is not None
            and _is_valid_unicast_mac(tx_mac)
            and _is_valid_unicast_mac(rx_mac)
            and info.frame_type not in {"CTS", "ACK"}
        ):
            self._last_directed[channel] = (burst_end_ts, timestamp, tx_mac, rx_mac)

        bssid = tx_mac if (tx_mac and _is_valid_unicast_mac(tx_mac)) else "UNATTRIBUTED"

        # 12-bit Sequence Control gap tracking per transmitter
        seq_gap = 0
        if info.fcs_ok and info.seq_num is not None and bssid != "UNATTRIBUTED":
            prev_seq = self._last_seq.get(bssid)
            if prev_seq is not None:
                diff = (int(info.seq_num) - prev_seq) % 4096
                if diff > 1 and diff < 512:
                    seq_gap = diff - 1
            self._last_seq[bssid] = int(info.seq_num)

        # 64-bit Beacon/ProbeResp TSF clock drift regression (ppm)
        tsf_drift_ppm: float | None = None
        if info.fcs_ok and info.tsf_us is not None and bssid != "UNATTRIBUTED":
            hist = self._tsf_history.setdefault(bssid, [])
            # Reset regression if AP rebooted / TSF jumped backward
            if hist and (info.tsf_us < hist[-1][1] or (timestamp - hist[-1][0]) > 30.0):
                hist.clear()
            hist.append((timestamp, int(info.tsf_us)))
            if len(hist) > 24:
                del hist[:-24]
            if len(hist) >= 2:
                t0, u0 = hist[0]
                dt_host_us = (
                    np.array([p[0] - t0 for p in hist], dtype=np.float64)
                ) * 1e6
                dt_tsf_us = np.array([float(p[1] - u0) for p in hist], dtype=np.float64)
                span_us = float(dt_host_us[-1] - dt_host_us[0])
                if span_us >= 25_000.0:
                    denom = float(np.dot(dt_host_us, dt_host_us))
                    if denom > 1e-6:
                        slope = float(np.dot(dt_host_us, dt_tsf_us) / denom)
                        ppm = (slope - 1.0) * 1e6
                        if abs(ppm) <= 250.0:
                            tsf_drift_ppm = ppm

        return bssid, rx_mac, via_sifs, seq_gap, tsf_drift_ppm


# ---------------------------------------------------------------------------
# Native 20 MSPS 802.11a/g/n OFDM Burst Detector & CSI/MAC Extractor
# ---------------------------------------------------------------------------


class NativeOFDMBurstDetector:
    """Vectorized NumPy/SciPy 20 MSPS 802.11 OFDM preamble, 52-bin CSI, & MAC decoder."""

    def __init__(
        self,
        sample_rate: float = 20e6,
        min_snr_db: float = 5.5,
        sc_threshold: float = 0.50,
        max_full_fcs_bytes: int = 2350,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.min_snr_db = float(min_snr_db)
        self.sc_threshold = float(sc_threshold)
        self.max_full_fcs_bytes = int(max_full_fcs_bytes)
        self._tracker = _LinkTelemetryTracker()
        self._overlap_iq = np.empty(0, dtype=np.complex64)
        self._overlap_ts = 0.0
        self._last_dwell_idx: int | None = None

    def reset_stream_state(self) -> None:
        self._overlap_iq = np.empty(0, dtype=np.complex64)

    def process_chunk(self, chunk: IQChunk) -> list[PacketEvent]:
        """Detect and decode all 802.11 OFDM bursts inside a conditioned 20 MSPS IQChunk."""
        if self._last_dwell_idx != chunk.dwell_index:
            self._overlap_iq = np.empty(0, dtype=np.complex64)
            self._last_dwell_idx = chunk.dwell_index

        if self._overlap_iq.size > 0:
            iq = np.concatenate([self._overlap_iq, chunk.samples])
            base_ts = self._overlap_ts
        else:
            iq = chunk.samples
            base_ts = chunk.timestamp

        events, keep_start = self._detect_with_cursor(
            iq,
            timestamp=base_ts,
            freq_hz=chunk.freq_hz,
            channel=chunk.channel,
            sample_rate=chunk.sample_rate,
            adc_clip_ratio=chunk.adc_clip_ratio,
        )

        # Retain unconsumed tail (or any incomplete straddling burst from before its L-STF)
        # while never backing up across an already-decoded burst.
        keep_start = min(max(0, keep_start), iq.size)
        if keep_start < iq.size:
            self._overlap_iq = iq[keep_start:].copy()
            self._overlap_ts = base_ts + keep_start / chunk.sample_rate
        else:
            self._overlap_iq = np.empty(0, dtype=np.complex64)
            self._overlap_ts = base_ts + iq.size / chunk.sample_rate
        return events

    def detect_in_samples(
        self,
        iq: np.ndarray,
        *,
        timestamp: float = 0.0,
        freq_hz: float = 2437e6,
        channel: int | None = None,
        sample_rate: float = 20e6,
        adc_clip_ratio: float = 0.0,
    ) -> list[PacketEvent]:
        """Run Schmidl-Cox + L-LTF sync + 52-subcarrier CSI + Viterbi MAC decode on I/Q array."""
        events, _ = self._detect_with_cursor(
            iq,
            timestamp=timestamp,
            freq_hz=freq_hz,
            channel=channel,
            sample_rate=sample_rate,
            adc_clip_ratio=adc_clip_ratio,
        )
        return events

    def _detect_with_cursor(
        self,
        iq: np.ndarray,
        *,
        timestamp: float = 0.0,
        freq_hz: float = 2437e6,
        channel: int | None = None,
        sample_rate: float = 20e6,
        adc_clip_ratio: float = 0.0,
    ) -> tuple[list[PacketEvent], int]:
        x = np.asarray(iq, dtype=np.complex64).reshape(-1)
        n_total = int(x.size)
        if n_total < 420:
            return [], 0

        ch = int(channel) if channel is not None else freq_hz_to_channel(freq_hz)
        fs = float(sample_rate)

        # Estimate background noise floor from lower 1st percentile / minimum of 32-sample block powers
        # so even >95%-duty-cycle burst buffers measure quiet guard noise rather than signal power.
        n_blocks = max(1, n_total // 32)
        pwr = x.real.astype(np.float32) ** 2 + x.imag.astype(np.float32) ** 2
        blk_pwr = np.mean(pwr[: n_blocks * 32].reshape(n_blocks, 32), axis=1)
        noise_est = min(float(np.percentile(blk_pwr, 1)), float(np.min(blk_pwr)) * 2.5)
        noise_var = max(noise_est, 1e-11)
        noise_floor_dbm = float(10.0 * math.log10(noise_var) - 45.0)
        min_power_gate = max(1e-10, 0.25 * noise_var)

        # Vectorized Schmidl-Cox lag-16 autocorrelation with window W=64 evaluated at stride=4
        w_sc = 64
        prod16 = x[:-16].conj() * x[16:]
        csum_p = np.empty(prod16.size + 1, dtype=np.complex128)
        csum_p[0] = 0.0
        np.cumsum(prod16, dtype=np.complex128, out=csum_p[1:])

        csum_r = np.empty(pwr.size + 1, dtype=np.float64)
        csum_r[0] = 0.0
        np.cumsum(pwr, dtype=np.float64, out=csum_r[1:])

        max_start = n_total - 400
        if max_start <= 0:
            return [], 0

        grid = np.arange(0, max_start, 4, dtype=np.intp)
        p_win = csum_p[grid + w_sc] - csum_p[grid]
        r_win = csum_r[grid + 16 + w_sc] - csum_r[grid + 16]
        active_mask = r_win >= (min_power_gate * w_sc)
        if not np.any(active_mask):
            return [], max(0, n_total - 416)

        metric = np.zeros(grid.size, dtype=np.float64)
        p_act = p_win[active_mask]
        r_act = r_win[active_mask]
        metric[active_mask] = (p_act.real**2 + p_act.imag**2) / np.maximum(
            r_act * r_act, 1e-20
        )

        cand_indices = np.flatnonzero(metric >= self.sc_threshold)
        if cand_indices.size == 0:
            return [], max(0, n_total - 416)

        events: list[PacketEvent] = []
        skip_until_sample = 0
        pending_incomplete_start: int | None = None
        ref_ltf = REF_LLTF_64.astype(np.complex64)

        idx_ptr = 0
        n_cands = int(cand_indices.size)
        while idx_ptr < n_cands:
            g_idx = int(cand_indices[idx_ptr])
            n0 = int(grid[g_idx])
            if n0 < skip_until_sample:
                idx_ptr += 1
                continue

            # Advance across contiguous Schmidl-Cox plateau (~100-140 samples) to find plateau center
            end_ptr = idx_ptr
            while (
                end_ptr + 1 < n_cands
                and (cand_indices[end_ptr + 1] - cand_indices[end_ptr]) <= 3
            ):
                if (grid[cand_indices[end_ptr + 1]] - n0) > 180:
                    break
                end_ptr += 1

            # Require minimum plateau width (>= 3 grid steps = 12 samples) to reject impulse spikes
            if (end_ptr - idx_ptr) < 2:
                idx_ptr = end_ptr + 1
                continue

            plateau_bins = cand_indices[idx_ptr : end_ptr + 1]
            best_bin = int(plateau_bins[int(np.argmax(metric[plateau_bins]))])
            p_stf = complex(p_win[best_bin])
            omega_coarse = math.atan2(p_stf.imag, p_stf.real) / 16.0

            search_lo = max(0, n0 - 16)
            search_hi = min(n_total, n0 + 380)
            if search_hi - search_lo < 260:
                pending_incomplete_start = max(skip_until_sample, max(0, n0 - 64))
                break

            seg = x[search_lo:search_hi].astype(np.complex128)
            n_vec = np.arange(seg.size, dtype=np.float64)
            seg_derot = seg * np.exp(-1j * omega_coarse * n_vec)

            # Cross-correlate with 64-sample L-LTF reference to locate twin T1/T2 peaks (+64 apart)
            corr = np.correlate(seg_derot.astype(np.complex64), ref_ltf, mode="valid")
            if corr.size <= 65:
                idx_ptr = end_ptr + 1
                continue

            corr_mag = np.abs(corr)
            seg_pwr = np.abs(seg_derot) ** 2
            seg_csum = np.concatenate([[0.0], np.cumsum(seg_pwr)])
            win_rms = np.sqrt(
                np.maximum(seg_csum[64 : 64 + corr.size] - seg_csum[: corr.size], 1e-12)
            )
            norm_corr = corr_mag / (win_rms * _REF_LLTF_NORM)

            dual_score = norm_corr[:-64] + norm_corr[64:]
            m1_rel = int(np.argmax(dual_score))
            dual_peak = float(dual_score[m1_rel])
            if dual_peak < 0.82:
                # Not a genuine 802.11 L-LTF twin-symbol preamble
                idx_ptr = end_ptr + 1
                continue

            t1_abs = search_lo + m1_rel
            if t1_abs + 128 + SYM_LEN > n_total:
                pending_incomplete_start = max(skip_until_sample, max(0, t1_abs - 224))
                break

            # Fine CFO from phase progression between T1 (64) and T2 (64)
            t1_coarse = seg_derot[m1_rel : m1_rel + 64]
            t2_coarse = seg_derot[m1_rel + 64 : m1_rel + 128]
            fine_prod = complex(np.vdot(t1_coarse, t2_coarse))
            d_omega = math.atan2(fine_prod.imag, fine_prod.real) / 64.0
            omega_total = omega_coarse + d_omega
            cfo_hz = float(omega_total * fs / (2.0 * math.pi))

            # Extract CFO-corrected L-LTF symbols T1 & T2 from original sample stream
            ltf_pair = x[t1_abs : t1_abs + 128].astype(np.complex128)
            ltf_pair *= np.exp(-1j * omega_total * np.arange(128, dtype=np.float64))
            y1 = np.fft.fft(ltf_pair[:64]) * 0.125
            y2 = np.fft.fft(ltf_pair[64:128]) * 0.125
            h_raw = (
                0.5 * (y1[ACTIVE_FFT_BINS] + y2[ACTIVE_FFT_BINS]) * L_LTF_SEQ_52
            ).astype(np.complex64)
            h_san = sanitize_csi(h_raw)

            sig_var = float(np.mean(np.abs(ltf_pair) ** 2))
            rssi_dbm = float(10.0 * math.log10(max(sig_var, 1e-12)) - 45.0)
            snr_db = float(10.0 * math.log10(max(sig_var / noise_var, 1.0)))
            if snr_db < self.min_snr_db and dual_peak < 1.15:
                idx_ptr = end_ptr + 1
                continue

            # Decode L-SIG (symbol immediately following T2 at t1_abs + 128)
            lsig_start = t1_abs + 128
            decoded_ev, consumed_samples, truncated = self._decode_lsig_and_mac(
                x,
                t1_abs=t1_abs,
                lsig_start=lsig_start,
                omega_total=omega_total,
                h_est=h_raw.astype(np.complex128),
                h_san=h_san,
                timestamp=timestamp + max(0, t1_abs - 192) / fs,
                freq_hz=freq_hz,
                channel=ch,
                rssi_dbm=rssi_dbm,
                noise_floor_dbm=noise_floor_dbm,
                snr_db=snr_db,
                cfo_hz=cfo_hz,
                adc_clip_ratio=adc_clip_ratio,
                sample_rate=fs,
            )
            if truncated:
                pending_incomplete_start = max(skip_until_sample, max(0, t1_abs - 224))
                break
            if decoded_ev is not None:
                events.append(decoded_ev)
            skip_until_sample = lsig_start + max(consumed_samples, SYM_LEN)
            idx_ptr = end_ptr + 1

        if (
            pending_incomplete_start is not None
            and (n_total - pending_incomplete_start) <= 24_000
        ):
            keep_start = pending_incomplete_start
        else:
            keep_start = max(skip_until_sample, n_total - 416)
        return events, keep_start

    def _equalize_symbol(
        self,
        x: np.ndarray,
        sym_start: int,
        t1_abs: int,
        omega_total: float,
        h_est: np.ndarray,
        h_abs_sq: np.ndarray,
        v_idx: int,
    ) -> tuple[np.ndarray, float]:
        """CFO-correct, FFT, channel-equalize, and pilot-CPE track a single 80-sample OFDM symbol."""
        idx64 = sym_start + CP_LEN
        raw64 = x[idx64 : idx64 + 64].astype(np.complex128)
        sample_offsets = np.arange(
            idx64 - t1_abs, idx64 - t1_abs + 64, dtype=np.float64
        )
        raw64 *= np.exp(-1j * omega_total * sample_offsets)
        y52 = (np.fft.fft(raw64) * 0.125)[ACTIVE_FFT_BINS]

        eq52 = y52 * np.conj(h_est) / np.maximum(h_abs_sq, 1e-10)
        exp_pilots = BASE_PILOT_SIGNS * PILOT_POLARITY_127[v_idx % 127]
        pilot_corr = np.sum(
            eq52[PILOT_POS_IN_52] * exp_pilots * h_abs_sq[PILOT_POS_IN_52]
        )
        cpe = math.atan2(pilot_corr.imag, pilot_corr.real)
        eq52 *= complex(math.cos(cpe), -math.sin(cpe))

        data48 = eq52[DATA_POS_IN_52]
        qbpsk_ratio = float(
            np.sum(data48.imag**2) / max(float(np.sum(data48.real**2)), 1e-12)
        )
        return data48, qbpsk_ratio

    def _equalize_symbols_batch(
        self,
        x: np.ndarray,
        data_start: int,
        n_sym: int,
        t1_abs: int,
        omega_total: float,
        h_est: np.ndarray,
        h_abs_sq: np.ndarray,
    ) -> np.ndarray:
        """Vectorized CFO derotation, 64-bin FFT, LS equalization, & pilot-CPE tracking."""
        raw_syms = x[data_start : data_start + n_sym * SYM_LEN].reshape(n_sym, SYM_LEN)[
            :, CP_LEN:
        ]
        raw64 = raw_syms.astype(np.complex128)
        sym_bases = (data_start + CP_LEN - t1_abs) + np.arange(n_sym, dtype=np.float64)[
            :, None
        ] * SYM_LEN
        offsets = sym_bases + np.arange(64, dtype=np.float64)[None, :]
        raw64 *= np.exp((-1j * omega_total) * offsets)

        y52 = (np.fft.fft(raw64, axis=1) * 0.125)[:, ACTIVE_FFT_BINS]
        inv_h = (np.conj(h_est) / np.maximum(h_abs_sq, 1e-10))[None, :]
        eq52 = y52 * inv_h

        pol = PILOT_POLARITY_127[(1 + np.arange(n_sym, dtype=np.intp)) % 127][:, None]
        exp_pilots = pol * BASE_PILOT_SIGNS[None, :]
        pilot_w = h_abs_sq[PILOT_POS_IN_52][None, :]
        pilot_corr = np.sum(eq52[:, PILOT_POS_IN_52] * exp_pilots * pilot_w, axis=1)
        cpe = np.angle(pilot_corr)
        eq52 *= np.exp(-1j * cpe)[:, None]
        return eq52[:, DATA_POS_IN_52]

    def _decode_lsig_and_mac(
        self,
        x: np.ndarray,
        *,
        t1_abs: int,
        lsig_start: int,
        omega_total: float,
        h_est: np.ndarray,
        h_san: np.ndarray,
        timestamp: float,
        freq_hz: float,
        channel: int,
        rssi_dbm: float,
        noise_floor_dbm: float,
        snr_db: float,
        cfo_hz: float,
        adc_clip_ratio: float,
        sample_rate: float,
    ) -> tuple[PacketEvent | None, int, bool]:
        h_abs_sq = h_est.real**2 + h_est.imag**2
        h_data_w = h_abs_sq[DATA_POS_IN_52] / max(float(np.mean(h_abs_sq)), 1e-10)

        lsig_data, _ = self._equalize_symbol(
            x, lsig_start, t1_abs, omega_total, h_est, h_abs_sq, v_idx=0
        )
        _, inv_48 = _INTERLEAVER_TABLES[48]
        lsig_llrs = _demap_llrs(lsig_data, h_data_w, 1)[inv_48]
        la, lb = _depuncture_llrs(lsig_llrs, "1/2")
        lsig_bits = _viterbi_decode_serial(la, lb, zero_terminated=True)

        rate_tuple = (
            int(lsig_bits[0]),
            int(lsig_bits[1]),
            int(lsig_bits[2]),
            int(lsig_bits[3]),
        )
        reserved_ok = int(lsig_bits[4]) == 0
        parity_ok = (int(np.sum(lsig_bits[:18])) & 1) == 0
        tail_ok = int(np.sum(lsig_bits[18:24])) == 0

        if not (
            reserved_ok and parity_ok and tail_ok and rate_tuple in LSIG_RATE_TABLE
        ):
            return None, SYM_LEN, False

        psdu_len = 0
        for b in range(12):
            psdu_len |= int(lsig_bits[5 + b]) << b

        if psdu_len < 14 or psdu_len > 2350:
            return None, SYM_LEN, False

        _, n_bpsc, n_cbps, n_dbps, code_rate = LSIG_RATE_TABLE[rate_tuple]
        n_sym_full = math.ceil((16 + psdu_len * 8 + 6) / n_dbps)
        total_burst_samples = (1 + n_sym_full) * SYM_LEN

        # Check whether symbol 1 is 802.11n HT-MF (90-deg rotated QBPSK HT-SIG1 after 6 Mbps L-SIG)
        data_start = lsig_start + SYM_LEN
        if data_start + SYM_LEN > x.size:
            return None, total_burst_samples, True

        if n_bpsc == 1:
            _, qbpsk_ratio = self._equalize_symbol(
                x, data_start, t1_abs, omega_total, h_est, h_abs_sq, v_idx=1
            )
            if qbpsk_ratio > 2.4:
                # Valid 802.11n/ac HT-MF preamble: L-LTF CSI is valid, payload is HT MCS
                aps = sorted(self._tracker._known_aps_by_ch.get(channel, set()))
                bssid_ht = aps[0] if len(aps) == 1 else "HT:OFDM:PREAMBLE"
                ev = PacketEvent(
                    timestamp=timestamp,
                    freq_hz=freq_hz,
                    channel=channel,
                    bssid=bssid_ht,
                    receiver_addr=None,
                    frame_type="DATA_PREAMBLE",
                    rssi_dbm=rssi_dbm,
                    noise_floor_dbm=noise_floor_dbm,
                    snr_db=snr_db,
                    csi=h_est.astype(np.complex64),
                    csi_sanitized=h_san,
                    cfo_hz=cfo_hz,
                    tsf_us=None,
                    tsf_drift_ppm=None,
                    seq_num=None,
                    seq_gap=0,
                    fcs_ok=True,
                    adc_clip_ratio=float(adc_clip_ratio),
                    attributed_via_sifs=False,
                )
                return ev, min(total_burst_samples, x.size - lsig_start), False

        verify_full_fcs = psdu_len <= self.max_full_fcs_bytes
        target_bytes = psdu_len if verify_full_fcs else min(psdu_len, 40)
        n_sym_decode = math.ceil((16 + target_bytes * 8 + 6) / n_dbps)

        if data_start + n_sym_decode * SYM_LEN > x.size:
            return None, total_burst_samples, True

        _, inv_perm = _INTERLEAVER_TABLES[n_cbps]
        data48_batch = self._equalize_symbols_batch(
            x, data_start, n_sym_decode, t1_abs, omega_total, h_est, h_abs_sq
        )
        w_batch = np.broadcast_to(h_data_w, (n_sym_decode, 48)).reshape(-1)
        raw_llrs = _demap_llrs(data48_batch.reshape(-1), w_batch, n_bpsc).reshape(
            n_sym_decode, n_cbps
        )
        all_llrs = raw_llrs[:, inv_perm].reshape(-1)

        la_d, lb_d = _depuncture_llrs(all_llrs, code_rate)
        scrambled_bits = _viterbi_decode(la_d, lb_d, zero_terminated=verify_full_fcs)
        descrambled = _descramble_ieee80211(scrambled_bits)

        mpdu_bits = descrambled[16 : 16 + target_bytes * 8]
        mpdu_bytes = _bits_to_bytes_lsb(mpdu_bits)
        mac_info = parse_80211_mac_frame(mpdu_bytes, fcs_present=verify_full_fcs)

        burst_end_ts = timestamp + (320 + total_burst_samples) / sample_rate
        bssid, rx_mac, via_sifs, seq_gap, tsf_ppm = self._tracker.attribute_and_track(
            mac_info,
            timestamp=timestamp,
            burst_end_ts=burst_end_ts,
            channel=channel,
        )

        return (
            PacketEvent(
                timestamp=timestamp,
                freq_hz=freq_hz,
                channel=channel,
                bssid=bssid,
                receiver_addr=rx_mac,
                frame_type=mac_info.frame_type,
                rssi_dbm=rssi_dbm,
                noise_floor_dbm=noise_floor_dbm,
                snr_db=snr_db,
                csi=h_est.astype(np.complex64),
                csi_sanitized=h_san,
                cfo_hz=cfo_hz,
                tsf_us=mac_info.tsf_us,
                tsf_drift_ppm=tsf_ppm,
                seq_num=mac_info.seq_num,
                seq_gap=seq_gap,
                fcs_ok=mac_info.fcs_ok,
                adc_clip_ratio=float(adc_clip_ratio),
                attributed_via_sifs=via_sifs,
            ),
            total_burst_samples,
            False,
        )


# ---------------------------------------------------------------------------
# Pure-struct Radiotap & 802.11 PCAP / PCAPNG Reader & Writer
# ---------------------------------------------------------------------------

DLT_IEEE802_11: int = 105
DLT_IEEE802_11_RADIO: int = 127


def _align_offset(off: int, align: int) -> int:
    rem = off % align
    return off if rem == 0 else off + (align - rem)


def _synthetic_csi_from_rssi(
    rssi_dbm: float, freq_hz: float, bssid: str, ts: float
) -> np.ndarray:
    """Construct a deterministic 52-subcarrier flat+multipath CSI vector for PCAP-only records."""
    amp = float(10.0 ** ((rssi_dbm + 48.0) / 20.0))
    amp = min(max(amp, 0.08), 2.0)
    seed_val = zlib.crc32(bssid.encode("ascii", errors="ignore")) & 0xFFFF
    k = np.asarray(SUBCARRIER_INDICES, dtype=np.float64)
    ripple = 1.0 + 0.12 * np.cos(0.18 * k + 0.01 * seed_val + 0.25 * ts)
    phase = 0.08 * np.sin(0.15 * k + 0.02 * seed_val)
    return (amp * ripple * np.exp(1j * phase)).astype(np.complex64)


def _parse_radiotap_header(buf: bytes) -> tuple[int, dict[str, Any]]:
    """Parse IEEE 802.11 Radiotap header using Python ``struct`` (without scapy)."""
    if len(buf) < 8:
        raise ValueError("Truncated Radiotap header")
    ver, _pad, it_len = struct.unpack_from("<BBH", buf, 0)
    if ver != 0 or it_len < 8 or it_len > len(buf):
        raise ValueError(f"Invalid Radiotap header version={ver} len={it_len}")

    # Read 32-bit present bitmasks (bit 31 indicates another extension bitmap word follows)
    present_words: list[int] = []
    off = 4
    while off + 4 <= it_len:
        word = struct.unpack_from("<I", buf, off)[0]
        present_words.append(word)
        off += 4
        if (word & 0x80000000) == 0:
            break

    meta: dict[str, Any] = {}
    p0 = present_words[0] if present_words else 0

    # Parse standard Radiotap fields in strict bit-index order with natural alignment
    if p0 & (1 << 0):  # TSFT: uint64 (align 8)
        off = _align_offset(off, 8)
        if off + 8 <= it_len:
            meta["tsft_us"] = struct.unpack_from("<Q", buf, off)[0]
            off += 8
    if (
        p0 & (1 << 1)
    ) and off + 1 <= it_len:  # Flags: uint8 (0x10 = FCS at end, 0x40 = bad FCS)
        flags = buf[off]
        meta["fcs_present"] = bool(flags & 0x10)
        meta["bad_fcs"] = bool(flags & 0x40)
        off += 1
    if (p0 & (1 << 2)) and off + 1 <= it_len:  # Rate: uint8 (500 kbps units)
        meta["rate_mbps"] = 0.5 * buf[off]
        off += 1
    if p0 & (1 << 3):  # Channel: uint16 freq_mhz + uint16 flags (align 2)
        off = _align_offset(off, 2)
        if off + 4 <= it_len:
            freq_mhz, _ch_flags = struct.unpack_from("<HH", buf, off)
            meta["freq_hz"] = float(freq_mhz) * 1e6
            off += 4
    if p0 & (1 << 4):  # FHSS: uint16 (align 2)
        off = _align_offset(off, 2)
        off += 2
    if (p0 & (1 << 5)) and off + 1 <= it_len:  # Antenna signal: int8 dBm
        meta["rssi_dbm"] = float(struct.unpack_from("<b", buf, off)[0])
        off += 1
    if (p0 & (1 << 6)) and off + 1 <= it_len:  # Antenna noise: int8 dBm
        meta["noise_dbm"] = float(struct.unpack_from("<b", buf, off)[0])
        off += 1

    return it_len, meta


def write_radiotap_pcap(
    path: str | Path,
    records: Sequence[tuple[float, bytes, float, float, bool]],
) -> Path:
    """Write ``(timestamp_s, mpdu_with_fcs, freq_hz, rssi_dbm, bad_fcs)`` tuples to Radiotap PCAP."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Global PCAP header: magic 0xA1B2C3D4, v2.4, tz=0, sigfigs=0, snaplen=65535, DLT=127
    chunks: list[bytes] = [
        struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, DLT_IEEE802_11_RADIO)
    ]
    # Present bits: Flags (1) | Rate (2) | Channel (3) | Signal dBm (5) | Noise dBm (6)
    present = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 6)
    for ts, mpdu, freq_hz, rssi_dbm, bad_fcs in records:
        flags = 0x10 | (0x40 if bad_fcs else 0x00)
        freq_mhz = round(float(freq_hz) / 1e6)
        sig_i8 = max(-120, min(0, round(float(rssi_dbm))))
        rtap = struct.pack("<BBHI", 0, 0, 16, present) + struct.pack(
            "<BBHHbb", flags, 12, freq_mhz, 0x00A0, sig_i8, -92
        )
        pkt = rtap + mpdu
        total_us = max(0, round(float(ts) * 1_000_000))
        ts_sec, ts_usec = divmod(total_us, 1_000_000)
        chunks.append(struct.pack("<IIII", ts_sec, ts_usec, len(pkt), len(pkt)) + pkt)
    out.write_bytes(b"".join(chunks))
    return out


def iter_pcap_packet_events(
    pcap_path: str | Path,
    *,
    default_channel: int = 6,
    duration_s: float | None = None,
) -> Iterator[PacketEvent]:
    """Yield ``PacketEvent`` objects from a Radiotap or raw 802.11 ``.pcap`` / ``.pcapng`` file."""
    raw_file = Path(pcap_path).read_bytes()
    if len(raw_file) < 24:
        return

    tracker = _LinkTelemetryTracker()
    default_freq = channel_to_freq_hz(default_channel)
    t0: float | None = None

    magic = struct.unpack_from("<I", raw_file, 0)[0]
    raw_packets: list[tuple[float, int, bytes]] = []

    if magic in (0xA1B2C3D4, 0xA1B23C4D, 0xD4C3B2A1, 0x4D3CB2A1):
        endian = "<" if magic in (0xA1B2C3D4, 0xA1B23C4D) else ">"
        ns_res = magic in (0xA1B23C4D, 0x4D3CB2A1)
        _m, _maj, _min, _tz, _sig, _snap, dlt = struct.unpack_from(
            f"{endian}IHHiIII", raw_file, 0
        )
        pos = 24
        scale = 1e-9 if ns_res else 1e-6
        while pos + 16 <= len(raw_file):
            ts_s, ts_sub, incl_len, _orig_len = struct.unpack_from(
                f"{endian}IIII", raw_file, pos
            )
            pos += 16
            if pos + incl_len > len(raw_file):
                break
            raw_packets.append(
                (
                    float(ts_s) + float(ts_sub) * scale,
                    int(dlt),
                    raw_file[pos : pos + incl_len],
                )
            )
            pos += incl_len
    elif magic == 0x0A0D0D0A:
        # PCAPNG parser (SHB 0x0A0D0D0A with LE/BE byte-order magic, IDB 0x00000001 + if_tsresol, EPB 0x00000006)
        pos = 0
        endian = "<"
        if_dlts: list[int] = []
        if_ts_scales: list[float] = []
        while pos + 12 <= len(raw_file):
            btype_probe = struct.unpack_from("<I", raw_file, pos)[0]
            if btype_probe == 0x0A0D0D0A:
                if pos + 16 > len(raw_file):
                    break
                bo_magic = struct.unpack_from("<I", raw_file, pos + 8)[0]
                if bo_magic == 0x1A2B3C4D:
                    endian = "<"
                elif bo_magic == 0x4D3C2B1A:
                    endian = ">"
                else:
                    raise ValueError(
                        f"Unrecognized PCAPNG SHB byte-order magic: 0x{bo_magic:08X}"
                    )
                blen = struct.unpack_from(f"{endian}I", raw_file, pos + 4)[0]
                if blen < 28 or pos + blen > len(raw_file):
                    break
                if_dlts.clear()
                if_ts_scales.clear()
                pos += blen
                continue

            btype, blen = struct.unpack_from(f"{endian}II", raw_file, pos)
            if blen < 12 or pos + blen > len(raw_file):
                break
            if btype == 0x00000001 and blen >= 20:
                link_type = struct.unpack_from(f"{endian}H", raw_file, pos + 8)[0]
                ts_scale = 1e-6
                opt_pos = pos + 16
                opt_end = pos + blen - 4
                while opt_pos + 4 <= opt_end:
                    opt_code, opt_len = struct.unpack_from(
                        f"{endian}HH", raw_file, opt_pos
                    )
                    opt_pos += 4
                    if opt_code == 0:
                        break
                    if opt_pos + opt_len > opt_end:
                        break
                    if opt_code == 9 and opt_len >= 1:
                        resol_byte = raw_file[opt_pos]
                        if (resol_byte & 0x80) == 0:
                            ts_scale = 10.0 ** (-int(resol_byte))
                        else:
                            ts_scale = 2.0 ** (-int(resol_byte & 0x7F))
                    opt_pos += _align_offset(opt_len, 4)
                if_dlts.append(int(link_type))
                if_ts_scales.append(float(ts_scale))
            elif btype == 0x00000006 and blen >= 32:
                if_id, ts_hi, ts_lo, cap_len, _pkt_len = struct.unpack_from(
                    f"{endian}IIIII", raw_file, pos + 8
                )
                scale = if_ts_scales[if_id] if if_id < len(if_ts_scales) else 1e-6
                ts64 = ((ts_hi << 32) | ts_lo) * scale
                dlt = if_dlts[if_id] if if_id < len(if_dlts) else DLT_IEEE802_11_RADIO
                pkt_end = min(pos + 28 + cap_len, pos + blen - 4)
                if pkt_end > pos + 28:
                    pkt_bytes = raw_file[pos + 28 : pkt_end]
                    raw_packets.append((float(ts64), dlt, pkt_bytes))
            pos += blen
    else:
        raise ValueError(f"Unrecognized PCAP/PCAPNG magic header: 0x{magic:08X}")

    for ts, dlt, pkt in raw_packets:
        if t0 is None:
            t0 = ts
        if duration_s is not None and (ts - t0) > duration_s:
            break

        meta: dict[str, Any] = {}
        if dlt == DLT_IEEE802_11_RADIO:
            try:
                rtap_len, meta = _parse_radiotap_header(pkt)
            except ValueError:
                continue
            mpdu = pkt[rtap_len:]
        elif dlt == DLT_IEEE802_11:
            mpdu = pkt
            meta["fcs_present"] = True
        else:
            continue

        freq_hz = float(meta.get("freq_hz", default_freq))
        try:
            ch = freq_hz_to_channel(freq_hz)
        except ValueError:
            ch = default_channel
            freq_hz = default_freq

        rssi_dbm = float(meta.get("rssi_dbm", -58.0))
        noise_dbm = float(meta.get("noise_dbm", -92.0))
        fcs_present = bool(meta.get("fcs_present", True))
        bad_fcs = bool(meta.get("bad_fcs", False))

        info = parse_80211_mac_frame(
            mpdu, fcs_present=fcs_present, fcs_known_bad=bad_fcs
        )
        rate_mbps = max(float(meta.get("rate_mbps", 6.0)), 1.0)
        airtime_s = 20e-6 + (len(mpdu) * 8.0) / (rate_mbps * 1e6)
        bssid, rx_mac, via_sifs, seq_gap, tsf_ppm = tracker.attribute_and_track(
            info,
            timestamp=ts,
            burst_end_ts=ts + airtime_s,
            channel=ch,
        )

        csi_raw = _synthetic_csi_from_rssi(rssi_dbm, freq_hz, bssid, ts)
        csi_san = sanitize_csi(csi_raw)
        yield PacketEvent(
            timestamp=ts,
            freq_hz=freq_hz,
            channel=ch,
            bssid=bssid,
            receiver_addr=rx_mac,
            frame_type=info.frame_type,
            rssi_dbm=rssi_dbm,
            noise_floor_dbm=noise_dbm,
            snr_db=rssi_dbm - noise_dbm,
            csi=csi_raw,
            csi_sanitized=csi_san,
            cfo_hz=0.0,
            tsf_us=info.tsf_us,
            tsf_drift_ppm=tsf_ppm,
            seq_num=info.seq_num,
            seq_gap=seq_gap,
            fcs_ok=info.fcs_ok,
            adc_clip_ratio=0.0,
            attributed_via_sifs=via_sifs,
        )


# ---------------------------------------------------------------------------
# Pure-struct + PMT GNU Radio ZMQ PDU Deserializer
# ---------------------------------------------------------------------------

_PMT_UVI_SPECS: dict[int, tuple[int, str]] = {
    0x00: (1, "u8"),
    0x01: (1, ">i1"),
    0x02: (2, ">u2"),
    0x03: (2, ">i2"),
    0x04: (4, ">u4"),
    0x05: (4, ">i4"),
    0x06: (8, ">u8"),
    0x07: (8, ">i8"),
    0x08: (4, ">f4"),
    0x09: (8, ">f8"),
    0x0A: (8, "c32"),
    0x0B: (16, "c64"),
}


def _parse_pmt_stream(buf: bytes, pos: int = 0, depth: int = 0) -> tuple[Any, int]:
    """Deserialize a GNU Radio PMT binary stream using Python ``struct``."""
    if depth > 64 or pos >= len(buf):
        raise ValueError("Truncated or deeply nested PMT stream")
    tag = buf[pos]
    pos += 1

    if tag == 0x00:  # PST_TRUE
        return True, pos
    if tag == 0x01:  # PST_FALSE
        return False, pos
    if tag == 0x06:  # PST_NULL
        return None, pos
    if tag == 0x02:  # PST_SYMBOL
        if pos + 2 > len(buf):
            raise ValueError("Truncated PMT symbol")
        slen = struct.unpack_from(">H", buf, pos)[0]
        pos += 2
        if pos + slen > len(buf):
            raise ValueError("Truncated PMT symbol body")
        return buf[pos : pos + slen].decode("utf-8", errors="replace"), pos + slen
    if tag == 0x03:  # PST_INT32
        if pos + 4 > len(buf):
            raise ValueError("Truncated PMT int32")
        return int(struct.unpack_from(">i", buf, pos)[0]), pos + 4
    if tag == 0x0B:  # PST_UINT64
        if pos + 8 > len(buf):
            raise ValueError("Truncated PMT uint64")
        return int(struct.unpack_from(">Q", buf, pos)[0]), pos + 8
    if tag == 0x04:  # PST_DOUBLE
        if pos + 8 > len(buf):
            raise ValueError("Truncated PMT double")
        return float(struct.unpack_from(">d", buf, pos)[0]), pos + 8
    if tag == 0x05:  # PST_COMPLEX
        if pos + 16 > len(buf):
            raise ValueError("Truncated PMT complex")
        re, im = struct.unpack_from(">dd", buf, pos)
        return complex(re, im), pos + 16
    if tag == 0x07:  # PST_PAIR
        car, pos = _parse_pmt_stream(buf, pos, depth + 1)
        cdr, pos = _parse_pmt_stream(buf, pos, depth + 1)
        return (car, cdr), pos
    if (
        tag == 0x09
    ):  # PST_DICT (recursive chain of 0x09 + pair(k, v) + rest_dict_or_null)
        kv_pair, pos = _parse_pmt_stream(buf, pos, depth + 1)
        rest, pos = _parse_pmt_stream(buf, pos, depth + 1)
        out_dict: dict[str, Any] = {}
        if isinstance(rest, dict):
            out_dict.update(rest)
        if isinstance(kv_pair, tuple) and len(kv_pair) == 2:
            out_dict[str(kv_pair[0])] = kv_pair[1]
        return out_dict, pos
    if tag in (0x08, 0x0C):  # PST_VECTOR / PST_TUPLE
        if pos + 4 > len(buf):
            raise ValueError("Truncated PMT vector/tuple")
        n_items = struct.unpack_from(">I", buf, pos)[0]
        pos += 4
        if n_items > 65536:
            raise ValueError("Unreasonable PMT vector length")
        items = []
        for _ in range(n_items):
            item, pos = _parse_pmt_stream(buf, pos, depth + 1)
            items.append(item)
        return items, pos
    if tag == 0x0A:  # PST_UNIFORM_VECTOR
        if pos + 6 > len(buf):
            raise ValueError("Truncated PMT uniform vector header")
        uvi = buf[pos]
        n_items = struct.unpack_from(">I", buf, pos + 1)[0]
        npad = buf[pos + 5]
        pos += 6 + npad
        if uvi not in _PMT_UVI_SPECS:
            raise ValueError(f"Unsupported PMT uniform vector subtype: 0x{uvi:02X}")
        elem_bytes, kind = _PMT_UVI_SPECS[uvi]
        byte_len = n_items * elem_bytes
        if pos + byte_len > len(buf):
            raise ValueError("Truncated PMT uniform vector payload")
        raw_seg = buf[pos : pos + byte_len]
        pos += byte_len
        if kind == "u8":
            return raw_seg, pos
        if kind == "c32":
            f32_be = np.frombuffer(raw_seg, dtype=">f4").astype(np.float32)
            return (f32_be[0::2] + 1j * f32_be[1::2]).astype(np.complex64), pos
        if kind == "c64":
            f64_be = np.frombuffer(raw_seg, dtype=">f8").astype(np.float64)
            return (f64_be[0::2] + 1j * f64_be[1::2]).astype(np.complex64), pos
        return np.frombuffer(raw_seg, dtype=kind).astype(np.float64), pos
    raise ValueError(f"Unsupported PMT tag: 0x{tag:02X}")


def decode_gnuradio_pmt_pdu(payload: bytes) -> tuple[dict[str, Any], bytes] | None:
    """Decode a GNU Radio ``zeromq.pub_msg_sink`` serialized PMT PDU ``cons(meta, u8vector)``."""
    if len(payload) < 8 or payload[0] not in (0x07, 0x0A):
        return None
    try:
        parsed, end_pos = _parse_pmt_stream(payload, 0)
    except ValueError:
        return None
    if end_pos < len(payload) - 4:
        return None
    if isinstance(parsed, bytes):
        return {}, parsed
    if isinstance(parsed, tuple) and len(parsed) == 2 and isinstance(parsed[1], bytes):
        meta = parsed[0] if isinstance(parsed[0], dict) else {}
        return meta, parsed[1]
    return None


# ---------------------------------------------------------------------------
# Unified Passive 802.11 Packet & CSI Capture Engine
# ---------------------------------------------------------------------------


class HackRFPacketCapture:
    """Unified strictly passive (0 dBm RX-only) 802.11 PacketEvent & L-LTF CSI capture engine.

    Supports:
      - ``backend="auto"|"soapy"|"libhackrf"|"osmosdr"|"hackrf_transfer"``: live 20 MSPS HackRF RX
      - ``backend="iq_file"``: recorded ``.cs8`` / ``.cf32`` / ``.npy`` baseband I/Q replay
      - ``backend="pcap"``: Radiotap / 802.11 ``.pcap`` & ``.pcapng`` offline replay
      - ``backend="gr_zmq"``: GNU Radio ``gr-ieee802-11`` ZMQ SUB PDU socket stream
    """

    def __init__(
        self,
        channels: Sequence[int] = (1, 6, 11),
        dwell_ms: float = 80.0,
        settle_ms: float = 3.0,
        sample_rate: float = 20e6,
        lna_db: int = 24,
        vga_db: int = 20,
        amp_enable: bool = False,
        bias_tee_enable: bool = False,
        dc_offset_removal: bool = True,
        backend: str = "auto",
        zmq_endpoint: str | None = None,
        pcap_path: str | Path | None = None,
        iq_path: str | Path | None = None,
        loop_file: bool = False,
    ) -> None:
        if amp_enable:
            raise ValueError(
                "Passive 0 dBm RX-only safety lock: HackRF RF AMP must remain False."
            )
        if bias_tee_enable:
            raise ValueError(
                "Passive 0 dBm RX-only safety lock: antenna bias-tee must remain False."
            )

        self.channels: tuple[int, ...] = tuple(int(c) for c in (channels or (1, 6, 11)))
        self.dwell_ms = float(dwell_ms)
        self.settle_ms = float(settle_ms)
        self.sample_rate = float(sample_rate)
        self.lna_db = int(lna_db)
        self.vga_db = int(vga_db)
        self.amp_enable: bool = False
        self.bias_tee_enable: bool = False
        self.dc_offset_removal: bool = bool(dc_offset_removal)
        self.zmq_endpoint = zmq_endpoint or "tcp://127.0.0.1:5555"
        self.pcap_path = Path(pcap_path) if pcap_path is not None else None
        self.iq_path = Path(iq_path) if iq_path is not None else None
        self.loop_file = bool(loop_file)

        self._iq_source: HackRFIQSource | None = None
        self._zmq_sock: Any = None
        self._zmq_ctx: Any = None
        self._detector = NativeOFDMBurstDetector(sample_rate=self.sample_rate)

        key = backend.strip().lower()
        if key == "pcap":
            if self.pcap_path is None or not self.pcap_path.is_file():
                raise FileNotFoundError(
                    f"PCAP file required for backend='pcap', got {self.pcap_path}"
                )
            self.active_backend = "pcap"
        elif key == "gr_zmq":
            self._open_zmq()
            self.active_backend = f"gr_zmq({self.zmq_endpoint})"
        else:
            self._iq_source = HackRFIQSource(
                channels=self.channels,
                dwell_ms=self.dwell_ms,
                settle_ms=self.settle_ms,
                sample_rate=self.sample_rate,
                lna_db=self.lna_db,
                vga_db=self.vga_db,
                amp_enable=False,
                bias_tee_enable=False,
                dc_offset_removal=self.dc_offset_removal,
                backend=key,
                iq_path=self.iq_path,
                loop_file=self.loop_file,
            )
            self.active_backend = self._iq_source.active_backend

    @property
    def max_adc_clip_ratio(self) -> float:
        if self._iq_source is not None:
            return float(self._iq_source.max_adc_clip_ratio)
        return 0.0

    def _open_zmq(self) -> None:
        try:
            import zmq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for --backend gr_zmq") from exc
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVTIMEO, 100)
        sock.connect(self.zmq_endpoint)
        self._zmq_ctx = ctx
        self._zmq_sock = sock

    def _iter_zmq_events(
        self,
        duration_s: float | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> Iterator[PacketEvent]:
        assert self._zmq_sock is not None
        import zmq  # type: ignore[import-not-found]

        tracker = _LinkTelemetryTracker()
        start_mono = time.monotonic()
        default_ch = self.channels[0]
        default_freq = channel_to_freq_hz(default_ch)

        while True:
            if stop_check is not None and stop_check():
                break
            if duration_s is not None and (time.monotonic() - start_mono) >= duration_s:
                break
            try:
                payload = self._zmq_sock.recv()
            except zmq.Again:
                continue

            now_ts = time.time()
            # Format 1: UTF-8 JSON serialized PacketEvent
            if payload.startswith(b"{"):
                with contextlib.suppress(
                    UnicodeDecodeError, ValueError, KeyError, TypeError
                ):
                    d = json.loads(payload.decode("utf-8"))
                    yield PacketEvent.from_dict(d)
                    continue

            # Format 2: GNU Radio zeromq.pub_msg_sink serialized PMT PDU cons(meta, u8vector)
            pmt_decoded = decode_gnuradio_pmt_pdu(payload)
            csi_override: np.ndarray | None = None
            fcs_present = True
            fcs_known_bad = False
            freq_hz = default_freq
            rssi_dbm = -56.0
            noise_dbm = -92.0
            cfo_hz = 0.0

            if pmt_decoded is not None:
                pmt_meta, mpdu = pmt_decoded
                norm_meta = {str(k).strip().lower(): v for k, v in pmt_meta.items()}
                for fk in (
                    "nominal frequency",
                    "frequency",
                    "freq",
                    "freq_hz",
                    "chan_freq",
                ):
                    if fk in norm_meta:
                        fv = float(norm_meta[fk])
                        freq_hz = fv * 1e6 if fv < 1e5 else fv
                        break
                if "noise" in norm_meta:
                    noise_dbm = float(norm_meta["noise"])
                if "rssi" in norm_meta:
                    rssi_dbm = float(norm_meta["rssi"])
                elif "dbm_antsignal" in norm_meta:
                    rssi_dbm = float(norm_meta["dbm_antsignal"])
                elif "snr" in norm_meta:
                    rssi_dbm = noise_dbm + float(norm_meta["snr"])
                for ck in ("freq offset", "freq_offset", "cfo", "cfo_hz"):
                    if ck in norm_meta:
                        cfo_hz = float(norm_meta[ck])
                        break
                if "csi" in norm_meta:
                    csi_cand = np.asarray(norm_meta["csi"], dtype=np.complex64).reshape(
                        -1
                    )
                    if csi_cand.size == 52:
                        csi_override = csi_cand
                    elif csi_cand.size == 64:
                        csi_override = csi_cand[ACTIVE_FFT_BINS]
                fcs_known_bad = bool(
                    norm_meta.get("bad_fcs", False) or norm_meta.get("fcs_error", False)
                )
                if "fcs_present" in norm_meta:
                    fcs_present = bool(norm_meta["fcs_present"])
                elif len(mpdu) >= 14:
                    exp_crc = zlib.crc32(mpdu[:-4]) & 0xFFFFFFFF
                    act_crc = struct.unpack_from("<I", mpdu, len(mpdu) - 4)[0]
                    # gr-ieee802-11 decode_mac strips trailing 4-byte FCS after verifying CRC-32
                    fcs_present = exp_crc == act_crc
                else:
                    fcs_present = False
            else:
                # Format 3: Raw Radiotap or 802.11 MPDU bytes over ZMQ
                mpdu = payload
                if len(payload) >= 16 and payload[0] == 0 and payload[1] == 0:
                    try:
                        rtap_len, meta = _parse_radiotap_header(payload)
                        mpdu = payload[rtap_len:]
                        freq_hz = float(meta.get("freq_hz", default_freq))
                        rssi_dbm = float(meta.get("rssi_dbm", -56.0))
                        noise_dbm = float(meta.get("noise_dbm", -92.0))
                        fcs_present = bool(meta.get("fcs_present", True))
                        fcs_known_bad = bool(meta.get("bad_fcs", False))
                    except ValueError:
                        pass

            info = parse_80211_mac_frame(
                mpdu, fcs_present=fcs_present, fcs_known_bad=fcs_known_bad
            )
            try:
                ch = freq_hz_to_channel(freq_hz)
            except ValueError:
                ch = default_ch
                freq_hz = default_freq
            bssid, rx_mac, via_sifs, seq_gap, tsf_ppm = tracker.attribute_and_track(
                info, timestamp=now_ts, burst_end_ts=now_ts + 120e-6, channel=ch
            )
            csi_raw = (
                csi_override
                if csi_override is not None
                else _synthetic_csi_from_rssi(rssi_dbm, freq_hz, bssid, now_ts)
            )
            yield PacketEvent(
                timestamp=now_ts,
                freq_hz=freq_hz,
                channel=ch,
                bssid=bssid,
                receiver_addr=rx_mac,
                frame_type=info.frame_type,
                rssi_dbm=rssi_dbm,
                noise_floor_dbm=noise_dbm,
                snr_db=rssi_dbm - noise_dbm,
                csi=csi_raw,
                csi_sanitized=sanitize_csi(csi_raw),
                cfo_hz=cfo_hz,
                tsf_us=info.tsf_us,
                tsf_drift_ppm=tsf_ppm,
                seq_num=info.seq_num,
                seq_gap=seq_gap,
                fcs_ok=info.fcs_ok,
                adc_clip_ratio=0.0,
                attributed_via_sifs=via_sifs,
            )

    def iter_events(
        self,
        duration_s: float | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> Iterator[PacketEvent]:
        """Stream decoded cleartext 802.11 ``PacketEvent`` observations."""
        if self.active_backend == "pcap":
            assert self.pcap_path is not None
            for ev in iter_pcap_packet_events(
                self.pcap_path,
                default_channel=self.channels[0],
                duration_s=duration_s,
            ):
                if stop_check is not None and stop_check():
                    break
                yield ev
            return

        if self._zmq_sock is not None:
            yield from self._iter_zmq_events(
                duration_s=duration_s, stop_check=stop_check
            )
            return

        assert self._iq_source is not None
        for chunk in self._iq_source.iter_chunks(
            duration_s=duration_s, chunk_ms=20.0, stop_check=stop_check
        ):
            for ev in self._detector.process_chunk(chunk):
                yield ev

    def close(self) -> None:
        if self._iq_source is not None:
            self._iq_source.close()
            self._iq_source = None
        if self._zmq_sock is not None:
            with contextlib.suppress(OSError, RuntimeError, AttributeError):
                self._zmq_sock.close(0)
            self._zmq_sock = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
