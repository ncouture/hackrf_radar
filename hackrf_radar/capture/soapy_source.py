"""Strictly passive (0 dBm RX-only) 20 MSPS HackRF One I/Q capture & channel hopper.

Supports SoapySDR primary backend with automatic live fallback to:
  1. SoapySDR (``driver=hackrf`` / ``gnuradio.soapy``)
  2. ``ctypes.CDLL("libhackrf.so.0")`` direct USB RX streaming
  3. ``osmosdr.source(args="numchan=1 hackrf=0,bias=0")``
  4. ``hackrf_transfer -r -`` subprocess RX streaming
  5. ``.cs8`` / ``.cf32`` / ``.npy`` recorded I/Q file replay
"""

from __future__ import annotations

import collections
import contextlib
import ctypes
import os
import select
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np

from .packet_event import WIFI_CHANNELS_HZ, channel_to_freq_hz

# Supported HackRF MAX2837 discrete baseband filter bandwidths (Hz)
HACKRF_BASEBAND_BWS_HZ: tuple[int, ...] = (
    1_750_000,
    2_500_000,
    3_500_000,
    5_000_000,
    5_500_000,
    6_000_000,
    7_000_000,
    8_000_000,
    9_000_000,
    10_000_000,
    12_000_000,
    14_000_000,
    15_000_000,
    20_000_000,
    24_000_000,
    28_000_000,
)

ADC_CLIP_THRESHOLD_INT8: int = 120
ADC_CLIP_THRESHOLD_NORM: float = 120.0 / 128.0


def quantize_lna_gain(lna_db: float) -> int:
    """Clamp and quantize HackRF RX IF LNA gain to 0..40 dB in 8 dB steps."""
    clamped = min(max(float(lna_db), 0.0), 40.0)
    return round(clamped / 8.0) * 8


def quantize_vga_gain(vga_db: float) -> int:
    """Clamp and quantize HackRF RX baseband VGA gain to 0..62 dB in 2 dB steps."""
    clamped = min(max(float(vga_db), 0.0), 62.0)
    return round(clamped / 2.0) * 2


def select_baseband_bw(sample_rate: float) -> int:
    """Select the appropriate MAX2837 baseband filter bandwidth for 20 MHz OFDM."""
    sr = round(sample_rate)
    best = HACKRF_BASEBAND_BWS_HZ[0]
    for bw in HACKRF_BASEBAND_BWS_HZ:
        if bw <= sr:
            best = bw
    return best


def compute_adc_clip_ratio_int8(raw_i8: np.ndarray) -> float:
    """Compute fraction of complex int8 samples with |I| >= 120 or |Q| >= 120."""
    n_vals = (raw_i8.size // 2) * 2
    if n_vals == 0:
        return 0.0
    pairs = raw_i8[:n_vals].reshape(-1, 2)
    # Cast to int16 before abs so -128 does not overflow int8
    i_abs = np.abs(pairs[:, 0].astype(np.int16))
    q_abs = np.abs(pairs[:, 1].astype(np.int16))
    clipped = (i_abs >= ADC_CLIP_THRESHOLD_INT8) | (q_abs >= ADC_CLIP_THRESHOLD_INT8)
    return float(np.mean(clipped))


def compute_adc_clip_ratio_cf32(iq: np.ndarray) -> float:
    """Compute fraction of normalized complex64 samples exceeding 120/128 full scale."""
    if iq.size == 0:
        return 0.0
    clipped = (np.abs(iq.real) >= ADC_CLIP_THRESHOLD_NORM) | (
        np.abs(iq.imag) >= ADC_CLIP_THRESHOLD_NORM
    )
    return float(np.mean(clipped))


@dataclass(slots=True)
class IQChunk:
    """Conditioned 20 MSPS RX I/Q block for a single channel dwell."""

    samples: np.ndarray  # complex64 normalized in [-1, 1], DC-offset removed
    timestamp: float  # host epoch seconds of first sample in chunk
    freq_hz: float  # tuned channel center frequency (Hz)
    channel: int  # Wi-Fi channel number
    sample_rate: float  # sample rate (Hz), typically 20e6
    adc_clip_ratio: float  # fraction of samples with |I|>=120 or |Q|>=120
    total_gain_db: float  # lna_db + vga_db (AMP locked to 0 dB)
    overrun_count: int  # cumulative USB / ring-buffer overrun count
    dwell_index: int = 0  # monotonically increments on every channel retune


class _DCBlocker:
    """Per-channel stateful DC-offset estimator & remover."""

    def __init__(self, alpha: float = 0.08) -> None:
        self.alpha = float(alpha)
        self._dc: complex | None = None

    def reset(self) -> None:
        self._dc = None

    def process(self, iq: np.ndarray) -> np.ndarray:
        if iq.size == 0:
            return iq
        block_mean = complex(np.mean(iq))
        if self._dc is None:
            self._dc = block_mean
        else:
            self._dc = (1.0 - self.alpha) * self._dc + self.alpha * block_mean
        return (iq - np.complex64(self._dc)).astype(np.complex64, copy=False)


# ---------------------------------------------------------------------------
# Backend 1: SoapySDR (driver=hackrf) / gnuradio.soapy
# ---------------------------------------------------------------------------


class _SoapyHackRFBackend:
    """Passive RX-only SoapySDR HackRF driver backend."""

    name = "soapy"

    def __init__(
        self,
        sample_rate: float,
        freq_hz: float,
        lna_db: int,
        vga_db: int,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.freq_hz = float(freq_hz)
        self.lna_db = quantize_lna_gain(lna_db)
        self.vga_db = quantize_vga_gain(vga_db)
        self.overruns = 0
        self._sdr: Any = None
        self._rx_stream: Any = None
        self._soapy_mod: Any = None
        self._use_cs8 = False

        try:
            import SoapySDR  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("SoapySDR Python bindings are not installed") from exc

        self._soapy_mod = SoapySDR
        # Verify hackrf factory exists before opening
        factories = (
            set(SoapySDR.listModules()) if hasattr(SoapySDR, "listModules") else set()
        )
        try:
            found = SoapySDR.Device.enumerate({"driver": "hackrf"})
        except (RuntimeError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"SoapySDR enumerate(driver=hackrf) failed: {exc}"
            ) from exc

        if not found:
            raise RuntimeError(
                f"SoapyHackRF support plugin not found (enumerated 0 hackrf devices; modules={len(factories)})"
            )

        sdr = SoapySDR.Device({"driver": "hackrf"})
        rx_dir = SoapySDR.SOAPY_SDR_RX
        if hasattr(sdr, "setGainMode"):
            sdr.setGainMode(rx_dir, 0, False)
        # Strictly passive RX: RF AMP forced to 0 dB, bias-tee disabled
        sdr.setGain(rx_dir, 0, "AMP", 0.0)
        if hasattr(sdr, "writeSetting"):
            with contextlib.suppress(RuntimeError, OSError, AttributeError):
                sdr.writeSetting("bias_tx", "false")
        sdr.setGain(rx_dir, 0, "LNA", float(self.lna_db))
        sdr.setGain(rx_dir, 0, "VGA", float(self.vga_db))
        sdr.setSampleRate(rx_dir, 0, self.sample_rate)
        sdr.setBandwidth(rx_dir, 0, float(select_baseband_bw(self.sample_rate)))
        sdr.setFrequency(rx_dir, 0, self.freq_hz)

        fmt = getattr(SoapySDR, "SOAPY_SDR_CS8", "CS8")
        try:
            self._rx_stream = sdr.setupStream(rx_dir, fmt, [0])
            self._use_cs8 = True
        except (RuntimeError, OSError, TypeError, ValueError):
            self._rx_stream = sdr.setupStream(rx_dir, SoapySDR.SOAPY_SDR_CF32, [0])
            self._use_cs8 = False

        sdr.activateStream(self._rx_stream)
        self._sdr = sdr

    def retune(self, freq_hz: float) -> None:
        self.freq_hz = float(freq_hz)
        if self._sdr is not None:
            rx_dir = self._soapy_mod.SOAPY_SDR_RX
            self._sdr.setGain(rx_dir, 0, "AMP", 0.0)
            self._sdr.setFrequency(rx_dir, 0, self.freq_hz)

    def read_samples(self, num_samples: int) -> tuple[np.ndarray, float]:
        """Return (complex64_samples, adc_clip_ratio)."""
        SoapySDR = self._soapy_mod
        if self._use_cs8:
            buf = np.empty(num_samples * 2, dtype=np.int8)
            sr = self._sdr.readStream(
                self._rx_stream, [buf], num_samples, timeoutUs=200_000
            )
            ret = int(sr.ret)
            if ret == getattr(SoapySDR, "SOAPY_SDR_OVERFLOW", -4):
                self.overruns += 1
                return np.empty(0, dtype=np.complex64), 0.0
            if ret <= 0:
                return np.empty(0, dtype=np.complex64), 0.0
            valid_i8 = buf[: ret * 2]
            clip = compute_adc_clip_ratio_int8(valid_i8)
            f32 = valid_i8.astype(np.float32) * (1.0 / 128.0)
            iq = (f32[0::2] + 1j * f32[1::2]).astype(np.complex64)
            return iq, clip

        buf_cf = np.empty(num_samples, dtype=np.complex64)
        sr = self._sdr.readStream(
            self._rx_stream, [buf_cf], num_samples, timeoutUs=200_000
        )
        ret = int(sr.ret)
        if ret == getattr(SoapySDR, "SOAPY_SDR_OVERFLOW", -4):
            self.overruns += 1
            return np.empty(0, dtype=np.complex64), 0.0
        if ret <= 0:
            return np.empty(0, dtype=np.complex64), 0.0
        iq = buf_cf[:ret].copy()
        return iq, compute_adc_clip_ratio_cf32(iq)

    def close(self) -> None:
        if self._sdr is not None:
            if self._rx_stream is not None:
                with contextlib.suppress(RuntimeError, OSError, AttributeError):
                    self._sdr.deactivateStream(self._rx_stream)
                    self._sdr.closeStream(self._rx_stream)
            self._rx_stream = None
            self._sdr = None


# ---------------------------------------------------------------------------
# Backend 2: ctypes libhackrf.so.0 direct USB RX streaming
# ---------------------------------------------------------------------------


class _LibHackRFStruct(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_void_p),
        ("buffer", ctypes.POINTER(ctypes.c_uint8)),
        ("buffer_length", ctypes.c_int),
        ("valid_length", ctypes.c_int),
        ("rx_ctx", ctypes.c_void_p),
        ("tx_ctx", ctypes.c_void_p),
    ]


_RX_CB_PROTO = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_LibHackRFStruct))


class _LibHackRFBackend:
    """Direct ctypes binding to libhackrf.so.0 in strict RX-only mode."""

    name = "libhackrf"

    def __init__(
        self,
        sample_rate: float,
        freq_hz: float,
        lna_db: int,
        vga_db: int,
        max_queue_blocks: int = 64,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.freq_hz = float(freq_hz)
        self.lna_db = quantize_lna_gain(lna_db)
        self.vga_db = quantize_vga_gain(vga_db)
        self.overruns = 0

        self._lib: Any = None
        self._dev = ctypes.c_void_p(None)
        self._closed = False
        self._tune_epoch = 0
        self._queue: collections.deque[tuple[int, bytes]] = collections.deque()
        self._max_queue_blocks = int(max_queue_blocks)
        self._cond = threading.Condition()
        self._leftover = b""

        lib = None
        last_dl_exc: OSError | None = None
        for soname in ("libhackrf.so.0", "/usr/local/lib/libhackrf.so.0"):
            try:
                lib = ctypes.CDLL(soname)
                break
            except OSError as exc:
                last_dl_exc = exc
        if lib is None:
            raise RuntimeError(
                f"Unable to load libhackrf.so.0: {last_dl_exc}"
            ) from last_dl_exc

        # Bind RX-only C API signatures
        lib.hackrf_init.argtypes = []
        lib.hackrf_init.restype = ctypes.c_int
        lib.hackrf_exit.argtypes = []
        lib.hackrf_exit.restype = ctypes.c_int
        lib.hackrf_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.hackrf_open.restype = ctypes.c_int
        lib.hackrf_close.argtypes = [ctypes.c_void_p]
        lib.hackrf_close.restype = ctypes.c_int
        lib.hackrf_set_sample_rate.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.hackrf_set_sample_rate.restype = ctypes.c_int
        lib.hackrf_set_baseband_filter_bandwidth.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        lib.hackrf_set_baseband_filter_bandwidth.restype = ctypes.c_int
        lib.hackrf_set_amp_enable.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
        lib.hackrf_set_amp_enable.restype = ctypes.c_int
        lib.hackrf_set_antenna_enable.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
        lib.hackrf_set_antenna_enable.restype = ctypes.c_int
        lib.hackrf_set_lna_gain.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.hackrf_set_lna_gain.restype = ctypes.c_int
        lib.hackrf_set_vga_gain.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.hackrf_set_vga_gain.restype = ctypes.c_int
        lib.hackrf_set_freq.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.hackrf_set_freq.restype = ctypes.c_int
        lib.hackrf_start_rx.argtypes = [ctypes.c_void_p, _RX_CB_PROTO, ctypes.c_void_p]
        lib.hackrf_start_rx.restype = ctypes.c_int
        lib.hackrf_stop_rx.argtypes = [ctypes.c_void_p]
        lib.hackrf_stop_rx.restype = ctypes.c_int

        rc = lib.hackrf_init()
        if rc != 0:
            raise RuntimeError(f"hackrf_init failed with code {rc}")

        dev = ctypes.c_void_p(None)
        rc = lib.hackrf_open(ctypes.byref(dev))
        if rc != 0 or not dev.value:
            lib.hackrf_exit()
            raise RuntimeError(
                f"hackrf_open failed with code {rc} (device busy or disconnected)"
            )

        self._lib = lib
        self._dev = dev

        try:
            # Force RF AMP OFF (0) and antenna bias-tee power OFF (0)
            self._check(
                lib.hackrf_set_amp_enable(dev, ctypes.c_uint8(0)), "set_amp_enable(0)"
            )
            self._check(
                lib.hackrf_set_antenna_enable(dev, ctypes.c_uint8(0)),
                "set_antenna_enable(0)",
            )
            self._check(
                lib.hackrf_set_sample_rate(dev, ctypes.c_double(self.sample_rate)),
                "set_sample_rate",
            )
            bw_hz = select_baseband_bw(self.sample_rate)
            self._check(
                lib.hackrf_set_baseband_filter_bandwidth(dev, ctypes.c_uint32(bw_hz)),
                "set_baseband_filter_bandwidth",
            )
            self._check(
                lib.hackrf_set_lna_gain(dev, ctypes.c_uint32(self.lna_db)),
                "set_lna_gain",
            )
            self._check(
                lib.hackrf_set_vga_gain(dev, ctypes.c_uint32(self.vga_db)),
                "set_vga_gain",
            )
            self._check(
                lib.hackrf_set_freq(dev, ctypes.c_uint64(round(self.freq_hz))),
                "set_freq",
            )

            self._cb_ref = _RX_CB_PROTO(self._on_rx_transfer)
            self._check(lib.hackrf_start_rx(dev, self._cb_ref, None), "start_rx")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(rc: int, op: str) -> None:
        if rc != 0:
            raise RuntimeError(f"libhackrf {op} failed with status {rc}")

    def _on_rx_transfer(self, transfer_ptr: Any) -> int:
        if self._closed:
            return -1
        tr = transfer_ptr.contents
        n_bytes = int(tr.valid_length)
        if n_bytes <= 0:
            return 0
        raw = ctypes.string_at(tr.buffer, n_bytes)
        epoch = self._tune_epoch
        with self._cond:
            if len(self._queue) >= self._max_queue_blocks:
                self._queue.popleft()
                self.overruns += 1
            self._queue.append((epoch, raw))
            self._cond.notify_all()
        return 0

    def retune(self, freq_hz: float) -> None:
        self.freq_hz = float(freq_hz)
        if self._dev.value and not self._closed:
            self._check(
                self._lib.hackrf_set_amp_enable(self._dev, ctypes.c_uint8(0)),
                "set_amp_enable(0)",
            )
            self._check(
                self._lib.hackrf_set_freq(
                    self._dev, ctypes.c_uint64(round(self.freq_hz))
                ),
                "set_freq",
            )
            with self._cond:
                self._tune_epoch += 1
                self._queue.clear()
                self._leftover = b""

    def read_samples(self, num_samples: int) -> tuple[np.ndarray, float]:
        needed_bytes = int(num_samples) * 2
        target_epoch = self._tune_epoch
        chunks: list[bytes] = []
        collected = 0
        if self._leftover:
            chunks.append(self._leftover)
            collected += len(self._leftover)
            self._leftover = b""

        deadline = time.monotonic() + max(0.35, (num_samples / self.sample_rate) * 4.0)
        while collected < needed_bytes and not self._closed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            with self._cond:
                while not self._queue and not self._closed:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        break
                    self._cond.wait(timeout=min(rem, 0.05))
                while self._queue and collected < needed_bytes:
                    ep, raw = self._queue.popleft()
                    if ep != target_epoch:
                        continue
                    chunks.append(raw)
                    collected += len(raw)

        if not chunks:
            return np.empty(0, dtype=np.complex64), 0.0

        blob = b"".join(chunks)
        if len(blob) > needed_bytes:
            self._leftover = blob[needed_bytes:]
            blob = blob[:needed_bytes]
        else:
            even_len = (len(blob) // 2) * 2
            self._leftover = blob[even_len:]
            blob = blob[:even_len]

        if not blob:
            return np.empty(0, dtype=np.complex64), 0.0

        raw_i8 = np.frombuffer(blob, dtype=np.int8)
        clip = compute_adc_clip_ratio_int8(raw_i8)
        f32 = raw_i8.astype(np.float32) * (1.0 / 128.0)
        iq = (f32[0::2] + 1j * f32[1::2]).astype(np.complex64)
        return iq, clip

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._cond:
            self._cond.notify_all()
        if self._lib is not None and self._dev.value:
            with contextlib.suppress(OSError, ctypes.ArgumentError, RuntimeError):
                self._lib.hackrf_stop_rx(self._dev)
                time.sleep(0.015)
            with contextlib.suppress(OSError, ctypes.ArgumentError, RuntimeError):
                self._lib.hackrf_close(self._dev)
            self._dev = ctypes.c_void_p(None)
        if self._lib is not None:
            with contextlib.suppress(OSError, ctypes.ArgumentError, RuntimeError):
                self._lib.hackrf_exit()
            self._lib = None


# ---------------------------------------------------------------------------
# Backend 3: gr-osmosdr (osmosdr.source("hackrf=0,bias=0"))
# ---------------------------------------------------------------------------


class _OsmoSDRBackend:
    """GNU Radio osmosdr HackRF RX-only backend."""

    name = "osmosdr"

    def __init__(
        self,
        sample_rate: float,
        freq_hz: float,
        lna_db: int,
        vga_db: int,
    ) -> None:
        try:
            import osmosdr  # type: ignore[import-not-found]
            from gnuradio import blocks, gr  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "gnuradio/osmosdr Python bindings are not installed"
            ) from exc

        self._gr = gr
        self._blocks = blocks
        self._osmosdr = osmosdr
        self.sample_rate = float(sample_rate)
        self.freq_hz = float(freq_hz)
        self.lna_db = quantize_lna_gain(lna_db)
        self.vga_db = quantize_vga_gain(vga_db)
        self.overruns = 0

        src = osmosdr.source(args="numchan=1 hackrf=0,bias=0")
        if src.get_num_channels() < 1:
            raise RuntimeError("osmosdr failed to open hackrf=0")
        src.set_sample_rate(self.sample_rate)
        src.set_bandwidth(float(select_baseband_bw(self.sample_rate)), 0)
        src.set_freq_corr(0, 0)
        src.set_dc_offset_mode(2, 0)
        src.set_iq_balance_mode(0, 0)
        src.set_gain_mode(False, 0)
        src.set_gain(0.0, "AMP", 0)
        src.set_if_gain(float(self.lna_db), 0)
        src.set_bb_gain(float(self.vga_db), 0)
        src.set_center_freq(self.freq_hz, 0)
        self._src = src

    def retune(self, freq_hz: float) -> None:
        self.freq_hz = float(freq_hz)
        self._src.set_gain(0.0, "AMP", 0)
        self._src.set_center_freq(self.freq_hz, 0)

    def read_samples(self, num_samples: int) -> tuple[np.ndarray, float]:
        tb = self._gr.top_block("hackrf_osmo_rx_burst")
        head = self._blocks.head(self._gr.sizeof_gr_complex, int(num_samples))
        sink = self._blocks.vector_sink_c(reserve_items=int(num_samples))
        tb.connect(self._src, head, sink)
        tb.run()
        tb.disconnect_all()
        iq = np.asarray(sink.data(), dtype=np.complex64)
        return iq, compute_adc_clip_ratio_cf32(iq)

    def close(self) -> None:
        self._src = None


# ---------------------------------------------------------------------------
# Backend 4: hackrf_transfer -r - CLI RX-only subprocess fallback
# ---------------------------------------------------------------------------


class _HackRFTransferBackend:
    """Subprocess fallback wrapping ``hackrf_transfer -r -`` in strict RX-only mode."""

    name = "hackrf_transfer"

    def __init__(
        self,
        sample_rate: float,
        freq_hz: float,
        lna_db: int,
        vga_db: int,
    ) -> None:
        bin_path = shutil.which("hackrf_transfer") or "/usr/local/bin/hackrf_transfer"
        if not Path(bin_path).is_file():
            raise RuntimeError("hackrf_transfer binary not found on PATH")
        self._bin_path = bin_path
        self.sample_rate = float(sample_rate)
        self.freq_hz = float(freq_hz)
        self.lna_db = quantize_lna_gain(lna_db)
        self.vga_db = quantize_vga_gain(vga_db)
        self.overruns = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._leftover = b""
        self._start_stream_and_probe()

    def _build_rx_cmd(self) -> list[str]:
        bw_hz = select_baseband_bw(self.sample_rate)
        # Strictly RX-only flags: -r - (stdout), -a 0 (RF AMP off), -p 0 (antenna power off)
        return [
            self._bin_path,
            "-r",
            "-",
            "-f",
            str(round(self.freq_hz)),
            "-s",
            str(round(self.sample_rate)),
            "-b",
            str(int(bw_hz)),
            "-a",
            "0",
            "-p",
            "0",
            "-l",
            str(int(self.lna_db)),
            "-g",
            str(int(self.vga_db)),
        ]

    def _collect_proc_error(
        self, proc: subprocess.Popen[bytes], stdout_prefix: bytes = b""
    ) -> str:
        stderr_bytes = b""
        if proc.stderr is not None:
            with contextlib.suppress(OSError, ValueError):
                rlist, _, _ = select.select([proc.stderr.fileno()], [], [], 0.05)
                if rlist:
                    stderr_bytes = os.read(proc.stderr.fileno(), 4096)
        msg = stderr_bytes.decode("utf-8", errors="replace").strip()
        if not msg and stdout_prefix.startswith(b"Usage:"):
            msg = "hackrf_transfer rejected arguments or failed to open device"
        return msg or "device busy or disconnected"

    def _start_stream_and_probe(self) -> None:
        self._stop_proc()
        self._leftover = b""
        proc = subprocess.Popen(
            self._build_rx_cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._proc = proc
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        probed = bytearray()
        deadline = time.monotonic() + 0.90
        while len(probed) < 4096:
            rc = proc.poll()
            rem = deadline - time.monotonic()
            if rem <= 0:
                break
            rlist, _, _ = select.select([fd], [], [], min(rem, 0.05))
            if rlist:
                chunk = os.read(fd, 16384)
                if not chunk:
                    break
                probed.extend(chunk)
                if probed.startswith(b"Usage:"):
                    break
            elif rc is not None:
                break

        rc = proc.poll()
        raw_probe = bytes(probed)
        if rc is not None or raw_probe.startswith(b"Usage:") or len(raw_probe) < 2:
            err_detail = self._collect_proc_error(proc, raw_probe)
            self._stop_proc()
            raise RuntimeError(
                f"hackrf_transfer failed ({rc if rc is not None else 'timeout'}): {err_detail}"
            )

        self._leftover = raw_probe

    def retune(self, freq_hz: float) -> None:
        self.freq_hz = float(freq_hz)
        self._start_stream_and_probe()

    def _stop_proc(self) -> None:
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=0.35)
            except (OSError, subprocess.SubprocessError):
                with contextlib.suppress(OSError, subprocess.SubprocessError):
                    proc.kill()
                    proc.wait(timeout=0.2)
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    with contextlib.suppress(OSError, ValueError):
                        pipe.close()

    def read_samples(self, num_samples: int) -> tuple[np.ndarray, float]:
        if self._proc is None:
            self._start_stream_and_probe()
        assert self._proc is not None and self._proc.stdout is not None

        needed_bytes = int(num_samples) * 2
        chunks: list[bytes] = []
        collected = 0
        if self._leftover:
            chunks.append(self._leftover)
            collected += len(self._leftover)
            self._leftover = b""

        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + max(0.45, (num_samples / self.sample_rate) * 4.0)
        while collected < needed_bytes:
            rem = deadline - time.monotonic()
            if rem <= 0:
                break
            rlist, _, _ = select.select([fd], [], [], min(rem, 0.05))
            if rlist:
                part = os.read(fd, min(65536, needed_bytes - collected + 4096))
                if not part:
                    break
                chunks.append(part)
                collected += len(part)
            elif self._proc.poll() is not None:
                break

        blob = b"".join(chunks)
        rc = self._proc.poll()
        if blob.startswith(b"Usage:") or (len(blob) < 2 and rc is not None and rc != 0):
            err_detail = self._collect_proc_error(self._proc, blob)
            self._stop_proc()
            raise RuntimeError(f"hackrf_transfer failed ({rc}): {err_detail}")

        if len(blob) > needed_bytes:
            self._leftover = blob[needed_bytes:]
            blob = blob[:needed_bytes]
        else:
            even_len = (len(blob) // 2) * 2
            self._leftover = blob[even_len:]
            blob = blob[:even_len]

        if not blob:
            return np.empty(0, dtype=np.complex64), 0.0
        if len(blob) < needed_bytes:
            self.overruns += 1
        raw_i8 = np.frombuffer(blob, dtype=np.int8)
        clip = compute_adc_clip_ratio_int8(raw_i8)
        f32 = raw_i8.astype(np.float32) * (1.0 / 128.0)
        iq = (f32[0::2] + 1j * f32[1::2]).astype(np.complex64)
        return iq, clip

    def close(self) -> None:
        self._stop_proc()


# ---------------------------------------------------------------------------
# Backend 5: Recorded I/Q file replay (.cs8 / .cf32 / .npy)
# ---------------------------------------------------------------------------


class _IQFileBackend:
    """Deterministic replay backend for .cs8 (int8), .cf32 (complex64), and .npy I/Q files."""

    name = "iq_file"

    def __init__(
        self,
        iq_path: str | Path,
        sample_rate: float,
        freq_hz: float,
        loop: bool = False,
    ) -> None:
        self.path = Path(iq_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"I/Q replay file not found: {self.path}")
        self.sample_rate = float(sample_rate)
        self.freq_hz = float(freq_hz)
        self.loop = bool(loop)
        self.overruns = 0
        self._cursor = 0

        suffix = self.path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(self.path)
            if arr.dtype == np.int8:
                self._raw_i8: np.ndarray | None = arr.reshape(-1)
                self._cf32: np.ndarray | None = None
            else:
                self._raw_i8 = None
                self._cf32 = np.asarray(arr, dtype=np.complex64).reshape(-1)
        elif suffix in {".cf32", ".fc32", ".cfile"}:
            self._raw_i8 = None
            self._cf32 = np.fromfile(self.path, dtype=np.complex64)
        else:
            # Default HackRF .cs8 / .sc8 / .s8 / .iq interleaved signed int8
            raw = np.fromfile(self.path, dtype=np.int8)
            even = (raw.size // 2) * 2
            self._raw_i8 = raw[:even]
            self._cf32 = None

    @property
    def total_samples(self) -> int:
        if self._raw_i8 is not None:
            return int(self._raw_i8.size // 2)
        assert self._cf32 is not None
        return int(self._cf32.size)

    def retune(self, freq_hz: float) -> None:
        self.freq_hz = float(freq_hz)

    def read_samples(self, num_samples: int) -> tuple[np.ndarray, float]:
        total = self.total_samples
        if total == 0 or (self._cursor >= total and not self.loop):
            return np.empty(0, dtype=np.complex64), 0.0

        if self._cursor >= total and self.loop:
            self._cursor = 0

        end = min(self._cursor + int(num_samples), total)
        if self._raw_i8 is not None:
            seg = self._raw_i8[self._cursor * 2 : end * 2]
            self._cursor = end
            clip = compute_adc_clip_ratio_int8(seg)
            f32 = seg.astype(np.float32) * (1.0 / 128.0)
            iq = (f32[0::2] + 1j * f32[1::2]).astype(np.complex64)
            return iq, clip

        assert self._cf32 is not None
        iq = self._cf32[self._cursor : end].copy()
        self._cursor = end
        return iq, compute_adc_clip_ratio_cf32(iq)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# High-level Passive RX Channel-Hopping I/Q Source
# ---------------------------------------------------------------------------


class HackRFIQSource:
    """Strictly passive (0 dBm RX-only) 20 MSPS HackRF I/Q source with channel hopping.

    Enforces:
      - Zero transmit capability (RX-only drivers, RF AMP locked to 0 dB, bias-tee off)
      - Manual LNA (0..40 dB, 8 dB steps) and VGA (0..62 dB, 2 dB steps) gain staging
      - 2.4 GHz (Ch 1/6/11) and 5 GHz UNII (Ch 36..165) channel hopping
      - Post-retune PLL settling sample flush (2–5 ms) & strict per-channel DC isolation
      - Real-time int8 ADC clipping (|I|>=120 or |Q|>=120) and USB overrun telemetry
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
        iq_path: str | Path | None = None,
        loop_file: bool = False,
    ) -> None:
        if amp_enable:
            raise ValueError(
                "RF AMP must remain False (0 dB) in strictly passive 0 dBm RX-only mode"
            )
        if bias_tee_enable:
            raise ValueError(
                "Antenna bias-tee power is forbidden in passive RX-only mode"
            )
        if not channels:
            raise ValueError("At least one Wi-Fi channel must be specified")

        self.channels: tuple[int, ...] = tuple(int(c) for c in channels)
        for ch in self.channels:
            channel_to_freq_hz(ch)  # validate channel map upfront

        self.sample_rate = float(sample_rate)
        self.dwell_ms = float(dwell_ms)
        # Enforce 2-5 ms PLL settling flush on live hardware (0 ms allowed for iq_file if requested)
        self.settle_ms = float(settle_ms)
        self.lna_db = quantize_lna_gain(lna_db)
        self.vga_db = quantize_vga_gain(vga_db)
        self.amp_enable: bool = False
        self.bias_tee_enable: bool = False
        self.dc_offset_removal: bool = bool(dc_offset_removal)

        self._dc_blocker = _DCBlocker(alpha=0.08)
        self._channel_idx = 0
        self._dwell_counter = 0
        self._samples_in_dwell = 0
        self._sim_clock = 0.0
        self.total_samples_read = 0
        self.max_adc_clip_ratio = 0.0
        self.backend_errors: dict[str, str] = {}

        first_freq = channel_to_freq_hz(self.channels[0])
        self._driver = self._open_backend(
            backend=backend,
            iq_path=iq_path,
            first_freq=first_freq,
            loop_file=loop_file,
        )
        self.active_backend: str = self._driver.name

        # Flush initial PLL settling samples on live hardware
        if self.active_backend != "iq_file":
            effective_settle_ms = min(max(self.settle_ms, 2.0), 5.0)
            settle_samples = round(self.sample_rate * (effective_settle_ms * 1e-3))
            if settle_samples > 0:
                self._driver.read_samples(settle_samples)

    def _open_backend(
        self,
        backend: str,
        iq_path: str | Path | None,
        first_freq: float,
        loop_file: bool,
    ) -> Any:
        key = backend.strip().lower()
        if key == "iq_file" or (key == "auto" and iq_path is not None):
            if iq_path is None:
                raise ValueError("iq_path is required when backend='iq_file'")
            return _IQFileBackend(
                iq_path=iq_path,
                sample_rate=self.sample_rate,
                freq_hz=first_freq,
                loop=loop_file,
            )

        builders = {
            "soapy": lambda: _SoapyHackRFBackend(
                self.sample_rate, first_freq, self.lna_db, self.vga_db
            ),
            "soapy_hackrf": lambda: _SoapyHackRFBackend(
                self.sample_rate, first_freq, self.lna_db, self.vga_db
            ),
            "libhackrf": lambda: _LibHackRFBackend(
                self.sample_rate, first_freq, self.lna_db, self.vga_db
            ),
            "osmosdr": lambda: _OsmoSDRBackend(
                self.sample_rate, first_freq, self.lna_db, self.vga_db
            ),
            "hackrf_transfer": lambda: _HackRFTransferBackend(
                self.sample_rate, first_freq, self.lna_db, self.vga_db
            ),
        }

        if key == "auto":
            order = ("soapy", "libhackrf", "osmosdr", "hackrf_transfer")
            for cand in order:
                try:
                    return builders[cand]()
                except (
                    RuntimeError,
                    OSError,
                    ValueError,
                    TypeError,
                    ctypes.ArgumentError,
                ) as exc:
                    self.backend_errors[cand] = str(exc)
            details = "; ".join(f"{k}: {v}" for k, v in self.backend_errors.items())
            raise RuntimeError(f"No live HackRF RX backend succeeded ({details})")

        if key not in builders:
            raise ValueError(f"Unsupported SDR backend: {backend!r}")
        return builders[key]()

    @property
    def current_channel(self) -> int:
        return self.channels[self._channel_idx]

    @property
    def current_freq_hz(self) -> float:
        return WIFI_CHANNELS_HZ.get(
            self.current_channel, channel_to_freq_hz(self.current_channel)
        )

    @property
    def total_gain_db(self) -> float:
        return float(self.lna_db + self.vga_db)

    @property
    def overrun_count(self) -> int:
        return int(getattr(self._driver, "overruns", 0))

    def hop_next_channel(self) -> int:
        """Advance to the next channel, reset per-channel state, and flush PLL settle samples."""
        if len(self.channels) <= 1:
            self._samples_in_dwell = 0
            return self.current_channel

        self._channel_idx = (self._channel_idx + 1) % len(self.channels)
        self._dwell_counter += 1
        self._samples_in_dwell = 0
        self._dc_blocker.reset()

        new_freq = self.current_freq_hz
        self._driver.retune(new_freq)

        if self.active_backend != "iq_file":
            effective_settle_ms = min(max(self.settle_ms, 2.0), 5.0)
            settle_samples = round(self.sample_rate * (effective_settle_ms * 1e-3))
            if settle_samples > 0:
                self._driver.read_samples(settle_samples)

        return self.current_channel

    def iter_chunks(
        self,
        duration_s: float | None = None,
        chunk_ms: float = 20.0,
        stop_check: Callable[[], bool] | None = None,
    ) -> Iterator[IQChunk]:
        """Yield DC-removed 20 MSPS IQChunks while hopping across configured channels."""
        chunk_samples = max(4096, round(self.sample_rate * (chunk_ms * 1e-3)))
        dwell_samples = max(
            chunk_samples, round(self.sample_rate * (self.dwell_ms * 1e-3))
        )
        start_wall = time.time()
        start_mono = time.monotonic()

        while True:
            if stop_check is not None and stop_check():
                break
            if duration_s is not None:
                elapsed = (
                    self._sim_clock
                    if self.active_backend == "iq_file"
                    else (time.monotonic() - start_mono)
                )
                if elapsed >= duration_s:
                    break

            if len(self.channels) > 1 and self._samples_in_dwell >= dwell_samples:
                self.hop_next_channel()

            to_read = min(chunk_samples, dwell_samples - self._samples_in_dwell)
            if to_read <= 0:
                to_read = chunk_samples

            ts = (
                start_wall + self._sim_clock
                if self.active_backend == "iq_file"
                else time.time()
            )
            raw_iq, clip_ratio = self._driver.read_samples(to_read)
            if raw_iq.size == 0:
                if self.active_backend == "iq_file":
                    break
                continue

            clean_iq = (
                self._dc_blocker.process(raw_iq) if self.dc_offset_removal else raw_iq
            )
            n_got = int(clean_iq.size)
            self._samples_in_dwell += n_got
            self.total_samples_read += n_got
            self._sim_clock += n_got / self.sample_rate
            self.max_adc_clip_ratio = max(self.max_adc_clip_ratio, clip_ratio)

            yield IQChunk(
                samples=clean_iq,
                timestamp=ts,
                freq_hz=self.current_freq_hz,
                channel=self.current_channel,
                sample_rate=self.sample_rate,
                adc_clip_ratio=clip_ratio,
                total_gain_db=self.total_gain_db,
                overrun_count=self.overrun_count,
                dwell_index=self._dwell_counter,
            )

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def __getattr__(name: str) -> Any:
    if name == "HackRFPacketCapture":
        from .gr_wifi_bridge import HackRFPacketCapture

        return HackRFPacketCapture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
