"""Tests for 52-subcarrier CSI sanitization, native 20 MSPS 802.11 OFDM decode, SIFS, & PCAP/IQ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hackrf_radar.capture import (
    SUBCARRIER_INDICES,
    HackRFPacketCapture,
    PacketEvent,
    compute_adc_clip_ratio_int8,
    sanitize_csi,
)
from hackrf_radar.capture.gr_wifi_bridge import (
    NativeOFDMBurstDetector,
    build_80211_mac_frame,
    synthesize_80211_ofdm_burst,
    write_radiotap_pcap,
)


def test_csi_sanitization_removes_sto_slope_and_random_offset() -> None:
    k = np.asarray(SUBCARRIER_INDICES, dtype=np.float64)
    # Underlying physical multipath response (amplitude ripple + nonlinear phase)
    amp = 0.85 + 0.22 * np.cos(0.19 * k)
    true_nl_phase = 0.28 * np.sin(0.14 * k)
    h_phys = amp * np.exp(1j * true_nl_phase)

    # Inject large STO phase slope across DC gap + random LO phase offset
    h_corrupted_1 = h_phys * np.exp(1j * (1.73 - 0.41 * k))
    h_corrupted_2 = h_phys * np.exp(1j * (-2.45 + 0.37 * k))

    s1 = sanitize_csi(h_corrupted_1)
    s2 = sanitize_csi(h_corrupted_2)

    assert s1.shape == (52,)
    assert s1.dtype == np.complex64
    np.testing.assert_allclose(np.abs(s1), amp, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(s1, s2, rtol=1e-3, atol=1e-3)

    # Edge cases: all-zero CSI & invalid subcarrier count
    zeros = sanitize_csi(np.zeros(52, dtype=np.complex64))
    assert np.all(zeros == 0.0)
    with pytest.raises(ValueError):
        sanitize_csi(np.ones(48, dtype=np.complex64))


def test_packet_event_serialization_and_int8_clip_overflow() -> None:
    # Verify -128 in signed int8 does not overflow abs()
    raw_i8 = np.array([-128, 0, 127, 10, 50, -50, 120, 0], dtype=np.int8)
    clip = compute_adc_clip_ratio_int8(raw_i8)
    assert clip == pytest.approx(0.75)

    csi = np.ones(52, dtype=np.complex64) * (0.6 + 0.2j)
    ev = PacketEvent(
        timestamp=123.456,
        freq_hz=2437e6,
        channel=6,
        bssid="AA:BB:CC:00:00:02",
        receiver_addr="AA:BB:CC:00:00:04",
        frame_type="ACK",
        rssi_dbm=-51.5,
        noise_floor_dbm=-92.0,
        snr_db=40.5,
        csi=csi,
        csi_sanitized=sanitize_csi(csi),
        cfo_hz=-3400.0,
        tsf_us=None,
        tsf_drift_ppm=None,
        seq_num=None,
        seq_gap=0,
        fcs_ok=True,
        adc_clip_ratio=0.002,
        attributed_via_sifs=True,
    )
    rebuilt = PacketEvent.from_dict(ev.to_dict())
    assert rebuilt.bssid == ev.bssid
    assert rebuilt.attributed_via_sifs is True
    np.testing.assert_allclose(rebuilt.csi, ev.csi, atol=1e-6)


def test_native_ofdm_detector_sifs_tsf_seqgap_and_fcs() -> None:
    det = NativeOFDMBurstDetector(sample_rate=20e6)
    k = np.asarray(SUBCARRIER_INDICES, dtype=np.float64)
    chan = (0.92 + 0.16 * np.cos(0.17 * k)) * np.exp(-1j * 0.06 * k)

    # Beacon #1 at t=0.000s (seq=4095, TSF=1_000_000) and Beacon #2 at t=0.100s (+10 ppm drift, seq=2 -> gap=2)
    bcn1 = build_80211_mac_frame(
        "BEACON", addr2="AA:BB:CC:00:00:01", seq_num=4095, tsf_us=1_000_000
    )
    bcn2 = build_80211_mac_frame(
        "BEACON", addr2="AA:BB:CC:00:00:01", seq_num=2, tsf_us=1_100_001
    )
    # Data (12 Mbps QPSK) -> SIFS ACK (24 Mbps 16-QAM) + corrupted FCS frame
    data_f = build_80211_mac_frame(
        "DATA_PREAMBLE", addr1="AA:BB:CC:00:00:05", addr2="AA:BB:CC:00:00:01", seq_num=3
    )
    ack_f = build_80211_mac_frame("ACK", addr1="AA:BB:CC:00:00:01")
    bad_f = build_80211_mac_frame(
        "QOS_NULL",
        addr1="AA:BB:CC:00:00:01",
        addr2="AA:BB:CC:00:00:05",
        corrupt_fcs=True,
    )

    w_b1 = synthesize_80211_ofdm_burst(
        bcn1,
        rate_mbps=6,
        channel_csi=chan,
        cfo_hz=11200.0,
        snr_db=30.0,
        pad_before=500,
        pad_after=500,
    )
    ev1 = det.detect_in_samples(w_b1, timestamp=10.0, freq_hz=2412e6, channel=1)
    assert len(ev1) == 1
    assert ev1[0].frame_type == "BEACON"
    assert ev1[0].bssid == "AA:BB:CC:00:00:01"
    assert ev1[0].seq_num == 4095
    assert ev1[0].fcs_ok is True
    assert abs(ev1[0].cfo_hz - 11200.0) < 1500.0

    # Second burst train 100ms later: Beacon #2 + Data -> SIFS ACK + FCS error
    w_b2 = synthesize_80211_ofdm_burst(
        bcn2,
        rate_mbps=6,
        channel_csi=chan,
        cfo_hz=11200.0,
        snr_db=30.0,
        pad_before=500,
        pad_after=300,
    )
    w_dat = synthesize_80211_ofdm_burst(
        data_f,
        rate_mbps=12,
        channel_csi=chan,
        cfo_hz=11200.0,
        snr_db=30.0,
        pad_before=300,
        pad_after=140,
    )
    w_ack = synthesize_80211_ofdm_burst(
        ack_f,
        rate_mbps=24,
        channel_csi=chan,
        cfo_hz=-5400.0,
        snr_db=30.0,
        pad_before=60,
        pad_after=400,
    )
    w_bad = synthesize_80211_ofdm_burst(
        bad_f,
        rate_mbps=54,
        channel_csi=chan,
        cfo_hz=0.0,
        snr_db=32.0,
        pad_before=400,
        pad_after=500,
    )

    train = np.concatenate([w_b2, w_dat, w_ack, w_bad])
    evs = det.detect_in_samples(train, timestamp=10.100, freq_hz=2412e6, channel=1)
    assert len(evs) == 4

    assert evs[0].frame_type == "BEACON"
    assert evs[0].seq_gap == 2  # 4095 -> 2 missed seq 0 and 1 across modulo-4096 wrap!
    assert evs[0].tsf_drift_ppm is not None
    assert abs(evs[0].tsf_drift_ppm - 10.0) < 12.0

    assert evs[1].frame_type == "DATA_PREAMBLE"
    assert evs[1].receiver_addr == "AA:BB:CC:00:00:05"

    assert evs[2].frame_type == "ACK"
    assert evs[2].bssid == "AA:BB:CC:00:00:05"
    assert evs[2].receiver_addr == "AA:BB:CC:00:00:01"
    assert evs[2].attributed_via_sifs is True

    assert evs[3].frame_type == "FCS_ERROR"
    assert evs[3].fcs_ok is False


def test_cs8_iq_file_and_radiotap_pcap_replay(tmp_path: Path) -> None:
    bcn = build_80211_mac_frame(
        "BEACON", addr2="AA:BB:CC:00:00:02", seq_num=7, tsf_us=2_000_000
    )
    rts = build_80211_mac_frame(
        "RTS", addr1="AA:BB:CC:00:00:04", addr2="AA:BB:CC:00:00:02"
    )
    cts = build_80211_mac_frame("CTS", addr1="AA:BB:CC:00:00:02")

    # 1. Test .cs8 interleaved int8 I/Q file replay via HackRFPacketCapture(backend="iq_file")
    wf = np.concatenate(
        [
            synthesize_80211_ofdm_burst(
                bcn,
                rate_mbps=6,
                snr_db=28.0,
                amplitude=0.35,
                pad_before=2000,
                pad_after=2000,
            ),
            np.zeros(40_000, dtype=np.complex64),
        ]
    )
    cs8_interleaved = np.empty(wf.size * 2, dtype=np.int8)
    cs8_interleaved[0::2] = np.clip(np.round(wf.real * 127.0), -127, 127).astype(
        np.int8
    )
    cs8_interleaved[1::2] = np.clip(np.round(wf.imag * 127.0), -127, 127).astype(
        np.int8
    )
    cs8_path = tmp_path / "sample.cs8"
    cs8_interleaved.tofile(cs8_path)

    with HackRFPacketCapture(
        channels=[6], backend="iq_file", iq_path=cs8_path
    ) as cap_iq:
        iq_events = list(cap_iq.iter_events())
    assert len(iq_events) == 1
    assert iq_events[0].bssid == "AA:BB:CC:00:00:02"
    assert iq_events[0].frame_type == "BEACON"

    # 2. Test Radiotap PCAP replay via HackRFPacketCapture(backend="pcap")
    pcap_path = tmp_path / "sample.pcap"
    write_radiotap_pcap(
        pcap_path,
        [
            (100.000000, bcn, 2437e6, -48.0, False),
            (100.010000, rts, 2437e6, -49.0, False),
            (100.010180, cts, 2437e6, -54.0, False),
        ],
    )
    with HackRFPacketCapture(
        channels=[6], backend="pcap", pcap_path=pcap_path
    ) as cap_pcap:
        pcap_events = list(cap_pcap.iter_events())
    assert len(pcap_events) == 3
    assert pcap_events[2].frame_type == "CTS"
    assert pcap_events[2].bssid == "AA:BB:CC:00:00:04"
    assert pcap_events[2].attributed_via_sifs is True

    # 3. 2.4 GHz ToDS Data frame learns AP BSSID for subsequent standalone CTS-to-self,
    #    and microsecond rounding carry (100.9999998 -> 101.0) preserves monotonic timestamps.
    qos_tods = build_80211_mac_frame(
        "QOS_NULL",
        addr1="00:00:5E:00:53:10",
        addr2="00:00:5E:00:53:20",
        addr3="00:00:5E:00:53:10",
        seq_num=12,
    )
    cts_self = build_80211_mac_frame("CTS", addr1="00:00:5E:00:53:10")
    pcap_cts_path = tmp_path / "cts_self.pcap"
    write_radiotap_pcap(
        pcap_cts_path,
        [
            (100.9999998, qos_tods, 2437e6, -50.0, False),
            (101.0500000, cts_self, 2437e6, -46.0, False),
        ],
    )
    with HackRFPacketCapture(
        channels=[6], backend="pcap", pcap_path=pcap_cts_path
    ) as cap_cts:
        cts_events = list(cap_cts.iter_events())
    assert len(cts_events) == 2
    assert cts_events[0].timestamp == pytest.approx(101.0, abs=1e-6)
    assert cts_events[1].frame_type == "CTS"
    assert cts_events[1].bssid == "00:00:5E:00:53:10"
    assert cts_events[1].attributed_via_sifs is False


def test_chunk_boundary_no_duplicate_and_straddling_beacon_preserved() -> None:
    from hackrf_radar.capture.gr_wifi_bridge import append_fcs
    from hackrf_radar.capture.soapy_source import IQChunk

    # 1. Short frame inside tail of chunk 0 must NOT be emitted again in chunk 1
    det = NativeOFDMBurstDetector(sample_rate=20e6)
    qos = build_80211_mac_frame(
        "QOS_NULL", addr1="AA:BB:CC:00:00:01", addr2="AA:BB:CC:00:00:05", seq_num=10
    )
    burst = synthesize_80211_ofdm_burst(
        qos, rate_mbps=24, snr_db=30.0, pad_before=100, pad_after=500
    )
    c0 = np.concatenate(
        [np.random.default_rng(1).normal(0, 1e-4, 10000).astype(np.complex64), burst]
    )
    c1 = np.random.default_rng(2).normal(0, 1e-4, 10000).astype(np.complex64)
    ev0 = det.process_chunk(IQChunk(c0, 0.0, 2412e6, 1, 20e6, 0.0, 44.0, 0, 0))
    ev1 = det.process_chunk(
        IQChunk(c1, c0.size / 20e6, 2412e6, 1, 20e6, 0.0, 44.0, 0, 0)
    )
    assert len(ev0) == 1
    assert len(ev1) == 0

    # 2. Multi-kilobyte sample Beacon straddling chunk 0 -> chunk 1 must be preserved & decoded once
    det2 = NativeOFDMBurstDetector(sample_rate=20e6)
    bcn_raw = build_80211_mac_frame(
        "BEACON", addr2="AA:BB:CC:00:00:01", seq_num=7, tsf_us=500_000
    )
    bcn_long = append_fcs(bcn_raw[:-4] + b"\x00" * 150)
    bcn_wf = synthesize_80211_ofdm_burst(
        bcn_long, rate_mbps=6, snr_db=30.0, pad_before=200, pad_after=200
    )
    stream = np.concatenate(
        [
            np.random.default_rng(3).normal(0, 1e-4, 10000).astype(np.complex64),
            bcn_wf,
            np.random.default_rng(4).normal(0, 1e-4, 10000).astype(np.complex64),
        ]
    )
    b0 = det2.process_chunk(
        IQChunk(stream[:12500], 0.0, 2412e6, 1, 20e6, 0.0, 44.0, 0, 0)
    )
    b1 = det2.process_chunk(
        IQChunk(stream[12500:], 12500 / 20e6, 2412e6, 1, 20e6, 0.0, 44.0, 0, 0)
    )
    assert len(b0) == 0
    assert len(b1) == 1
    assert b1[0].fcs_ok is True
    assert b1[0].seq_num == 7


def test_1500_byte_high_duty_cycle_and_fcs_verification() -> None:
    from hackrf_radar.capture.gr_wifi_bridge import append_fcs

    base = build_80211_mac_frame(
        "DATA_PREAMBLE",
        addr1="AA:BB:CC:00:00:05",
        addr2="AA:BB:CC:00:00:01",
        seq_num=99,
    )
    big_good = append_fcs(base[:-4] + b"\x55" * 1460)
    big_bad = big_good[:-1] + bytes([big_good[-1] ^ 0xFF])

    wf_good = synthesize_80211_ofdm_burst(
        big_good, rate_mbps=24, snr_db=25.0, pad_before=300, pad_after=300
    )
    wf_bad = synthesize_80211_ofdm_burst(
        big_bad, rate_mbps=24, snr_db=25.0, pad_before=300, pad_after=300
    )

    det = NativeOFDMBurstDetector()
    ev_good = det.detect_in_samples(wf_good, freq_hz=2412e6, channel=1)
    ev_bad = det.detect_in_samples(wf_bad, freq_hz=2412e6, channel=1)
    assert len(ev_good) == 1 and ev_good[0].fcs_ok is True
    assert (
        len(ev_bad) == 1
        and ev_bad[0].fcs_ok is False
        and ev_bad[0].frame_type == "FCS_ERROR"
    )


def test_pcap_big_endian_nanosecond_and_pcapng_tsresol(tmp_path: Path) -> None:
    import struct

    from hackrf_radar.capture.gr_wifi_bridge import (
        DLT_IEEE802_11_RADIO,
        iter_pcap_packet_events,
    )

    bcn = build_80211_mac_frame(
        "BEACON", addr2="AA:BB:CC:00:00:03", seq_num=11, tsf_us=900_000
    )
    rtap = struct.pack("<BBHI", 0, 0, 8, 0) + bcn

    # 1. Big-endian nanosecond PCAP (0x4D3CB2A1 -> unpacked <I is 0x4D3CB2A1 when written >I 0xA1B23C4D)
    be_pcap = tmp_path / "be_ns.pcap"
    hdr = struct.pack(">IHHiIII", 0xA1B23C4D, 2, 4, 0, 0, 65535, DLT_IEEE802_11_RADIO)
    p1 = struct.pack(">IIII", 200, 100_000_000, len(rtap), len(rtap)) + rtap
    p2 = struct.pack(">IIII", 200, 350_000_000, len(rtap), len(rtap)) + rtap
    be_pcap.write_bytes(hdr + p1 + p2)
    evs_be = list(iter_pcap_packet_events(be_pcap, duration_s=1.0))
    assert len(evs_be) == 2
    assert evs_be[1].timestamp - evs_be[0].timestamp == pytest.approx(0.25, abs=1e-6)

    # 2. Big-endian PCAPNG with IDB nanosecond if_tsresol=9
    pcapng = tmp_path / "be_ns.pcapng"
    shb = struct.pack(">IIIHHqI", 0x0A0D0D0A, 28, 0x1A2B3C4D, 1, 0, -1, 28)
    opt_ts = struct.pack(">HHB3xHH", 9, 1, 9, 0, 0)
    idb_len = 20 + len(opt_ts)
    idb = (
        struct.pack(">IIHHI", 1, idb_len, DLT_IEEE802_11_RADIO, 0, 65535)
        + opt_ts
        + struct.pack(">I", idb_len)
    )
    blocks = [shb, idb]
    for i in range(4):
        ts_ns = int((50.0 + 0.1 * i) * 1e9)
        pad_pkt = rtap + b"\x00" * ((4 - len(rtap) % 4) % 4)
        el = 32 + len(pad_pkt)
        blocks.append(
            struct.pack(
                ">IIIIIII",
                6,
                el,
                0,
                (ts_ns >> 32) & 0xFFFFFFFF,
                ts_ns & 0xFFFFFFFF,
                len(rtap),
                len(rtap),
            )
            + pad_pkt
            + struct.pack(">I", el)
        )
    pcapng.write_bytes(b"".join(blocks))
    evs_ng = list(iter_pcap_packet_events(pcapng, duration_s=1.0))
    assert len(evs_ng) == 4
    assert [round(e.timestamp, 1) for e in evs_ng] == [50.0, 50.1, 50.2, 50.3]


def test_gnuradio_pmt_deserializer_and_gr_zmq_streaming() -> None:
    import struct
    import time

    from hackrf_radar.capture.gr_wifi_bridge import decode_gnuradio_pmt_pdu

    # Build authentic GNU Radio PMT binary PDU: cons(dict(rssi=-47.5, csi=c32vector(52)), u8vector(bcn_no_fcs))
    bcn_no_fcs = build_80211_mac_frame(
        "BEACON", addr2="AA:BB:CC:00:00:01", seq_num=33, tsf_us=777_000
    )[:-4]
    sym_rssi = b"\x02" + struct.pack(">H", 4) + b"rssi"
    val_rssi = b"\x04" + struct.pack(">d", -47.5)
    sym_csi = b"\x02" + struct.pack(">H", 3) + b"csi"
    csi_be = b"".join(struct.pack(">ff", 0.75, -0.25) for _ in range(52))
    val_csi = b"\x0a\x0a" + struct.pack(">I", 52) + b"\x01\x00" + csi_be
    pmt_dict = (
        b"\x09\x07" + sym_rssi + val_rssi + b"\x09\x07" + sym_csi + val_csi + b"\x06"
    )
    pmt_u8 = b"\x0a\x00" + struct.pack(">I", len(bcn_no_fcs)) + b"\x01\x00" + bcn_no_fcs
    raw_pdu = b"\x07" + pmt_dict + pmt_u8

    decoded = decode_gnuradio_pmt_pdu(raw_pdu)
    assert decoded is not None
    meta, mpdu = decoded
    assert meta["rssi"] == pytest.approx(-47.5)
    assert len(meta["csi"]) == 52
    assert mpdu == bcn_no_fcs

    try:
        import zmq  # type: ignore[import-not-found]
    except ImportError:
        return

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    try:
        with HackRFPacketCapture(
            channels=[1], backend="gr_zmq", zmq_endpoint=f"tcp://127.0.0.1:{port}"
        ) as cap:
            time.sleep(0.12)
            pub.send(raw_pdu)
            evs = list(cap.iter_events(duration_s=0.22))
        assert len(evs) == 1
        assert evs[0].frame_type == "BEACON"
        assert evs[0].bssid == "AA:BB:CC:00:00:01"
        assert evs[0].fcs_ok is True
        assert evs[0].seq_num == 33
        assert evs[0].rssi_dbm == pytest.approx(-47.5)
        assert float(np.abs(evs[0].csi[0])) == pytest.approx(
            float(np.hypot(0.75, 0.25)), rel=1e-3
        )
    finally:
        pub.close(0)
