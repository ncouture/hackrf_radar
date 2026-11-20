# HackRF passive Wi-Fi tomographic radar

<img src="./public/hackrf-radar.png" width="640">


[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/ncouture/hackrf_radar)

`hackrf-radar` is a passive (`0 dBm` receive-only) indoor multi-human sensing system built around a single HackRF One software-defined radio. It captures ambient IEEE 802.11a/g/n traffic across 2.4 GHz and 5 GHz channels without associating to any network, transmitting any RF energy, or decrypting protected payloads. From cleartext `L-STF`/`L-LTF` preambles and MAC headers, the pipeline estimates 2D human presence maps, walking trajectories, and stationary respiratory rates (`12` to `24 BPM`).

## System architecture

The codebase is organized into four layers under `hackrf_radar/`:

* `hackrf_radar/capture/` (`packet_event.py`, `soapy_source.py`, `gr_wifi_bridge.py`, `simulator.py`) manages 20 MSPS 8-bit I/Q acquisition, 2.4/5 GHz channel hopping, cleartext 802.11 OFDM preamble synchronization, 52-subcarrier `L-LTF` channel state information (`CSI`) extraction, MAC frame attribution, and synthetic physics simulation.
* `hackrf_radar/tomography/` (`geometry.py`, `features.py`, `solver.py`) computes First Fresnel Zone elliptical sensitivity weights ($\Delta_{ij} < \lambda/2$), extracts link-level RSSI perturbation, sanitized CSI turbulence, and respiratory power spectral density (`0.1` to `25 Hz`), solves non-negative total-variation regularized Radio Tomographic Imaging (`fista_tv`, `nnls`, `sart`), and runs multi-target Kalman tracking.
* `hackrf_radar/web/` (`serializer.py`, `server.py`, `static/`) provides a Python standard-library HTTP and Server-Sent Events (`SSE`) server with a browser UI for live 2D tomography heatmaps, 52-subcarrier `L-LTF` CSI inspection, solver convergence, vital-sign PSD plots, and runtime solver controls.
* `hackrf_radar/config.py`, `hackrf_radar/plotter.py`, `hackrf_radar/cli.py`, and `run_radar.py` provide passive front-end safety locks, JSON room configuration, a 20+ FPS dashboard (`pyqtgraph`, blitted `matplotlib`, `rich` ANSI Truecolor TUI, and `web`), JSONL session recording/replay, and PNG snapshot export.

Because a single HackRF receiver forms a radial fan of chords toward room transmitters, the capture layer also tracks Short Interframe Space (`SIFS`, `10 us` at 2.4 GHz and `16 us` at 5 GHz) control exchanges (`RTS`/`CTS`, `Data`/`ACK`, `BlockAck`). Attributing 1-address `CTS` and `ACK` frames to the preceding frame's receiver address adds cross-room peer chords that suppress single-receiver fan intersection ghosts during multi-human inversion.

## Tomographic pipeline

The `hackrf_radar/tomography/` package (`geometry.py`, `features.py`, and `solver.py`) converts cleartext `PacketEvent` streams into 2D attenuation maps, target tracks, and respiratory estimates.

### First Fresnel Zone and bistatic weights

`RoomGeometry` in `hackrf_radar/tomography/geometry.py` discretizes a `room_width` by `room_height` space (default `6.0 m x 5.0 m`) into `grid_shape = (Nx, Ny)` voxels (default `(40, 34)`, `N = 1360` cells with `dx = 0.15 m` and `dy = 0.147 m`). Voxel centers lie at `((i + 0.5) * dx, (j + 0.5) * dy)` and flatten in row-major `(Ny, Nx)` order into `voxel_coords` of shape `(1360, 2)`.

Transmitter coordinates come from two paths:

* Configured anchors in `known_tx_positions`, normalized to uppercase colon-hex by `canonicalize_mac` and validated on lookup by `is_unicast_station_mac` (rejecting IEEE 802 group or multicast addresses where `(first_octet & 0x01) != 0`).
* Auto-discovered BSSIDs in `observe_node` and `_compute_auto_position`, which wait for `min_auto_packets = 3` observations, blend a SHA-256 hash of the BSSID with a golden-ratio sequence slot around the inset room perimeter, pull the node radially toward `rx_pos` when median RSSI gives a shorter log-distance estimate (`d_est = 10 ** ((P0 - RSSI_med) / (10 * n))` with `P0 = -40.0 dBm` at 2.4 GHz, `-46.5 dBm` at `>= 4.9 GHz`, and `n = 2.4`, applied when `d_est < 0.82 * dist_rx_perim`), and lock the position after 20 RSSI samples.

For link $i$ between $\mathbf{p}_{\text{tx}}$ and $\mathbf{p}_{\text{rx}}$ with baseline distance $d_i = \|\mathbf{p}_{\text{tx}} - \mathbf{p}_{\text{rx}}\|_2$, `compute_excess_path` evaluates the bistatic excess path length at every voxel $\mathbf{x}_j$:

$$
\Delta_{ij} = \max\left(0,\; \|\mathbf{x}_j - \mathbf{p}_{\text{tx}}\|_2 + \|\mathbf{x}_j - \mathbf{p}_{\text{rx}}\|_2 - d_i\right)
$$

Loci of constant $\Delta_{ij} > 0$ form confocal ellipses with foci at $\mathbf{p}_{\text{tx}}$ and $\mathbf{p}_{\text{rx}}$. Given free-space wavelength $\lambda = c / \max(f, 10^6\text{ Hz})$ from `wavelength_m`, `compute_ffz_weights` sets $\Delta_{\max} = \max(0.5 \cdot \text{clearance\_scale} \cdot \lambda,\; 10^{-6}\text{ m})$ and assigns $C^1$-smooth raised-cosine weights normalized by $d_i^{-1/2}$:

$$
W_{ij}^{\text{FFZ}} = \begin{cases} \dfrac{1}{\sqrt{\max(d_i, 0.05)}} \cos^2\left(\dfrac{\pi \Delta_{ij}}{2 \Delta_{\max}}\right), & \Delta_{ij} < \Delta_{\max} \\ 0, & \Delta_{ij} \ge \Delta_{\max} \end{cases}
$$

With default `clearance_scale = 1.0`, $\Delta_{\max} = \lambda / 2$, so support is strictly the First Fresnel Zone $\Delta_{ij} < \lambda / 2$ (semi-major axis $a = d_i/2 + \lambda/4$, semi-minor axis $b = \sqrt{a^2 - (d_i/2)^2} \approx \frac{1}{2}\sqrt{\lambda d_i}$), and weights stay finite directly on $\mathbf{p}_{\text{tx}}$ and $\mathbf{p}_{\text{rx}}$.

`RoomGeometry.get_or_create_chord` caches each `LinkChord` by sub-band (`2.4G`, `5G_LO` for `>= 4.9 GHz`, and `5G_HI` for `>= 5.45 GHz`) with three sensitivity vectors:

* `ffz_weights` sets `clearance_scale = 1.0` for strict $\Delta_{ij} < \lambda / 2$ support.
* `body_weights` widens `clearance_scale` to `max(1.0, (0.5 * lam + 0.035) / (0.5 * lam))`, extending the cutoff to $\Delta_{ij} < \lambda/2 + 0.035\text{ m}$ to match human torso width (`~0.30 m`) so narrow 5 GHz Fresnel ellipses (where the strict excess-path cutoff is `lambda/2 ~ 2.6 cm`) do not fall between `15 cm` grid centers.
* `shell_weights` (`compute_bistatic_shell_weights`) models off-LOS CSI multipath across all $\Delta_{ij} \ge 0$ using wavelength-scaled exponential decay $\exp(-\Delta_{ij}/\ell_{\text{eff}})$ (where $\ell_{\text{eff}} = \max(0.45\sqrt{\lambda/\lambda_{2.4\text{G}}},\; 0.05)\text{ m}$) multiplied by softened bistatic range attenuation $\frac{\max(d_i, 0.25)}{\sqrt{r_{\text{tx}}^2 + r_0^2}\sqrt{r_{\text{rx}}^2 + r_0^2}}$ ($r_0 = 0.35\text{ m}$) and $d_i^{-1/2}$.

In `hackrf_radar/tomography/solver.py`, `TomographicRadarEngine._build_linear_system` constructs forward matrix rows from `0.86 * body_weights + 0.14 * shell_weights` and reserves strict `ffz_weights` ($\Delta_{ij} < \lambda / 2$) for null-chord ghost carving on quiet links.

### RSSI perturbation, sanitized CSI turbulence, and 0.1 to 25 Hz PSD

Upstream of tomography, `sanitize_csi` in `hackrf_radar/capture/packet_event.py` removes timing and oscillator phase nuisance terms from the 52 active 20 MHz 802.11 `L-LTF` subcarriers ($k \in [-26..-1, +1..+26]$, `312.5 kHz` spacing). Because the HackRF is not phase-locked or clock-locked to room transmitters, each packet carries a random LO phase offset $\alpha$ and symbol/packet-detection timing slope $\beta k$. `sanitize_csi` unwraps the lower ($[-26..-1]$) and upper ($[+1..+26]$) half-bands separately, steps across the 2-bin DC null ($k = -1 \to +1$) using the magnitude-product-weighted median bin-to-bin phase difference, fits $\hat{\phi}_k = \alpha + \beta k$ with weights $w_k = (|H_k|/\max|H|)^2 + 10^{-6}$, and outputs $H_{\text{san}}[k] = |H[k]| \exp(j(\phi_{\text{unwrapped}}[k] - \alpha - \beta k))$.

In `hackrf_radar/tomography/features.py`, `LinkWindowExtractor.ingest` buffers packets into 12-second deques (`_SingleStreamStats`, with `max_history_s = vital_window_s + 2.0`). Standalone frames go into `_direct_streams` (`TX -> RX`), whereas SIFS control replies (`ACK`, `CTS`, `BLOCK_ACK`, or `attributed_via_sifs=True`) between localized anchors go into `_peer_streams` (`TX_A -> Peer_B`), and non-SIFS frames between peer nodes record 12-bit MAC sequence-number gaps (`seq_gap`) in both directions of `_peer_seq_gaps`.

At each step, `extract` inspects packets over the full window `[now - window_s, now]` (default `1.5 s`, tracking `packet_count` and `fcs_error_rate` across all packets and features over FCS-valid packets) and a fast sub-window `[now - min(window_s, 0.55), now]` (falling back to the full FCS-valid window when fewer than `2` valid packets exist for RSSI or fewer than `3` exist for dynamics):

* Direct RSSI perturbation (`rssi_delta_db`) takes `inst_rssi` as the fast-window median. Under empty-room calibration (`calibrate_baseline`), $\Delta\text{RSSI} = \max(0, \text{baseline\_rssi} - \text{inst\_rssi})$. When uncalibrated, `_fit_uncalibrated_path_loss` normalizes every direct chord's 75th-percentile RSSI to a 1 m 2.4 GHz equivalent $P_{0,\text{equiv}} = \text{RSSI}_{75} + \Delta_{\text{band}} + 22\log_{10}(\max(d_i, 0.35))$ ($\Delta_{\text{band}} = 6.5\text{ dB}$ for `>= 4.9 GHz`), uses the 90th percentile across links (`max` when `< 3` links) as the unobstructed reference $P_{0,\text{ref}}$, and computes $\Delta\text{RSSI} = \max(0, \hat{R}_i - \text{inst\_rssi} - 0.55)$.
* Sanitized CSI turbulence (`csi_turbulence`) stacks fast-window `csi_sanitized` vectors into $C \in \mathbb{C}^{M \times 52}$ with mean $\bar{\mathbf{c}}$ and normalizes the complex temporal standard deviation by the per-subcarrier RMS magnitude:

$$
\sigma_{\text{CSI}} = \frac{\sqrt{\frac{1}{52M}\sum_{m=1}^{M}\sum_{k=1}^{52} \left|C_{m,k} - \bar{c}_k\right|^2}}{\|\bar{\mathbf{c}}\|_2 / \sqrt{52}}
$$

* Feature extraction also computes calibrated baseline vector shift $\|\bar{\mathbf{c}} - \mathbf{c}_{\text{base}}\|_2 / \|\bar{\mathbf{c}}\|_2$ and adjacent-subcarrier `multipath_spread`, combines noise-gated $\Delta\text{RSSI}$, $\sigma_{\text{CSI}}$, RSSI 15th-to-85th interpercentile spread, calibrated CSI shift, and a `+0.22` vital-sign bonus into `perturbation_score` (clamped to `[0, 1.5]` and zeroed below `0.07`), and subtracts `tx_a`'s direct radial RSSI drop and CSI turbulence from `_peer_streams` while adding `seq_gap_rate` to isolate cross-room blockage on SIFS peer chords (zeroed below `0.08`).

For vital signs, `_compute_vital_psd` runs over `vital_window_s = 10.0 s` without uniform resampling (`PSD_SAMPLE_RATE_HZ = 50.0` and `PSD_NFFT = 4096` are unused constants). Instead, it evaluates `scipy.signal.lombscargle` on irregular packet arrival timestamps across 512 frequencies (`psd_freqs_hz`): 320 fine bins in `[0.10, 1.20] Hz` (`~0.00345 Hz` spacing) concatenated with 192 coarse bins in `[1.25, 25.0] Hz`.

The PSD pipeline executes these checks and transforms:

1. Requires at least `18` FCS-valid packets, `2.4 s` span, and `16` distinct timestamps.
2. Flags macroscopic walking shadow crossings (`is_macro_transit`) if `P95 - P5 > 2.85 dB` or half-window medians shift by `> 1.25 dB`.
3. Applies timestamp-linear ordinary-least-squares detrending (`_detrend_time`, immune to channel-hopping dwell gaps) to packet RSSI (`rssi_dt`) and quadrature CSI phase perturbation $\phi_m = \frac{1}{52}\sum_k \operatorname{Im}(C_{m,k}\bar{c}_k^*) / \max(|\bar{c}_k|^2, 10^{-8})$ scaled by `8.0` (`phase_dt`).
4. Forms the variance-weighted spectrum $P(f) = \operatorname{Var}(r_{\text{dt}})L_{\text{RSSI}}(f) + \operatorname{Var}(\phi_{\text{dt}})L_{\text{phase}}(f)$ and normalized coherence $\gamma(f) = P(f)/(\operatorname{Var}(r_{\text{dt}}) + \operatorname{Var}(\phi_{\text{dt}})) \in [0, 1]$.
5. Selects the strongest strictly interior local maximum inside `[RESP_MIN_HZ, RESP_MAX_HZ] = [0.18, 0.45] Hz` (`10.8` to `27.0 BPM`), rejecting candidates when `is_macro_transit` holds, $\gamma(f_{\text{pk}}) < 0.30$, median sideband SNR (`[0.10, 2.20] Hz` outside `+-0.065 Hz` of $f_{\text{pk}}$) `< 9.0 dB`, 85th-percentile sideband SNR `< 4.5 dB`, peak power `< 0.032`, or `< 0.18 Hz` skirt `> 1.15 * P(f_pk)`.
6. Refines frequency via 3-point parabolic interpolation on $\ln P(f)$ and requires at least `1.25` observed breathing cycles (`eff_span * f_est >= 1.25`) within `[10.5, 27.5] BPM`.

### Non-negative RTI inversion (fista_tv, nnls, sart)

In `hackrf_radar/tomography/solver.py`, `_build_linear_system` constructs $\mathbf{W} \in \mathbb{R}^{M \times N}$ and $\mathbf{y} \in \mathbb{R}^M$ from active chord summaries:

* Multiplies `0.86 * body_weights + 0.14 * shell_weights` by Gaussian spatial tapers around `rx_pos` (`_rx_taper`, radius `0.55 m`) and anchor endpoints (`0.45 m`), plus a linear room-wall margin taper (`_wall_taper`, clamped to `[0.20, 1.0]` within `0.58 m` of walls) to prevent spoke-fan pileup at shared vertices, then normalizes each row by `6.0 * max(raw_w)`.
* Doubles row weight (`row_mult = 2.0`) on active unperturbed links (`perturbation_score < 0.05`, `packet_count >= 3`) so $y_i \approx 0$ carves away ghost intersections, while accumulating their strict `ffz_weights` into `quiet_mask_acc`.
* Computes a wide-angle chord-intersection consistency prior `cross_prior` over pairs of hot chords ($s_i, s_k \ge 0.12$) crossing at $|\sin\theta_{ik}| \ge 0.22$ (`>= 12.7 deg`), attenuating near shared endpoints and quiet chords. In `step()`, columns are scaled by $\mathbf{W}_{\text{eff}} = \mathbf{W}\operatorname{diag}(\mathbf{c}_{\text{prior}})$.

All three non-negative solvers return an exact zero `(Ny, Nx)` map when $\mathbf{W}_{\text{eff}}$ is empty or $\max(\mathbf{y}) \le 10^{-6}$, and record per-iteration objectives when `history_out` is supplied:

* `fista_tv` (`solve_fista_tv`, `max_iter = 45`, defaults `lam_l1 = 0.006`, `lam_l2 = 0.020`, `lam_tv = 0.006`) minimizes non-negative isotropic 2D Total Variation plus Elastic-Net regularized least squares over $\mathbf{x} \ge \mathbf{0}$:

$$
J(\mathbf{x}) = \frac{1}{2}\|\mathbf{W}_{\text{eff}}\mathbf{x} - \mathbf{y}\|_2^2 + \frac{\lambda_{\ell_2}}{2}\|\mathbf{x}\|_2^2 + \lambda_{\ell_1}\mathbf{1}^\top\mathbf{x} + \lambda_{\text{TV}}\sum_{r,c}\sqrt{(x_{r,c+1}-x_{r,c})^2 + (x_{r+1,c}-x_{r,c})^2}
$$

It estimates the Lipschitz bound $L = 1.05(\|\mathbf{W}_{\text{eff}}^\top\mathbf{W}_{\text{eff}}\|_2 + \lambda_{\ell_2})$ via 8 power iterations (`_estimate_lipschitz`), takes a projected gradient and $\ell_1$ step $\mathbf{z}_k = \max(0, \mathbf{v}_k - \frac{1}{L}(\mathbf{W}_{\text{eff}}^\top(\mathbf{W}_{\text{eff}}\mathbf{v}_k - \mathbf{y}) + \lambda_{\ell_2}\mathbf{v}_k + \lambda_{\ell_1}))$, applies 3 inner iterations of Chambolle dual projection with non-negativity clamping (`_prox_tv2d_nonneg`, $\tau = 0.24$), and updates Nesterov momentum $t_{k+1} = \frac{1}{2}(1 + \sqrt{1 + 4t_k^2})$.

* `nnls` (`solve_nnls_regularized`, `max_iter = 34`, default `lam_l1 = 0.004`, `lam_l2 = max(lam_l2, lam_tv)` in `step()`) drops the TV term, minimizing $\frac{1}{2}\|\mathbf{W}_{\text{eff}}\mathbf{x} - \mathbf{y}\|_2^2 + \frac{\lambda_{\ell_2}}{2}\|\mathbf{x}\|_2^2 + \lambda_{\ell_1}\mathbf{1}^\top\mathbf{x}$ over $\mathbf{x} \ge \mathbf{0}$ via Nesterov-accelerated projected gradient descent, followed by a `sigma = 0.65` voxel Gaussian filter.
* `sart` (`solve_sart`, `max_iter = 45`) runs Simultaneous Algebraic Reconstruction Technique with over-relaxation $\omega = 1.35$, row sums $R_i = \max(\sum_j W_{ij}, 10^{-6})$, regularized column sums $C_j = \max(\sum_i W_{ij} + 0.38\max_k\sum_i W_{ik}, 10^{-6})$, and shrinkage $s = 0.0004\max(\mathbf{y})$, followed by a `sigma = 0.55` voxel Gaussian filter:

$$
\mathbf{x}_{k+1} = \max\left(\mathbf{0},\; \mathbf{x}_k + \omega\,\mathbf{C}^{-1}\mathbf{W}_{\text{eff}}^\top\mathbf{R}^{-1}(\mathbf{y} - \mathbf{W}_{\text{eff}}\mathbf{x}_k) - s\right)
$$

After inversion, `step()` applies temporal exponential smoothing ($\mathbf{D}_t = 0.64\,\mathbf{D}_{\text{raw}} + 0.36\,\mathbf{D}_{t-1}$) to stabilize channel-hopping dwell transitions.

### Multi-target Kalman tracking and breathing attribution

`_extract_peaks` smooths `density_map` (`sigma = 1.0`), requires peak density `>= 0.14`, thresholds at `max(0.15, 0.37 * D_max)`, detects local maxima with `ndimage.maximum_filter`, computes sub-voxel center-of-mass coordinates inside a `+-0.42 m` patch, and applies greedy non-maximum suppression with `min_sep_m = 0.90 m` (up to 8 detections).

`MultiTargetKalmanTracker` and `_KalmanTrackState` maintain a 4D constant-velocity state $[x, y, v_x, v_y]^\top$ with initial covariance $\operatorname{diag}(0.18, 0.18, 0.45, 0.45)$:

* Prediction ($\Delta t = \operatorname{clip}(t - t_{\text{last}}, 0, 0.6)\text{ s}$) decays velocity by $e^{-0.15\Delta t}$, clamps $(x, y)$ `0.25 m` inside room walls, and adds process noise $\mathbf{Q} = \operatorname{diag}(0.05\Delta t, 0.05\Delta t, 1.15\Delta t, 1.15\Delta t)$.
* Global assignment builds the Euclidean distance matrix and runs `scipy.optimize.linear_sum_assignment` within `gate_distance_m = 1.65 m`.
* Update applies linear position measurement $\mathbf{H} = [\mathbf{I}_2 \;\; \mathbf{0}_{2\times 2}]$ with $\mathbf{R} = \operatorname{diag}(0.045, 0.045)\text{ m}^2$, caps speed at `1.8 m/s`, and updates `smoothed_speed` via a `0.60 / 0.40` EMA.
* Lifecycle rules coast unmatched tracks until `misses * dt_miss > 0.42 s` (or `hits < 4` with `misses >= 2`), spawn new tracks when unmatched detections sit `>= 0.85 m` from existing tracks, confirm after `hits >= 4` (for `step_count >= 6`), and label `smoothed_speed >= 0.14 m/s` as `"walking"` and slower targets as `"stationary"`.

`_attribute_breathing` projects per-chord linear PSDs onto each stationary track's voxel weighted by `sens_target / (1.0 + 8.0 * walker_load)` (down-weighting chords crossed by concurrent walkers) and smooths the `[10.5, 27.5] BPM` rate with a `0.50 / 0.50` EMA.

## Hardware setup and gain staging

All live capture paths enforce `0 dBm` passive reception in both configuration (`RxSettings` in `hackrf_radar/config.py`) and driver initialization (`hackrf_radar/capture/soapy_source.py`):

* Transmit APIs (`hackrf_start_tx`, `soapy_hackrf_sink`, `hackrf_transfer -t`) are absent and forbidden.
* The HackRF One MGA-81563 `+14 dB` RF front-end amplifier is locked off (`amp_enable = False`, `-a 0`). Enabling the RF amplifier near indoor Wi-Fi access points saturates the MAX5864 8-bit ADC and creates intermodulation artifacts across subcarriers.
* Antenna port DC bias-tee power is locked off (`bias_tee_enable = False`).
* Manual gain staging defaults to `lna_db = 24` (MAX2837 IF LNA, `0..40 dB` in `8 dB` steps) and `vga_db = 20` (baseband VGA, `0..62 dB` in `2 dB` steps) at `20 MSPS`.
* Watch the `ADC Clip` indicator in the passive Wi-Fi HUD during live operation. If the fraction of int8 samples with `|I| >= 120` or `|Q| >= 120` rises above `1%`, lower `--lna-db` by one `8 dB` step (to `16`) or reduce `--vga-db` to `14`. If packet SNR is low in a large room, raise `--vga-db` in `4 dB` increments while keeping `AMP` off.
* Place the HackRF One with a 2.4/5 GHz dipole antenna near one wall (for example `(3.0, 0.35)` m on the south wall of a `6.0 m x 5.0 m` room) with vertical polarization matching nearby access points. After each channel hop (`Ch 1: 2412 MHz`, `Ch 6: 2437 MHz`, `Ch 11: 2462 MHz`, plus optional 5 GHz UNII channels `36..165`), the capture loop discards `2` to `5 ms` of post-tune PLL settling samples and isolates per-channel state.

## GNU Radio bridge and native OFDM fallback

`hackrf_radar/capture/gr_wifi_bridge.py` and `grc/hackrf_wifi_passive_radar.grc` support multiple live and offline ingestion paths so capture works across varied host installations:

* When `SoapyHackRF` is installed, `HackRFPacketCapture(backend="auto")` streams 20 MSPS CS8 I/Q via `SoapySDR`. When `libHackRFSupport.so` is absent on the host, `"auto"` falls back automatically to `ctypes.CDLL("libhackrf.so.0")`, `gnuradio-osmosdr` (`osmosdr.source("hackrf=0")`), or `hackrf_transfer -r -`.
* When GNU Radio 3.10 and `gr-ieee802-11` are present, `grc/hackrf_wifi_passive_radar.grc` (or the programmatic flowgraph builder) decodes 802.11 frames and publishes PDUs over ZMQ (`--backend gr_zmq --zmq-endpoint tcp://127.0.0.1:5555`) or Radiotap PCAP (`--mode replay --input capture.pcap`), parsed with Python's `struct` module without requiring `scapy`.
* When binary `gr-ieee802-11` is not installed, `gr_wifi_bridge.py` executes a native NumPy/SciPy 20 MSPS 802.11a/g/n OFDM burst detector: Schmidl-Cox `L-STF` coarse timing and carrier frequency offset (`CFO`) estimation, `L-LTF` cross-correlation fine symbol alignment, 64-point FFT channel estimation across all 52 active subcarriers (`k in [-26..-1, +1..+26]`, tracking pilots `{-21, -7, +7, +21}`), linear sampling/packet-detection phase slope (`STO`/`PDD`) and random phase offset sanitization, 6 Mbps BPSK `L-SIG` and MAC header demodulation, and IEEE 802.11 CRC-32 (`FCS`) verification.
* From cleartext MAC headers, the bridge extracts `Beacon` 64-bit `TSF` timestamps to regress per-BSSID clock drift (`ppm`), tracks 12-bit `Sequence Control` gaps per transmitter, and counts `FCS` CRC drops as interference telemetry.

## Calibrating BSSID anchor positions

Tomographic spatial accuracy depends on knowing the `(x, y)` coordinates (in room meters, origin `(0, 0)` at the south-west corner) of the HackRF receiver and major ambient Wi-Fi transmitters.

1. Measure room width (`room_width`) and length (`room_height`) in meters and record the HackRF antenna position in `rx.rx_pos` inside a copy of `configs/example_room.json`.
2. Run a short live scan (`./run_radar.py --mode live --duration 5 --ui tui --record scan.jsonl`) to list active BSSIDs, channels, and frame rates in your space.
3. For fixed access points and stationary smart-home devices (plugs, speakers, printers, thermostats, cameras) whose locations you know, add entries under `"anchors"` in your JSON config with canonical uppercase MAC addresses (`"AA:BB:CC:00:00:01"`), `x`, `y`, and `channel`.
4. Keep `"auto_discovery": {"enabled": true}` in the JSON file so uncalibrated BSSIDs observed on air are placed automatically along the room perimeter using log-distance RSSI ranging and chord consistency.
5. For stationary rooms, pass `--calibrate-s 3.0` when nobody is inside the room so the engine records a clean empty-room baseline before tracking occupancy.

## Realtime web interface

`hackrf-radar` includes a browser interface served directly from Python's standard library (`http.server.ThreadingHTTPServer` and `text/event-stream`), requiring no extra web dependencies. Pass `--ui web` to run the browser dashboard standalone, or add `--web` alongside `tui`, `matplotlib`, or `pyqtgraph` to mirror frames to both displays.

The page displays the physical room voxel attenuation map with First Fresnel Zone ellipses, direct and SIFS peer chords, Kalman-tracked human positions and breathing rates, per-chord 52-subcarrier `L-LTF` CSI amplitude and sanitized phase, solver objective convergence (`J(x_k)`), `0.1` to `25 Hz` respiratory PSD, and live RF HUD metrics. Clicking any chord pins its Fresnel zone and subcarrier CSI profile; controls in the lower deck switch solvers (`fista_tv`, `nnls`, `sart`), tune regularization `lambda` and sliding window duration, trigger empty-room baseline calibration, pause rendering, or drag simulated targets in `--mode sim`.

HTTP endpoints bind to loopback (`127.0.0.1:8765`) by default and keep passive `0 dBm` RX-only locks enforced:

* `GET /` serves the Luminous light/dark web application with a strict Content Security Policy (no inline scripts).
* `GET /api/stream` streams realtime `config` and `frame` events over Server-Sent Events with drop-oldest backpressure so slow browser tabs do not stall capture.
* `GET /api/frame`, `GET /api/config`, and `GET /api/health` return the latest serialized `TomographyFrame`, room settings, and server health as JSON.
* `POST /api/control` accepts same-origin JSON updates for solver method, regularization, observation window, baseline calibration, and simulation target placement while rejecting any `amp_enable`, `bias_tee_enable`, or transmit keys.

## Command-line usage

Run the multi-human simulator (1, 2, or 3+ targets) with the realtime web interface, interactive GUI, or headless PNG snapshot export:

```bash
# Realtime browser tomography UI at http://127.0.0.1:8765/ (streams continuously until Ctrl+C)
python3 run_radar.py --mode sim --humans 2 --scenario walk_and_breathe --ui web

# Concurrent Rich TUI + realtime web UI on a custom port
python3 run_radar.py --mode sim --humans 2 --ui tui --web --web-port 8080

# 2-human walk + stationary breathing simulation (auto-selects pyqtgraph or matplotlib)
python3 run_radar.py --mode sim --humans 2 --scenario walk_and_breathe

# 3-human simulation saving a headless PNG snapshot
MPLBACKEND=Agg python3 run_radar.py --mode sim --humans 3 --duration 5.0 --no-gui --ui matplotlib --save-snapshot sim_3humans.png

# High-FPS PyQtGraph dashboard under system Python (interactive X11 or offscreen)
/usr/bin/python3 run_radar.py --mode sim --humans 2 --ui pyqtgraph --duration 5.0

# Rich ANSI Truecolor half-block TUI for SSH and terminal sessions
python3 run_radar.py --mode sim --humans 2 --ui tui --duration 5.0
```

Capture live ambient Wi-Fi passively from an attached HackRF One (`0 dBm` RX-only, `AMP` off) and record `PacketEvent` telemetry for offline replay:

```bash
# Live passive HackRF capture across channels 1, 6, and 11 with realtime browser UI
/usr/bin/python3 run_radar.py --mode live --config configs/example_room.json --channels 1 6 11 --lna-db 24 --vga-db 20 --ui web --record session.jsonl

# Replay a recorded JSONL session, Radiotap PCAP, or raw CS8/CF32 I/Q capture
python3 run_radar.py --mode replay --input session.jsonl --ui matplotlib --save-snapshot replay_check.png
```

## Dual-Python environment and testing

Every module runs under both standard Python 3.12 (`numpy`, `scipy`, `matplotlib`, `rich`) and system `/usr/bin/python3` 3.14 (which additionally provides `pyqtgraph`, `PyQt6`, `SoapySDR`, `gnuradio`, and `osmosdr`):

```bash
python3 -m pytest tests/
/usr/bin/python3 -m pytest tests/
```
