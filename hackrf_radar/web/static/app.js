(function () {
  'use strict';

  const SPEED_OF_LIGHT = 299792458.0;
  const THEME_STORAGE_KEY = 'hackrf-radar-color-scheme';
  const PREFS_STORAGE_KEY = 'hackrf-radar-ui-prefs-v1';
  const SUBCARRIERS = [];
  for (let k = -26; k <= 26; k += 1) {
    if (k !== 0) SUBCARRIERS.push(k);
  }
  const PILOTS = [-21, -7, 7, 21];

  const reducedMotionQuery = window.matchMedia
    ? window.matchMedia('(prefers-reduced-motion: reduce)')
    : null;
  const darkSchemeQuery = window.matchMedia
    ? window.matchMedia('(prefers-color-scheme: dark)')
    : null;

  const ui = {
    paused: false,
    plotTab: 'convergence', // 'convergence' | 'psd'
    sifsPulseEnabled: !(reducedMotionQuery && reducedMotionQuery.matches),
    showGroundTruth: true,
    pinnedLinkId: null,
    hoveredLinkId: null,
    kbLinkIndex: -1,
    draggingTargetId: null,
    lastTargetSendMs: 0,
    isEditingSlider: false,
    sifsPhase: 0,
    lastAnimTs: performance.now(),
    offscreenGrid: document.createElement('canvas'),
    controlTimers: {},
    reconnectTimer: null,
    reconnectDelayMs: 600,
    eventSource: null,
    config: {
      room_name: 'Passive Wi-Fi Tomographic Room',
      room_width: 6.0,
      room_height: 5.0,
      sim_interactive: false,
      control_enabled: false,
    },
    frame: null,
  };

  function getCssVar(name, fallback) {
    try {
      const val = getComputedStyle(document.documentElement)
        .getPropertyValue(name)
        .trim();
      return val || fallback || '';
    } catch (_) {
      return fallback || '';
    }
  }

  function hexToRgb(colorStr) {
    if (!colorStr) return [49, 134, 255];
    const m = colorStr.trim();
    if (m.startsWith('rgb')) {
      const nums = m.match(/[\d.]+/g);
      if (nums && nums.length >= 3) return [+nums[0], +nums[1], +nums[2]];
    }
    let hex = m.replace('#', '');
    if (hex.length === 3) {
      hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    }
    const num = parseInt(hex, 16);
    if (Number.isNaN(num)) return [49, 134, 255];
    return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
  }

  function announce(msg) {
    const live = document.getElementById('live-status');
    if (!live || !msg) return;
    live.textContent = '';
    window.setTimeout(() => {
      live.textContent = msg;
    }, 30);
  }

  function loadPrefs() {
    try {
      const raw = window.localStorage.getItem(PREFS_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (typeof parsed.sifsPulseEnabled === 'boolean') {
        ui.sifsPulseEnabled = parsed.sifsPulseEnabled;
      }
      if (typeof parsed.showGroundTruth === 'boolean') {
        ui.showGroundTruth = parsed.showGroundTruth;
      }
      if (parsed.plotTab === 'convergence' || parsed.plotTab === 'psd') {
        ui.plotTab = parsed.plotTab;
      }
    } catch (_) {}
  }

  function savePrefs() {
    try {
      window.localStorage.setItem(
        PREFS_STORAGE_KEY,
        JSON.stringify({
          sifsPulseEnabled: ui.sifsPulseEnabled,
          showGroundTruth: ui.showGroundTruth,
          plotTab: ui.plotTab,
        })
      );
    } catch (_) {}
  }

  // Two-state dark mode toggle following modern-web-guidance dark-mode-toggle pattern
  function getSystemScheme() {
    return darkSchemeQuery && darkSchemeQuery.matches ? 'dark' : 'light';
  }

  function getStoredScheme() {
    try {
      const v = window.localStorage.getItem(THEME_STORAGE_KEY);
      return v === 'light' || v === 'dark' ? v : null;
    } catch (_) {
      return null;
    }
  }

  function getRenderedScheme() {
    return getStoredScheme() || getSystemScheme();
  }

  function applyColorScheme(storedOverride) {
    const meta = document.querySelector('meta[name="color-scheme"]');
    if (storedOverride === 'light' || storedOverride === 'dark') {
      if (meta) meta.setAttribute('content', storedOverride);
      document.documentElement.setAttribute('data-theme', storedOverride);
    } else {
      if (meta) meta.setAttribute('content', 'light dark');
      document.documentElement.removeAttribute('data-theme');
    }
    const rendered = storedOverride || getSystemScheme();
    const label = document.getElementById('theme-label');
    const btn = document.getElementById('btn-theme');
    if (label) {
      label.textContent = rendered === 'dark' ? 'Light mode' : 'Dark mode';
    }
    if (btn) {
      btn.setAttribute(
        'aria-label',
        rendered === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'
      );
    }
    renderAllCanvases();
  }

  function toggleColorScheme() {
    const rendered = getRenderedScheme();
    const target = rendered === 'dark' ? 'light' : 'dark';
    const sys = getSystemScheme();
    let nextStored = null;
    if (target !== sys) {
      nextStored = target;
      try {
        window.localStorage.setItem(THEME_STORAGE_KEY, target);
      } catch (_) {}
    } else {
      try {
        window.localStorage.removeItem(THEME_STORAGE_KEY);
      } catch (_) {}
    }
    applyColorScheme(nextStored);
    announce('Switched to ' + target + ' color scheme.');
  }

  function updateSliderProgress(input) {
    if (!input) return;
    const mn = parseFloat(input.min || '0');
    const mx = parseFloat(input.max || '100');
    const val = parseFloat(input.value);
    const pct = mx > mn ? ((val - mn) / (mx - mn)) * 100 : 50;
    input.style.setProperty('--progress', Math.max(0, Math.min(100, pct)).toFixed(1) + '%');
  }

  function setConnectionState(state, text) {
    const pill = document.getElementById('conn-pill');
    const label = document.getElementById('conn-text');
    if (pill) pill.setAttribute('data-state', state);
    if (label) label.textContent = text;
  }

  async function sendControl(payload, announceMsg) {
    try {
      const resp = await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({}));
        const reason = errBody.error || 'Control request rejected';
        announce(reason);
        return false;
      }
      if (announceMsg) announce(announceMsg);
      return true;
    } catch (_) {
      announce('Unable to reach radar control endpoint.');
      return false;
    }
  }

  function scheduleControl(key, payload, delayMs) {
    if (ui.controlTimers[key]) {
      window.clearTimeout(ui.controlTimers[key]);
    }
    ui.controlTimers[key] = window.setTimeout(() => {
      delete ui.controlTimers[key];
      sendControl(payload);
    }, delayMs);
  }

  function applyConfig(cfg) {
    if (!cfg || typeof cfg !== 'object') return;
    ui.config = Object.assign({}, ui.config, cfg);

    const w = Number(ui.config.room_width) || 6.0;
    const h = Number(ui.config.room_height) || 5.0;
    const heading = document.getElementById('room-heading');
    if (heading) {
      heading.textContent = w.toFixed(1) + 'm × ' + h.toFixed(1) + 'm Room & Attenuation Grid';
    }

    const hint = document.getElementById('room-hint');
    if (hint) {
      hint.textContent = ui.config.sim_interactive
        ? 'Drag target or click chord'
        : 'Click chord to inspect';
    }

    const resetBtn = document.getElementById('btn-reset-target');
    if (resetBtn) {
      resetBtn.classList.toggle('is-hidden', !ui.config.sim_interactive);
    }

    const controlEnabled = ui.config.control_enabled !== false;
    ['slider-lambda', 'slider-window', 'btn-fista', 'btn-nnls', 'btn-sart', 'btn-calibrate'].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.disabled = !controlEnabled;
    });
  }

  function getActiveLink(frame) {
    const links = frame && Array.isArray(frame.links) ? frame.links : [];
    if (!links.length) return null;
    if (ui.hoveredLinkId) {
      const found = links.find((l) => l.link_id === ui.hoveredLinkId);
      if (found) return found;
    }
    if (ui.pinnedLinkId) {
      const found = links.find((l) => l.link_id === ui.pinnedLinkId);
      if (found) return found;
    }
    let best = links[0];
    let bestScore = -1;
    for (let i = 0; i < links.length; i += 1) {
      const score = Number(links[i].perturbation_score) || 0;
      if (score > bestScore) {
        bestScore = score;
        best = links[i];
      }
    }
    return best;
  }

  function sizeCanvas(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const w = canvas.clientWidth || 320;
    const h = canvas.clientHeight || 180;
    const targetW = Math.max(1, Math.floor(w * dpr));
    const targetH = Math.max(1, Math.floor(h * dpr));
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function getRoomViewport(w, h, roomW, roomH) {
    const padX = 28;
    const padTop = 18;
    const padBot = 26;
    const availW = Math.max(80, w - padX * 2);
    const availH = Math.max(80, h - padTop - padBot);
    const scale = Math.min(availW / roomW, availH / roomH);
    const drawW = roomW * scale;
    const drawH = roomH * scale;
    const ox = (w - drawW) * 0.5;
    const oy = padTop + (availH - drawH) * 0.5;
    return {
      ox,
      oy,
      drawW,
      drawH,
      scale,
      roomW,
      roomH,
      toPxX(mx) {
        return ox + mx * scale;
      },
      toPxY(my) {
        // Mathematical room Y: 0m at south wall (bottom), roomH at north wall (top)
        return oy + (roomH - my) * scale;
      },
      toRoomX(px) {
        return (px - ox) / scale;
      },
      toRoomY(py) {
        return roomH - (py - oy) / scale;
      },
    };
  }

  function renderRoomCanvas() {
    const canvas = document.getElementById('roomCanvas');
    if (!canvas) return;
    const { ctx, w, h } = sizeCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const frame = ui.frame;
    const roomW = (frame && frame.room && frame.room.width) || ui.config.room_width || 6.0;
    const roomH = (frame && frame.room && frame.room.height) || ui.config.room_height || 5.0;
    const vp = getRoomViewport(w, h, roomW, roomH);

    const surfHex = getCssVar('--surface', '#fdfcfc');
    const onSurf = getCssVar('--on-surface', '#000000');
    const onVar = getCssVar('--on-surface-variant', 'rgba(0,0,0,0.62)');
    const strokeDef = getCssVar('--stroke-default', '#e3e3e3');
    const chart1 = getCssVar('--chart-1', '#2575fc');
    const chart3 = getCssVar('--chart-3', '#228a49');
    const chart8 = getCssVar('--chart-8', '#997112');
    const errorCol = getCssVar('--error', '#de2d29');
    const monoFont = "11px 'Google Sans Code', monospace";

    ctx.save();
    ctx.beginPath();
    ctx.roundRect(vp.ox, vp.oy, vp.drawW, vp.drawH, 10);
    ctx.clip();

    // 1. Live reconstructed voxel attenuation heatmap
    if (frame && frame.density_map && frame.density_map.nx > 0 && frame.density_map.ny > 0) {
      const nx = frame.density_map.nx;
      const ny = frame.density_map.ny;
      const vals = frame.density_map.values || [];
      const maxVal = Math.max(0.65, Number(frame.density_map.max) || 0);

      if (ui.offscreenGrid.width !== nx || ui.offscreenGrid.height !== ny) {
        ui.offscreenGrid.width = nx;
        ui.offscreenGrid.height = ny;
      }
      const offCtx = ui.offscreenGrid.getContext('2d');
      const imgData = offCtx.createImageData(nx, ny);
      const bgRgb = hexToRgb(surfHex);
      const midRgb = hexToRgb(chart1);
      const hotRgb = hexToRgb(chart8);
      const peakRgb = hexToRgb(errorCol);

      for (let gy = 0; gy < ny; gy += 1) {
        // Flip Y so row 0 (south wall y=0) draws at the bottom of the canvas
        const dstRow = ny - 1 - gy;
        for (let gx = 0; gx < nx; gx += 1) {
          const v = Number(vals[gy * nx + gx]) || 0;
          const t = Math.min(1, Math.max(0, v / maxVal));
          let r;
          let g;
          let b;
          if (t < 0.45) {
            const u = t / 0.45;
            r = bgRgb[0] + (midRgb[0] - bgRgb[0]) * u;
            g = bgRgb[1] + (midRgb[1] - bgRgb[1]) * u;
            b = bgRgb[2] + (midRgb[2] - bgRgb[2]) * u;
          } else if (t < 0.8) {
            const u = (t - 0.45) / 0.35;
            r = midRgb[0] + (hotRgb[0] - midRgb[0]) * u;
            g = midRgb[1] + (hotRgb[1] - midRgb[1]) * u;
            b = midRgb[2] + (hotRgb[2] - midRgb[2]) * u;
          } else {
            const u = (t - 0.8) / 0.2;
            r = hotRgb[0] + (peakRgb[0] - hotRgb[0]) * u;
            g = hotRgb[1] + (peakRgb[1] - hotRgb[1]) * u;
            b = hotRgb[2] + (peakRgb[2] - hotRgb[2]) * u;
          }
          const p = (dstRow * nx + gx) * 4;
          imgData.data[p] = r;
          imgData.data[p + 1] = g;
          imgData.data[p + 2] = b;
          imgData.data[p + 3] = t > 0.01 ? Math.round(28 + 220 * Math.pow(t, 0.72)) : 0;
        }
      }
      offCtx.putImageData(imgData, 0, 0);
      const solverMethod = frame.solver ? frame.solver.method : 'fista_tv';
      ctx.imageSmoothingEnabled = solverMethod !== 'fista_tv';
      ctx.drawImage(ui.offscreenGrid, vp.ox, vp.oy, vp.drawW, vp.drawH);
    }

    // 2. Subtle 1 m metric grid lines
    ctx.strokeStyle = strokeDef;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 1;
    for (let mx = 1; mx < roomW; mx += 1) {
      const gx = vp.toPxX(mx);
      ctx.beginPath();
      ctx.moveTo(gx, vp.oy);
      ctx.lineTo(gx, vp.oy + vp.drawH);
      ctx.stroke();
    }
    for (let my = 1; my < roomH; my += 1) {
      const gy = vp.toPxY(my);
      ctx.beginPath();
      ctx.moveTo(vp.ox, gy);
      ctx.lineTo(vp.ox + vp.drawW, gy);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    const links = frame && Array.isArray(frame.links) ? frame.links : [];
    const activeLink = getActiveLink(frame);
    const c1Rgb = hexToRgb(chart1);
    const c8Rgb = hexToRgb(chart8);
    const errRgb = hexToRgb(errorCol);
    const onVarRgb = hexToRgb(onVar);
    const c3Rgb = hexToRgb(chart3);

    // 3. First Fresnel Zone ellipse (Delta = lambda / 2) on active/inspected chord
    if (activeLink) {
      const x1 = activeLink.tx_pos[0];
      const y1 = activeLink.tx_pos[1];
      const x2 = activeLink.rx_pos[0];
      const y2 = activeLink.rx_pos[1];
      const dx = x2 - x1;
      const dy = y2 - y1;
      const dist = Math.hypot(dx, dy);
      if (dist > 0.05) {
        const freq = Math.max(1e6, Number(activeLink.freq_hz) || 2437e6);
        const lam = SPEED_OF_LIGHT / freq;
        const deltaMax = 0.5 * lam;
        const semiMajor = 0.5 * (dist + deltaMax);
        const semiMinor = 0.5 * Math.sqrt(deltaMax * (2 * dist + deltaMax));
        const px1 = vp.toPxX(x1);
        const py1 = vp.toPxY(y1);
        const px2 = vp.toPxX(x2);
        const py2 = vp.toPxY(y2);
        const ecx = 0.5 * (px1 + px2);
        const ecy = 0.5 * (py1 + py2);
        const angle = Math.atan2(py2 - py1, px2 - px1);

        ctx.save();
        ctx.translate(ecx, ecy);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.ellipse(0, 0, semiMajor * vp.scale, semiMinor * vp.scale, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(' + c1Rgb[0] + ',' + c1Rgb[1] + ',' + c1Rgb[2] + ',0.14)';
        ctx.fill();
        ctx.strokeStyle = chart1;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.restore();
      }
    }

    // 4. Direct & SIFS peer RF chords weighted by perturbation_score
    const anchors = frame && frame.room && Array.isArray(frame.room.anchors) ? frame.room.anchors : [];
    const activePulseIdx =
      ui.sifsPulseEnabled && anchors.length > 0
        ? Math.floor(ui.sifsPhase * anchors.length) % anchors.length
        : -1;
    const pulseBssid = activePulseIdx >= 0 ? anchors[activePulseIdx].bssid : null;

    for (let i = 0; i < links.length; i += 1) {
      const lnk = links[i];
      const isSelected = activeLink && lnk.link_id === activeLink.link_id;
      const score = Math.min(1, Math.max(0, (Number(lnk.perturbation_score) || 0) / 1.0));
      const isPulse =
        ui.sifsPulseEnabled &&
        pulseBssid &&
        (lnk.tx_bssid === pulseBssid || lnk.rx_id === pulseBssid);

      ctx.beginPath();
      ctx.moveTo(vp.toPxX(lnk.tx_pos[0]), vp.toPxY(lnk.tx_pos[1]));
      ctx.lineTo(vp.toPxX(lnk.rx_pos[0]), vp.toPxY(lnk.rx_pos[1]));
      if (lnk.is_peer) {
        ctx.setLineDash([5, 4]);
      } else {
        ctx.setLineDash([]);
      }

      if (isSelected) {
        ctx.strokeStyle = chart1;
        ctx.lineWidth = 2.4;
        ctx.globalAlpha = 0.96;
      } else if (score > 0.14) {
        const r = Math.round(c8Rgb[0] + (errRgb[0] - c8Rgb[0]) * score);
        const g = Math.round(c8Rgb[1] + (errRgb[1] - c8Rgb[1]) * score);
        const b = Math.round(c8Rgb[2] + (errRgb[2] - c8Rgb[2]) * score);
        ctx.strokeStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
        ctx.lineWidth = 1.0 + 1.9 * score;
        ctx.globalAlpha = 0.28 + 0.65 * score;
      } else {
        ctx.strokeStyle = 'rgb(' + onVarRgb[0] + ',' + onVarRgb[1] + ',' + onVarRgb[2] + ')';
        ctx.lineWidth = isPulse ? 1.3 : 0.75;
        ctx.globalAlpha = isPulse ? 0.38 : 0.15;
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // 5. Optional Ground Truth markers (sim mode)
    const gts = frame && Array.isArray(frame.ground_truth) ? frame.ground_truth : [];
    if (ui.showGroundTruth && gts.length > 0) {
      for (let i = 0; i < gts.length; i += 1) {
        const gt = gts[i];
        const gx = vp.toPxX(gt.x);
        const gy = vp.toPxY(gt.y);
        const rad = 0.36 * vp.scale;
        ctx.beginPath();
        ctx.arc(gx, gy, Math.max(9, rad), 0, Math.PI * 2);
        ctx.strokeStyle = chart3;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Crosshair
        ctx.beginPath();
        ctx.moveTo(gx - 6, gy);
        ctx.lineTo(gx + 6, gy);
        ctx.moveTo(gx, gy - 6);
        ctx.lineTo(gx, gy + 6);
        ctx.lineWidth = 1.8;
        ctx.strokeStyle = chart3;
        ctx.stroke();
      }
    }

    // 6. Tracked humans: trails, velocity vectors, circles, and badges
    const tracks = frame && Array.isArray(frame.tracks) ? frame.tracks : [];
    for (let i = 0; i < tracks.length; i += 1) {
      const trk = tracks[i];
      const tx = vp.toPxX(trk.x);
      const ty = vp.toPxY(trk.y);

      if (Array.isArray(trk.trail) && trk.trail.length > 1) {
        ctx.beginPath();
        for (let j = 0; j < trk.trail.length; j += 1) {
          const px = vp.toPxX(trk.trail[j][0]);
          const py = vp.toPxY(trk.trail[j][1]);
          if (j === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.strokeStyle = 'rgba(' + c1Rgb[0] + ',' + c1Rgb[1] + ',' + c1Rgb[2] + ',0.68)';
        ctx.lineWidth = 2.0;
        ctx.stroke();
      }

      // Velocity vector arrow
      if ((Number(trk.speed_mps) || 0) >= 0.08) {
        const vxPx = tx + trk.vx * 0.65 * vp.scale;
        const vyPx = ty - trk.vy * 0.65 * vp.scale;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(vxPx, vyPx);
        ctx.strokeStyle = chart8;
        ctx.lineWidth = 2.2;
        ctx.stroke();
      }

      // Target body disk
      const rPx = Math.max(10, 0.38 * vp.scale);
      ctx.beginPath();
      ctx.arc(tx, ty, rPx, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + c3Rgb[0] + ',' + c3Rgb[1] + ',' + c3Rgb[2] + ',0.22)';
      ctx.fill();
      ctx.lineWidth = 2.2;
      ctx.strokeStyle = chart3;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(tx, ty, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = onSurf;
      ctx.fill();

      // Callout badge
      const badgeText =
        trk.breathing_bpm != null
          ? 'T' + trk.track_id + ' · ' + Math.round(trk.breathing_bpm) + ' BPM'
          : 'T' + trk.track_id + ' · ' + (Number(trk.speed_mps) || 0).toFixed(2) + ' m/s';
      ctx.font = "600 11px 'Google Sans Code', monospace";
      const tw = ctx.measureText(badgeText).width;
      const bx = Math.min(vp.ox + vp.drawW - tw - 12, Math.max(vp.ox + 4, tx + 10));
      const by = Math.max(vp.oy + 18, ty - 12);
      ctx.fillStyle = surfHex;
      ctx.globalAlpha = 0.88;
      ctx.beginPath();
      ctx.roundRect(bx - 4, by - 12, tw + 8, 16, 5);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = chart3;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = onSurf;
      ctx.textAlign = 'left';
      ctx.fillText(badgeText, bx, by);
    }

    ctx.restore();

    // Room frame
    ctx.beginPath();
    ctx.roundRect(vp.ox, vp.oy, vp.drawW, vp.drawH, 10);
    ctx.strokeStyle = strokeDef;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // 7. Ambient Wi-Fi AP/IoT nodes
    for (let i = 0; i < anchors.length; i += 1) {
      const anc = anchors[i];
      const nx = vp.toPxX(anc.x);
      const ny = vp.toPxY(anc.y);
      const isEndpoint =
        activeLink && (activeLink.tx_bssid === anc.bssid || activeLink.rx_id === anc.bssid);
      const isPulse = ui.sifsPulseEnabled && i === activePulseIdx;

      if (isPulse) {
        const subPulse = (ui.sifsPhase * anchors.length) % 1;
        ctx.beginPath();
        ctx.arc(nx, ny, 5 + subPulse * 11, 0, Math.PI * 2);
        ctx.strokeStyle =
          'rgba(' + c1Rgb[0] + ',' + c1Rgb[1] + ',' + c1Rgb[2] + ',' + (0.72 * (1 - subPulse)).toFixed(2) + ')';
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(nx, ny, isEndpoint ? 5.5 : 4.2, 0, Math.PI * 2);
      ctx.fillStyle = isEndpoint ? chart1 : surfHex;
      ctx.fill();
      ctx.lineWidth = isEndpoint ? 2 : 1.5;
      ctx.strokeStyle = isEndpoint ? onSurf : onVar;
      ctx.stroke();
    }

    // 8. HackRF One RX marker (triangle)
    const rxPos = (frame && frame.room && frame.room.rx_pos) || [roomW * 0.5, 0.35];
    const rxX = vp.toPxX(rxPos[0]);
    const rxY = vp.toPxY(rxPos[1]);
    ctx.beginPath();
    ctx.moveTo(rxX, rxY - 7);
    ctx.lineTo(rxX - 6.5, rxY + 5);
    ctx.lineTo(rxX + 6.5, rxY + 5);
    ctx.closePath();
    ctx.fillStyle = chart3;
    ctx.fill();
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = onSurf;
    ctx.stroke();

    // Axis scale labels
    ctx.fillStyle = onVar;
    ctx.font = monoFont;
    ctx.textAlign = 'left';
    ctx.fillText('0m', vp.ox + 2, vp.oy + vp.drawH + 15);
    ctx.textAlign = 'center';
    ctx.fillText('HackRF RX (0 dBm)', rxX, Math.min(h - 4, vp.oy + vp.drawH + 15));
    ctx.textAlign = 'right';
    ctx.fillText(roomW.toFixed(1) + 'm × ' + roomH.toFixed(1) + 'm', vp.ox + vp.drawW - 2, vp.oy + vp.drawH + 15);
  }

  function renderCSICanvas() {
    const canvas = document.getElementById('csiCanvas');
    if (!canvas) return;
    const { ctx, w, h } = sizeCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const onVar = getCssVar('--on-surface-variant', 'rgba(0,0,0,0.62)');
    const strokeDef = getCssVar('--stroke-default', '#e3e3e3');
    const chart1 = getCssVar('--chart-1', '#2575fc');
    const chart2 = getCssVar('--chart-2', '#5b6e9e');
    const chart10 = getCssVar('--chart-10', '#c84288');

    const padL = 36;
    const padR = 10;
    const padT = 10;
    const padB = 20;
    const pw = Math.max(40, w - padL - padR);
    const ph = Math.max(30, h - padT - padB);

    const frame = ui.frame;
    const activeLink = getActiveLink(frame);
    const ampRaw =
      (activeLink && Array.isArray(activeLink.csi_amp) && activeLink.csi_amp.length === 52
        ? activeLink.csi_amp
        : frame && frame.csi && Array.isArray(frame.csi.amp) && frame.csi.amp.length === 52
          ? frame.csi.amp
          : null);
    const phaseRaw =
      (activeLink && Array.isArray(activeLink.csi_phase) && activeLink.csi_phase.length === 52
        ? activeLink.csi_phase
        : frame && frame.csi && Array.isArray(frame.csi.phase) && frame.csi.phase.length === 52
          ? frame.csi.phase
          : null);
    const baseRaw =
      activeLink && Array.isArray(activeLink.baseline_csi_amp) && activeLink.baseline_csi_amp.length === 52
        ? activeLink.baseline_csi_amp
        : null;

    const mapX = (k) => padL + ((k + 26) / 52) * pw;
    const yMin = -24;
    const yMax = 6;
    const mapY = (db) => padT + ((yMax - Math.max(yMin, Math.min(yMax, db))) / (yMax - yMin)) * ph;

    // Gridlines
    ctx.strokeStyle = strokeDef;
    ctx.lineWidth = 1;
    [-20, -10, 0].forEach((db) => {
      const yy = mapY(db);
      ctx.beginPath();
      ctx.moveTo(padL, yy);
      ctx.lineTo(padL + pw, yy);
      ctx.stroke();

      ctx.fillStyle = onVar;
      ctx.font = "10px 'Google Sans Code', monospace";
      ctx.textAlign = 'right';
      ctx.fillText(db + 'dB', padL - 4, yy + 3);
    });

    // DC null + pilot markers
    ctx.save();
    ctx.strokeStyle = strokeDef;
    ctx.setLineDash([2, 2]);
    [0].concat(PILOTS).forEach((kMark) => {
      const mx = mapX(kMark);
      ctx.beginPath();
      ctx.moveTo(mx, padT);
      ctx.lineTo(mx, padT + ph);
      ctx.stroke();
    });
    ctx.restore();

    if (ampRaw) {
      let refAmp = 0.001;
      for (let i = 0; i < ampRaw.length; i += 1) {
        if (ampRaw[i] > refAmp) refAmp = ampRaw[i];
      }
      if (baseRaw) {
        for (let i = 0; i < baseRaw.length; i += 1) {
          if (baseRaw[i] > refAmp) refAmp = baseRaw[i];
        }
      }
      const toDb = (a) => 20 * Math.log10(Math.max(a, 1e-4) / refAmp);

      // Baseline dashed curve
      if (baseRaw) {
        ctx.beginPath();
        ctx.strokeStyle = chart2;
        ctx.lineWidth = 1.4;
        ctx.setLineDash([4, 3]);
        for (let i = 0; i < SUBCARRIERS.length; i += 1) {
          const k = SUBCARRIERS[i];
          const px = mapX(k);
          const py = mapY(toDb(baseRaw[i]));
          if (i === 0 || k === 1) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Sanitized phase overlay mapped to [-pi, +pi]
      if (phaseRaw) {
        ctx.beginPath();
        ctx.strokeStyle = chart10;
        ctx.globalAlpha = 0.58;
        ctx.lineWidth = 1.1;
        for (let i = 0; i < SUBCARRIERS.length; i += 1) {
          const k = SUBCARRIERS[i];
          const px = mapX(k);
          const normPh = (phaseRaw[i] + Math.PI) / (2 * Math.PI);
          const py = padT + (1 - Math.max(0, Math.min(1, normPh))) * ph;
          if (i === 0 || k === 1) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // Measured |H_k| split across DC null
      const c1Rgb = hexToRgb(chart1);
      ctx.strokeStyle = chart1;
      ctx.lineWidth = 2.0;
      [[0, 25], [26, 51]].forEach((span) => {
        ctx.beginPath();
        const pts = [];
        for (let i = span[0]; i <= span[1]; i += 1) {
          const px = mapX(SUBCARRIERS[i]);
          const py = mapY(toDb(ampRaw[i]));
          pts.push({ px, py });
          if (i === span[0]) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();

        if (pts.length > 1) {
          ctx.beginPath();
          ctx.moveTo(pts[0].px, padT + ph);
          for (let j = 0; j < pts.length; j += 1) ctx.lineTo(pts[j].px, pts[j].py);
          ctx.lineTo(pts[pts.length - 1].px, padT + ph);
          ctx.closePath();
          ctx.fillStyle = 'rgba(' + c1Rgb[0] + ',' + c1Rgb[1] + ',' + c1Rgb[2] + ',0.14)';
          ctx.fill();
        }
      });
    } else {
      ctx.fillStyle = onVar;
      ctx.font = "12px 'Google Sans Flex', sans-serif";
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for 802.11 L-LTF CSI stream…', padL + pw * 0.5, padT + ph * 0.55);
    }

    ctx.fillStyle = onVar;
    ctx.font = "10px 'Google Sans Code', monospace";
    ctx.textAlign = 'center';
    ctx.fillText('-26', mapX(-26) + 6, h - 4);
    ctx.fillText('DC', mapX(0), h - 4);
    ctx.fillText('+26', mapX(26) - 6, h - 4);
  }

  function renderConvergenceOrPsdCanvas() {
    const canvas = document.getElementById('convCanvas');
    if (!canvas) return;
    const { ctx, w, h } = sizeCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const onVar = getCssVar('--on-surface-variant', 'rgba(0,0,0,0.62)');
    const strokeDef = getCssVar('--stroke-default', '#e3e3e3');
    const chart1 = getCssVar('--chart-1', '#2575fc');
    const chart3 = getCssVar('--chart-3', '#228a49');

    const padL = 34;
    const padR = 10;
    const padT = 10;
    const padB = 18;
    const pw = Math.max(40, w - padL - padR);
    const ph = Math.max(24, h - padT - padB);

    ctx.strokeStyle = strokeDef;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT + ph);
    ctx.lineTo(padL + pw, padT + ph);
    ctx.stroke();

    const frame = ui.frame;
    if (!frame) return;

    if (ui.plotTab === 'psd') {
      const freqs = (frame.psd && frame.psd.freqs_hz) || [];
      const power = (frame.psd && frame.psd.power_db) || [];
      if (freqs.length < 2) return;

      const fMin = 0.1;
      const fMax = 25.0;
      const logMin = Math.log10(fMin);
      const logSpan = Math.log10(fMax) - logMin;
      const mapX = (f) => padL + ((Math.log10(Math.max(fMin, Math.min(fMax, f))) - logMin) / logSpan) * pw;
      const yMin = -45;
      const yMax = 2;
      const mapY = (db) => padT + ((yMax - Math.max(yMin, Math.min(yMax, db))) / (yMax - yMin)) * ph;

      // Highlight 0.18-0.45 Hz respiratory band (10.8-27.0 BPM)
      const rx0 = mapX(0.18);
      const rx1 = mapX(0.45);
      const c3Rgb = hexToRgb(chart3);
      ctx.fillStyle = 'rgba(' + c3Rgb[0] + ',' + c3Rgb[1] + ',' + c3Rgb[2] + ',0.14)';
      ctx.fillRect(rx0, padT, Math.max(2, rx1 - rx0), ph);

      ctx.beginPath();
      ctx.strokeStyle = chart1;
      ctx.lineWidth = 1.8;
      for (let i = 0; i < freqs.length; i += 1) {
        const px = mapX(freqs[i]);
        const py = mapY(power[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      if (frame.psd.peak_resp_hz != null) {
        const pkX = mapX(frame.psd.peak_resp_hz);
        ctx.save();
        ctx.strokeStyle = chart3;
        ctx.lineWidth = 1.4;
        ctx.setLineDash([3, 2]);
        ctx.beginPath();
        ctx.moveTo(pkX, padT);
        ctx.lineTo(pkX, padT + ph);
        ctx.stroke();
        ctx.restore();
      }

      ctx.fillStyle = onVar;
      ctx.font = "10px 'Google Sans Code', monospace";
      ctx.textAlign = 'left';
      ctx.fillText('0.1 Hz', padL, h - 3);
      ctx.textAlign = 'center';
      ctx.fillText('Resp 0.18–0.45 Hz', padL + pw * 0.48, h - 3);
      ctx.textAlign = 'right';
      ctx.fillText('25 Hz', padL + pw, h - 3);
      return;
    }

    // Solver objective convergence curve J(x_k)
    const curve = (frame.solver && frame.solver.convergence) || [];
    if (!curve.length) return;

    let maxJ = 0.05;
    let minJ = Infinity;
    for (let i = 0; i < curve.length; i += 1) {
      if (curve[i] > maxJ) maxJ = curve[i];
      if (curve[i] < minJ) minJ = curve[i];
    }
    if (!Number.isFinite(minJ)) minJ = 0;
    const span = Math.max(0.05, maxJ - minJ);
    const n = curve.length;
    const mapX = (idx) => (n <= 1 ? padL + pw * 0.5 : padL + (idx / (n - 1)) * pw);
    const mapY = (val) => padT + (1 - (val - minJ) / (span * 1.08)) * ph;

    ctx.beginPath();
    ctx.strokeStyle = chart1;
    ctx.lineWidth = 2.0;
    for (let i = 0; i < n; i += 1) {
      const px = mapX(i);
      const py = mapY(curve[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    const endX = mapX(n - 1);
    const endY = mapY(curve[n - 1]);
    ctx.beginPath();
    ctx.arc(endX, endY, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = chart1;
    ctx.fill();

    ctx.fillStyle = onVar;
    ctx.font = "10px 'Google Sans Code', monospace";
    ctx.textAlign = 'right';
    ctx.fillText(maxJ.toFixed(2), padL - 4, padT + 7);
    ctx.fillText(minJ.toFixed(2), padL - 4, padT + ph);
    ctx.textAlign = 'left';
    ctx.fillText('k=1', padL, h - 3);
    ctx.textAlign = 'right';
    ctx.fillText('k=' + n, padL + pw, h - 3);
  }

  function renderReadouts() {
    const frame = ui.frame;
    if (!frame) return;

    const room = frame.room || {};
    const hud = frame.hud || {};
    const solver = frame.solver || {};
    const tracks = Array.isArray(frame.tracks) ? frame.tracks : [];
    const gts = Array.isArray(frame.ground_truth) ? frame.ground_truth : [];

    // Header subtitle
    const sub = document.getElementById('subtitle-text');
    if (sub) {
      const nx = room.grid_nx || 40;
      const ny = room.grid_ny || 34;
      sub.textContent =
        (hud.backend_label || 'passive-rx') +
        ' · ' +
        (room.width || 6).toFixed(1) +
        'm × ' +
        (room.height || 5).toFixed(1) +
        'm · ' +
        nx +
        '×' +
        ny +
        ' voxels · FFZ Δ < λ/2';
    }

    // Room target readout
    const targetRd = document.getElementById('target-readout');
    if (targetRd) {
      if (tracks.length === 0) {
        targetRd.textContent = 'Empty room (0 tracks)';
      } else {
        const parts = tracks.slice(0, 3).map((t) => {
          const tail = t.breathing_bpm != null ? ' ' + Math.round(t.breathing_bpm) + ' BPM' : '';
          return 'T' + t.track_id + ' (' + t.x.toFixed(2) + 'm, ' + t.y.toFixed(2) + 'm)' + tail;
        });
        targetRd.textContent = parts.join(' · ');
      }
    }

    // Accessible tracked humans strip built via programmatic DOM nodes
    const strip = document.getElementById('tracks-strip');
    if (strip) {
      const nodes = [];
      if (tracks.length === 0) {
        const emptyChip = document.createElement('span');
        emptyChip.className = 'track-chip track-chip--empty lumi-body-s';
        emptyChip.textContent = 'No active human tracks detected';
        nodes.push(emptyChip);
      } else {
        for (let i = 0; i < tracks.length; i += 1) {
          const t = tracks[i];
          const chip = document.createElement('span');
          chip.className = 'track-chip lumi-body-s';
          const metric =
            t.breathing_bpm != null
              ? Math.round(t.breathing_bpm) + ' BPM'
              : t.speed_mps.toFixed(2) + ' m/s';
          chip.textContent =
            'T' +
            t.track_id +
            ' [' +
            t.activity +
            '] · (' +
            t.x.toFixed(2) +
            'm, ' +
            t.y.toFixed(2) +
            'm) · ' +
            metric;
          nodes.push(chip);
        }
      }
      strip.replaceChildren(...nodes);
    }

    // Show ground-truth switch only when ground truth is provided by backend (sim mode)
    const gtWrap = document.getElementById('gt-switch-wrap');
    if (gtWrap) {
      gtWrap.classList.toggle('is-hidden', gts.length === 0);
    }

    // Active chord readout
    const activeLink = getActiveLink(frame);
    const chordRd = document.getElementById('chord-readout');
    if (chordRd) {
      if (activeLink) {
        const pinTag = ui.pinnedLinkId === activeLink.link_id ? ' [Pinned]' : '';
        const peerTag = activeLink.is_peer ? ' SIFS' : '';
        chordRd.textContent =
          activeLink.tx_label +
          '→' +
          activeLink.rx_label +
          peerTag +
          ' · ΔRSSI ' +
          Number(activeLink.rssi_delta_db || 0).toFixed(1) +
          ' dB' +
          pinTag;
      } else {
        chordRd.textContent = 'No active RF chords yet';
      }
    }

    // Bottom-right heading & cost / vital readout
    const solverTitle = document.getElementById('solver-title');
    const objRd = document.getElementById('obj-readout');
    const methodNames = {
      fista_tv: 'FISTA-TV',
      nnls: 'Regularized NNLS',
      sart: 'SART',
    };
    const mName = methodNames[solver.method] || String(solver.method || 'FISTA-TV');
    if (ui.plotTab === 'psd') {
      if (solverTitle) solverTitle.textContent = 'Vital-Sign PSD (0.1–25 Hz)';
      if (objRd) {
        objRd.textContent =
          frame.psd && frame.psd.peak_resp_bpm != null
            ? 'Peak ' + Number(frame.psd.peak_resp_bpm).toFixed(1) + ' BPM'
            : 'Vital Peak: none';
      }
    } else {
      const nIters = Array.isArray(solver.convergence) ? solver.convergence.length : 0;
      if (solverTitle) solverTitle.textContent = mName + ' Convergence (' + nIters + ' iters)';
      if (objRd) {
        const lastJ = nIters > 0 ? Number(solver.convergence[nIters - 1]) : null;
        objRd.textContent = lastJ != null ? 'Cost J(x) = ' + lastJ.toFixed(2) : 'Cost J(x) = —';
      }
    }

    // Telemetry grid
    const setTxt = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };
    setTxt('tele-k', String(hud.bssid_count != null ? hud.bssid_count : 0));
    const peerPart = hud.peer_chord_count > 0 ? ' (' + hud.peer_chord_count + 'p)' : '';
    setTxt('tele-m', String(hud.chord_count != null ? hud.chord_count : 0) + peerPart);
    setTxt('tele-n', String((room.grid_nx || 40) * (room.grid_ny || 34)));
    setTxt('tele-rate', Math.round(Number(hud.packet_rate_hz) || 0) + ' pkt/s');
    setTxt('tele-tracks', String(tracks.length));
    setTxt(
      'tele-vital',
      hud.peak_resp_bpm != null ? Number(hud.peak_resp_bpm).toFixed(1) + ' BPM' : 'none'
    );

    const adcEl = document.getElementById('tele-adc');
    if (adcEl) {
      const clipPct = (Number(hud.adc_clip_max) || 0) * 100;
      const isWarn = (Number(hud.adc_clip_max) || 0) >= 0.02;
      adcEl.textContent = (isWarn ? 'WARN ' : 'OK ') + clipPct.toFixed(1) + '%';
      adcEl.classList.toggle('tele-val--warn', isWarn);
    }
    setTxt(
      'tele-fcs',
      Number(hud.fcs_drop_pct || 0).toFixed(1) + '% · ' + String(hud.seq_gaps || 0)
    );

    // Sync solver controls from authoritative server frame when user isn't dragging a slider
    if (!ui.isEditingSlider) {
      const lamSlider = document.getElementById('slider-lambda');
      const lamVal = document.getElementById('slider-lambda-val');
      if (lamSlider && solver.lam_tv != null) {
        const clamped = Math.max(0.001, Math.min(0.15, Number(solver.lam_tv)));
        lamSlider.value = clamped.toFixed(3);
        updateSliderProgress(lamSlider);
        if (lamVal) lamVal.textContent = clamped.toFixed(3);
      }

      const winSlider = document.getElementById('slider-window');
      const winVal = document.getElementById('slider-window-val');
      if (winSlider && solver.window_s != null) {
        const clampedW = Math.max(0.4, Math.min(4.0, Number(solver.window_s)));
        winSlider.value = clampedW.toFixed(1);
        updateSliderProgress(winSlider);
        if (winVal) winVal.textContent = clampedW.toFixed(1) + ' s';
      }
    }

    document.querySelectorAll('[data-solver]').forEach((btn) => {
      const active = btn.dataset.solver === solver.method;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
      btn.setAttribute('tabindex', active ? '0' : '-1');
    });
  }

  function renderAllCanvases() {
    renderRoomCanvas();
    renderCSICanvas();
    renderConvergenceOrPsdCanvas();
  }

  function handleIncomingFrame(frameObj) {
    if (!frameObj || typeof frameObj !== 'object') return;
    if (frameObj.status === 'waiting') return;
    if (ui.paused) return;
    ui.frame = frameObj;
    const fps = Number(frameObj.server_fps) || 0;
    setConnectionState(
      'live',
      fps > 0 ? 'Live · ' + fps.toFixed(1) + ' FPS' : 'Live · #' + (frameObj.seq || 0)
    );
    renderReadouts();
    renderAllCanvases();
  }

  function connectStream() {
    if (ui.eventSource) {
      ui.eventSource.close();
      ui.eventSource = null;
    }
    if (ui.reconnectTimer) {
      window.clearTimeout(ui.reconnectTimer);
      ui.reconnectTimer = null;
    }
    setConnectionState('connecting', 'Connecting…');

    const es = new EventSource('/api/stream');
    ui.eventSource = es;

    es.addEventListener('open', () => {
      ui.reconnectDelayMs = 600;
      if (!ui.paused) {
        setConnectionState('live', 'Connected');
      }
    });

    es.addEventListener('config', (ev) => {
      try {
        applyConfig(JSON.parse(ev.data));
      } catch (_) {}
    });

    es.addEventListener('frame', (ev) => {
      try {
        handleIncomingFrame(JSON.parse(ev.data));
      } catch (_) {}
    });

    es.addEventListener('error', () => {
      if (es.readyState === EventSource.CLOSED) {
        setConnectionState('reconnecting', 'Reconnecting…');
        const delay = ui.reconnectDelayMs;
        ui.reconnectDelayMs = Math.min(5000, Math.round(ui.reconnectDelayMs * 1.5));
        ui.reconnectTimer = window.setTimeout(connectStream, delay);
      } else {
        setConnectionState('reconnecting', 'Stream interrupted…');
      }
    });
  }

  function findClosestLink(mx, my) {
    const links = ui.frame && Array.isArray(ui.frame.links) ? ui.frame.links : [];
    let bestId = null;
    let bestDist = 0.22;
    for (let i = 0; i < links.length; i += 1) {
      const l = links[i];
      const vx = l.rx_pos[0] - l.tx_pos[0];
      const vy = l.rx_pos[1] - l.tx_pos[1];
      const lenSq = vx * vx + vy * vy || 1e-6;
      const t = Math.max(0, Math.min(1, ((mx - l.tx_pos[0]) * vx + (my - l.tx_pos[1]) * vy) / lenSq));
      const px = l.tx_pos[0] + t * vx;
      const py = l.tx_pos[1] + t * vy;
      const d = Math.hypot(mx - px, my - py);
      if (d < bestDist) {
        bestDist = d;
        bestId = l.link_id;
      }
    }
    return bestId;
  }

  function findClosestDraggableTarget(mx, my) {
    if (!ui.config.sim_interactive || !ui.frame) return null;
    const pool =
      Array.isArray(ui.frame.ground_truth) && ui.frame.ground_truth.length > 0
        ? ui.frame.ground_truth
        : ui.frame.tracks || [];
    let bestId = null;
    let bestDist = 0.65;
    for (let i = 0; i < pool.length; i += 1) {
      const item = pool[i];
      const d = Math.hypot(mx - item.x, my - item.y);
      if (d < bestDist) {
        bestDist = d;
        bestId = item.id != null ? item.id : item.track_id;
      }
    }
    return bestId;
  }

  function bindControls() {
    const roomCanvas = document.getElementById('roomCanvas');
    if (roomCanvas) {
      const toRoom = (ev) => {
        const rect = roomCanvas.getBoundingClientRect();
        const roomW = (ui.frame && ui.frame.room && ui.frame.room.width) || ui.config.room_width || 6.0;
        const roomH = (ui.frame && ui.frame.room && ui.frame.room.height) || ui.config.room_height || 5.0;
        const vp = getRoomViewport(rect.width, rect.height, roomW, roomH);
        return {
          mx: vp.toRoomX(ev.clientX - rect.left),
          my: vp.toRoomY(ev.clientY - rect.top),
          roomW,
          roomH,
        };
      };

      roomCanvas.addEventListener('pointerdown', (ev) => {
        const { mx, my } = toRoom(ev);
        const dragId = findClosestDraggableTarget(mx, my);
        if (dragId != null) {
          ui.draggingTargetId = dragId;
          roomCanvas.setPointerCapture(ev.pointerId);
          roomCanvas.style.cursor = 'grabbing';
          sendControl({ sim_target: { id: dragId, x: mx, y: my } });
          return;
        }
        const hitId = findClosestLink(mx, my);
        if (hitId) {
          ui.pinnedLinkId = ui.pinnedLinkId === hitId ? null : hitId;
          renderReadouts();
          renderAllCanvases();
          announce(ui.pinnedLinkId ? 'Pinned chord ' + hitId : 'Cleared pinned chord.');
        }
      });

      roomCanvas.addEventListener('pointermove', (ev) => {
        const { mx, my, roomW, roomH } = toRoom(ev);
        if (ui.draggingTargetId != null) {
          const now = performance.now();
          if (now - ui.lastTargetSendMs >= 55) {
            ui.lastTargetSendMs = now;
            const cx = Math.max(0.3, Math.min(roomW - 0.3, mx));
            const cy = Math.max(0.3, Math.min(roomH - 0.3, my));
            sendControl({ sim_target: { id: ui.draggingTargetId, x: cx, y: cy } });
          }
          return;
        }
        const dragId = findClosestDraggableTarget(mx, my);
        if (dragId != null) {
          roomCanvas.style.cursor = 'grab';
          return;
        }
        const hitId = findClosestLink(mx, my);
        roomCanvas.style.cursor = hitId ? 'pointer' : 'default';
        if (hitId !== ui.hoveredLinkId) {
          ui.hoveredLinkId = hitId;
          renderReadouts();
          renderAllCanvases();
        }
      });

      const endDrag = (ev) => {
        if (ui.draggingTargetId != null) {
          ui.draggingTargetId = null;
          roomCanvas.style.cursor = 'default';
          try {
            roomCanvas.releasePointerCapture(ev.pointerId);
          } catch (_) {}
        }
      };
      roomCanvas.addEventListener('pointerup', endDrag);
      roomCanvas.addEventListener('pointercancel', endDrag);
      roomCanvas.addEventListener('pointerleave', () => {
        if (ui.hoveredLinkId !== null) {
          ui.hoveredLinkId = null;
          renderReadouts();
          renderAllCanvases();
        }
      });

      roomCanvas.addEventListener('keydown', (ev) => {
        const links = ui.frame && Array.isArray(ui.frame.links) ? ui.frame.links : [];
        if (!links.length) return;
        if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') {
          ev.preventDefault();
          ui.kbLinkIndex = (ui.kbLinkIndex + 1) % links.length;
          ui.hoveredLinkId = links[ui.kbLinkIndex].link_id;
          renderReadouts();
          renderAllCanvases();
          announce('Inspecting chord ' + ui.hoveredLinkId);
        } else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') {
          ev.preventDefault();
          ui.kbLinkIndex = (ui.kbLinkIndex - 1 + links.length) % links.length;
          ui.hoveredLinkId = links[ui.kbLinkIndex].link_id;
          renderReadouts();
          renderAllCanvases();
          announce('Inspecting chord ' + ui.hoveredLinkId);
        } else if (ev.key === 'Enter' || ev.key === ' ') {
          if (ui.hoveredLinkId) {
            ev.preventDefault();
            ui.pinnedLinkId = ui.pinnedLinkId === ui.hoveredLinkId ? null : ui.hoveredLinkId;
            renderReadouts();
            renderAllCanvases();
            announce(ui.pinnedLinkId ? 'Pinned chord ' + ui.pinnedLinkId : 'Unpinned chord.');
          }
        } else if (ev.key === 'Escape') {
          ui.pinnedLinkId = null;
          ui.hoveredLinkId = null;
          ui.kbLinkIndex = -1;
          renderReadouts();
          renderAllCanvases();
          announce('Cleared chord selection.');
        }
      });
    }

    // Sliders
    const lamSlider = document.getElementById('slider-lambda');
    const lamVal = document.getElementById('slider-lambda-val');
    if (lamSlider) {
      updateSliderProgress(lamSlider);
      lamSlider.addEventListener('pointerdown', () => {
        ui.isEditingSlider = true;
      });
      lamSlider.addEventListener('pointerup', () => {
        ui.isEditingSlider = false;
      });
      lamSlider.addEventListener('change', () => {
        ui.isEditingSlider = false;
      });
      lamSlider.addEventListener('input', () => {
        const val = parseFloat(lamSlider.value);
        updateSliderProgress(lamSlider);
        if (lamVal) lamVal.textContent = val.toFixed(3);
        scheduleControl('lam_tv', { lam_tv: val }, 80);
      });
    }

    const winSlider = document.getElementById('slider-window');
    const winVal = document.getElementById('slider-window-val');
    if (winSlider) {
      updateSliderProgress(winSlider);
      winSlider.addEventListener('pointerdown', () => {
        ui.isEditingSlider = true;
      });
      winSlider.addEventListener('pointerup', () => {
        ui.isEditingSlider = false;
      });
      winSlider.addEventListener('change', () => {
        ui.isEditingSlider = false;
      });
      winSlider.addEventListener('input', () => {
        const val = parseFloat(winSlider.value);
        updateSliderProgress(winSlider);
        if (winVal) winVal.textContent = val.toFixed(1) + ' s';
        scheduleControl('window_s', { window_s: val }, 80);
      });
    }

    // Solver segmented tabs
    const solverBtns = Array.from(document.querySelectorAll('[data-solver]'));
    solverBtns.forEach((btn, idx) => {
      btn.addEventListener('click', () => {
        const method = btn.dataset.solver;
        solverBtns.forEach((b) => {
          const isAct = b === btn;
          b.classList.toggle('is-active', isAct);
          b.setAttribute('aria-selected', isAct ? 'true' : 'false');
          b.setAttribute('tabindex', isAct ? '0' : '-1');
        });
        sendControl({ solver_method: method }, 'Switched solver to ' + btn.textContent + '.');
      });
      btn.addEventListener('keydown', (ev) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(ev.key)) return;
        ev.preventDefault();
        let nextIdx = idx;
        if (ev.key === 'ArrowRight') nextIdx = (idx + 1) % solverBtns.length;
        else if (ev.key === 'ArrowLeft') nextIdx = (idx - 1 + solverBtns.length) % solverBtns.length;
        else if (ev.key === 'Home') nextIdx = 0;
        else if (ev.key === 'End') nextIdx = solverBtns.length - 1;
        solverBtns[nextIdx].focus();
        solverBtns[nextIdx].click();
      });
    });

    // Bottom-right Convergence / Vital PSD mini-tabs
    const plotBtns = Array.from(document.querySelectorAll('[data-plot]'));
    const syncPlotTabs = () => {
      plotBtns.forEach((b) => {
        const isAct = b.dataset.plot === ui.plotTab;
        b.classList.toggle('is-active', isAct);
        b.setAttribute('aria-selected', isAct ? 'true' : 'false');
        b.setAttribute('tabindex', isAct ? '0' : '-1');
      });
    };
    syncPlotTabs();
    plotBtns.forEach((btn, idx) => {
      btn.addEventListener('click', () => {
        ui.plotTab = btn.dataset.plot === 'psd' ? 'psd' : 'convergence';
        syncPlotTabs();
        savePrefs();
        renderReadouts();
        renderConvergenceOrPsdCanvas();
      });
      btn.addEventListener('keydown', (ev) => {
        if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
        ev.preventDefault();
        const nextBtn = plotBtns[(idx + 1) % plotBtns.length];
        nextBtn.focus();
        nextBtn.click();
      });
    });

    // SIFS Pulse switch
    const swSifs = document.getElementById('switch-sifs');
    if (swSifs) {
      swSifs.checked = ui.sifsPulseEnabled;
      swSifs.addEventListener('change', () => {
        ui.sifsPulseEnabled = swSifs.checked;
        savePrefs();
        renderRoomCanvas();
      });
    }

    // Ground truth switch
    const swGt = document.getElementById('switch-gt');
    if (swGt) {
      swGt.checked = ui.showGroundTruth;
      swGt.addEventListener('change', () => {
        ui.showGroundTruth = swGt.checked;
        savePrefs();
        renderRoomCanvas();
      });
    }

    // Action buttons
    const btnCal = document.getElementById('btn-calibrate');
    if (btnCal) {
      btnCal.addEventListener('click', () => {
        sendControl({ action: 'calibrate_baseline' }, 'Empty-room baseline calibrated.');
      });
    }

    const btnReset = document.getElementById('btn-reset-target');
    if (btnReset) {
      btnReset.addEventListener('click', () => {
        sendControl({ action: 'reset_sim_targets' }, 'Simulated targets reset to scenario path.');
      });
    }

    const btnPause = document.getElementById('btn-pause');
    if (btnPause) {
      btnPause.addEventListener('click', () => {
        ui.paused = !ui.paused;
        btnPause.textContent = ui.paused ? 'Resume' : 'Pause';
        if (ui.paused) {
          setConnectionState('paused', 'Paused');
          announce('Tomography display paused.');
        } else {
          setConnectionState('live', 'Live');
          announce('Tomography display resumed.');
        }
      });
    }

    const btnTheme = document.getElementById('btn-theme');
    if (btnTheme) {
      btnTheme.addEventListener('click', toggleColorScheme);
    }
  }

  function tickAnimation(now) {
    const dt = Math.min(0.05, Math.max(0, (now - ui.lastAnimTs) / 1000));
    ui.lastAnimTs = now;
    if (ui.sifsPulseEnabled && !ui.paused && ui.frame) {
      ui.sifsPhase = (ui.sifsPhase + dt * 0.42) % 1;
      renderRoomCanvas();
    }
    window.requestAnimationFrame(tickAnimation);
  }

  async function init() {
    loadPrefs();
    applyColorScheme(getStoredScheme());
    bindControls();

    if (darkSchemeQuery && darkSchemeQuery.addEventListener) {
      darkSchemeQuery.addEventListener('change', () => {
        applyColorScheme(getStoredScheme());
      });
    }
    window.addEventListener('storage', (ev) => {
      if (ev.key === THEME_STORAGE_KEY) {
        applyColorScheme(getStoredScheme());
      }
    });

    const container = document.querySelector('.widget-container');
    if (container && typeof ResizeObserver === 'function') {
      new ResizeObserver(() => {
        renderAllCanvases();
      }).observe(container);
    }

    // Hydrate initial config & latest frame snapshot immediately
    try {
      const cfgResp = await fetch('/api/config');
      if (cfgResp.ok) applyConfig(await cfgResp.json());
    } catch (_) {}

    try {
      const frResp = await fetch('/api/frame');
      if (frResp.ok) {
        const data = await frResp.json();
        if (data && data.status !== 'waiting') handleIncomingFrame(data);
      }
    } catch (_) {}

    connectStream();
    window.requestAnimationFrame(tickAnimation);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
