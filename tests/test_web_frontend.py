"""Contract, security, accessibility, and asset-serving tests for the Luminous web frontend."""

from __future__ import annotations

import http.client
import re
import shutil
import subprocess
from urllib.parse import urlparse

import pytest

from hackrf_radar.web import RealtimeWebServer, TomographyStateHub
from hackrf_radar.web.server import ALLOWED_STATIC_ASSETS, STATIC_DIR


def _fetch(url: str) -> tuple[int, dict[str, str], bytes]:
    parsed = urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5.0)
    try:
        conn.request("GET", parsed.path or "/")
        resp = conn.getresponse()
        body = resp.read()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, hdrs, body
    finally:
        conn.close()


def test_frontend_assets_served_with_strict_csp_and_mime_types() -> None:
    hub = TomographyStateHub(room_width=6.0, room_height=5.0)
    with RealtimeWebServer(hub, host="127.0.0.1", port=0) as srv:
        base = srv.url.rstrip("/")

        st, hdrs, html_bytes = _fetch(f"{base}/")
        assert st == 200
        assert "text/html" in hdrs["content-type"]
        assert "script-src 'self'" in hdrs["content-security-policy"]
        assert "unsafe-inline" not in hdrs["content-security-policy"]
        html = html_bytes.decode("utf-8")
        assert "<title>HackRF Passive Wi-Fi Tomographic Imaging</title>" in html
        assert 'id="roomCanvas"' in html
        assert 'id="csiCanvas"' in html
        assert 'id="convCanvas"' in html

        for asset_name, expected_ct in ALLOWED_STATIC_ASSETS.items():
            st_a, hdrs_a, body_a = _fetch(f"{base}/static/{asset_name}")
            assert st_a == 200, f"Expected 200 for /static/{asset_name}"
            assert hdrs_a["content-type"] == expected_ct
            assert hdrs_a["x-content-type-options"] == "nosniff"
            assert len(body_a) > 64


def test_html_accessibility_dark_mode_and_zero_inline_scripts() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    # Language, title, color-scheme meta, landmarks, single h1, skip link, live region
    assert '<html lang="en">' in html
    assert '<meta name="color-scheme" content="light dark">' in html
    assert html.count("<h1") == 1
    assert "<header" in html and "<main" in html
    assert 'class="skip-link visually-hidden"' in html
    assert 'aria-live="polite"' in html
    assert 'for="slider-lambda"' in html
    assert 'for="slider-window"' in html
    assert 'for="switch-sifs"' in html

    # Pause/Resume dynamic-label button must not set aria-pressed
    pause_match = re.search(r"<button[^>]*id=\"btn-pause\"[^>]*>", html)
    assert pause_match is not None
    assert "aria-pressed" not in pause_match.group(0)

    # Strict CSP compatibility: every <script> tag must have a src attribute and empty body
    for match in re.finditer(r"<script([^>]*)>(.*?)</script>", html, flags=re.DOTALL):
        attrs, body = match.group(1), match.group(2)
        assert "src=" in attrs, "Inline script detected in index.html"
        assert body.strip() == "", "Non-empty inline script body detected"

    # Zero inline event handlers (onclick=, onchange=, etc.)
    assert re.search(r"\son[a-z]+\s*=", html, flags=re.IGNORECASE) is None


def test_css_luminous_tokens_dark_mode_reduced_motion_and_forced_colors() -> None:
    lumi_css = (STATIC_DIR / "luminous.css").read_text(encoding="utf-8")
    app_css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")

    assert "color-scheme: light dark;" in lumi_css
    assert ':has(> head > meta[name="color-scheme"][content="dark"])' in lumi_css
    assert "@media (prefers-color-scheme: dark)" in lumi_css
    assert "@media (forced-colors: active)" in lumi_css
    assert "@supports not (scrollbar-color: auto)" in lumi_css
    assert "--lumi-sys-color--surface:" in lumi_css

    assert "@media (prefers-reduced-motion: no-preference)" in app_css
    assert "clip-path: inset(50%)" in app_css


def test_js_no_dangerous_dom_sinks_and_valid_syntax() -> None:
    forbidden_sinks = (
        "innerHTML",
        "outerHTML",
        "document.write",
        "eval(",
        "new Function",
    )
    js_files = [STATIC_DIR / "app.js", STATIC_DIR / "theme-init.js"]
    for js_path in js_files:
        text = js_path.read_text(encoding="utf-8")
        for sink in forbidden_sinks:
            assert sink not in text, f"Forbidden DOM sink {sink!r} found in {js_path}"

    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    for endpoint in ("/api/stream", "/api/control", "/api/config", "/api/frame"):
        assert endpoint in app_js
    for solver_name in ("fista_tv", "nnls", "sart"):
        assert solver_name in app_js

    node_bin = shutil.which("node")
    if node_bin is not None:
        for js_path in js_files:
            proc = subprocess.run(
                [node_bin, "--check", str(js_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0, (
                f"node --check failed for {js_path}: {proc.stderr}"
            )
    else:
        pytest.skip("node not available for JS syntax check")
