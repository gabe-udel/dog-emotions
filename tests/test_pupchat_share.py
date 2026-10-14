"""The phone link, and the two ways it could be worse than useless.

A QR pointing at a dead hostname fails on the phone silently, with nothing on screen to
explain why - worse than no QR at all. And the code is a credential, so anything that
hands it out more widely than the owner's own screen is a takeover, not a convenience.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pupchat import share
from pupchat.share import TUNNEL_RE, public_url, qr_png

BANNER = """2026-10-14T10:00:00Z INF Thank you for trying Cloudflare Tunnel.
2026-10-14T10:00:01Z INF |  https://brave-dog-walks-fast.trycloudflare.com  |
2026-10-14T10:00:02Z INF Registered tunnel connection
"""


@pytest.fixture
def tunnel_log(tmp_path, monkeypatch):
    path = tmp_path / "tunnel.log"
    monkeypatch.setenv("PUPCHAT_TUNNEL_LOG", str(path))
    monkeypatch.delenv("PUPCHAT_PUBLIC_URL", raising=False)
    monkeypatch.setattr(share, "STARTED_AT", time.time() - 1)
    return path


# ------------------------------------------------------------------ the address

def test_the_address_is_read_from_cloudflareds_own_log(tunnel_log):
    """A quick tunnel's hostname is random and new on every run. There is nothing to
    configure and nothing to remember, so it has to be read."""
    tunnel_log.write_text(BANNER, encoding="utf-8")
    url, reason = public_url()
    assert url == "https://brave-dog-walks-fast.trycloudflare.com"
    assert reason == "live"


def test_a_reconnect_wins_over_the_first_banner(tunnel_log):
    """cloudflared logs the banner again when it reconnects, and the newest is live."""
    tunnel_log.write_text(
        BANNER + "\nINF reconnecting\nINF | https://second-name-here.trycloudflare.com |",
        encoding="utf-8")
    assert public_url()[0] == "https://second-name-here.trycloudflare.com"


def test_yesterdays_log_is_refused(tunnel_log):
    """THE load-bearing check. An old log parses perfectly and yields a hostname that is
    dead; a QR pointing at it fails on the phone with nothing to explain why."""
    tunnel_log.write_text(BANNER, encoding="utf-8")
    old = time.time() - 86400
    import os
    os.utime(tunnel_log, (old, old))
    url, reason = public_url()
    assert url is None
    assert "earlier run" in reason


def test_a_log_with_no_banner_yet_says_it_is_connecting(tunnel_log):
    tunnel_log.write_text("INF starting\n", encoding="utf-8")
    url, reason = public_url()
    assert url is None and "connecting" in reason


def test_no_tunnel_at_all_says_how_to_start_one(monkeypatch):
    monkeypatch.delenv("PUPCHAT_TUNNEL_LOG", raising=False)
    monkeypatch.delenv("PUPCHAT_PUBLIC_URL", raising=False)
    url, reason = public_url()
    assert url is None
    assert "Share Pupchat.bat" in reason


def test_an_explicit_address_overrides_the_log(monkeypatch):
    """So a real deployment behind a real domain needs no cloudflared at all."""
    monkeypatch.setenv("PUPCHAT_PUBLIC_URL", "https://pupchat.example/")
    assert public_url() == ("https://pupchat.example", "configured")


def test_only_a_trycloudflare_host_is_accepted():
    assert TUNNEL_RE.findall("see https://a-b-c.trycloudflare.com now") == \
        ["https://a-b-c.trycloudflare.com"]
    assert TUNNEL_RE.findall("https://evil.example.com/trycloudflare.com") == []


def test_a_huge_log_is_not_read_whole(tunnel_log):
    """cloudflared appends a line per request, so this file grows all session."""
    tunnel_log.write_text(BANNER + ("INF request\n" * 200000), encoding="utf-8")
    assert tunnel_log.stat().st_size > share.TAIL_BYTES
    # The banner is now far outside the tail, which is the honest outcome: better to
    # say "still connecting" than to read 2 MB on every poll.
    assert public_url()[0] is None


# ------------------------------------------------------------------ the QR itself

def test_the_qr_is_a_real_png():
    png = qr_png("https://brave-dog-walks-fast.trycloudflare.com/?k=" + "x" * 43)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 200


def test_the_qr_reads_back_as_the_url_it_was_given():
    """The one test that proves the image is scannable rather than merely square."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    url = "https://brave-dog-walks-fast.trycloudflare.com/?k=" + "A1b2C3" * 7
    png = qr_png(url)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    decoded, *_ = cv2.QRCodeDetector().detectAndDecode(img)
    assert decoded == url


def test_the_qr_has_a_quiet_zone():
    """A code with no white border around it is unreliable to scan."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    png = qr_png("https://x.trycloudflare.com", scale=4, quiet=4)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img[0, 0] == 255 and img[-1, -1] == 255


# ------------------------------------------------------------------ who may see it

@pytest.fixture
def clients(tmp_path, monkeypatch):
    import pupchat.library as L
    from pupchat.api import build_app
    from pupchat.library import Library

    monkeypatch.setattr(L, "_LIBRARY", Library(tmp_path / "library.json"))
    monkeypatch.setattr(L, "OWNER_FILE", tmp_path / "owner.json")
    app = build_app()
    return (TestClient(app, client=("127.0.0.1", 50000)),
            TestClient(app, client=("203.0.113.7", 40000)))


def test_the_phone_link_is_local_only(clients, tunnel_log):
    """Owner-only would not be enough. The QR IS the owner credential, so the first
    visitor handed the link could otherwise re-serve it to anyone."""
    tunnel_log.write_text(BANNER, encoding="utf-8")
    local, remote = clients
    assert local.get("/api/share/url").status_code == 200
    assert remote.get("/api/share/url").status_code == 403
    assert remote.get("/api/share/qr.png").status_code == 403
    assert remote.post("/api/share/rotate").status_code == 403


def test_the_link_carries_the_token_and_the_qr_matches(clients, tunnel_log):
    tunnel_log.write_text(BANNER, encoding="utf-8")
    local, _ = clients
    body = local.get("/api/share/url").json()
    assert body["url"].startswith("https://brave-dog-walks-fast.trycloudflare.com/?k=")
    assert local.get("/api/share/qr.png").headers["content-type"] == "image/png"


def test_the_qr_is_never_cached(clients, tunnel_log):
    """The hostname changes with every tunnel and the token changes on rotate."""
    tunnel_log.write_text(BANNER, encoding="utf-8")
    local, _ = clients
    assert "no-store" in local.get("/api/share/qr.png").headers["cache-control"]


def test_rotating_invalidates_the_old_code(clients, tunnel_log):
    """A leaked code must be one click from dead."""
    tunnel_log.write_text(BANNER, encoding="utf-8")
    local, _ = clients
    before = local.get("/api/share/url").json()["url"]
    local.post("/api/share/rotate")
    assert local.get("/api/share/url").json()["url"] != before


def test_with_no_tunnel_there_is_no_qr_and_a_reason(clients, monkeypatch):
    monkeypatch.delenv("PUPCHAT_TUNNEL_LOG", raising=False)
    monkeypatch.delenv("PUPCHAT_PUBLIC_URL", raising=False)
    local, _ = clients
    body = local.get("/api/share/url").json()
    assert body["url"] is None and body["qr"] is None
    assert local.get("/api/share/qr.png").status_code == 503
