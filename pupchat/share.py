"""The address to put Pupchat on a phone, and the QR code that carries it.

The owner's phone cannot reach http://127.0.0.1:8000, and the brief ruled out offering a
LAN address - a 192.168.x.x link that works or not depending on the Wi-Fi is exactly the
papercut to avoid. So the phone goes through the Cloudflare tunnel, and the only way to
know that hostname is to read it out of cloudflared's own log: a quick tunnel's name is
random and new on every run, with nothing to configure and nothing to remember.

The QR is generated with OpenCV, which is already a hard dependency of the pipeline - no
pip install, no vendored JavaScript, no CDN.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

TUNNEL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")

STARTED_AT = time.time()
"""When this process began. A log written before that belongs to an earlier run."""

TAIL_BYTES = 65536
"""cloudflared appends a line per request, so the banner is near the start but the file
grows. Reading the tail finds the most recent banner without loading the whole thing."""


def public_url(now: float | None = None) -> tuple[str | None, str]:
    """This server's address on the internet, or why there isn't one.

    The staleness check is the load-bearing half. Yesterday's logfile parses perfectly
    and yields a hostname that is dead, and a QR pointing at a dead hostname is worse
    than no QR at all: it fails on the phone, silently, with nothing on screen to explain
    why.
    """
    override = os.getenv("PUPCHAT_PUBLIC_URL", "").strip()
    if override:
        return override.rstrip("/"), "configured"

    raw = os.getenv("PUPCHAT_TUNNEL_LOG", "").strip()
    if not raw:
        return None, "no tunnel: start Share Pupchat.bat"

    path = Path(raw)
    if not path.exists():
        return None, "no tunnel yet: cloudflared has not written its log"
    try:
        if path.stat().st_mtime < STARTED_AT - 5:
            return None, "that tunnel log is from an earlier run"
        with path.open("rb") as fh:
            fh.seek(max(0, path.stat().st_size - TAIL_BYTES))
            tail = fh.read().decode("utf-8", "replace")
    except OSError as e:
        return None, f"could not read the tunnel log ({e})"

    hits = TUNNEL_RE.findall(tail)
    # The last one: a reconnect logs the banner again, and the newest is the live one.
    return (hits[-1], "live") if hits else (None, "the tunnel is still connecting")


def qr_png(url: str, scale: int = 8, quiet: int = 4) -> bytes:
    """A scannable PNG of `url`.

    `INTER_NEAREST` is not a preference. Any smoothing interpolation blurs the module
    edges and the code stops scanning. The quiet zone is not decoration either: a QR
    with no white border around it is unreliable to read.
    """
    import cv2
    import numpy as np                              # noqa: F401 - cv2 needs it loaded

    encoder = getattr(cv2, "QRCodeEncoder_create", None)
    if encoder is None:
        raise RuntimeError("this OpenCV build has no QR encoder")
    matrix = encoder().encode(url)
    matrix = cv2.copyMakeBorder(matrix, quiet, quiet, quiet, quiet,
                                cv2.BORDER_CONSTANT, value=255)
    matrix = cv2.resize(matrix, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", matrix)
    if not ok:
        raise RuntimeError("could not encode the QR image")
    return buf.tobytes()
