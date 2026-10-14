"""Put src/ on the path for the test session only.

Library modules under src/ deliberately do no sys.path manipulation of their own - this
is test infrastructure, and the one place it is allowed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

WEBAPP = ROOT / "pupchat" / "webapp"

TEXT_SUFFIXES = {".html", ".js", ".css", ".json", ".svg", ".txt", ".md"}

VENDORED = {"assets/tailwind.js"}
"""Third-party code copied in verbatim by tools/vendor_assets.py.

Excluded from the substring checks, because scanning a minified library for short
markers is all false positives and no signal: "sk-" matches `mask-border` 31 times, and
Tailwind prints its own documentation URLs in error messages, which are strings it never
fetches. What those checks are really about is what WE wrote.
"""


def webapp_files(vendored: bool = False) -> dict[str, str]:
    """Every text file the browser can be served, as {relative path: contents}.

    Globbed rather than named. The safety rules here - no API key in anything shipped,
    nothing fetched from Google, video streamed rather than embedded - are properties of
    *the front end*, not of a file called app.js. Pinning filenames meant that splitting
    one script into modules silently dropped four safety tests instead of failing them,
    which is the worst outcome available to a test whose job is to catch a leak.
    """
    out = {}
    for p in sorted(WEBAPP.rglob("*")):
        rel = p.relative_to(WEBAPP).as_posix()
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if rel in VENDORED and not vendored:
            continue
        out[rel] = p.read_text(encoding="utf-8", errors="replace")
    return out


def webapp_text(vendored: bool = False) -> str:
    """Everything shippable, concatenated - for 'this string must appear nowhere'."""
    return "\n".join(webapp_files(vendored).values())
