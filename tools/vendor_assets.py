"""Download everything the page loads from someone else's server, once.

Run by hand; the output is committed. The page must work with no internet, and it must
not tell Google who is looking at it:

    .venv\\Scripts\\python.exe tools\\vendor_assets.py

What it fetches and why each is allowed to be redistributed:

* **Plus Jakarta Sans** - SIL OFL 1.1, which permits bundling.
* **Material Symbols Outlined** - Apache 2.0. The full variable font is ~3.9 MB of woff2.
  That is fine over a laptop's loopback and rude over a tunnel to a phone on mobile data,
  so it is subset to the glyphs the design actually uses. The markup keeps working
  unchanged because the ligature features are retained: `<span>pets</span>` still renders
  a paw.
* **Tailwind Play CDN** - MIT. Vendored as-is rather than compiled, because a real
  compiled stylesheet needs the Tailwind CLI and therefore Node, which this project does
  not have. 400 KB of vendored JS is the smaller change.
* **The Stitch logo** - exported as part of the user's own design.

Deliberately NOT fetched: the seven sample dog photos. They are invented content and the
app replaces them with the owner's own photo or a monogram.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEBAPP = ROOT / "pupchat" / "webapp"
ASSETS = WEBAPP / "assets"
FONTS = ASSETS / "fonts"
STITCH = (ROOT / "outputs" / "pupchat" / "pupchat app"
          / "stitch_pupchat_ai_dog_behaviorist")

# Google serves woff2 only to a user agent it believes supports it. With urllib's default
# agent it returns ttf, which is roughly twice the size for no benefit.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

JAKARTA_CSS = ("https://fonts.googleapis.com/css2"
               "?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap")
SYMBOLS_CSS = ("https://fonts.googleapis.com/css2"
               "?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1")
TAILWIND_JS = "https://cdn.tailwindcss.com"

# Icon name -> private-use codepoint. Needed because subsetting by ligature text does not
# work on this font: every icon name is spelled out of the same lowercase alphabet, so
# keeping the letters keeps every ligature rule in the font reachable, and the "subset"
# comes out the same size as the original (measured: 1112 KB -> 995 KB). Subsetting by
# codepoint keeps the icons we name; the `--text` pass alongside it keeps the letters and
# the ligature rules that spell them.
CODEPOINTS = ("https://raw.githubusercontent.com/google/material-design-icons/master/"
              "variablefont/MaterialSymbolsOutlined%5BFILL%2CGRAD%2Copsz%2Cwght%5D"
              ".codepoints")

OFL = "https://raw.githubusercontent.com/google/fonts/main/ofl/plusjakartasans/OFL.txt"
APACHE = "https://raw.githubusercontent.com/google/material-design-icons/master/LICENSE"

# The glyphs the three Stitch screens use, plus the handful the rebuild adds (a mobile
# menu, the QR panel, delete and back). Subsetting by ligature TEXT, so this list is the
# literal set of words that must keep working.
ICONS = """
add add_a_photo add_comment arrow_back arrow_forward arrow_forward_ios arrow_right
arrow_upward badge bookmark_border cake chat_bubble_outline check check_circle
chevron_right cleaning_services close content_copy delete dock_to_right done_all
door_front download edit edit_square expand_more explore fence forum help_outline
history_edu icecream lightbulb local_shipping medical_services menu mic more_vert
navigation notifications notifications_active open_in_new pets play_arrow psychology
qr_code_2 radio_button_checked refresh scale school search settings share smart_display
smart_toy spa sports_kabaddi stethoscope thumb_up thunderstorm tune verified video_file
video_library videocam visibility vital_signs volume_off warning
""".split()


def fetch(url: str, *, binary: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return data if binary else data.decode("utf-8")


def localise(css: str, prefix: str) -> tuple[str, dict[str, str]]:
    """Rewrite every remote url() to a local file, and say what to download.

    Returns the rewritten CSS and {local filename: remote url}.
    """
    wanted: dict[str, str] = {}

    def swap(m: re.Match) -> str:
        url = m.group(1).strip("'\"")
        name = f"{prefix}-{len(wanted):02d}.woff2"
        for have, u in wanted.items():          # the same face can appear twice
            if u == url:
                name = have
                break
        wanted[name] = url
        return f"url('fonts/{name}')"

    return re.sub(r"url\(([^)]+)\)", swap, css), wanted


def subset_symbols(src: Path, dest: Path, codepoints: dict[str, int]) -> None:
    """Cut the icon font down to the glyphs this design uses.

    Two selections together, and both are needed:

    * `--unicodes` names the icon glyphs by their private-use codepoint. This is what
      makes the subset small. Selecting by ligature text instead does nothing useful,
      because every icon name is spelled from the same lowercase alphabet, so retaining
      the letters leaves every ligature in the font reachable.
    * `--text` keeps those letters and, with `--layout-features`, the ligature rules that
      turn them into the icons. Without this the markup would have to change to numeric
      character references, and the brief says to use the exported HTML as-is:
      `<span class="material-symbols-outlined">pets</span>` must keep rendering a paw.

    `--no-layout-closure` is what makes the two cooperate. By default fontTools EXPANDS
    the glyph set to include anything the retained layout rules can produce - and since
    every icon name is spelled from the retained letters, that is every icon in the font.
    Measured: 1112 KB in, 995 KB out, i.e. no subsetting at all. With closure off, the
    rules are instead pruned down to the glyphs actually asked for.
    """
    missing = [n for n in ICONS if n not in codepoints]
    if missing:
        print(f"   !! not in this font, skipping: {', '.join(missing)}")
    use = [n for n in ICONS if n in codepoints]
    unicodes = ",".join(f"U+{codepoints[n]:04X}" for n in use)

    words = dest.parent / "_icons.txt"
    words.write_text("\n".join(use), encoding="utf-8")
    cmd = [sys.executable, "-m", "fontTools.subset", str(src),
           f"--text-file={words}", f"--unicodes={unicodes}",
           # rclt is the one that matters: this font carries its ligatures under
           # 'rclt' and 'rlig', NOT 'liga'. Subsetting with the usual liga/calt list
           # silently produced a font with zero ligature rules - 18 KB of icons that
           # rendered as nothing, because the markup asks for them by name.
           "--layout-features+=rlig,rclt,liga,dlig,calt,ccmp",
           "--no-layout-closure",
           "--flavor=woff2", f"--output-file={dest}"]
    subprocess.run(cmd, check=True, capture_output=True)
    words.unlink()
    verify_icons(dest, use)


def ligature_words(path: Path) -> set[str]:
    """Every word the font still turns into an icon.

    Unwraps LookupType 7 (Extension), which is how this font stores them - reading
    `st.ligatures` without unwrapping finds nothing and makes a broken subset look fine.
    """
    from fontTools.ttLib import TTFont
    font = TTFont(path)
    cmap = font.getBestCmap()
    rev = {g: chr(cp) for cp, g in cmap.items()}
    out: set[str] = set()
    for lookup in font["GSUB"].table.LookupList.Lookup:
        for sub in lookup.SubTable:
            inner = sub.ExtSubTable if lookup.LookupType == 7 else sub
            for first, arr in (getattr(inner, "ligatures", None) or {}).items():
                for lig in arr:
                    out.add(rev.get(first, "?")
                            + "".join(rev.get(c, "?") for c in lig.Component))
    return out


def verify_icons(path: Path, names: list[str]) -> None:
    """Refuse to ship a subset whose icons do not render.

    A font missing its ligature rules is not a visible failure at build time - the file
    is written, it is a valid font, and it is even smaller. It only shows up as blank
    spaces where the icons should be.
    """
    have = ligature_words(path)
    missing = [n for n in names if n not in have]
    if missing:
        raise SystemExit(
            f"the icon subset lost {len(missing)} of {len(names)} icons, e.g. "
            f"{', '.join(missing[:6])}. The ligature features were dropped - check the "
            f"--layout-features list against the font's own feature tags.")


def icon_codepoints() -> dict[str, int]:
    """`name <hex>` per line, straight from Google's own font repository."""
    out: dict[str, int] = {}
    for line in fetch(CODEPOINTS).splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = int(parts[1], 16)
    return out


def main() -> None:
    FONTS.mkdir(parents=True, exist_ok=True)
    css_parts: list[str] = []

    # ---------------------------------------------------------------- body font
    print("Plus Jakarta Sans ...")
    css, wanted = localise(fetch(JAKARTA_CSS), "plus-jakarta-sans")
    for name, url in wanted.items():
        (FONTS / name).write_bytes(fetch(url, binary=True))
        print(f"   {name}  {(FONTS / name).stat().st_size / 1024:.0f} KB")
    css_parts.append(css)

    # ---------------------------------------------------------------- icon font
    print("Material Symbols Outlined ...")
    sym_css = fetch(SYMBOLS_CSS)
    url = re.search(r"url\(([^)]+)\)", sym_css).group(1).strip("'\"")
    full = FONTS / "_material-symbols-full.woff2"
    full.write_bytes(fetch(url, binary=True))
    out = FONTS / "material-symbols-outlined.woff2"
    subset_symbols(full, out, icon_codepoints())
    print(f"   {full.stat().st_size / 1024:.0f} KB  ->  "
          f"{out.stat().st_size / 1024:.0f} KB  ({len(ICONS)} icons)")
    full.unlink()

    sym_css = re.sub(r"url\([^)]+\)", "url('fonts/material-symbols-outlined.woff2')",
                     sym_css)
    css_parts.append(sym_css)

    # The class that makes the ligatures work. It comes only from Google's stylesheet
    # today, so vendoring the font without it renders the word, not the icon.
    css_parts.append("""
.material-symbols-outlined {
  font-family: 'Material Symbols Outlined';
  font-weight: normal;
  font-style: normal;
  line-height: 1;
  letter-spacing: normal;
  text-transform: none;
  display: inline-block;
  white-space: nowrap;
  word-wrap: normal;
  direction: ltr;
  -webkit-font-feature-settings: 'liga';
  -webkit-font-smoothing: antialiased;
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}
""")

    (ASSETS / "fonts.css").write_text(
        "/* Vendored by tools/vendor_assets.py. Do not edit by hand. */\n"
        + "\n".join(css_parts), encoding="utf-8")

    # ---------------------------------------------------------------- tailwind
    print("Tailwind Play CDN ...")
    js = fetch(TAILWIND_JS, binary=True)
    (ASSETS / "tailwind.js").write_bytes(js)
    print(f"   tailwind.js  {len(js) / 1024:.0f} KB")

    # ---------------------------------------------------------------- the logo
    print("Stitch logo ...")
    html = (STITCH / "pupchat_new_chat" / "code.html").read_text(encoding="utf-8")
    m = re.search(r"https://lh3\.googleusercontent\.com/aida/[A-Za-z0-9_-]+", html)
    if not m:
        print("   !! no logo url in the export; the page falls back to the paw glyph")
    else:
        png = fetch(m.group(0), binary=True)
        (ASSETS / "logo.png").write_bytes(png)
        print(f"   logo.png  {len(png) / 1024:.1f} KB")

    # ---------------------------------------------------------------- licences
    # Both fonts are redistributable only WITH their licence, so a failure here is a
    # licensing problem, not a cosmetic one. Say so loudly rather than carrying on quietly.
    print("licences ...")
    for name, url in (("LICENSE-plus-jakarta-sans.txt", OFL),
                      ("LICENSE-material-symbols.txt", APACHE)):
        try:
            (FONTS / name).write_text(fetch(url), encoding="utf-8")
            print(f"   {name}  {(FONTS / name).stat().st_size / 1024:.1f} KB")
        except Exception as e:
            print(f"   !! could not fetch {name} ({e}). Both fonts may only be "
                  f"redistributed WITH their licence - add it by hand before shipping.")

    print("\nDone. Everything the page needs is now in pupchat/webapp/assets/.")


if __name__ == "__main__":
    main()
