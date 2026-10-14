"""Merge the three Stitch screens into one page. Run once; index.html is the source after.

The export is three standalone HTML files that are 99.5% the same: each carries its own
copy of the sidebar and header (~14 KB of the 26-41 KB). Keeping them separate would mean
a full page load on every click - the sidebar blinking, the fonts re-parsing - and three
copies of any sidebar fix.

So: one shell, three screen bodies as siblings, each tagged `data-view`. The router shows
one at a time and keeps real URLs (/, /chat/<id>, /dogs/<name>). Every screen's body is
copied VERBATIM; only the remote URLs are rewritten, because the page must not call
Google.

This script is mechanical on purpose. It does the merge and the URL swap and nothing else
- the fake sample content is removed afterwards, by hand, against the committed file.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STITCH = (ROOT / "outputs" / "pupchat" / "pupchat app"
          / "stitch_pupchat_ai_dog_behaviorist")
OUT = ROOT / "pupchat" / "webapp" / "index.html"

# The exact string line 4 of every file ends with. Everything before it is the shell.
MAIN_OPEN = ('<main class="relative pt-16 bg-surface min-h-[calc(100vh-4rem)] w-full">'
             '<div class="flex flex-col w-full">')

SCREENS = [
    # folder, data-view, needs its own wrapper?
    ("pupchat_new_chat", "welcome", False),
    ("pupchat_chat_barking_at_guests", "chat", True),
    ("pupchat_dog_profile_milo", "dog", False),
]


def body_of(folder: str) -> str:
    """The screen between <main> and </main>, with its inline scripts dropped.

    The exported scripts are demo stubs - the new-chat one ends in `alert(...)` - and
    they define globals (`handleSend`, `removeMedia`) that would collide with the real
    modules. The markup they drove is kept; only the behaviour is replaced.
    """
    text = (STITCH / folder / "code.html").read_text(encoding="utf-8")
    body = text[text.index(MAIN_OPEN) + len(MAIN_OPEN): text.rindex("</main>")]
    body = re.sub(r"<script\b.*?</script>", "", body, flags=re.S).rstrip()
    # Each body closes one more tag than it opens: the div MAIN_OPEN left open.
    assert body.endswith("</div>"), f"{folder}: unexpected body ending"
    return body[: body.rindex("</div>")].rstrip()


def tag_view(body: str, view: str, wrap: bool) -> str:
    """Mark a screen so the router can show and hide it.

    A wrapper is used only where the screen has several roots. The welcome screen's root
    is `flex-1`, which resolves against its parent, so putting a plain div above it would
    collapse the layout - there the attribute goes on the existing element and nothing
    moves.
    """
    hidden = "" if view == "welcome" else "hidden "
    if wrap:
        return (f'\n<div data-view="{view}" class="{hidden}flex-col w-full">\n'
                f'{body}\n</div>\n')
    body = body.strip()
    m = re.match(r"<(\w+)\s", body)
    assert m, f"{view}: body does not start with an element"
    insert = f'<{m.group(1)} data-view="{view}" '
    out = insert + body[len(m.group(0)):]
    if hidden:                      # the Tailwind class, never the HTML attribute:
        out = re.sub(r'class="', 'class="hidden ', out, count=1)   # [hidden]{display:none}
    return "\n" + out + "\n"        # loses to Tailwind's display utilities


def localise(head: str) -> str:
    """Point every asset at this server. Nothing may be fetched from Google."""
    # Five link tags: two preconnects and three stylesheets (Material Symbols twice).
    head = re.sub(r'<link[^>]*?(?:fonts\.googleapis\.com|fonts\.gstatic\.com)[^>]*?/?>',
                  "", head)
    head = head.replace("</head>", '<link href="assets/fonts.css" rel="stylesheet"/>\n'
                                   "</head>", 1)
    head = head.replace('<script src="https://cdn.tailwindcss.com"></script>',
                        '<script src="assets/tailwind.js"></script>')
    head = re.sub(r'src="https://lh3\.googleusercontent\.com/aida/[^"]*"',
                  'src="assets/logo.png"', head)
    # The page paints before Tailwind's in-browser compiler runs. Without this the flash
    # is white against a warm page; hiding the body instead would trade a flash for a
    # blank screen, which the brief rules out.
    head = head.replace("<body ", '<body style="background:#fef8f3" ', 1)
    return head


def main() -> None:
    first = (STITCH / SCREENS[0][0] / "code.html").read_text(encoding="utf-8")
    head = localise(first[: first.index(MAIN_OPEN)])

    parts = [head, MAIN_OPEN]
    for folder, view, wrap in SCREENS:
        parts.append(f"\n<!-- ===================== {view} ===================== -->")
        parts.append(tag_view(body_of(folder), view, wrap))
    parts.append('</div></main></div>\n'
                 '<script type="module" src="js/main.js"></script>\n'
                 "</body></html>\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  {OUT.stat().st_size / 1024:.0f} KB")

    left = re.findall(r"https?://[^\"')\s]+", OUT.read_text(encoding="utf-8"))
    remote = [u for u in left if "googleusercontent" in u or "google" in u
              or "tailwindcss.com" in u]
    print("remote urls still present:", remote or "none")


if __name__ == "__main__":
    main()
