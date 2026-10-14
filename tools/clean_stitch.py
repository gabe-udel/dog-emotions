"""Strip the invented sample content out of the merged page. Run once, after merge_stitch.

The Stitch export is a mockup: two dogs called Milo and Bella, five conversations, a user
called Sarah on a "Premium Plan", three baseline videos, and a panel of measurements the
pipeline cannot produce. None of it is real and none of it may ship.

Everything removed here goes to one of two places:

* a `<template>`, when the page needs to stamp out many of them from real data
* a friendly empty state, for when there is nothing yet

The design's own markup is reused in both cases - a template IS the Stitch row, moved. So
the look is still the export's; only the content changes.

Run:  .venv\\Scripts\\python.exe tools\\clean_stitch.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "pupchat" / "webapp" / "index.html"

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
        "source", "track", "wbr"}
TAG = re.compile(r"<(/?)([a-zA-Z][\w-]*)([^>]*?)(/?)>")


def element_at(html: str, start: int) -> tuple[int, int]:
    """The [start, end) span of the element whose opening tag begins at `start`.

    A deliberately small substitute for an HTML parser, which this project does not have
    installed. It only has to cope with Stitch's output: well-formed, quoted attributes,
    and no script or style left by this point.
    """
    m = TAG.match(html, start)
    if not m:
        raise ValueError(f"no tag at {start}: {html[start:start + 60]!r}")
    if m.group(2).lower() in VOID or m.group(4):
        return start, m.end()
    depth = 0
    for t in TAG.finditer(html, start):
        if t.group(2).lower() in VOID or t.group(4):
            continue
        depth += -1 if t.group(1) else 1
        if depth == 0:
            return start, t.end()
    raise ValueError(f"unclosed <{m.group(2)}> at {start}")


class Page:
    def __init__(self, path: Path):
        self.path = path
        self.html = path.read_text(encoding="utf-8")
        self.log: list[str] = []
        self.templates: list[str] = []

    # ------------------------------------------------------------------ locating
    def _open_tag_before(self, needle: str, tag: str, nth: int = 0) -> int:
        """Index of the `tag` element that contains the nth occurrence of `needle`."""
        pos = -1
        for _ in range(nth + 1):
            pos = self.html.index(needle, pos + 1)
        return self.html.rindex(tag, 0, pos + 1)

    def _after(self, marker: str) -> int:
        """Index of the first element after the comment containing `marker`.

        Matched loosely: Stitch writes `<!-- Label -->` with padding spaces, and pinning
        the exact spacing makes every edit here fragile for no benefit.
        """
        m = re.search(r"<!--[^>]*?" + re.escape(marker) + r"[^>]*?-->", self.html)
        if not m:
            raise SystemExit(f"no comment containing {marker!r}")
        return self.html.index("<", m.end())

    # ------------------------------------------------------------------ editing
    def _replace(self, a: int, b: int, new: str, label: str) -> str:
        old = self.html[a:b]
        self.html = self.html[:a] + new + self.html[b:]
        self.log.append(f"{label:44s} {len(old):6d} -> {len(new):5d}")
        return old

    def cut(self, needle: str, label: str, *, tag: str = "<", nth: int = 0) -> str:
        a, b = element_at(self.html, self._open_tag_before(needle, tag, nth))
        return self._replace(a, b, "", f"cut    {label}")

    def cut_after(self, marker: str, label: str) -> str:
        a, b = element_at(self.html, self._after(marker))
        return self._replace(a, b, "", f"cut    {label}")

    def swap(self, needle: str, new: str, label: str, *, tag: str = "<",
             nth: int = 0) -> str:
        a, b = element_at(self.html, self._open_tag_before(needle, tag, nth))
        return self._replace(a, b, new, f"swap   {label}")

    def inner(self, needle: str, new: str, label: str, *, tag: str = "<",
              nth: int = 0) -> str:
        a, b = element_at(self.html, self._open_tag_before(needle, tag, nth))
        open_end = self.html.index(">", a) + 1
        close_start = self.html.rindex("<", a, b)
        return self._replace(open_end, close_start, new, f"inner  {label}")

    def attr(self, needle: str, extra: str, label: str, *, tag: str = "<",
             nth: int = 0) -> None:
        """Add an attribute to an existing element without moving anything."""
        a = self._open_tag_before(needle, tag, nth)
        m = TAG.match(self.html, a)
        self._replace(a, a + len(m.group(0).split(" ", 1)[0]) + 1,
                      f"{m.group(0).split(' ', 1)[0]} {extra} ", f"attr   {label}")


    def card_content(self, heading: str, new: str, label: str) -> str:
        """Replace a profile card's body, keeping its chrome and its heading.

        The cards are the design's; only what is inside them changes. A card is an
        <article> whose first child is the header strip and whose second is the content
        grid, so the content is the element right after the header closes.
        """
        h = self.html.index(f">{heading}</h2>")
        art = self.html.rindex("<article", 0, h)
        a0, a1 = element_at(self.html, art)
        head_start = self.html.index("<", art + len("<article"))
        _, head_end = element_at(self.html, head_start)
        body_start = self.html.index("<", head_end)
        _, body_end = element_at(self.html, body_start)
        assert body_end <= a1, f"{heading}: content ran past the card"
        return self._replace(body_start, body_end, new, f"card   {label}")

    def link(self, needle: str, href: str, label: str, nth: int = 0) -> None:
        """Give the <a> containing `needle` a real destination.

        By content rather than by pattern: the export wraps its link text in spans at
        varying depth, so a regex over the opening tag plus the first words is brittle
        in a way that only shows up as a silently unchanged `href="#"`.
        """
        a = self._open_tag_before(needle, "<a", nth)
        end = self.html.index(">", a)
        tag = self.html[a:end + 1]
        if 'href="#"' not in tag:
            raise SystemExit(f"{label}: that <a> has no href=\"#\" ({tag[:80]!r})")
        self._replace(a, end + 1, tag.replace('href="#"', f'href="{href}"'),
                      f"link   {label}")

    def sub(self, pattern: str, new: str, label: str, expect: int = 0) -> None:
        self.html, n = re.subn(pattern, new, self.html)
        if n == 0:
            raise SystemExit(f"no match: {label}  ({pattern[:70]!r})")
        if expect and n != expect:
            raise SystemExit(f"{label}: expected {expect}, matched {n}")
        self.log.append(f"{'sub    ' + label:44s} x{n}")

    def text(self, old: str, new: str, label: str, expect: int = 0) -> None:
        n = self.html.count(old)
        if n == 0:
            raise SystemExit(f"not found: {label}  ({old[:70]!r})")
        if expect and n != expect:
            raise SystemExit(f"{label}: expected {expect}, found {n}")
        self.html = self.html.replace(old, new)
        self.log.append(f"{'text   ' + label:44s} x{n}")

    def template(self, name: str, html: str) -> None:
        self.templates.append(f'<template id="tpl-{name}">{html}</template>')

    def assemble(self) -> None:
        """Put the templates back into the document.

        Everything cut into a template leaves `self.html`, so an edit aimed at template
        content silently finds nothing until this has run. Assembling before the final
        pass is what lets one rule - "no asset may come from Google" - be applied to the
        live markup and the templates alike.
        """
        block = "\n\n".join(self.templates)
        hook = '<script type="module" src="js/main.js"></script>'
        self.html = self.html.replace(hook, f"{block}\n\n{hook}")
        self.templates = []

    def write(self) -> None:
        self.path.write_text(self.html, encoding="utf-8")


EMPTY = ('<p data-empty="{k}" class="px-3 py-2 font-body-sm text-body-sm text-outline">'
         "{t}</p>")


FIELD = """<div class="bg-surface-container-low p-4 rounded-xl">
<span class="font-label-sm text-label-sm uppercase tracking-wider text-outline block mb-1 font-semibold">{label}</span>
<{el} data-field="{key}" placeholder="{hint}" rows="{rows}"
 class="w-full bg-transparent font-body-md text-body-md text-on-surface font-medium
 border border-transparent rounded-lg px-1.5 -mx-1.5 py-0.5 resize-none
 placeholder:text-outline/70 placeholder:font-normal
 focus:outline-none focus:border-outline-variant focus:bg-surface-container-lowest
 transition-colors"></{el}>
</div>"""


def field(key: str, label: str, hint: str, rows: int = 0) -> str:
    """One editable value in the design's own tile.

    An input rather than a value plus an Edit button: the border only appears on focus,
    so the card still reads as the mockup until you click into it, and there is no edit
    mode to leave half-finished.
    """
    el = "textarea" if rows else "input"
    return FIELD.format(label=label, key=key, hint=hint, el=el, rows=rows or 1)


GRID = '<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">{}</div>'
ONE = '<div class="grid grid-cols-1 gap-4">{}</div>'


def main() -> None:
    p = Page(PAGE)

    # ---------------------------------------------------------------- sidebar: dogs
    p.template("dog-row", p.cut(">Milo</p>", "sample dog Milo", tag="<a"))
    p.cut(">Bella</p>", "sample dog Bella", tag="<a")
    p.inner("data-active-classes", EMPTY.format(k="dogs",
            t="No dogs yet. Press Add to make one."),
            "dog list -> empty state", tag="<nav")

    # ---------------------------------------------------------------- sidebar: chats
    # nth=1: both navs carry data-active-classes, and the dog one comes first. Without
    # this the chat empty state overwrites the dog one and the chat list stays blank.
    p.template("chat-row", p.cut(">Barking at guests</span>", "sample chat", tag="<a"))
    for t in ("Leash pulling on walks", "Is this rough play?",
              "Whining during car rides", "Separation anxiety at noon"):
        p.cut(f">{t}</span>", f"sample chat: {t}", tag="<a")
    p.inner("data-active-classes", EMPTY.format(k="chats",
            t="No chats yet. Add a clip and ask a question."),
            "chat list -> empty state", tag="<nav", nth=1)

    # ------------------------------------------------- sidebar footer: who is using it
    # "Sarah & Milo / Premium Plan" is invented, and "Premium" is wrong in kind too:
    # these weights are non-commercial, so the app can never be sold. The space is worth
    # more as the credit the licence actually requires.
    p.swap('alt="Sarah &amp; Milo"', """
<div class="flex flex-col gap-2 min-w-0 flex-1">
  <div class="flex items-center gap-2 font-label-sm text-label-sm text-outline">
    <span id="healthDot" class="w-2 h-2 rounded-full bg-outline shrink-0"></span>
    <span id="healthText" class="truncate">connecting…</span>
  </div>
  <p id="creditLine" class="font-label-sm text-label-sm text-outline/80 leading-snug"></p>
  <p class="font-label-sm text-label-sm text-outline/70 leading-snug">
    Non-commercial demo. Research prototype, not a product.</p>
  <p id="shareNotice" class="hidden font-label-sm text-label-sm text-outline/80 leading-snug"></p>
</div>""", "user block -> health + credits", tag="<a")

    # ---------------------------------------------------------------- dead controls
    p.cut('data-path="settings"', "settings gear", tag="<a")
    p.cut('data-path="help"', "help icon", tag="<a")
    p.cut('data-path="notifications"', "notifications bell", tag="<a")
    p.cut('alt="Profile"', "header avatar", tag="<img")
    p.text('<div class="h-5 w-px bg-outline-variant mx-1"></div>', "",
           "divider left by the avatar")
    for _ in range(2):
        p.cut(">mic</span>", "mic button", tag="<button")

    # ---------------------------------------------------------------- the chat view
    # The fabricated measurements go FIRST, so they cannot survive inside the template
    # taken from the card that contains them. Zone 2 / Rhythmic Staccato / High Rigid
    # Sweep / Every 3-4 barks: two of those need audio, which this project has none of,
    # and "alert"/"aroused" are in config.BANNED_WORDS exactly so that a conclusion is
    # never dressed up as a measurement.
    p.cut_after("Visual Signal Diagram", "fabricated arousal panel")

    p.template("user-turn", p.cut_after("1. USER MESSAGE ITEM", "sample question"))
    p.template("answer-card",
               p.cut_after("2. PUPCHAT AI RESPONSE CARD", "sample answer"))
    p.template("day-break", p.cut_after("Timeline Break", "sample date chip"))

    # What is left of the conversation column is where real turns are appended.
    p.attr("Conversation Column", 'data-stream', "mark the message stream", tag="<div")

    # The popover ships visible (no `hidden` class) and the export's own script only
    # toggled it, so on a fresh load it covers the composer.
    p.text('<div class="w-60 mb-2 bg-surface-container-lowest',
           '<div class="hidden w-60 mb-2 bg-surface-container-lowest',
           "hide the attach popover until asked", expect=1)

    # ---------------------------------------------------------------- welcome screen
    p.text("What’s Milo up to today?",
           'What’s <span data-field="dogName">your dog</span> up to today?',
           "hero greeting")
    p.text("decode Milo’s body language", "decode your dog’s body language",
           "hero subtitle")
    p.text('placeholder="Ask about Milo..."', 'placeholder="Ask about your dog…"',
           "welcome composer placeholder")
    p.text('placeholder="Ask about Milo…"', 'placeholder="Ask about your dog…"',
           "chat composer placeholder")
    p.text(">Milo_backyard_zoomies.mp4<", "><", "sample attachment name")

    # The prompt cards are good; their inline onclick is a demo stub that names the
    # sample dog. The question becomes data for the real module to read.
    p.text("Why does Milo bark whenever guests arrive at the front door?",
           "Why does my dog bark when guests arrive?", "prompt card question")
    p.text("onclick=\"insertPrompt('", 'data-prompt="', "prompt handler -> data")
    p.text("')\" type=\"button\"", '" type="button"', "prompt handler tail")

    # "Active Milo Baselines: Sound-sensitive (Thunder)" is invented. The strip stays,
    # driven by the real count, and hidden when there are none.
    p.swap("Active Milo Baselines:", """
<div data-baseline-strip class="hidden items-center gap-2 px-4 py-2 rounded-full bg-surface-container-low">
  <span class="material-symbols-outlined text-[16px] text-primary">verified</span>
  <span class="font-label-sm text-label-sm text-on-surface-variant">
    <span data-field="baselineCount">0</span> baseline clips saved</span>
  <a data-link="dog" class="font-label-sm text-label-sm text-primary hover:underline cursor-pointer">View profile</a>
</div>""", "baseline strip -> real count", tag="<div")

    # ---------------------------------------------------------------- the dog switcher
    p.text('Chatting about: <span class="font-semibold text-primary">Milo</span>',
           'Chatting about: <span class="font-semibold text-primary" '
           'data-field="dogName">no dog yet</span>', "welcome context pill")
    p.text('Chatting about: <strong class="text-primary font-bold">Milo</strong>',
           'Chatting about: <strong class="text-primary font-bold" '
           'data-field="dogName">no dog yet</strong>', "chat sub-header")
    p.text('Chatting about: <strong class="text-primary font-semibold">Milo</strong>',
           'Chatting about: <strong class="text-primary font-semibold" '
           'data-field="dogName">no dog yet</strong>', "header switcher")
    p.text(">Golden Retriever · 2y 3m<", ' data-field="dogMeta"><', "chat breed chip")
    p.text(">Golden Retriever · 2y<", ' data-field="dogMeta"><', "breed chips")
    p.text("Milo's Profile &amp; Baselines", "Profile &amp; Baselines", "profile link")

    # ---------------------------------------------------------------- the dog profile
    p.text('rounded-full">Milo</span>', 'rounded-full" data-field="dogName">this dog</span>',
           "breadcrumb")
    p.cut("Baseline Profile Synchronized", "fake sync badge", tag="<div")
    p.cut("Microchipped", "fake clinic line", tag="<p")
    # By button, not by the "Action Buttons" comment: the welcome composer's send row
    # is commented "Action Buttons / Send", comes first in the document, and matched -
    # which quietly removed the send button from the whole app.
    p.cut("Share Profile", "Share Profile button", tag="<button")
    p.cut("Edit Profile", "Edit Profile button", tag="<button")

    # The photo, or the first letter. One element either way, so there is no empty frame
    # before a photo is added.
    p.swap('alt="Milo the Golden Retriever Mix"', """
<div data-avatar="dog" class="w-28 h-28 sm:w-32 sm:h-32 rounded-full bg-secondary-container
 text-on-secondary-container flex items-center justify-center shadow-md
 ring-4 ring-surface-container-low overflow-hidden bg-cover bg-center cursor-pointer"
 title="Click to add a photo">
  <span data-field="dogInitial" class="font-display-lg text-display-lg font-semibold">?</span>
</div>
<input data-input="photo" type="file" accept="image/*" class="hidden"/>""",
           "profile photo -> monogram", tag="<img")
    p.text('font-bold tracking-tight">Milo</h1>',
           'font-bold tracking-tight" data-field="dogName">this dog</h1>', "profile title")
    p.text("Golden Retriever Mix • 3 years old",
           '<span data-field="dogMeta">no details yet</span>', "profile subtitle")

    # Four vitals the mockup invents (weight, BCS, neuter status, temperament) become
    # four the app actually knows.
    p.text(">3 yrs 2 mos<", ' data-field="statAge">—<', "vitals: age")
    p.text(">64 lbs (BCS 5/9)<", ' data-field="statBreed">—<', "vitals: breed")
    p.text(">Neutered Male<", ' data-field="statBaselines">0<', "vitals: baselines")
    p.text(">Biddable &amp; Alert<", ' data-field="statChats">0<', "vitals: chats")
    p.text(">Weight<", ">Breed<", "vitals label: weight -> breed")
    p.text(">Status<", ">Baselines<", "vitals label: status -> baselines")
    p.text(">Temperament<", ">Chats<", "vitals label: temperament -> chats")

    # The four cards keep their chrome and headings; their contents become the seven
    # fields the pipeline actually sends to the model (pupchat/webapp app.js profileText).
    p.card_content("Basics", GRID.format(
        field("name", "Name", "What do you call them?")
        + field("breed", "Breed", "Labrador, collie cross, not sure…")
        + field("age", "Age", "2 years, 8 months, a puppy…")), "Basics -> name/breed/age")
    p.card_content("Health &amp; Wellness", ONE.format(
        field("health", "Health", "Anything a vet has said, or nothing yet.", rows=3)),
        "Health -> one field")
    p.card_content("Training History", ONE.format(
        field("training", "Training", "What they know, and how you taught it.", rows=3)),
        "Training -> one field")
    p.card_content("Behavior Notes", ONE.format(
        field("triggers", "Known triggers", "Doorbell, other dogs, the vacuum…", rows=2)
        + field("notes", "Anything else", "What you would tell a new sitter.", rows=3)),
        "Behaviour -> triggers + notes")

    # ---------------------------------------------------------- baselines & past chats
    p.template("baseline-tile", p.cut_after("Video Card 1", "sample baseline 1"))
    p.cut_after("Video Card 2", "sample baseline 2")
    p.cut_after("Video Card 3", "sample baseline 3")
    p.text(">3 Uploaded<", ' data-field="baselineCount">0 saved<', "baseline count badge")
    p.text("Short baseline videos help Pupchat recognize Milo's unique body language, "
           "neutral tail carriage, and typical vocal cadence during assessments.",
           "A short clip of a calm, ordinary moment. It gives Pupchat something to "
           "compare against, so it can say what is unusual for this dog rather than "
           "just what is unusual for dogs.", "baseline explainer")
    p.text("Teach Pupchat about Milo", "Teach Pupchat about this dog", "baseline heading")

    # Hooks for the two lists the dog page draws. Without them the renderer finds its
    # container by matching a Tailwind class string, which breaks the moment the design
    # is re-exported with different spacing.
    p.text('<div class="grid grid-cols-1 sm:grid-cols-2 gap-3.5">',
           '<div data-baselines class="grid grid-cols-1 sm:grid-cols-2 gap-3.5">',
           "baseline grid hook", expect=1)
    p.text('<div class="space-y-2.5" role="list">',
           '<div data-past-chats class="space-y-2.5" role="list">',
           "past chats hook", expect=1)

    p.template("past-chat-row", p.cut(">Barking at guests<", "sample past chat", tag="<a"))
    for t in ("Leash pulling on morning walks", "Is this rough play with neigh",
              "Whining during highway car"):
        p.cut(f">{t}", f"sample past chat: {t[:24]}", tag="<a")
    p.text("Past Chats about Milo", "Past chats", "past chats heading")
    p.text("Start new consultation for Milo", "Start a new chat about this dog",
           "new chat link")

    # ---------------------------------------------------------------- the last photos
    # Seven invented dog photos. Each becomes the same monogram the sidebar already uses,
    # so a dog with no photo looks deliberate rather than broken, and the element is
    # ready for the owner's own picture.
    def monogram(size: str, text: str, extra: str = "") -> str:
        return (f'<div data-avatar="dog" class="{size} rounded-full bg-secondary-container'
                f' text-on-secondary-container flex items-center justify-center shrink-0'
                f' overflow-hidden bg-cover bg-center {extra}">'
                f'<span data-field="dogInitial" class="{text} font-semibold">?</span></div>')

    p.swap("A radiant golden retriever", monogram(
        "w-28 h-28", "font-display-lg text-display-lg", "shadow-sm"),
        "welcome hero photo", tag="<img")
    p.swap("A close-up portrait of a cheerful golden retriever", monogram(
        "w-full h-full", "font-label-sm text-label-sm"),
        "welcome context-pill photo", tag="<img")
    p.swap('alt="Milo portrait"', monogram(
        "w-10 h-10", "font-label-md text-label-md", "shadow-sm ring-2 ring-primary-fixed"),
        "chat sub-header photo", tag="<img")
    p.text("<!-- Milo Profile Identity Header -->", "<!-- Profile header -->",
           "stale comment")

    # ---------------------------------------------------------------- phones
    # The export is fixed-width desktop: a 288px sidebar with everything else pushed
    # right by exactly that. Four class edits make it a drawer below `lg`, and the
    # desktop rendering is byte-identical because every original rule is kept, just
    # behind the `lg:` breakpoint.
    p.text('<aside class="fixed left-0 top-0 h-screen w-72',
           '<aside data-sidebar class="fixed left-0 top-0 h-screen w-72 '
           '-translate-x-full lg:translate-x-0 transition-transform duration-200',
           "sidebar -> drawer on small screens")
    p.text('<div class="pl-72 min-h-screen bg-surface"',
           '<div class="lg:pl-72 min-h-screen bg-surface"', "content offset")
    p.text('<header class="fixed top-0 left-72 right-0',
           '<header class="fixed top-0 left-0 lg:left-72 right-0', "header offset")
    p.text('class="fixed bottom-0 left-72 right-0',
           'class="fixed bottom-0 left-0 lg:left-72 right-0', "composer dock offset")

    # A menu button, and a backdrop that closes the drawer. Both hidden on desktop.
    p.text('<header class="fixed top-0 left-0 lg:left-72 right-0',
           '<div data-scrim class="hidden lg:hidden fixed inset-0 bg-inverse-surface/40 z-30"></div>'
           '<header class="fixed top-0 left-0 lg:left-72 right-0', "sidebar backdrop")
    p.sub(r'(<header class="fixed top-0 left-0 lg:left-72[^"]*"[^>]*>)',
          r'''\1<button data-menu type="button" title="Menu"
 class="lg:hidden p-2 -ml-2 mr-1 rounded-full text-on-surface-variant hover:bg-surface-container">
 <span class="material-symbols-outlined text-[22px]">menu</span></button>''',
          "menu button", expect=1)

    # ---------------------------------------------------------------- the error surface
    # Errors are transient UI, not conversation. An earlier version routed them through
    # the chat store, which created a chat titled "(error)" for every rejected upload and
    # read exactly like history not being saved.
    p.text("<!-- Profile header -->", """
<div id="videoModal" class="hidden fixed inset-0 z-50 bg-inverse-surface/80 backdrop-blur-sm
 flex items-center justify-center p-4">
  <div class="relative w-full max-w-3xl">
    <button id="videoModalClose" type="button" title="Close"
      class="absolute -top-3 -right-3 z-10 w-10 h-10 rounded-full bg-surface-container-lowest
      text-on-surface shadow-lg flex items-center justify-center hover:bg-surface-container">
      <span class="material-symbols-outlined text-[22px]">close</span>
    </button>
    <video id="videoModalPlayer" class="w-full rounded-2xl bg-black shadow-2xl"
      controls autoplay playsinline></video>
  </div>
</div>
<div id="toast" class="hidden fixed top-4 left-1/2 -translate-x-1/2 z-50 max-w-lg w-[90%]"
  <div class="flex items-start gap-3 px-4 py-3 rounded-xl bg-error-container text-on-error-container shadow-lg">
    <span class="material-symbols-outlined text-[20px] shrink-0">warning</span>
    <p id="toastText" class="font-body-sm text-body-sm flex-1"></p>
    <button id="toastClose" type="button" class="material-symbols-outlined text-[18px] shrink-0 hover:opacity-70">close</button>
  </div>
</div>
<!-- Profile header -->""", "toast banner")

    # ---------------------------------------------------------------- dead links
    # Every <a> in the export points at "#", which scrolls to the top and reads as
    # broken. Each one either gets a real path or stops being a link.
    p.text('data-path="new-chat" href="#"', 'data-path="new-chat" href="/"', "New -> /")
    p.text('data-path="add-dog" href="#"', 'data-path="add-dog" href="/dogs/new"',
           "Add -> /dogs/new")
    p.text('data-path="dog-profile" href="#"', 'data-path="dog-profile" href="/dogs"',
           "profile link")
    p.link(">My Dogs<", "/dogs", "breadcrumb My Dogs", nth=1)   # nth=0 is the sidebar
    p.link(">add_comment<", "/", "start a new chat")
    p.cut("View All", "View All link", tag="<a")

    # The three attach choices are actions, not destinations. Recording and photos are
    # not built, so only the one that works survives.
    p.swap("Upload video", '<button data-action="upload" type="button" '
           'class="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl '
           'hover:bg-surface-container transition-colors text-left">'
           '<span class="material-symbols-outlined text-[20px] text-primary">video_file'
           '</span><span class="font-label-md text-label-md text-on-surface">'
           'Upload a clip</span></button>', "upload action", tag="<a")
    p.cut("Record video", "record video (not built)", tag="<a")
    p.cut("Add photo", "add photo (not built)", tag="<a")

    # Declared in the page rather than created in JS, so the hook test can see it.
    p.text('<div data-past-chats class="space-y-2.5" role="list">',
           '<div data-past-chats class="space-y-2.5" role="list">'
           '<p data-empty-chats class="hidden font-body-sm text-body-sm text-outline py-2">'
           "No chats about this dog yet.</p>", "past chats empty state", expect=1)

    # ---------------------------------------------------------------- new templates
    # The export has no "working on it" state - its mockup is always of a finished
    # answer. Built from the same card chrome so it belongs to the design.
    p.templates.append("""<template id="tpl-progress">
<div data-progress class="flex">
  <div class="w-full max-w-2xl px-5 py-4 rounded-2xl bg-surface-container-lowest border border-outline-variant/60 shadow-sm space-y-3">
    <div class="flex items-center gap-2.5">
      <span class="material-symbols-outlined text-[20px] text-primary">visibility</span>
      <p class="font-label-lg text-label-lg text-on-surface font-semibold">Pupchat is watching the video…</p>
    </div>
    <ol data-steps class="space-y-1.5"></ol>
    <div data-bar-wrap class="hidden h-1.5 rounded-full bg-surface-container overflow-hidden">
      <div data-bar class="h-full bg-primary transition-all duration-300" style="width:0%"></div>
    </div>
    <p data-eta class="font-body-sm text-body-sm text-outline"></p>
  </div>
</div></template>""")

    p.templates.append("""<template id="tpl-progress-step">
<li class="flex items-center gap-2.5 font-body-sm text-body-sm text-outline">
  <span data-icon class="material-symbols-outlined text-[18px] text-outline-variant">radio_button_checked</span>
  <span data-label class="flex-1"></span>
  <span data-detail class="font-label-sm text-label-sm text-outline/70"></span>
</li></template>""")

    p.templates.append("""<template id="tpl-error-turn">
<div class="flex">
  <div class="max-w-2xl px-4 py-3 rounded-2xl bg-error-container text-on-error-container border border-error/20">
    <p data-field="text" class="font-body-sm text-body-sm"></p>
  </div>
</div></template>""")

    # From here on the templates are part of the document again, so one rule covers both.
    p.assemble()

    p.sub(r'src="https://lh3\.googleusercontent\.com/aida/[A-Za-z0-9_-]+"',
          'src="assets/logo.png"', "assistant badge logo", expect=1)
    p.text("Behavioral Breakdown · Milo",
           '<span data-field="dogName">this dog</span>', "answer card subtitle")
    p.swap("A golden retriever dog barking excitedly", '<video data-clip '
           'class="w-full h-full object-cover" muted loop playsinline '
           'preload="metadata"></video>',
           "sample video thumbnail -> real clip", tag="<img")
    # The whole attribute, not just its prefix: an inline background-image is the one
    # place a half-removed URL still loads.
    p.sub(r"""style="background-image: url\('[^']*'\)\"""", 'data-thumb=""',
          "baseline tile photo", expect=1)

    print("\n".join(p.log))

    # The one check that cannot be skipped. Three of the export's images live in
    # `style="background-image:url(...)"` rather than `src`, so a grep for src= passes
    # while the page still phones Google on every visit.
    left = re.findall(r"https?://[^\"')\s]+", p.html)
    if left:
        raise SystemExit("still loading from the internet:\n  "
                         + "\n  ".join(sorted(set(u[:90] for u in left))))

    p.write()
    print(f"\nwrote {PAGE.relative_to(ROOT)}  {PAGE.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
