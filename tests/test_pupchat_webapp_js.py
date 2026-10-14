"""Structural checks on the front-end modules.

There is no Node and no browser in this environment, so none of this runs the code. What
it can do is catch the failures that would otherwise only show up as a blank page with a
message in a console nobody is watching: a module that imports a file that isn't there, a
template the renderer asks for that the page doesn't define, a hook that got renamed on
one side only.

Those are exactly the mistakes a rename causes, and they are silent in a browser.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import WEBAPP, webapp_files

JS_DIR = WEBAPP / "js"


def js_files() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(JS_DIR.glob("*.js"))}


#: A `/` in one of these positions starts a regex literal rather than a division.
_REGEX_AFTER = set("(,=:[!&|?{};\n+-*%~^<>") | {""}


def strip_js(src: str) -> str:
    """Remove strings, template literals, comments and regex literals.

    Crude, and only used for counting delimiters. Over-removal is safe: it can make a
    balance check pass that should fail, never the other way round.

    Regex literals have to go, and that is not optional - `/^\\d+[.)]\\s+/` contains a
    lone ")" and `{1,4}` a lone brace pair, so leaving them in reports a perfectly good
    module as unbalanced.
    """
    out, i, n = [], 0, len(src)
    prev = ""
    while i < n:
        c = src[i]
        if c in "\"'`":
            quote = c
            i += 1
            while i < n and src[i] != quote:
                i += 2 if src[i] == "\\" else 1
            i += 1
            prev = quote
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i)
            i = n if j < 0 else j + 2
        elif c == "/" and prev in _REGEX_AFTER:
            i += 1
            in_class = False
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == "[":
                    in_class = True
                elif src[i] == "]":
                    in_class = False
                elif src[i] == "/" and not in_class:
                    break
                elif src[i] == "\n":
                    break               # not a regex after all; bail out safely
                i += 1
            i += 1
        else:
            out.append(c)
            if not c.isspace():
                prev = c
            elif c == "\n":
                prev = "\n"
            i += 1
    return "".join(out)


# ------------------------------------------------------------------ the modules

def test_the_page_loads_a_module_entry_point():
    html = webapp_files()["index.html"]
    assert 'type="module"' in html, "the scripts are not ES modules"
    assert re.search(r'<script[^>]+src="js/main\.js"', html)


@pytest.mark.parametrize("name", [p.name for p in sorted(JS_DIR.glob("*.js"))])
def test_every_module_is_balanced(name):
    """A stray brace is a parse error, and a parse error is a blank page."""
    src = strip_js((JS_DIR / name).read_text(encoding="utf-8"))
    for open_c, close_c in ("{}", "()", "[]"):
        assert src.count(open_c) == src.count(close_c), (
            f"{name}: {open_c}{close_c} unbalanced "
            f"({src.count(open_c)} vs {src.count(close_c)})")


def test_every_import_resolves():
    """A typo'd import kills the whole module graph, so nothing on the page works."""
    for name, src in js_files().items():
        for spec in re.findall(r'from\s+"([^"]+)"', src):
            assert spec.startswith("./"), f"{name} imports {spec} from somewhere remote"
            assert (JS_DIR / spec[2:]).exists(), f"{name} imports missing {spec}"


def test_no_module_is_orphaned():
    """Every module must be reachable from main.js, or it is dead code pretending to
    be wired up."""
    files = js_files()
    seen, queue = {"main.js"}, ["main.js"]
    while queue:
        src = files[queue.pop()]
        for spec in re.findall(r'from\s+"\./([^"]+)"', src):
            if spec not in seen:
                seen.add(spec)
                queue.append(spec)
    assert seen == set(files), f"not reachable from main.js: {set(files) - seen}"


# ------------------------------------------------------------------ the contract

def test_every_template_the_code_asks_for_exists():
    """tpl("dog-row") throws at runtime if the page has no such template, and it throws
    inside a render loop, so the sidebar simply stops drawing."""
    html = webapp_files()["index.html"]
    defined = set(re.findall(r'<template id="tpl-([^"]+)"', html))
    asked = set()
    for src in js_files().values():
        asked |= set(re.findall(r'tpl\("([^"]+)"\)', src))
    assert asked, "no templates are used at all"
    assert asked <= defined, f"missing templates: {sorted(asked - defined)}"


def test_every_dom_hook_the_code_uses_is_in_the_page():
    """data-* hooks are the join between the design and the code. A rename on one side
    only fails silently - querySelector returns null and the feature quietly does
    nothing."""
    html = webapp_files()["index.html"]
    used = set()
    for src in js_files().values():
        used |= set(re.findall(r'\[data-([a-z-]+)(?:="[^"]*")?\]', src))
    # `data-view` and `data-field` are written by the code as well as read.
    for hook in used - {"view", "field"}:
        assert f"data-{hook}" in html, f"nothing in the page has data-{hook}"


def test_every_element_id_the_code_touches_exists():
    html = webapp_files()["index.html"]
    ids = set(re.findall(r'id="([^"]+)"', html))
    used = set()
    for src in js_files().values():
        used |= set(re.findall(r'\$\("([^"]+)"\)', src))
        used |= set(re.findall(r'getElementById\("([^"]+)"\)', src))
    # dogMenu is created by the code when the switcher is first opened.
    for name in used - {"dogMenu"}:
        assert name in ids, f"no element with id={name}"


def test_no_dead_link_is_left_in_the_page():
    """href="#" scrolls to the top and looks broken. Every link either goes somewhere
    or is a button."""
    html = webapp_files()["index.html"]
    live = html[: html.index("<template")]
    assert 'href="#"' not in live, "a dead link survived in the live page"
