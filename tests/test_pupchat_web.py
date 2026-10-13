"""The Gradio page.

The bug this file exists for: `main()` shipped calling `launch(show_api=False)`, which
Gradio 6 removed, so `python -m pupchat.web` raised on startup. It went unnoticed
because the page was only ever checked by calling `build()` - constructing a Blocks
never touches launch's signature, so the page "built fine" and could not run.

Nothing here binds a port or opens a browser.
"""
from __future__ import annotations

import inspect

import pytest


def test_the_page_constructs():
    from pupchat.web import build
    assert build() is not None


def test_main_calls_launch_with_arguments_gradio_accepts(monkeypatch):
    """Binds main()'s kwargs against the REAL launch signature, so a removed or
    renamed argument fails here instead of at startup."""
    import gradio as gr

    real = inspect.signature(gr.Blocks.launch)
    captured: dict = {}

    def fake_launch(self, *args, **kwargs):
        real.bind(self, *args, **kwargs)        # TypeError on an unknown kwarg
        captured.update(kwargs)
        return None

    monkeypatch.setattr(gr.Blocks, "launch", fake_launch)
    from pupchat.web import main
    main(port=7999, open_browser=False)

    assert captured["server_port"] == 7999
    assert captured["inbrowser"] is False


def test_an_unknown_launch_kwarg_would_be_caught(monkeypatch):
    """Proves the guard above actually bites, rather than passing vacuously."""
    import gradio as gr

    real = inspect.signature(gr.Blocks.launch)

    def fake_launch(self, *args, **kwargs):
        real.bind(self, *args, **kwargs)
        return None

    monkeypatch.setattr(gr.Blocks, "launch", fake_launch)
    with pytest.raises(TypeError):
        gr.Blocks().launch(show_api=False)       # the argument that shipped broken


def test_the_page_has_the_pieces_the_workflow_needs():
    """Upload, ask, read the evidence, flag an answer."""
    from pupchat.web import build
    kinds = [type(b).__name__ for b in build().blocks.values()]
    assert "Video" in kinds
    assert "Chatbot" in kinds
    assert kinds.count("Button") >= 2          # show measurements, flag
    assert "Accordion" in kinds                # the flag panel
    assert kinds.count("Textbox") >= 3         # question, evidence, flag note


def test_the_licence_notice_is_on_the_page():
    """Non-commercial weights; a user of the page has to be told."""
    from pupchat.config import LICENCE_NOTICE
    from pupchat.web import build
    text = " ".join(getattr(b, "value", "") or "" for b in build().blocks.values()
                    if isinstance(getattr(b, "value", None), str))
    assert LICENCE_NOTICE in text


def test_the_page_warns_that_the_first_run_is_slow():
    """A multi-minute wait with no warning reads as a hang."""
    from pupchat.web import INTRO
    assert "minutes" in INTRO.lower()
