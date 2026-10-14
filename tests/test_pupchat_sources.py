"""Source links, built from metadata rather than from the model's prose.

Nothing here touches the network or spends anything: the grounding metadata is handed in
as plain dicts, which is also the point of `sources._get` accepting both dicts and the
SDK's objects.

The thing being protected is narrow and important. Asking a model to cite its sources is
the most reliable way to get invented ones, and a plausible fake citation inside a warm,
confident answer is the one kind of wrongness a reader cannot catch. So every link comes
from the API's own metadata and none from the answer text - not because invented links
are unlikely, but because there is no code path that could produce one.
"""
from __future__ import annotations

import pytest

from pupchat.config import grounding_enabled
from pupchat.sources import citation_shaped, domain_of, extract, mark


def meta(chunks=(), supports=(), entry=None):
    return {"candidates": [{"grounding_metadata": {
        "grounding_chunks": list(chunks),
        "grounding_supports": list(supports),
        "search_entry_point": ({"rendered_content": entry} if entry else None),
    }}]}


def web(uri, title="", domain=""):
    return {"web": {"uri": uri, "title": title, "domain": domain}}


def support(end, indices):
    return {"segment": {"end_index": end}, "grounding_chunk_indices": list(indices)}


# ------------------------------------------------------------------ extracting

def test_sources_are_numbered_in_order():
    resp = meta([web("https://a.test/1", "Stress signals", "rspca.org.uk"),
                 web("https://b.test/2", "Body language", "akc.org")])
    sources, _ = extract(resp)
    assert [s["n"] for s in sources] == [1, 2]
    assert [s["title"] for s in sources] == ["Stress signals", "Body language"]
    assert [s["domain"] for s in sources] == ["rspca.org.uk", "akc.org"]


def test_the_api_supplied_domain_is_preferred_over_the_uri():
    """The uri is a redirect through vertexaisearch.cloud.google.com, so parsing it
    would name Google as the source of everything."""
    resp = meta([web("https://vertexaisearch.cloud.google.com/grounding-api/xyz",
                     "Stress signals", "rspca.org.uk")])
    assert extract(resp)[0][0]["domain"] == "rspca.org.uk"


def test_a_chunk_with_no_web_page_is_skipped():
    resp = meta([{"retrieved_context": {"title": "internal"}},
                 web("https://a.test/1", "Real page", "a.test")])
    sources, _ = extract(resp)
    assert len(sources) == 1 and sources[0]["n"] == 1


def test_no_search_means_no_sources_and_no_error():
    """The model decides whether to search. Not searching is common and not a failure;
    the UI then shows nothing rather than "no sources found"."""
    assert extract({"candidates": [{}]}) == ([], None)
    assert extract({"candidates": []}) == ([], None)
    assert extract(None) == ([], None)


def test_the_search_suggestions_box_is_carried_through():
    """Google's terms require it be shown alongside a grounded answer."""
    _, html = extract(meta([web("https://a.test/1")], entry="<div>search</div>"))
    assert html == "<div>search</div>"


def test_domain_of_strips_www():
    assert domain_of("https://www.rspca.org.uk/advice") == "rspca.org.uk"
    assert domain_of("not a url") == ""


# ------------------------------------------------------------------ markers

def test_a_marker_lands_where_the_api_says():
    text = "His ears are forward. He is watching the door."
    resp = meta([web("https://a.test/1")], [support(len("His ears are forward."), [0])])
    out = mark(text, resp, extract(resp)[0])
    assert out == "His ears are forward.[1] He is watching the door."


def test_two_sources_at_one_place_become_one_marker():
    text = "His ears are forward."
    resp = meta([web("https://a.test/1"), web("https://b.test/2")],
                [support(len(text), [0]), support(len(text), [1])])
    assert mark(text, resp, extract(resp)[0]).endswith("[1][2]")


def test_offsets_are_utf8_bytes_not_characters():
    """The API's end_index counts bytes. On an answer containing one curly quote the two
    diverge, and every marker after it lands mid-word."""
    text = "He’s watching. All calm."          # the quote is 3 bytes, 1 character
    first = "He’s watching."
    resp = meta([web("https://a.test/1")],
                [support(len(first.encode("utf-8")), [0])])
    assert mark(text, resp, extract(resp)[0]) == "He’s watching.[1] All calm."


def test_an_offset_past_the_end_does_not_lose_text():
    text = "Short answer."
    resp = meta([web("https://a.test/1")], [support(9999, [0])])
    out = mark(text, resp, extract(resp)[0])
    assert out.startswith("Short answer.") and out.endswith("[1]")


def test_no_supports_leaves_the_answer_alone():
    text = "His ears are forward."
    assert mark(text, meta([web("https://a.test/1")]), []) == text


def test_marking_an_empty_answer_is_safe():
    assert mark("", meta([web("https://a.test/1")], [support(0, [0])]), []) == ""


# ------------------------------------------------------------------ invented ones

@pytest.mark.parametrize("text", [
    "Research by Smith et al. shows this.",
    "One study (Johnson, 2019) found the opposite.",
    "See doi:10.1234/abcd for details.",
    "Read more at https://example.com/made-up",
    "Published in the Journal of Veterinary Behavior.",
])
def test_citation_shaped_text_is_caught(text):
    """The model is told to write no citations at all, so anything matching here is the
    rule being broken - and it is the failure a reader cannot check for themselves."""
    assert citation_shaped(text)


@pytest.mark.parametrize("text", [
    "Research on canine stress signals finds that ears pinned back often means worry.",
    "The standard desensitisation protocol is to start below the threshold.",
    "His ears are forward and his weight is over his front paws.",
])
def test_plain_language_sourcing_is_not_flagged(text):
    """The prompt asks for plain-words attribution. Flagging that would make the rule
    unfollowable."""
    assert citation_shaped(text) == []


# ------------------------------------------------------------------ the switch

def test_grounding_is_off_unless_asked_for(monkeypatch):
    """A grounded request can cost more per answer, so this is the owner's call."""
    monkeypatch.delenv("PUPCHAT_GROUNDING", raising=False)
    assert grounding_enabled() is False
    monkeypatch.setenv("PUPCHAT_GROUNDING", "1")
    assert grounding_enabled() is True


def test_the_request_carries_no_search_tool_when_it_is_off(monkeypatch):
    monkeypatch.delenv("PUPCHAT_GROUNDING", raising=False)
    from pupchat.config import Config
    from pupchat.providers.gemini_provider import GeminiProvider

    import tempfile
    from pathlib import Path
    clip = Path(tempfile.gettempdir()) / "pupchat_test_clip.mp4"
    clip.write_bytes(b"\x00" * 32)

    payload = GeminiProvider().build_request("evidence", "why?", clip, Config())
    assert "tools" not in payload["config"]


def test_the_request_asks_for_search_when_it_is_on(monkeypatch):
    monkeypatch.setenv("PUPCHAT_GROUNDING", "1")
    from pupchat.config import Config
    from pupchat.providers.gemini_provider import GeminiProvider

    import tempfile
    from pathlib import Path
    clip = Path(tempfile.gettempdir()) / "pupchat_test_clip.mp4"
    clip.write_bytes(b"\x00" * 32)

    payload = GeminiProvider().build_request("evidence", "why?", clip, Config())
    assert payload["config"]["tools"] == [{"google_search": {}}]


def test_the_openai_provider_is_untouched(monkeypatch):
    """Grounding is a Gemini feature. OpenAI returns no sources and must not pretend."""
    monkeypatch.setenv("PUPCHAT_GROUNDING", "1")
    from pupchat.config import Config
    from pupchat.providers.openai_provider import OpenAIProvider

    payload = OpenAIProvider().build_request("evidence", "why?", [], Config())
    assert "tools" not in payload
