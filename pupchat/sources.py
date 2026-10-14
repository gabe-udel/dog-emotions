"""Where the advice came from, built from the provider's metadata.

Asking a model to cite its sources is the most reliable way to get invented ones, and a
plausible fake citation inside a warm, confident answer is the one kind of wrongness a
reader has no way to notice. So the model is told to write no citations at all, and every
link here is read out of Gemini's grounding metadata instead.

That is the whole design: **a fabricated URL has nowhere to come from.** It is not that
invented links are unlikely - there is no code path that could produce one.

What this cannot do is tell you a cited page is correct. Grounding says where an answer
was found, not that the finding is right, and the search cannot be restricted to chosen
domains - the prompt's preference for veterinary and welfare sources is a preference, not
a filter. The UI says both.

Off by default (`PUPCHAT_GROUNDING=1` to enable): a grounded request can cost more per
answer than a plain one, and that is the owner's decision to make knowingly.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse


def _get(obj, name, default=None):
    """Field access that works on both the SDK's objects and plain dicts.

    Tests use dicts so they never need the SDK or a network; production gets pydantic
    models. Writing this once is cheaper than mirroring the whole object tree.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def domain_of(uri: str) -> str:
    try:
        host = urlparse(uri).netloc.lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def extract(response) -> tuple[list[dict], str | None]:
    """(sources, search_html) from a provider response.

    Returns an empty list when the model chose not to search, which is common and not an
    error - the UI then shows no Sources section at all rather than "no sources found".
    """
    candidates = _get(response, "candidates") or []
    if not candidates:
        return [], None
    meta = _get(candidates[0], "grounding_metadata")
    if meta is None:
        return [], None

    sources: list[dict] = []
    for i, chunk in enumerate(_get(meta, "grounding_chunks") or []):
        web = _get(chunk, "web")
        uri = _get(web, "uri")
        if not uri:
            continue                      # a retrieved-context chunk, not a web page
        title = _get(web, "title") or ""
        # `domain` is supplied by the API, which matters: the uri is a redirect through
        # vertexaisearch.cloud.google.com, so parsing it would name Google every time.
        dom = _get(web, "domain") or domain_of(uri)
        sources.append({"n": len(sources) + 1, "index": i, "title": title or dom or uri,
                        "uri": uri, "domain": dom})

    search_html = _get(_get(meta, "search_entry_point"), "rendered_content")
    return sources, search_html


def mark(text: str, response, sources: list[dict] | None = None) -> str:
    """Insert [1]-style markers where the API says each source was used.

    Offsets are **byte** offsets into the UTF-8 encoding of the answer, not character
    positions. On an answer containing a single curly quote the two diverge, and every
    marker after it lands mid-word - so the work is done on bytes and decoded once at the
    end.
    """
    if not text:
        return text
    candidates = _get(response, "candidates") or []
    meta = _get(candidates[0], "grounding_metadata") if candidates else None
    supports = _get(meta, "grounding_supports") or []
    if not supports:
        return text

    by_index = {s["index"]: s["n"] for s in (sources or [])}
    raw = text.encode("utf-8")

    # Group by position: two supports can end at the same place, and inserting twice
    # would give "[1][2]" as separate runs rather than one tidy marker.
    at: dict[int, list[int]] = {}
    for support in supports:
        end = _get(_get(support, "segment"), "end_index")
        if end is None:
            continue
        end = max(0, min(int(end), len(raw)))
        for idx in _get(support, "grounding_chunk_indices") or []:
            n = by_index.get(int(idx))
            if n and n not in at.setdefault(end, []):
                at[end].append(n)

    if not at:
        return text
    out = bytearray()
    cursor = 0
    for end in sorted(at):
        out += raw[cursor:end]
        out += ("".join(f"[{n}]" for n in sorted(at[end]))).encode("utf-8")
        cursor = end
    out += raw[cursor:]
    return out.decode("utf-8", "replace")


CITATION_SHAPED = re.compile(
    r"\bet al\.?"
    r"|\(\s*[A-Z][A-Za-z'’-]+(?:\s*(?:&|and|,)\s*[A-Z][A-Za-z'’-]+)*,?\s*(?:19|20)\d\d\s*\)"
    r"|\bdoi\s*:\s*\S+"
    r"|\bhttps?://\S+"
    r"|\bJournal of [A-Z]",
    re.IGNORECASE,
)


def citation_shaped(text: str) -> list[str]:
    """Text in the answer body that looks like a citation the model wrote itself.

    The model is told to write none, so anything matching here is a rule being broken -
    and the one failure mode that matters, because a reader cannot check it. Reported so
    it is visible rather than silently trusted.
    """
    return [m.group(0) for m in CITATION_SHAPED.finditer(text or "")]
