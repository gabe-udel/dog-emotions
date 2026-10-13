"""Checks on an answer's shape, so the prompt's rules can be tested without the API.

These are pure functions over a string. The tests run them against fixtures - no key, no
network, no spend - and the CLI runs the same functions against live answers, so what is
asserted offline and what is measured online are the same code.

What they cannot check is whether the advice is any good. These catch format drift and
leaked measurement jargon; a kind, wrong answer passes every one of them.
"""
from __future__ import annotations

import re

SECTIONS = ("What I see", "What it likely means", "Do", "Don't", "Try this next")

# A heading may arrive bolded, as a markdown header, or bare with a colon. The rule
# being tested is that the section is present and in order, not its decoration.
# Built by concatenation rather than str.format: the pattern contains `{1,4}`, which
# format() reads as a replacement field.
def _heading_re(name: str) -> re.Pattern:
    return re.compile(
        r"(?:^|\n)[ \t]*(?:#{1,4}[ \t]*)?(?:\*\*|__)?[ \t]*"
        + re.escape(name)
        + r"[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*(?:\n|$)", re.I)

PROFESSIONAL = re.compile(
    r"\b(veterinar(?:y|ian|ians)|vet|behaviou?rist|behaviou?rists|"
    r"certified trainer|professional trainer)\b", re.I)

# "110°", "110 degrees", "-0.11 face-widths": numbers straight out of the evidence.
DEGREES = re.compile(r"-?\d+(?:\.\d+)?\s*(?:°|º|degrees?\b)", re.I)
JARGON = re.compile(
    r"-?\d+(?:\.\d+)?\s*(?:face-?widths?|body-?lengths?|px|pixels?|Hz)\b"
    r"|\bSNR\b|\bnoise floor\b|\bcoverage\b|\bwithin-clip\b|\bkeypoints?\b", re.I)

AVERSIVE = re.compile(
    r"\b(shock collar|e-?collar|prong collar|choke (?:chain|collar)|"
    r"alpha roll|dominance|scruff|punish(?:ment)?|correction collar)\b", re.I)


def section_positions(text: str) -> dict[str, int]:
    """Character offset of each section heading found, or absent from the dict."""
    out: dict[str, int] = {}
    for name in SECTIONS:
        m = _heading_re(name).search(text)
        if m:
            out[name] = m.start()
    return out


def has_all_sections(text: str) -> bool:
    return len(section_positions(text)) == len(SECTIONS)


def sections_in_order(text: str) -> bool:
    pos = section_positions(text)
    if len(pos) != len(SECTIONS):
        return False
    offsets = [pos[name] for name in SECTIONS]
    return offsets == sorted(offsets)


def looks_like_new_video_answer(text: str) -> bool:
    """The five-section format. A follow-up must NOT look like this."""
    return has_all_sections(text) and sections_in_order(text)


def raw_degrees(text: str) -> list[str]:
    return [m.group(0) for m in DEGREES.finditer(text)]


def leaked_jargon(text: str) -> list[str]:
    return [m.group(0) for m in JARGON.finditer(text)]


# "Don't punish him for growling" is the advice we WANT. A plain keyword match flags it
# as aversive, which is backwards - the prompt's whole "Don't" section is built from
# sentences like that. So a hit only counts when it is not already being warned against.
_NEGATION = re.compile(
    r"\b(don'?t|do not|never|avoid|instead of|rather than|without|no need to|"
    r"shouldn'?t|should not|not recommend\w*|stay away from)\b", re.I)


def aversive_advice(text: str) -> list[str]:
    """Aversive methods being RECOMMENDED. Mentions that advise against them pass."""
    out = []
    for start, sentence in _sentences(text):
        for m in AVERSIVE.finditer(sentence):
            before = sentence[:m.start()]
            if _NEGATION.search(before):
                continue            # "don't punish ..." - the right advice
            out.append(m.group(0))
    return out


def _sentences(text: str) -> list[tuple[int, str]]:
    out, start = [], 0
    for m in re.finditer(r"[.!?](?:\s|$)|\n", text):
        chunk = text[start:m.end()].strip()
        if chunk:
            out.append((start, chunk))
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append((start, tail))
    return out


def referral_sentences(text: str) -> list[tuple[int, str]]:
    return [(i, s) for i, s in _sentences(text) if PROFESSIONAL.search(s)]


def referral_report(text: str, tail_fraction: float = 0.25) -> dict:
    """How many referrals, and whether the one allowed sits at the end.

    `tail_fraction` of the answer counts as "the end". A referral in the opening
    paragraph is what the rewrite was meant to stop.
    """
    found = referral_sentences(text)
    n = len(text) or 1
    cutoff = n * (1.0 - tail_fraction)
    return {
        "count": len(found),
        "at_most_one": len(found) <= 1,
        "at_end": all(i >= cutoff for i, _ in found) if found else True,
        "sentences": [s for _, s in found],
        "positions": [round(i / n, 3) for i, _ in found],
    }


def check(text: str, *, new_video: bool) -> dict:
    """Every rule at once. `new_video` selects the format expectation."""
    ref = referral_report(text)
    five = looks_like_new_video_answer(text)
    return {
        "five_sections": five,
        "format_ok": five if new_video else not five,
        "missing_sections": [s for s in SECTIONS if s not in section_positions(text)],
        "raw_degrees": raw_degrees(text),
        "leaked_jargon": leaked_jargon(text),
        "aversive_advice": aversive_advice(text),
        "referral": ref,
        "passed": (
            (five if new_video else not five)
            and not raw_degrees(text)
            and not aversive_advice(text)
            and ref["at_most_one"] and ref["at_end"]
        ),
    }


def render(result: dict, label: str = "") -> str:
    tick = lambda b: "PASS" if b else "FAIL"            # noqa: E731
    r = result["referral"]
    lines = [f"ANSWER CHECK {label}".strip(),
             f"  overall                 {tick(result['passed'])}",
             f"  five-section format     {tick(result['format_ok'])}"
             + (f"   missing: {result['missing_sections']}"
                if result["missing_sections"] and result["five_sections"] is False
                and not result["format_ok"] else ""),
             f"  no raw degrees          {tick(not result['raw_degrees'])}"
             + (f"   found: {result['raw_degrees']}" if result["raw_degrees"] else ""),
             f"  referral at most once   {tick(r['at_most_one'])}   count {r['count']}",
             f"  referral at the end     {tick(r['at_end'])}"
             + (f"   position {r['positions']}" if r["positions"] else ""),
             f"  no aversive advice      {tick(not result['aversive_advice'])}",
             f"  no leaked jargon        {tick(not result['leaked_jargon'])}"
             + (f"   found: {result['leaked_jargon']}"
                if result["leaked_jargon"] else "")]
    return "\n".join(lines)
