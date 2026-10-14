"""Checks on an answer's shape, so the prompt's rules can be tested without the API.

These are pure functions over a string. The tests run them against fixtures - no key, no
network, no spend - and the CLI runs the same functions against live answers, so what is
asserted offline and what is measured online are the same code.

What they cannot check is whether the advice is any good. These catch format drift and
leaked measurement jargon; a kind, wrong answer passes every one of them.
"""
from __future__ import annotations

import re

from .sources import citation_shaped

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
    r"shouldn'?t|should not|not recommend\w*|stay away from|"
    # Added when the feelings check went in: "he doesn't look anxious" is a description
    # of what is NOT there. The original list had "don't" but not "doesn't", which also
    # meant "he doesn't need a prong collar" was being read as recommending one.
    r"doesn'?t|does not|isn'?t|is not|wasn'?t|was not|aren'?t|are not|"
    r"hasn'?t|has not|no sign of|nothing like)\b", re.I)


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


CORE = ("What I see", "What it likely means", "Do")
"""What a new-video answer should cover in substance.

"Don't" and "Try this next" are good structure, not a safety rule. A two-paragraph answer
to "is this normal?" is better without them, and failing it for that was the check
enforcing a format rather than a standard.
"""

ANSWER_BANNED = (
    "happy", "happiness", "sad", "sadness", "angry", "anger", "anxious", "anxiety",
    "stressed", "stress", "fearful", "afraid", "scared", "frightened", "frustrated",
    "frustration", "aggressive", "aggression", "depressed", "distressed",
    "pain", "painful", "injured", "injury", "sick", "illness",
)
"""Feelings and clinical claims, banned inside "What I see" only.

Deliberately NARROWER than config.BANNED_WORDS. That list is built for the evidence
layer, where no interpretation is allowed at all, so it also bans "relaxed", "tense" and
"alert" as interpretation in body-language clothes. In the model's ANSWER those describe
muscle tone, which is body - and interpreting is explicitly the model's job. Failing "his
mouth was relaxed" would be this check fighting the prompt it is meant to enforce.

What stays banned there is the split the prompt asks for: describe the body in "What I
see", say what it means in the next section.
"""


def banned_in_what_i_see(text: str) -> list[str]:
    """Feeling words inside the description section.

    Negation-aware, reusing the same sentence walk as `aversive_advice`: "he doesn't look
    anxious" is a description of what is absent, not a conclusion being smuggled in.
    """
    spans = section_positions(text)
    if "What I see" not in spans:
        return []
    start = spans["What I see"]
    later = [p for s, p in spans.items() if p > start]
    end = min(later) if later else len(text)

    found: list[str] = []
    for _, sentence in _sentences(text[start:end]):
        low = sentence.lower()
        for word in ANSWER_BANNED:
            if re.search(rf"\b{re.escape(word)}\b", low):
                before = low[: low.index(word)]
                if not _NEGATION.search(before):
                    found.append(word)
    return sorted(set(found))


def word_count(text: str) -> int:
    return len((text or "").split())


def check(text: str, *, new_video: bool) -> dict:
    """Every rule at once. `new_video` selects the format expectation.

    The five headings stopped being a pass condition when the prompt started allowing a
    different shape - "use fewer headings, or differently named ones, when the question
    calls for it". They are still reported, because the shape is worth seeing; they are
    no longer a failure. What a follow-up must still not do is re-run the whole format.
    """
    ref = referral_report(text)
    five = looks_like_new_video_answer(text)
    spans = section_positions(text)
    words = word_count(text)
    too_long = words > (450 if new_video else 220)
    invented = citation_shaped(text)
    feelings = banned_in_what_i_see(text)
    return {
        # ---- described, never a failure
        "five_sections": five,
        "format_ok": five if new_video else not five,
        "missing_sections": [s for s in SECTIONS if s not in spans],
        "core_covered": [s for s in CORE if s in spans],
        "word_count": words,
        "too_long": too_long,
        "leaked_jargon": leaked_jargon(text),
        # ---- the rules that still fail
        "raw_degrees": raw_degrees(text),
        "aversive_advice": aversive_advice(text),
        "citation_shaped": invented,
        "feelings_in_what_i_see": feelings,
        "referral": ref,
        "passed": (
            bool((text or "").strip())
            and not raw_degrees(text)
            and not aversive_advice(text)
            # The model is told to write no citations at all, so one it wrote itself is
            # the single failure a reader cannot check for themselves.
            and not invented
            and not feelings
            and ref["at_most_one"] and ref["at_end"]
            and (True if new_video else not five)
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
