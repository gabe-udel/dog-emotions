"""The answer rules from prompts/system.md, checked as code.

These run against fixtures, never the API: no key, no network, no spend. The same
functions run against live answers, so what is asserted here and what is measured on a
real call cannot drift apart.

What they do NOT check is whether the advice is good. A warm, confident, wrong answer
passes every test in this file.
"""
from __future__ import annotations

import re

import pytest

from pupchat.answer_check import (SECTIONS, aversive_advice, check, leaked_jargon,
                                  looks_like_new_video_answer, raw_degrees,
                                  referral_report, section_positions,
                                  sections_in_order)
from pupchat.client import system_prompt

NEW_VIDEO_ANSWER = """**What I see**
Your dog's ears are pinned flat back against his head and his lips are pulled back.
His body is held still and tight.

**What it likely means**
He likely feels uneasy about something nearby and is asking for space. He may also be
startled rather than frightened, though the pinned ears make discomfort more likely.

**Do**
Give him room and let him move away on his own.
Keep your voice low and calm.
Let him come to you instead of reaching for him.

**Don't**
Don't lean over him, it can feel threatening.
Don't scold the snarl, it is his warning system.
Don't pull him toward the thing that worries him.

**Try this next**
1. Stand a few steps further away than where he reacts.
2. Drop a treat on the floor and let him eat it.
3. Repeat for two minutes, once a day.

If this keeps happening, a certified behaviorist can help you work through it.
"""

FOLLOW_UP_ANSWER = """Not right now. He's asking for space, so reaching for him could
push him past what he can cope with. Wait until his ears come forward and his mouth
softens, then let him approach you first.
"""

BAD_NUMBERS = """**What I see**
His ears are angled 110° back and sit 0.11 face-widths below eye level.
"""

BAD_EARLY_REFERRAL = """Because this involves bite risk, please consult a certified
behaviorist straight away.

**What I see**
His ears are pinned back.
"""

BAD_TWO_REFERRALS = """He looks uneasy. A veterinarian should check him over.
Give him space and keep things calm.
A certified behaviorist can also help with this.
"""

BAD_AVERSIVE = """**Do**
Use a prong collar to correct the behaviour when he growls.
"""


# ------------------------------------------------------- five sections, new video

def test_a_new_video_answer_has_all_five_headings():
    assert looks_like_new_video_answer(NEW_VIDEO_ANSWER)
    assert set(section_positions(NEW_VIDEO_ANSWER)) == set(SECTIONS)


def test_the_headings_must_be_in_order():
    scrambled = "\n".join([
        "**Try this next**", "1. do a thing",
        "**What I see**", "ears back",
        "**What it likely means**", "unease",
        "**Do**", "give space",
        "**Don't**", "no looming"])
    assert not sections_in_order(scrambled)
    assert not looks_like_new_video_answer(scrambled)


def test_a_missing_heading_fails():
    without_do = NEW_VIDEO_ANSWER.replace("**Do**", "")
    assert not looks_like_new_video_answer(without_do)
    assert "Do" in check(without_do, new_video=True)["missing_sections"]


@pytest.mark.parametrize("style", [
    "**{}**", "## {}", "{}:", "__{}__", "### {} :",
])
def test_heading_decoration_does_not_matter(style):
    """The rule is that the section exists and is in order, not how it is decorated."""
    body = "\n\n".join(f"{style.format(name)}\nsome text here." for name in SECTIONS)
    assert looks_like_new_video_answer(body), style


# ------------------------------------------------------- follow-ups

def test_a_follow_up_does_not_repeat_the_five_headings():
    assert not looks_like_new_video_answer(FOLLOW_UP_ANSWER)
    assert check(FOLLOW_UP_ANSWER, new_video=False)["format_ok"]


def test_a_follow_up_that_repeats_the_format_fails():
    assert not check(NEW_VIDEO_ANSWER, new_video=False)["format_ok"]


def test_a_new_video_answer_without_the_format_fails():
    assert not check(FOLLOW_UP_ANSWER, new_video=True)["format_ok"]


# ------------------------------------------------------- no raw measurements

def test_raw_degree_values_are_caught():
    assert raw_degrees("his ears are angled 110° back")
    assert raw_degrees("angled 110 degrees back")
    assert raw_degrees("tilted -45.5° to the side")


def test_plain_language_passes():
    assert raw_degrees("his ears are pinned flat back against his head") == []
    assert raw_degrees(NEW_VIDEO_ANSWER) == []
    assert raw_degrees(FOLLOW_UP_ANSWER) == []


def test_measurement_jargon_is_caught_too():
    assert leaked_jargon("0.11 face-widths below eye level")
    assert leaked_jargon("the keypoints were unreliable")
    assert leaked_jargon("coverage was only 44%")
    assert leaked_jargon(NEW_VIDEO_ANSWER) == []


def test_the_bad_example_fails_the_whole_check():
    r = check(BAD_NUMBERS, new_video=True)
    assert r["raw_degrees"] and not r["passed"]


# ------------------------------------------------------- referrals

def test_one_referral_at_the_end_is_allowed():
    r = referral_report(NEW_VIDEO_ANSWER)
    assert r["count"] == 1 and r["at_most_one"] and r["at_end"]


def test_no_referral_at_all_is_allowed():
    r = referral_report(FOLLOW_UP_ANSWER)
    assert r["count"] == 0 and r["at_most_one"] and r["at_end"]


def test_leading_with_a_referral_fails():
    """The behaviour the rewrite was meant to stop."""
    r = referral_report(BAD_EARLY_REFERRAL)
    assert r["count"] == 1
    assert not r["at_end"]
    assert not check(BAD_EARLY_REFERRAL, new_video=True)["passed"]


def test_two_referrals_fail():
    r = referral_report(BAD_TWO_REFERRALS)
    assert r["count"] == 2 and not r["at_most_one"]


@pytest.mark.parametrize("word", ["vet", "veterinarian", "behaviorist",
                                  "behaviourist", "certified trainer"])
def test_every_referral_wording_is_detected(word):
    assert referral_report(f"Give him space. Then see a {word} about it.")["count"] == 1


# ------------------------------------------------------- training advice

def test_aversive_tools_are_caught():
    for bad in ("prong collar", "shock collar", "choke chain", "alpha roll"):
        assert aversive_advice(f"try a {bad} for this"), bad
    assert not check(BAD_AVERSIVE, new_video=True)["passed"]


def test_reward_based_advice_passes():
    assert aversive_advice(NEW_VIDEO_ANSWER) == []


@pytest.mark.parametrize("sentence", [
    "Don't scold or punish him for showing teeth.",
    "Never use a prong collar on a fearful dog.",
    "Avoid choke chains entirely.",
    "Reward him instead of punishing the growl.",
    "Build confidence rather than using a shock collar.",
])
def test_advising_against_an_aversive_is_not_flagged(sentence):
    """The prompt's whole Don't section is built from sentences like these. A
    keyword match with no sense of negation flags the correct advice as the error."""
    assert aversive_advice(sentence) == [], sentence


@pytest.mark.parametrize("sentence", [
    "Use a prong collar to correct the behaviour.",
    "A quick leash correction with a choke chain will stop it.",
    "You should punish him when he growls.",
])
def test_actually_recommending_an_aversive_is_still_flagged(sentence):
    assert aversive_advice(sentence), sentence


# ------------------------------------------------------- the prompt itself

def test_the_prompt_states_every_section_heading():
    p = system_prompt()
    for name in SECTIONS:
        assert name in p, name


def test_the_prompt_forbids_raw_numbers_and_aversives():
    p = system_prompt().lower()
    assert "never quote raw numbers" in p
    assert "shock" in p and "prong" in p and "choke" in p


def _flat(text: str) -> str:
    """Lowercase, markdown emphasis removed, whitespace collapsed.

    The prompt is a wrapped markdown file, so a rule can be split across a line break
    or carry ** around part of it. The test is about the rule being present, not about
    where the author happened to wrap the line."""
    return re.sub(r"\s+", " ", re.sub(r"\*\*|__", "", text)).lower()


def test_the_prompt_keeps_the_rules_the_rewrite_did_not_cover():
    """Carried over from the previous prompt because the new text does not mention
    them, and each one guards against a specific way of overclaiming."""
    p = _flat(system_prompt())
    assert "measured against ground truth" in p
    assert "reliability estimated, not yet verified" in p
    assert "no information, not an absence" in p
    assert "never invent a moment" in p
    assert "never diagnose" in p


def test_the_prompt_puts_the_referral_at_the_end():
    p = system_prompt().lower()
    assert "put the advice first" in p
    assert "don't lead with a referral" in p


def test_the_full_check_passes_on_a_good_answer():
    assert check(NEW_VIDEO_ANSWER, new_video=True)["passed"]
    assert check(FOLLOW_UP_ANSWER, new_video=False)["passed"]
