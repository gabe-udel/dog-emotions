"""Flagged answers.

The corpus only has value if a flagged case can be re-run later, which means the exact
inputs must survive. A path into videos/ does not: rename the file and the record is
worthless, replace it and the record is actively misleading.
"""
from __future__ import annotations

import json

import pytest

from pupchat.feedback import flag_answer, load_flags


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "dog.mp4"
    p.write_bytes(b"\x00" * 512)
    return p


def test_flag_writes_a_record(tmp_path, clip):
    rec = flag_answer(clip, "What is he doing?", "He is wagging.", "wrong, tail is down",
                      evidence_text="tail not measurable", provider="gemini",
                      model="gemini-3.8-flash", video_hash="abc123",
                      flag_dir=tmp_path / "flagged")
    assert rec["question"] == "What is he doing?"
    assert rec["answer"] == "He is wagging."
    assert rec["note"] == "wrong, tail is down"
    assert rec["model"] == "gemini-3.8-flash"


def test_the_clip_is_copied_not_referenced(tmp_path, clip):
    """So renaming or replacing the original cannot corrupt the record."""
    d = tmp_path / "flagged"
    rec = flag_answer(clip, "q", "a", "n", video_hash="abc123", flag_dir=d)
    copies = list(d.glob("*.mp4"))
    assert len(copies) == 1
    assert copies[0].read_bytes() == clip.read_bytes()
    clip.unlink()
    assert copies[0].exists()


def test_evidence_is_saved_with_the_answer(tmp_path, clip):
    """Re-running a flagged case needs what the model was told, not just what it said."""
    d = tmp_path / "flagged"
    flag_answer(clip, "q", "a", "n", evidence_text="ear tips below eye level",
                video_hash="h", flag_dir=d)
    assert "ear tips below eye level" in load_flags(d)[0]["evidence_text"]


def test_flags_accumulate_one_json_object_per_line(tmp_path, clip):
    d = tmp_path / "flagged"
    flag_answer(clip, "q1", "a1", "n1", video_hash="h", flag_dir=d)
    flag_answer(clip, "q2", "a2", "n2", video_hash="h", flag_dir=d)
    rows = load_flags(d)
    assert [r["question"] for r in rows] == ["q1", "q2"]
    for line in (d / "flags.jsonl").read_text().strip().split("\n"):
        json.loads(line)


def test_each_flag_also_gets_its_own_file(tmp_path, clip):
    d = tmp_path / "flagged"
    rec = flag_answer(clip, "q", "a", "n", video_hash="h", flag_dir=d)
    assert (d / f"{rec['id']}.json").exists()


def test_an_empty_note_is_allowed(tmp_path, clip):
    """Flagging without typing a reason still records the case; forcing a note would
    cost the flag."""
    rec = flag_answer(clip, "q", "a", "", video_hash="h", flag_dir=tmp_path / "f")
    assert rec["note"] == ""


def test_a_missing_clip_is_recorded_as_missing_not_silently_dropped(tmp_path):
    d = tmp_path / "flagged"
    rec = flag_answer(tmp_path / "gone.mp4", "q", "a", "n", flag_dir=d)
    assert rec["clip"] is None
    assert "not found" in rec["clip_missing_reason"]


def test_load_flags_on_an_empty_directory(tmp_path):
    assert load_flags(tmp_path / "nothing") == []
