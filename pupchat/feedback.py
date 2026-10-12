"""Flagged answers: what was asked, what came back, and why it was wrong.

The point of the flag button is to build a corpus of failures that can be re-run later
against a changed prompt, a changed model or a changed feature set. That only works if
the exact inputs are kept, so the clip is COPIED rather than referenced - a path into
`videos/` is worthless once the file is renamed, and worse than worthless if it is
replaced by different footage under the same name.

Stored next to the evidence that produced the answer, keyed by the same video hash the
rest of the pipeline uses.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .config import OUT_DIR

FLAG_DIR = OUT_DIR / "flagged"
FLAG_LOG = FLAG_DIR / "flags.jsonl"


def flag_answer(video: str | Path, question: str, answer: str, note: str, *,
                evidence_text: str = "", provider: str = "", model: str = "",
                video_hash: str = "", copy_clip: bool = True,
                flag_dir: Path | None = None) -> dict:
    """Append one flagged answer. Returns the record written."""
    flag_dir = Path(flag_dir) if flag_dir is not None else FLAG_DIR
    flag_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    video = Path(video)
    record = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "id": f"{stamp}-{video_hash or video.stem}",
        "video_name": video.name,
        "video_hash": video_hash,
        "question": question,
        "answer": answer,
        "note": note,
        "evidence_text": evidence_text,
        "provider": provider,
        "model": model,
    }

    if copy_clip and video.exists():
        dest = flag_dir / f"{record['id']}{video.suffix}"
        if not dest.exists():
            shutil.copy2(video, dest)
        record["clip"] = str(dest.relative_to(flag_dir.parent.parent)) \
            if flag_dir.is_relative_to(OUT_DIR.parent) else str(dest)
    else:
        record["clip"] = None
        record["clip_missing_reason"] = ("copy disabled" if not copy_clip
                                         else f"{video} not found")

    (flag_dir / f"{record['id']}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    with (flag_dir / "flags.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def load_flags(flag_dir: Path | None = None) -> list[dict]:
    flag_dir = Path(flag_dir) if flag_dir is not None else FLAG_DIR
    log = flag_dir / "flags.jsonl"
    if not log.exists():
        return []
    return [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
