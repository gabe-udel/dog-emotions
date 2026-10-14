"""Dogs and chats, on this machine's disk.

The front end used to keep both in localStorage, which made them per-browser: a phone
could never see them, and clearing site data threw them away. They live here instead.

One JSON file, held in memory, rewritten whole under one lock. A database would be the
right answer for a product and the wrong one for a demo whose entire dataset is a few
hundred kilobytes - the cost of being wrong here is one lost edit, not a corrupt index.

**Who may write.** There is no login. The owner is whoever is running the server, which
means localhost - plus any device holding the owner token, because the owner's phone
arrives through the Cloudflare tunnel and so is not local. A visitor to a shared demo can
upload a clip and ask about it, but sees no dogs and no chats, and nothing they do is
stored. Anonymous strangers writing records to someone's disk is unbounded growth and
content liability, and `guard.py` caps spend and CPU, not disk.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .config import OUT_DIR

LIB_DIR = OUT_DIR / "library"
LIB_FILE = LIB_DIR / "library.json"
BAK_FILE = LIB_DIR / "library.bak.json"
OWNER_FILE = LIB_DIR / "owner.json"
PHOTO_DIR = LIB_DIR / "photos"

VERSION = 1

DOG_FIELDS = ("name", "breed", "age", "health", "training", "triggers", "notes")
"""Exactly the seven the model is given. Adding a field here without adding it to
`profile_text` would put something on the page that never reaches the answer."""

SEARCH_CHARS = 2000
"""Per chat. Enough that searching finds a phrase from the middle of a conversation,
small enough that the sidebar listing stays cheap."""


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def slug(name: str) -> str:
    """A dog's name as it appears in a URL.

    Lossy on purpose: /dogs/<name> is for people to read and share, so the id stays the
    real key and this is only a lookup hint.
    """
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s or "dog"


def profile_text(dog: dict | None) -> str:
    """The dog, as prose, for the model.

    Empty fields are left out rather than sent as "Health: ". A blank label invites the
    model to guess whether it means healthy or unknown, and those are different.
    """
    if not dog:
        return ""
    bits = []
    if dog.get("name"):
        bits.append(f"The dog's name is {dog['name']}.")
    for key, label in (("breed", "Breed"), ("age", "Age"), ("health", "Health"),
                       ("training", "Training"), ("triggers", "Known triggers"),
                       ("notes", "Notes from the owner")):
        if dog.get(key):
            bits.append(f"{label}: {dog[key]}")
    return "\n".join(bits)


class Library:
    """The whole store. One instance per process; every method takes the lock."""

    def __init__(self, path: Path = LIB_FILE):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._state = self._load()

    # ------------------------------------------------------------------ disk
    def _blank(self) -> dict:
        return {"version": VERSION, "active_dog_id": None, "dogs": [], "chats": []}

    def _load(self) -> dict:
        """Current file, then the backup, then empty.

        A corrupt file must never stop the app starting. Coming up with no dogs is
        recoverable and obvious; refusing to start is neither.
        """
        for candidate in (self.path, self.path.with_name(BAK_FILE.name)):
            try:
                state = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(state, dict) and isinstance(state.get("dogs"), list):
                state.setdefault("chats", [])
                state.setdefault("active_dog_id", None)
                state.setdefault("version", VERSION)
                if candidate != self.path:
                    print(f"[library] {self.path.name} was unreadable; "
                          f"recovered from {candidate.name}")
                return state
            print(f"[library] ignoring {candidate.name}: not a library file")
        return self._blank()

    def _save(self) -> None:
        """Atomic: temp file in the SAME directory, fsync, os.replace.

        os.replace is atomic on Windows too (MoveFileEx with REPLACE_EXISTING), which a
        write-in-place is not: a crash halfway through would leave truncated JSON, and
        every dog and chat with it.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        if self.path.exists():
            try:
                shutil.copy2(self.path, self.path.with_name(BAK_FILE.name))
            except OSError:
                pass
        for attempt in range(4):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                # Antivirus and the search indexer hold brief handles on Windows.
                time.sleep(0.05 * (attempt + 1))
        os.replace(tmp, self.path)          # last try; let it raise

    # ------------------------------------------------------------------ owner
    def owner_token(self) -> str:
        """The secret in the QR link. Created on first use and kept out of git."""
        with self._lock:
            try:
                tok = json.loads(OWNER_FILE.read_text(encoding="utf-8"))["token"]
                if isinstance(tok, str) and len(tok) >= 20:
                    return tok
            except (OSError, ValueError, KeyError, TypeError):
                pass
            return self.rotate_token()

    def rotate_token(self) -> str:
        with self._lock:
            tok = secrets.token_urlsafe(32)
            OWNER_FILE.parent.mkdir(parents=True, exist_ok=True)
            OWNER_FILE.write_text(json.dumps({"token": tok}), encoding="utf-8")
            return tok

    # ------------------------------------------------------------------ dogs
    def dogs(self) -> list[dict]:
        with self._lock:
            return [dict(d) for d in self._state["dogs"]]

    def dog(self, key: str | None) -> dict | None:
        """By id, or by name-slug, so /dogs/<name> resolves without an id in the URL."""
        if not key:
            return None
        with self._lock:
            for d in self._state["dogs"]:
                if d["id"] == key:
                    return dict(d)
            for d in self._state["dogs"]:
                if slug(d.get("name", "")) == slug(key):
                    return dict(d)
        return None

    def add_dog(self, data: dict) -> dict:
        with self._lock:
            now = time.time()
            dog = {"id": _id("d"), "photo": None, "baselines": [],
                   "created": now, "updated": now}
            dog.update({k: str(data.get(k) or "") for k in DOG_FIELDS})
            self._state["dogs"].append(dog)
            if not self._state["active_dog_id"]:
                self._state["active_dog_id"] = dog["id"]
            self._save()
            return dict(dog)

    def update_dog(self, dog_id: str, data: dict) -> dict | None:
        with self._lock:
            for d in self._state["dogs"]:
                if d["id"] == dog_id:
                    for k in DOG_FIELDS:
                        if k in data:
                            d[k] = str(data[k] or "")
                    d["updated"] = time.time()
                    self._save()
                    return dict(d)
        return None

    def delete_dog(self, dog_id: str) -> int:
        """Remove a dog. Its chats stay, detached.

        Deleting the conversations too would be a surprise: the clips are measured and
        the answers are the useful part, and nothing says removing a profile means
        throwing those away.
        """
        with self._lock:
            before = len(self._state["dogs"])
            self._state["dogs"] = [d for d in self._state["dogs"] if d["id"] != dog_id]
            if before == len(self._state["dogs"]):
                return -1
            detached = 0
            for c in self._state["chats"]:
                if c.get("dog_id") == dog_id:
                    c["dog_id"] = None
                    detached += 1
            if self._state["active_dog_id"] == dog_id:
                self._state["active_dog_id"] = (
                    self._state["dogs"][0]["id"] if self._state["dogs"] else None)
            for p in PHOTO_DIR.glob(f"{dog_id}.*"):
                p.unlink(missing_ok=True)
            self._save()
            return detached

    def set_photo(self, dog_id: str, suffix: str) -> dict | None:
        with self._lock:
            for d in self._state["dogs"]:
                if d["id"] == dog_id:
                    d["photo"] = f"{dog_id}{suffix}"
                    d["updated"] = time.time()
                    self._save()
                    return dict(d)
        return None

    def photo_path(self, dog_id: str) -> Path | None:
        d = self.dog(dog_id)
        if not d or not d.get("photo"):
            return None
        p = PHOTO_DIR / d["photo"]
        return p if p.exists() else None

    def add_baseline(self, dog_id: str, entry: dict) -> dict | None:
        with self._lock:
            for d in self._state["dogs"]:
                if d["id"] == dog_id:
                    d.setdefault("baselines", [])
                    if not any(b["hash"] == entry["hash"] for b in d["baselines"]):
                        d["baselines"].append(entry)
                    d["updated"] = time.time()
                    self._save()
                    return dict(d)
        return None

    def remove_baseline(self, dog_id: str, vhash: str) -> dict | None:
        with self._lock:
            for d in self._state["dogs"]:
                if d["id"] == dog_id:
                    d["baselines"] = [b for b in d.get("baselines", [])
                                      if b["hash"] != vhash]
                    d["updated"] = time.time()
                    self._save()
                    return dict(d)
        return None

    def active_dog_id(self) -> str | None:
        with self._lock:
            return self._state["active_dog_id"]

    def set_active_dog(self, dog_id: str | None) -> None:
        with self._lock:
            self._state["active_dog_id"] = dog_id
            self._save()

    # ------------------------------------------------------------------ chats
    def chat_summaries(self) -> list[dict]:
        """Enough for the sidebar: no messages, no provider history.

        Those two are most of the bytes, and listing is the most frequent read.
        """
        with self._lock:
            return [{k: c.get(k) for k in ("id", "dog_id", "title", "hash",
                                           "video_name", "duration", "created",
                                           "updated", "search_text")}
                    | {"n_messages": len(c.get("messages", []))}
                    for c in sorted(self._state["chats"],
                                    key=lambda c: -(c.get("updated") or 0))]

    def chat(self, chat_id: str) -> dict | None:
        with self._lock:
            for c in self._state["chats"]:
                if c["id"] == chat_id:
                    return json.loads(json.dumps(c))
        return None

    def add_chat(self, dog_id: str | None = None, title: str = "") -> dict:
        with self._lock:
            now = time.time()
            chat = {"id": _id("c"), "dog_id": dog_id, "title": title or "New chat",
                    "hash": None, "video_name": None, "duration": None,
                    "history": [], "messages": [], "search_text": "",
                    "pending_job_id": None, "created": now, "updated": now}
            self._state["chats"].append(chat)
            self._save()
            return json.loads(json.dumps(chat))

    def update_chat(self, chat_id: str, **fields) -> dict | None:
        allowed = {"title", "dog_id", "hash", "video_name", "duration", "history",
                   "pending_job_id"}
        with self._lock:
            for c in self._state["chats"]:
                if c["id"] == chat_id:
                    for k, v in fields.items():
                        if k in allowed:
                            c[k] = v
                    c["updated"] = time.time()
                    self._save()
                    return json.loads(json.dumps(c))
        return None

    def append_message(self, chat_id: str, message: dict) -> dict | None:
        """Add one turn and keep the search text current.

        The server owns the transcript: it appends the question when the job starts and
        the answer when it finishes. One writer means a crash mid-answer leaves a
        visible failed turn rather than a silent gap.
        """
        with self._lock:
            for c in self._state["chats"]:
                if c["id"] != chat_id:
                    continue
                message = dict(message)
                message.setdefault("at", time.time())
                c.setdefault("messages", []).append(message)
                if c["title"] in ("", "New chat") and message.get("role") == "user":
                    c["title"] = (message.get("text") or "New chat")[:60]
                if message.get("role") in ("user", "assistant"):
                    c["search_text"] = (
                        f"{c.get('search_text', '')} {message.get('text', '')}"
                    ).strip()[-SEARCH_CHARS:]
                c["updated"] = time.time()
                self._save()
                return json.loads(json.dumps(c))
        return None

    def delete_chat(self, chat_id: str) -> bool:
        with self._lock:
            before = len(self._state["chats"])
            self._state["chats"] = [c for c in self._state["chats"]
                                    if c["id"] != chat_id]
            if before == len(self._state["chats"]):
                return False
            self._save()
            return True

    # ------------------------------------------------------------------ views
    def snapshot(self, *, owner: bool) -> dict[str, Any]:
        """What /api/library returns.

        A visitor gets the shape with nothing in it, so the page renders its empty
        states rather than breaking on missing keys - and never learns that this
        machine has dogs on it at all.
        """
        if not owner:
            return {"owner": False, "dogs": [], "chats": [], "active_dog_id": None}
        return {"owner": True, "dogs": self.dogs(), "chats": self.chat_summaries(),
                "active_dog_id": self.active_dog_id()}


_LIBRARY: Library | None = None


def library() -> Library:
    """The process-wide store. One instance, so one lock actually serialises writes."""
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = Library()
    return _LIBRARY
