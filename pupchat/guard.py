"""Guard rails for a Pupchat that strangers can reach.

Locally none of this matters: one person, one clip at a time, their own key, their own
disk. Over a tunnel every one of those assumptions breaks, and the four that break
expensively are handled here.

**Spend.** A measured ask costs about $0.0048 (1,874 in / 896 out on gemini-3.8-flash).
One visitor with a loop is therefore a real bill, so there is a per-IP daily cap AND a
global one. The global cap is the load-bearing half: per-IP counting can be defeated by
anyone willing to forge a header (see `client_ip`), and a cap that can be defeated is
not a budget. At the default 300 asks/day the worst case is about $1.44.

**Memory.** The size limit was already enforced, but only after `await file.read()` had
pulled the whole body into RAM - a 10 GB POST would be a 10 GB allocation before the
413. `read_capped` streams and stops at the limit.

**The event loop.** `analyse()` is ~93 s of blocking CPU for a 7.5 s clip (measured,
12 cores). Called directly from an `async def` it blocks the entire loop, so while one
visitor is measured nobody else can even load the page. It now runs in a worker thread
behind a semaphore of one, which keeps the server responsive and makes the wait
reportable instead of silent.

**Other people's videos.** They land on the host's disk. In share mode they are swept
after a day; the page says so.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from starlette.concurrency import run_in_threadpool

CLIENT_IP_HEADERS = ("cf-connecting-ip", "x-forwarded-for", "x-real-ip")
"""Checked in order. A Cloudflare tunnel terminates locally, so every request arrives
from 127.0.0.1 and `request.client.host` identifies the tunnel, not the visitor."""

LOCAL_IPS = {"127.0.0.1", "::1", "localhost", ""}


def client_ip(request) -> str:
    """The visitor's address, as well as it can be known.

    Deliberately trusting a header, which is normally wrong. Behind a tunnel it is the
    only source of the visitor's address, and the honest limitation is that a visitor
    can forge it and so land in someone else's bucket - or a fresh one per request.
    That is why the global cap exists and why the per-IP cap is a courtesy rather than
    a control: it stops ordinary over-use, not a determined abuser.
    """
    headers = getattr(request, "headers", {}) or {}
    for name in CLIENT_IP_HEADERS:
        raw = headers.get(name)
        if raw:
            # X-Forwarded-For is a chain, client first.
            return raw.split(",")[0].strip()
    client = getattr(request, "client", None)
    return getattr(client, "host", "") or ""


def is_local(ip: str) -> bool:
    """Localhost is exempt from every cap.

    So running Pupchat on your own machine behaves exactly as it did before this
    module existed - no counters, no 429s - while tunnel traffic, which arrives with a
    forwarding header, is counted.
    """
    return ip in LOCAL_IPS


class BudgetExceeded(Exception):
    def __init__(self, message: str, retry_after_s: int = 3600):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class Budget:
    """Daily counters, per IP and in total, persisted so a restart is not a reset.

    Kept in one small JSON file rather than a database because the whole point is to
    be a demo. The counters are keyed by UTC date and the file is rewritten on each
    increment; at demo volumes that is a few hundred writes a day.
    """

    def __init__(self, path: Path, cfg):
        self.path = Path(path)
        self.cfg = cfg
        self._lock = threading.Lock()

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load(self) -> dict:
        today = self._today()
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        if state.get("date") != today:
            state = {"date": today, "ip": {}, "total": {}}
        state.setdefault("ip", {})
        state.setdefault("total", {})
        return state

    def _save(self, state: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except OSError:
            pass          # a demo must not 500 because a counter could not be written

    def limits(self, kind: str) -> tuple[int, int]:
        c = self.cfg
        if kind == "ask":
            return c.asks_per_ip_per_day, c.total_asks_per_day
        return c.analyses_per_ip_per_day, c.total_analyses_per_day

    def spend(self, kind: str, ip: str) -> None:
        """Count one use, or raise BudgetExceeded without counting it."""
        if is_local(ip):
            return
        per_ip, total = self.limits(kind)
        noun = "question" if kind == "ask" else "video"
        with self._lock:
            state = self._load()
            used_ip = int(state["ip"].get(ip, {}).get(kind, 0))
            used_all = int(state["total"].get(kind, 0))
            if used_ip >= per_ip:
                raise BudgetExceeded(
                    f"this demo allows {per_ip} {noun}s per visitor per day, and you "
                    f"have used {used_ip}. It resets at midnight UTC.")
            if used_all >= total:
                raise BudgetExceeded(
                    f"this demo has answered its {total} {noun}s for today - it runs "
                    f"on one person's home machine and their own API key. Try "
                    f"tomorrow; it resets at midnight UTC.")
            state["ip"].setdefault(ip, {})[kind] = used_ip + 1
            state["total"][kind] = used_all + 1
            self._save(state)

    def snapshot(self) -> dict:
        with self._lock:
            state = self._load()
        return {"date": state["date"], "total": state["total"],
                "visitors": len(state["ip"])}


class QueueFull(Exception):
    pass


class AnalysisQueue:
    """One analysis at a time, in a worker thread, with a bounded waiting room.

    Bounded because an unbounded queue is a worse experience than a refusal: the
    tenth person in line waits a quarter of an hour for a page that never said so.
    Being told "six people ahead of you, come back shortly" is honest and instant.
    """

    def __init__(self, max_waiting: int):
        self.max_waiting = max_waiting
        self._lock = threading.Lock()
        self._depth = 0
        self._busy = threading.Semaphore(1)

    @property
    def depth(self) -> int:
        """How many analyses are in the system, running one included."""
        with self._lock:
            return self._depth

    async def run(self, fn, *args, **kwargs):
        with self._lock:
            if self._depth > self.max_waiting:
                raise QueueFull(self._depth)
            self._depth += 1
        try:
            # Acquire in the thread pool too: a blocking semaphore held across an
            # await would pin the event loop, which is the problem this class exists
            # to solve.
            await run_in_threadpool(self._busy.acquire)
            try:
                return await run_in_threadpool(fn, *args, **kwargs)
            finally:
                self._busy.release()
        finally:
            with self._lock:
                self._depth -= 1


async def read_capped(upload, max_bytes: int) -> bytes:
    """Read an upload, refusing it as soon as it passes the limit.

    `await upload.read()` with a size check afterwards still allocates the whole body
    first, so the check protects the pipeline and not the host. Reading in chunks makes
    the limit mean what it says.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(
                f"that file is over the {max_bytes / 1e6:.0f} MB limit. A 5-10 second "
                f"clip is normally well under it - check it is not 4K.")
        chunks.append(chunk)
    return b"".join(chunks)


def sweep(uploads: Path, out_dir: Path, older_than_hours: float) -> list[str]:
    """Delete uploaded clips past the retention window, and their analyses.

    Returns the hashes removed. Only ever touches files under `uploads` and cache
    directories whose name is exactly an upload's hash, so nothing else under
    outputs/ is at risk - the keypoint caches are the expensive artefact and deleting
    one by accident costs 93 seconds of CPU to rebuild.
    """
    uploads, out_dir = Path(uploads), Path(out_dir)
    if not uploads.exists():
        return []
    cutoff = time.time() - older_than_hours * 3600
    removed: list[str] = []
    for clip in sorted(uploads.glob("*")):
        if not clip.is_file():
            continue
        try:
            if clip.stat().st_mtime >= cutoff:
                continue
            vhash = clip.stem
            clip.unlink()
            cache = out_dir / vhash
            if cache.is_dir() and vhash:
                shutil.rmtree(cache, ignore_errors=True)
            removed.append(vhash)
        except OSError:
            continue
    return removed
