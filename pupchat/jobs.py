"""Long work, reported while it runs.

Measuring a clip is about 93 seconds of CPU. The page used to send the request and wait,
which looks identical to the server having crashed - so people reload, and reloading
starts a second measurement behind the first.

Work now gets a job id. The page polls `GET /api/jobs/{id}` and shows named steps. Not
Server-Sent Events: Cloudflare quick tunnels do not support them, and this app is meant
to be shared through one.

The honesty problem this module exists to solve is that the expensive stage is silent.
`[3/4] body pose` is one blocking call over every frame and emits nothing until it ends,
so there is no true fraction to show for most of the wait. Rather than interpolate a bar
against the clock - which would be a fabricated measurement in an app whose whole point
is not fabricating measurements - it shows which step is running and how long it is
expected to take, and stops showing a number once that estimate is spent.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .config import OUT_DIR

TIMING_FILE = OUT_DIR / "library" / "timing.json"

STEPS = ("queued", "finding", "measuring", "asking", "done", "failed")

STEP_LABEL = {
    "queued": "Waiting for the machine to be free",
    "finding": "Finding your dog in the video",
    "measuring": "Measuring how they move",
    "asking": "Asking Pupchat's AI",
    "done": "Done",
    "failed": "That did not work",
}

DEFAULT_SECONDS_PER_FRAME = 1.0
"""Measured: 93 s for a 7.52 s clip, which strided to ~94 keypoint frames, on 12 cores.
Only a starting point - `seconds_per_frame` prefers this machine's own history."""

ASK_SECONDS = 15.0
"""Gemini measured at 13.1 s on a real call, plus a little for the upload."""


def classify(line: str) -> tuple[str | None, str]:
    """A progress line from the pipeline -> (step, detail for the owner).

    The tags are the contract. They are stable strings in pipeline.py and client.py, and
    this is a pure function so the mapping can be tested against all of them without
    spending a second of CPU.

    An unrecognised line returns (None, "") and must leave the step alone: progress also
    carries continuation lines, and letting one of those reset the step would make the
    display jump backwards.
    """
    s = line.strip()
    if s.startswith("[cache]"):
        return "asking", "already measured"
    if s.startswith("[1/4]"):
        return "finding", ""
    if s.startswith("[2/4]"):
        return "finding", "loading the model"
    if s.startswith("[3/4]"):
        return "measuring", "finding the body"
    if s.startswith("[smooth]"):
        return "measuring", "steadying the tracking"
    if s.startswith("[4/4]"):
        return "measuring", "writing up the measurements"
    if s.startswith("[vision]"):
        return "asking", "sending the video"
    if s.startswith(("[gemini]", "[openai]")):
        if "retrying" in s:
            return "asking", "the AI is busy - retrying"
        if "WARNING" in s:
            return None, ""
        return "asking", ""
    m = re.match(r"face (\d+)/(\d+)", s)
    if m:
        return "measuring", f"frame {m.group(1)} of {m.group(2)}"
    m = re.match(r"(\d+) frames at", s)
    if m:
        return None, ""                    # handled by `frames`, not by step
    return None, ""


def frames_in(line: str) -> int | None:
    """The true frame count, announced early enough to sharpen the estimate."""
    m = re.match(r"\s*(\d+) frames at", line)
    return int(m.group(1)) if m else None


def seconds_per_frame(path: Path = TIMING_FILE) -> float:
    """From this machine's own completed runs, not a number baked in here.

    93 s is true of one clip on one laptop. The estimate the owner reads should come
    from the machine the page is running on.
    """
    try:
        samples = json.loads(path.read_text(encoding="utf-8"))["seconds_per_frame"]
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_SECONDS_PER_FRAME
    good = [float(x) for x in samples if isinstance(x, (int, float)) and 0 < x < 60]
    if len(good) < 3:
        return DEFAULT_SECONDS_PER_FRAME
    good.sort()
    return good[len(good) // 2]


def record_timing(seconds: float, frames: int, path: Path = TIMING_FILE) -> None:
    """Keep the last 20 runs. Median, so one unlucky run does not move the estimate."""
    if frames <= 0 or seconds <= 0:
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    samples = [x for x in data.get("seconds_per_frame", [])
               if isinstance(x, (int, float))]
    samples.append(seconds / frames)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seconds_per_frame": samples[-20:]}, indent=2),
                        encoding="utf-8")
    except OSError:
        pass


def estimate(frames: int | None, *, cached: bool, queue_ahead: int = 0,
             ask: bool = True) -> float:
    """Seconds this job is expected to take, start to finish."""
    if cached:
        return ASK_SECONDS if ask else 1.0
    n = frames if frames and frames > 0 else 90
    measure = n * seconds_per_frame()
    return queue_ahead * measure + measure + (ASK_SECONDS if ask else 0.0)


@dataclass
class Job:
    id: str
    kind: str                       # "analyse" | "ask"
    step: str = "queued"
    detail: str = ""
    frames: int | None = None
    frames_done: int = 0
    eta_total_s: float = 0.0
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    result: dict | None = None
    error: str | None = None
    task: object = None             # holds the asyncio Task so it is not collected

    def report(self, line: str) -> None:
        n = frames_in(line)
        if n:
            self.frames = n
            # The count is only known after decoding, so this is the first point the
            # estimate can stop being a guess about clip length.
            self.eta_total_s = estimate(n, cached=False,
                                        ask=(self.kind == "ask"))
        step, detail = classify(line)
        if step:
            self.step = step
            self.detail = detail
        m = re.match(r"\s*face (\d+)/(\d+)", line)
        if m:
            self.frames_done = int(m.group(1))

    def snapshot(self) -> dict:
        elapsed = (self.finished or time.time()) - self.started
        remaining = self.eta_total_s - elapsed
        return {
            "id": self.id,
            "kind": self.kind,
            "step": self.step,
            "step_label": STEP_LABEL.get(self.step, self.step),
            "detail": self.detail,
            "frames": self.frames,
            "frames_done": self.frames_done,
            "elapsed_s": round(elapsed, 1),
            # None once the estimate is spent. Counting down to "0s left" and staying
            # there is the small dishonesty that makes people reload.
            "remaining_s": round(remaining) if remaining > 0 else None,
            "over_estimate": self.eta_total_s > 0 and remaining <= 0
                             and self.step not in ("done", "failed"),
            # A real fraction only where one exists: during the face loop. The body-pose
            # stage is silent, and a bar interpolated against the clock there would be a
            # measurement nobody made.
            "fraction": (round(self.frames_done / self.frames, 3)
                         if self.frames and self.frames_done else None),
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    def __init__(self, keep: int = 40, ttl_s: float = 3600.0):
        self.keep = keep
        self.ttl_s = ttl_s
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, eta_total_s: float = 0.0) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, eta_total_s=eta_total_s)
        with self._lock:
            self._jobs[job.id] = job
            self._evict()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def finish(self, job: Job, result: dict | None = None,
               error: str | None = None) -> None:
        job.finished = time.time()
        job.step = "failed" if error else "done"
        job.detail = ""
        job.result = result
        job.error = error

    def _evict(self) -> None:
        now = time.time()
        stale = [k for k, j in self._jobs.items()
                 if j.finished and now - j.finished > self.ttl_s]
        for k in stale:
            del self._jobs[k]
        if len(self._jobs) > self.keep:
            oldest = sorted(self._jobs.values(), key=lambda j: j.started)
            for j in oldest[: len(self._jobs) - self.keep]:
                if j.finished:
                    self._jobs.pop(j.id, None)
