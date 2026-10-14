"""The four things that change when strangers can reach Pupchat.

Each test here corresponds to a way a shared URL costs the host something: money,
memory, responsiveness, or other people's video sitting on their disk. None of them
can be checked by running the app locally, because localhost is exempt from all of it
by design - which is exactly why they need tests.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pupchat.api import build_app
from conftest import webapp_files
from pupchat.config import ShareConfig
from pupchat.guard import (AnalysisQueue, Budget, BudgetExceeded, QueueFull,
                           client_ip, is_local, read_capped, sweep)


ANSWER = {"answer": "He is standing still.", "history": [], "provider": "gemini",
          "model": "gemini-3.8-flash", "usage": {}, "cost_usd": 0.0, "seconds": 0.1,
          "truncated": False, "evidence_text": "", "hash": "abc"}
"""A canned provider result. No test may need a key or spend money."""


class FakeRequest:
    def __init__(self, headers=None, host="127.0.0.1"):
        self.headers = headers or {}
        self.client = type("C", (), {"host": host})()


# ------------------------------------------------------------------ who is calling

def test_a_tunnel_visitor_is_identified_by_the_forwarded_header():
    """A Cloudflare tunnel terminates on the host, so every request arrives from
    127.0.0.1. Without reading the header, every visitor shares one bucket AND is
    treated as local - which would exempt the whole internet from every cap."""
    r = FakeRequest({"cf-connecting-ip": "203.0.113.7"})
    assert client_ip(r) == "203.0.113.7"
    assert not is_local(client_ip(r))


def test_a_forwarded_chain_takes_the_client_end():
    r = FakeRequest({"x-forwarded-for": "203.0.113.7, 70.41.3.18, 150.172.238.178"})
    assert client_ip(r) == "203.0.113.7"


def test_your_own_browser_is_local_and_exempt():
    assert is_local(client_ip(FakeRequest()))


def test_a_header_free_remote_caller_is_still_counted():
    assert not is_local(client_ip(FakeRequest(host="198.51.100.4")))


# ------------------------------------------------------------------ spend

def budget(tmp_path, **kw) -> Budget:
    return Budget(tmp_path / "b.json", ShareConfig(**kw))


def test_a_visitor_is_cut_off_at_their_daily_allowance(tmp_path):
    b = budget(tmp_path, asks_per_ip_per_day=3, total_asks_per_day=99)
    for _ in range(3):
        b.spend("ask", "203.0.113.7")
    with pytest.raises(BudgetExceeded) as e:
        b.spend("ask", "203.0.113.7")
    assert "3 questions per visitor" in str(e.value)


def test_one_visitor_hitting_their_cap_does_not_block_another(tmp_path):
    b = budget(tmp_path, asks_per_ip_per_day=1, total_asks_per_day=99)
    b.spend("ask", "203.0.113.7")
    b.spend("ask", "198.51.100.4")              # must not raise
    with pytest.raises(BudgetExceeded):
        b.spend("ask", "203.0.113.7")


def test_the_total_cap_holds_even_when_every_request_claims_a_new_address(tmp_path):
    """The load-bearing one. Per-IP counting rests on a header the visitor controls,
    so it cannot bound the bill on its own - rotating the address defeats it
    completely. The global cap is what makes the worst case a known number."""
    b = budget(tmp_path, asks_per_ip_per_day=100, total_asks_per_day=5)
    for i in range(5):
        b.spend("ask", f"203.0.113.{i}")
    with pytest.raises(BudgetExceeded) as e:
        b.spend("ask", "203.0.113.99")
    assert "today" in str(e.value)


def test_a_refusal_does_not_consume_the_allowance(tmp_path):
    """Otherwise a visitor who hits the cap keeps incrementing it, and the counter
    stops meaning 'uses' - which would also make the reset time a lie."""
    b = budget(tmp_path, asks_per_ip_per_day=1, total_asks_per_day=99)
    b.spend("ask", "203.0.113.7")
    for _ in range(3):
        with pytest.raises(BudgetExceeded):
            b.spend("ask", "203.0.113.7")
    assert b.snapshot()["total"]["ask"] == 1


def test_your_own_use_is_never_counted(tmp_path):
    b = budget(tmp_path, asks_per_ip_per_day=1, total_asks_per_day=1)
    for _ in range(50):
        b.spend("ask", "127.0.0.1")
    assert b.snapshot()["total"] == {}


def test_counters_survive_a_restart(tmp_path):
    """A cap that resets when the host reboots their machine is not a daily cap."""
    budget(tmp_path, asks_per_ip_per_day=2).spend("ask", "203.0.113.7")
    fresh = budget(tmp_path, asks_per_ip_per_day=2)
    fresh.spend("ask", "203.0.113.7")
    with pytest.raises(BudgetExceeded):
        fresh.spend("ask", "203.0.113.7")


def test_yesterdays_counters_do_not_count_against_today(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"date": "2001-01-01",
                                "ip": {"203.0.113.7": {"ask": 99}},
                                "total": {"ask": 99}}), encoding="utf-8")
    Budget(path, ShareConfig(asks_per_ip_per_day=2)).spend("ask", "203.0.113.7")


def test_asks_and_videos_have_separate_allowances(tmp_path):
    """They cost different things - one is money, the other is 93 s of CPU - so
    exhausting one must not block the other."""
    b = budget(tmp_path, asks_per_ip_per_day=1, analyses_per_ip_per_day=1)
    b.spend("ask", "203.0.113.7")
    b.spend("analyse", "203.0.113.7")
    with pytest.raises(BudgetExceeded):
        b.spend("ask", "203.0.113.7")


def test_an_unwritable_counter_file_does_not_take_the_demo_down(tmp_path):
    """Refusing to answer because a counter could not be saved is a worse failure
    than losing the count."""
    b = Budget(tmp_path / "nope" / "deep" / "b.json", ShareConfig())
    (tmp_path / "nope").write_text("I am a file, not a directory", encoding="utf-8")
    b.spend("ask", "203.0.113.7")          # must not raise


# ------------------------------------------------------------------ memory

class FakeUpload:
    """An UploadFile-ish object that yields `total` bytes in 1 MB reads."""

    def __init__(self, total: int):
        self.left = total
        self.served = 0

    async def read(self, n: int = -1) -> bytes:
        take = min(n, self.left)
        self.left -= take
        self.served += take
        return b"\0" * take


def test_an_oversized_upload_is_refused_before_it_is_all_in_memory():
    """`await file.read()` then checking len() allocates the whole body first, so a
    10 GB POST is a 10 GB allocation before the 413. The cap has to be enforced while
    reading or it protects the pipeline and not the host."""
    up = FakeUpload(50 * 1024 * 1024)
    with pytest.raises(ValueError) as e:
        asyncio.run(read_capped(up, 4 * 1024 * 1024))
    assert "limit" in str(e.value)
    assert up.served <= 5 * 1024 * 1024, "read far past the limit before refusing"


def test_an_upload_inside_the_limit_is_returned_whole():
    up = FakeUpload(3 * 1024 * 1024)
    data = asyncio.run(read_capped(up, 8 * 1024 * 1024))
    assert len(data) == 3 * 1024 * 1024


# ------------------------------------------------------------------ the queue

def test_only_one_analysis_runs_at_a_time():
    """Two at once on a 12-core machine is slower than two in sequence and makes both
    look broken."""
    q = AnalysisQueue(max_waiting=8)
    live, peak = [0], [0]

    def work():
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        live[0] -= 1
        return "done"

    async def main():
        return await asyncio.gather(*(q.run(work) for _ in range(5)))

    assert asyncio.run(main()) == ["done"] * 5
    assert peak[0] == 1, f"{peak[0]} ran concurrently"


def test_the_waiting_room_is_bounded():
    """Being refused instantly beats waiting ten minutes behind people the page never
    told you about."""
    q = AnalysisQueue(max_waiting=1)
    release = asyncio.Event()

    async def main():
        async def blocker():
            await q.run(lambda: time.sleep(0.3))

        tasks = [asyncio.create_task(blocker()) for _ in range(2)]
        await asyncio.sleep(0.05)
        with pytest.raises(QueueFull):
            await q.run(lambda: None)
        await asyncio.gather(*tasks)

    asyncio.run(main())


def test_the_queue_drains_even_when_the_work_raises():
    """A leaked slot would permanently shrink the waiting room, and the app would get
    progressively less available with every error."""
    q = AnalysisQueue(max_waiting=2)

    def boom():
        raise RuntimeError("measurement failed")

    async def main():
        for _ in range(4):
            with pytest.raises(RuntimeError):
                await q.run(boom)
        return q.depth

    assert asyncio.run(main()) == 0


def test_the_server_stays_responsive_while_an_analysis_runs():
    """The reason the queue uses a thread. Awaiting blocking CPU on the event loop
    stalled every other request - including the page itself - for the full 93 s."""
    q = AnalysisQueue(max_waiting=4)
    served = []

    async def main():
        task = asyncio.create_task(q.run(lambda: time.sleep(0.4)))
        for _ in range(5):
            await asyncio.sleep(0.02)
            served.append(q.depth)          # the loop is still turning
        await task

    asyncio.run(main())
    assert len(served) == 5
    assert all(d == 1 for d in served), served


# ------------------------------------------------------------------ retention

def test_retention_removes_old_clips_and_their_analyses(tmp_path):
    uploads, out = tmp_path / "uploads", tmp_path / "out"
    uploads.mkdir()
    old = uploads / "aaaaaaaaaaaaaaaa.mp4"
    old.write_bytes(b"old")
    cache = out / "aaaaaaaaaaaaaaaa"
    cache.mkdir(parents=True)
    (cache / "evidence.json").write_text("{}", encoding="utf-8")

    long_ago = time.time() - 48 * 3600
    import os
    os.utime(old, (long_ago, long_ago))

    assert sweep(uploads, out, older_than_hours=24) == ["aaaaaaaaaaaaaaaa"]
    assert not old.exists()
    assert not cache.exists()


def test_retention_leaves_recent_clips_alone(tmp_path):
    uploads, out = tmp_path / "uploads", tmp_path / "out"
    uploads.mkdir()
    fresh = uploads / "bbbbbbbbbbbbbbbb.mp4"
    fresh.write_bytes(b"new")
    assert sweep(uploads, out, older_than_hours=24) == []
    assert fresh.exists()


def test_retention_touches_nothing_outside_the_upload_folder(tmp_path):
    """The keypoint caches are the expensive artefact - 93 s of CPU each - and
    outputs/ also holds measurement files the research pipeline wrote. A sweep that
    guessed at directory names could delete either."""
    uploads, out = tmp_path / "uploads", tmp_path / "out"
    uploads.mkdir()
    (out / "unrelated_cache").mkdir(parents=True)
    (out / "feature_error.json").write_text("{}", encoding="utf-8")
    keep = uploads / "cccccccccccccccc.mp4"
    keep.write_bytes(b"x")
    long_ago = time.time() - 48 * 3600
    import os
    os.utime(keep, (long_ago, long_ago))

    sweep(uploads, out, older_than_hours=24)
    assert (out / "unrelated_cache").exists()
    assert (out / "feature_error.json").exists()


def test_a_missing_upload_folder_is_not_an_error(tmp_path):
    assert sweep(tmp_path / "gone", tmp_path / "out", older_than_hours=1) == []


# ------------------------------------------------------------------ through the API

def test_health_reports_the_visitor_facing_limits():
    """The page shows these, so they come from config rather than being written twice."""
    body = TestClient(build_app()).get("/api/health").json()["share"]
    for key in ("enabled", "retention_hours", "asks_per_day", "videos_per_day",
                "queue_depth"):
        assert key in body, key


def test_a_remote_caller_over_the_cap_gets_429_with_retry_after(monkeypatch):
    """A 500 would look like a broken demo; a 429 with Retry-After is a status a
    browser and a human both understand."""
    import pupchat.api as A

    app = build_app()
    app.state.budget.cfg = ShareConfig(asks_per_ip_per_day=0, total_asks_per_day=0)
    monkeypatch.setattr(A, "_resolve", lambda v, h: Path("clip.mp4"))

    r = TestClient(app).post("/api/ask",
                             data={"question": "what is he doing?", "hash": "abc"},
                             headers={"CF-Connecting-IP": "203.0.113.7"})
    assert r.status_code == 429
    assert r.headers.get("retry-after")
    assert "per day" in r.json()["detail"]


def test_your_own_requests_are_not_rate_limited(monkeypatch):
    """The caps must be invisible when you are the only user, or developing against
    your own server would mean tripping over them."""
    import pupchat.api as A

    app = build_app()
    app.state.budget.cfg = ShareConfig(asks_per_ip_per_day=0, total_asks_per_day=0)
    monkeypatch.setattr(A, "_resolve", lambda v, h: Path("clip.mp4"))

    import pupchat.client as C
    monkeypatch.setattr(C, "ask_pupchat", lambda *a, **k: ANSWER)

    # TestClient reports its peer as "testclient", so the local address is set
    # explicitly - the point of the test is the 127.0.0.1 path, not the default.
    local = TestClient(app, client=("127.0.0.1", 50000))
    r = local.post("/api/ask",
                   data={"question": "what is he doing?", "hash": "abc"})
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------ the page


def page() -> str:
    return webapp_files()["index.html"]


def scripts() -> str:
    """Every script the page ships, concatenated.

    Named files were the old way, and splitting app.js into modules would have dropped
    these four checks silently rather than failing them.
    """
    return "\n".join(t for n, t in webapp_files().items() if n.endswith(".js"))


def test_the_page_credits_the_datasets_it_is_built_on():
    """CC BY-NC is BY. The credit is a licence condition, so it belongs where a
    visitor sees it and not only in the repo."""
    html = page()
    assert 'id="creditLine"' in html
    assert "Non-commercial" in html


def test_the_credit_text_comes_from_the_api_not_the_html():
    """Two copies of a licence string drift. The page renders what /api/health says,
    so the notice and the server can never disagree."""
    js = scripts()
    assert "attribution" in js
    assert "creditLine" in js
    # the licence sentence itself must NOT be duplicated into the front end
    assert "CC BY-NC" not in page().replace('id="creditLine"', "")


def test_the_visitor_notice_states_retention_and_the_caps_from_config():
    """Hard-coding "24 hours" would make the page lie the moment retention_hours
    changed, and a wrong privacy promise is worse than none."""
    js = scripts()
    assert "share.retention_hours" in js
    assert "share.asks_per_day" in js
    assert "share.videos_per_day" in js
    assert "deleted after" in js


def test_the_visitor_notice_is_hidden_unless_the_server_is_shared():
    """On a local run there is no stranger whose video needs removing, so the notice
    would be noise - and claiming a retention policy that is not running is worse."""
    tag = re.search(r'<p id="shareNotice"[^>]*>', page())
    assert tag, "no shareNotice element"
    classes = re.search(r'class="([^"]*)"', tag.group(0)).group(1).split()
    assert "hidden" in classes, classes
    assert "share.enabled" in scripts()
