"""Progress, and whether what it tells the owner is true.

`classify` is a pure function over the pipeline's own log lines, so the whole stage
mapping can be checked against the real strings without spending a second of CPU. That
matters: the alternative is measuring a clip every time this is touched.

The estimate gets its own tests because the dishonest version is the easy one to write.
A bar interpolated against the clock looks better and means nothing, since the stage that
dominates the wait reports nothing at all while it runs.
"""
from __future__ import annotations

import json
import time

import pytest

from pupchat.jobs import (ASK_SECONDS, DEFAULT_SECONDS_PER_FRAME, Job, JobStore,
                          classify, estimate, frames_in, record_timing,
                          seconds_per_frame)

# The exact strings pipeline.py and client.py emit. If one of these changes, the page
# silently stops moving, so they are pinned here rather than paraphrased.
REAL_LINES = [
    ("[cache] reusing measurements for clip.mp4 (ab65fb44eb3a3f0d)", "asking"),
    ("[1/4] decoding clip.mp4 (7.5s at 25.0 fps)", "finding"),
    ("      94 frames at 12.5 fps (strided from 25.0)", None),
    ("[2/4] loading the cascade (snapshot-best-008.pt)", "finding"),
    ("[3/4] body pose (this is the slow part on CPU)", "measuring"),
    ("      face 20/94", "measuring"),
    ("[smooth] One Euro min_cutoff=0.5 beta=5.0: face jitter 1.2 -> 1.1 px", "measuring"),
    ("[4/4] features and evidence", "measuring"),
    ("      wrote outputs/pupchat/ab65fb44eb3a3f0d", None),
    ("[vision] sending clip.mp4 whole (4.6 MB) - Gemini samples it", "asking"),
    ("[gemini] gemini-3.8-flash", "asking"),
]


@pytest.mark.parametrize("line,step", REAL_LINES)
def test_each_real_progress_line_maps_to_the_right_step(line, step):
    assert classify(line)[0] == step


def test_an_unknown_line_leaves_the_step_alone():
    """Progress carries continuation lines. Letting one reset the step would make the
    display jump backwards mid-measurement."""
    job = Job(id="j", kind="analyse")
    job.report("[3/4] body pose (this is the slow part on CPU)")
    job.report("      something nobody anticipated")
    assert job.step == "measuring"


def test_the_steps_only_go_forwards_on_a_real_run():
    job = Job(id="j", kind="ask")
    seen = []
    for line, _ in REAL_LINES[1:]:
        job.report(line)
        seen.append(job.step)
    order = ["finding", "measuring", "asking"]
    assert [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]] == order


def test_a_cached_clip_jumps_straight_to_asking():
    """One line, then done. The page must not sit on "measuring" for a clip that was
    measured last week."""
    job = Job(id="j", kind="ask")
    job.report("[cache] reusing measurements for clip.mp4 (abc)")
    assert job.step == "asking"


def test_a_retry_says_so_rather_than_looking_stuck():
    job = Job(id="j", kind="ask")
    job.report("[gemini] transient error, retrying in 4s (attempt 2/4): 503")
    assert job.step == "asking"
    assert "retry" in job.detail.lower()


def test_the_frame_count_is_picked_up_from_the_log():
    assert frames_in("      94 frames at 12.5 fps (strided from 25.0)") == 94
    assert frames_in("[1/4] decoding clip.mp4") is None


def test_the_frame_count_sharpens_the_estimate():
    """Before decoding there is only a guess. The true count arrives a second in."""
    job = Job(id="j", kind="analyse", eta_total_s=estimate(None, cached=False))
    before = job.eta_total_s
    job.report("      200 frames at 12.5 fps (strided from 25.0)")
    assert job.frames == 200
    assert job.eta_total_s > before


def test_the_face_loop_gives_a_real_fraction():
    job = Job(id="j", kind="analyse")
    job.report("      94 frames at 12.5 fps")
    job.report("      face 47/94")
    assert job.snapshot()["fraction"] == pytest.approx(0.5, abs=0.01)


def test_there_is_no_fraction_during_the_silent_stage():
    """[3/4] body pose is one blocking call over every frame and reports nothing until
    it ends. A bar there would be interpolated against the clock - a measurement nobody
    made, in an app whose whole point is not making those."""
    job = Job(id="j", kind="analyse")
    job.report("      94 frames at 12.5 fps")
    job.report("[3/4] body pose (this is the slow part on CPU)")
    assert job.snapshot()["fraction"] is None


# ------------------------------------------------------------------ the estimate

def test_a_cached_clip_is_estimated_as_just_the_question():
    assert estimate(94, cached=True) == ASK_SECONDS


def test_the_estimate_grows_with_the_queue():
    alone = estimate(94, cached=False, queue_ahead=0)
    third = estimate(94, cached=False, queue_ahead=2)
    assert third > alone


def test_the_estimate_counts_down_and_then_stops_counting():
    """Showing "0s left" for another forty seconds is the small dishonesty that makes
    people reload - which starts a second measurement behind the first."""
    job = Job(id="j", kind="analyse", eta_total_s=10.0)
    assert job.snapshot()["remaining_s"] == 10
    job.started = time.time() - 30
    snap = job.snapshot()
    assert snap["remaining_s"] is None
    assert snap["over_estimate"] is True


def test_a_finished_job_is_not_over_estimate():
    """Only a job still running can be late."""
    store = JobStore()
    job = store.create("analyse", eta_total_s=1.0)
    job.started = time.time() - 30
    store.finish(job, result={"hash": "abc"})
    assert job.snapshot()["over_estimate"] is False


def test_timing_comes_from_this_machine_once_there_is_enough_of_it(tmp_path):
    """93 s is true of one clip on one laptop. The number the owner reads should come
    from the machine the page is running on."""
    path = tmp_path / "timing.json"
    assert seconds_per_frame(path) == DEFAULT_SECONDS_PER_FRAME

    for _ in range(3):
        record_timing(seconds=40.0, frames=20, path=path)        # 2.0 s/frame
    assert seconds_per_frame(path) == pytest.approx(2.0)


def test_timing_keeps_only_the_recent_runs(tmp_path):
    path = tmp_path / "timing.json"
    for i in range(40):
        record_timing(seconds=float(i + 1), frames=1, path=path)
    assert len(json.loads(path.read_text())["seconds_per_frame"]) == 20


def test_a_nonsense_timing_sample_is_ignored(tmp_path):
    path = tmp_path / "timing.json"
    path.write_text(json.dumps({"seconds_per_frame": ["x", -1, 0, 9999]}),
                    encoding="utf-8")
    assert seconds_per_frame(path) == DEFAULT_SECONDS_PER_FRAME


def test_an_unwritable_timing_file_does_not_break_a_job(tmp_path):
    bad = tmp_path / "nope"
    bad.write_text("I am a file", encoding="utf-8")
    record_timing(seconds=10.0, frames=10, path=bad / "deep" / "timing.json")


# ------------------------------------------------------------------ the store

def test_a_finished_job_keeps_its_result():
    store = JobStore()
    job = store.create("analyse")
    store.finish(job, result={"hash": "abc"})
    assert store.get(job.id).snapshot()["result"] == {"hash": "abc"}
    assert store.get(job.id).snapshot()["step"] == "done"


def test_a_failed_job_says_why():
    """The page shows this to the owner, so "failed" on its own is not enough."""
    store = JobStore()
    job = store.create("ask")
    store.finish(job, error="that clip is 12.2s")
    snap = store.get(job.id).snapshot()
    assert snap["step"] == "failed"
    assert "12.2s" in snap["error"]


def test_old_finished_jobs_are_evicted_but_running_ones_are_not():
    store = JobStore(keep=3, ttl_s=0.0)
    done = [store.create("analyse") for _ in range(5)]
    for j in done:
        store.finish(j, result={})
    running = store.create("analyse")
    store.create("analyse")                      # triggers eviction
    assert store.get(running.id) is not None
    assert sum(store.get(j.id) is not None for j in done) <= 3
