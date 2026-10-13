"""The HTTP surface a Stitch-exported front end calls.

Stitch generates a front end only, so these endpoints are the whole contract between it
and the pipeline. Nothing here touches the AI provider: /api/ask is covered by the
mocked client tests, and these check routing, caching behaviour, error shapes and - the
one that actually bit - JSON that a strict parser will accept.
"""
from __future__ import annotations

import json
from pathlib import Path
import math

import pytest
from fastapi.testclient import TestClient

from pupchat.api import build_app, json_safe


@pytest.fixture
def client():
    return TestClient(build_app())


# ------------------------------------------------------------------ NaN handling

def test_nan_becomes_null_not_a_crash():
    """Python's json.dumps writes bare NaN; Starlette and every strict parser refuse
    it. This cost a 500 on /api/trace before it was handled."""
    out = json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2.5}})
    assert out == {"a": None, "b": [1.0, None], "c": {"d": 2.5}}
    json.dumps(out, allow_nan=False)          # the real requirement


def test_json_safe_leaves_ordinary_values_alone():
    payload = {"n": 3, "s": "text", "f": 1.5, "b": True, "none": None}
    assert json_safe(payload) == payload


def test_a_missing_measurement_is_null_so_a_ui_can_test_for_it():
    """`null` is testable in a front end; NaN is not even parseable."""
    assert json_safe(float("nan")) is None


# ------------------------------------------------------------------ routing

def test_health_reports_the_configured_provider(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["provider"] in ("openai", "gemini")
    assert "prototype" in body["licence"].lower()


def test_health_carries_the_licence_notice(client):
    """Non-commercial weights. A front end built on this needs to be told, so the
    constraint travels with the API rather than living only in the docs.

    The notice used to say "must not be deployed publicly", which was stricter than
    either licence: both restrict commercial use, not public hosting. What must stay
    pinned is the real constraint - non-commercial - and the fact that it is stated.
    """
    notice = client.get("/api/health").json()["licence"].lower()
    assert "cc by-nc" in notice
    assert "commercially" in notice
    assert "never sold" in notice
    assert "not a vet" in notice


def test_health_carries_the_attribution_the_licence_requires(client):
    """CC BY-NC is BY: credit is a condition, not a courtesy. It travels with the API
    so a front end that only reads JSON cannot leave it behind."""
    credit = client.get("/api/health").json()["attribution"]
    assert "DogFLW" in credit
    assert "DeepLabCut" in credit or "SuperAnimal" in credit


def test_unanalysed_hash_is_a_404_not_a_500(client):
    for path in ("/api/evidence/deadbeef", "/api/track/deadbeef",
                 "/api/trace/deadbeef", "/api/overlay/deadbeef"):
        assert client.get(path).status_code == 404, path


def test_ask_without_a_known_video_is_a_404(client):
    r = client.post("/api/ask", data={"question": "what is he doing?",
                                      "hash": "deadbeef"})
    assert r.status_code == 404
    assert "upload" in r.json()["detail"].lower()


def test_cors_is_open_so_a_separate_front_end_can_call_it(client):
    """A Stitch export opened from the filesystem or served on another port is a
    different origin, and a silent CORS failure looks like a broken backend."""
    r = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_every_documented_route_exists(client):
    paths = {r.path for r in build_app().routes if hasattr(r, "path")}
    for expected in ("/api/health", "/api/analyse", "/api/ask", "/api/evidence/{vhash}",
                     "/api/track/{vhash}", "/api/trace/{vhash}",
                     "/api/overlay/{vhash}", "/api/flag", "/api/flags"):
        assert expected in paths, expected


def test_flags_endpoint_returns_a_list(client):
    r = client.get("/api/flags")
    assert r.status_code == 200
    assert isinstance(r.json()["flags"], list)


# ------------------------------------------------------------------ upload identity

@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    """Point the upload store at a temp folder.

    Uploads used to live in the system temp directory, where test litter was
    invisible; they are project-local now, so a test that called _store_upload for
    real left 3-byte .mp4 files in outputs/ for the retention sweep to find.
    """
    import pupchat.api as A
    monkeypatch.setattr(A, "UPLOADS", tmp_path / "uploads")
    return tmp_path / "uploads"


def test_the_same_bytes_under_two_names_are_one_file(upload_dir):
    """Analysis is minutes of CPU, so re-uploading the same clip must not redo it."""
    from pupchat.api import _store_upload
    a = _store_upload("first.mp4", b"identical bytes")
    b = _store_upload("second.mp4", b"identical bytes")
    assert a == b
    assert a.parent == upload_dir


def test_different_bytes_get_different_files(upload_dir):
    from pupchat.api import _store_upload
    a = _store_upload("clip.mp4", b"one")
    b = _store_upload("clip.mp4", b"two")
    assert a != b


# ------------------------------------------------------------------ upload limits

def test_health_publishes_the_clip_length_window(client):
    """The page reads these rather than hard-coding 5 and 10, so one config change
    moves both the browser check and the server check."""
    u = client.get("/api/health").json()["upload"]
    assert u["min_seconds"] == 5.0 and u["max_seconds"] == 10.0


def test_health_says_whether_a_key_exists_but_never_what_it_is(client):
    """The page needs to know it can ask a question. It must never learn the secret."""
    body = client.get("/api/health").text
    assert "key_configured" in body
    assert "AIza" not in body and "sk-" not in body


@pytest.mark.parametrize("duration,expect", [(3.1, "too few frames"),
                                             (12.2, "too long to measure")])
def test_duration_gate_explains_which_way_it_failed(tmp_path, duration, expect,
                                                    monkeypatch):
    """A bare 'invalid video' tells the owner nothing about how to fix it."""
    import pupchat.api as A
    from pupchat.config import Config

    monkeypatch.setattr(A, "probe", lambda p: (30.0, int(duration * 30), duration))
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00")
    with pytest.raises(ValueError) as e:
        A.check_duration(f, Config())
    assert expect in str(e.value)
    assert f"{duration:.1f}s" in str(e.value)


def test_a_clip_inside_the_window_passes(tmp_path, monkeypatch):
    import pupchat.api as A
    from pupchat.config import Config

    monkeypatch.setattr(A, "probe", lambda p: (30.0, 210, 7.0))
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00")
    assert A.check_duration(f, Config()) == 7.0


def test_the_boundaries_are_tolerant_of_container_rounding(tmp_path, monkeypatch):
    """A clip trimmed to 5.0s routinely reports 4.97, and rejecting it would be
    indistinguishable from a bug to whoever trimmed it."""
    import pupchat.api as A
    from pupchat.config import Config

    for d in (4.9, 10.1):
        monkeypatch.setattr(A, "probe", lambda p, d=d: (30.0, int(d * 30), d))
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00")
        assert A.check_duration(f, Config()) == d


# ------------------------------------------------------------------ the page

def test_the_page_is_served_at_the_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Pupchat" in r.text
    assert "app.js" in r.text


def test_the_page_script_is_served(client):
    assert client.get("/app.js").status_code == 200


def test_no_key_reaches_the_browser(client):
    """The whole reason the API exists rather than calling Gemini from the page."""
    blob = client.get("/").text + client.get("/app.js").text
    for marker in ("AIzaSy", "sk-", "GEMINI_API_KEY=", "OPENAI_API_KEY="):
        assert marker not in blob, marker


def test_api_routes_win_over_static_files(client):
    """The static mount is at /, so it must be added last or it would shadow /api/*."""
    assert client.get("/api/health").json()["ok"] is True


# ------------------------------------------------------------------ video playback

def test_an_uploaded_clip_can_be_played_back(client):
    """The browser never holds the bytes, so showing the user what they sent means
    streaming it from the server by the hash the chat already stores."""
    r = client.get("/api/video/ab65fb44eb3a3f0d")
    if r.status_code == 404:
        pytest.skip("that clip is not in this machine's upload cache")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("video/")
    assert len(r.content) > 1000


def test_an_unknown_video_id_is_a_404(client):
    assert client.get("/api/video/deadbeefdeadbeef").status_code == 404


# ------------------------------------------------------------------ page wiring

def test_errors_do_not_become_chats():
    """The reported bug: toastError() routed through ensureChat(), so every rejected
    upload created a chat titled "(error)" with no conversation in it - which reads
    exactly like chat history not being saved."""
    js = (Path(__file__).resolve().parent.parent
          / "pupchat" / "webapp" / "app.js").read_text(encoding="utf-8")
    assert 'ensureChat("(error)")' not in js
    assert "function toastError" in js
    assert 'getElementById' in js or '$("toast")' in js


def test_the_page_has_a_toast_and_a_video_preview():
    html = (Path(__file__).resolve().parent.parent
            / "pupchat" / "webapp" / "index.html").read_text(encoding="utf-8")
    assert 'id="toast"' in html
    assert 'id="attachPreview"' in html
    assert "<video" in html


def test_the_chat_bubble_streams_the_clip_rather_than_storing_it():
    js = (Path(__file__).resolve().parent.parent
          / "pupchat" / "webapp" / "app.js").read_text(encoding="utf-8")
    assert "/api/video/" in js
    # a data: URL here would mean the bytes went into localStorage, which is a ~5 MB
    # quota against a multi-megabyte clip
    assert "data:video" not in js


# ------------------------------------------------------------------ startup

def test_a_busy_port_stops_the_server_instead_of_shadowing_it(monkeypatch):
    """uvicorn's own behaviour here is to log the bind failure and exit 0, leaving the
    OTHER process answering the browser. When that one is an older build it serves
    stale code, and a missing endpoint looks like a backend bug rather than a second
    server. Refusing to start is the only honest outcome."""
    import pupchat.api as A

    monkeypatch.setattr(A, "port_is_taken", lambda *a, **k: True)
    monkeypatch.setattr(A, "build_app", lambda: pytest.fail("must not reach uvicorn"))
    with pytest.raises(SystemExit) as e:
        A.main()
    msg = str(e.value).lower()
    assert "already in use" in msg
    assert "old code" in msg          # names the consequence, not just the symptom


def test_port_check_reports_a_free_port_as_free():
    from pupchat.api import port_is_taken
    assert port_is_taken(port=1, host="127.0.0.1") is False
