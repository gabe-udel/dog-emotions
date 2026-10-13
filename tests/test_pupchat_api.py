"""The HTTP surface a Stitch-exported front end calls.

Stitch generates a front end only, so these endpoints are the whole contract between it
and the pipeline. Nothing here touches the AI provider: /api/ask is covered by the
mocked client tests, and these check routing, caching behaviour, error shapes and - the
one that actually bit - JSON that a strict parser will accept.
"""
from __future__ import annotations

import json
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
    constraint travels with the API rather than living only in the docs."""
    notice = client.get("/api/health").json()["licence"].lower()
    assert "cc by-nc" in notice
    assert "commercially" in notice
    assert "not be deployed" in notice


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

def test_the_same_bytes_under_two_names_are_one_file(tmp_path):
    """Analysis is minutes of CPU, so re-uploading the same clip must not redo it."""
    from pupchat.api import _store_upload
    a = _store_upload("first.mp4", b"identical bytes")
    b = _store_upload("second.mp4", b"identical bytes")
    assert a == b


def test_different_bytes_get_different_files():
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
