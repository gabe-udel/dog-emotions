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
