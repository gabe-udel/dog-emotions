"""Dogs and chats on disk, and who is allowed to touch them.

Two things are being protected here. The first is the owner's data surviving: a restart,
a cleared browser, a crash halfway through a write. The second is that a stranger on a
shared demo never sees it - which cannot be checked by using the app, because every
local request is the owner by definition.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pupchat.library import Library, profile_text, slug


@pytest.fixture
def lib(tmp_path):
    return Library(tmp_path / "library.json")


# ------------------------------------------------------------------ dogs

def test_a_dog_survives_a_restart(tmp_path):
    """The headline improvement over localStorage, so it gets the first test."""
    a = Library(tmp_path / "library.json")
    dog = a.add_dog({"name": "Biscuit", "breed": "Collie cross"})

    b = Library(tmp_path / "library.json")          # a fresh process
    assert [d["name"] for d in b.dogs()] == ["Biscuit"]
    assert b.dog(dog["id"])["breed"] == "Collie cross"


def test_only_the_seven_fields_the_model_gets_are_stored(lib):
    """A field on the page that never reaches the answer is worse than no field: the
    owner fills it in and reasonably expects it to matter."""
    dog = lib.add_dog({"name": "Biscuit", "favourite_toy": "rope", "id": "hack"})
    assert "favourite_toy" not in dog
    assert dog["id"] != "hack"
    assert set(("name", "breed", "age", "health", "training", "triggers",
                "notes")) <= set(dog)


def test_a_dog_can_be_found_by_name_for_the_url(lib):
    """/dogs/Biscuit has to resolve without an id in the address bar."""
    lib.add_dog({"name": "Biscuit"})
    assert lib.dog("Biscuit")["name"] == "Biscuit"
    assert lib.dog("biscuit")["name"] == "Biscuit"
    assert lib.dog("nobody") is None


def test_slug_is_url_safe():
    assert slug("Biscuit") == "biscuit"
    assert slug("Mr. Bojangles III") == "mr-bojangles-iii"
    assert slug("   ") == "dog"


def test_deleting_a_dog_keeps_its_chats(lib):
    """The clips are measured and the answers are the useful part. Nothing about
    removing a profile says to throw those away."""
    dog = lib.add_dog({"name": "Biscuit"})
    chat = lib.add_chat(dog["id"], "Barking")
    assert lib.delete_dog(dog["id"]) == 1
    assert lib.chat(chat["id"])["dog_id"] is None


def test_deleting_an_unknown_dog_says_so(lib):
    assert lib.delete_dog("d_nope") == -1


def test_the_first_dog_becomes_the_active_one(lib):
    """Otherwise the first question after adding a dog carries no profile at all."""
    dog = lib.add_dog({"name": "Biscuit"})
    assert lib.active_dog_id() == dog["id"]
    second = lib.add_dog({"name": "Pepper"})
    assert lib.active_dog_id() == dog["id"], "adding a dog must not steal the selection"
    lib.set_active_dog(second["id"])
    assert lib.active_dog_id() == second["id"]


def test_deleting_the_active_dog_picks_another(lib):
    a = lib.add_dog({"name": "Biscuit"})
    b = lib.add_dog({"name": "Pepper"})
    lib.delete_dog(a["id"])
    assert lib.active_dog_id() == b["id"]


def test_the_same_baseline_is_not_added_twice(lib):
    """Re-uploading a clip hits the analysis cache and returns the same hash."""
    dog = lib.add_dog({"name": "Biscuit"})
    for _ in range(3):
        dog = lib.add_baseline(dog["id"], {"hash": "abc123", "name": "rug.mp4",
                                           "duration": 6.0})
    assert len(dog["baselines"]) == 1
    dog = lib.remove_baseline(dog["id"], "abc123")
    assert dog["baselines"] == []


# ------------------------------------------------------------------ the profile text

def test_an_empty_field_is_left_out_of_the_profile():
    """A blank label invites the model to guess whether "Health:" means healthy or
    unknown, and those are different answers."""
    text = profile_text({"name": "Biscuit", "breed": "Collie", "health": "",
                         "age": "", "training": "", "triggers": "", "notes": ""})
    assert "Biscuit" in text and "Collie" in text
    assert "Health" not in text


def test_no_dog_is_an_empty_profile_not_a_crash():
    assert profile_text(None) == ""
    assert profile_text({}) == ""


# ------------------------------------------------------------------ chats

def test_a_chat_titles_itself_from_the_first_question(lib):
    chat = lib.add_chat(None)
    assert chat["title"] == "New chat"
    chat = lib.append_message(chat["id"], {"role": "user", "text": "Why does he bark?"})
    assert chat["title"] == "Why does he bark?"


def test_search_text_follows_the_whole_conversation(lib):
    """Searching only titles would miss the thing you actually remember, which is
    usually a phrase from the middle of an answer."""
    chat = lib.add_chat(None)
    lib.append_message(chat["id"], {"role": "user", "text": "why the barking"})
    lib.append_message(chat["id"], {"role": "assistant",
                                    "text": "his ears are forward"})
    summary = next(c for c in lib.chat_summaries() if c["id"] == chat["id"])
    assert "ears are forward" in summary["search_text"]


def test_search_text_is_capped(lib):
    """It is returned for every chat in the sidebar listing, so it cannot grow without
    limit."""
    chat = lib.add_chat(None)
    for _ in range(50):
        lib.append_message(chat["id"], {"role": "assistant", "text": "x" * 500})
    summary = next(c for c in lib.chat_summaries() if c["id"] == chat["id"])
    assert len(summary["search_text"]) <= 2000


def test_an_error_turn_does_not_become_the_title(lib):
    """An error is feedback about something that did not happen. A sidebar full of
    chats named after failures reads exactly like history not being saved."""
    chat = lib.add_chat(None)
    lib.append_message(chat["id"], {"role": "error", "text": "that clip is 12.2s"})
    assert lib.chat(chat["id"])["title"] == "New chat"


def test_the_sidebar_listing_leaves_out_the_heavy_parts(lib):
    """Messages and provider history are most of the bytes, and listing is the most
    frequent read on the page."""
    chat = lib.add_chat(None)
    lib.append_message(chat["id"], {"role": "user", "text": "hello"})
    lib.update_chat(chat["id"], history=[{"role": "user", "text": "hello"}])
    summary = next(c for c in lib.chat_summaries() if c["id"] == chat["id"])
    assert "messages" not in summary and "history" not in summary
    assert summary["n_messages"] == 1


def test_chats_are_listed_newest_first(lib):
    a = lib.add_chat(None, "older")
    time.sleep(0.01)
    b = lib.add_chat(None, "newer")
    assert [c["id"] for c in lib.chat_summaries()][:2] == [b["id"], a["id"]]


def test_update_chat_ignores_fields_it_does_not_own(lib):
    """`messages` is appended through one path on purpose, so a crash mid-answer leaves
    a visible failed turn instead of a silent gap."""
    chat = lib.add_chat(None)
    lib.update_chat(chat["id"], messages=[{"role": "user", "text": "snuck in"}],
                    id="c_hack", title="fine")
    out = lib.chat(chat["id"])
    assert out["messages"] == []
    assert out["id"] == chat["id"]
    assert out["title"] == "fine"


# ------------------------------------------------------------------ the file itself

def test_a_corrupt_file_falls_back_to_the_backup(tmp_path):
    """Coming up with yesterday's data beats not coming up."""
    path = tmp_path / "library.json"
    a = Library(path)
    a.add_dog({"name": "Biscuit"})
    a.add_dog({"name": "Pepper"})          # the second save writes the .bak
    path.write_text("{ this is not json", encoding="utf-8")

    b = Library(path)
    assert [d["name"] for d in b.dogs()] == ["Biscuit"]


def test_a_corrupt_file_with_no_backup_is_an_empty_library(tmp_path):
    path = tmp_path / "library.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json at all", encoding="utf-8")
    assert Library(path).dogs() == []


def test_a_file_of_the_wrong_shape_is_refused(tmp_path):
    """Valid JSON that is not a library - someone's notes, a half-written edit - must
    not be loaded as one and then saved over."""
    path = tmp_path / "library.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"hello": "world"}', encoding="utf-8")
    assert Library(path).dogs() == []


def test_the_file_is_never_left_half_written(tmp_path):
    """Written to a temp file and moved into place, so a crash mid-write cannot leave
    truncated JSON where every dog and chat used to be."""
    path = tmp_path / "library.json"
    lib = Library(path)
    for i in range(10):
        lib.add_dog({"name": f"dog{i}", "notes": "x" * 5000})
    json.loads(path.read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"


def test_concurrent_writers_do_not_lose_each_other(tmp_path):
    """The job worker appends an answer on a thread while the page is reading."""
    lib = Library(tmp_path / "library.json")

    def add(n):
        for i in range(10):
            lib.add_dog({"name": f"t{n}-{i}"})

    threads = [threading.Thread(target=add, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(lib.dogs()) == 40
    assert len(Library(tmp_path / "library.json").dogs()) == 40


# ------------------------------------------------------------------ through the API

@pytest.fixture
def api(tmp_path, monkeypatch):
    """A client whose library is a temp file.

    Without this the suite writes dogs into the real outputs/pupchat/library, which is
    the user's own data. Tests must never touch it.
    """
    import pupchat.library as L
    import pupchat.api as A

    store = Library(tmp_path / "library.json")
    monkeypatch.setattr(L, "_LIBRARY", store)
    monkeypatch.setattr(L, "PHOTO_DIR", tmp_path / "photos")
    monkeypatch.setattr(L, "OWNER_FILE", tmp_path / "owner.json")
    monkeypatch.setattr(A, "PHOTO_DIR", tmp_path / "photos")
    from pupchat.api import build_app
    return TestClient(build_app(), client=("127.0.0.1", 50000)), store


@pytest.fixture
def stranger(api):
    client, store = api
    return TestClient(client.app, client=("203.0.113.7", 40000)), store


def test_the_owner_can_add_and_change_a_dog(api):
    client, _ = api
    dog = client.post("/api/dogs", json={"name": "Biscuit"}).json()
    assert dog["name"] == "Biscuit"
    out = client.put(f"/api/dogs/{dog['id']}", json={"breed": "Collie"}).json()
    assert out["breed"] == "Collie"
    assert client.get("/api/library").json()["dogs"][0]["breed"] == "Collie"
    assert client.delete(f"/api/dogs/{dog['id']}").json()["ok"] is True


def test_a_stranger_sees_no_dogs_and_no_chats(api, stranger):
    """A shared demo must not show visitors the owner's dog, whose profile holds
    health notes and behaviour history."""
    owner, _ = api
    owner.post("/api/dogs", json={"name": "Biscuit"})
    owner.post("/api/chats", json={"title": "Barking"})

    guest, _ = stranger
    body = guest.get("/api/library").json()
    assert body["owner"] is False
    assert body["dogs"] == [] and body["chats"] == []


def test_a_stranger_cannot_write(stranger):
    guest, _ = stranger
    assert guest.post("/api/dogs", json={"name": "Hack"}).status_code == 403
    assert guest.put("/api/active-dog", json={"dog_id": "x"}).status_code == 403
    assert guest.post("/api/chats", json={}).status_code == 403


def test_the_refusal_explains_what_a_visitor_can_still_do(stranger):
    """A bare 403 reads as a broken demo."""
    guest, _ = stranger
    detail = guest.post("/api/dogs", json={"name": "x"}).json()["detail"].lower()
    assert "upload a clip" in detail


def test_the_owner_token_makes_a_remote_device_the_owner(api, stranger):
    """The owner's own phone arrives through the tunnel, so it is not local and can be
    recognised no other way."""
    owner, store = api
    owner.post("/api/dogs", json={"name": "Biscuit"})

    guest, _ = stranger
    assert guest.get("/api/library").json()["dogs"] == []
    with_token = guest.get("/api/library",
                           headers={"X-Pupchat-Owner": store.owner_token()})
    assert [d["name"] for d in with_token.json()["dogs"]] == ["Biscuit"]


def test_a_wrong_token_is_still_a_stranger(api, stranger):
    guest, _ = stranger
    r = guest.get("/api/library", headers={"X-Pupchat-Owner": "not-the-token"})
    assert r.json()["owner"] is False


def test_the_library_never_hands_out_the_token(api):
    """It is a credential. Anything that echoed it would turn a read into a takeover."""
    client, store = api
    client.post("/api/dogs", json={"name": "Biscuit"})
    assert store.owner_token() not in client.get("/api/library").text
    assert store.owner_token() not in client.get("/api/health").text


def test_the_qr_link_claims_the_token_then_cleans_the_url(api):
    """A token left in the address bar ends up in history, in a screenshot of the
    phone, and in whatever link gets pasted next."""
    client, store = api
    r = client.get(f"/?k={store.owner_token()}", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    cookie = r.headers.get("set-cookie", "")
    assert "pupchat_owner=" in cookie
    assert "HttpOnly" in cookie


def test_a_bad_token_in_the_link_sets_no_cookie(api):
    client, _ = api
    r = client.get("/?k=guessing", follow_redirects=False)
    assert r.status_code == 303
    assert "pupchat_owner=" not in r.headers.get("set-cookie", "")


def test_a_photo_must_actually_be_an_image(api):
    """A .jpg that is really HTML, served back from this origin, would be stored XSS on
    the owner's own page."""
    client, _ = api
    dog = client.post("/api/dogs", json={"name": "Biscuit"}).json()
    r = client.put(f"/api/dogs/{dog['id']}/photo",
                   files={"file": ("sneaky.jpg", b"<script>alert(1)</script>",
                                   "image/jpeg")})
    assert r.status_code == 400
    assert "image" in r.json()["detail"].lower()


def test_a_real_photo_is_stored_and_served_back(api):
    client, _ = api
    dog = client.post("/api/dogs", json={"name": "Biscuit"}).json()

    from PIL import Image
    import io as _io
    buf = _io.BytesIO()
    Image.new("RGB", (8, 8), (200, 180, 160)).save(buf, format="PNG")

    r = client.put(f"/api/dogs/{dog['id']}/photo",
                   files={"file": ("b.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and r.json()["photo"].endswith(".png")
    got = client.get(f"/api/dogs/{dog['id']}/photo")
    assert got.status_code == 200 and got.content[:4] == b"\x89PNG"


def test_a_dog_photo_is_not_stored_where_the_sweep_can_delete_it(api, tmp_path):
    """guard.sweep() clears the uploads folder after 24h in share mode. A dog's photo
    there would vanish after a day of sharing."""
    import pupchat.api as A
    client, _ = api
    dog = client.post("/api/dogs", json={"name": "Biscuit"}).json()

    from PIL import Image
    import io as _io
    buf = _io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    client.put(f"/api/dogs/{dog['id']}/photo",
               files={"file": ("b.png", buf.getvalue(), "image/png")})

    stored = list((tmp_path / "photos").glob("*"))
    assert stored, "the photo was not written where expected"
    for p in stored:
        assert A.UPLOADS not in p.parents
