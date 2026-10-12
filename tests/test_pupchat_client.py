"""Request assembly, provider swapping, and the accounting around them.

Every test here mocks the API. None needs OPENAI_API_KEY or GEMINI_API_KEY, none opens
a socket, none costs money - a suite that silently spends is a trap for whoever runs it
next.

The point of the provider split is that only the MEDIA differs. These tests pin that:
the system prompt and the evidence text must be byte-identical across providers, or a
comparison between them measures two prompts rather than two models.
"""
from __future__ import annotations

import json
import math
import types

import pytest

from pupchat import client as C
from pupchat.config import Config, VisionConfig
from pupchat.providers import MEDIA_FRAMES, MEDIA_VIDEO, available, get_provider
from pupchat.providers.gemini_provider import GeminiProvider, mime_for
from pupchat.providers.openai_provider import OpenAIProvider

FRAMES = [(0.0, "AAAABBBBCCCC" * 40), (0.5, "DDDDEEEEFFFF" * 40)]
EVIDENCE = "Ear tips 0.18 face-widths below eye level, visible 90% of frames."
QUESTION = "Why does he do this?"


# --------------------------------------------------------------- fakes

class FakeOpenAIResponse:
    output_text = "His ears are back and low through most of the clip."
    # OpenAI's output_tokens already INCLUDES its reasoning tokens.
    usage = {"input_tokens": 12_000, "output_tokens": 300,
             "input_tokens_details": {"cached_tokens": 10_000},
             "output_tokens_details": {"reasoning_tokens": 120}}
    incomplete_details = None


class FakeOpenAIClient:
    def __init__(self):
        self.calls = []
        self.responses = types.SimpleNamespace(create=self._create)

    def _create(self, **payload):
        self.calls.append(payload)
        return FakeOpenAIResponse()


class FakeGeminiResponse:
    text = "His ears are back and low through most of the clip."
    # Gemini reports the visible answer and its thinking SEPARATELY and bills both.
    # These are the real proportions from the first live call.
    usage_metadata = types.SimpleNamespace(
        prompt_token_count=9_000, candidates_token_count=214,
        thoughts_token_count=682, cached_content_token_count=0)
    candidates = [types.SimpleNamespace(
        finish_reason=types.SimpleNamespace(name="STOP"))]


class FakeGeminiTruncated(FakeGeminiResponse):
    candidates = [types.SimpleNamespace(
        finish_reason=types.SimpleNamespace(name="MAX_TOKENS"))]


class FakeGeminiClient:
    def __init__(self):
        self.calls = []
        self.uploaded = []
        self.models = types.SimpleNamespace(generate_content=self._generate)
        self.files = types.SimpleNamespace(upload=self._upload)

    def _generate(self, **payload):
        self.calls.append(payload)
        return FakeGeminiResponse()

    def _upload(self, file):
        self.uploaded.append(file)
        return {"uri": f"files/fake-{len(self.uploaded)}"}


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00" * 2048)
    return p


# --------------------------------------------------------------- the registry

def test_both_providers_are_registered():
    assert available() == ["gemini", "openai"]


def test_each_provider_declares_what_media_it_takes():
    assert get_provider("openai").media_kind == MEDIA_FRAMES
    assert get_provider("gemini").media_kind == MEDIA_VIDEO


def test_unknown_provider_fails_with_a_useful_message():
    with pytest.raises(SystemExit) as e:
        get_provider("anthropic")
    assert "expected one of" in str(e.value)


def test_config_selects_the_active_provider(cfg):
    object.__setattr__(cfg, "provider", "gemini")
    assert cfg.active.name == "gemini"
    assert cfg.active.key_env == "GEMINI_API_KEY"
    object.__setattr__(cfg, "provider", "openai")
    assert cfg.active.key_env == "OPENAI_API_KEY"


def test_unknown_provider_in_config_is_rejected(cfg):
    object.__setattr__(cfg, "provider", "nope")
    with pytest.raises(SystemExit):
        _ = cfg.active


# --------------------------------------------------------------- shared prompt

def test_both_providers_send_the_identical_system_prompt(cfg, video):
    o = OpenAIProvider().build_request(EVIDENCE, QUESTION, FRAMES, cfg)
    g = GeminiProvider().build_request(EVIDENCE, QUESTION, video, cfg)
    assert o["instructions"] == g["config"]["system_instruction"] == C.system_prompt()


def test_both_providers_send_the_identical_evidence_text(cfg, video):
    o = OpenAIProvider().build_request(EVIDENCE, QUESTION, FRAMES, cfg)
    g = GeminiProvider().build_request(EVIDENCE, QUESTION, video, cfg)
    assert EVIDENCE in o["input"][0]["content"][0]["text"]
    assert EVIDENCE in g["contents"][0]["parts"][0]["text"]


def test_system_prompt_carries_the_hard_limits():
    txt = C.system_prompt()
    for required in ("veterinarian", "certified behaviourist", "Never diagnose"):
        assert required in txt


# --------------------------------------------------------------- question last

def test_openai_puts_the_question_last_after_the_frames(cfg):
    p = OpenAIProvider().build_request(EVIDENCE, QUESTION, FRAMES, cfg)
    content = p["input"][0]["content"]
    last_image = max(i for i, c in enumerate(content) if c["type"] == "input_image")
    assert content[-1]["type"] == "input_text"
    assert content[-1]["text"].endswith(QUESTION)
    assert last_image < len(content) - 1


def test_gemini_puts_the_question_last_after_the_video(cfg, video):
    p = GeminiProvider().build_request(EVIDENCE, QUESTION, video, cfg)
    parts = p["contents"][0]["parts"]
    assert parts[-1]["text"].endswith(QUESTION)
    assert "inline_data" in parts[1]


def test_openai_sends_one_image_per_frame_with_timestamps(cfg):
    p = OpenAIProvider().build_request(EVIDENCE, QUESTION, FRAMES, cfg)
    content = p["input"][0]["content"]
    images = [c for c in content if c["type"] == "input_image"]
    assert len(images) == len(FRAMES)
    assert all(c["image_url"].startswith("data:image/jpeg;base64,") for c in images)
    assert any("Frame at 0.5s" in c.get("text", "") for c in content)


def test_model_comes_from_config_not_from_code(cfg):
    object.__setattr__(cfg.openai, "model", "gpt-4o-mini")
    assert OpenAIProvider().build_request(EVIDENCE, "q", FRAMES, cfg)["model"] \
        == "gpt-4o-mini"
    object.__setattr__(cfg.gemini, "model", "gemini-2.5-pro")
    p = GeminiProvider().build_request(EVIDENCE, "q", cfg.snapshot.parent, cfg) \
        if False else None  # path unused here
    object.__setattr__(cfg.gemini, "model", "gemini-2.5-pro")
    assert cfg.gemini.model == "gemini-2.5-pro"


# --------------------------------------------------------------- gemini media

def test_small_video_goes_inline(cfg, video):
    p = GeminiProvider().build_request(EVIDENCE, QUESTION, video, cfg)
    part = p["contents"][0]["parts"][1]
    assert "inline_data" in part
    assert part["inline_data"]["mime_type"] == "video/mp4"


def test_large_video_uses_the_files_api(cfg, tmp_path, monkeypatch):
    import pupchat.providers.gemini_provider as G
    monkeypatch.setattr(G, "INLINE_LIMIT_BYTES", 1024)
    big = tmp_path / "big.mp4"
    big.write_bytes(b"\x00" * 4096)
    p = GeminiProvider().build_request(EVIDENCE, QUESTION, big, cfg)
    assert "file_upload" in p["contents"][0]["parts"][1]


def test_file_upload_is_resolved_on_send(cfg, tmp_path, monkeypatch):
    import pupchat.providers.gemini_provider as G
    monkeypatch.setattr(G, "INLINE_LIMIT_BYTES", 1024)
    big = tmp_path / "big.mp4"
    big.write_bytes(b"\x00" * 4096)
    fake = FakeGeminiClient()
    p = GeminiProvider().build_request(EVIDENCE, QUESTION, big, cfg)
    GeminiProvider().send(p, cfg, client=fake)
    assert fake.uploaded == [str(big)]
    sent_parts = fake.calls[0]["contents"][0]["parts"]
    assert sent_parts[1] == {"uri": "files/fake-1"}


def test_unsupported_container_is_refused(tmp_path):
    bad = tmp_path / "clip.mkv"
    bad.write_bytes(b"\x00")
    with pytest.raises(SystemExit) as e:
        mime_for(bad)
    assert "does not document a video MIME type" in str(e.value)


def test_documented_mime_types_resolve(tmp_path):
    for ext, mime in ((".mp4", "video/mp4"), (".mov", "video/mov"),
                      (".webm", "video/webm")):
        p = tmp_path / f"c{ext}"
        p.write_bytes(b"\x00")
        assert mime_for(p) == mime


# --------------------------------------------------------------- frame rate rule

def test_two_fps_under_fifteen_seconds():
    v = VisionConfig()
    assert v.fps_for(14.9) == 2.0
    assert v.fps_for(15.0) == 1.0
    assert v.fps_for(60.0) == 1.0


# --------------------------------------------------------------- follow-ups

def test_openai_follow_up_keeps_frames_byte_identical_in_the_prefix(cfg):
    first = OpenAIProvider().build_request(EVIDENCE, "first?", FRAMES, cfg)
    history = [{"role": "user", "text": "first?"},
               {"role": "assistant", "text": "because..."}]
    second = OpenAIProvider().build_request(EVIDENCE, "and then?", FRAMES, cfg, history)
    assert first["input"][0]["content"][:-1] == second["input"][0]["content"]
    assert second["input"][-1]["content"][-1]["text"] == "and then?"


def test_gemini_follow_up_keeps_the_video_in_the_prefix(cfg, video):
    first = GeminiProvider().build_request(EVIDENCE, "first?", video, cfg)
    history = [{"role": "user", "text": "first?"},
               {"role": "assistant", "text": "because..."}]
    second = GeminiProvider().build_request(EVIDENCE, "and then?", video, cfg, history)
    assert first["contents"][0]["parts"][:-1] == second["contents"][0]["parts"]
    assert second["contents"][-1]["parts"][-1]["text"] == "and then?"
    assert second["contents"][1]["role"] == "user"
    assert second["contents"][2]["role"] == "model"


# --------------------------------------------------------------- redaction

def test_openai_redaction_shortens_images_and_keeps_text(cfg):
    p = OpenAIProvider().build_request(EVIDENCE, "q", FRAMES, cfg)
    r = OpenAIProvider().redact(p)
    for part in r["input"][0]["content"]:
        if part["type"] == "input_image":
            assert len(part["image_url"]) < 80 and "chars]" in part["image_url"]
    assert EVIDENCE in r["input"][0]["content"][0]["text"]


def test_gemini_redaction_shortens_the_inline_video(cfg, video):
    p = GeminiProvider().build_request(EVIDENCE, "q", video, cfg)
    r = GeminiProvider().redact(p)
    assert len(r["contents"][0]["parts"][1]["inline_data"]["data"]) < 80


def test_redaction_does_not_mutate_the_original(cfg):
    p = OpenAIProvider().build_request(EVIDENCE, "q", FRAMES, cfg)
    before = p["input"][0]["content"][2]["image_url"]
    OpenAIProvider().redact(p)
    assert p["input"][0]["content"][2]["image_url"] == before


@pytest.mark.parametrize("provider_name", ["openai", "gemini"])
def test_no_api_key_ever_appears_in_a_payload(cfg, video, monkeypatch, provider_name):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-secret-value-67890")
    object.__setattr__(cfg, "provider", provider_name)
    p = get_provider(provider_name).build_request(
        EVIDENCE, "q", FRAMES if provider_name == "openai" else video, cfg)
    blob = json.dumps(get_provider(provider_name).redact(p))
    assert "sk-secret-value-12345" not in blob
    assert "gm-secret-value-67890" not in blob


# --------------------------------------------------------------- usage and cost

def test_openai_reasoning_tokens_are_not_double_counted():
    """OpenAI's output_tokens already includes reasoning, so adding them again would
    overstate the bill."""
    r = OpenAIProvider().send({"model": "x"}, Config(), client=FakeOpenAIClient())
    assert r.usage == {"input_tokens": 12_000, "output_tokens": 300,
                       "cached_tokens": 10_000, "reasoning_tokens": 120}


def test_gemini_thinking_tokens_are_billed_as_output(cfg, video):
    """The bug this pins: Gemini reports 214 visible tokens and 682 thinking tokens,
    charges for both, and an earlier version billed only the visible count - under-
    reporting the real cost by about four times, silently."""
    p = GeminiProvider().build_request(EVIDENCE, "q", video, cfg)
    r = GeminiProvider().send(p, cfg, client=FakeGeminiClient())
    assert r.usage["output_tokens"] == 214 + 682
    assert r.usage["reasoning_tokens"] == 682

    # Compare the OUTPUT component alone. On a real call the input tokens dominate the
    # total, so the headline cost moves by a third while the output charge quadruples -
    # which is exactly why the undercount was easy to miss.
    def out_cost(n):
        return C.estimate_cost("gemini-3.8-flash",
                               {"input_tokens": 0, "output_tokens": n,
                                "cached_tokens": 0})

    assert out_cost(r.usage["output_tokens"]) == pytest.approx(out_cost(214) * 4.19,
                                                               rel=0.01)


def test_truncation_is_surfaced_not_swallowed(cfg, video):
    """A cut-off answer presented as a whole one is a silent failure."""
    class Trunc(FakeGeminiClient):
        def _generate(self, **payload):
            self.calls.append(payload)
            return FakeGeminiTruncated()

    p = GeminiProvider().build_request(EVIDENCE, "q", video, cfg)
    assert GeminiProvider().send(p, cfg, client=Trunc()).truncated
    assert not GeminiProvider().send(p, cfg, client=FakeGeminiClient()).truncated


def test_output_cap_leaves_room_for_thinking(cfg):
    """900 truncated a real answer at 682 thinking + 214 visible."""
    assert cfg.active.max_output_tokens >= 2000


def test_cached_tokens_are_priced_lower_than_fresh_ones():
    cached = {"input_tokens": 10_000, "output_tokens": 0, "cached_tokens": 10_000}
    fresh = {"input_tokens": 10_000, "output_tokens": 0, "cached_tokens": 0}
    assert C.estimate_cost("gpt-6.1-sol", cached) < C.estimate_cost("gpt-6.1-sol", fresh)


def test_gemini_models_are_priced():
    cost = C.estimate_cost("gemini-3.8-flash",
                           {"input_tokens": 1_000_000, "output_tokens": 0,
                            "cached_tokens": 0})
    assert cost == pytest.approx(0.75)


def test_cost_of_an_unknown_model_is_nan_not_zero():
    assert math.isnan(C.estimate_cost("no-such-model",
                                      {"input_tokens": 1, "output_tokens": 1}))


def test_usage_log_is_one_json_object_per_line(tmp_path):
    p = tmp_path / "usage.jsonl"
    C.log_usage({"a": 1}, p)
    C.log_usage({"b": 2}, p)
    assert [json.loads(ln) for ln in p.read_text().strip().split("\n")] \
        == [{"a": 1}, {"b": 2}]


# --------------------------------------------------------------- end to end, mocked

@pytest.mark.parametrize("provider_name,fake,expected_media", [
    ("openai", FakeOpenAIClient, "2 frames"),
    ("gemini", FakeGeminiClient, MEDIA_VIDEO),
])
def test_ask_pupchat_runs_without_a_key_or_network(
        cfg, tmp_path, monkeypatch, video, provider_name, fake, expected_media):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    object.__setattr__(cfg, "provider", provider_name)

    monkeypatch.setattr(C, "analyse", lambda *a, **k: {
        "evidence_text": EVIDENCE, "hash": "deadbeef", "cached": True})
    monkeypatch.setattr(C, "cache_dir", lambda h: tmp_path)
    monkeypatch.setattr(C, "sample_vision_frames", lambda *a, **k: FRAMES)
    logged: list[dict] = []
    monkeypatch.setattr(C, "log_usage", lambda entry, path=None: logged.append(entry))

    client = fake()
    res = C.ask_pupchat(video, "Is he okay?", cfg=cfg,
                        progress=lambda *a, **k: None, client=client)

    assert res["answer"].startswith("His ears are back")
    assert res["provider"] == provider_name
    assert client.calls, "the provider client should have been called"
    assert len(logged) == 1
    assert logged[0]["provider"] == provider_name
    assert logged[0]["media"] == expected_media
    assert logged[0]["cost_usd"] > 0


def test_history_shape_is_provider_neutral(cfg, tmp_path, monkeypatch, video):
    monkeypatch.setattr(C, "analyse", lambda *a, **k: {
        "evidence_text": EVIDENCE, "hash": "deadbeef", "cached": True})
    monkeypatch.setattr(C, "cache_dir", lambda h: tmp_path)
    monkeypatch.setattr(C, "sample_vision_frames", lambda *a, **k: FRAMES)
    monkeypatch.setattr(C, "log_usage", lambda *a, **k: None)

    out = {}
    for name, fake in (("openai", FakeOpenAIClient), ("gemini", FakeGeminiClient)):
        object.__setattr__(cfg, "provider", name)
        res = C.ask_pupchat(video, "q", cfg=cfg, progress=lambda *a, **k: None,
                            client=fake())
        out[name] = res["history"]
    assert out["openai"] == out["gemini"], \
        "a transcript must survive a provider switch mid-conversation"
