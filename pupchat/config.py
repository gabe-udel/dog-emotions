"""Every tunable for the Pupchat prototype, in one place.

Nothing here imports the vision stack, so tests and the evidence layer can load the
config without pulling in torch.

The model name is a config value and never a literal in the call site, so swapping
models is an edit here rather than a search through the code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "pupchat"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system.md"

ERROR_JSON = ROOT / "outputs" / "feature_error.json"
SNR_JSON = ROOT / "outputs" / "feature_snr.json"

DEFAULT_SNAPSHOT = ROOT / "face_project" / "face38" / "snapshot-best-008.pt"

# Non-commercial weights. Said out loud in the CLI and the web page, not only in docs.
LICENCE_NOTICE = (
    "Prototype only. DogFLW is CC BY-NC 4.0 and SuperAnimal follows the DeepLabCut "
    "Model Zoo terms, so this must not be deployed publicly or used commercially. "
    "Educational guidance only - not a vet or a certified behaviourist."
)


@dataclass(frozen=True)
class VisionConfig:
    """Frames sent to OpenAI. Separate from the keypoint pass on purpose.

    These two rates answer different questions and must not be conflated. The keypoint
    pass needs enough temporal resolution for per-second derivatives; the vision pass
    needs enough frames to see what happened without blowing the token budget.
    """

    fps_short: float = 2.0
    """Sampling rate for clips under `short_clip_s`."""

    fps_long: float = 1.0
    """Sampling rate for longer clips."""

    short_clip_s: float = 15.0

    max_frames: int = 32
    """Hard cap regardless of rate. The API allows 1,500 images per request; this is a
    cost ceiling, not an API limit."""

    jpeg_quality: int = 80
    width: int = 768
    """Frames are resized before base64 encoding. Vision models tile large images, so
    sending 1080p costs tokens without adding detail at this task."""

    def fps_for(self, duration_s: float) -> float:
        return self.fps_short if duration_s < self.short_clip_s else self.fps_long


@dataclass(frozen=True)
class KeypointConfig:
    """The pass that produces measurements. Slow: roughly 1.2 s per frame on CPU."""

    target_fps: float = 10.0
    """Source video is strided down to about this rate. The STRIDED rate is what gets
    handed to aggregate(), because every per-second derivative and the oscillation rate
    are computed from it - passing the source fps would scale them by the stride."""

    max_frames: int = 120
    """At ~1.2 s/frame this bounds a run at about 2.5 minutes."""

    width: int = 960
    pcut: float = 0.35
    """Confidence cut for FEATURES. Display cutoffs live elsewhere and must not be
    confused with this one."""

    ransac: bool = True
    ransac_ms: float = 230.0
    ransac_thresh: float = 0.05


@dataclass(frozen=True)
class SmoothingConfig:
    """One Euro over keypoint tracks, before features are computed.

    Defaults chosen from a measured sweep, NOT from the paper. On this data the
    trade-off is direct and there is no free lunch:

        beta   jitter vs raw   3 Hz kept   5 Hz kept
        0.7        -17%           58%         43%     <- flattens fast motion
        2.0        -10%           81%         71%
        5.0         -5%           92%         87%     <- default
       20.0         -1%           98%         96%

    beta=0.7 cuts jitter most but keeps less of a 3 Hz wag than the rolling median
    already in the pipeline (68.6%), which would make it a regression. beta=5.0 is the
    setting that satisfies "do not flatten fast motion" while still removing something.
    The honest summary is that the gain is small: see docs/PUPCHAT_PROTOTYPE.md.
    """

    enabled: bool = True
    min_cutoff: float = 0.5
    beta: float = 5.0
    d_cutoff: float = 1.0


@dataclass(frozen=True)
class SpikeConfig:
    """Flagging frames where a feature jumps and snaps back."""

    enabled: bool = True
    sigmas: float = 4.0
    """Threshold as a multiple of the feature's OWN measured error."""
    window: int = 3
    """Frames either side used for the local baseline."""
    max_width: int = 3
    """Longer than this and it is a movement, not a spike."""


@dataclass(frozen=True)
class StillnessConfig:
    min_seconds: float = 2.0
    """A run shorter than this is not reported."""


@dataclass(frozen=True)
class EvidenceConfig:
    min_coverage: float = 0.40
    """Below this a feature is reported as unavailable, with a reason."""

    event_sigmas: float = 2.0
    """An excursion counts when the series leaves its clip median by more than this
    many times the feature's OWN measured error. The threshold comes from the
    measurement rather than from taste."""

    event_min_ms: float = 150.0
    """Shorter excursions are not reported; they are indistinguishable from jitter."""

    max_events_per_feature: int = 3
    max_words: int = 300


# Emotion and interpretation words the evidence text must never contain.
#
# The list is deliberately wider than "emotions": "relaxed", "alert" and "submissive"
# are interpretations wearing body-language clothes, and letting them through would
# hand the model a conclusion while pretending to hand it a measurement. Editable -
# tests read this list rather than a copy.
BANNED_WORDS: tuple[str, ...] = (
    # affect
    "happy", "happiness", "sad", "sadness", "angry", "anger", "anxious", "anxiety",
    "stressed", "stress", "fearful", "fear", "afraid", "scared", "frightened",
    "excited", "excitement", "calm", "content", "depressed", "distressed", "distress",
    "frustrated", "frustration", "nervous", "upset", "joyful", "joy", "playful",
    "aggressive", "aggression", "threatened", "comfortable", "uncomfortable",
    "confident", "insecure", "submissive", "dominant", "defensive",
    # interpretation wearing body-language clothes
    "relaxed", "tense", "alert", "arousal", "aroused", "agitated", "wary", "uneasy",
    "friendly", "hostile", "curious", "bored", "attentive",
    # clinical
    "pain", "painful", "injured", "injury", "sick", "illness", "distressing",
)


@dataclass(frozen=True)
class ProviderSettings:
    """One AI provider. The two differ in what media they can take, which is the whole
    reason the provider is swappable rather than just the model name."""

    name: str
    model: str
    key_env: str
    max_output_tokens: int = 3000
    """Generous because reasoning models spend this budget before the answer starts.
    Measured on the first real Gemini call: 682 thinking tokens against 214 visible,
    so a 900 cap truncated the answer mid-word."""

    timeout_s: float = 180.0

    def key(self) -> str:
        k = os.getenv(self.key_env, "").strip()
        if not k:
            raise SystemExit(
                f"{self.key_env} is not set.\n"
                f"  Create a .env file next to CLAUDE.md containing:\n"
                f"    {self.key_env}=...\n"
                f"  .env is gitignored. See docs/PUPCHAT_PROTOTYPE.md.")
        return k


def _openai_settings() -> ProviderSettings:
    return ProviderSettings(
        name="openai",
        model=os.getenv("PUPCHAT_OPENAI_MODEL",
                        os.getenv("PUPCHAT_MODEL", "gpt-6.1-sol")),
        key_env="OPENAI_API_KEY")


def _gemini_settings() -> ProviderSettings:
    return ProviderSettings(
        name="gemini",
        model=os.getenv("PUPCHAT_GEMINI_MODEL", "gemini-3.8-flash"),
        key_env="GEMINI_API_KEY")


# Kept as an alias so `cfg.openai` still reads naturally at the call sites.
OpenAIConfig = ProviderSettings


# USD per million tokens, for the local cost log only - a stale entry misreports the
# log and nothing else. Checked 2026-10-11 against
# https://developers.openai.com/api/docs/pricing and
# https://ai.google.dev/gemini-api/docs/pricing
PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-6.1-sol":  {"input": 2.00, "cached": 0.10, "output": 10.00},
    "gpt-6-astra":  {"input": 10.00, "cached": 1.00, "output": 50.00},
    "gpt-6-luna":   {"input": 0.10, "cached": 0.01, "output": 0.50},
    "gpt-4o":       {"input": 2.50, "cached": 1.25, "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15, "cached": 0.075, "output": 0.60},
    # Gemini. Flash prices are the promotional rate through 2026-12-31; they double
    # after that, so this table needs revisiting then.
    "gemini-3.8-flash":      {"input": 0.75, "cached": 0.075, "output": 3.75},
    "gemini-3.5-flash-lite": {"input": 0.30, "cached": 0.03, "output": 2.50},
    "gemini-3.1-pro-preview": {"input": 2.00, "cached": 0.20, "output": 12.00},
    "gemini-2.5-pro":        {"input": 1.25, "cached": 0.125, "output": 10.00},
}

USAGE_LOG = OUT_DIR / "usage.jsonl"

PROVIDERS = ("openai", "gemini")

# Set PUPCHAT_FREE_TIER=1 when the key is on a free tier. PRICING holds paid rates, so
# without this the usage log reports money that was never charged.
FREE_TIER = os.getenv("PUPCHAT_FREE_TIER", "").strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Config:
    vision: VisionConfig = field(default_factory=VisionConfig)
    keypoint: KeypointConfig = field(default_factory=KeypointConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    spikes: SpikeConfig = field(default_factory=SpikeConfig)
    stillness: StillnessConfig = field(default_factory=StillnessConfig)
    openai: ProviderSettings = field(default_factory=_openai_settings)
    gemini: ProviderSettings = field(default_factory=_gemini_settings)
    provider: str = field(
        default_factory=lambda: os.getenv("PUPCHAT_PROVIDER", "openai").lower())
    snapshot: Path = DEFAULT_SNAPSHOT

    @property
    def active(self) -> ProviderSettings:
        """Settings for the selected provider."""
        if self.provider not in PROVIDERS:
            raise SystemExit(
                f"unknown provider {self.provider!r}; expected one of {PROVIDERS}. "
                f"Set PUPCHAT_PROVIDER in .env or pass --provider.")
        return getattr(self, self.provider)


def load_env() -> None:
    """Read .env if present. Never raises - a missing file is normal until someone
    sets a key, and the error belongs at the point of use."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def system_prompt(path: Path = PROMPT_PATH) -> str:
    """The one prompt both providers use. Shared on purpose: swapping providers must
    change who interprets the video, never what they were told about it."""
    return path.read_text(encoding="utf-8")
