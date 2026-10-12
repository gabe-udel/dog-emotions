"""`python -m pupchat ask --video <file> --question "<text>"`, then follow-ups."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import LICENCE_NOTICE, Config, load_env


def _utf8_stdout() -> None:
    """The model answers in Unicode; a cp1252 console turns its apostrophes and dashes
    into mojibake, or raises outright. Reconfigure rather than strip the characters."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _banner() -> None:
    print("=" * 74)
    print("Pupchat prototype")
    print(LICENCE_NOTICE)
    print("=" * 74)


def cmd_ask(a: argparse.Namespace) -> int:
    from .client import ask_pupchat

    cfg = Config()
    if a.provider:
        object.__setattr__(cfg, "provider", a.provider)
    if a.model:
        object.__setattr__(cfg.active, "model", a.model)

    _banner()
    if a.dry_run:
        # Build and print the request without sending it. Lets the exact payload be
        # inspected, and reviewed in a diff, without a key and without a network call.
        from .client import build_payload_only
        import json as _json
        print(_json.dumps(build_payload_only(a.video, a.question, cfg), indent=2))
        return 0
    res = ask_pupchat(a.video, a.question, cfg=cfg, show_request=a.show_request)

    if a.show_evidence:
        print("\n--- evidence ---")
        print(res["evidence_text"])
        print("--- end evidence ---")

    print("\n" + "=" * 74)
    print(res["answer"])
    print("=" * 74)
    c = res["cost_usd"]
    u = res["usage"]
    cost_s = f" | ${c:.4f}" if c == c else ""
    think = u.get("reasoning_tokens", 0)
    think_s = f" ({think} thinking)" if think else ""
    print(f"[{res['provider']}/{res['model']} | {res['seconds']:.1f}s | "
          f"{u.get('input_tokens', '?')} in / {u.get('output_tokens', '?')} out"
          f"{think_s}{cost_s}]")
    if res.get("truncated"):
        print("[WARNING] the answer was cut off at the token limit.")

    if a.no_follow_up:
        return 0

    history = res["history"]
    print("\nAsk a follow-up, or press Enter to finish. "
          "The video is not reprocessed.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            return 0
        res = ask_pupchat(a.video, q, history=history, cfg=cfg)
        history = res["history"]
        print("\n" + res["answer"])
        c = res["cost_usd"]
        if c == c:
            cached = (res["usage"].get("input_tokens_details") or {}).get(
                "cached_tokens", 0)
            print(f"\n[${c:.4f} | {cached} cached tokens]")


def cmd_evidence(a: argparse.Namespace) -> int:
    """Measurements only. Needs no API key, so it is the way to check the vision half."""
    from .pipeline import analyse

    _banner()
    res = analyse(a.video, Config(), force=a.force)
    print("\n" + res["evidence_text"])
    if a.json:
        print("\n" + json.dumps(res["evidence"], indent=2))
    print(f"\n[cached={res['cached']}]  {res['dir']}")
    return 0


def cmd_trace(a: argparse.Namespace) -> int:
    """Why each feature came out the way it did. Reads the cache, computes nothing."""
    from .pipeline import analyse
    from .trace import (frame_trace, keypoint_coverage, load_keypoints,
                        render_coverage, render_frame)

    cfg = Config()
    res = analyse(a.video, cfg, progress=print if a.verbose else (lambda *x, **k: None))
    kp_path = res.get("keypoints")
    if kp_path is None or not Path(kp_path).exists():
        print("This video was measured before keypoint caching existed.")
        print("Re-run with --force to regenerate:")
        print(f'  python -m pupchat evidence --video "{a.video}" --force')
        return 1

    kp = load_keypoints(kp_path)
    pcut = cfg.keypoint.pcut
    print(render_coverage(keypoint_coverage(kp, pcut), pcut))

    n = len(res["frames"])
    if a.time is not None:
        idx = int(round(a.time * float(res["meta"]["keypoint_fps"])))
    else:
        idx = a.frame if a.frame is not None else n // 2
    idx = max(0, min(n - 1, idx))

    print()
    print("=" * 56)
    print(render_frame(frame_trace(kp, res["frames"], idx, pcut), pcut,
                       only_failed=a.failed_only))
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    load_env()
    ap = argparse.ArgumentParser(prog="pupchat", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="ask a question about a video")
    ask.add_argument("--video", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--provider", default=None, choices=("openai", "gemini"),
                     help="overrides PUPCHAT_PROVIDER")
    ask.add_argument("--model", default=None, help="overrides the configured model")
    ask.add_argument("--show-evidence", action="store_true")
    ask.add_argument("--show-request", action="store_true",
                     help="print the request before sending it, media shortened")
    ask.add_argument("--no-follow-up", action="store_true")
    ask.add_argument("--dry-run", action="store_true",
                     help="build and print the request; send nothing, need no key")
    ask.set_defaults(func=cmd_ask)

    ev = sub.add_parser("evidence", help="measurements only; no API key needed")
    ev.add_argument("--video", required=True)
    ev.add_argument("--json", action="store_true")
    ev.add_argument("--force", action="store_true", help="ignore the cache")
    ev.set_defaults(func=cmd_evidence)

    tr = sub.add_parser("trace", help="show why each feature succeeded or failed")
    tr.add_argument("--video", required=True)
    tr.add_argument("--frame", type=int, default=None, help="frame index (default: mid)")
    tr.add_argument("--time", type=float, default=None, help="seconds; overrides --frame")
    tr.add_argument("--failed-only", action="store_true")
    tr.add_argument("--verbose", action="store_true")
    tr.set_defaults(func=cmd_trace)

    a = ap.parse_args(argv)
    if not Path(a.video).exists():
        print(f"no such file: {a.video}", file=sys.stderr)
        return 2
    return a.func(a)
