"""Build the small dog-emotion video dataset.

Downloads short clips from Pexels (free licence, no attribution required but
recorded anyway) into videos/{happy,sad,angry}/. Labels come from the search
query, not from any annotator, so treat them as weak labels -- see
videos/EMOTION_DATASET.md.

    .venv\\Scripts\\python.exe src\\fetch_emotion_videos.py
    .venv\\Scripts\\python.exe src\\fetch_emotion_videos.py --per-class 20 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_ROOT = ROOT / "videos"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Queries are ordered: the first one is the on-the-nose search, the rest widen
# the net so a class can still reach its quota when the literal query runs dry.
QUERIES: dict[str, list[str]] = {
    "happy": [
        "happy dog",
        "dog playing",
        "dog wagging tail",
        "excited dog",
        "playful puppy",
        "dog running grass",
    ],
    "sad": [
        "sad dog",
        "lonely dog",
        "dog lying down",
        "sleepy dog",
        "dog waiting window",
        "old dog resting",
    ],
    "angry": [
        "angry dog",
        "dog barking",
        "aggressive dog",
        "dog growling",
        "guard dog",
        "dog teeth",
    ],
}

# Face keypoints need pixels on the head, but 4K clips are a waste of disk for a
# demo. Prefer 720p, fall back to whatever is closest under the cap.
TARGET_W = 1280
MAX_W = 1920
MAX_BYTES = 120 * 1024 * 1024

FILE_RE = re.compile(
    r"https://videos\.pexels\.com/video-files/(\d+)/([^\"'\\ ]+?_(\d+)_(\d+)_\d+fps\.mp4)"
)
SLUG_RE = re.compile(r'"/video/([a-z0-9-]+?)-(\d+)/"')


@dataclass
class Clip:
    vid: str
    slug: str = ""
    query: str = ""
    # width -> url
    renditions: dict[int, str] = field(default_factory=dict)

    @property
    def page_url(self) -> str:
        return f"https://www.pexels.com/video/{self.slug}-{self.vid}/"

    def pick(self) -> tuple[int, str] | None:
        """Closest rendition to 720p that is not larger than MAX_W."""
        usable = {w: u for w, u in self.renditions.items() if w <= MAX_W}
        if not usable:
            return None
        width = min(usable, key=lambda w: (abs(w - TARGET_W), -w))
        return width, usable[width]


def _curl() -> str:
    """Pexels 403s Python's TLS fingerprint but serves curl, so shell out."""
    exe = shutil.which("curl") or r"C:\Windows\System32\curl.exe"
    if not Path(exe).exists():
        sys.exit("curl not found; it is required (Python's TLS stack is blocked)")
    return exe


CURL = _curl()


def get(url: str, dest: Path | None = None, timeout: int = 180) -> bytes:
    cmd = [CURL, "-sS", "-L", "--fail", "--max-time", str(timeout), "-A", UA]
    if dest is not None:
        cmd += ["-o", str(dest)]
    cmd.append(url)
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.decode("utf-8", "replace").strip() or f"curl exit {proc.returncode}"
        )
    return proc.stdout


def search(query: str, page: int) -> dict[str, Clip]:
    """One page of Pexels video search, parsed into Clips keyed by video id."""
    url = (
        "https://www.pexels.com/search/videos/"
        + urllib.parse.quote(query)
        + f"/?page={page}"
    )
    try:
        html = get(url, timeout=60).decode("utf-8", "replace")
    except (RuntimeError, OSError) as exc:
        print(f"    ! search failed ({query} p{page}): {exc}")
        return {}

    clips: dict[str, Clip] = {}
    for m in FILE_RE.finditer(html):
        vid, width = m.group(1), int(m.group(3))
        clip = clips.setdefault(vid, Clip(vid=vid, query=query))
        clip.renditions[width] = m.group(0)

    for slug, vid in SLUG_RE.findall(html):
        if vid in clips and not clips[vid].slug:
            clips[vid].slug = slug

    return clips


def download(url: str, dest: Path) -> int:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        get(url, dest=tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    size = tmp.stat().st_size
    if size > MAX_BYTES:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{size / 1e6:.0f} MB exceeds cap")
    if size < 10_000:
        tmp.unlink(missing_ok=True)
        raise ValueError("suspiciously small, likely an error page")
    tmp.replace(dest)
    return size


def collect(label: str, want: int, seen: set[str]) -> list[Clip]:
    """Walk this label's queries until `want` unique unseen clips are found."""
    picked: list[Clip] = []
    for query in QUERIES[label]:
        for page in (1, 2, 3):
            if len(picked) >= want:
                return picked
            found = search(query, page)
            if not found:
                break
            fresh = 0
            for vid, clip in found.items():
                if vid in seen or len(picked) >= want:
                    continue
                if clip.pick() is None:
                    continue
                seen.add(vid)
                picked.append(clip)
                fresh += 1
            print(f"    {query!r} p{page}: {len(found)} results, {fresh} new")
            if fresh == 0:
                break
            time.sleep(1.0)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true", help="search only, no download")
    ap.add_argument(
        "--classes", nargs="*", default=list(QUERIES), choices=list(QUERIES)
    )
    args = ap.parse_args()

    seen: set[str] = set()
    manifest: dict[str, list[dict]] = {}

    for label in args.classes:
        print(f"\n[{label}] searching for {args.per_class} clips")
        clips = collect(label, args.per_class, seen)
        print(f"[{label}] {len(clips)} clips selected")

        out_dir = VIDEO_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []

        for i, clip in enumerate(clips, 1):
            width, url = clip.pick()
            dest = out_dir / f"{label}_{i:02d}_{clip.vid}.mp4"
            row = {
                "file": dest.name,
                "label": label,
                "query": clip.query,
                "pexels_id": clip.vid,
                "page": clip.page_url,
                "download_url": url,
                "width": width,
            }
            if args.dry_run:
                print(f"    [dry] {dest.name} <- {width}px")
                rows.append(row)
                continue
            if dest.exists():
                print(f"    = {dest.name} (exists)")
                row["bytes"] = dest.stat().st_size
                rows.append(row)
                continue
            try:
                size = download(url, dest)
            except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the run
                print(f"    ! {dest.name}: {exc}")
                continue
            row["bytes"] = size
            rows.append(row)
            print(f"    + {dest.name}  {width}px  {size / 1e6:.1f} MB")

        manifest[label] = rows

    out = VIDEO_ROOT / "emotion_manifest.json"
    if not args.dry_run:
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nmanifest -> {out}")

    total = sum(len(v) for v in manifest.values())
    mb = sum(r.get("bytes", 0) for v in manifest.values() for r in v) / 1e6
    print(f"total {total} clips, {mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
