"""High-resolution overlay of all 46 DogFLW landmark indices on one sample face.

CLAUDE.md 6 is explicit that the landmark *indices* are exact but the *names* in
keypoint_scheme.py are inferred from the mean face shape, because DogFLW ships no
landmark manual. This renders index -> position directly on a real dog so the naming can
actually be eyeballed and corrected.

46 labels on one face collide badly, so labels are pushed radially out from the face
centroid and then relaxed apart, with a leader line back to the true point.

  python src/show_landmark_indices.py                   # auto-pick a big frontal face
  python src/show_landmark_indices.py -r                 # one random face
  python src/show_landmark_indices.py -r -n 6            # six random faces
  python src/show_landmark_indices.py -r -n 6 --seed 0   # ...reproducibly
  python src/show_landmark_indices.py -r --ear-type pointy --split test
  python src/show_landmark_indices.py --id n02085620_1235,n02086646_3670

--min-face defaults to 220 px because DogFLW contains plenty of small thumbnails, and
uniform random sampling hits them often enough to be annoying; pass --min-face 0 to
sample the dataset as it really is.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "src")
from keypoint_scheme import DOGFLW_NAMES, REGION_OF
from draw import FACE_CONTOURS, REGION_COLOR

# draw.py stores BGR for OpenCV; matplotlib wants RGB 0-1.
RGB = {k: (b / 255, g / 255, r / 255) for k, (r, g, b) in REGION_COLOR.items()}
REGION_ORDER = ["ear", "head", "eye", "nose", "muzzle", "mouth"]


def pick_image(recs: list[dict]) -> dict:
    """Big face in *absolute pixels*, shot near head-on.

    Scoring on the face's share of the frame instead picks tiny thumbnails that happen to
    be all dog, which is the opposite of what a high-resolution overlay wants. There also
    has to be room around the bbox for the crop margin, so oversized faces are not better
    without bound.
    """
    best, best_score = None, -1.0
    for r in recs:
        lm = np.asarray(r["landmarks"], float)
        x1, y1, x2, y2 = r["bbox_xyxy"]
        face_px = min(x2 - x1, y2 - y1)                 # short side, in pixels
        room = min(1.0, min(r["width"], r["height"]) / (2.1 * face_px))
        # 18/19 = eye_right_lateral / eye_left_lateral, 32 = nose_bottom
        dr = np.linalg.norm(lm[18] - lm[32])
        dl = np.linalg.norm(lm[19] - lm[32])
        frontality = min(dr, dl) / max(dr, dl) if max(dr, dl) > 0 else 0.0
        score = face_px * room * frontality ** 3
        if score > best_score:
            best, best_score = r, score
    return best


def place_labels(pts: np.ndarray, radius: float, min_sep: float,
                 iters: int = 400) -> np.ndarray:
    """Nudge each label off its point, then relax only where labels actually collide.

    Labels are kept close to their own point rather than banished to a ring: with 46 dense
    landmarks, a radial starburst makes every leader line long and crossing, which is
    exactly what stops you tracing a number back to a dot.
    """
    c = pts.mean(axis=0)
    d = pts - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n[n == 0] = 1.0
    lab = pts + d / n * radius

    for _ in range(iters):
        moved = False
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                diff = lab[i] - lab[j]
                dist = np.hypot(*diff)
                if dist < min_sep:
                    if dist < 1e-6:
                        diff, dist = np.array([1.0, 0.0]), 1.0
                    push = (min_sep - dist) / 2 * diff / dist
                    lab[i] += push
                    lab[j] -= push
                    moved = True
        # a label must not drift so far that its leader line stops being readable
        off = lab - pts
        far = np.linalg.norm(off, axis=1)
        cap = radius * 3.2
        over = far > cap
        if over.any():
            lab[over] = pts[over] + off[over] / far[over, None] * cap
        if not moved:
            break
    return lab


def load_ear_types(path: Path) -> dict[str, str]:
    """image id -> ear_type, from data/ear_types.csv (built by src/ear_types.py)."""
    import csv
    with open(path, newline="", encoding="utf8") as fh:
        return {r["id"]: r["ear_type"] for r in csv.DictReader(fh)}


def face_px(rec: dict) -> float:
    x1, y1, x2, y2 = rec["bbox_xyxy"]
    return min(x2 - x1, y2 - y1)


def render(rec: dict, args, out_path: Path) -> None:
    lm = np.asarray(rec["landmarks"], float)
    assert lm.shape == (46, 2), lm.shape
    img = np.asarray(Image.open(args.root / rec["file"]).convert("RGB"))

    # crop generously around the face so the labels have somewhere to live
    x1, y1, x2, y2 = rec["bbox_xyxy"]
    mx, my = (x2 - x1) * args.margin, (y2 - y1) * args.margin
    cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    cx2, cy2 = int(min(img.shape[1], x2 + mx)), int(min(img.shape[0], y2 + my))
    img = img[cy1:cy2, cx1:cx2]
    pts = lm - [cx1, cy1]

    # scale off the real landmark spread: the annotation bbox ignores ears hanging far
    # below it, which is what blew the layout out on a long-eared breed.
    extent = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    radius = extent * 0.055
    min_sep = extent * 0.050
    labels = place_labels(pts, radius, min_sep)

    h, w = img.shape[:2]
    pad = extent * 0.06
    xlo = min(0.0, labels[:, 0].min() - pad)
    xhi = max(float(w), labels[:, 0].max() + pad)
    ylo = min(0.0, labels[:, 1].min() - pad)
    yhi = max(float(h), labels[:, 1].max() + pad)

    # Fixed panel width in inches so a 250 px face and a 1600 px face get labels of the
    # same *relative* size. Sizing the panel in pixels instead let the index bubbles -
    # which are sized in points - completely swamp any small image.
    PANEL_W, LEGEND_W = 9.5, 4.4
    panel_h = PANEL_W * (yhi - ylo) / (xhi - xlo)
    fig, (ax, axl) = plt.subplots(
        1, 2, figsize=(PANEL_W + LEGEND_W, max(panel_h, 8.4)),
        gridspec_kw={"width_ratios": [PANEL_W, LEGEND_W]})
    fig.patch.set_facecolor("#11131a")

    ax.imshow(img)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(yhi, ylo)
    ax.axis("off")

    for contour in FACE_CONTOURS:
        seg = pts[contour]
        ax.plot(seg[:, 0], seg[:, 1], "-", lw=1.5,
                color=RGB[REGION_OF[contour[0]]], alpha=0.55, zorder=2)

    for i in range(46):
        col = RGB[REGION_OF[i]]
        ax.plot([pts[i, 0], labels[i, 0]], [pts[i, 1], labels[i, 1]],
                "-", lw=0.8, color=col, alpha=0.5, zorder=3)
        ax.plot(*pts[i], "o", ms=6.5, mfc=col, mec="black", mew=1.2, zorder=4)
        ax.text(*labels[i], str(i), color="white", fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.28", fc=col, ec="black",
                          lw=1.0, alpha=0.95))

    ax.set_title(f"DogFLW landmark indices 0-45\n{rec['id']}  ({rec['split']} split)",
                 color="white", fontsize=13, pad=14)

    # ---- index -> inferred name, grouped by region ----
    axl.axis("off")
    axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    y = 0.985
    axl.text(0.0, y, "index → inferred name", color="white",
             fontsize=11.5, fontweight="bold", va="top")
    y -= 0.030
    axl.text(0.0, y, "names are derived, not authoritative (CLAUDE.md §6)",
             color="#96a0b5", fontsize=8.2, va="top", style="italic")
    y -= 0.032
    for region in REGION_ORDER:
        idxs = [i for i in range(46) if REGION_OF[i] == region]
        axl.text(0.0, y, f"{region}  ({len(idxs)})", color=RGB[region],
                 fontsize=10, fontweight="bold", va="top")
        y -= 0.023
        for i in idxs:
            axl.text(0.045, y, f"{i:>2}", color=RGB[region], fontsize=8.6,
                     va="top", family="monospace", fontweight="bold")
            axl.text(0.135, y, DOGFLW_NAMES[i], color="#d2d8e4", fontsize=8.6,
                     va="top", family="monospace")
            y -= 0.0205
        y -= 0.008

    # No swatch legend over the image: on a portrait crop it sat on the dog's face, and
    # the region headings in the right-hand panel are already colour-coded.

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight": the axis limits grow to fit the labels, which can otherwise
    # push the two-line title outside the canvas.
    fig.savefig(out_path, dpi=args.dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    px = Image.open(out_path).size
    print(f"  {rec['id']:<20} {rec['split']:<5} face {face_px(rec):>4.0f}px "
          f"-> {out_path}  ({px[0]}x{px[1]}, {out_path.stat().st_size/1e6:.1f} MB)")


def open_file(p: Path) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["cmd", "/c", "start", "", str(p)], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=False)
        else:
            subprocess.run(["xdg-open", str(p)], check=False)
    except Exception as e:          # display is a convenience, never fatal
        print(f"(could not open viewer: {e})")


def main():
    ap = argparse.ArgumentParser(
        description="Overlay the 46 DogFLW landmark indices on dataset images.")
    ap.add_argument("--id", default=None,
                    help="render this exact image id (repeatable via comma-separated list)")
    ap.add_argument("-r", "--random", action="store_true",
                    help="sample randomly instead of auto-picking the clearest face")
    ap.add_argument("-n", type=int, default=1, help="how many images to render")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for --random, so a run can be reproduced")
    ap.add_argument("--split", choices=["train", "test"], default=None)
    ap.add_argument("--ear-type", choices=["pointy", "floppy", "half_floppy"], default=None,
                    help="only sample this ear type (needs data/ear_types.csv)")
    ap.add_argument("--min-face", type=float, default=220,
                    help="skip faces whose short side is under this many pixels "
                         "(0 = no filter); random sampling otherwise finds thumbnails")
    ap.add_argument("--annotations", type=Path, default=Path("data/dogflw/annotations.json"))
    ap.add_argument("--root", type=Path, default=Path("data/dogflw"))
    ap.add_argument("--ear-csv", type=Path, default=Path("data/ear_types.csv"))
    ap.add_argument("--out", type=Path, default=None,
                    help="output path for a single image (default: --out-dir/<id>.png)")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/figures/indices"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--margin", type=float, default=0.16,
                    help="crop margin around the face bbox, as a fraction of its size")
    ap.add_argument("--no-show", action="store_true", help="do not open the PNG(s)")
    args = ap.parse_args()

    if not args.annotations.exists():
        sys.exit(f"not found: {args.annotations}\nRun src/extract_dogflw.py first.")
    recs = json.load(open(args.annotations))
    by_id = {r["id"]: r for r in recs}

    # ---- explicit ids win over any sampling ----
    if args.id:
        wanted = [s.strip() for s in args.id.split(",") if s.strip()]
        missing = [i for i in wanted if i not in by_id]
        if missing:
            sys.exit(f"no image(s) with id: {', '.join(missing)}")
        chosen = [by_id[i] for i in wanted]
    else:
        pool = recs
        if args.split:
            pool = [r for r in pool if r["split"] == args.split]
        if args.min_face:
            pool = [r for r in pool if face_px(r) >= args.min_face]
        if args.ear_type:
            if not args.ear_csv.exists():
                sys.exit(f"{args.ear_csv} not found - run src/ear_types.py first.")
            ear = load_ear_types(args.ear_csv)
            pool = [r for r in pool if ear.get(r["id"]) == args.ear_type]
        if not pool:
            sys.exit("no images match those filters (try lowering --min-face)")

        if args.random:
            import random
            rng = random.Random(args.seed)
            chosen = rng.sample(pool, min(args.n, len(pool)))
        else:
            # deterministic: the clearest faces first
            chosen = sorted(pool, key=lambda r: -face_px(r))[:args.n]
            if args.n == 1:
                chosen = [pick_image(pool)]

    filters = [f"split={args.split}" if args.split else "",
               f"ear={args.ear_type}" if args.ear_type else "",
               f"min_face={args.min_face:g}px" if args.min_face else ""]
    print(f"rendering {len(chosen)} image(s)  "
          f"[{', '.join(f for f in filters if f) or 'no filters'}]"
          f"{f'  seed={args.seed}' if args.random and args.seed is not None else ''}")

    written = []
    for rec in chosen:
        out = (args.out if args.out and len(chosen) == 1
               else args.out_dir / f"landmark_indices_{rec['id']}.png")
        render(rec, args, out)
        written.append(out)

    if not args.no_show:
        # opening a dozen viewer windows is worse than opening none
        if len(written) <= 3:
            for p in written:
                open_file(p)
        else:
            print(f"({len(written)} files written; not auto-opening. "
                  f"See {args.out_dir})")


if __name__ == "__main__":
    main()
