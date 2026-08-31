"""Extract DogFLW from the Kaggle zip: PNG -> JPEG (native resolution, coords unchanged)."""
import argparse, io, json, sys, zipfile
from pathlib import Path
from PIL import Image

# Default lives inside the project rather than /tmp so it works on Windows too.
ZIP = Path("data/dogflw.zip")
OUT = Path("data/dogflw")
KAGGLE_URL = "https://www.kaggle.com/datasets/georgemartvel/dogflw"


def find_zip() -> Path | None:
    """Look where a browser download would plausibly have left the Kaggle zip."""
    seen = set()
    for d in (Path("data"), Path("."), Path.home() / "Downloads", Path.home() / "Desktop"):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.zip")):
            if p in seen:
                continue
            seen.add(p)
            if "dogflw" in p.name.lower():
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=None,
                    help=f"DogFLW Kaggle zip (default: {ZIP}, else auto-found in Downloads)")
    ap.add_argument("--out", type=Path, default=OUT, help=f"output dir (default: {OUT})")
    args = ap.parse_args()
    out = args.out

    zip_path = args.zip or (ZIP if ZIP.exists() else find_zip())
    if zip_path is None or not zip_path.exists():
        sys.exit(f"DogFLW zip not found (looked in data\\, .\\, ~/Downloads, ~/Desktop).\n"
                 f"Download it from {KAGGLE_URL}\n"
                 f"then re-run, or pass --zip <path>.")
    print(f"reading {zip_path}  ({zip_path.stat().st_size/1e6:.0f} MB)", flush=True)
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()
    recs = []
    for split in ("train", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        imgs = sorted(n for n in names if f"/{split}/images/" in n and n.endswith(".png"))
        for i, n in enumerate(imgs):
            stem = Path(n).stem
            lab = f"DogFLW/{split}/labels/{stem}.json"
            if lab not in names:
                print("MISSING LABEL", lab); continue
            ann = json.loads(z.read(lab))
            lms = [[float(x), float(y)] for x, y in ann["landmarks"]]
            try:
                bbox = [float(v) for v in ann["bounding_boxes"]]
                bbox_src = "dataset"
            except ValueError:
                # 3 images ship an empty bbox; rebuild it from the landmark hull
                # with the ~10% margin the dataset documents.
                xs = [x for x, _ in lms]; ys = [y for _, y in lms]
                w, h = max(xs) - min(xs), max(ys) - min(ys)
                bbox = [min(xs) - .1 * w, min(ys) - .1 * h,
                        max(xs) + .1 * w, max(ys) + .1 * h]
                bbox_src = "derived"
            im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
            rel = f"images/{split}/{stem}.jpg"
            im.save(out / rel, "JPEG", quality=95, subsampling=0)
            recs.append({
                "id": stem, "split": split, "file": rel,
                "width": im.width, "height": im.height,
                "landmarks": lms,
                "bbox_xyxy": bbox, "bbox_src": bbox_src,
            })
            if i % 500 == 0:
                print(f"{split} {i}/{len(imgs)}", flush=True)
    (out / "annotations.json").write_text(json.dumps(recs))
    n_tr = sum(r["split"] == "train" for r in recs)
    print(f"DONE train={n_tr} test={len(recs)-n_tr} total={len(recs)}")


if __name__ == "__main__":
    main()
