"""Extract DogFLW from the Kaggle zip: PNG -> JPEG (native resolution, coords unchanged)."""
import io, json, sys, zipfile
from pathlib import Path
from PIL import Image

ZIP = Path("/tmp/dogflw.zip")
OUT = Path("data/dogflw")

def main():
    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    recs = []
    for split in ("train", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
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
            im.save(OUT / rel, "JPEG", quality=95, subsampling=0)
            recs.append({
                "id": stem, "split": split, "file": rel,
                "width": im.width, "height": im.height,
                "landmarks": lms,
                "bbox_xyxy": bbox, "bbox_src": bbox_src,
            })
            if i % 500 == 0:
                print(f"{split} {i}/{len(imgs)}", flush=True)
    (OUT / "annotations.json").write_text(json.dumps(recs))
    n_tr = sum(r["split"] == "train" for r in recs)
    print(f"DONE train={n_tr} test={len(recs)-n_tr} total={len(recs)}")


if __name__ == "__main__":
    main()
