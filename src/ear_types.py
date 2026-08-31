"""Map every DogFLW image to its breed's ear type, from the zip's 'Ear Types.docx'.

DogFLW image ids are Stanford Dogs synset ids (n02085620_11477 -> n02085620 = Chihuahua),
and the Kaggle zip ships one undocumented extra file, 'DogFLW/Ear Types.docx', which
carries four Python list literals: `codes` (every breed) plus `dogs_with_pointy_ears`,
`dogs_with_floppy_ears` and `dogs_with_half_floppy_ears`.

Ear carriage is a real confound for this project: two of the nine merged SuperAnimal
keypoints are `ear_right_tip` / `ear_left_tip`, and a pricked Chihuahua ear and a dropped
Basset ear put that landmark in completely different places relative to the skull. This
gives you the stratifying variable to check whether error concentrates in one group.

The .docx is read straight out of the Kaggle zip so this needs no manual unzip step, and
parsed as XML (a .docx is a zip of XML) rather than pulling in python-docx.
"""
from __future__ import annotations
import argparse, ast, re, sys, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOCX_IN_ZIP = "DogFLW/Ear Types.docx"
EAR_LISTS = {
    "pointy": "dogs_with_pointy_ears",
    "floppy": "dogs_with_floppy_ears",
    "half_floppy": "dogs_with_half_floppy_ears",
}


def docx_text(data: bytes) -> str:
    """Visible paragraph text of a .docx supplied as bytes."""
    import io
    root = ET.fromstring(zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml"))
    return "\n".join(
        "".join(t.text or "" for t in p.iter(f"{W}t")) for p in root.iter(f"{W}p")
    )


def parse_list(text: str, name: str) -> list[str]:
    """Pull `name = [...]` out of the document text and literal-eval it."""
    m = re.search(rf"{name}\s*=\s*\[", text)
    if not m:
        raise KeyError(f"no list named {name!r} in the document")
    start = m.end() - 1
    depth, i = 0, start
    while i < len(text):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    else:
        raise ValueError(f"unterminated list literal for {name!r}")
    return list(ast.literal_eval(text[start:i + 1]))


def synset(code: str) -> str:
    """'n02085620-Chihuahua' or 'n02085620_11477' -> 'n02085620'."""
    return re.split(r"[-_]", code, maxsplit=1)[0]


def breed_table(text: str) -> pd.DataFrame:
    """One row per breed: synset, breed name, ear type."""
    codes = parse_list(text, "codes")
    rows = {synset(c): {"breed_code": c, "breed": c.split("-", 1)[1]} for c in codes}

    seen: dict[str, str] = {}
    for ear, list_name in EAR_LISTS.items():
        for c in parse_list(text, list_name):
            s = synset(c)
            if s in seen:
                print(f"  WARNING: {c} appears in both {seen[s]} and {ear}", file=sys.stderr)
            seen[s] = ear
            rows.setdefault(s, {"breed_code": c, "breed": c.split("-", 1)[1]})
            rows[s]["ear_type"] = ear

    df = pd.DataFrame.from_dict(rows, orient="index").rename_axis("synset").reset_index()
    if "ear_type" not in df:
        df["ear_type"] = pd.NA
    return df[["synset", "breed_code", "breed", "ear_type"]]


def main() -> pd.DataFrame:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=Path("archive.zip"),
                    help="DogFLW Kaggle zip containing 'Ear Types.docx'")
    ap.add_argument("--docx", type=Path, default=None,
                    help="use a standalone .docx instead of reading it from --zip")
    ap.add_argument("--annotations", type=Path,
                    default=Path("data/dogflw/annotations.json"),
                    help="DogFLW annotations.json from extract_dogflw.py")
    ap.add_argument("--out", type=Path, default=Path("data/ear_types.csv"))
    args = ap.parse_args()

    if args.docx:
        text = docx_text(args.docx.read_bytes())
    else:
        if not args.zip.exists():
            sys.exit(f"zip not found: {args.zip} (pass --zip or --docx)")
        text = docx_text(zipfile.ZipFile(args.zip).read(DOCX_IN_ZIP))

    breeds = breed_table(text)
    print(f"breeds in document: {len(breeds)}")
    print(breeds["ear_type"].value_counts(dropna=False).to_string())

    unclassified = breeds[breeds["ear_type"].isna()]
    if len(unclassified):
        print(f"\n{len(unclassified)} breed(s) in `codes` with no ear list:")
        print(unclassified[["synset", "breed"]].to_string(index=False))

    if not args.annotations.exists():
        sys.exit(f"\nannotations not found: {args.annotations}\n"
                 f"Run src/extract_dogflw.py first.")

    imgs = pd.read_json(args.annotations)[["id", "split", "file"]]
    imgs["synset"] = imgs["id"].map(synset)
    df = imgs.merge(breeds, on="synset", how="left")
    df["filename"] = df["file"].str.rsplit("/", n=1).str[-1]
    df = df[["filename", "id", "split", "synset", "breed", "ear_type", "file"]]

    print(f"\nimages: {len(df)}")
    print(df["ear_type"].value_counts(dropna=False).to_string())
    missing = df[df["ear_type"].isna()]
    if len(missing):
        print(f"\n{len(missing)} image(s) with no ear type, "
              f"{missing['synset'].nunique()} distinct breed(s):")
        print(missing.groupby(["synset", "breed"], dropna=False)
                     .size().rename("images").to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows)")
    print("\nhead:")
    print(df.head(8).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
