"""Hand-label the withers, the one keypoint StanfordExtra leaves empty.

StanfordExtra v12 carries a "Withers" slot at joint index 22 and never fills it - flag 0
in all 2,545 images that overlap DogFLW. That point is SuperAnimal's `back_base`, and
together with `back_end` it defines `L`, the body length every body feature divides by.
Without it, per-keypoint body accuracy is measurable but per-FEATURE accuracy is not.

Sixty clicks closes that gap.

Two rules this tool follows, and they decide whether the result means anything:

**SuperAnimal's prediction is never drawn.** Showing the model's guess for the very point
being labelled would anchor the click to it, and the run would measure agreement rather
than accuracy.

**Only ground truth is drawn as context.** The front leg tops and tail start come from
StanfordExtra's human annotations, so they orient the eye without leaking the answer.

    .venv\\Scripts\\python.exe src\\label_withers.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

import splits as splits_mod
import superanimal as sa

SE_JSON = "data/stanfordextra_v12/StanfordExtra_V12/StanfordExtra_v12.json"
CONTEXT = {2: ("front left top", "#4da3ff"), 8: ("front right top", "#4da3ff"),
           12: ("tail start", "#ffb347")}
MAX_W, MAX_H = 1000, 720

GUIDE = ("WITHERS = the ridge between the shoulder blades, where the neck meets the back.\n"
         "On a standing dog it is the highest point of the back, just behind the neck.\n"
         "Blue = front leg tops   Orange = tail start   (human annotations, for orientation)")


def build_pool(seed: int, n: int) -> list[tuple[str, Path, dict]]:
    """Images worth labelling: annotated elsewhere, and with a spine to compare against."""
    se = {Path(r["img_path"]).stem: r for r in json.loads(Path(SE_JSON).read_text())}
    recs = {r["id"]: r for r in json.loads(Path("data/dogflw/annotations.json").read_text())}
    z = np.load("data/sa_dogboxes.npz", allow_pickle=True)
    poses = {i: p for i, p in zip(z["ids"].tolist(), z["poses"])}
    SA = sa.sa_bodyparts()
    ib, ie = SA.index("back_base"), SA.index("back_end")
    sp = splits_mod.load("data/splits.json")

    def collect(ids) -> list:
        out = []
        for i in ids:
            if i not in se or i not in poses or i not in recs:
                continue
            p = poses[i]
            if p[ib, 2] < 0.35 or p[ie, 2] < 0.35:
                continue                  # nothing to score the click against
            f = Path("data/dogflw") / recs[i]["file"]
            if f.exists():
                out.append((i, f, se[i]))
        return out

    # Val first: every other measurement on this project is reported on val, and 167
    # images there clear the filters - well past the 60 needed. Other splits are only
    # drawn on if val cannot fill the quota. SuperAnimal is stock and never saw DogFLW,
    # so this is for comparability, not leakage.
    pool = collect(sp.val)
    if len(pool) < n:
        seen = {i for i, _, _ in pool}
        pool += [t for t in collect(list(sp.test) + list(sp.train)) if t[0] not in seen]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))[:n]
    return [pool[k] for k in sorted(idx)]


class Labeller:
    def __init__(self, root: tk.Tk, pool: list, out: Path) -> None:
        self.root, self.pool, self.out = root, pool, out
        self.labels: dict[str, dict] = {}
        if out.exists():
            self.labels = json.loads(out.read_text())
        self.i = 0
        self.point: tuple[float, float] | None = None
        self.scale = 1.0
        self.photo = None

        root.title("Withers labelling")
        tk.Label(root, text=GUIDE, justify="left", fg="#333",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(8, 4))
        self.canvas = tk.Canvas(root, bg="#222", highlightthickness=0)
        self.canvas.pack(padx=10)
        self.canvas.bind("<Button-1>", self.on_click)

        bar = tk.Frame(root)
        bar.pack(fill="x", padx=10, pady=8)
        self.status = tk.Label(bar, text="", font=("Segoe UI", 10, "bold"))
        self.status.pack(side="left")
        tk.Button(bar, text="Skip  (s)", command=self.skip).pack(side="right", padx=4)
        tk.Button(bar, text="Next  (enter)", command=self.nxt).pack(side="right", padx=4)
        tk.Button(bar, text="Back", command=self.prev).pack(side="right", padx=4)

        root.bind("<Return>", lambda e: self.nxt())
        root.bind("s", lambda e: self.skip())
        root.bind("<Escape>", lambda e: root.quit())
        self.show()

    # ---------------------------------------------------------------- display

    def show(self) -> None:
        while self.i < len(self.pool) and self.pool[self.i][0] in self.labels:
            self.i += 1
        if self.i >= len(self.pool):
            self.finish()
            return
        sid, path, se = self.pool[self.i]
        img = Image.open(path).convert("RGB")
        self.scale = min(MAX_W / img.width, MAX_H / img.height, 1.0)
        disp = img.resize((int(img.width * self.scale), int(img.height * self.scale)))
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.configure(width=disp.width, height=disp.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        J = np.asarray(se["joints"], dtype=float)
        for k, (name, colour) in CONTEXT.items():
            if J[k, 2] <= 0:
                continue
            x, y = J[k, 0] * self.scale, J[k, 1] * self.scale
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline=colour, width=2)
            self.canvas.create_text(x + 10, y - 10, text=name, fill=colour,
                                    anchor="w", font=("Segoe UI", 8))
        self.point = None
        done = len(self.labels)
        self.status.configure(
            text=f"{done}/{len(self.pool)} done   ·   {sid}   ·   click the withers")

    def on_click(self, ev) -> None:
        self.point = (ev.x / self.scale, ev.y / self.scale)
        self.canvas.delete("mark")
        self.canvas.create_line(ev.x - 12, ev.y, ev.x + 12, ev.y, fill="#39d353",
                                width=2, tags="mark")
        self.canvas.create_line(ev.x, ev.y - 12, ev.x, ev.y + 12, fill="#39d353",
                                width=2, tags="mark")

    # ---------------------------------------------------------------- actions

    def record(self, value: dict) -> None:
        self.labels[self.pool[self.i][0]] = value
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(self.labels, indent=2))   # save every click
        self.i += 1
        self.show()

    def nxt(self) -> None:
        if self.point is None:
            self.status.configure(text="click the withers first, or press s to skip")
            return
        self.record({"x": self.point[0], "y": self.point[1], "visible": True})

    def skip(self) -> None:
        self.record({"visible": False})

    def prev(self) -> None:
        if self.i > 0:
            self.i -= 1
            self.labels.pop(self.pool[self.i][0], None)
            self.out.write_text(json.dumps(self.labels, indent=2))
            self.show()

    def finish(self) -> None:
        n = sum(1 for v in self.labels.values() if v.get("visible"))
        self.canvas.delete("all")
        self.status.configure(text=f"done - {n} labelled, {len(self.labels) - n} skipped "
                                   f"-> {self.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/withers_labels.json")
    a = ap.parse_args()

    pool = build_pool(a.seed, a.n)
    if not pool:
        raise SystemExit("no usable images found")
    print(f"{len(pool)} images queued -> {a.out}")
    root = tk.Tk()
    Labeller(root, pool, Path(a.out))
    root.mainloop()


if __name__ == "__main__":
    main()
