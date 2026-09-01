"""A small desktop window for running the DogFLW-trained model on a video.

Point-and-click alternative to run_video.py: pick a file, press Run, watch the log.
Inference happens in a subprocess on a worker thread so the window stays responsive
and can be cancelled; all output is streamed into the log pane.

    .venv\\Scripts\\python.exe src\\video_app.py
"""
from __future__ import annotations
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
WEIGHTS = ROOT / "model_weights"
FALLBACK_CONFIG = WEIGHTS / "pytorch_config.yaml"
SCRIPT = ROOT / "src" / "run_video.py"
OUTDIR = ROOT / "outputs"
VIDEO_TYPES = [("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.wmv"),
               ("All files", "*.*")]

# Friendly names, and a short tag that goes into the output filename so two models
# run on the same clip do not overwrite each other.
KNOWN = {
    "superanimal_quadruped_dogface_final":
        ("Fine-tuned  (sigma 17)  -  recommended, draws all 46", "s17"),
    "superanimal_quadruped_dogface_sigma8":
        ("Fine-tuned  (sigma 8)  -  hides its own weak points", "s8"),
    "superanimal_quadruped_hrnet_w32_dogface":
        ("Untrained warm start  (reference only)", "warmstart"),
}

# Display order, best first. sigma 17 leads on COVERAGE, which is what matters when a
# person is judging the output by eye. Measured on video, points drawn out of 46 at
# cutoff 0.30: sigma 17 draws 46.0, sigma 8 draws 33.5 - and of the 14 ear landmarks,
# 14.0 versus 5.0. sigma 8 is genuinely better calibrated (its confidence tracks
# correctness more sharply, bad/good 0.46 vs 0.75) and marginally better on eyes and
# nose, but it expresses that by scoring its weak channels so low they disappear from
# the display. Keep it for anyone who wants only high-trust points; not the default.
PREFERRED = ["s17", "s8", "warmstart"]


# There is no landmark-set control any more. It existed to hide points measured
# unlearnable, and ear_correct.py fixed the last of those - the ear region went from
# 0.0882 to 0.0631 NME, taking the two worst points to 0.066 and 0.080. All 46 face
# landmarks are now worth drawing, so keypoint_scheme.UNRELIABLE is empty and
# run_video.py's --landmarks flag has nothing to filter.

# (label, extra run_video.py flags). First entry is the default.
DISPLAY_MODES = [
    ("Points + colour legend", ["--bare", "--no-lines", "--legend"]),
    ("Points + legend + skeleton lines", ["--bare", "--legend"]),
    ("Everything: title, legend, face zoom", []),
    ("Side-by-side vs original", ["--compare", "--bare", "--no-lines", "--legend"]),
]


def discover_models() -> list[tuple[str, Path, Path, str]]:
    """(label, weights, config, tag) for every .pt in model_weights/.

    A model's config is <stem>.yaml when present, otherwise the shared
    pytorch_config.yaml - the architecture only has to match the checkpoint.
    """
    out = []
    for pt in sorted(WEIGHTS.glob("*.pt")):
        label, tag = KNOWN.get(pt.stem, (pt.stem, pt.stem[:12]))
        cfg = pt.with_suffix(".yaml")
        if not cfg.exists():
            cfg = FALLBACK_CONFIG
        out.append((label, pt, cfg, tag))
    # trained models first, warm start last
    out.sort(key=lambda r: (PREFERRED.index(r[3]) if r[3] in PREFERRED else len(PREFERRED),
                            r[0]))
    return out


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.video: Path | None = None
        self.out: Path | None = None
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue[str | None] = queue.Queue()

        root.title("Dog face keypoints")
        root.minsize(720, 520)
        pad = {"padx": 14, "pady": 6}

        head = ttk.Frame(root)
        head.pack(fill="x", **pad)
        ttk.Label(head, text="Dog face keypoints",
                  font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(head, foreground="#666",
                  text="SuperAnimal-Quadruped fine-tuned on DogFLW - 76 keypoints "
                       "(39 body + 37 facial).").pack(anchor="w")

        mf = ttk.LabelFrame(root, text="1. Model")
        mf.pack(fill="x", **pad)
        mrow = ttk.Frame(mf)
        mrow.pack(fill="x", padx=10, pady=10)
        self.models = discover_models()
        self.model_box = ttk.Combobox(mrow, state="readonly", width=46,
                                      values=[m[0] for m in self.models])
        if self.models:
            self.model_box.current(0)
        self.model_box.pack(side="left")
        self.model_box.bind("<<ComboboxSelected>>", self.on_model)
        ttk.Button(mrow, text="Refresh", command=self.refresh_models).pack(side="left", padx=8)
        self.lbl_model = ttk.Label(mf, foreground="#888", font=("Consolas", 8))
        self.lbl_model.pack(anchor="w", padx=10, pady=(0, 8))

        pick = ttk.LabelFrame(root, text="2. Choose a video")
        pick.pack(fill="x", **pad)
        row = ttk.Frame(pick)
        row.pack(fill="x", padx=10, pady=10)
        self.btn_pick = ttk.Button(row, text="Choose video…", command=self.choose)
        self.btn_pick.pack(side="left")
        self.lbl_file = ttk.Label(row, text="no file selected", foreground="#888")
        self.lbl_file.pack(side="left", padx=12)

        opts = ttk.LabelFrame(root, text="3. Options")
        opts.pack(fill="x", **pad)
        orow = ttk.Frame(opts)
        orow.pack(fill="x", padx=10, pady=10)
        ttk.Label(orow, text="Frames to process:").pack(side="left")
        self.frames = tk.StringVar(value="150")
        ttk.Spinbox(orow, from_=1, to=100000, increment=25, width=8,
                    textvariable=self.frames).pack(side="left", padx=(8, 18))
        ttk.Label(orow, text="Display:").pack(side="left")
        self.mode_box = ttk.Combobox(orow, state="readonly", width=34,
                                     values=[m[0] for m in DISPLAY_MODES])
        self.mode_box.current(0)
        self.mode_box.pack(side="left", padx=(8, 0))


        crow = ttk.Frame(opts)
        crow.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(crow, text="Hide keypoints below confidence:").pack(side="left")
        # 0.35 is safe again now that "Reliable 32" removes the ear contours by
        # measurement rather than by threshold: with sigma 17 + head-crop it still
        # draws ~29 of the 30 detected landmarks. On "All 46" it costs ~3 ear points.
        self.pcut = tk.DoubleVar(value=0.35)
        self.pcut_lbl = ttk.Label(crow, text="0.35", width=5,
                                  font=("Consolas", 9), foreground="#2f6fb0")
        ttk.Scale(crow, from_=0.05, to=0.95, orient="horizontal", length=190,
                  variable=self.pcut,
                  command=lambda v: self.pcut_lbl.configure(
                      text=f"{float(v):.2f}")).pack(side="left", padx=(8, 6))
        self.pcut_lbl.pack(side="left")
        hrow = ttk.Frame(opts)
        hrow.pack(fill="x", padx=10, pady=(2, 0))
        # Off by default. It measures ~18% better NME on the 30 reliable landmarks
        # (0.0377 -> 0.0310) but that is ~1.3 px on a 220 px face, while it costs a lot
        # of visible detail: with sub-pixel decoding on, pass 1 alone renders 44.4 of 46
        # face landmarks at distinct pixels and the head-cropped pass only 25.4.
        self.headcrop = tk.BooleanVar(value=False)
        ttk.Checkbutton(hrow, variable=self.headcrop,
                        text="Zoom to the head before placing face points (two-pass, slower)"
                        ).pack(side="left")
        self.ema = tk.BooleanVar(value=True)
        ttk.Checkbutton(hrow, variable=self.ema,
                        text="Steady jittery points").pack(side="left", padx=(18, 0))
        ttk.Label(opts, foreground="#888", justify="left",
                  text="Two-pass re-crops to the head so the face fills the share of the frame\n"
                       "the model trained on. ~18% better NME on the reliable landmarks, but\n"
                       "worth ~1 px on this footage - and it costs visible detail: 25 of 46\n"
                       "face points land on distinct pixels versus 44 with it off. Leave it off\n"
                       "unless you are measuring rather than looking.\n"
                  ).pack(anchor="w", padx=10, pady=(0, 2))
        ttk.Label(opts, foreground="#888", justify="left",
                  text="A visibility control, NOT a quality knob - measured on video, confidence\n"
                       "runs opposite to accuracy. Raising it hides weak landmarks rather than\n"
                       "improving them. With the settings above, 0.20 draws all 46 points; 0.30\n"
                       "drops about 1 ear landmark in 10 and 0.35 drops 3 in 14.\n"
                       "Roughly 1 second per frame; side-by-side runs both models, so about double."
                  ).pack(anchor="w", padx=10, pady=(0, 8))
        self.compare = None      # superseded by the Display mode above

        run = ttk.Frame(root)
        run.pack(fill="x", **pad)
        self.btn_run = ttk.Button(run, text="Run", command=self.start, state="disabled")
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(run, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=8)
        self.btn_open = ttk.Button(run, text="Open result", command=self.open_result,
                                   state="disabled")
        self.btn_open.pack(side="left")
        self.bar = ttk.Progressbar(run, mode="indeterminate", length=180)
        self.bar.pack(side="right")

        logf = ttk.LabelFrame(root, text="Log")
        logf.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(logf, height=14, wrap="word", font=("Consolas", 9),
                           background="#12141b", foreground="#d6dae4",
                           insertbackground="#d6dae4", relief="flat")
        self.log.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        sb.pack(side="right", fill="y", pady=8, padx=(0, 8))
        self.log.configure(yscrollcommand=sb.set, state="disabled")

        self.status = ttk.Label(root, text="Ready", foreground="#666", anchor="w")
        self.status.pack(fill="x", padx=14, pady=(0, 10))

        missing = [p for p in (PY, SCRIPT) if not p.exists()]
        if missing or not self.models:
            # A fresh clone has neither the venv nor the checkpoints - the .pt files are
            # 113 MB, over GitHub's per-file limit, so they ship as a Release instead.
            # Say exactly which piece is absent and where it comes from.
            if not PY.exists():
                self.write(
                    "The Python environment is not built yet.\n\n"
                    "  Close this window and double-click  setup.bat\n"
                    "  (one time, 5-15 minutes - it downloads about 1 GB)\n\n")
            elif not self.models:
                self.write(
                    f"No model files (.pt) found in:\n  {WEIGHTS}\n\n"
                    "The trained model is 113 MB, over GitHub's 100 MB file limit, so it\n"
                    "is not in the repository. Get it from the Releases page:\n\n"
                    "  https://github.com/gabe-udel/dog-emotions/releases\n\n"
                    "Download  superanimal_quadruped_dogface_final.zip,  unzip it, and put\n"
                    "the .pt file inside into the folder above, then reopen this app.\n"
                    "(It is zipped only because GitHub Releases rejects the .pt extension.)\n"
                    "The .yaml config it needs is already in the repository.\n\n")
            for p in missing:
                self.write(f"Missing: {p}\n")
            self.btn_pick.configure(state="disabled")
            self.model_box.configure(state="disabled")
            self.set_status("Setup incomplete - see the log", "#b3402f")
        else:
            self.write(f"{len(self.models)} model(s) available. Output: {OUTDIR}\n\n"
                       f"Choose a video to begin.\n")
        self.on_model()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def current_model(self):
        i = self.model_box.current()
        return self.models[i] if 0 <= i < len(self.models) else None

    def on_model(self, _evt=None):
        m = self.current_model()
        if not m:
            self.lbl_model.configure(text="")
            return
        _, pt, cfg, _tag = m
        self.lbl_model.configure(
            text=f"{pt.name}  ({pt.stat().st_size/1e6:.0f} MB)   config: {cfg.name}")

    def refresh_models(self):
        """Re-scan model_weights/ - lets a training run that finished mid-session appear."""
        keep = self.model_box.get()
        self.models = discover_models()
        self.model_box.configure(values=[m[0] for m in self.models])
        if self.models:
            labels = [m[0] for m in self.models]
            self.model_box.current(labels.index(keep) if keep in labels else 0)
            self.write(f"\nFound {len(self.models)} model(s): "
                       f"{', '.join(m[1].name for m in self.models)}\n")
        self.on_model()

    # ---- helpers ----
    def write(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_status(self, text: str, color: str = "#666"):
        self.status.configure(text=text, foreground=color)

    # ---- actions ----
    def choose(self):
        p = filedialog.askopenfilename(title="Choose a video", filetypes=VIDEO_TYPES)
        if not p:
            return
        self.video = Path(p)
        self.lbl_file.configure(text=self.video.name, foreground="#000")
        self.btn_run.configure(state="normal")
        self.btn_open.configure(state="disabled")
        self.write(f"\nSelected: {self.video}\n")
        self.set_status("Ready to run")

    def start(self):
        if not self.video:
            return
        try:
            n = int(self.frames.get())
            if n < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Frames", "Frames to process must be a whole number of 1 or more.")
            return

        m = self.current_model()
        if not m:
            messagebox.showerror("Model", "No model selected.")
            return
        _label, weights, config, tag = m
        # every option that changes the output goes into the filename, so runs with
        # different settings sit side by side instead of overwriting each other
        if self.headcrop.get():
            tag += "_hc"

        OUTDIR.mkdir(parents=True, exist_ok=True)
        # tag in the name so running two models on one clip keeps both results
        self.out = OUTDIR / f"{self.video.stem}_{tag}.mp4"
        cmd = [str(PY), str(SCRIPT), "--video", str(self.video), "--out", str(self.out),
               "--config", str(config), "--snapshot", str(weights),
               "--width", "960", "--smooth", "3", "--max-frames", str(n),
               "--pcutoff", f"{self.pcut.get():.2f}"]
        mi = self.mode_box.current()
        cmd += DISPLAY_MODES[mi if 0 <= mi < len(DISPLAY_MODES) else 0][1]
        if self.headcrop.get():
            # 0.55 = measured optimum; the scale gate skips frames already near it
            cmd += ["--head-crop", "0.55", "--gate", "scale"]
        if self.ema.get():
            cmd += ["--ema", "0.5"]

        self.btn_run.configure(state="disabled")
        self.btn_pick.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_open.configure(state="disabled")
        self.bar.start(12)
        self.set_status("Running - loading the models takes a few seconds", "#2f6fb0")
        self.write(f"\n{'-'*62}\nRunning {n} frames -> {self.out.name}\n{'-'*62}\n")

        threading.Thread(target=self.worker, args=(cmd,), daemon=True).start()
        self.root.after(80, self.drain)

    def worker(self, cmd: list[str]):
        # 3, not 6: this often runs while a training job already holds most of the
        # cores. Oversubscribing makes both slower. Inference on a short clip is not
        # the bottleneck in this workflow, so yield the cores.
        env = dict(os.environ, OMP_NUM_THREADS="3", PYTHONUNBUFFERED="1")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                encoding="utf-8", errors="replace", creationflags=flags)
            for line in self.proc.stdout:
                self.q.put(line)
            self.proc.wait()
            self.q.put(f"\n[exit code {self.proc.returncode}]\n")
        except Exception as e:                      # surface it in the log, never a traceback dialog
            self.q.put(f"\nFailed to start: {e}\n")
        finally:
            self.q.put(None)

    def drain(self):
        done = False
        try:
            while True:
                item = self.q.get_nowait()
                if item is None:
                    done = True
                    break
                self.write(item)
        except queue.Empty:
            pass
        if done:
            self.finish()
        else:
            self.root.after(80, self.drain)

    def finish(self):
        self.bar.stop()
        self.btn_run.configure(state="normal")
        self.btn_pick.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        rc = self.proc.returncode if self.proc else -1
        if rc == 0 and self.out and self.out.exists():
            mb = self.out.stat().st_size / 1e6
            self.write(f"\nWrote {self.out}  ({mb:.1f} MB)\n")
            self.btn_open.configure(state="normal")
            self.set_status(f"Done - {self.out.name}", "#1f7a52")
        elif rc and rc < 0:
            self.set_status("Stopped", "#666")
        else:
            self.set_status("Failed - see the log above", "#b3402f")
        self.proc = None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.write("\nStopping…\n")
            self.proc.terminate()

    def open_result(self):
        if self.out and self.out.exists():
            os.startfile(str(self.out)) if sys.platform == "win32" else \
                subprocess.run(["xdg-open", str(self.out)], check=False)

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askokcancel("Quit", "A run is in progress. Stop it and quit?"):
                return
            self.proc.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
