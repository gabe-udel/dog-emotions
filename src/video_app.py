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
FACE_PROJECT = ROOT / "face_project"
SCRIPT = ROOT / "src" / "run_video.py"
OUTDIR = ROOT / "outputs"
VIDEO_TYPES = [("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.wmv"),
               ("All files", "*.*")]

# (label, extra run_video.py flags). First entry is the default.
#
# The old build offered a "side-by-side vs original" mode and a head-crop toggle. Both
# belonged to the unified 76-channel model: the comparison contrasted it with stock
# SuperAnimal, and the head crop was an optional second pass. The cascade always runs a
# face crop and always runs stock SuperAnimal for the body, so neither has a job.
DISPLAY_MODES = [
    ("Points + colour legend", ["--no-lines"]),
    ("Points + legend + contour lines", []),
    ("Face only, no body keypoints", ["--no-lines", "--no-body"]),
]


def discover_models() -> list[tuple[str, Path, Path, str]]:
    """(label, snapshot, config, tag) for every face-model checkpoint on disk.

    Two places are searched: `face_project/<run>/` where training writes its snapshots
    alongside the pytorch_config.yaml that describes them, and `model_weights/` for a
    released checkpoint that ships with a matching .yaml. A snapshot is only offered if
    its config can be found - the architecture has to match or loading fails obscurely.
    """
    out: list[tuple[str, Path, Path, str]] = []
    for run in sorted(FACE_PROJECT.glob("*/")):
        cfg = run / "pytorch_config.yaml"
        if not cfg.exists():
            continue
        for pt in sorted(run.glob("snapshot-*.pt"), reverse=True):
            out.append((f"{run.name} / {pt.stem}", pt, cfg, f"{run.name}-{pt.stem}"))
    for pt in sorted(WEIGHTS.glob("*.pt")):
        cfg = pt.with_suffix(".yaml")
        if cfg.exists():
            out.append((f"released / {pt.stem}", pt, cfg, pt.stem[:16]))
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
        # Both corrections were fitted against the OLD unified model and have not been
        # re-measured on the cascade, so both start off. Turning one on is an
        # experiment, not an improvement, until evaluate_face.py says otherwise on val.
        self.ear_correct = tk.BooleanVar(value=False)
        ttk.Checkbutton(hrow, variable=self.ear_correct,
                        text="Ear-type bias correction (unmeasured on this model)"
                        ).pack(side="left")
        self.shape_refine = tk.BooleanVar(value=False)
        ttk.Checkbutton(hrow, variable=self.shape_refine,
                        text="Shape model for skull top (unmeasured)"
                        ).pack(side="left", padx=(18, 0))
        ttk.Label(opts, foreground="#888", justify="left",
                  text="The two checkboxes are carried over from the previous architecture and are\n"
                       "OFF until re-measured. Both learn a correction to one specific model's\n"
                       "errors, and this is a different model - fit them with postfit.py and score\n"
                       "them with evaluate_face.py --split val before trusting either.\n"
                  ).pack(anchor="w", padx=10, pady=(0, 2))
        ttk.Label(opts, foreground="#888", justify="left",
                  text="Confidence is a visibility control, NOT a quality knob - on the previous\n"
                       "model it ran opposite to accuracy across crop scales. Raising it hides\n"
                       "weak landmarks rather than improving them. That relationship has not been\n"
                       "re-checked on the cascade either.\n"
                       "Roughly 2 seconds per frame: two HRNet passes, body then face."
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
                    "No face-model checkpoint found. Two places are searched:\n\n"
                    f"  {FACE_PROJECT}\\<run>\\snapshot-*.pt   (written by training)\n"
                    f"  {WEIGHTS}\\*.pt                        (a released checkpoint)\n\n"
                    "A snapshot is only listed when a pytorch_config.yaml sits beside it,\n"
                    "because the architecture has to match the weights.\n\n"
                    "To train one (about a day on this CPU):\n\n"
                    "  .venv\\Scripts\\python.exe src\\splits.py\n"
                    "  .venv\\Scripts\\python.exe src\\build_face_coco.py\n"
                    "  .venv\\Scripts\\python.exe src\\train_face.py --run-name face1\n\n"
                    "Or download a released checkpoint into model_weights\\ along with its\n"
                    ".yaml:\n\n  https://github.com/gabe-udel/dog-emotions/releases\n\n")
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
        if self.ear_correct.get():
            tag += "_ear"
        if self.shape_refine.get():
            tag += "_shape"

        OUTDIR.mkdir(parents=True, exist_ok=True)
        self.out = OUTDIR / f"{self.video.stem}_{tag}.mp4"
        cmd = [str(PY), str(SCRIPT), "--video", str(self.video), "--out", str(self.out),
               "--config", str(config), "--snapshot", str(weights),
               "--width", "960", "--smooth", "3", "--max-frames", str(n),
               "--pcut", f"{self.pcut.get():.2f}"]
        mi = self.mode_box.current()
        cmd += DISPLAY_MODES[mi if 0 <= mi < len(DISPLAY_MODES) else 0][1]
        if self.ear_correct.get():
            cmd.append("--ear-correct")
        if self.shape_refine.get():
            cmd.append("--shape-refine")

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
