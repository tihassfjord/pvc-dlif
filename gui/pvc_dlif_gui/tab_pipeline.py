"""Pipeline tab: run stages 00-08 against a config file.

Each stage is the same ``scripts/NN_*.py`` you would run from a terminal - the
GUI only builds the command line and streams the output back.  That keeps one
implementation of the science and makes anything you do here reproducible by
copying the printed command.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .jobs import ProcessJob, python_exe
from .widgets import LabelledEntry, LogPane, PathPicker, ToolTip

# (id, script, label, tooltip)
STAGES = [
    ("00", "00_build_manifest.py", "Inventory dataset",
     "Scans the archive and writes manifest.json: what exists, what is usable, and why not."),
    ("01", "01_prepare_data.py", "Convert + fit crop",
     "DICOM -> 4D NIfTI at native resolution, and recovers the per-scan crop window."),
    ("02", "02_run_pvc.py", "Partial volume correction",
     "RL and RVC over every scan, plus per-frame recovery/noise diagnostics."),
    ("03", "03_make_dlif_inputs.py", "Build network inputs",
     "One input tree per experimental condition, all sharing the uncorrected crop."),
    ("04", "04_infer_baseline.py", "Pretrained inference",
     "Predictions from the deployed DLIF model."),
    ("05", "05_retrain.py", "Retrain",
     "Cross-validated retraining per condition. Long; wants a GPU."),
    ("06", "06_evaluate.py", "Evaluate",
     "Curve metrics, bias/variance, paired statistics, kinetics."),
    ("07", "07_motion_subset.py", "Motion subset",
     "Exploratory: detection, correction, and the cost of resampling."),
    ("08", "08_report.py", "Report",
     "Thesis figures and \\input-ready LaTeX tables."),
]


def _pilot_args(stage_id: str) -> list[str]:
    """The per-stage pilot settings defined in scripts/run_all.py."""
    import importlib.util
    import sys
    from pathlib import Path as _Path

    module_path = _Path(__file__).resolve().parents[2] / "scripts" / "run_all.py"
    spec = importlib.util.spec_from_file_location("_run_all", module_path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_run_all", module)
    spec.loader.exec_module(module)
    return list(getattr(module, "PILOT_ARGS", {}).get(stage_id, []))


class PipelineTab(ttk.Frame):

    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.job: ProcessJob | None = None
        self._queue: list[tuple[str, str]] = []      # stages still to run
        self._build()

    # ---------------------------------------------------------------- #
    def _build(self) -> None:
        top = ttk.LabelFrame(self, text="Configuration", padding=8)
        top.pack(fill="x")
        self.config_path = PathPicker(
            top, "Config", kind="file",
            value=str(self.app.repo_root / "configs" / "thesis.yaml"),
            filetypes=(("YAML", "*.yaml *.yml"), ("All", "*.*")),
            tooltip="The single source of truth. Every stage reads it; nothing is "
                    "hard-coded elsewhere.")
        self.config_path.pack(fill="x", pady=2)

        row = ttk.Frame(top)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Open in editor", command=self._open_config).pack(side="left")
        ttk.Button(row, text="Check paths", command=self._check_config).pack(side="left", padx=6)

        # --- stage selection ------------------------------------------
        mid = ttk.LabelFrame(self, text="Stages", padding=8)
        mid.pack(fill="x", pady=(8, 0))

        self.selected: dict[str, tk.BooleanVar] = {}
        for index, (stage_id, _script, label, tip) in enumerate(STAGES):
            var = tk.BooleanVar(value=True)
            self.selected[stage_id] = var
            check = ttk.Checkbutton(mid, text=f"{stage_id}  {label}", variable=var)
            check.grid(row=index % 5, column=index // 5, sticky="w", padx=(0, 24), pady=1)
            ToolTip(check, tip)

        buttons = ttk.Frame(mid)
        buttons.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(buttons, text="All", width=6,
                   command=lambda: self._set_all(True)).pack(side="left")
        ttk.Button(buttons, text="None", width=6,
                   command=lambda: self._set_all(False)).pack(side="left", padx=4)

        # --- options --------------------------------------------------
        opts = ttk.LabelFrame(self, text="Options", padding=8)
        opts.pack(fill="x", pady=(8, 0))
        line = ttk.Frame(opts)
        line.pack(fill="x")

        self.limit = LabelledEntry(line, "Limit scans", "", width=6,
                                   tooltip="Process at most N scans. Blank = all.")
        self.limit.pack(side="left")
        self.ids = LabelledEntry(line, "IDs", "", width=24,
                                 tooltip="Space-separated scan IDs, e.g. AA1 AA2. Blank = all.")
        self.ids.pack(side="left", padx=12)

        self.dry_run = tk.BooleanVar(value=False)
        dry = ttk.Checkbutton(line, text="Dry run", variable=self.dry_run)
        dry.pack(side="left", padx=8)
        ToolTip(dry, "Report what would happen without writing anything.")

        self.pilot = tk.BooleanVar(value=False)
        pilot = ttk.Checkbutton(line, text="Pilot", variable=self.pilot)
        pilot.pack(side="left")
        ToolTip(pilot, "Few scans, few folds, few epochs - minutes, to prove the wiring. "
                       "Pilot numbers are not results.")

        # --- run ------------------------------------------------------
        run_row = ttk.Frame(self)
        run_row.pack(fill="x", pady=8)
        self.run_button = ttk.Button(run_row, text="Run selected stages", command=self._run)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(run_row, text="Cancel", command=self._cancel,
                                        state="disabled")
        self.cancel_button.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(run_row, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=12)
        self.status = ttk.Label(run_row, text="idle")
        self.status.pack(side="left")

        self.log = LogPane(self, height=18)
        self.log.pack(fill="both", expand=True)

    # ---------------------------------------------------------------- #
    def _set_all(self, value: bool) -> None:
        for var in self.selected.values():
            var.set(value)

    def _open_config(self) -> None:
        """Hand the YAML to whatever the OS uses for text files."""
        import os
        import subprocess
        import sys
        path = self.config_path.path()
        if not path or not path.exists():
            messagebox.showwarning("Not found", "That config file does not exist.")
            return
        if sys.platform.startswith("win"):
            os.startfile(str(path))                      # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open" if sys.platform.startswith("linux") else "open", str(path)])

    def _check_config(self) -> None:
        """Load the config and report which of the four paths actually exist."""
        path = self.config_path.path()
        if not path or not path.exists():
            messagebox.showwarning("Not found", "Pick a config file first.")
            return
        self.log.clear()
        try:
            from pvc_dlif.config import load_config
            config = load_config(path)
        except Exception as exc:                          # noqa: BLE001
            self.log.write(f"ERROR: {exc}")
            return
        for name in ("dicom_root", "dlif_data_root", "dlif_repo", "work"):
            try:
                value = getattr(config, name)
            except Exception as exc:                      # noqa: BLE001
                self.log.write(f"{name:16s} ERROR: {exc}")
                continue
            mark = "ok    " if Path(value).exists() else "MISSING"
            self.log.write(f"{name:16s} {mark}  {value}")
        self.log.write("\nwork/ is created on demand; the other three must exist.")

    # ---------------------------------------------------------------- #
    def _stage_args(self, stage_id: str) -> list[str]:
        """Command-line arguments for one stage.

        Pilot mode reuses the per-stage settings from ``run_all.py`` rather than
        inventing its own, so the GUI's pilot is the same pilot.
        """
        args = ["--config", str(self.config_path.get())]
        if self.pilot.get():
            args += _pilot_args(stage_id)
        elif self.limit.get():
            args += ["--limit", self.limit.get()]
        if self.ids.get():
            args += ["--ids", *self.ids.get().split()]
        if self.dry_run.get():
            args += ["--dry-run"]
        return args

    def _run(self) -> None:
        chosen = [(sid, script) for sid, script, _l, _t in STAGES if self.selected[sid].get()]
        if not chosen:
            messagebox.showwarning("Nothing selected", "Tick at least one stage.")
            return
        if not self.config_path.path() or not self.config_path.path().exists():
            messagebox.showwarning("No config", "Pick a config file that exists.")
            return
        self.log.clear()
        self._queue = list(chosen)
        self._set_running(True)
        self._run_next()

    def _run_next(self) -> None:
        """Pop the next stage off the queue; stop the run if one fails."""
        if not self._queue:
            self._set_running(False)
            self.log.write("\nAll selected stages finished.", "ok")
            self.status.configure(text="done")
            self.app.analysis_tab.refresh_hint()
            return
        stage_id, script = self._queue.pop(0)
        self.status.configure(text=f"stage {stage_id}")
        self.log.write(f"\n===== stage {stage_id} =====")
        command = [python_exe(), str(self.app.repo_root / "scripts" / script),
                   *self._stage_args(stage_id)]
        self.job = ProcessJob(self.log.write, self._on_stage_done, self)
        self.job.start(command, cwd=self.app.repo_root)

    def _on_stage_done(self, ok: bool) -> None:
        if not ok:
            self._queue.clear()
            self._set_running(False)
            self.status.configure(text="failed")
            self.log.write("Stage failed - stopping here.", "error")
            return
        self._run_next()

    def _cancel(self) -> None:
        self._queue.clear()
        if self.job:
            self.job.cancel()
        self.status.configure(text="cancelling…")

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()
