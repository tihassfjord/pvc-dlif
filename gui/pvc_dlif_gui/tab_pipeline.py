"""Pipeline tab: the nine stages, what each has produced, and running them.

Status comes from the files under ``<work>`` every time it is refreshed, never
from memory, so it survives crashes and stages run from a terminal.  Two
results are shown as coloured banners rather than log lines because they are
the two numbers the whole study rests on: how many scans are usable (stage
00), and how faithfully the rebuilt inputs reproduce the distributed ones
(stage 01).

Stages run as *detached* subprocesses (see ``detached.py``): closing the
window does not kill a run, and reopening it re-attaches to the log.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from . import detached
from .jobs import python_exe
from .widgets import Banner, LabelledEntry, LogPane, PathPicker, ToolTip

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
     "Cross-validated retraining per condition. Days on one GPU."),
    ("06", "06_evaluate.py", "Evaluate",
     "Curve metrics, bias/variance, paired statistics, kinetics."),
    ("07", "07_motion_subset.py", "Motion subset",
     "Exploratory: detection, correction, and the cost of resampling."),
    ("08", "08_report.py", "Report",
     "Thesis figures and \\input-ready LaTeX tables."),
]
SCRIPT_OF = {sid: script for sid, script, _l, _t in STAGES}

LIGHT = {"done": "●", "partial": "◐", "missing": "○", "running": "▶"}
LIGHT_COLOUR = {"done": "#2e7d32", "partial": "#e69500", "missing": "#999", "running": "#1565c0"}


def _pilot_args(stage_id: str) -> list[str]:
    """The per-stage pilot settings defined in scripts/run_all.py."""
    import importlib.util
    import sys

    module_path = Path(__file__).resolve().parents[2] / "scripts" / "run_all.py"
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
        self.state: dict | None = None          # the detached run being tailed
        self._tail: detached.LogTail | None = None
        self._queue: list[str] = []
        self._build()
        self.after(400, self.refresh_status)
        self.after(600, self._reattach)

    # ================================================================ #
    # Layout
    # ================================================================ #
    def _build(self) -> None:
        top = ttk.LabelFrame(self, text="Configuration", padding=8)
        top.pack(fill="x")
        self.config_path = PathPicker(
            top, "Config", kind="file",
            value=str(self.app.repo_root / "configs" / "thesis.yaml"),
            filetypes=(("YAML", "*.yaml *.yml"), ("All", "*.*")),
            tooltip="The single source of truth. Every stage reads it.")
        self.config_path.pack(fill="x", pady=2)
        self.config_path.var.trace_add("write", lambda *_: self.after(300, self.refresh_status))

        # ----- stage table --------------------------------------------
        mid = ttk.LabelFrame(self, text="Stages   (status is read from the work folder)", padding=8)
        mid.pack(fill="x", pady=(8, 0))

        # Banners live between the config and the stage list (packed on show)
        self.banner_manifest = Banner(self, before=mid)
        self.banner_prepare = Banner(self, before=mid)
        header = ("", "", "stage", "progress", "produced")
        for col, text in enumerate(header):
            ttk.Label(mid, text=text, foreground="#666").grid(row=0, column=col, sticky="w", padx=4)

        self.selected: dict[str, tk.BooleanVar] = {}
        self.lights: dict[str, tk.Label] = {}
        self.progress_labels: dict[str, ttk.Label] = {}
        self.detail_labels: dict[str, ttk.Label] = {}
        self.run_buttons: dict[str, ttk.Button] = {}
        for row, (stage_id, _script, label, tip) in enumerate(STAGES, start=1):
            var = tk.BooleanVar(value=True)
            self.selected[stage_id] = var
            ttk.Checkbutton(mid, variable=var).grid(row=row, column=0, sticky="w")
            light = tk.Label(mid, text=LIGHT["missing"], fg=LIGHT_COLOUR["missing"], font=("Segoe UI", 12))
            light.grid(row=row, column=1, sticky="w")
            self.lights[stage_id] = light
            name = ttk.Label(mid, text=f"{stage_id}  {label}", width=28, anchor="w")
            name.grid(row=row, column=2, sticky="w", padx=4)
            ToolTip(name, tip)
            self.progress_labels[stage_id] = ttk.Label(mid, text="", width=12, anchor="w")
            self.progress_labels[stage_id].grid(row=row, column=3, sticky="w", padx=4)
            self.detail_labels[stage_id] = ttk.Label(mid, text="", foreground="#555", anchor="w")
            self.detail_labels[stage_id].grid(row=row, column=4, sticky="w", padx=4)
            button = ttk.Button(mid, text="Run", width=6,
                                command=lambda s=stage_id: self._run_stages([s]))
            button.grid(row=row, column=5, sticky="e", padx=4)
            self.run_buttons[stage_id] = button
        mid.columnconfigure(4, weight=1)

        bar = ttk.Frame(mid)
        bar.grid(row=len(STAGES) + 1, column=0, columnspan=6, sticky="ew", pady=(6, 0))
        ttk.Button(bar, text="All", width=6, command=lambda: self._set_all(True)).pack(side="left")
        ttk.Button(bar, text="None", width=6, command=lambda: self._set_all(False)).pack(side="left", padx=4)
        ttk.Button(bar, text="Refresh status", command=self.refresh_status).pack(side="left", padx=12)
        ttk.Button(bar, text="Open work folder", command=self._open_work).pack(side="left")
        pack = ttk.Button(bar, text="Pack for cluster…", command=self._pack_for_cluster)
        pack.pack(side="right")
        ToolTip(pack, "Bundle everything stage 05 needs into one folder to copy to a cluster: "
                      "inputs, curves, manifest, model code, this package, job templates. "
                      "See docs/cluster.md.")
        collect = ttk.Button(bar, text="Collect from cluster…", command=self._collect_from_cluster)
        collect.pack(side="right", padx=6)
        ToolTip(collect, "Merge a bundle's work/models (copied back from the cluster) into the "
                         "local work tree; reports what is still missing.")

        # ----- options ------------------------------------------------
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
        ToolTip(pilot, "Few scans, one fold, one run, five epochs - minutes, to prove the wiring. "
                       "Pilot numbers are not results.")
        self.sweep = tk.BooleanVar(value=False)
        sweep = ttk.Checkbutton(line, text="PVC sweep", variable=self.sweep)
        sweep.pack(side="left", padx=8)
        ToolTip(sweep, "Stage 02 runs the whole iteration grid (triples the work).")

        # ----- run ----------------------------------------------------
        run_row = ttk.Frame(self)
        run_row.pack(fill="x", pady=8)
        self.run_button = ttk.Button(run_row, text="Run selected stages", command=self._run_selected)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(run_row, text="Stop", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(run_row, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=12)
        self.status = ttk.Label(run_row, text="idle")
        self.status.pack(side="left")

        self.log = LogPane(self, height=14)
        self.log.pack(fill="both", expand=True)

    # ================================================================ #
    # Config / status
    # ================================================================ #
    def _config(self):
        path = self.config_path.path()
        if not path or not path.exists():
            return None
        try:
            from pvc_dlif.config import load_config
            return load_config(path)
        except Exception as exc:                            # noqa: BLE001
            self.status.configure(text=f"config error: {exc}")
            return None

    def _set_all(self, value: bool) -> None:
        for var in self.selected.values():
            var.set(value)

    def _open_work(self) -> None:
        config = self._config()
        if config is None:
            return
        _open_in_file_manager(config.work)

    def refresh_status(self) -> None:
        """Re-read the work tree and update lights, counts and banners."""
        config = self._config()
        if config is None:
            for stage_id in self.lights:
                self._set_light(stage_id, "missing", "", "")
            return
        try:
            from pvc_dlif.status import manifest_summary, prepare_summary, stage_statuses
            statuses = stage_statuses(config)
            manifest = manifest_summary(config)
            prepare = prepare_summary(config)
        except Exception as exc:                            # noqa: BLE001
            self.status.configure(text=f"status error: {exc}")
            return

        running = self.state["stage"] if self.state else None
        if running and running not in self.lights:          # a helper tool, not a stage
            running = None
        for status in statuses:
            state = "running" if status.stage_id == running else status.state
            self._set_light(status.stage_id, state, status.progress_text, status.detail)

        # ----- the two banners ----------------------------------------
        if manifest:
            reasons = ", ".join(f"{v} {k}" for k, v in manifest["exclusion_reasons"].items()) or "none"
            self.banner_manifest.show(
                f"n usable = {manifest['n_usable']}   (of {manifest['n_total']} in the archive; "
                f"excluded: {reasons}; missing DICOM {manifest['n_missing_dicom']}, "
                f"missing input {manifest['n_missing_dlif_img']}, missing AIF {manifest['n_missing_aif']})",
                "ok" if manifest["n_usable"] > 0 else "error")
        else:
            self.banner_manifest.hide()

        if prepare:
            r = prepare.get("agreement_with_distributed_median_r")
            below = prepare.get("scans_below_0999", [])
            if r is None:
                self.banner_prepare.show("Stage 01: no scan could be compared against a distributed input.", "warn")
            else:
                text = (f"crop agreement with the distributed inputs: median r = {r:.5f}   "
                        f"({prepare.get('n_exact_matches')} of {prepare.get('n_matched_to_distributed')} exact)")
                if below:
                    text += "   -   below 0.999: " + ", ".join(f"{s} ({v:.3f})" for s, v in below)
                self.banner_prepare.show(text, "ok" if r >= 0.999 and not below else "warn")
        else:
            self.banner_prepare.hide()

    def _set_light(self, stage_id: str, state: str, progress: str, detail: str) -> None:
        self.lights[stage_id].configure(text=LIGHT[state], fg=LIGHT_COLOUR[state])
        self.progress_labels[stage_id].configure(text=progress)
        self.detail_labels[stage_id].configure(text=detail)

    # ================================================================ #
    # Running stages (detached)
    # ================================================================ #
    def _stage_args(self, stage_id: str) -> list[str]:
        """Command-line arguments for one stage; pilot mode reuses run_all's."""
        args = ["--config", str(self.config_path.get())]
        if self.pilot.get():
            args += _pilot_args(stage_id)
        elif self.limit.get():
            args += ["--limit", self.limit.get()]
        if self.ids.get():
            args += ["--ids", *self.ids.get().split()]
        if self.dry_run.get():
            args += ["--dry-run"]
        if stage_id == "02" and self.sweep.get():
            args += ["--sweep"]
        if stage_id == "07":
            args += ["--screen"]
        return args

    def _run_selected(self) -> None:
        chosen = [sid for sid, *_ in STAGES if self.selected[sid].get()]
        if not chosen:
            messagebox.showwarning("Nothing selected", "Tick at least one stage.")
            return
        self._run_stages(chosen)

    def _run_stages(self, stage_ids: list[str]) -> None:
        if self.state:
            messagebox.showinfo("Busy", f"Stage {self.state['stage']} is still running.")
            return
        config = self._config()
        if config is None:
            messagebox.showwarning("No config", "Pick a config file that loads.")
            return
        self.log.clear()
        self._queue = list(stage_ids)
        self._launch_next(config.work)

    def _launch_next(self, work: Path) -> None:
        if not self._queue:
            self._set_running(False)
            self.status.configure(text="done")
            self.log.write("\nAll selected stages finished.", "ok")
            self.refresh_status()
            self.app.analysis_tab.refresh_hint()
            return
        stage_id = self._queue.pop(0)
        command = [python_exe(), str(self.app.repo_root / "scripts" / SCRIPT_OF[stage_id]),
                   *self._stage_args(stage_id)]
        self.log.write(f"\n===== stage {stage_id} =====")
        self.log.write("$ " + " ".join(command))
        try:
            self.state = detached.launch(command, work, stage_id, cwd=self.app.repo_root,
                                         queue=self._queue)
        except Exception as exc:                            # noqa: BLE001
            self.log.write(f"ERROR: could not start stage {stage_id}: {exc}")
            self._queue.clear()
            self._set_running(False)
            return
        self._attach(work)

    def _attach(self, work: Path) -> None:
        """Start tailing the running stage's log."""
        assert self.state is not None
        self._work = work
        self._tail = detached.LogTail(Path(self.state["log_path"]))
        self._set_running(True)
        self.status.configure(text=f"stage {self.state['stage']} (pid {self.state['pid']})")
        self.refresh_status()
        self.after(400, self._poll)

    def _poll(self) -> None:
        """Every 400 ms: show new log lines; when the process is gone, move on."""
        if self.state is None or self._tail is None:
            return
        for line in self._tail.read_new():
            if not line.startswith(detached.MARKER):
                self.log.write(line)
        if detached.pid_alive(int(self.state["pid"])):
            self._ticks = getattr(self, "_ticks", 0) + 1
            if self._ticks % 8 == 0:              # ~every 3 s: refresh the heartbeat line
                self._show_heartbeat()
            self.after(400, self._poll)
            return

        # Process gone - drain the last lines, judge the outcome, continue.
        for line in self._tail.read_new():
            if not line.startswith(detached.MARKER):
                self.log.write(line)
        code = detached.exit_code_from_log(Path(self.state["log_path"]), int(self.state["pid"]))
        stage_id = self.state["stage"]
        detached.write_state(self._work, None)
        self.state = None
        self._tail = None
        if code == 0:
            self.log.write(f"stage {stage_id} finished", "ok")
            self._launch_next(self._work)
        else:
            why = "was stopped" if code is None else f"exited with {code}"
            self.log.write(f"stage {stage_id} {why} - stopping here (see the log above)", "error")
            self._queue.clear()
            self._set_running(False)
            self.status.configure(text="failed")
            self.refresh_status()

    def _show_heartbeat(self) -> None:
        """Put stage 05's live progress into its detail label and the status."""
        if self.state is None or self.state.get("stage") != "05":
            return
        config = self._config()
        if config is None:
            return
        try:
            from pvc_dlif.status import training_progress
            progress = training_progress(config)
        except Exception:                                   # noqa: BLE001
            return
        if not progress:
            self.detail_labels["05"].configure(text="starting (loading scans into memory)…")
            return
        age = progress["age_seconds"]
        eta = progress.get("eta_seconds", 0.0)
        eta_text = f"{eta / 60:.0f} min" if eta < 5400 else f"{eta / 3600:.1f} h"
        line = (f"{progress['condition']} {progress['fold']} {progress['run']}: "
                f"epoch {progress['epoch']}/{progress['epochs']}, "
                f"{progress['seconds_per_epoch']:.1f} s/epoch, run ETA {eta_text}, "
                f"val {progress['val_loss']:.4f} (best {progress['best_val_loss']:.4f})")
        if progress.get("finished"):
            line += "  - finished, next run starting"
        elif age > 10 * max(5.0, progress["seconds_per_epoch"]):
            line += f"  - NO UPDATE FOR {age / 60:.0f} MIN, may be stuck"
        else:
            line += f"  - updated {age:.0f} s ago"
        self.detail_labels["05"].configure(text=line)
        self.status.configure(text=f"stage 05 (pid {self.state['pid']}) epoch {progress['epoch']}/{progress['epochs']}")

    def _reattach(self) -> None:
        """On startup: is a stage from a previous GUI session still running?"""
        config = self._config()
        if config is None:
            return
        state = detached.read_state(config.work)
        if not state:
            return
        if detached.pid_alive(int(state.get("pid", 0))):
            self.state = state
            self._queue = list(state.get("queue", []))
            self.log.write(f"Re-attached to stage {state['stage']} started {state.get('started')} "
                           f"(pid {state['pid']}); {len(self._queue)} stage(s) queued after it.", "warn")
            self._attach(config.work)
        else:
            # It finished (or died) while no GUI was watching. Show its tail.
            log_path = Path(state.get("log_path", ""))
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                self.log.write(f"Stage {state['stage']} ran unattended; last lines of its log:", "warn")
                for line in lines[-15:]:
                    self.log.write("  " + line)
            detached.write_state(config.work, None)
            self.refresh_status()

    # ================================================================ #
    # Cluster helpers - both just run the scripts in scripts/cluster/
    # ================================================================ #
    def _pack_for_cluster(self) -> None:
        from tkinter import filedialog
        if self.state:
            messagebox.showinfo("Busy", "Wait for the running stage to finish first.")
            return
        config = self._config()
        if config is None:
            return
        target = filedialog.askdirectory(title="Folder to create the cluster bundle in",
                                         initialdir=str(config.work.parent))
        if not target:
            return
        self._run_tool("pack", ["scripts/cluster/pack_stage05.py", "--out", target])

    def _collect_from_cluster(self) -> None:
        from tkinter import filedialog
        if self.state:
            messagebox.showinfo("Busy", "Wait for the running stage to finish first.")
            return
        source = filedialog.askdirectory(title="The bundle's work/models folder copied back from the cluster")
        if not source:
            return
        self._run_tool("collect", ["scripts/cluster/collect_stage05.py", "--from", source])

    def _run_tool(self, label: str, script_and_args: list[str]) -> None:
        """Run a helper script the same way stages run, so its output is tailed."""
        config = self._config()
        if config is None:
            return
        script, *rest = script_and_args
        command = [python_exe(), str(self.app.repo_root / script), "--config",
                   str(self.config_path.get()), *rest]
        self.log.clear()
        self.log.write(f"===== {label} =====")
        self.log.write("$ " + " ".join(command))
        try:
            self.state = detached.launch(command, config.work, label, cwd=self.app.repo_root)
        except Exception as exc:                            # noqa: BLE001
            self.log.write(f"ERROR: could not start {label}: {exc}")
            return
        self._queue = []
        self._attach(config.work)

    def _cancel(self) -> None:
        if self.state is None:
            return
        if not messagebox.askyesno("Stop", f"Stop stage {self.state['stage']} now? Its outputs are "
                                           "resumable, so a later run picks up where it left off."):
            return
        self._queue.clear()
        detached.terminate(int(self.state["pid"]))
        self.status.configure(text="stopping…")

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        for button in self.run_buttons.values():
            button.configure(state="disabled" if running else "normal")
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()


def _open_in_file_manager(path: Path) -> None:
    import os
    import subprocess
    import sys
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(str(path))                             # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
