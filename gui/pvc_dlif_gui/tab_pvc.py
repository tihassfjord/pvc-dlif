"""PVC tab: deconvolve one image, a folder of them, or a set of thesis scans.

Two ways to choose the input.  *Pick files* takes any NIfTI (3D or 4D) - for
trying things outside the 70.  *Thesis dataset* lists the usable scan IDs
from ``manifest.json`` and reads their native series from ``<work>/native``,
so this tab can stand in for stage 02 on a subset.

Either way the work goes through :func:`pvc_dlif.pvc.runner.run_batch`, the
same function stage 02 calls.  That gives resumability (existing outputs are
skipped), the per-frame diagnostics, and one implementation to trust.  The
GUI only collects parameters and shows what came back.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .jobs import ThreadJob
from .widgets import CheckList, LabelledEntry, LogPane, PathPicker, TableView, ToolTip

# LabPET8 values measured in the project thesis; the config overrides them.
DEFAULT_FWHM = (0.864, 0.874, 0.994)


class PVCTab(ttk.Frame):

    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.job: ThreadJob | None = None
        self.stop_event = threading.Event()
        self._build()
        self.after(300, self._prefill_from_config)

    # ================================================================ #
    # Layout
    # ================================================================ #
    def _build(self) -> None:
        # ----- input ------------------------------------------------------
        io_box = ttk.LabelFrame(self, text="Input", padding=8)
        io_box.pack(fill="x")

        mode_row = ttk.Frame(io_box)
        mode_row.pack(fill="x")
        self.mode = tk.StringVar(value="files")
        for value, text in (("files", "Pick files"), ("thesis", "Thesis dataset")):
            ttk.Radiobutton(mode_row, text=text, value=value, variable=self.mode,
                            command=self._switch_mode).pack(side="left", padx=(0, 12))

        # Pick-files panel
        self.files_panel = ttk.Frame(io_box)
        self.inputs = PathPicker(
            self.files_panel, "Images", kind="files",
            tooltip="One or more NIfTI files. 3D or 4D; a 4D series is corrected frame by frame.")
        self.inputs.pack(fill="x", pady=2)
        row = ttk.Frame(self.files_panel)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Add whole folder…", command=self._add_folder).pack(side="left")
        ttk.Button(row, text="Clear", command=lambda: self.inputs.var.set("")).pack(side="left", padx=4)
        self.count_label = ttk.Label(row, text="0 files")
        self.count_label.pack(side="left", padx=10)
        self.inputs.var.trace_add("write", lambda *_: self._update_count())

        # Thesis-dataset panel
        self.thesis_panel = ttk.Frame(io_box)
        head = ttk.Frame(self.thesis_panel)
        head.pack(fill="x")
        self.thesis_info = ttk.Label(head, text="", foreground="#555")
        self.thesis_info.pack(side="left")
        ttk.Button(head, text="Reload from manifest", command=self._load_manifest).pack(side="right")
        self.scan_list = CheckList(self.thesis_panel, height=6, columns=8)
        self.scan_list.pack(fill="x", pady=(4, 0))

        self.outdir = PathPicker(
            io_box, "Output folder", kind="dir",
            tooltip="Outputs land in <folder>/<method>_i<k>/<scan>.nii.gz with a diagnostics "
                    "JSON beside each. For the thesis dataset this defaults to <work>/pvc, "
                    "which is exactly where stage 02 writes.")
        self.outdir.pack(fill="x", pady=(6, 2))

        # ----- correction -------------------------------------------------
        par_box = ttk.LabelFrame(self, text="Correction", padding=8)
        par_box.pack(fill="x", pady=(8, 0))

        line1 = ttk.Frame(par_box)
        line1.pack(fill="x", pady=2)
        ttk.Label(line1, text="Method").pack(side="left", padx=(0, 6))
        self.use_rl = tk.BooleanVar(value=True)
        self.use_rvc = tk.BooleanVar(value=False)
        rl = ttk.Checkbutton(line1, text="RL", variable=self.use_rl)
        rvc = ttk.Checkbutton(line1, text="RVC", variable=self.use_rvc)
        rl.pack(side="left")
        rvc.pack(side="left", padx=(4, 12))
        ToolTip(rl, "Richardson-Lucy: multiplicative, non-negative by construction.")
        ToolTip(rvc, "Reblurred Van Cittert: additive, faster, noisier. Uses alpha.")

        self.iterations = LabelledEntry(line1, "Iterations", 15,
                                        tooltip="Fixed count applied to every frame.")
        self.iterations.pack(side="left")
        self.sweep_on = tk.BooleanVar(value=False)
        sweep = ttk.Checkbutton(line1, text="Sweep", variable=self.sweep_on)
        sweep.pack(side="left", padx=(12, 2))
        self.sweep_values = LabelledEntry(line1, "", "10, 15, 20", width=12,
                                          tooltip="Iteration counts to run instead of the single value. "
                                                  "This is the sensitivity analysis, not a search.")
        self.sweep_values.pack(side="left")
        ToolTip(sweep, "Run every count in the list instead of the single value.")
        self.alpha = LabelledEntry(line1, "Alpha", 1.5, width=6,
                                   tooltip="RVC relaxation parameter. Ignored for RL.")
        self.alpha.pack(side="left", padx=(14, 0))

        line2 = ttk.Frame(par_box)
        line2.pack(fill="x", pady=4)
        ttk.Label(line2, text="PSF FWHM (mm)").pack(side="left", padx=(0, 6))
        self.fwhm = [LabelledEntry(line2, axis, value, width=7,
                                   tooltip=f"Point-spread FWHM along {axis} in mm. Prefilled "
                                           "from psf.fwhm_mm in the config.")
                     for axis, value in zip("xyz", DEFAULT_FWHM)]
        for entry in self.fwhm:
            entry.pack(side="left", padx=3)

        ttk.Label(line2, text="   Backend").pack(side="left", padx=(14, 4))
        self.backend = tk.StringVar(value="petpvc")
        backend_box = ttk.Combobox(line2, textvariable=self.backend, width=22, state="readonly",
                                   values=("petpvc", "numpy (development only)"))
        backend_box.pack(side="left")
        ToolTip(backend_box, "petpvc: the validated toolbox - anything you report comes from it.\n"
                             "numpy: built-in fallback so the GUI works without the binary.")

        self.workers = LabelledEntry(line2, "Workers", 4, width=5,
                                     tooltip="Frames corrected in parallel within a scan.")
        self.workers.pack(side="left", padx=12)
        self.resume = tk.BooleanVar(value=True)
        resume = ttk.Checkbutton(line2, text="Skip existing", variable=self.resume)
        resume.pack(side="left")
        ToolTip(resume, "Outputs that already exist are not recomputed.")

        # ----- run --------------------------------------------------------
        run_row = ttk.Frame(self)
        run_row.pack(fill="x", pady=8)
        self.run_button = ttk.Button(run_row, text="Run PVC", command=self._run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(run_row, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=6)
        ToolTip(self.stop_button, "Stops at the next scan boundary; the current scan finishes.")
        self.progress = ttk.Progressbar(run_row, mode="determinate", length=260)
        self.progress.pack(side="left", padx=12)
        self.status = ttk.Label(run_row, text="idle")
        self.status.pack(side="left")

        # ----- log + results ---------------------------------------------
        panes = ttk.Notebook(self)
        panes.pack(fill="both", expand=True)
        self.log = LogPane(panes, height=12)
        panes.add(self.log, text="Log")
        self.results = TableView(panes, height=12)
        panes.add(self.results, text="Results")
        self.frames_table = TableView(panes, height=12)
        panes.add(self.frames_table, text="Per frame")
        self.panes = panes

        self._switch_mode()

    # ================================================================ #
    # Input helpers
    # ================================================================ #
    def _switch_mode(self) -> None:
        if self.mode.get() == "files":
            self.thesis_panel.pack_forget()
            self.files_panel.pack(fill="x", before=self.outdir)
        else:
            self.files_panel.pack_forget()
            self.thesis_panel.pack(fill="x", before=self.outdir)
            if not self.scan_list.vars:
                self._load_manifest()

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Folder of NIfTI images")
        if not folder:
            return
        found = sorted(str(p) for p in Path(folder).glob("*.nii*"))
        if not found:
            messagebox.showinfo("Nothing found", "No .nii or .nii.gz files in that folder.")
            return
        existing = self.inputs.get()
        self.inputs.var.set(";".join(filter(None, [existing, ";".join(found)])))

    def _update_count(self) -> None:
        n = len(self.inputs.paths())
        self.count_label.configure(text=f"{n} file{'' if n == 1 else 's'}")

    def _config(self):
        """The loaded config, or None with the reason shown in the status bar."""
        path = self.app.pipeline_tab.config_path.path()
        if not path or not path.exists():
            return None
        try:
            from pvc_dlif.config import load_config
            return load_config(path)
        except Exception as exc:                            # noqa: BLE001
            self.thesis_info.configure(text=f"config error: {exc}")
            return None

    def _prefill_from_config(self) -> None:
        """PSF, iterations and alpha from the config, so defaults match the thesis."""
        config = self._config()
        if config is None:
            return
        for entry, value in zip(self.fwhm, config.fwhm_mm):
            entry.var.set(str(value))
        self.iterations.var.set(str(config.iterations_primary))
        self.sweep_values.var.set(", ".join(str(k) for k in config.iterations_grid))
        self.alpha.var.set(str(config.get("pvc.rvc_alpha", 1.5)))
        self.workers.var.set(str(config.get("pvc.workers", 4)))
        methods = {str(m).upper() for m in config.pvc_methods}
        self.use_rl.set("RL" in methods)
        self.use_rvc.set("RVC" in methods)

    def _load_manifest(self) -> None:
        """Fill the scan list with usable IDs that have a native series."""
        config = self._config()
        if config is None:
            self.thesis_info.configure(text="No config loaded - set one on the Pipeline tab.")
            return
        from pvc_dlif.status import manifest_summary
        summary = manifest_summary(config)
        if summary is None:
            self.thesis_info.configure(text="No manifest.json - run stage 00 first.")
            self.scan_list.set_items([])
            return
        ids = summary["usable_ids"]
        ready = [i for i in ids if (config.dir_native / f"{i}.nii.gz").exists()]
        self.scan_list.set_items(ready, checked=False)
        missing = len(ids) - len(ready)
        self.thesis_info.configure(
            text=f"{len(ready)} usable scans with a native series"
                 + (f"; {missing} still need stage 01" if missing else ""))
        if not self.outdir.get():
            self.outdir.var.set(str(config.dir_pvc))

    # ================================================================ #
    # Parameters -> settings
    # ================================================================ #
    def _iteration_list(self) -> list[int]:
        if not self.sweep_on.get():
            return [self.iterations.as_int(15)]
        counts = [int(float(p)) for p in self.sweep_values.get().replace(";", ",").split(",") if p.strip()]
        return counts or [self.iterations.as_int(15)]

    def _scan_pairs(self) -> list[tuple[str, Path]]:
        """``(scan_id, path)`` pairs for whichever input mode is active."""
        if self.mode.get() == "files":
            pairs = []
            for path in self.inputs.paths():
                scan_id = path.name.replace(".nii.gz", "").replace(".nii", "")
                pairs.append((scan_id, path))
            return pairs
        config = self._config()
        if config is None:
            return []
        return [(scan_id, config.dir_native / f"{scan_id}.nii.gz") for scan_id in self.scan_list.selected()]

    # ================================================================ #
    # Run / stop
    # ================================================================ #
    def _run(self) -> None:
        pairs = self._scan_pairs()
        out_dir = self.outdir.path()
        methods = [m for m, on in (("RL", self.use_rl.get()), ("RVC", self.use_rvc.get())) if on]
        if not pairs:
            messagebox.showwarning("No input", "Pick at least one image or scan.")
            return
        if not methods:
            messagebox.showwarning("No method", "Tick RL, RVC or both.")
            return
        if out_dir is None:
            messagebox.showwarning("No output folder", "Pick where the results should go.")
            return
        missing = [str(p) for _, p in pairs if not p.exists()]
        if missing:
            messagebox.showerror("Missing input", "Not found:\n" + "\n".join(missing[:8]))
            return

        counts = self._iteration_list()
        backend = "numpy" if self.backend.get().startswith("numpy") else "petpvc"
        config = self._config()
        executable = config.petpvc_exe if config is not None else None

        params = dict(
            methods=methods, counts=counts,
            alpha=self.alpha.as_float(1.5),
            fwhm=tuple(entry.as_float(d) for entry, d in zip(self.fwhm, DEFAULT_FWHM)),
            backend=backend, executable=executable,
            workers=max(1, self.workers.as_int(1)), resume=self.resume.get(),
        )

        self.stop_event.clear()
        self._n_scans = len(pairs)
        self._n_settings = len(methods) * len(counts)
        self.progress.configure(maximum=self._n_scans * self._n_settings, value=0)
        self.log.clear()
        self._set_running(True)
        self._processed_ids = [scan_id for scan_id, _ in pairs]
        self._out_dir = out_dir

        self.job = ThreadJob(self._on_line, self._on_done, self)
        self.job.start(lambda log: _run_batch_job(pairs, out_dir, params, log, self.stop_event),
                       capture_logger="pvc_dlif")

    def _stop(self) -> None:
        self.stop_event.set()
        self.status.configure(text="stopping after this scan…")

    def _on_line(self, line: str) -> None:
        if line.startswith("__progress__"):
            self.progress.step(1)
            done = int(self.progress["value"])
            scan = min(self._n_scans, done // max(1, self._n_settings) + 1)
            self.status.configure(text=f"scan {scan} of {self._n_scans}")
            return
        if "frames" in line and "/" in line:
            # "INFO   AA1 rl_i15: 20/42 frames" -> frame progress in the status label
            try:
                fragment = line.split(":")[-1].strip().split()[0]
                current = self.status.cget("text").split(",")[0]
                self.status.configure(text=f"{current}, frame {fragment}")
            except (IndexError, ValueError):
                pass
        self.log.write(line)

    def _on_done(self, ok: bool) -> None:
        self._set_running(False)
        stopped = self.stop_event.is_set()
        self.status.configure(text="stopped" if stopped else ("done" if ok else "failed"))
        self.log.write("Stopped at a scan boundary." if stopped else
                       ("Finished." if ok else "Finished with errors."),
                       "warn" if stopped else ("ok" if ok else "error"))
        self._show_results()

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        if running:
            self.status.configure(text="starting…")

    # ================================================================ #
    # Results
    # ================================================================ #
    def _show_results(self) -> None:
        """Summarise the diagnostics JSONs for the scans that were just processed."""
        try:
            import pandas as pd
            from pvc_dlif.report.assemble import frame_diagnostics_table
        except ImportError:
            return
        frames = frame_diagnostics_table(self._out_dir)
        if frames.empty:
            return
        frames = frames[frames["scan_id"].isin(self._processed_ids)]
        if frames.empty:
            return
        summary = (frames.groupby(["scan_id", "tag"])
                   .agg(frames=("frame", "count"),
                        noise_amplification=("noise_amplification", "median"),
                        peak_recovery=("peak_recovery", "median"),
                        negative_fraction=("negative_fraction_after", "mean"))
                   .reset_index())
        self.results.show(summary)
        self.frames_table.show(frames)
        self.panes.select(self.results)
        del pd


# ==================================================================== #
# The work itself - a plain function so it is testable without a window
# ==================================================================== #
def _run_batch_job(pairs, out_dir: Path, params: dict, log, stop_event) -> dict:
    """Build the settings list and hand everything to ``run_batch``.

    Returns the report ``run_batch`` produced.  Progress ticks are sent as the
    literal line ``__progress__`` which the tab turns into a bar step.
    """
    from pvc_dlif.pvc.deconvolution import PVCSettings
    from pvc_dlif.pvc.psf import PSF
    from pvc_dlif.pvc.runner import run_batch

    psf = PSF(tuple(params["fwhm"]))
    settings = [PVCSettings(method=m, iterations=k, psf=psf, alpha=params["alpha"])
                for m in params["methods"] for k in params["counts"]]
    log(f"{len(pairs)} scan(s) x {len(settings)} setting(s): "
        + ", ".join(s.tag for s in settings) + f"   PSF {psf.fwhm_mm} mm   backend {params['backend']}")

    report = {"written": [], "skipped": [], "failed": [], "warnings": [], "stopped": False}
    # One scan per call so progress can be reported between scans; run_batch
    # itself handles the settings loop, resume and diagnostics for that scan.
    for index, (scan_id, path) in enumerate(pairs, start=1):
        if stop_event.is_set():
            report["stopped"] = True
            break
        log(f"--- scan {index} of {len(pairs)}: {scan_id}")
        partial = run_batch([(scan_id, path)], settings, out_dir,
                            backend_name=params["backend"], executable=params["executable"],
                            workers=params["workers"], resume=params["resume"],
                            should_stop=stop_event.is_set)
        for key in ("written", "skipped", "failed", "warnings"):
            report[key].extend(partial[key])
        for _ in settings:
            log("__progress__")
        if partial["failed"]:
            for failure in partial["failed"]:
                log(f"ERROR {failure['scan_id']} {failure['settings']}: {failure['error']}")

    log(f"written {len(report['written'])}, skipped {len(report['skipped'])}, "
        f"failed {len(report['failed'])}")
    return report
