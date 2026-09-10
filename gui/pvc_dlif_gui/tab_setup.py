"""Setup tab: the config's paths and key parameters, and a preflight check.

Two jobs.  First, edit the handful of settings worth changing without opening
the YAML - the four paths, the primary iteration count and methods, the
cross-validation size, the training device, and the motion-affected IDs.
Saving round-trips through ``load_config`` so an invalid edit is refused
with the same message the command line would give.

Second, a preflight check: is PETPVC installed, is there a GPU, do the paths
exist, is there disk space.  Every failed item says what to do about it.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .jobs import ThreadJob
from .widgets import LabelledEntry, PathPicker, ToolTip


class SetupTab(ttk.Frame):

    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.job: ThreadJob | None = None
        self._build()
        # Populate from the config the Pipeline tab points at, if it loads.
        self.after(200, self.reload)

    # ---------------------------------------------------------------- #
    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Config file:").pack(side="left")
        self.config_label = ttk.Label(top, text="", foreground="#555")
        self.config_label.pack(side="left", padx=6)
        ttk.Button(top, text="Reload", command=self.reload).pack(side="right")

        # --- paths ------------------------------------------------------
        paths = ttk.LabelFrame(self, text="Paths", padding=8)
        paths.pack(fill="x", pady=(8, 0))
        self.paths = {
            "paths.dicom_root": PathPicker(paths, "DICOM root", "dir",
                                           tooltip="Folder of dPET_dcm_<ID> exports."),
            "paths.dlif_data_root": PathPicker(paths, "DLIF data", "dir",
                                               tooltip="Contains AIF_SUV/, IMG_SUV_*/, VOI_SUV/."),
            "paths.dlif_repo": PathPicker(paths, "DLIF repository", "dir",
                                          tooltip="Clone of the group's DLIF repo holding "
                                                  "src/models and the pretrained weights."),
            "paths.work": PathPicker(paths, "Work (output)", "dir",
                                     tooltip="Everything the pipeline generates. Can be deleted "
                                             "and rebuilt."),
            "paths.petpvc_exe": PathPicker(paths, "petpvc (optional)", "file",
                                           filetypes=(("Executable", "*"),),
                                           tooltip="Leave empty to look petpvc up on PATH."),
        }
        for picker in self.paths.values():
            picker.pack(fill="x", pady=2)

        # --- parameters -------------------------------------------------
        params = ttk.LabelFrame(self, text="Parameters exposed here (everything else stays in the YAML)",
                                padding=8)
        params.pack(fill="x", pady=(8, 0))
        row1 = ttk.Frame(params)
        row1.pack(fill="x", pady=2)
        self.iterations = LabelledEntry(row1, "Primary iterations", 15, width=6,
                                        tooltip="pvc.iterations_primary - fixed across all frames. "
                                                "Must be in pvc.iterations_grid.")
        self.iterations.pack(side="left")
        self.methods = LabelledEntry(row1, "Methods", "RL, RVC", width=12,
                                     tooltip="pvc.methods - comma-separated: RL, RVC.")
        self.methods.pack(side="left", padx=14)
        self.workers = LabelledEntry(row1, "PVC workers", 4, width=5,
                                     tooltip="pvc.workers - frames corrected in parallel.")
        self.workers.pack(side="left")

        row2 = ttk.Frame(params)
        row2.pack(fill="x", pady=2)
        self.folds = LabelledEntry(row2, "CV folds", 10, width=5,
                                   tooltip="dlif.cv.n_folds. 17 in the 2024 regime, 10 in 2026.")
        self.folds.pack(side="left")
        self.runs = LabelledEntry(row2, "Runs per fold", 10, width=5,
                                  tooltip="dlif.cv.n_runs - repeats that feed the variance term. "
                                          "1 in the 2024 regime, 10 in 2026.")
        self.runs.pack(side="left", padx=14)
        ttk.Label(row2, text="Device").pack(side="left")
        self.device = tk.StringVar(value="cuda")
        device_box = ttk.Combobox(row2, textvariable=self.device, width=6, state="readonly",
                                  values=("cuda", "cpu"))
        device_box.pack(side="left", padx=4)
        ToolTip(device_box, "dlif.train.device. Stage 05 on CPU is not realistic.")
        self.regime_label = ttk.Label(row2, text="", foreground="#555")
        self.regime_label.pack(side="left", padx=(18, 0))
        ToolTip(self.regime_label,
                "dlif.regime - which published training protocol stage 05 reproduces. "
                "It sets epochs, learning rate, loss, folds and runs together, and the "
                "two fields to the left edit the active one. Change it in the YAML.")

        row3 = ttk.Frame(params)
        row3.pack(fill="x", pady=2)
        self.motion_ids = LabelledEntry(row3, "Motion-affected IDs", "", width=48,
                                        tooltip="motion.affected_ids - space or comma separated. "
                                                "Confirm with the group before filling in; stage 07 "
                                                "--screen proposes candidates.")
        self.motion_ids.pack(side="left")

        buttons = ttk.Frame(params)
        buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(buttons, text="Save to config", command=self.save).pack(side="left")
        self.save_status = ttk.Label(buttons, text="", foreground="#555")
        self.save_status.pack(side="left", padx=10)

        # --- preflight --------------------------------------------------
        pre = ttk.LabelFrame(self, text="Preflight", padding=8)
        pre.pack(fill="both", expand=True, pady=(8, 0))
        bar = ttk.Frame(pre)
        bar.pack(fill="x")
        ttk.Button(bar, text="Run preflight check", command=self.preflight).pack(side="left")
        self.pre_status = ttk.Label(bar, text="", foreground="#555")
        self.pre_status.pack(side="left", padx=10)

        self.tree = ttk.Treeview(pre, columns=("status", "item", "detail"), show="headings", height=14)
        self.tree.heading("status", text="")
        self.tree.heading("item", text="Check")
        self.tree.heading("detail", text="Found / what to do")
        self.tree.column("status", width=40, anchor="center", stretch=False)
        self.tree.column("item", width=180, anchor="w", stretch=False)
        self.tree.column("detail", width=700, anchor="w")
        self.tree.tag_configure("ok", foreground="#2e7d32")
        self.tree.tag_configure("warn", foreground="#8a6d3b")
        self.tree.tag_configure("error", foreground="#c62828")
        self.tree.tag_configure("fix", foreground="#555")
        vsb = ttk.Scrollbar(pre, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True, pady=(6, 0))
        vsb.pack(side="right", fill="y", pady=(6, 0))

    # ---------------------------------------------------------------- #
    def config_path(self) -> Path | None:
        return self.app.pipeline_tab.config_path.path()

    def reload(self) -> None:
        """Fill the form from the YAML. Invalid config -> show the error, keep the form."""
        path = self.config_path()
        self.config_label.configure(text=str(path) if path else "(none)")
        if not path or not path.exists():
            return
        try:
            from pvc_dlif.config import load_config
            config = load_config(path)
        except Exception as exc:                            # noqa: BLE001
            self.save_status.configure(text=f"config does not load: {exc}", foreground="#c62828")
            return
        raw = config.raw
        for dotted, picker in self.paths.items():
            value = config.get(dotted)
            picker.var.set("" if value is None else str(value))
        self.iterations.var.set(str(config.get("pvc.iterations_primary", 15)))
        self.methods.var.set(", ".join(str(m) for m in config.get("pvc.methods", [])))
        self.workers.var.set(str(config.get("pvc.workers", 4)))
        self.folds.var.set(str(config.get("dlif.cv.n_folds", 10)))
        self.runs.var.set(str(config.get("dlif.cv.n_runs", 10)))
        self.device.set(str(config.get("dlif.train.device", "cuda")))
        if config.regime == "base":
            self.regime_label.configure(text="no regime set")
        else:
            self.regime_label.configure(
                text=f"regime {config.regime}:  "
                     f"{config.get('dlif.train.epochs')} epochs, "
                     f"lr {config.get('dlif.train.learning_rate')}, "
                     f"{config.get('dlif.train.loss')}"
            )
        self.motion_ids.var.set(" ".join(str(i) for i in config.get("motion.affected_ids", [])))
        self.save_status.configure(text="loaded", foreground="#555")
        del raw

    def save(self) -> None:
        """Write the form back into the YAML, keeping every comment."""
        path = self.config_path()
        if not path or not path.exists():
            messagebox.showwarning("No config", "Pick a config on the Pipeline tab first.")
            return
        from pvc_dlif.config_edit import save_values

        methods = [m.strip().upper() for m in self.methods.get().replace(";", ",").split(",") if m.strip()]
        motion = [i.strip() for i in self.motion_ids.get().replace(",", " ").split() if i.strip()]
        values = {
            "pvc.iterations_primary": self.iterations.as_int(15),
            "pvc.methods": methods,
            "pvc.workers": max(1, self.workers.as_int(1)),
            "dlif.cv.n_folds": max(1, self.folds.as_int(10)),
            "dlif.cv.n_runs": max(1, self.runs.as_int(10)),
            "dlif.train.device": self.device.get(),
            "motion.affected_ids": motion,
        }
        for dotted, picker in self.paths.items():
            text = picker.get()
            values[dotted] = text if text else None
        try:
            save_values(path, values)
        except Exception as exc:                            # noqa: BLE001
            messagebox.showerror("Not saved", str(exc))
            self.save_status.configure(text="not saved - invalid", foreground="#c62828")
            return
        self.save_status.configure(text="saved", foreground="#2e7d32")
        self.app.pipeline_tab.refresh_status()

    # ---------------------------------------------------------------- #
    def preflight(self) -> None:
        """Run the checks off the Tk thread (torch import alone can take seconds)."""
        self.tree.delete(*self.tree.get_children())
        self.pre_status.configure(text="checking…")
        path = self.config_path()
        results: list = []

        def work(log):
            from pvc_dlif.preflight import run_checks
            config, error = None, ""
            try:
                from pvc_dlif.config import load_config
                config = load_config(path) if path and path.exists() else None
                error = "" if config else "no config file"
            except Exception as exc:                        # noqa: BLE001
                error = str(exc)
            results.extend(run_checks(config, error))

        self.job = ThreadJob(lambda _line: None, lambda ok: self._show_checks(results, ok), self)
        self.job.start(work)

    def _show_checks(self, checks: list, ok: bool) -> None:
        errors = warns = 0
        for check in checks:
            if check.ok:
                tag, mark = "ok", "✓"
            elif check.severity == "warn":
                tag, mark = "warn", "!"
                warns += 1
            else:
                tag, mark = "error", "✗"
                errors += 1
            self.tree.insert("", "end", values=(mark, check.name, check.detail), tags=(tag,))
            if not check.ok and check.fix:
                self.tree.insert("", "end", values=("", "", "→ " + check.fix), tags=("fix",))
        if not ok:
            self.pre_status.configure(text="preflight itself failed - see console", foreground="#c62828")
        elif errors:
            self.pre_status.configure(text=f"{errors} blocking, {warns} warnings", foreground="#c62828")
        elif warns:
            self.pre_status.configure(text=f"ready with {warns} warning(s)", foreground="#8a6d3b")
        else:
            self.pre_status.configure(text="ready", foreground="#2e7d32")
