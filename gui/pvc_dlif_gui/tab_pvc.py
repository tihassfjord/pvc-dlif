"""PVC tab: deconvolve one image or a batch of them.

Accepts 3D volumes and 4D dynamic series (each frame corrected independently,
which is what the pipeline does).  Every parameter that changes the result is
on screen, and the exact settings are written next to each output as a small
JSON sidecar so a result can always be traced back to how it was made.
"""

from __future__ import annotations

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from .jobs import ThreadJob
from .widgets import LabelledEntry, LogPane, PathPicker, ToolTip

# Defaults are the LabPET8 values measured in the project thesis.
DEFAULT_FWHM = (0.864, 0.874, 0.994)


class PVCTab(ttk.Frame):

    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.job: ThreadJob | None = None
        self._build()

    # ---------------------------------------------------------------- #
    # Layout
    # ---------------------------------------------------------------- #
    def _build(self) -> None:
        # --- inputs ---------------------------------------------------
        io_box = ttk.LabelFrame(self, text="Input / output", padding=8)
        io_box.pack(fill="x")

        self.inputs = PathPicker(
            io_box, "Images", kind="files",
            tooltip="One or more NIfTI files (.nii/.nii.gz). 3D or 4D; "
                    "4D series are corrected frame by frame.")
        self.inputs.pack(fill="x", pady=2)

        row = ttk.Frame(io_box)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Add whole folder…", command=self._add_folder).pack(side="left")
        ttk.Button(row, text="Clear", command=lambda: self.inputs.var.set("")).pack(side="left", padx=4)
        self.count_label = ttk.Label(row, text="0 files")
        self.count_label.pack(side="left", padx=10)
        self.inputs.var.trace_add("write", lambda *_: self._update_count())

        self.outdir = PathPicker(
            io_box, "Output folder", kind="dir",
            tooltip="Corrected images land here as <name>_<method>_i<iterations>.nii.gz")
        self.outdir.pack(fill="x", pady=2)

        # --- parameters -----------------------------------------------
        par_box = ttk.LabelFrame(self, text="Correction", padding=8)
        par_box.pack(fill="x", pady=(8, 0))

        line1 = ttk.Frame(par_box)
        line1.pack(fill="x", pady=2)

        ttk.Label(line1, text="Method").pack(side="left", padx=(0, 4))
        self.method = tk.StringVar(value="RL")
        method_box = ttk.Combobox(line1, textvariable=self.method, width=6,
                                  state="readonly", values=("RL", "RVC"))
        method_box.pack(side="left")
        ToolTip(method_box,
                "RL: Richardson-Lucy, multiplicative, non-negative.\n"
                "RVC: Reblurred Van Cittert, additive, faster but noisier.")

        self.iterations = LabelledEntry(
            line1, "Iterations", 15, tooltip="Fixed iteration count, applied to every frame.")
        self.iterations.pack(side="left", padx=12)

        self.alpha = LabelledEntry(
            line1, "Alpha", 1.5, tooltip="RVC relaxation parameter. Ignored for RL.")
        self.alpha.pack(side="left", padx=(0, 12))

        self.no_stop = tk.BooleanVar(value=True)
        stop_check = ttk.Checkbutton(line1, text="Run exact iteration count",
                                     variable=self.no_stop)
        stop_check.pack(side="left")
        ToolTip(stop_check, "Disables PETPVC's internal stopping criterion (-s 0) so the "
                            "iteration count you set is the count that runs.")

        line2 = ttk.Frame(par_box)
        line2.pack(fill="x", pady=6)
        ttk.Label(line2, text="PSF FWHM (mm)").pack(side="left", padx=(0, 6))
        self.fwhm = [LabelledEntry(line2, axis, value, width=7,
                                   tooltip=f"Point-spread FWHM along {axis}, in millimetres.")
                     for axis, value in zip("xyz", DEFAULT_FWHM)]
        for entry in self.fwhm:
            entry.pack(side="left", padx=3)

        ttk.Label(line2, text="   Backend").pack(side="left", padx=(16, 4))
        self.backend = tk.StringVar(value="petpvc")
        backend_box = ttk.Combobox(line2, textvariable=self.backend, width=8,
                                   state="readonly", values=("petpvc", "numpy"))
        backend_box.pack(side="left")
        ToolTip(backend_box,
                "petpvc: the validated toolbox - use this for anything you will report.\n"
                "numpy: built-in fallback, for when the binary is not installed.")

        self.workers = LabelledEntry(line2, "Workers", 4, width=5,
                                     tooltip="Frames corrected in parallel, per scan.")
        self.workers.pack(side="left", padx=12)

        # --- sweep ----------------------------------------------------
        sweep_box = ttk.LabelFrame(self, text="Iteration sweep (optional)", padding=8)
        sweep_box.pack(fill="x", pady=(8, 0))
        self.sweep_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(sweep_box, text="Run every count below instead of the single value above",
                        variable=self.sweep_on).pack(side="left")
        self.sweep_values = LabelledEntry(sweep_box, "", "10, 15, 20", width=18,
                                          tooltip="Comma-separated iteration counts.")
        self.sweep_values.pack(side="left", padx=8)

        # --- run ------------------------------------------------------
        run_row = ttk.Frame(self)
        run_row.pack(fill="x", pady=8)
        self.run_button = ttk.Button(run_row, text="Run PVC", command=self._run)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(run_row, text="Cancel", command=self._cancel,
                                        state="disabled")
        self.cancel_button.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(run_row, mode="determinate", length=280)
        self.progress.pack(side="left", padx=12)
        self.status = ttk.Label(run_row, text="idle")
        self.status.pack(side="left")

        self.log = LogPane(self, height=14)
        self.log.pack(fill="both", expand=True)

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #
    def _add_folder(self) -> None:
        """Append every NIfTI in a chosen folder to the file list."""
        from tkinter import filedialog
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

    def _iteration_list(self) -> list[int]:
        if not self.sweep_on.get():
            return [self.iterations.as_int(15)]
        counts = []
        for part in self.sweep_values.get().replace(";", ",").split(","):
            part = part.strip()
            if part:
                counts.append(int(float(part)))
        return counts or [self.iterations.as_int(15)]

    # ---------------------------------------------------------------- #
    # Run
    # ---------------------------------------------------------------- #
    def _run(self) -> None:
        files = self.inputs.paths()
        out_dir = self.outdir.path()
        if not files:
            messagebox.showwarning("No input", "Pick at least one image.")
            return
        if out_dir is None:
            messagebox.showwarning("No output folder", "Pick where the results should go.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        counts = self._iteration_list()
        total = len(files) * len(counts)
        self.progress.configure(maximum=total, value=0)
        self.log.clear()
        self._set_running(True)

        settings = dict(
            method=self.method.get(),
            alpha=self.alpha.as_float(1.5),
            fwhm=tuple(entry.as_float(d) for entry, d in zip(self.fwhm, DEFAULT_FWHM)),
            disable_stopping=self.no_stop.get(),
            backend=self.backend.get(),
            workers=max(1, self.workers.as_int(1)),
        )

        self.job = ThreadJob(self._on_line, self._on_done, self)
        self.job.start(lambda log: _correct_all(files, out_dir, counts, settings, log,
                                                self.job.cancelled if self.job else False,
                                                should_stop=lambda: bool(self.job and self.job.cancelled)))

    def _cancel(self) -> None:
        if self.job:
            self.job.cancel()
            self.status.configure(text="cancelling…")

    def _on_line(self, line: str) -> None:
        if line.startswith("__progress__"):
            self.progress.step(1)
            return
        self.log.write(line)

    def _on_done(self, ok: bool) -> None:
        self._set_running(False)
        self.status.configure(text="done" if ok else "failed")
        self.log.write("Finished." if ok else "Finished with errors.", "ok" if ok else "error")

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        self.status.configure(text="running…" if running else "idle")


# -------------------------------------------------------------------- #
# The actual work - a plain function, so it is testable without any GUI.
# -------------------------------------------------------------------- #
def _correct_all(files, out_dir: Path, counts, settings: dict, log, _unused=False,
                 should_stop=lambda: False) -> None:
    """Correct every (file, iteration count) pair, writing a NIfTI and a sidecar."""
    import numpy as np
    import nibabel as nib

    from pvc_dlif.pvc.deconvolution import PVCSettings, make_backend
    from pvc_dlif.pvc.psf import PSF

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    psf = PSF(tuple(settings["fwhm"]))
    backend = make_backend(settings["backend"])
    log(f"Backend: {backend.name}   PSF FWHM: {psf.fwhm_mm} mm")

    for path in files:
        if should_stop():
            log("cancelled")
            return
        image = nib.load(str(path))
        data = np.asarray(image.dataobj, dtype=np.float32)
        zooms = [float(z) for z in image.header.get_zooms()[:3]]
        is_4d = data.ndim == 4

        sampling = psf.fwhm_voxels(zooms)
        log(f"\n{path.name}: shape {data.shape}, voxel {tuple(round(z, 4) for z in zooms)} mm, "
            f"FWHM {tuple(round(s, 2) for s in sampling)} voxels")
        for warning in psf.check_sampling(zooms, path.stem):
            log("WARNING: " + warning)

        for iterations in counts:
            if should_stop():
                log("cancelled")
                return
            pvc = PVCSettings(method=settings["method"], iterations=iterations, psf=psf,
                              alpha=settings["alpha"],
                              disable_stopping_criterion=settings["disable_stopping"])
            log(f"  {pvc.tag} …")

            if is_4d:
                # NIfTI is (x, y, z, t); the backend wants one volume at a time.
                corrected = np.empty_like(data)
                for t in range(data.shape[3]):
                    corrected[..., t] = backend.correct_volume(data[..., t], zooms, pvc)
            else:
                corrected = backend.correct_volume(data, zooms, pvc)

            stem = path.name.replace(".nii.gz", "").replace(".nii", "")
            target = out_dir / f"{stem}_{pvc.tag}.nii.gz"
            nib.save(nib.Nifti1Image(corrected.astype(np.float32), image.affine, image.header),
                     str(target))

            # Sidecar: what was run, on what, when.
            target.with_suffix("").with_suffix(".pvc.json").write_text(json.dumps({
                "source": str(path),
                "method": pvc.method,
                "iterations": pvc.iterations,
                "alpha": pvc.alpha,
                "psf_fwhm_mm": list(psf.fwhm_mm),
                "disable_stopping_criterion": pvc.disable_stopping_criterion,
                "backend": backend.name,
                "voxel_mm": zooms,
                "created": datetime.now().isoformat(timespec="seconds"),
            }, indent=2), encoding="utf-8")

            ratio = float(corrected.max() / data.max()) if data.max() > 0 else float("nan")
            log(f"    -> {target.name}   peak x{ratio:.3f}")
            log("__progress__")
