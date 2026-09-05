"""Analysis tab: read what stages 06 and 08 produced and look at it.

Four views:

* **Summary** - the headline from ``summary.json``: median RMSE per condition
  and every condition against the reference with CI, effect size and adjusted p.
* **Tables** - every result table, sortable, exportable.
* **Plots** - the thesis figures, drawn by the same functions stage 08 uses
  (``pvc_dlif.report.figures``), embedded in the window with a Save button.
* **Per scan** - one scan: its curves under every condition, its metric rows,
  its PVC diagnostics.  For finding and describing cases in the Discussion.

Plus an *Export for thesis* button that copies ``<work>/report`` to a folder
ready to drop into Overleaf.

Nothing is computed here.  If a table is missing the tab says which stage
produces it rather than drawing an empty axis.
"""

from __future__ import annotations

import gc
import json
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .widgets import LogPane, PathPicker, TableView, ToolTip

# table stem -> the stage that writes it, for the "missing" message
PRODUCED_BY = {
    "curve_metrics": "06", "comparisons": "06", "bias_variance": "06",
    "bias_variance_per_scan": "06", "error_by_time_bin": "06", "failure_modes": "06",
    "rmse_by_group": "06", "kinetics": "06", "pvc_frame_diagnostics": "08",
}
KEY_COLUMNS = {"condition", "scan_id", "run", "group", "fold", "metric", "tag", "frame", "model"}


class AnalysisTab(ttk.Frame):

    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.tables: dict[str, object] = {}       # stem -> DataFrame
        self.summary: dict = {}
        self.predictions = None
        self.diagnostics = None
        self._build()

    # ================================================================ #
    # Layout
    # ================================================================ #
    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x")
        self.results_dir = PathPicker(
            top, "Results folder", kind="dir",
            tooltip="Usually <work>/results. Predictions are read from the sibling "
                    "predictions/ folder and PVC diagnostics from pvc/.")
        self.results_dir.pack(fill="x", side="left", expand=True)
        ttk.Button(top, text="Load", width=8, command=self.load).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="From config", width=12, command=self._from_config).pack(side="left", padx=4)
        export = ttk.Button(top, text="Export for thesis…", command=self._export_for_thesis)
        export.pack(side="left", padx=(12, 0))
        ToolTip(export, "Copies <work>/report (figures as PDF/PNG and the LaTeX table fragments) "
                        "to a folder of your choice, ready for Overleaf.")

        self.status = ttk.Label(self, text="nothing loaded", foreground="#888")
        self.status.pack(anchor="w", pady=(4, 6))

        self.views = ttk.Notebook(self)
        self.views.pack(fill="both", expand=True)

        self.summary_pane = LogPane(self.views, height=20)
        self.views.add(self.summary_pane, text="Summary")

        self.tables_nb = ttk.Notebook(self.views)
        self.views.add(self.tables_nb, text="Tables")
        self.table_views: dict[str, TableView] = {}

        self.plot_pane = _PlotPane(self.views, self)
        self.views.add(self.plot_pane, text="Plots")

        self.scan_pane = _ScanBrowser(self.views, self)
        self.views.add(self.scan_pane, text="Per scan")

    # ================================================================ #
    # Loading
    # ================================================================ #
    def config(self):
        path = self.app.pipeline_tab.config_path.path()
        if not path or not path.exists():
            return None
        try:
            from pvc_dlif.config import load_config
            return load_config(path)
        except Exception:                                   # noqa: BLE001
            return None

    def refresh_hint(self) -> None:
        """Called after a pipeline run so the folder is prefilled."""
        if not self.results_dir.get():
            self._from_config(quiet=True)

    def _from_config(self, quiet: bool = False) -> None:
        config = self.config()
        if config is None:
            if not quiet:
                messagebox.showwarning("No config", "Set a config file that loads on the Pipeline tab.")
            return
        self.results_dir.var.set(str(config.dir_results))

    def load(self) -> None:
        folder = self.results_dir.path()
        if not folder or not folder.exists():
            messagebox.showwarning("Not found", "Pick an existing results folder.")
            return
        from pvc_dlif.report.assemble import frame_diagnostics_table, load_predictions, read_table

        self.tables.clear()
        found, missing = [], []
        for stem in PRODUCED_BY:
            frame = read_table(folder / stem)
            if frame is None:
                missing.append(stem)
            else:
                self.tables[stem] = frame
                found.append(stem)

        summary_path = folder / "summary.json"
        self.summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

        # Neighbours of results/: predictions/ and pvc/
        work = folder.parent
        self.predictions = load_predictions(work / "predictions")
        pvc_dir = work / "pvc"
        self.diagnostics = frame_diagnostics_table(pvc_dir) if pvc_dir.exists() else None
        if self.diagnostics is not None and self.diagnostics.empty:
            self.diagnostics = self.tables.get("pvc_frame_diagnostics")

        self.status.configure(
            text=f"loaded {len(found)} table(s)"
                 + (f"; predictions for {self.predictions['scan_id'].nunique()} scans" if self.predictions is not None else "; no predictions")
                 + (f"; missing: {', '.join(missing)}" if missing else ""),
            foreground="#333")

        self._fill_summary(missing)
        self._fill_tables()
        self.plot_pane.on_data_loaded()
        self.scan_pane.on_data_loaded()

    def _fill_summary(self, missing: list[str]) -> None:
        pane = self.summary_pane
        pane.clear()
        if not self.summary:
            pane.write("No summary.json in this folder - run stage 06.", "warn")
        else:
            pane.write(f"reference condition : {self.summary.get('reference')}")
            pane.write(f"scans               : {self.summary.get('n_scans')}")
            pane.write(f"conditions          : {', '.join(self.summary.get('conditions', []))}")
            pane.write("")
            pane.write("median RMSE by condition")
            for name, value in (self.summary.get("rmse_median") or {}).items():
                pane.write(f"  {name:24s} {value}")
            pane.write("")
            pane.write("versus the reference  (negative delta = lower error than the reference)")
            pane.write(f"  {'condition':24s} {'delta':>9s} {'95% CI':>20s} {'effect':>8s} {'p adj':>8s}")
            for row in self.summary.get("rmse_vs_reference", []):
                ci = row.get("ci") or [float("nan"), float("nan")]
                p = row.get("p_adjusted")
                better = p is not None and p < 0.05 and row.get("delta_median", 0) < 0
                worse = p is not None and p < 0.05 and row.get("delta_median", 0) > 0
                pane.write(
                    f"  {row.get('condition', ''):24s} {row.get('delta_median', float('nan')):9.4f}"
                    f" {f'[{ci[0]:.4f}, {ci[1]:.4f}]':>20s}"
                    f" {row.get('effect_size', float('nan')):8.3f}"
                    f" {('n/a' if p is None else f'{p:.4f}'):>8s}",
                    "ok" if better else ("error" if worse else None))
            bv = self.summary.get("bias_variance") or []
            if bv:
                pane.write("")
                pane.write("bias / variance   (MSE = bias^2 + variance; variance needs repeats)")
                for row in bv:
                    pane.write(f"  {str(row.get('condition', '')):24s} mse {row.get('mse', float('nan')):.5f}"
                               f"   bias^2 {row.get('bias_squared', float('nan')):.5f}"
                               f"   var {row.get('variance', float('nan')):.5f}")
        if missing:
            pane.write("")
            for stem in missing:
                pane.write(f"missing {stem} - produced by stage {PRODUCED_BY[stem]}", "warn")

    def _fill_tables(self) -> None:
        for tab_id in self.tables_nb.tabs():
            self.tables_nb.forget(tab_id)
        self.table_views.clear()
        for stem, frame in self.tables.items():
            view = TableView(self.tables_nb, height=18)
            view.show(frame)                                  # type: ignore[arg-type]
            self.tables_nb.add(view, text=stem)
            self.table_views[stem] = view

    # ================================================================ #
    # Export
    # ================================================================ #
    def _export_for_thesis(self) -> None:
        folder = self.results_dir.path()
        report_dir = folder.parent / "report" if folder else None
        if report_dir is None or not report_dir.exists():
            messagebox.showwarning("Nothing to export", "No report/ folder next to results/ - run stage 08.")
            return
        target = filedialog.askdirectory(title="Folder to copy figures and tables into")
        if not target:
            return
        copied = 0
        for path in sorted(report_dir.iterdir()):
            if path.suffix.lower() in {".pdf", ".png", ".tex", ".csv"}:
                shutil.copy2(path, Path(target) / path.name)
                copied += 1
        messagebox.showinfo("Exported", f"{copied} file(s) copied to\n{target}")


# ==================================================================== #
# Plots
# ==================================================================== #
class _FigureHost(ttk.Frame):
    """Holds one matplotlib figure with a toolbar and a Save button."""

    def __init__(self, parent):
        super().__init__(parent)
        self.figure = None
        self.holder = ttk.Frame(self)
        self.holder.pack(fill="both", expand=True)

    def show(self, figure) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import matplotlib.pyplot as plt

        for child in self.holder.winfo_children():
            child.destroy()
        # The old toolbar's PhotoImages must be collected *here*, on the Tk
        # thread.  Left to the garbage collector they can be finalised inside a
        # worker thread, which Tk does not tolerate and which can stall the
        # event loop.
        gc.collect()
        self.figure = figure
        canvas = FigureCanvasTkAgg(figure, master=self.holder)
        canvas.draw()
        NavigationToolbar2Tk(canvas, self.holder).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        # pyplot keeps its own reference; drop it so figures don't pile up.
        plt.close(figure)

    def save(self) -> None:
        if self.figure is None:
            messagebox.showinfo("Nothing to save", "Draw a plot first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=(("PDF", "*.pdf"), ("PNG", "*.png"), ("SVG", "*.svg")))
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches="tight")


class _PlotPane(ttk.Frame):
    """Picker + figure host. Every figure comes from pvc_dlif.report.figures."""

    PLOTS = [
        ("Paired metric", "One line per scan from the reference to each condition - what the "
                          "Wilcoxon test actually sees."),
        ("Bias / variance", "MSE split into bias^2 and variance per condition (stacked)."),
        ("Error by time bin", "Where along the curve each condition wins or loses."),
        ("Iteration sweep", "Median metric against deconvolution iteration count, per method. "
                            "A sensitivity analysis, not a search."),
        ("Recovery vs noise", "Per-frame peak recovery against noise amplification, coloured by "
                              "count level, for one PVC tag."),
        ("Curve overlay", "Every condition's predicted input function against the arterial curve "
                          "for one scan, with a bolus-peak inset."),
    ]

    def __init__(self, parent, tab: AnalysisTab):
        super().__init__(parent, padding=6)
        self.tab = tab

        controls = ttk.Frame(self)
        controls.pack(fill="x")
        ttk.Label(controls, text="Plot").pack(side="left", padx=(0, 4))
        self.plot_name = tk.StringVar(value=self.PLOTS[0][0])
        chooser = ttk.Combobox(controls, textvariable=self.plot_name, state="readonly", width=18,
                               values=[name for name, _ in self.PLOTS])
        chooser.pack(side="left")
        chooser.bind("<<ComboboxSelected>>", lambda _e: self._describe())

        ttk.Label(controls, text="Metric").pack(side="left", padx=(12, 4))
        self.metric = tk.StringVar(value="rmse")
        self.metric_box = ttk.Combobox(controls, textvariable=self.metric, width=14, state="readonly",
                                       values=("rmse",))
        self.metric_box.pack(side="left")

        ttk.Label(controls, text="Tag").pack(side="left", padx=(12, 4))
        self.tag = tk.StringVar()
        self.tag_box = ttk.Combobox(controls, textvariable=self.tag, width=10, state="readonly")
        self.tag_box.pack(side="left")
        ToolTip(self.tag_box, "PVC setting, for 'Recovery vs noise'.")

        ttk.Label(controls, text="Scan").pack(side="left", padx=(12, 4))
        self.scan = tk.StringVar()
        self.scan_box = ttk.Combobox(controls, textvariable=self.scan, width=9, state="readonly")
        self.scan_box.pack(side="left")
        ToolTip(self.scan_box, "For 'Curve overlay'.")

        ttk.Button(controls, text="Draw", command=self.draw).pack(side="left", padx=12)
        ttk.Button(controls, text="Save…", command=lambda: self.host.save()).pack(side="left")

        self.hint = ttk.Label(self, text="", foreground="#666", wraplength=800, justify="left")
        self.hint.pack(anchor="w", pady=(6, 4))
        self._describe()

        self.host = _FigureHost(self)
        self.host.pack(fill="both", expand=True)

    def _describe(self) -> None:
        for name, text in self.PLOTS:
            if name == self.plot_name.get():
                self.hint.configure(text=text)

    def on_data_loaded(self) -> None:
        metrics = self.tab.tables.get("curve_metrics")
        if metrics is not None:
            numeric = [c for c in metrics.columns                                  # type: ignore[attr-defined]
                       if c not in KEY_COLUMNS and metrics[c].dtype.kind in "fiu"]
            self.metric_box.configure(values=numeric or ["rmse"])
            if self.metric.get() not in numeric and numeric:
                self.metric.set("rmse" if "rmse" in numeric else numeric[0])
        if self.tab.diagnostics is not None and len(self.tab.diagnostics):
            tags = sorted(self.tab.diagnostics["tag"].unique())
            self.tag_box.configure(values=tags)
            if tags and self.tag.get() not in tags:
                self.tag.set(tags[0])
        if self.tab.predictions is not None:
            scans = sorted(str(s) for s in self.tab.predictions["scan_id"].unique())
            self.scan_box.configure(values=scans)
            if scans and self.scan.get() not in scans:
                self.scan.set(scans[0])

    def draw(self) -> None:
        if not self.tab.tables and self.tab.predictions is None:
            messagebox.showinfo("No data", "Load a results folder first.")
            return
        try:
            figure = self._build_figure()
        except Exception as exc:                            # noqa: BLE001
            messagebox.showerror("Plot failed", str(exc))
            return
        self.host.show(figure)

    def _build_figure(self):
        from pvc_dlif.report import assemble, figures

        choice = self.plot_name.get()
        metric = self.metric.get() or "rmse"
        metrics = self.tab.tables.get("curve_metrics")
        reference = self.tab.summary.get("reference")

        if choice == "Paired metric":
            if metrics is None:
                raise RuntimeError("curve_metrics is missing - run stage 06.")
            if reference not in set(metrics["condition"]):
                reference = sorted(metrics["condition"].unique())[0]
            return figures.plot_paired_metric(
                metrics, metric=metric, reference=reference,
                title=f"{metric.upper()} per scan, each condition against {reference}")

        if choice == "Bias / variance":
            table = self.tab.tables.get("bias_variance")
            if table is None:
                raise RuntimeError("bias_variance is missing - run stage 06.")
            return figures.plot_bias_variance(table, title="Squared error decomposed into bias and variance")

        if choice == "Error by time bin":
            table = self.tab.tables.get("error_by_time_bin")
            if table is None:
                raise RuntimeError("error_by_time_bin is missing - run stage 06.")
            column = metric if metric in table.columns else "rmse"
            return figures.plot_error_by_time_bin(table, metric=column,
                                                  title="Prediction error across the time-activity curve")

        if choice == "Iteration sweep":
            if metrics is None:
                raise RuntimeError("curve_metrics is missing - run stage 06.")
            config = self.tab.config()
            if config is None:
                raise RuntimeError("Needs the config (for which condition uses which iteration count).")
            sweep = assemble.iteration_sweep_table(metrics, config.conditions, metric=metric)
            if sweep["iterations"].nunique() < 2:
                raise RuntimeError("Only one iteration count in the results - run stage 02 with the "
                                   "sweep and evaluate the extra conditions first.")
            return figures.plot_iteration_sweep(sweep, metric=metric,
                                                title="Sensitivity to the deconvolution iteration count")

        if choice == "Recovery vs noise":
            diag = self.tab.diagnostics
            if diag is None or diag.empty:
                raise RuntimeError("No PVC diagnostics found - run stage 02 (or the PVC tab).")
            tag = self.tag.get()
            subset = diag[diag["tag"] == tag] if tag else diag
            return figures.plot_recovery_noise(subset, title=f"Recovery against noise, per frame ({tag})")

        # Curve overlay
        if self.tab.predictions is None:
            raise RuntimeError("No prediction tables found - run stage 04 (and 05).")
        scan_id = self.scan.get()
        curves, truth = assemble.curves_for_scan(self.tab.predictions, scan_id)
        if truth is None:
            raise RuntimeError(f"No predictions for {scan_id}.")
        return figures.plot_curve_overlay(curves, truth, title=f"Scan {scan_id}")


# ==================================================================== #
# Per-scan browser
# ==================================================================== #
class _ScanBrowser(ttk.Frame):
    """Everything known about one scan, side by side."""

    def __init__(self, parent, tab: AnalysisTab):
        super().__init__(parent, padding=6)
        self.tab = tab

        controls = ttk.Frame(self)
        controls.pack(fill="x")
        ttk.Label(controls, text="Scan").pack(side="left", padx=(0, 4))
        self.scan = tk.StringVar()
        self.scan_box = ttk.Combobox(controls, textvariable=self.scan, width=10, state="readonly")
        self.scan_box.pack(side="left")
        self.scan_box.bind("<<ComboboxSelected>>", lambda _e: self.show())
        ttk.Button(controls, text="◀", width=3, command=lambda: self._step(-1)).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="▶", width=3, command=lambda: self._step(1)).pack(side="left", padx=2)
        ttk.Label(controls, text="Sort by").pack(side="left", padx=(12, 4))
        self.order = tk.StringVar(value="scan id")
        order_box = ttk.Combobox(controls, textvariable=self.order, width=22, state="readonly",
                                 values=("scan id", "reference rmse (worst first)", "reference rmse (best first)"))
        order_box.pack(side="left")
        order_box.bind("<<ComboboxSelected>>", lambda _e: self._reorder(jump=True))
        ToolTip(order_box, "Worst-first is the quickest way to find the interesting cases.")
        ttk.Button(controls, text="Save figure…", command=lambda: self.host.save()).pack(side="right")

        self.info = ttk.Label(self, text="", foreground="#555")
        self.info.pack(anchor="w", pady=(4, 2))

        split = ttk.Panedwindow(self, orient="vertical")
        split.pack(fill="both", expand=True)
        self.host = _FigureHost(split)
        split.add(self.host, weight=3)
        lower = ttk.Notebook(split)
        split.add(lower, weight=2)
        self.metric_rows = TableView(lower, height=6, with_export=False)
        lower.add(self.metric_rows, text="metrics")
        self.diag_rows = TableView(lower, height=6, with_export=False)
        lower.add(self.diag_rows, text="PVC diagnostics")
        self.kinetic_rows = TableView(lower, height=6, with_export=False)
        lower.add(self.kinetic_rows, text="kinetics")

        self._scans: list[str] = []

    def on_data_loaded(self) -> None:
        self._reorder()

    def _reorder(self, jump: bool = False) -> None:
        metrics = self.tab.tables.get("curve_metrics")
        ids: list[str] = []
        if self.tab.predictions is not None:
            ids = sorted(str(s) for s in self.tab.predictions["scan_id"].unique())
        elif metrics is not None:
            ids = sorted(str(s) for s in metrics["scan_id"].unique())
        reference = self.tab.summary.get("reference")
        if metrics is not None and reference in set(metrics["condition"]) and self.order.get() != "scan id":
            ref = (metrics[metrics["condition"] == reference].groupby("scan_id")["rmse"].mean())
            ids = sorted(ids, key=lambda s: -ref.get(s, float("nan")) if "worst" in self.order.get()
                         else ref.get(s, float("nan")))
        self._scans = ids
        self.scan_box.configure(values=ids)
        if ids and (jump or self.scan.get() not in ids):
            self.scan.set(ids[0])
        if ids:
            self.show()

    def _step(self, delta: int) -> None:
        if not self._scans:
            return
        index = self._scans.index(self.scan.get()) if self.scan.get() in self._scans else 0
        self.scan.set(self._scans[(index + delta) % len(self._scans)])
        self.show()

    def show(self) -> None:
        scan_id = self.scan.get()
        if not scan_id:
            return
        from pvc_dlif.report import assemble, figures

        # Figure
        if self.tab.predictions is not None:
            curves, truth = assemble.curves_for_scan(self.tab.predictions, scan_id)
            if truth is not None:
                try:
                    self.host.show(figures.plot_curve_overlay(curves, truth, title=f"Scan {scan_id}"))
                except Exception as exc:                    # noqa: BLE001
                    self.info.configure(text=f"figure failed: {exc}")

        # Tables filtered to this scan
        metrics = self.tab.tables.get("curve_metrics")
        if metrics is not None:
            rows = metrics[metrics["scan_id"].astype(str) == scan_id]
            self.metric_rows.show(rows)
            line = f"{scan_id}: {len(rows)} metric rows"
            if "group" in rows.columns and len(rows):
                line += f", group {rows['group'].iloc[0]}"
            self.info.configure(text=line)
        if self.tab.diagnostics is not None:
            diag = self.tab.diagnostics
            self.diag_rows.show(diag[diag["scan_id"].astype(str) == scan_id])
        kinetics = self.tab.tables.get("kinetics")
        if kinetics is not None:
            self.kinetic_rows.show(kinetics[kinetics["scan_id"].astype(str) == scan_id])
