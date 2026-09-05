"""Analysis tab: read what stage 06 wrote and draw it.

Reads the result tables out of ``<work>/results`` and offers the five views the
thesis actually needs:

* **Summary**      - headline RMSE per condition and the test against the reference.
* **Metrics**      - the full per-scan, per-condition metric table.
* **Comparisons**  - Wilcoxon, effect size, bootstrap CI, Holm-adjusted p.
* **Bias/variance**- the MSE decomposition per condition.
* **Curves**       - predicted vs measured input function for one scan.

Nothing is computed here.  If a table is missing the tab says which stage
produces it rather than silently drawing an empty axis.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .widgets import LogPane, PathPicker, TableView, ToolTip

# table stem -> the stage that writes it, for the "missing" message
PRODUCED_BY = {
    "curve_metrics": "06",
    "comparisons": "06",
    "bias_variance": "06",
    "error_by_time_bin": "06",
    "failure_modes": "06",
    "rmse_by_group": "06",
    "kinetics": "06",
}


class AnalysisTab(ttk.Frame):

    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.tables: dict[str, object] = {}       # stem -> DataFrame
        self.summary: dict = {}
        self._build()

    # ---------------------------------------------------------------- #
    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x")
        self.results_dir = PathPicker(
            top, "Results folder", kind="dir",
            tooltip="Usually <work>/results. Contains curve_metrics.csv, comparisons.csv, "
                    "bias_variance.csv and summary.json.")
        self.results_dir.pack(fill="x", side="left", expand=True)
        ttk.Button(top, text="Load", width=8, command=self._load).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="From config", width=12, command=self._from_config).pack(side="left", padx=4)

        self.status = ttk.Label(self, text="nothing loaded", foreground="#888")
        self.status.pack(anchor="w", pady=(4, 6))

        self.views = ttk.Notebook(self)
        self.views.pack(fill="both", expand=True)

        # --- summary --------------------------------------------------
        self.summary_pane = LogPane(self.views, height=20)
        self.views.add(self.summary_pane, text="Summary")

        # --- three plain tables --------------------------------------
        self.metric_table = TableView(self.views, height=20)
        self.views.add(self.metric_table, text="Metrics")
        self.comparison_table = TableView(self.views, height=20)
        self.views.add(self.comparison_table, text="Comparisons")
        self.bv_table = TableView(self.views, height=20)
        self.views.add(self.bv_table, text="Bias / variance")

        # --- plots ----------------------------------------------------
        self.plot_pane = _PlotPane(self.views, self)
        self.views.add(self.plot_pane, text="Plots")

    # ---------------------------------------------------------------- #
    def refresh_hint(self) -> None:
        """Called after a pipeline run so the folder is prefilled."""
        if not self.results_dir.get():
            self._from_config(quiet=True)

    def _from_config(self, quiet: bool = False) -> None:
        """Fill the results folder from the config the Pipeline tab points at."""
        path = self.app.pipeline_tab.config_path.path()
        if not path or not path.exists():
            if not quiet:
                messagebox.showwarning("No config", "Set a config file on the Pipeline tab first.")
            return
        try:
            from pvc_dlif.config import load_config
            self.results_dir.var.set(str(load_config(path).dir_results))
        except Exception as exc:                          # noqa: BLE001
            if not quiet:
                messagebox.showerror("Config error", str(exc))

    # ---------------------------------------------------------------- #
    def _load(self) -> None:
        folder = self.results_dir.path()
        if not folder or not folder.exists():
            messagebox.showwarning("Not found", "Pick an existing results folder.")
            return

        import pandas as pd

        self.tables.clear()
        found, missing = [], []
        for stem in PRODUCED_BY:
            frame = _read_table(folder / stem)
            if frame is None:
                missing.append(stem)
            else:
                self.tables[stem] = frame
                found.append(f"{stem} ({len(frame)})")

        summary_path = folder / "summary.json"
        self.summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

        self.status.configure(
            text=f"loaded {len(found)} table(s)" + (f"; missing: {', '.join(missing)}" if missing else ""),
            foreground="#333")

        self._fill_summary(missing)
        for stem, view in (("curve_metrics", self.metric_table),
                           ("comparisons", self.comparison_table),
                           ("bias_variance", self.bv_table)):
            frame = self.tables.get(stem)
            if frame is not None:
                view.show(frame)                          # type: ignore[arg-type]
        self.plot_pane.on_data_loaded()

    def _fill_summary(self, missing: list[str]) -> None:
        """Human-readable headline: what beat the reference, and by how much."""
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
            pane.write("versus the reference  (negative delta = better)")
            header = f"  {'condition':24s} {'delta':>9s} {'95% CI':>20s} {'effect':>8s} {'p adj':>8s}"
            pane.write(header)
            for row in self.summary.get("rmse_vs_reference", []):
                ci = row.get("ci") or [float("nan"), float("nan")]
                p = row.get("p_adjusted")
                tag = ("ok" if p is not None and p < 0.05 and row.get("delta_median", 0) < 0
                       else None)
                pane.write(
                    f"  {row.get('condition', ''):24s} {row.get('delta_median', float('nan')):9.4f}"
                    f" {f'[{ci[0]:.4f}, {ci[1]:.4f}]':>20s}"
                    f" {row.get('effect_size', float('nan')):8.3f}"
                    f" {('n/a' if p is None else f'{p:.4f}'):>8s}", tag)
        if missing:
            pane.write("")
            for stem in missing:
                pane.write(f"missing {stem}.csv - produced by stage {PRODUCED_BY[stem]}", "warn")


# -------------------------------------------------------------------- #
class _PlotPane(ttk.Frame):
    """Matplotlib canvas plus a picker for which plot to draw."""

    PLOTS = [
        ("RMSE by condition", "Paired per-scan RMSE, one line per scan, so the paired "
                              "structure of the design is visible rather than averaged away."),
        ("Metric spread", "Distribution of a chosen metric across scans, by condition."),
        ("Bias / variance", "MSE split into bias^2 and variance per condition."),
        ("Error by time bin", "Where in the curve each condition wins or loses."),
        ("Predicted curve", "One scan: every condition's predicted input function against "
                            "the measured arterial curve."),
    ]

    def __init__(self, parent, tab: AnalysisTab):
        super().__init__(parent, padding=6)
        self.tab = tab
        self.canvas = None

        controls = ttk.Frame(self)
        controls.pack(fill="x")

        ttk.Label(controls, text="Plot").pack(side="left", padx=(0, 4))
        self.plot_name = tk.StringVar(value=self.PLOTS[0][0])
        chooser = ttk.Combobox(controls, textvariable=self.plot_name, state="readonly",
                               width=22, values=[name for name, _ in self.PLOTS])
        chooser.pack(side="left")
        chooser.bind("<<ComboboxSelected>>", lambda _e: self._describe())

        ttk.Label(controls, text="Metric").pack(side="left", padx=(14, 4))
        self.metric = tk.StringVar(value="rmse")
        self.metric_box = ttk.Combobox(controls, textvariable=self.metric, width=12,
                                       state="readonly", values=("rmse",))
        self.metric_box.pack(side="left")

        ttk.Label(controls, text="Scan").pack(side="left", padx=(14, 4))
        self.scan = tk.StringVar()
        self.scan_box = ttk.Combobox(controls, textvariable=self.scan, width=10, state="readonly")
        self.scan_box.pack(side="left")
        ToolTip(self.scan_box, "Only used by the 'Predicted curve' plot.")

        ttk.Button(controls, text="Draw", command=self.draw).pack(side="left", padx=12)
        ttk.Button(controls, text="Save…", command=self._save).pack(side="left")

        self.hint = ttk.Label(self, text="", foreground="#666", wraplength=760, justify="left")
        self.hint.pack(anchor="w", pady=(6, 4))
        self._describe()

        self.holder = ttk.Frame(self)
        self.holder.pack(fill="both", expand=True)

    # ---------------------------------------------------------------- #
    def _describe(self) -> None:
        for name, text in self.PLOTS:
            if name == self.plot_name.get():
                self.hint.configure(text=text)

    def on_data_loaded(self) -> None:
        """Repopulate the metric and scan pickers from the loaded tables."""
        metrics = self.tab.tables.get("curve_metrics")
        if metrics is None:
            return
        numeric = [c for c in metrics.columns                        # type: ignore[attr-defined]
                   if c not in {"condition", "scan_id", "run", "group", "fold"}]
        self.metric_box.configure(values=numeric or ["rmse"])
        if self.metric.get() not in numeric and numeric:
            self.metric.set("rmse" if "rmse" in numeric else numeric[0])
        scans = sorted(str(s) for s in metrics["scan_id"].unique())  # type: ignore[index]
        self.scan_box.configure(values=scans)
        if scans and not self.scan.get():
            self.scan.set(scans[0])

    # ---------------------------------------------------------------- #
    def draw(self) -> None:
        if not self.tab.tables:
            messagebox.showinfo("No data", "Load a results folder first.")
            return
        try:
            figure = self._build_figure()
        except Exception as exc:                          # noqa: BLE001
            messagebox.showerror("Plot failed", str(exc))
            return
        self._show(figure)

    def _build_figure(self):
        """Dispatch to the requested plot. Returns a matplotlib Figure."""
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import numpy as np

        metrics = self.tab.tables.get("curve_metrics")
        choice = self.plot_name.get()
        metric = self.metric.get() or "rmse"

        if choice == "RMSE by condition":
            if metrics is None:
                raise RuntimeError("curve_metrics is missing - run stage 06.")
            # Average over repeats, then one grey line per scan across conditions.
            wide = (metrics.groupby(["scan_id", "condition"])[metric].mean()
                    .unstack("condition"))
            figure, axis = plt.subplots(figsize=(7.5, 4.5))
            for _, row in wide.iterrows():
                axis.plot(wide.columns, row.values, color="0.7", linewidth=0.7, marker="o",
                          markersize=2, zorder=1)
            axis.plot(wide.columns, wide.median(), color="#c0392b", linewidth=2.2,
                      marker="s", label="median", zorder=3)
            axis.set_ylabel(metric)
            axis.set_title(f"Paired {metric} across conditions (n = {len(wide)})")
            axis.tick_params(axis="x", rotation=20)
            axis.legend()

        elif choice == "Metric spread":
            if metrics is None:
                raise RuntimeError("curve_metrics is missing - run stage 06.")
            grouped = metrics.groupby(["scan_id", "condition"])[metric].mean().unstack("condition")
            figure, axis = plt.subplots(figsize=(7.5, 4.5))
            axis.boxplot([grouped[c].dropna().values for c in grouped.columns],
                         labels=list(grouped.columns), showmeans=True)
            axis.set_ylabel(metric)
            axis.set_title(f"{metric} by condition")
            axis.tick_params(axis="x", rotation=20)

        elif choice == "Bias / variance":
            table = self.tab.tables.get("bias_variance")
            if table is None:
                raise RuntimeError("bias_variance is missing - run stage 06.")
            figure, axis = plt.subplots(figsize=(7.5, 4.5))
            x = np.arange(len(table))
            bias = table["bias_squared"] if "bias_squared" in table else table.iloc[:, 1]
            variance = table["variance"] if "variance" in table else table.iloc[:, 2]
            axis.bar(x, bias, label="bias²", color="#2980b9")
            axis.bar(x, variance, bottom=bias, label="variance", color="#e67e22")
            axis.set_xticks(x)
            axis.set_xticklabels(table["condition"] if "condition" in table else x, rotation=20)
            axis.set_ylabel("MSE")
            axis.set_title("MSE decomposition")
            axis.legend()

        elif choice == "Error by time bin":
            table = self.tab.tables.get("error_by_time_bin")
            if table is None:
                raise RuntimeError("error_by_time_bin is missing - run stage 06.")
            column = metric if metric in table.columns else "rmse"
            figure, axis = plt.subplots(figsize=(7.5, 4.5))
            for condition, group in table.groupby("condition"):
                group = group.sort_values(group.columns[1])
                axis.plot(group.iloc[:, 1].astype(str), group[column], marker="o", label=condition)
            axis.set_ylabel(column)
            axis.set_xlabel("time bin")
            axis.set_title(f"{column} by frame time bin")
            axis.legend(fontsize=8)

        else:                                             # Predicted curve
            figure = self._curve_figure(plt)

        figure.tight_layout()
        return figure

    def _curve_figure(self, plt):
        """Predicted vs measured input function for the selected scan."""
        import pandas as pd

        folder = self.tab.results_dir.path()
        scan_id = self.scan.get()
        if not scan_id:
            raise RuntimeError("Pick a scan.")
        # Predictions live one level up, next to results/, in predictions/.
        predictions_dir = folder.parent / "predictions" if folder else None
        frames = []
        for name in ("pretrained", "retrained"):
            for suffix in (".parquet", ".csv"):
                path = (predictions_dir / name).with_suffix(suffix) if predictions_dir else None
                if path and path.exists():
                    frames.append(pd.read_parquet(path) if suffix == ".parquet"
                                  else pd.read_csv(path))
                    break
        if not frames:
            raise RuntimeError("No prediction tables found - run stage 04 (and 05).")

        table = pd.concat(frames, ignore_index=True)
        table = table[table["scan_id"].astype(str) == scan_id]
        if table.empty:
            raise RuntimeError(f"No predictions for {scan_id}.")

        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        collapsed = (table.groupby(["condition", "frame"])
                     .agg(predicted=("predicted", "mean"),
                          truth=("truth", "first"),
                          time_min=("time_min", "first")).reset_index())
        for condition, group in collapsed.groupby("condition"):
            group = group.sort_values("frame")
            axis.plot(group["time_min"], group["predicted"], linewidth=1.4, label=condition)
        reference = collapsed.drop_duplicates("frame").sort_values("frame")
        axis.plot(reference["time_min"], reference["truth"], "k--", linewidth=2,
                  label="arterial (measured)")
        axis.set_xlabel("time (min)")
        axis.set_ylabel("activity (SUV)")
        axis.set_title(f"Input function - {scan_id}")
        axis.legend(fontsize=8)
        return figure

    # ---------------------------------------------------------------- #
    def _show(self, figure) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        for child in self.holder.winfo_children():
            child.destroy()
        self.figure = figure
        canvas = FigureCanvasTkAgg(figure, master=self.holder)
        canvas.draw()
        NavigationToolbar2Tk(canvas, self.holder).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas = canvas

    def _save(self) -> None:
        """Write the current figure at thesis resolution."""
        from tkinter import filedialog
        if getattr(self, "figure", None) is None:
            messagebox.showinfo("Nothing to save", "Draw a plot first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=(("PDF", "*.pdf"), ("PNG", "*.png"), ("SVG", "*.svg")))
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", path)


def _read_table(stem: Path):
    """Read <stem>.parquet if present, else <stem>.csv, else None."""
    import pandas as pd
    parquet = stem.with_suffix(".parquet")
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:                                 # noqa: BLE001 - pyarrow optional
            pass
    csv = stem.with_suffix(".csv")
    return pd.read_csv(csv) if csv.exists() else None
