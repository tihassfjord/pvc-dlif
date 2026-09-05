"""Thesis figures.

Every figure is built from the result tables the evaluation stage writes, so
regenerating them after a rerun is one command and no figure can drift out of
step with the numbers in the text.

Style choices are deliberate and shared: a single colour per condition across
every figure, uncorrected always drawn first and in grey, and paired data drawn
as paired data (lines connecting the same scan) rather than as two independent
distributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = [
    "CONDITION_COLOURS", "save_figure", "plot_curve_overlay", "plot_paired_metric",
    "plot_bias_variance", "plot_error_by_time_bin", "plot_iteration_sweep",
    "plot_recovery_noise", "plot_motion_trace",
]

# One colour per condition, used everywhere.  Grey for the reference so the
# eye reads the corrected conditions as the change.
CONDITION_COLOURS: dict[str, str] = {
    "baseline_pretrained": "#6b7280",
    "baseline_retrained": "#9ca3af",
    "rl_pretrained": "#2563eb",
    "rl_retrained": "#1d4ed8",
    "rvc_pretrained": "#ea580c",
    "rvc_retrained": "#c2410c",
    "mc_baseline_retrained": "#059669",
    "mc_rl_retrained": "#047857",
}


def _colour(condition: str, index: int = 0) -> str:
    palette = ["#2563eb", "#ea580c", "#059669", "#7c3aed", "#dc2626", "#0891b2"]
    return CONDITION_COLOURS.get(condition, palette[index % len(palette)])


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )
    return plt


def save_figure(fig, path: Path, formats: Sequence[str] = ("pdf", "png")) -> list[Path]:
    """Write a figure in every requested format and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in formats:
        target = path.with_suffix(f".{suffix}")
        fig.savefig(target)
        written.append(target)
    fig.clf()
    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


def plot_curve_overlay(
    curves: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    truth: tuple[Sequence[float], Sequence[float]],
    title: str = "",
    inset_max_min: float = 3.0,
):
    """Predicted input functions against the arterial ground truth.

    An inset zooms on the first minutes: the bolus peak occupies a handful of
    5-8 s frames and is invisible on a 40-minute axis, yet it is where the
    spill-out bias and the compartment fits are most sensitive.
    """
    plt = _style()

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    t_truth, y_truth = np.asarray(truth[0], float), np.asarray(truth[1], float)
    ax.plot(t_truth, y_truth, "o-", color="black", lw=1.6, ms=3.2,
            label="Arterial sampling", zorder=5)

    for index, (name, (t, y)) in enumerate(curves.items()):
        ax.plot(np.asarray(t, float), np.asarray(y, float), "-", lw=1.5,
                color=_colour(name, index), label=name, alpha=0.9)

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Activity [SUV]")
    if title:
        ax.set_title(title)
    # Legend below the axes: the inset occupies the upper right, and a legend
    # that overlaps it is the fastest way to make a figure unreadable in print.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=min(len(curves) + 1, 4), fontsize=8)

    inset = ax.inset_axes([0.42, 0.38, 0.55, 0.55])
    inset.plot(t_truth, y_truth, "o-", color="black", lw=1.2, ms=2.4)
    for index, (name, (t, y)) in enumerate(curves.items()):
        inset.plot(np.asarray(t, float), np.asarray(y, float), "-", lw=1.2,
                   color=_colour(name, index), alpha=0.9)
    inset.set_xlim(0, inset_max_min)
    peak_window = y_truth[t_truth <= inset_max_min]
    if peak_window.size:
        inset.set_ylim(0, float(peak_window.max()) * 1.15)
    inset.tick_params(labelsize=7)
    inset.set_title("Bolus peak", fontsize=7.5)

    return fig


def plot_paired_metric(
    metrics_table: Any,
    metric: str,
    reference: str,
    conditions: Sequence[str] | None = None,
    aggregate_runs: str = "mean",
    title: str = "",
):
    """Paired dot plot: one line per scan, reference to condition.

    Drawing the pairing makes the thing the Wilcoxon test actually uses visible
    -- how consistently scans move, not just where the two medians sit.
    """
    plt = _style()

    collapsed = (
        metrics_table.groupby(["condition", "scan_id"], dropna=False)[metric]
        .agg(aggregate_runs)
        .reset_index()
    )
    wide = collapsed.pivot(index="scan_id", columns="condition", values=metric)
    targets = [c for c in (conditions or list(wide.columns)) if c != reference and c in wide.columns]

    fig, axes = plt.subplots(1, max(len(targets), 1), figsize=(2.6 * max(len(targets), 1), 3.8),
                             sharey=True, squeeze=False)
    for index, condition in enumerate(targets):
        ax = axes[0][index]
        pairs = wide[[reference, condition]].dropna()
        for _, row in pairs.iterrows():
            improved = row[condition] < row[reference]
            ax.plot([0, 1], [row[reference], row[condition]], "-",
                    color=("#2563eb" if improved else "#dc2626"), alpha=0.35, lw=0.9)
        ax.plot(np.zeros(len(pairs)), pairs[reference], "o", color="#6b7280", ms=3.5, alpha=0.8)
        ax.plot(np.ones(len(pairs)), pairs[condition], "o", color=_colour(condition, index), ms=3.5, alpha=0.8)
        ax.plot([0, 1], [pairs[reference].median(), pairs[condition].median()], "-",
                color="black", lw=2.2, zorder=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([reference, condition], rotation=20, ha="right", fontsize=7.5)
        ax.set_xlim(-0.3, 1.3)
        n_better = int((pairs[condition] < pairs[reference]).sum())
        ax.set_title(f"{n_better}/{len(pairs)} improved", fontsize=8)

    axes[0][0].set_ylabel(metric.upper())
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_bias_variance(decomposition: Any, title: str = ""):
    """Stacked bias-squared and variance per condition.

    The stack is the point: if the total barely moves while the two components
    trade against each other, an RMSE-only reading would call it a null result.
    """
    plt = _style()

    table = decomposition.sort_values("mse")
    fig, ax = plt.subplots(figsize=(1.5 * max(len(table), 3), 3.6))
    x = np.arange(len(table))

    ax.bar(x, table["bias_squared"], color="#2563eb", label="Bias$^2$")
    ax.bar(x, table["variance"], bottom=table["bias_squared"], color="#f59e0b", label="Variance")
    ax.plot(x, table["mse"], "k_", ms=18, mew=2, label="MSE")

    # A single deterministic model has no variance term to estimate; say so
    # above the bar rather than letting a zero read as a measured result.
    headroom = float(table["mse"].max()) * 0.03
    for index, row in enumerate(table.itertuples()):
        if not bool(getattr(row, "estimable", True)):
            ax.text(index, float(row.mse) + headroom, "no repeats",
                    ha="center", va="bottom", fontsize=6.5, color="#6b7280")
    ax.set_ylim(0, float(table["mse"].max()) * 1.14)

    ax.set_xticks(x)
    ax.set_xticklabels(table["condition"], rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean squared error [SUV$^2$]")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_error_by_time_bin(binned: Any, metric: str = "rmse", title: str = ""):
    """Error against frame timing, one line per condition.

    This is where the recovery-noise trade-off across the time-activity curve
    becomes visible: early frames have very low counts and late ones
    comparatively high, and PVC is not expected to behave the same in both.
    """
    plt = _style()

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    order = sorted(binned["time_bin"].unique(), key=lambda s: float(str(s).split("-")[0].replace(">=", "")))
    positions = {label: i for i, label in enumerate(order)}

    for index, (condition, group) in enumerate(binned.groupby("condition")):
        group = group.copy()
        group["_x"] = group["time_bin"].map(positions)
        group = group.sort_values("_x")
        ax.plot(group["_x"], group[metric], "o-", lw=1.5, ms=4,
                color=_colour(condition, index), label=condition)

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.set_xlabel("Frame time")
    ax.set_ylabel(metric.upper())
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_iteration_sweep(table: Any, metric: str = "rmse", title: str = ""):
    """Metric against deconvolution iteration count, per method.

    The iteration count is fixed for the main analysis; this figure is the
    sensitivity analysis that shows how much that choice matters, rather than
    tuning it to the best result.
    """
    plt = _style()

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for index, (method, group) in enumerate(table.groupby("method")):
        group = group.sort_values("iterations")
        ax.plot(group["iterations"], group[metric], "o-", lw=1.6, ms=5,
                color=_colour(str(method), index), label=str(method))
        if "ci_low" in group.columns and "ci_high" in group.columns:
            ax.fill_between(group["iterations"], group["ci_low"], group["ci_high"],
                            color=_colour(str(method), index), alpha=0.15)

    ax.set_xlabel("Deconvolution iterations")
    ax.set_ylabel(metric.upper())
    ax.legend(fontsize=8, title="Method", title_fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_recovery_noise(frame_diagnostics: Any, title: str = ""):
    """Noise amplification against peak recovery, coloured by frame count level.

    The phantom result -- recovery bought with noise -- restated on the dynamic
    data, with each point one frame of one scan.  Points in the lower right are
    where PVC pays off; the upper left is where it does not.
    """
    plt = _style()

    table = frame_diagnostics.dropna(subset=["noise_amplification", "peak_recovery"])
    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    counts = table["counts_proxy"].to_numpy(dtype=float)
    positive = counts[counts > 0]
    colour_values = np.log10(np.where(counts > 0, counts, positive.min() if positive.size else 1.0))

    scatter = ax.scatter(
        table["peak_recovery"], table["noise_amplification"],
        c=colour_values, cmap="viridis", s=12, alpha=0.7, edgecolors="none",
    )
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.axvline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Blood-pool peak recovery (after / before)")
    ax.set_ylabel("Noise amplification (%STD after / before)")
    fig.colorbar(scatter, ax=ax, label="log$_{10}$ frame count proxy")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_motion_trace(trace: Mapping[str, Any], threshold_mm: float = 0.5, title: str = ""):
    """Frame-to-frame displacement over the scan, for one animal."""
    plt = _style()

    displacement = np.asarray(trace["displacement_mm"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(np.arange(displacement.size), displacement, "o-", ms=3, lw=1.2, color="#2563eb")
    ax.axhline(threshold_mm, color="#dc2626", ls="--", lw=1.0, label=f"{threshold_mm} mm")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Centre-of-mass displacement [mm]")
    ax.legend(fontsize=8)
    ax.set_title(title or f"{trace.get('scan_id', '')}: cyclic score {trace.get('cyclic_score', float('nan')):.2f}")
    fig.tight_layout()
    return fig
