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
    "plot_psf_sweep", "plot_recovery_noise", "plot_motion_trace",
    "plot_effects_forest", "plot_ratio_agreement", "plot_condition_ladder",
]

# Diverging pair for "better" and "worse", with a neutral for "cannot tell".
# Blue and red read as opposites and separate under every simulated colour
# vision deficiency; the neutral midpoint is grey rather than a third hue, so
# an interval crossing zero reads as absence rather than as a third category.
BETTER, WORSE, UNDECIDED = "#2a78d6", "#e34948", "#8c8c8c"

# One colour per condition, used everywhere.  Grey for the reference so the
# eye reads the corrected conditions as the change.
CONDITION_COLOURS: dict[str, str] = {
    # The two uncorrected references are grey on purpose: the eye then reads
    # the corrected conditions as the change, which is what they are.
    "baseline_pretrained": "#8c8c8c",
    "baseline_retrained": "#8c8c8c",
}

# Categorical hues in fixed order.  Validated as a set: every adjacent pair
# separates under simulated protan, deutan and tritan vision as well as for
# normal vision, against a near-white surface.  Assigned in order and never
# cycled -- past six series a figure is showing too much and should be split,
# which is why the figures that use this cap what they draw rather than
# generating a seventh hue.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948", "#eda100"]


def _colour(condition: str, index: int = 0) -> str:
    return CONDITION_COLOURS.get(condition, CATEGORICAL[index % len(CATEGORICAL)])


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
    conditions: Sequence[str] | None = None,
    max_series: int = 4,
    labels: Mapping[str, str] | None = None,
):
    """Predicted input functions against the arterial ground truth.

    An inset zooms on the first minutes: the bolus peak occupies a handful of
    5-8 s frames and is invisible on a 40-minute axis, yet it is where the
    spill-out bias and the compartment fits are most sensitive.

    Only a handful of conditions are drawn.  Every condition in this study
    predicts roughly the same curve -- the differences that matter are a few
    per cent at the peak -- so overlaying all of them produces a band of
    indistinguishable lines in which nothing can be read.  ``conditions``
    names the ones to draw; otherwise the first ``max_series`` are taken.
    Identity comes from the legend, where each name sits beside its swatch,
    rather than from colour alone; the curves converge in the tail, so a direct
    label there would stack four names on one point.
    """
    plt = _style()

    names = [c for c in (conditions or list(curves)) if c in curves][:max_series]
    if not names:
        names = list(curves)[:max_series]
    # The legend is the only place a curve is identified, so it carries the
    # reader's name for the condition where one is given rather than the
    # internal key.
    shown = dict(labels or {})

    # Two panels.  The predicted curves agree with the measured one and with
    # each other to within a few per cent over most of the acquisition, so
    # drawn as absolute activity they are one thick line and nothing can be
    # read from them.  The lower panel divides each prediction by the measured
    # curve, which is where the disagreement actually lives: on that axis the
    # curves separate, and the distance from 1.0 is the error itself.
    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(6.8, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )
    t_truth, y_truth = np.asarray(truth[0], float), np.asarray(truth[1], float)
    ax.plot(t_truth, y_truth, "o-", color="#111111", lw=2.0, ms=3.2,
            label="Arterial sampling", zorder=6)

    # The ratio is undefined where the measured curve is near zero, so the
    # pre-injection frames are left out of the lower panel rather than drawn as
    # a spike that is an artefact of dividing by nothing.
    # One per cent of the peak: below the pre-injection frames, well below the
    # tail. A higher threshold silently truncates the lower panel before the
    # late frames, which is where the conditions differ most in relative terms.
    usable = y_truth > 0.01 * float(y_truth.max())

    for index, name in enumerate(names):
        t, y = curves[name]
        t, y = np.asarray(t, float), np.asarray(y, float)
        ax.plot(t, y, "-", lw=1.8, color=_colour(name, index),
                label=shown.get(name, name), zorder=4)
        n = min(t.size, usable.size)
        mask = usable[:n]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = y[:n][mask] / y_truth[:n][mask]
        ax_res.plot(t[:n][mask], ratio, "-", lw=1.8,
                    color=_colour(name, index), zorder=4)

    ax_res.axhline(1.0, color="#111111", lw=1.4, zorder=5)
    ax_res.set_xlabel("Time [min]")
    ax_res.set_ylabel("predicted / measured")
    ax_res.set_ylim(0.4, 1.6)
    ax_res.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])

    ax.set_ylabel("Activity [SUV]")
    ax.set_xlim(left=0)
    # Logarithmic, because the bolus peak is an order of magnitude above the
    # tail and on a linear axis it takes the whole height, leaving the forty
    # minutes that carry most of the area squashed against the bottom where no
    # difference between conditions can be seen.  The peak keeps its own linear
    # inset, so nothing is lost by scaling the main panel for the tail.
    # The floor comes from the measured curve, not from the predictions: a
    # single near-zero pre-injection frame in one prediction would otherwise
    # drag the axis down a decade and leave most of the panel empty.
    ax.set_yscale("log")
    # and it comes from the tail rather than the whole curve, because the
    # pre-injection frames sit near zero and would open a decade of empty axis
    # below the part of the curve anyone reads.
    after_peak = y_truth[np.arange(y_truth.size) > int(np.argmax(y_truth))]
    tail = after_peak[after_peak > 0]
    if tail.size:
        ax.set_ylim(bottom=float(tail.min()) * 0.7, top=float(y_truth.max()) * 1.6)
    if title:
        ax.set_title(title, fontsize=10)
    # Below the lower panel, so it clears both.
    ax_res.legend(handles=ax.get_legend_handles_labels()[0],
                  labels=ax.get_legend_handles_labels()[1],
                  loc="upper center", bbox_to_anchor=(0.5, -0.30),
                  ncol=min(len(names) + 1, 3), fontsize=8, frameon=False)

    # Opaque, and above the main artists: a transparent inset lets the curve
    # underneath show through its panel, which reads as data.
    inset = ax.inset_axes([0.46, 0.42, 0.50, 0.52], zorder=10)
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    for side in inset.spines.values():
        side.set_edgecolor("#c9c9c6")
    inset.plot(t_truth, y_truth, "o-", color="#111111", lw=1.6, ms=2.4, zorder=6)
    for index, name in enumerate(names):
        t, y = curves[name]
        inset.plot(np.asarray(t, float), np.asarray(y, float), "-", lw=1.5,
                   color=_colour(name, index), zorder=4)
    inset.set_xlim(0, inset_max_min)
    peak_window = y_truth[t_truth <= inset_max_min]
    if peak_window.size:
        inset.set_ylim(0, float(peak_window.max()) * 1.25)
    inset.tick_params(labelsize=7)
    inset.set_title("Bolus peak (linear)", fontsize=7.5)

    # Set explicitly rather than with tight_layout, which cannot account for
    # the inset and warns about it.
    fig.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.22)
    return fig


def plot_paired_metric(
    metrics_table: Any,
    metric: str,
    reference: str,
    conditions: Sequence[str] | None = None,
    aggregate_runs: str = "mean",
    title: str = "",
    ylim: tuple[float, float] | None = None,
    target: float | None = None,
):
    """Paired dot plot: one line per scan, reference to condition.

    Drawing the pairing makes the thing the Wilcoxon test actually uses visible
    -- how consistently scans move, not just where the two medians sit.

    ``ylim`` fixes the vertical scale.  Panels within one figure already share
    it; passing the same range to every figure makes them comparable across
    arms too, which matters because otherwise a small effect drawn on a narrow
    axis looks larger than a big one drawn on a wide axis.

    ``target`` is the value that would be perfect.  For an error metric that is
    zero and is usually off the axis, so it is left unset and lower is simply
    better.  For a ratio it is 1.0, and setting it changes what the figure
    claims: a line is drawn at 1.0 and each scan is coloured by whether it
    moved *towards* that value, not by whether it went down.  Without it a
    condition that overshoots from 0.98 to 1.02 would be drawn as a worsening
    of the same size as one that fell from 0.98 to 0.94, when in fact the
    first got closer to correct.
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
        if target is None:
            closer = pairs[condition] < pairs[reference]
        else:
            closer = (pairs[condition] - target).abs() < (pairs[reference] - target).abs()
        for scan, row in pairs.iterrows():
            ax.plot([0, 1], [row[reference], row[condition]], "-",
                    color=(BETTER if closer[scan] else WORSE), alpha=0.35, lw=0.9)
        ax.plot(np.zeros(len(pairs)), pairs[reference], "o", color="#6b7280", ms=3.5, alpha=0.8)
        ax.plot(np.ones(len(pairs)), pairs[condition], "o", color=_colour(condition, index), ms=3.5, alpha=0.8)
        ax.plot([0, 1], [pairs[reference].median(), pairs[condition].median()], "-",
                color="black", lw=2.2, zorder=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([reference, condition], rotation=20, ha="right", fontsize=7.5)
        ax.set_xlim(-0.3, 1.3)
        if target is not None:
            ax.axhline(target, color="#4a4a4a", lw=0.9, ls="--", zorder=1)
        ax.set_title(f"{int(closer.sum())}/{len(pairs)} improved", fontsize=8)

    axes[0][0].set_ylabel(metric.upper()
                          + (f"   (perfect = {target:g})" if target is not None else ""))
    if ylim is not None:
        axes[0][0].set_ylim(*ylim)
    if target is not None:
        axes[0][-1].annotate("perfect", xy=(1.0, target), xycoords=("axes fraction", "data"),
                             xytext=(4, 0), textcoords="offset points",
                             va="center", fontsize=7, color="#52514e")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_effects_forest(comparisons: Any, metric: str = "rmse", title: str = "",
                        reference_label: bool = True, lower_is_better: bool = True):
    """Every paired comparison on one axis: effect with confidence interval.

    This is the figure the study reduces to.  A paired dot plot per condition
    shows one comparison well but makes two comparisons hard to weigh against
    each other, because each panel carries its own axis.  Here every condition
    is a row on a single shared scale, so the size of one effect relative to
    another is the length of a line rather than a number to be looked up.

    ``comparisons`` is the table stage 06 writes, filtered to one metric.  Rows
    are grouped by the ``family`` column (the analysis arms), because a
    comparison is only meaningful against its own reference.

    Colour carries direction and certainty together: blue where the interval
    lies wholly on the improving side, red where it lies wholly on the
    worsening side, grey where it spans zero and the sign is undetermined.
    Grey is not a third outcome, it is the absence of one.

    ``lower_is_better`` must be turned off for a ratio metric, where the target
    is 1.0 rather than zero.  A shift of -0.04 in a ratio is an improvement
    only if the reference was above 1.0, and the difference alone does not say
    whether it was; the figure then reports significance and magnitude without
    asserting a direction it cannot justify.
    """
    plt = _style()
    import matplotlib.patheffects as pe

    table = comparisons
    if "metric" in table.columns:
        table = table[table["metric"] == metric]
    if not len(table):
        raise ValueError(f"no comparisons for metric {metric!r}")

    has_family = "family" in table.columns
    groups = (list(table.groupby("family", sort=False)) if has_family
              else [("", table)])

    rows: list[tuple] = []
    for family, group in groups:
        group = group.sort_values("median_difference")
        rows.append(("header", family, group["reference"].iloc[0]
                     if "reference" in group.columns else ""))
        for _, r in group.iterrows():
            rows.append(("row", r, None))

    height = 0.34 * len(rows) + 1.2
    fig, ax = plt.subplots(figsize=(7.4, height))

    ax.axvline(0.0, color="#4a4a4a", lw=1.0, zorder=2)

    ticks, labels, header_rows = [], [], []
    for index, (kind, payload, extra) in enumerate(rows):
        y = len(rows) - index
        if kind == "header":
            ticks.append(y)
            header_rows.append(len(labels))
            labels.append(str(payload).replace("_", " "))
            if reference_label and extra:
                ax.annotate(f"vs {extra}", xy=(0.0, y), xycoords=("axes fraction", "data"),
                            xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=7, color="#6b7280")
            continue

        r = payload
        low, high = float(r["ci_low"]), float(r["ci_high"])
        delta = float(r["median_difference"])
        spans_zero = low <= 0.0 <= high
        if spans_zero:
            colour = UNDECIDED
        elif lower_is_better:
            colour = BETTER if delta < 0 else WORSE
        else:
            # A ratio: the shift is real but which side of 1.0 it lands on is
            # not encoded here, so the mark states significance, not benefit.
            colour = "#4a3aa7"

        ax.plot([low, high], [y, y], "-", color=colour, lw=2.0, solid_capstyle="butt", zorder=3)
        ax.plot([delta], [y], "o", color=colour, ms=7, zorder=4,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        ticks.append(y)
        labels.append(str(r["condition"]))

        bits = []
        if "effect_size" in r and np.isfinite(r["effect_size"]):
            bits.append(f"r = {float(r['effect_size']):+.2f}")
        if "p_adjusted" in r and np.isfinite(r["p_adjusted"]):
            p = float(r["p_adjusted"])
            bits.append("p < 0.001" if p < 0.001 else f"p = {p:.3f}")
        if bits:
            ax.annotate("   ".join(bits), xy=(1.0, y), xycoords=("axes fraction", "data"),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color="#52514e")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    # Bold the arm headings.  Done after the fact rather than with mathtext,
    # which turns an underscore in a condition name into a subscript.
    for position in header_rows:
        ax.get_yticklabels()[position].set_fontweight("bold")
    ax.set_ylim(0.3, len(rows) + 0.7)
    ax.set_xlabel(f"change in {metric.upper()} against the reference"
                  + ("   (left of the line is better)" if lower_is_better
                     else "   (a shift either way is a change, not an improvement)"))
    ax.tick_params(axis="y", length=0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", visible=False)

    if title:
        ax.set_title(title, fontsize=10)
    # Room on the right for the annotations, which sit outside the axes.
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    return fig


def plot_ratio_agreement(
    metrics_table: Any,
    metric: str,
    reference: str,
    conditions: Sequence[str] | None = None,
    target: float = 1.0,
    tolerance: float = 0.02,
    aggregate_runs: str = "mean",
    title: str = "",
):
    """How close a ratio sits to the value that would be correct, per condition.

    A ratio of predicted to true is right at ``target``, not at zero, and it can
    miss in two directions.  Drawn as a paired dot plot it is close to
    unreadable: seventy scans start at seventy different places, the lines cross
    the target from both sides, and the only summary on offer is a count of
    which moved down.

    Here each condition is one row: a dot at the median, a thick bar over the
    interquartile range, a thin line over the 5th to 95th percentile, against a
    marked target.  That shows the two things that decide whether a condition is
    any good -- whether it is centred on the target, and how tightly -- and it
    shows over- and under-estimation as position rather than as sign.

    Colour states where the median sits, not how it compares with the
    reference: grey inside ``tolerance`` of the target, blue below it, red
    above.  Colouring by "closer than the reference" was the first attempt and
    it was misleading -- it painted a median of 1.003 as a win over 0.994, a
    distinction of three parts in a thousand that no reader should be invited
    to take seriously.  An absolute statement about each condition can be read
    on its own and does not manufacture a difference out of noise.
    """
    plt = _style()

    collapsed = (metrics_table.groupby(["condition", "scan_id"], dropna=False)[metric]
                 .agg(aggregate_runs).reset_index())
    names = [reference] + [c for c in (conditions or []) if c != reference]
    names = [n for n in names if n in set(collapsed["condition"])]
    if not names:
        raise ValueError("no condition to draw")

    stats = {}
    for name in names:
        values = collapsed[collapsed["condition"] == name][metric].dropna()
        stats[name] = (values.median(), values.quantile(0.25), values.quantile(0.75),
                       values.quantile(0.05), values.quantile(0.95), values)
    fig, ax = plt.subplots(figsize=(6.8, 0.38 * len(names) + 1.5))
    band = tolerance * target
    ax.axvspan(target - band, target + band, color="#4a4a4a", alpha=0.07, zorder=1)
    ax.axvline(target, color="#4a4a4a", lw=1.0, ls="--", zorder=2)

    for index, name in enumerate(names):
        y = len(names) - index
        median, q1, q3, p5, p95, values = stats[name]
        if abs(median - target) <= band:
            colour = UNDECIDED
        else:
            colour = BETTER if median < target else WORSE
        ax.plot([p5, p95], [y, y], "-", color=colour, lw=1.2, alpha=0.65, zorder=3)
        ax.plot([q1, q3], [y, y], "-", color=colour, lw=5.0, alpha=0.85,
                solid_capstyle="butt", zorder=4)
        ax.plot([median], [y], "o", color=colour, ms=7, zorder=5,
                markeredgecolor="white", markeredgewidth=1.4)

        within = float((values.sub(target).abs() <= 0.10 * target).mean())
        ax.annotate(f"median {median:.3f}     {within:.0%} within ±10 %",
                    xy=(1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color="#52514e")

    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels(list(reversed(names)), fontsize=8)
    ax.set_ylim(0.4, len(names) + 0.6)
    ax.set_xlabel(f"{metric.replace('_', ' ')}"
                  f"      ← underestimates    {target:g} = correct    overestimates →"
                  f"\nband: median within ±{tolerance:.0%} of correct")
    ax.tick_params(axis="y", length=0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.74, 1))
    return fig


def plot_bias_variance(decomposition: Any, title: str = "", reference: str | None = None):
    """Stacked bias-squared and variance per condition.

    The stack is the point: if the total barely moves while the two components
    trade against each other, an RMSE-only reading would call it a null result.
    """
    plt = _style()

    table = decomposition.sort_values("mse")
    fig, ax = plt.subplots(figsize=(1.5 * max(len(table), 3), 3.6))
    x = np.arange(len(table))

    # Hatching on the upper segment is secondary encoding: the two segments
    # stay distinguishable in greyscale and in print, where the hue does not
    # survive.  The thin white edge is the 2px surface gap between fills.
    ax.bar(x, table["bias_squared"], color="#2a78d6", label="Bias$^2$",
           edgecolor="white", linewidth=0.8)
    ax.bar(x, table["variance"], bottom=table["bias_squared"], color="#eb6834",
           label="Variance", edgecolor="white", linewidth=0.8, hatch="///")
    ax.plot(x, table["mse"], "k_", ms=18, mew=2, label="MSE")

    # The reference condition is the thing every other bar is judged against,
    # so its total is drawn as a line across the figure rather than left for
    # the reader to hold in their head.
    reference_row = (table[table["condition"] == reference] if reference is not None
                     else table.iloc[0:0])
    if len(reference_row):
        level = float(reference_row["mse"].iloc[0])
        ax.axhline(level, color="#4a4a4a", lw=0.9, ls=":", zorder=1)
        ax.annotate("reference MSE", xy=(1.0, level), xycoords=("axes fraction", "data"),
                    xytext=(-3, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7, color="#52514e")

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


def plot_error_by_time_bin(binned: Any, metric: str = "rmse", title: str = "",
                           reference: str | None = None, max_series: int = 5):
    """Error against frame timing, one line per condition.

    This is where the recovery-noise trade-off across the time-activity curve
    becomes visible: early frames have very low counts and late ones
    comparatively high, and PVC is not expected to behave the same in both.

    An arm can hold ten conditions and most of them trace nearly the same
    shape, so beyond ``max_series`` the figure keeps the reference and the
    conditions furthest from it and says how many it left out.  Showing the
    extremes and naming the omission is more honest than showing everything
    illegibly, and more honest than silently showing the first few.
    """
    plt = _style()

    order = sorted(binned["time_bin"].unique(),
                   key=lambda s: float(str(s).split("-")[0].replace(">=", "")))
    positions = {label: i for i, label in enumerate(order)}

    overall = binned.groupby("condition")[metric].mean()
    names = list(overall.index)
    omitted = 0
    if len(names) > max_series:
        anchor = float(overall[reference]) if reference in overall.index else float(overall.median())
        ranked = (overall - anchor).abs().sort_values(ascending=False)
        keep = [c for c in ranked.index if c != reference][: max_series - (reference is not None)]
        omitted = len(names) - len(keep) - (1 if reference in overall.index else 0)
        names = ([reference] if reference in overall.index else []) + keep

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for index, condition in enumerate(names):
        group = binned[binned["condition"] == condition].copy()
        group["_x"] = group["time_bin"].map(positions)
        group = group.sort_values("_x")
        is_reference = condition == reference
        ax.plot(group["_x"], group[metric], "o-",
                lw=2.4 if is_reference else 1.6, ms=5 if is_reference else 4,
                color=_colour(condition, index), label=condition,
                zorder=5 if is_reference else 3)

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.set_xlabel("Frame time")
    ax.set_ylabel(metric.upper())
    ax.legend(fontsize=8, frameon=False)
    if omitted:
        ax.annotate(f"{omitted} further condition(s) omitted; the reference and the "
                    "most distant are shown",
                    xy=(0, -0.30), xycoords="axes fraction", fontsize=7, color="#6b7280")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def _sweep(table: Any, x: str, metric: str, xlabel: str, title: str,
           reference_level: float | None = None, reference_label: str = "uncorrected"):
    """One line per (method, model) of ``metric`` against ``x``.

    Pretrained and retrained are different models evaluated on the same grid,
    so they are separate series; drawing them as one line would connect points
    that do not belong to the same curve.
    """
    plt = _style()

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    keys = ["method", "model"] if "model" in table.columns else ["method"]
    index = 0
    for key, group in table.groupby(keys):
        group = group.sort_values(x)
        if group[x].nunique() < 2:
            # One point is not a sweep; drawing it adds a lone marker and a
            # legend entry that suggest a curve that was never measured.
            continue
        label = " / ".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
        colour = _colour(label, index)
        ax.plot(group[x], group[metric], "o-", lw=1.6, ms=5, color=colour, label=label)
        if "ci_low" in group.columns and "ci_high" in group.columns:
            ax.fill_between(group[x], group["ci_low"], group["ci_high"],
                            color=colour, alpha=0.15)
        index += 1

    # The uncorrected condition is not a point on a correction-strength axis,
    # so it cannot be a marker on the line; drawn as a rule it still gives the
    # reader the comparison that matters -- whether correcting at all moved the
    # quantity, and in which direction.
    if reference_level is not None and np.isfinite(reference_level):
        ax.axhline(reference_level, color="#4a4a4a", lw=0.9, ls="--", zorder=1)
        ax.annotate(reference_label, xy=(1.0, reference_level),
                    xycoords=("axes fraction", "data"),
                    xytext=(-3, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7, color="#52514e")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.replace("_", " "))
    if index:
        ax.legend(fontsize=8, title="Method / model", title_fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_iteration_sweep(table: Any, metric: str = "rmse", title: str = "",
                         reference_level: float | None = None):
    """Metric against deconvolution iteration count, per method and model.

    The iteration count is fixed for the main analysis; this figure is the
    sensitivity analysis that shows how much that choice matters, rather than
    tuning it to the best result.
    """
    fig, _ = _sweep(table, "iterations", metric, "Deconvolution iterations", title,
                    reference_level=reference_level)
    return fig


def plot_psf_sweep(table: Any, metric: str = "rmse", title: str = "",
                   reference_level: float | None = None):
    """Metric against the PSF scale the correction assumed, per method and model.

    The measured PSF has uncertainty and rests on assumptions -- static point
    source, centre of the field, spatially invariant -- so the correction is
    run again with the FWHM deliberately scaled.  A flat line means the result
    does not hinge on the PSF being exactly right; a steep one means it does.
    The vertical rule marks the measured PSF, which is the analysis as run.
    """
    fig, ax = _sweep(table, "psf_scale", metric, "Assumed PSF FWHM (x measured)", title,
                     reference_level=reference_level)
    ax.axvline(1.0, color="#6b7280", lw=0.9, ls="--", zorder=0)
    scales = sorted(set(float(v) for v in table["psf_scale"]))
    if scales:
        ax.set_xticks(scales)
        ax.set_xticklabels([f"{v:g}" for v in scales])
    fig.tight_layout()
    return fig


def plot_condition_ladder(
    metrics_table: Any,
    metric: str,
    order: Sequence[str],
    reference: str,
    labels: Mapping[str, str] | None = None,
    target: float | None = None,
    title: str = "",
    aggregate_runs: str = "mean",
    n_boot: int = 2000,
    seed: int = 0,
    group_edges: Sequence[int] = (),
    within_groups: Sequence[Sequence[str]] = (),
):
    """A metric across an ordered ladder of conditions, drawn as paired data.

    This exists for the effect that a single-scan curve cannot show.  The step
    from uncorrected to corrected is large enough to see on one scan; the step
    from one deconvolution iteration count to the next is a few tenths of a
    percent of the peak, which no honest zoom on one curve will reveal.  It is
    nevertheless real and highly consistent, and the reason it is real is
    visible only in the pairing: almost every scan moves the same way.

    The left panel is the absolute value, one thin line per scan plus the
    median, so the size of the uncorrected-to-corrected step is read against
    the spread it has to beat.  The right panel is the paired difference from
    ``reference`` with a bootstrap interval on the median, on whatever scale
    those differences need -- which is where the monotone trend along the
    correction-strength axis becomes legible.

    ``within_groups`` names the rungs that belong to one method, in increasing
    strength.  Each group is then also drawn in an inset as a difference from
    its own first rung.  That panel exists because the intervals in the main
    right panel are intervals on a difference from the uncorrected condition,
    and they overlap between rungs: a reader who compares them would conclude
    that the strength axis is flat.  It is not -- it is small, and it is
    measured within scan, so it has to be drawn that way.
    """
    plt = _style()

    collapsed = (
        metrics_table.groupby(["condition", "scan_id"], dropna=False)[metric]
        .agg(aggregate_runs)
        .reset_index()
    )
    wide = collapsed.pivot(index="scan_id", columns="condition", values=metric)
    order = [c for c in order if c in wide.columns]
    wide = wide[order].dropna()
    names = dict(labels or {})
    ticks = [names.get(c, c) for c in order]
    positions = np.arange(len(order))

    usable = [[c for c in g if c in wide.columns] for g in within_groups]
    usable = [g for g in usable if len(g) > 1]
    n_panels = 3 if usable else 2
    widths = [1.05, 1.0, 0.85] if usable else [1.05, 1.0]
    fig, axes = plt.subplots(1, n_panels, figsize=(4.7 * n_panels - 0.6, 4.0),
                             gridspec_kw={"width_ratios": widths})
    left, right = axes[0], axes[1]
    step_ax = axes[2] if usable else None

    values = wide.to_numpy()
    for row in values:
        left.plot(positions, row, "-", color="#9aa0a6", lw=0.6, alpha=0.30, zorder=1)
    median = np.median(values, axis=0)
    for index, condition in enumerate(order):
        left.plot(index, median[index], "o", ms=7, zorder=5,
                  color=_colour(condition, max(index - 1, 0)))
    left.plot(positions, median, "-", color="black", lw=2.0, zorder=4)
    if target is not None:
        left.axhline(target, color="#4a4a4a", lw=0.9, ls="--", zorder=2)
        left.annotate("perfect", xy=(1.0, target), xycoords=("axes fraction", "data"),
                      xytext=(-3, 3), textcoords="offset points",
                      ha="right", va="bottom", fontsize=7.5, color="#52514e")
    left.set_ylabel(metric.replace("_", " ") + "   (one line per scan)")
    left.set_title(f"Per scan, n = {len(wide)}", fontsize=9)

    # Paired differences.  The reference sits at exactly zero by construction
    # and is drawn as the rule rather than as a point with no interval.
    rng = np.random.default_rng(seed)
    base = wide[reference].to_numpy()
    xs, meds, los, his = [], [], [], []
    for index, condition in enumerate(order):
        if condition == reference:
            continue
        diff = wide[condition].to_numpy() - base
        draws = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        boot = np.median(diff[draws], axis=1)
        xs.append(index)
        meds.append(float(np.median(diff)))
        los.append(float(np.percentile(boot, 2.5)))
        his.append(float(np.percentile(boot, 97.5)))
    right.axhline(0.0, color="#4a4a4a", lw=0.9, ls="--", zorder=1)
    right.annotate(names.get(reference, reference), xy=(0.0, 0.0),
                   xycoords=("axes fraction", "data"), xytext=(2, 4),
                   textcoords="offset points", fontsize=7.5, color="#52514e")
    for x, m, lo, hi in zip(xs, meds, los, his):
        colour = _colour(order[x], max(x - 1, 0))
        right.plot([x, x], [lo, hi], "-", color=colour, lw=2.0, solid_capstyle="round")
        right.plot(x, m, "o", ms=7, color=colour, zorder=5)
    right.plot(xs, meds, "-", color="black", lw=1.4, zorder=4, alpha=0.75)
    right.set_ylabel("paired difference from " + names.get(reference, reference))
    right.set_title("Median difference, 95 % bootstrap interval", fontsize=9)

    if step_ax is not None:
        tick_pos, tick_lab, anchor_slots = [], [], []
        slot = 0.0
        for group in usable:
            anchor = wide[group[0]].to_numpy()
            anchor_slots.append(slot)
            step_ax.plot(slot, 0.0, "o", ms=6, mfc="white", mec="#6b7280", mew=1.4, zorder=5)
            tick_pos.append(slot)
            tick_lab.append(names.get(group[0], group[0]))
            slot += 1
            for condition in group[1:]:
                diff = wide[condition].to_numpy() - anchor
                draws = rng.integers(0, len(diff), size=(n_boot, len(diff)))
                boot = np.median(diff[draws], axis=1)
                colour = _colour(condition, max(order.index(condition) - 1, 0))
                step_ax.plot([slot, slot],
                             [np.percentile(boot, 2.5), np.percentile(boot, 97.5)],
                             "-", color=colour, lw=2.0, solid_capstyle="round")
                step_ax.plot(slot, np.median(diff), "o", ms=7, color=colour, zorder=5)
                tick_pos.append(slot)
                tick_lab.append(names.get(condition, condition))
                slot += 1
            slot += 0.6
        step_ax.axhline(0.0, color="#4a4a4a", lw=0.9, ls="--", zorder=1)
        step_ax.set_xticks(tick_pos)
        step_ax.set_xticklabels(tick_lab, rotation=25, ha="right", fontsize=8)
        step_ax.set_xlim(-0.6, slot - 1.0)
        step_ax.set_ylabel("paired difference from the weakest setting")
        step_ax.set_title("Within method, same scan", fontsize=9)

    for ax in (left, right):
        ax.set_xticks(positions)
        ax.set_xticklabels(ticks, rotation=25, ha="right", fontsize=8)
        ax.set_xlim(-0.5, len(order) - 0.5)
        # A rule between method blocks stops the eye reading the ladder as one
        # continuous axis, which it is not: RL and RVC are different methods.
        for edge in group_edges:
            if 0 < edge < len(order):
                ax.axvline(edge - 0.5, color="#d0d0d0", lw=0.9, zorder=0)

    if title:
        fig.suptitle(title, fontsize=10)
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
