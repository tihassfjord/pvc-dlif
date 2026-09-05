"""LaTeX tables for the thesis.

Written as ``booktabs`` fragments to be ``\\input{}`` into the thesis source, so
the numbers in the text come from the result tables rather than being copied by
hand.

The formatting rules encode a few conventions worth being consistent about:
a p-value is always shown next to its effect size and interval, adjusted
p-values are labelled as adjusted, and a number is never printed to more digits
than the data support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = [
    "format_p", "format_value", "latex_table", "comparison_table",
    "metrics_summary_table", "bias_variance_table", "write_table",
]

_LATEX_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def escape(text: Any) -> str:
    out = str(text)
    for char, replacement in _LATEX_ESCAPES.items():
        out = out.replace(char, replacement)
    return out


def format_p(value: float, adjusted: bool = False) -> str:
    """Format a p-value the way a reader expects to see it."""
    if value is None or not np.isfinite(value):
        return "--"
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.3f}"


def format_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(value)
    if not np.isfinite(number):
        return "--"
    if number != 0 and (abs(number) < 10 ** (-digits) or abs(number) >= 10 ** 5):
        return f"{number:.{digits}e}".replace("e", r"\times 10^{") + "}"
    return f"{number:.{digits}f}"


def latex_table(
    rows: Sequence[Sequence[str]],
    header: Sequence[str],
    caption: str = "",
    label: str = "",
    column_spec: str | None = None,
    notes: str = "",
) -> str:
    """Assemble a ``booktabs`` table body."""
    spec = column_spec or ("l" + "r" * (len(header) - 1))
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}" if caption else "",
        rf"  \label{{{label}}}" if label else "",
        rf"  \begin{{tabular}}{{{spec}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    lines += ["    " + " & ".join(row) + r" \\" for row in rows]
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if notes:
        lines.append(rf"  \par\vspace{{2pt}}\footnotesize {notes}")
    lines.append(r"\end{table}")
    return "\n".join(line for line in lines if line)


def comparison_table(
    comparisons: Any,
    metrics: Sequence[str] | None = None,
    caption: str = "Paired comparison of each condition against the reference.",
    label: str = "tab:pvc-comparison",
    alpha: float = 0.05,
    n_boot: int | None = None,
    ci_level: float = 0.95,
) -> str:
    """One row per (metric, condition): difference, interval, effect size, p.

    The confidence interval is on the median paired difference, so a reader can
    see the size of the effect and its uncertainty rather than only whether it
    cleared a threshold.
    """
    table = comparisons.copy()
    if metrics:
        table = table[table["metric"].isin(metrics)]

    rows: list[list[str]] = []
    for _, row in table.iterrows():
        significant = (
            row.get("p_adjusted") is not None
            and np.isfinite(row.get("p_adjusted", np.nan))
            and row["p_adjusted"] < alpha
        )
        condition = escape(row["condition"])
        if significant:
            condition = rf"\textbf{{{condition}}}"
        rows.append(
            [
                escape(row["metric"]),
                condition,
                str(int(row["n_pairs"])),
                format_value(row["median_reference"]),
                format_value(row["median_condition"]),
                format_value(row["median_difference"]),
                f"[{format_value(row['ci_low'])}, {format_value(row['ci_high'])}]",
                format_value(row["effect_size"], 2),
                format_p(row["p_value"]),
                format_p(row.get("p_adjusted", float("nan")), adjusted=True),
            ]
        )

    correction = table["correction"].iloc[0] if "correction" in table.columns and len(table) else "none"
    level = f"{ci_level:.0%}".replace("%", r"\%")
    resamples = (
        rf"{level} percentile bootstrap ({n_boot:,} resamples)".replace(",", r"\,")
        if n_boot else rf"{level} percentile bootstrap"
    )
    return latex_table(
        rows,
        header=[
            "Metric", "Condition", r"$n$", "Median ref.", "Median cond.",
            r"$\Delta$ median", rf"{level} CI", r"$r_{rb}$", r"$p$", r"$p_{\mathrm{adj}}$",
        ],
        caption=caption,
        label=label,
        column_spec="llrrrrrrrr",
        notes=(
            rf"Paired Wilcoxon signed-rank test; $r_{{rb}}$ is the matched-pairs rank-biserial "
            rf"effect size; {resamples} confidence interval on the median paired difference; "
            rf"$p_{{\mathrm{{adj}}}}$ is {escape(correction)}-corrected within each metric. "
            rf"Bold marks $p_{{\mathrm{{adj}}}} < {alpha}$. A negative $\Delta$ means the "
            rf"condition reduced the metric relative to the reference."
        ),
    )


def metrics_summary_table(
    metrics_table: Any,
    metrics: Sequence[str] = ("rmse", "nrmse", "auc_ratio", "peak_height_ratio", "peak_time_error"),
    caption: str = "Prediction accuracy by condition, median (interquartile range).",
    label: str = "tab:pvc-metrics",
    aggregate_runs: str = "mean",
) -> str:
    """Median and IQR of each metric, one column per metric, one row per condition."""
    collapsed = (
        metrics_table.groupby(["condition", "scan_id"], dropna=False)[list(metrics)]
        .agg(aggregate_runs)
        .reset_index()
    )

    rows: list[list[str]] = []
    for condition, group in collapsed.groupby("condition"):
        cells = [escape(condition), str(group["scan_id"].nunique())]
        for metric in metrics:
            values = group[metric].dropna()
            if values.empty:
                cells.append("--")
                continue
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            cells.append(f"{format_value(median)} ({format_value(q1)}--{format_value(q3)})")
        rows.append(cells)

    return latex_table(
        rows,
        header=["Condition", r"$n$", *[escape(m) for m in metrics]],
        caption=caption,
        label=label,
        column_spec="lr" + "r" * len(metrics),
        notes=(
            "Values are medians across scans with the interquartile range in parentheses. "
            "Repeats of a retrained condition are averaged per scan first, so every scan "
            "contributes one observation."
        ),
    )


def bias_variance_table(
    decomposition: Any,
    caption: str = "Decomposition of mean squared error into bias and variance.",
    label: str = "tab:bias-variance",
) -> str:
    """Bias-squared, variance and their share of MSE, per condition."""
    rows: list[list[str]] = []
    for _, row in decomposition.iterrows():
        estimable = bool(row.get("estimable", True))
        variance_cell = format_value(row["variance"], 4) if estimable else "--"
        share_cell = format_value(row.get("variance_share", float("nan")), 2) if estimable else "--"
        rows.append(
            [
                escape(row["condition"]),
                str(int(row.get("n_scans", 0))),
                str(int(row.get("n_repeats", 1))),
                format_value(row["mse"], 4),
                format_value(row["bias_squared"], 4),
                variance_cell,
                share_cell,
                format_value(row.get("signed_bias", float("nan")), 4),
            ]
        )

    return latex_table(
        rows,
        header=[
            "Condition", r"$n$", "Repeats", "MSE", r"Bias$^2$", "Variance",
            "Var. share", "Signed bias",
        ],
        caption=caption,
        label=label,
        column_spec="lrrrrrrr",
        notes=(
            "The variance term requires repeated training runs and is therefore only "
            "estimable for the retrained conditions; a dash marks a condition with a single "
            "deterministic model. Signed bias is positive when the predicted input function "
            "over-estimates the arterial ground truth."
        ),
    )


def write_table(latex: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    return path
