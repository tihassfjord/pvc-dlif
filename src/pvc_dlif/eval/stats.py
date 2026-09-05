"""Paired statistics for comparing conditions.

The design is paired: every scan contributes a prediction under every
condition, so the comparison of two conditions is a comparison of matched
pairs.  With ~70-90 scans and metrics that are bounded, skewed or ratio-valued,
a non-parametric paired test is the honest choice, which is why the Wilcoxon
signed-rank test is the default.

Three things are reported for every comparison, and all three belong in the
thesis: the p-value, an effect size (matched-pairs rank-biserial correlation),
and a bootstrap confidence interval for the median paired difference.  A
p-value alone would not say whether a difference matters, and with several
conditions compared against one reference the p-values need correcting -- Holm
by default, which controls the family-wise error rate without assuming
independence.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

__all__ = [
    "PairedComparison", "paired_test", "compare_conditions",
    "adjust_pvalues", "bootstrap_ci", "rank_biserial",
]


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation.

    ``(W+ - W-) / (W+ + W-)``, the effect size that goes with the Wilcoxon
    signed-rank test.  It runs from -1 to +1; the sign follows the sign of the
    typical difference and the magnitude says how consistently the pairs move in
    that direction.  Ties (zero differences) are dropped, as the test does.
    """
    from scipy.stats import rankdata

    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    return float((positive - negative) / total) if total > 0 else 0.0


def bootstrap_ci(
    differences: np.ndarray,
    statistic: str = "median",
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a paired statistic.

    Resamples *pairs*, not observations, which is what keeps the pairing intact.
    """
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    func = np.median if statistic == "median" else np.mean
    indices = rng.integers(0, values.size, size=(int(n_boot), values.size))
    samples = func(values[indices], axis=1)
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(low), float(high))


@dataclass
class PairedComparison:
    """One condition compared against the reference on one metric."""

    metric: str
    condition: str
    reference: str
    n_pairs: int
    median_reference: float
    median_condition: float
    median_difference: float
    mean_difference: float
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float
    p_adjusted: float | None
    effect_size: float
    effect_size_name: str
    test: str
    n_zero_differences: int
    note: str | None = None

    @property
    def improves(self) -> bool | None:
        """Whether the condition moved the metric in the better direction.

        ``None`` when the metric has no obvious better direction (for example a
        ratio that should be near 1); the caller decides in that case.
        """
        if not np.isfinite(self.median_difference):
            return None
        return self.median_difference < 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["improves"] = self.improves
        return payload


def paired_test(
    condition_values: Sequence[float],
    reference_values: Sequence[float],
    metric: str = "",
    condition: str = "condition",
    reference: str = "reference",
    test: str = "wilcoxon",
    alternative: str = "two-sided",
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> PairedComparison:
    """Compare two conditions on one metric over matched scans.

    ``condition_values[i]`` and ``reference_values[i]`` must be the same scan.
    Pairs where either value is missing are dropped, and the number actually
    used is reported.
    """
    from scipy import stats

    a = np.asarray(condition_values, dtype=np.float64)
    b = np.asarray(reference_values, dtype=np.float64)
    if a.size != b.size:
        raise ValueError(f"unpaired input: {a.size} vs {b.size} values")

    valid = np.isfinite(a) & np.isfinite(b)
    dropped = int((~valid).sum())
    a, b = a[valid], b[valid]
    differences = a - b
    n_zero = int(np.sum(differences == 0))

    note = None
    if dropped:
        note = f"{dropped} pair(s) dropped for missing values"

    if a.size < 3:
        return PairedComparison(
            metric=metric, condition=condition, reference=reference, n_pairs=int(a.size),
            median_reference=float(np.median(b)) if b.size else float("nan"),
            median_condition=float(np.median(a)) if a.size else float("nan"),
            median_difference=float(np.median(differences)) if differences.size else float("nan"),
            mean_difference=float(np.mean(differences)) if differences.size else float("nan"),
            ci_low=float("nan"), ci_high=float("nan"),
            statistic=float("nan"), p_value=float("nan"), p_adjusted=None,
            effect_size=float("nan"), effect_size_name="rank_biserial", test=test,
            n_zero_differences=n_zero,
            note=(note + "; " if note else "") + "too few pairs for a test",
        )

    if np.all(differences == 0):
        statistic, p_value = float("nan"), 1.0
        note = (note + "; " if note else "") + "all paired differences are zero"
    elif test == "wilcoxon":
        result = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    elif test == "ttest":
        result = stats.ttest_rel(a, b, alternative=alternative)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    elif test == "sign":
        n_positive = int(np.sum(differences > 0))
        n_used = int(np.sum(differences != 0))
        result = stats.binomtest(n_positive, n_used, 0.5, alternative=alternative)
        statistic, p_value = float(n_positive), float(result.pvalue)
    else:
        raise ValueError(f"unknown test {test!r}")

    ci_low, ci_high = bootstrap_ci(differences, "median", n_boot, alpha, seed)

    return PairedComparison(
        metric=metric, condition=condition, reference=reference, n_pairs=int(a.size),
        median_reference=float(np.median(b)),
        median_condition=float(np.median(a)),
        median_difference=float(np.median(differences)),
        mean_difference=float(np.mean(differences)),
        ci_low=ci_low, ci_high=ci_high,
        statistic=statistic, p_value=p_value, p_adjusted=None,
        effect_size=rank_biserial(differences), effect_size_name="rank_biserial",
        test=test, n_zero_differences=n_zero, note=note,
    )


def adjust_pvalues(p_values: Sequence[float], method: str = "holm") -> list[float]:
    """Correct a family of p-values.

    ``holm`` controls the family-wise error rate and makes no independence
    assumption -- appropriate when several conditions are each compared against
    the same reference.  ``bh`` (Benjamini-Hochberg) controls the false
    discovery rate and is the right choice for a wider exploratory sweep such as
    the per-metric grid.
    """
    values = np.asarray(p_values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan)
    p = values[finite]
    n = p.size
    if n == 0:
        return [float(v) for v in out]

    order = np.argsort(p)
    ordered = p[order]
    adjusted = np.empty_like(ordered)

    if method == "holm":
        running = 0.0
        for i, value in enumerate(ordered):
            running = max(running, (n - i) * value)
            adjusted[i] = min(running, 1.0)
    elif method in {"bh", "fdr_bh", "benjamini-hochberg"}:
        running = 1.0
        for i in range(n - 1, -1, -1):
            running = min(running, ordered[i] * n / (i + 1))
            adjusted[i] = min(running, 1.0)
    elif method in {"bonferroni"}:
        adjusted = np.minimum(ordered * n, 1.0)
    elif method in {"none", None}:
        adjusted = ordered
    else:
        raise ValueError(f"unknown correction method {method!r}")

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite] = restored
    return [float(v) for v in out]


def compare_conditions(
    metrics_table: Any,
    reference: str,
    metrics: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    aggregate_runs: str = "mean",
    test: str = "wilcoxon",
    alternative: str = "two-sided",
    correction: str = "holm",
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Compare every condition against the reference, on every metric.

    ``metrics_table`` is the per-curve table from
    :func:`pvc_dlif.eval.curve_metrics.metrics_frame`.  Conditions with
    repeated runs are collapsed to one value per scan first (mean by default),
    so each scan contributes exactly one paired observation.

    Multiple-comparison correction is applied *within* each metric across
    conditions, which matches the question being asked: for this metric, does
    any condition differ from the reference?
    """
    import pandas as pd

    table = metrics_table.copy()
    if "condition" not in table.columns or "scan_id" not in table.columns:
        raise ValueError("metrics_table needs 'condition' and 'scan_id' columns")

    reserved = {"scan_id", "condition", "model", "fold", "run"}
    if metrics is None:
        metrics = [
            c for c in table.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(table[c])
        ]

    collapsed = (
        table.groupby(["condition", "scan_id"], dropna=False)[list(metrics)]
        .agg(aggregate_runs)
        .reset_index()
    )

    available = list(collapsed["condition"].unique())
    if reference not in available:
        raise ValueError(f"reference condition {reference!r} not in the table ({available})")
    targets = [c for c in (conditions or available) if c != reference]

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        ref_series = collapsed[collapsed["condition"] == reference].set_index("scan_id")[metric]
        comparisons: list[PairedComparison] = []
        for condition in targets:
            cond_series = collapsed[collapsed["condition"] == condition].set_index("scan_id")[metric]
            shared = ref_series.index.intersection(cond_series.index)
            if len(shared) == 0:
                LOGGER.warning("No shared scans between %s and %s", condition, reference)
                continue
            if len(shared) < len(ref_series):
                LOGGER.warning(
                    "%s vs %s on %s: %d of %d scans are shared; the design is unbalanced",
                    condition, reference, metric, len(shared), len(ref_series),
                )
            comparisons.append(
                paired_test(
                    cond_series.loc[shared].to_numpy(),
                    ref_series.loc[shared].to_numpy(),
                    metric=metric, condition=condition, reference=reference,
                    test=test, alternative=alternative, n_boot=n_boot, alpha=alpha, seed=seed,
                )
            )

        adjusted = adjust_pvalues([c.p_value for c in comparisons], correction)
        for comparison, p_adj in zip(comparisons, adjusted):
            comparison.p_adjusted = p_adj
            rows.append({**comparison.to_dict(), "correction": correction})

    return pd.DataFrame(rows)
