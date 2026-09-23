"""
Exploratory data analysis engine.

Reusable, chart-free analysis functions covering univariate, bivariate,
multivariate and outlier analysis. Everything returns a structured dataclass or
a tidy dataframe; rendering is the visualisation layer's job. Keeping them apart
means the same numbers feed the dashboard, the tests and the written report.

Every function guards against too little data. An analysis computed on three
rows is worse than no analysis, because it looks authoritative and is not, so
small selections return an explicit "insufficient data" result instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import settings
from src.logging_setup import get_logger
from src.schema import ID_COLUMN, ColumnRegistry, Role, build_registry

logger = get_logger(__name__)

CorrelationMethod = Literal["pearson", "spearman", "kendall"]

#: Correlations weaker than this are treated as negligible when ranking drivers.
WEAK_CORRELATION: Final[float] = 0.10
MODERATE_CORRELATION: Final[float] = 0.30
STRONG_CORRELATION: Final[float] = 0.50


# --------------------------------------------------------------------------- #
# Result structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NumericSummary:
    """Descriptive statistics for one numeric column."""

    column: str
    count: int
    missing: int
    mean: float
    median: float
    std: float
    minimum: float
    q1: float
    q3: float
    maximum: float
    skewness: float
    kurtosis: float

    @property
    def iqr(self) -> float:
        """Interquartile range."""
        return self.q3 - self.q1

    @property
    def coefficient_of_variation(self) -> float | None:
        """Std / |mean|. None when the mean is zero, which makes it undefined."""
        if self.mean == 0:
            return None
        return self.std / abs(self.mean)

    @property
    def mean_median_gap_pct(self) -> float | None:
        """Percentage gap between mean and median - a plain-language skew signal."""
        if self.median == 0:
            return None
        return (self.mean - self.median) / abs(self.median) * 100.0

    @property
    def skew_description(self) -> str:
        """Readable interpretation of the skewness value."""
        if self.skewness > 1.0:
            return "strongly right-skewed"
        if self.skewness > 0.5:
            return "moderately right-skewed"
        if self.skewness < -1.0:
            return "strongly left-skewed"
        if self.skewness < -0.5:
            return "moderately left-skewed"
        return "approximately symmetric"

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "column": self.column,
            "count": self.count,
            "missing": self.missing,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.minimum,
            "q1": self.q1,
            "q3": self.q3,
            "max": self.maximum,
            "iqr": self.iqr,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "skew_description": self.skew_description,
            "mean_median_gap_pct": self.mean_median_gap_pct,
        }


@dataclass(frozen=True)
class FrequencyTable:
    """Frequency distribution of one categorical column."""

    column: str
    counts: dict[str, int]
    percentages: dict[str, float]
    n_categories: int
    n_rows: int
    missing: int

    @property
    def top_category(self) -> str | None:
        """Most frequent category."""
        return next(iter(self.counts), None)

    @property
    def top_share(self) -> float | None:
        """Percentage share of the most frequent category."""
        top = self.top_category
        return self.percentages.get(top) if top is not None else None

    def to_frame(self) -> pd.DataFrame:
        """Tidy dataframe for tables and charts."""
        return pd.DataFrame(
            {
                "category": list(self.counts),
                "count": list(self.counts.values()),
                "percentage": [self.percentages[key] for key in self.counts],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "column": self.column,
            "n_categories": self.n_categories,
            "counts": dict(self.counts),
            "percentages": {k: round(v, 4) for k, v in self.percentages.items()},
            "top_category": self.top_category,
            "top_share": self.top_share,
            "missing": self.missing,
        }


@dataclass(frozen=True)
class CorrelationPair:
    """One pair of correlated columns."""

    left: str
    right: str
    correlation: float

    @property
    def strength(self) -> str:
        """Readable strength label for the absolute correlation."""
        magnitude = abs(self.correlation)
        if magnitude >= STRONG_CORRELATION:
            return "strong"
        if magnitude >= MODERATE_CORRELATION:
            return "moderate"
        if magnitude >= WEAK_CORRELATION:
            return "weak"
        return "negligible"

    @property
    def direction(self) -> str:
        """Sign of the relationship."""
        return "positive" if self.correlation > 0 else "negative"

    def describe(self) -> str:
        """Association-only sentence. Never implies causation."""
        return (
            f"'{self.left}' and '{self.right}' show a {self.strength} {self.direction} "
            f"association (r = {self.correlation:.3f})."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "correlation": round(self.correlation, 6),
            "strength": self.strength,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation matrix plus its ranked pairs."""

    method: str
    matrix: pd.DataFrame
    pairs: tuple[CorrelationPair, ...]
    n_rows_used: int
    note: str = ""

    def top_pairs(self, limit: int = 10, exclude_same_family: bool = True) -> list[CorrelationPair]:
        """Strongest associations by absolute correlation.

        Args:
            limit: Maximum pairs to return.
            exclude_same_family: Drop pairs that derive from the same underlying
                measure - different months of one family (``bill_amt_m1`` vs
                ``bill_amt_m2``), a column against its own aggregate
                (``bill_amt_m3`` vs ``avg_bill_6m``), or an alias
                (``utilisation_m1`` vs ``utilisation_latest``). These correlate
                near-perfectly by construction and would otherwise fill the
                ranking with restatements instead of genuine relationships.

        Returns:
            Ranked pairs.
        """
        candidates = list(self.pairs)
        if exclude_same_family:
            candidates = [p for p in candidates if not _same_family(p.left, p.right)]
        return sorted(candidates, key=lambda p: abs(p.correlation), reverse=True)[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_rows_used": self.n_rows_used,
            "columns": list(self.matrix.columns),
            "top_pairs": [p.to_dict() for p in self.top_pairs()],
            "note": self.note,
        }


@dataclass(frozen=True)
class GroupComparison:
    """Comparison of a numeric measure across the levels of a categorical column."""

    group_column: str
    value_column: str
    table: pd.DataFrame
    overall_mean: float
    overall_median: float

    @property
    def highest_group(self) -> str | None:
        """Group with the highest mean."""
        if self.table.empty or "mean" not in self.table.columns:
            return None
        return str(self.table["mean"].idxmax())

    @property
    def lowest_group(self) -> str | None:
        """Group with the lowest mean."""
        if self.table.empty or "mean" not in self.table.columns:
            return None
        return str(self.table["mean"].idxmin())

    @property
    def spread_ratio(self) -> float | None:
        """Highest group mean divided by lowest. None when the lowest is zero."""
        if self.table.empty or "mean" not in self.table.columns:
            return None
        low = float(self.table["mean"].min())
        high = float(self.table["mean"].max())
        if low == 0:
            return None
        return high / low

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_column": self.group_column,
            "value_column": self.value_column,
            "overall_mean": self.overall_mean,
            "overall_median": self.overall_median,
            "highest_group": self.highest_group,
            "lowest_group": self.lowest_group,
            "spread_ratio": self.spread_ratio,
            "groups": self.table.reset_index().to_dict(orient="records"),
        }


@dataclass(frozen=True)
class CategoryOutcome:
    """Outcome rate of a binary target across the levels of a categorical column."""

    category_column: str
    target_column: str
    table: pd.DataFrame
    overall_rate: float
    min_group_size: int

    @property
    def highest_category(self) -> str | None:
        """Category with the highest outcome rate among adequately sized groups."""
        eligible = self.table[self.table["count"] >= self.min_group_size]
        if eligible.empty:
            return None
        return str(eligible["rate_pct"].idxmax())

    @property
    def lowest_category(self) -> str | None:
        """Category with the lowest outcome rate among adequately sized groups."""
        eligible = self.table[self.table["count"] >= self.min_group_size]
        if eligible.empty:
            return None
        return str(eligible["rate_pct"].idxmin())

    @property
    def rate_spread_pp(self) -> float | None:
        """Gap between highest and lowest rate, in percentage points."""
        eligible = self.table[self.table["count"] >= self.min_group_size]
        if eligible.empty:
            return None
        return float(eligible["rate_pct"].max() - eligible["rate_pct"].min())

    def to_dict(self) -> dict[str, Any]:
        return {
            "category_column": self.category_column,
            "target_column": self.target_column,
            "overall_rate_pct": round(self.overall_rate, 4),
            "highest_category": self.highest_category,
            "lowest_category": self.lowest_category,
            "rate_spread_pp": self.rate_spread_pp,
            "categories": self.table.reset_index().to_dict(orient="records"),
        }


@dataclass(frozen=True)
class InsufficientData:
    """Returned in place of a result when there are too few rows to analyse."""

    analysis: str
    n_rows: int
    required_rows: int
    reason: str = ""

    @property
    def message(self) -> str:
        """User-facing explanation."""
        base = (
            f"{self.analysis} needs at least {self.required_rows} rows but the "
            f"current selection has {self.n_rows}."
        )
        return f"{base} {self.reason}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "n_rows": self.n_rows,
            "required_rows": self.required_rows,
            "message": self.message,
            "insufficient_data": True,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: Maps a column to its *derivation family*: the underlying source measure it was
#: computed from. Two columns in the same family are near-tautologically
#: correlated - ``utilisation_latest`` IS ``utilisation_m1``, and
#: ``total_paid_6m`` is exactly six times ``avg_payment_6m`` - so reporting their
#: correlation as an insight is noise that crowds out real relationships.
_DERIVATION_FAMILIES: Final[Mapping[str, tuple[str, ...]]] = {
    "balance": (
        "bill_amt_m",
        "avg_bill_6m",
        "bill_volatility",
        "bill_trend_slope",
        "balance_growth_6m",
    ),
    "payment": ("pay_amt_m", "total_paid_6m", "avg_payment_6m"),
    "delinquency": (
        "pay_status_m",
        "delinquent_months_count",
        "max_delinquency",
        "current_delinquency",
        "ever_delinquent",
        "is_currently_delinquent",
        "delinquency_trend",
    ),
    "utilisation": ("utilisation_", "is_high_utilisation", "is_over_limit"),
    "repayment_ratio": (
        "repayment_ratio_",
        "is_full_payer",
        "is_revolver",
        "months_zero_payment",
    ),
    "credit_limit": ("credit_limit",),
}

#: Which families are *computed from* which others. A correlation across such a
#: pair is circular by construction: utilisation is defined as balance divided by
#: credit limit, so correlating utilisation with balance rediscovers the formula
#: rather than revealing anything about client behaviour.
_FAMILY_DEPENDENCIES: Final[Mapping[str, frozenset[str]]] = {
    "utilisation": frozenset({"balance", "credit_limit"}),
    "repayment_ratio": frozenset({"payment", "balance"}),
}


def _derivation_family(column: str) -> str | None:
    """Return the derivation family of a column, or None when it is standalone."""
    for family, patterns in _DERIVATION_FAMILIES.items():
        for pattern in patterns:
            if column == pattern or column.startswith(pattern):
                return family
    return None


def _same_family(left: str, right: str) -> bool:
    """True when a correlation between two columns is structurally redundant.

    Redundant in three ways, all of which produce a near-perfect correlation that
    says nothing about client behaviour:

    1. **Same family** - different months of one measure (``bill_amt_m1`` vs
       ``bill_amt_m2``), or a measure against its own aggregate or alias
       (``bill_amt_m3`` vs ``avg_bill_6m``, ``utilisation_m1`` vs
       ``utilisation_latest``).
    2. **Derived from** - one family is computed from the other
       (``utilisation`` vs ``bill_amt``, since utilisation = balance / limit).
    3. Either direction of (2).
    """
    left_family = _derivation_family(left)
    right_family = _derivation_family(right)

    if left_family is None or right_family is None:
        return False
    if left_family == right_family:
        return True

    return (
        right_family in _FAMILY_DEPENDENCIES.get(left_family, frozenset())
        or left_family in _FAMILY_DEPENDENCIES.get(right_family, frozenset())
    )


def _analysable_numeric_columns(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> list[str]:
    """Numeric columns worth analysing.

    Excludes the surrogate identifier (numeric but meaningless) and any column
    with fewer than two distinct values.
    """
    reg = registry or build_registry(df)
    id_column = reg.one(Role.CLIENT_ID) or ID_COLUMN

    columns: list[str] = []
    for column in df.select_dtypes(include="number").columns:
        name = str(column)
        if name == id_column:
            continue
        if df[column].nunique(dropna=True) < 2:
            continue
        columns.append(name)
    return columns


def _finite(series: pd.Series) -> pd.Series:
    """Numeric series with NaN and infinities removed."""
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


# --------------------------------------------------------------------------- #
# Univariate
# --------------------------------------------------------------------------- #


def summarise_numeric(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    registry: ColumnRegistry | None = None,
) -> list[NumericSummary]:
    """Compute descriptive statistics for numeric columns.

    Reports mean and median together, deliberately. Financial columns here are
    strongly right-skewed (the verified statement balance averages NT$51,223
    against a median of NT$22,382), so quoting only the mean would overstate the
    typical client.

    Args:
        df: Dataframe to analyse.
        columns: Columns to summarise. Defaults to all analysable numeric columns.
        registry: Optional pre-built registry.

    Returns:
        One :class:`NumericSummary` per column, skipping any with no finite values.
    """
    targets = (
        [c for c in columns if c in df.columns]
        if columns is not None
        else _analysable_numeric_columns(df, registry)
    )

    summaries: list[NumericSummary] = []
    for column in targets:
        raw = pd.to_numeric(df[column], errors="coerce")
        values = _finite(raw)
        if values.empty:
            continue
        summaries.append(
            NumericSummary(
                column=str(column),
                count=int(values.size),
                missing=int(raw.isna().sum()),
                mean=float(values.mean()),
                median=float(values.median()),
                std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
                minimum=float(values.min()),
                q1=float(values.quantile(0.25)),
                q3=float(values.quantile(0.75)),
                maximum=float(values.max()),
                skewness=float(values.skew()) if values.size > 2 else 0.0,
                kurtosis=float(values.kurtosis()) if values.size > 3 else 0.0,
            )
        )

    logger.info("Summarised %d numeric column(s).", len(summaries))
    return summaries


def summary_frame(summaries: Sequence[NumericSummary]) -> pd.DataFrame:
    """Turn summaries into a display-ready dataframe.

    Args:
        summaries: Summaries to tabulate.

    Returns:
        One row per column. Empty with the right columns when nothing is passed.
    """
    if not summaries:
        return pd.DataFrame(
            columns=[
                "column", "count", "missing", "mean", "median", "std",
                "min", "q1", "q3", "max", "iqr", "skewness", "skew_description",
            ]
        )
    return pd.DataFrame([s.to_dict() for s in summaries])


def frequency_table(
    df: pd.DataFrame, column: str, top_n: int | None = None, dropna: bool = False
) -> FrequencyTable | InsufficientData:
    """Build a frequency distribution for a categorical column.

    Args:
        df: Dataframe to analyse.
        column: Column to tabulate.
        top_n: Keep only the most frequent ``top_n`` categories.
        dropna: Exclude missing values from the counts.

    Returns:
        A :class:`FrequencyTable`, or :class:`InsufficientData` when the frame is
        empty or the column is absent.
    """
    if column not in df.columns:
        return InsufficientData(
            analysis=f"Frequency analysis of '{column}'",
            n_rows=len(df),
            required_rows=1,
            reason=f"Column '{column}' is not present in the data.",
        )
    if df.empty:
        return InsufficientData(
            analysis=f"Frequency analysis of '{column}'", n_rows=0, required_rows=1
        )

    series = df[column]
    counts = series.value_counts(dropna=dropna, sort=True)
    if top_n is not None:
        counts = counts.head(top_n)

    total = int(counts.sum())
    counts_dict = {str(key): int(value) for key, value in counts.items()}
    percentages = {
        key: (value / total * 100.0 if total else 0.0) for key, value in counts_dict.items()
    }

    return FrequencyTable(
        column=str(column),
        counts=counts_dict,
        percentages=percentages,
        n_categories=int(series.nunique(dropna=True)),
        n_rows=int(len(df)),
        missing=int(series.isna().sum()),
    )


def distribution_bins(
    df: pd.DataFrame, column: str, bins: int = 30
) -> pd.DataFrame | InsufficientData:
    """Bin a numeric column into a histogram-ready table.

    Returning binned counts rather than raw values keeps the payload small and
    makes the numbers testable, unlike a chart object.

    Args:
        df: Dataframe to analyse.
        column: Numeric column to bin.
        bins: Number of equal-width bins.

    Returns:
        Columns ``bin_left``, ``bin_right``, ``bin_center``, ``count``,
        ``percentage``; or :class:`InsufficientData`.
    """
    if column not in df.columns:
        return InsufficientData(
            analysis=f"Distribution of '{column}'",
            n_rows=len(df),
            required_rows=2,
            reason=f"Column '{column}' is not present.",
        )

    values = _finite(df[column])
    if values.size < 2:
        return InsufficientData(
            analysis=f"Distribution of '{column}'",
            n_rows=int(values.size),
            required_rows=2,
            reason="At least two finite values are needed to form bins.",
        )
    if values.nunique() == 1:
        return InsufficientData(
            analysis=f"Distribution of '{column}'",
            n_rows=int(values.size),
            required_rows=2,
            reason="All values are identical, so there is no distribution to show.",
        )

    counts, edges = np.histogram(values.to_numpy(), bins=bins)
    total = counts.sum()
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": (edges[:-1] + edges[1:]) / 2,
            "count": counts,
            "percentage": counts / total * 100.0 if total else np.zeros_like(counts, dtype=float),
        }
    )


# --------------------------------------------------------------------------- #
# Bivariate
# --------------------------------------------------------------------------- #


def correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    method: CorrelationMethod = "spearman",
    registry: ColumnRegistry | None = None,
    min_rows: int | None = None,
) -> CorrelationResult | InsufficientData:
    """Compute a correlation matrix and rank its pairs.

    Defaults to **Spearman**, not Pearson. Pearson measures linear association
    and is distorted by the heavy right skew in every monetary column here;
    Spearman ranks values first, so it captures monotonic relationships without
    being dragged around by a handful of very large balances. Pearson remains
    available for comparison.

    Args:
        df: Dataframe to analyse.
        columns: Columns to include. Defaults to all analysable numeric columns.
        method: ``spearman``, ``pearson`` or ``kendall``.
        registry: Optional pre-built registry.
        min_rows: Minimum rows required.

    Returns:
        A :class:`CorrelationResult`, or :class:`InsufficientData`.
    """
    threshold = min_rows if min_rows is not None else settings.min_rows_for_analysis

    targets = (
        [c for c in columns if c in df.columns]
        if columns is not None
        else _analysable_numeric_columns(df, registry)
    )

    if len(df) < threshold:
        return InsufficientData(
            analysis=f"{method.title()} correlation analysis",
            n_rows=len(df),
            required_rows=threshold,
            reason="Correlations computed on very few rows are unstable and misleading.",
        )
    if len(targets) < 2:
        return InsufficientData(
            analysis=f"{method.title()} correlation analysis",
            n_rows=len(df),
            required_rows=threshold,
            reason=f"At least two numeric columns are needed; found {len(targets)}.",
        )

    numeric = df[targets].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    matrix = numeric.corr(method=method, numeric_only=False)

    pairs: list[CorrelationPair] = []
    column_list = list(matrix.columns)
    for i, left in enumerate(column_list):
        for right in column_list[i + 1 :]:
            value = matrix.loc[left, right]
            if pd.notna(value):
                pairs.append(
                    CorrelationPair(left=str(left), right=str(right), correlation=float(value))
                )

    logger.info(
        "Computed %s correlation across %d column(s), %d pair(s).",
        method,
        len(column_list),
        len(pairs),
    )
    return CorrelationResult(
        method=method,
        matrix=matrix,
        pairs=tuple(pairs),
        n_rows_used=int(numeric.dropna().shape[0]),
        note=(
            "Spearman rank correlation is the default because the monetary columns "
            "are strongly right-skewed, which distorts Pearson. Correlation "
            "indicates association only, never causation."
            if method == "spearman"
            else "Correlation indicates association only, never causation."
        ),
    )


def compare_groups(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    min_group_size: int = 5,
) -> GroupComparison | InsufficientData:
    """Compare a numeric measure across the levels of a categorical column.

    Args:
        df: Dataframe to analyse.
        group_column: Categorical column defining the groups.
        value_column: Numeric column to compare.
        min_group_size: Groups smaller than this are dropped, since their means
            are too noisy to report.

    Returns:
        A :class:`GroupComparison`, or :class:`InsufficientData`.
    """
    missing = [c for c in (group_column, value_column) if c not in df.columns]
    if missing:
        return InsufficientData(
            analysis=f"Group comparison of '{value_column}' by '{group_column}'",
            n_rows=len(df),
            required_rows=min_group_size,
            reason=f"Missing column(s): {', '.join(missing)}.",
        )

    working = df[[group_column, value_column]].copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    working = working.dropna()

    if working.empty:
        return InsufficientData(
            analysis=f"Group comparison of '{value_column}' by '{group_column}'",
            n_rows=0,
            required_rows=min_group_size,
            reason="No rows remain once missing values are excluded.",
        )

    grouped = working.groupby(group_column, observed=True)[value_column].agg(
        count="count", mean="mean", median="median", std="std", minimum="min", maximum="max"
    )
    grouped = grouped[grouped["count"] >= min_group_size]

    if grouped.empty:
        return InsufficientData(
            analysis=f"Group comparison of '{value_column}' by '{group_column}'",
            n_rows=int(len(working)),
            required_rows=min_group_size,
            reason=f"No group reaches the minimum size of {min_group_size}.",
        )

    return GroupComparison(
        group_column=str(group_column),
        value_column=str(value_column),
        table=grouped.sort_values("mean", ascending=False),
        overall_mean=float(working[value_column].mean()),
        overall_median=float(working[value_column].median()),
    )


def category_outcome_rates(
    df: pd.DataFrame,
    category_column: str,
    target_column: str,
    min_group_size: int = 30,
) -> CategoryOutcome | InsufficientData:
    """Compute a binary outcome rate for each level of a categorical column.

    The core "which groups differ?" analysis. Groups below ``min_group_size``
    stay in the table but are excluded from the highest/lowest comparison, so a
    12-client category cannot be announced as the riskiest segment.

    Args:
        df: Dataframe to analyse.
        category_column: Categorical column defining the groups.
        target_column: Binary 0/1 outcome column.
        min_group_size: Minimum group size to qualify for ranking.

    Returns:
        A :class:`CategoryOutcome`, or :class:`InsufficientData`.
    """
    missing = [c for c in (category_column, target_column) if c not in df.columns]
    if missing:
        return InsufficientData(
            analysis=f"Outcome rate of '{target_column}' by '{category_column}'",
            n_rows=len(df),
            required_rows=min_group_size,
            reason=f"Missing column(s): {', '.join(missing)}.",
        )

    working = df[[category_column, target_column]].copy()
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    working = working.dropna()

    if working.empty:
        return InsufficientData(
            analysis=f"Outcome rate of '{target_column}' by '{category_column}'",
            n_rows=0,
            required_rows=min_group_size,
            reason="No rows remain once missing values are excluded.",
        )

    grouped = working.groupby(category_column, observed=True)[target_column].agg(
        count="count", events="sum"
    )
    grouped["rate_pct"] = grouped["events"] / grouped["count"] * 100.0
    overall = float(working[target_column].mean() * 100.0)
    grouped["lift_vs_overall_pp"] = grouped["rate_pct"] - overall
    grouped["meets_min_size"] = grouped["count"] >= min_group_size

    return CategoryOutcome(
        category_column=str(category_column),
        target_column=str(target_column),
        table=grouped.sort_values("rate_pct", ascending=False),
        overall_rate=overall,
        min_group_size=min_group_size,
    )


def numeric_outcome_comparison(
    df: pd.DataFrame,
    numeric_column: str,
    target_column: str,
    min_rows: int | None = None,
) -> dict[str, Any] | InsufficientData:
    """Compare a numeric column between the two outcome classes.

    Reports Cohen's d alongside the group means, because with 30,000 rows almost
    any difference reaches statistical significance; the effect size is what says
    whether it matters in practice.

    Args:
        df: Dataframe to analyse.
        numeric_column: Numeric column to compare.
        target_column: Binary 0/1 outcome column.
        min_rows: Minimum rows required.

    Returns:
        Summary statistics per class plus the difference and effect size, or
        :class:`InsufficientData`.
    """
    threshold = min_rows if min_rows is not None else settings.min_rows_for_analysis
    missing = [c for c in (numeric_column, target_column) if c not in df.columns]
    if missing:
        return InsufficientData(
            analysis=f"Comparison of '{numeric_column}' by outcome",
            n_rows=len(df),
            required_rows=threshold,
            reason=f"Missing column(s): {', '.join(missing)}.",
        )

    working = df[[numeric_column, target_column]].copy()
    working[numeric_column] = pd.to_numeric(working[numeric_column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    working = working.dropna()

    if len(working) < threshold:
        return InsufficientData(
            analysis=f"Comparison of '{numeric_column}' by outcome",
            n_rows=int(len(working)),
            required_rows=threshold,
        )

    positive = working.loc[working[target_column] == 1, numeric_column]
    negative = working.loc[working[target_column] == 0, numeric_column]

    if positive.empty or negative.empty:
        return InsufficientData(
            analysis=f"Comparison of '{numeric_column}' by outcome",
            n_rows=int(len(working)),
            required_rows=threshold,
            reason="Both outcome classes must be present to compare them.",
        )

    mean_positive = float(positive.mean())
    mean_negative = float(negative.mean())

    # Pooled standard deviation for Cohen's d.
    n1, n2 = len(positive), len(negative)
    var1 = float(positive.var(ddof=1)) if n1 > 1 else 0.0
    var2 = float(negative.var(ddof=1)) if n2 > 1 else 0.0
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if n1 + n2 > 2 else 0.0
    cohens_d = (mean_positive - mean_negative) / pooled if pooled > 0 else None

    return {
        "column": numeric_column,
        "target": target_column,
        "n_positive": n1,
        "n_negative": n2,
        "mean_positive": mean_positive,
        "mean_negative": mean_negative,
        "median_positive": float(positive.median()),
        "median_negative": float(negative.median()),
        "difference": mean_positive - mean_negative,
        "relative_difference_pct": (
            (mean_positive - mean_negative) / abs(mean_negative) * 100.0
            if mean_negative != 0
            else None
        ),
        "cohens_d": cohens_d,
        "effect_size_label": _effect_size_label(cohens_d),
        "interpretation_note": (
            "Difference in group means. Cohen's d expresses the gap in pooled "
            "standard deviations, which matters because with a large sample even a "
            "trivial difference becomes statistically significant. This is an "
            "observed association, not evidence of causation."
        ),
    }


def _effect_size_label(cohens_d: float | None) -> str:
    """Conventional descriptive label for a Cohen's d value."""
    if cohens_d is None:
        return "undefined"
    magnitude = abs(cohens_d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Multivariate
# --------------------------------------------------------------------------- #


def cross_tabulate(
    df: pd.DataFrame,
    row_column: str,
    column_column: str,
    value_column: str | None = None,
    aggfunc: str = "mean",
) -> pd.DataFrame | InsufficientData:
    """Build a two-way table, optionally aggregating a third column.

    Used for the risk-by-two-dimensions views, for example default rate by
    credit-limit band crossed with age band.

    Args:
        df: Dataframe to analyse.
        row_column: Column for table rows.
        column_column: Column for table columns.
        value_column: Numeric column to aggregate. When omitted, counts are used.
        aggfunc: Aggregation applied to ``value_column``.

    Returns:
        The pivot table, or :class:`InsufficientData`.
    """
    needed = [row_column, column_column] + ([value_column] if value_column else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return InsufficientData(
            analysis=f"Cross-tabulation of '{row_column}' by '{column_column}'",
            n_rows=len(df),
            required_rows=1,
            reason=f"Missing column(s): {', '.join(missing)}.",
        )
    if df.empty:
        return InsufficientData(
            analysis=f"Cross-tabulation of '{row_column}' by '{column_column}'",
            n_rows=0,
            required_rows=1,
        )

    if value_column is None:
        return pd.crosstab(df[row_column], df[column_column])

    working = df[needed].copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    return pd.pivot_table(
        working,
        index=row_column,
        columns=column_column,
        values=value_column,
        aggfunc=aggfunc,
        observed=True,
    )


def segment_profile(
    df: pd.DataFrame,
    segment_column: str,
    metric_columns: Sequence[str],
    target_column: str | None = None,
) -> pd.DataFrame | InsufficientData:
    """Profile segments by averaging a set of metrics within each one.

    Generic enough to serve both the band-based views in this phase and the
    clustering-based segments added later.

    Args:
        df: Dataframe to analyse.
        segment_column: Column identifying the segment.
        metric_columns: Numeric columns to average per segment.
        target_column: Optional binary outcome; its rate is added per segment.

    Returns:
        One row per segment, or :class:`InsufficientData`.
    """
    if segment_column not in df.columns:
        return InsufficientData(
            analysis=f"Segment profile by '{segment_column}'",
            n_rows=len(df),
            required_rows=1,
            reason=f"Column '{segment_column}' is not present.",
        )

    available = [c for c in metric_columns if c in df.columns]
    if not available:
        return InsufficientData(
            analysis=f"Segment profile by '{segment_column}'",
            n_rows=len(df),
            required_rows=1,
            reason="None of the requested metric columns are present.",
        )
    if df.empty:
        return InsufficientData(
            analysis=f"Segment profile by '{segment_column}'", n_rows=0, required_rows=1
        )

    working = df[[segment_column, *available] + ([target_column] if target_column and target_column in df.columns else [])].copy()
    for column in available:
        working[column] = pd.to_numeric(working[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    grouped = working.groupby(segment_column, observed=True)
    profile = grouped[available].mean()
    profile.insert(0, "n_clients", grouped.size())
    profile["share_of_portfolio_pct"] = profile["n_clients"] / len(working) * 100.0

    if target_column and target_column in working.columns:
        rates = grouped[target_column].mean() * 100.0
        profile[f"{target_column}_rate_pct"] = rates

    return profile


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #


def outlier_analysis(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    iqr_multiplier: float | None = None,
    registry: ColumnRegistry | None = None,
) -> pd.DataFrame | InsufficientData:
    """Quantify outliers per numeric column for analyst review.

    Complements :func:`src.data_quality.detect_outliers`: that function scores
    data quality, this one supports interpretation by reporting how extreme the
    tails are and how much of the total they carry.

    Args:
        df: Dataframe to analyse.
        columns: Columns to examine. Defaults to all analysable numeric columns.
        iqr_multiplier: Fence multiplier. Defaults to the configured value.
        registry: Optional pre-built registry.

    Returns:
        One row per column with bounds, counts and tail share, or
        :class:`InsufficientData`.
    """
    multiplier = iqr_multiplier if iqr_multiplier is not None else settings.outlier_iqr_multiplier
    targets = (
        [c for c in columns if c in df.columns]
        if columns is not None
        else _analysable_numeric_columns(df, registry)
    )

    if df.empty or not targets:
        return InsufficientData(
            analysis="Outlier analysis",
            n_rows=len(df),
            required_rows=4,
            reason="Requires a non-empty frame with at least one numeric column.",
        )

    records: list[dict[str, Any]] = []
    for column in targets:
        values = _finite(df[column])
        if values.size < 4:
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1

        if iqr > 0:
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            method = f"IQR ({multiplier}x)"
        else:
            # A degenerate quartile spread: most values are identical, so the IQR
            # fence collapses to a point and would flag nothing. Without a
            # fallback a single extreme value among 50 identical ones would be
            # missed entirely. MAD is the robust alternative for this case.
            median = float(values.median())
            mad = float((values - median).abs().median())
            if mad > 0:
                # 1.4826 scales MAD to be comparable with a standard deviation
                # under normality, so the multiplier keeps a familiar meaning.
                spread = 1.4826 * mad * multiplier
                lower, upper = median - spread, median + spread
                method = f"MAD ({multiplier}x, IQR was zero)"
            elif values.nunique() > 1:
                # Even MAD is zero, so the majority share one value. Anything
                # differing from that value is the outlier.
                lower = upper = median
                method = "deviation from the modal value (IQR and MAD both zero)"
            else:
                # Genuinely constant: nothing to report.
                continue

        low_mask = values < lower
        high_mask = values > upper
        n_out = int(low_mask.sum() + high_mask.sum())
        total_abs = float(values.abs().sum())

        records.append(
            {
                "column": column,
                "n_values": int(values.size),
                "method": method,
                "lower_bound": lower,
                "upper_bound": upper,
                "n_below": int(low_mask.sum()),
                "n_above": int(high_mask.sum()),
                "n_outliers": n_out,
                "outlier_pct": n_out / values.size * 100.0,
                "max_value": float(values.max()),
                "share_of_total_in_tail_pct": (
                    float(values[high_mask].abs().sum()) / total_abs * 100.0
                    if total_abs > 0
                    else 0.0
                ),
                "skewness": float(values.skew()) if values.size > 2 else 0.0,
            }
        )

    if not records:
        return InsufficientData(
            analysis="Outlier analysis",
            n_rows=len(df),
            required_rows=4,
            reason="No column had enough distinct finite values to assess.",
        )

    frame = pd.DataFrame(records).sort_values("outlier_pct", ascending=False)
    logger.info("Outlier analysis covered %d column(s).", len(frame))
    return frame


# --------------------------------------------------------------------------- #
# Convenience bundle
# --------------------------------------------------------------------------- #


def run_eda(
    df: pd.DataFrame,
    registry: ColumnRegistry | None = None,
    correlation_method: CorrelationMethod = "spearman",
) -> dict[str, Any]:
    """Run the standard EDA bundle in one call.

    Args:
        df: Dataframe carrying cleaned data and engineered features.
        registry: Optional pre-built registry.
        correlation_method: Correlation method to use.

    Returns:
        A dictionary of the individual analysis results.
    """
    reg = registry or build_registry(df)
    logger.info("Running EDA bundle on %d rows.", len(df))

    numeric_summaries = summarise_numeric(df, registry=reg)
    correlations = correlation_matrix(df, method=correlation_method, registry=reg)
    outliers = outlier_analysis(df, registry=reg)

    categorical_targets = [
        column
        for column in ("sex", "education", "marriage", "limit_band", "age_band", "delinquency_trend")
        if column in df.columns
    ]
    frequencies = {column: frequency_table(df, column) for column in categorical_targets}

    target_column = reg.one(Role.TARGET)
    category_outcomes: dict[str, Any] = {}
    if target_column:
        for column in categorical_targets:
            category_outcomes[column] = category_outcome_rates(df, column, target_column)

    return {
        "numeric_summaries": numeric_summaries,
        "numeric_summary_frame": summary_frame(numeric_summaries),
        "correlations": correlations,
        "outliers": outliers,
        "frequencies": frequencies,
        "category_outcomes": category_outcomes,
        "n_rows": int(len(df)),
    }
