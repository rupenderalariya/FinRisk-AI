"""
Financial feature engineering.

Every feature is derived from columns that are **verified to exist** at runtime.
If the source columns are absent the feature is skipped and the reason is
recorded in the :class:`FeatureReport`, which is what the data dictionary and
the Methodology page render. Nothing is ever invented to fill a gap.

What this dataset does and does not support
-------------------------------------------
The brief lists debt-to-income, savings rate and expense ratio as example
features. This dataset has **no income, savings or expense column**, so those
are impossible and are deliberately NOT produced. What it does have is a
six-month panel of statement balances, payments and repayment statuses, which
supports a richer set of genuinely behavioural features:

======================  ===========================================
Available here          Formula
======================  ===========================================
Credit utilisation      statement balance / credit limit
Repayment ratio         amount paid / previous month's statement
Delinquency measures    counts and severity from repayment status
Balance trajectory      OLS slope of balance across six months
Credit-limit band       quantile band of credit limit (capacity proxy)
======================  ===========================================

``credit_limit`` is a **credit limit**, never income. It is used as a coarse
proxy for credit capacity and is labelled as such everywhere.

Panel alignment
---------------
Month 1 is the most recent month (September 2005) and month 6 the oldest
(April 2005). A payment recorded in month *m* settles the statement issued in
month *m+1*, so repayment ratios divide ``pay_amt_m<m>`` by
``bill_amt_m<m+1>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import settings
from src.logging_setup import get_logger
from src.schema import (
    CHRONOLOGICAL_MONTHS,
    DELINQUENCY_THRESHOLD,
    ColumnRegistry,
    Role,
    build_registry,
)

logger = get_logger(__name__)

#: Median repayment ratio at or above which a client is treated as a full payer.
FULL_PAYER_RATIO: Final[float] = 0.95
#: Median repayment ratio below which a revolving-credit user is identified.
REVOLVER_RATIO: Final[float] = 0.30

#: Upper clip applied when *averaging* repayment ratios. A prior statement of a
#: few NT$ against a normal payment yields a ratio in the thousands, which is
#: arithmetically correct but makes an average meaningless and would dominate any
#: distance-based model. 2.0 (repaid twice over) is well beyond normal behaviour,
#: so the clip affects only pathological denominators. Uncapped values are kept
#: in ``repayment_ratio_mean``.
REPAYMENT_RATIO_CAP: Final[float] = 2.0

AGE_BAND_EDGES: Final[tuple[int, ...]] = (0, 29, 39, 49, 59, 200)
AGE_BAND_LABELS: Final[tuple[str, ...]] = ("Under 30", "30-39", "40-49", "50-59", "60+")

DELINQUENCY_TREND_LABELS: Final[tuple[str, ...]] = ("Improving", "Stable", "Worsening")

N_LIMIT_BANDS: Final[int] = 4


# --------------------------------------------------------------------------- #
# Reporting structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureDefinition:
    """Documentation for one engineered feature.

    These objects are the single source of truth for the engineered-feature
    section of ``data_dictionary.md``, so the documentation cannot drift from
    the implementation.
    """

    name: str
    formula: str
    source_columns: tuple[str, ...]
    interpretation: str
    category: str
    created: bool = True
    skip_reason: str | None = None
    used_for_analytics: bool = True
    used_for_ml: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the docs generator and UI."""
        return {
            "name": self.name,
            "formula": self.formula,
            "source_columns": list(self.source_columns),
            "interpretation": self.interpretation,
            "category": self.category,
            "created": self.created,
            "skip_reason": self.skip_reason,
            "used_for_analytics": self.used_for_analytics,
            "used_for_ml": self.used_for_ml,
        }


@dataclass(frozen=True)
class FeatureReport:
    """Outcome of a feature-engineering run."""

    definitions: tuple[FeatureDefinition, ...]
    skipped_groups: Mapping[str, str] = field(default_factory=dict)

    @property
    def created(self) -> tuple[FeatureDefinition, ...]:
        """Definitions for features that were successfully created."""
        return tuple(d for d in self.definitions if d.created)

    @property
    def skipped(self) -> tuple[FeatureDefinition, ...]:
        """Definitions for features that could not be created."""
        return tuple(d for d in self.definitions if not d.created)

    @property
    def created_columns(self) -> tuple[str, ...]:
        """Names of the created feature columns."""
        return tuple(d.name for d in self.created)

    def by_category(self) -> dict[str, list[FeatureDefinition]]:
        """Group created definitions by their category, preserving order."""
        grouped: dict[str, list[FeatureDefinition]] = {}
        for definition in self.created:
            grouped.setdefault(definition.category, []).append(definition)
        return grouped

    def ml_feature_names(self) -> tuple[str, ...]:
        """Created features marked as suitable model inputs."""
        return tuple(d.name for d in self.created if d.used_for_ml)

    def summary(self) -> str:
        """One-line human-readable outcome."""
        return (
            f"Created {len(self.created)} feature(s) across {len(self.by_category())} "
            f"category/categories; skipped {len(self.skipped)}."
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the UI and report."""
        return {
            "summary": self.summary(),
            "created_count": len(self.created),
            "skipped_count": len(self.skipped),
            "created_columns": list(self.created_columns),
            "definitions": [d.to_dict() for d in self.definitions],
            "skipped_groups": dict(self.skipped_groups),
        }


# --------------------------------------------------------------------------- #
# Safe arithmetic
# --------------------------------------------------------------------------- #


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series | float,
    invalid_denominators: Sequence[float] = (0.0,),
    require_positive_denominator: bool = False,
) -> pd.Series:
    """Divide two series, returning ``NaN`` instead of ``inf`` or raising.

    Division by zero is the single most common source of silent corruption in
    financial ratios: plain pandas division yields ``inf``, which then poisons
    every mean, correlation and model that touches it. This returns ``NaN``, an
    honest "undefined", which pandas excludes from aggregations by default.

    Args:
        numerator: Series to divide.
        denominator: Series or scalar to divide by.
        invalid_denominators: Denominator values treated as undefined.
        require_positive_denominator: Also treat any negative denominator as
            undefined. Use for ratios where a negative base is meaningless.

    Returns:
        The quotient, with ``NaN`` wherever the denominator was unusable.

    Examples:
        >>> import pandas as pd
        >>> safe_divide(pd.Series([10.0, 5.0]), pd.Series([2.0, 0.0])).tolist()
        [5.0, nan]
    """
    numerator_numeric = pd.to_numeric(numerator, errors="coerce").astype("float64")

    if isinstance(denominator, pd.Series):
        denominator_numeric = pd.to_numeric(denominator, errors="coerce").astype("float64")
    else:
        denominator_numeric = pd.Series(
            float(denominator), index=numerator_numeric.index, dtype="float64"
        )

    unusable = denominator_numeric.isna()
    for invalid in invalid_denominators:
        unusable = unusable | (denominator_numeric == invalid)
    if require_positive_denominator:
        unusable = unusable | (denominator_numeric <= 0)

    safe_denominator = denominator_numeric.where(~unusable, other=np.nan)

    # Both operands are NaN-safe here, so no divide-by-zero warning is possible.
    result = numerator_numeric / safe_denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _ols_slope(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Least-squares slope for each row of ``values`` against a shared ``x``.

    Closed-form and fully vectorised, so 30,000 six-point regressions cost one
    pass instead of 30,000 Python-level calls.

    Args:
        values: 2-D array, one row per client, one column per period.
        x: 1-D array of period positions shared by every row.

    Returns:
        1-D array of slopes. ``NaN`` where a row has fewer than two finite points.
    """
    finite = np.isfinite(values)
    n_points = finite.sum(axis=1)

    x_matrix = np.broadcast_to(x, values.shape)
    safe_values = np.where(finite, values, 0.0)
    safe_x = np.where(finite, x_matrix, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = safe_x.sum(axis=1) / n_points
        mean_y = safe_values.sum(axis=1) / n_points

        dx = np.where(finite, safe_x - mean_x[:, None], 0.0)
        dy = np.where(finite, safe_values - mean_y[:, None], 0.0)

        covariance = (dx * dy).sum(axis=1)
        variance = (dx * dx).sum(axis=1)
        slope = np.where(variance > 0, covariance / variance, np.nan)

    return np.where(n_points >= 2, slope, np.nan)


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #


def add_utilisation_features(
    df: pd.DataFrame, registry: ColumnRegistry, threshold: float | None = None
) -> tuple[pd.DataFrame, list[FeatureDefinition]]:
    """Add credit-utilisation features.

    Utilisation is the central capacity measure available here and stands in for
    the debt-to-income ratio that this dataset cannot support.

    Args:
        df: Cleaned dataframe.
        registry: Column registry.
        threshold: High-utilisation cut-off. Defaults to the configured value.

    Returns:
        ``(dataframe, definitions)``.
    """
    high_threshold = threshold if threshold is not None else settings.high_utilisation_threshold
    result = df.copy()
    definitions: list[FeatureDefinition] = []

    bills = registry.panel_columns(Role.BILL_AMOUNT)
    limit_column = registry.one(Role.CREDIT_LIMIT)

    if not bills or limit_column is None:
        missing = []
        if not bills:
            missing.append("monthly statement balance columns")
        if limit_column is None:
            missing.append("credit limit column")
        reason = f"Requires {' and '.join(missing)}, which are not present."
        definitions.append(
            FeatureDefinition(
                name="utilisation_*",
                formula="bill_amt_m<m> / credit_limit",
                source_columns=("bill_amt_m1..m6", "credit_limit"),
                interpretation="Share of the credit limit in use.",
                category="Credit utilisation",
                created=False,
                skip_reason=reason,
            )
        )
        logger.warning("Skipped utilisation features: %s", reason)
        return result, definitions

    limits = pd.to_numeric(result[limit_column], errors="coerce")
    monthly_columns: list[str] = []

    for month, bill_column in bills.items():
        name = f"utilisation_m{month}"
        # A non-positive limit cannot be a denominator -> NaN, never inf.
        result[name] = safe_divide(
            result[bill_column], limits, require_positive_denominator=True
        )
        monthly_columns.append(name)
        definitions.append(
            FeatureDefinition(
                name=name,
                formula=f"{bill_column} / {limit_column}",
                source_columns=(bill_column, limit_column),
                interpretation=(
                    f"Proportion of the credit limit used in month {month}. "
                    "Values above 1.0 mean the account is over its limit; negative "
                    "values mean the account is in credit."
                ),
                category="Credit utilisation",
                used_for_ml=False,  # summaries below carry this signal more compactly
            )
        )

    utilisation_frame = result[monthly_columns]

    latest_month = min(bills)
    result["utilisation_latest"] = result[f"utilisation_m{latest_month}"]
    definitions.append(
        FeatureDefinition(
            name="utilisation_latest",
            formula=f"bill_amt_m{latest_month} / {limit_column}",
            source_columns=(bills[latest_month], limit_column),
            interpretation=(
                "Utilisation in the most recent month. The primary point-in-time "
                "measure of how much of the available credit is committed."
            ),
            category="Credit utilisation",
        )
    )

    result["utilisation_mean_6m"] = utilisation_frame.mean(axis=1, skipna=True)
    definitions.append(
        FeatureDefinition(
            name="utilisation_mean_6m",
            formula="mean(utilisation_m1..m6)",
            source_columns=tuple(monthly_columns),
            interpretation=(
                "Average utilisation across the observed months. Smooths one-off "
                "spikes and describes sustained reliance on credit."
            ),
            category="Credit utilisation",
        )
    )

    result["utilisation_max_6m"] = utilisation_frame.max(axis=1, skipna=True)
    definitions.append(
        FeatureDefinition(
            name="utilisation_max_6m",
            formula="max(utilisation_m1..m6)",
            source_columns=tuple(monthly_columns),
            interpretation="Peak utilisation observed, i.e. the worst-case credit stretch.",
            category="Credit utilisation",
        )
    )

    result["utilisation_volatility"] = utilisation_frame.std(axis=1, ddof=0, skipna=True)
    definitions.append(
        FeatureDefinition(
            name="utilisation_volatility",
            formula="std(utilisation_m1..m6)",
            source_columns=tuple(monthly_columns),
            interpretation=(
                "Variability of utilisation. High values indicate erratic borrowing "
                "rather than a steady balance."
            ),
            category="Credit utilisation",
        )
    )

    result["is_high_utilisation"] = (
        result["utilisation_latest"] > high_threshold
    ).fillna(False)
    definitions.append(
        FeatureDefinition(
            name="is_high_utilisation",
            formula=f"utilisation_latest > {high_threshold}",
            source_columns=("utilisation_latest",),
            interpretation=(
                f"True when the most recent utilisation exceeds {high_threshold:.0%} of "
                "the credit limit. Configurable via HIGH_UTILISATION_THRESHOLD."
            ),
            category="Credit utilisation",
        )
    )

    result["is_over_limit"] = (result["utilisation_max_6m"] > 1.0).fillna(False)
    definitions.append(
        FeatureDefinition(
            name="is_over_limit",
            formula="utilisation_max_6m > 1.0",
            source_columns=("utilisation_max_6m",),
            interpretation=(
                "True when the statement balance exceeded the credit limit in at "
                "least one month."
            ),
            category="Credit utilisation",
        )
    )

    logger.info("Added %d utilisation feature(s).", len(definitions))
    return result, definitions


def add_repayment_features(
    df: pd.DataFrame, registry: ColumnRegistry
) -> tuple[pd.DataFrame, list[FeatureDefinition]]:
    """Add repayment-behaviour features.

    The alignment matters: a payment made in month *m* settles the statement
    issued in month *m+1*. Dividing by the same month's balance would be wrong
    and would badly distort the ratio.

    Args:
        df: Cleaned dataframe.
        registry: Column registry.

    Returns:
        ``(dataframe, definitions)``.
    """
    result = df.copy()
    definitions: list[FeatureDefinition] = []

    bills = registry.panel_columns(Role.BILL_AMOUNT)
    payments = registry.panel_columns(Role.PAYMENT_AMOUNT)

    if not bills or not payments:
        reason = (
            "Requires both monthly statement balance and monthly payment columns; "
            "at least one family is absent."
        )
        definitions.append(
            FeatureDefinition(
                name="repayment_ratio_*",
                formula="pay_amt_m<m> / bill_amt_m<m+1>",
                source_columns=("pay_amt_m1..m6", "bill_amt_m1..m6"),
                interpretation="Share of the previous statement that was repaid.",
                category="Repayment behaviour",
                created=False,
                skip_reason=reason,
            )
        )
        logger.warning("Skipped repayment features: %s", reason)
        return result, definitions

    ratio_columns: list[str] = []
    for month, payment_column in payments.items():
        prior_bill_column = bills.get(month + 1)
        if prior_bill_column is None:
            # The oldest month has no earlier statement to compare against.
            continue
        name = f"repayment_ratio_m{month}"
        result[name] = safe_divide(
            result[payment_column],
            result[prior_bill_column],
            require_positive_denominator=True,
        )
        ratio_columns.append(name)
        definitions.append(
            FeatureDefinition(
                name=name,
                formula=f"{payment_column} / {prior_bill_column}",
                source_columns=(payment_column, prior_bill_column),
                interpretation=(
                    f"Fraction of the month-{month + 1} statement repaid by the "
                    f"month-{month} payment. Undefined (null) when the prior "
                    "statement was zero or negative."
                ),
                category="Repayment behaviour",
                used_for_ml=False,
            )
        )

    if ratio_columns:
        ratio_frame = result[ratio_columns]

        result["repayment_ratio_mean"] = ratio_frame.mean(axis=1, skipna=True)
        definitions.append(
            FeatureDefinition(
                name="repayment_ratio_mean",
                formula=f"mean({', '.join(ratio_columns)})",
                source_columns=tuple(ratio_columns),
                interpretation=(
                    "Average share of each statement repaid. Near 1.0 indicates a "
                    "client who clears the balance; near 0 indicates minimal payment. "
                    "CAVEAT: when a prior statement was very small but positive (say "
                    "NT$5), an ordinary payment produces an enormous ratio, so this "
                    "unbounded mean is heavily right-skewed. Use "
                    "repayment_ratio_median for a typical value and "
                    "repayment_ratio_capped_mean for averaging."
                ),
                category="Repayment behaviour",
                used_for_ml=False,  # unbounded; would dominate any distance metric
            )
        )

        result["repayment_ratio_median"] = ratio_frame.median(axis=1, skipna=True)
        definitions.append(
            FeatureDefinition(
                name="repayment_ratio_median",
                formula=f"median({', '.join(ratio_columns)})",
                source_columns=tuple(ratio_columns),
                interpretation=(
                    "Typical repayment share, resistant to a single unusual month. "
                    "Still unbounded above, so a client with several tiny prior "
                    "statements can score very high."
                ),
                category="Repayment behaviour",
                used_for_ml=False,  # unbounded; capped variant is the model input
            )
        )

        # A bounded companion. Ratios are clipped to [0, 2] before averaging so the
        # measure stays interpretable and is safe to use in distance-based methods
        # such as K-Means, where a single value of 4,444 would dominate everything.
        result["repayment_ratio_capped_mean"] = (
            ratio_frame.clip(lower=0.0, upper=REPAYMENT_RATIO_CAP).mean(axis=1, skipna=True)
        )
        definitions.append(
            FeatureDefinition(
                name="repayment_ratio_capped_mean",
                formula=(
                    f"mean(clip({', '.join(ratio_columns)}, 0, {REPAYMENT_RATIO_CAP}))"
                ),
                source_columns=tuple(ratio_columns),
                interpretation=(
                    f"Average repayment share with each month clipped at "
                    f"{REPAYMENT_RATIO_CAP:.0%} of the statement. Clipping is applied "
                    "because a near-zero prior balance can inflate the raw ratio into "
                    "the thousands, which would distort both the average and any "
                    "distance-based model. A value of 1.0 means the statement was "
                    f"typically cleared; {REPAYMENT_RATIO_CAP:.1f} means it was repaid "
                    "at least twice over. The uncapped values remain available in "
                    "repayment_ratio_mean."
                ),
                category="Repayment behaviour",
            )
        )

        median_ratio = result["repayment_ratio_median"]
        result["is_full_payer"] = (median_ratio >= FULL_PAYER_RATIO).fillna(False)
        definitions.append(
            FeatureDefinition(
                name="is_full_payer",
                formula=f"repayment_ratio_median >= {FULL_PAYER_RATIO}",
                source_columns=("repayment_ratio_median",),
                interpretation=(
                    "Client typically clears the statement in full - a transactor "
                    "rather than a borrower."
                ),
                category="Repayment behaviour",
            )
        )

        result["is_revolver"] = (median_ratio < REVOLVER_RATIO).fillna(False)
        definitions.append(
            FeatureDefinition(
                name="is_revolver",
                formula=f"repayment_ratio_median < {REVOLVER_RATIO}",
                source_columns=("repayment_ratio_median",),
                interpretation=(
                    "Client typically repays less than "
                    f"{REVOLVER_RATIO:.0%} of the statement, carrying a revolving "
                    "balance forward."
                ),
                category="Repayment behaviour",
            )
        )

    payment_columns = list(payments.values())
    payment_frame = result[payment_columns].apply(pd.to_numeric, errors="coerce")

    result["months_zero_payment"] = (payment_frame == 0).sum(axis=1).astype("int64")
    definitions.append(
        FeatureDefinition(
            name="months_zero_payment",
            formula="count(pay_amt_m1..m6 == 0)",
            source_columns=tuple(payment_columns),
            interpretation=(
                "Number of observed months with no payment at all. A direct count of "
                "non-payment events."
            ),
            category="Repayment behaviour",
        )
    )

    result["total_paid_6m"] = payment_frame.sum(axis=1, skipna=True)
    definitions.append(
        FeatureDefinition(
            name="total_paid_6m",
            formula="sum(pay_amt_m1..m6)",
            source_columns=tuple(payment_columns),
            interpretation="Total amount repaid across the observed months (NT$).",
            category="Repayment behaviour",
        )
    )

    result["avg_payment_6m"] = payment_frame.mean(axis=1, skipna=True)
    definitions.append(
        FeatureDefinition(
            name="avg_payment_6m",
            formula="mean(pay_amt_m1..m6)",
            source_columns=tuple(payment_columns),
            interpretation="Average monthly payment (NT$).",
            category="Repayment behaviour",
        )
    )

    logger.info("Added %d repayment feature(s).", len(definitions))
    return result, definitions


def add_delinquency_features(
    df: pd.DataFrame, registry: ColumnRegistry
) -> tuple[pd.DataFrame, list[FeatureDefinition]]:
    """Add delinquency features from the monthly repayment-status columns.

    Only status codes >= 1 are used. Those are the documented "months of payment
    delay" values, so the features avoid depending on the undocumented -2 and 0
    codes entirely.

    Args:
        df: Cleaned dataframe.
        registry: Column registry.

    Returns:
        ``(dataframe, definitions)``.
    """
    result = df.copy()
    definitions: list[FeatureDefinition] = []

    statuses = registry.panel_columns(Role.PAYMENT_STATUS)
    if not statuses:
        reason = "Requires monthly repayment-status columns, which are not present."
        definitions.append(
            FeatureDefinition(
                name="delinquency_*",
                formula="derived from pay_status_m1..m6",
                source_columns=("pay_status_m1..m6",),
                interpretation="Delinquency frequency, severity and direction.",
                category="Delinquency",
                created=False,
                skip_reason=reason,
            )
        )
        logger.warning("Skipped delinquency features: %s", reason)
        return result, definitions

    status_columns = list(statuses.values())
    status_frame = result[status_columns].apply(pd.to_numeric, errors="coerce")
    delinquent_mask = status_frame >= DELINQUENCY_THRESHOLD

    result["delinquent_months_count"] = delinquent_mask.sum(axis=1).astype("int64")
    definitions.append(
        FeatureDefinition(
            name="delinquent_months_count",
            formula=f"count(pay_status_m1..m6 >= {DELINQUENCY_THRESHOLD})",
            source_columns=tuple(status_columns),
            interpretation=(
                "Number of observed months in which the client was behind on payment. "
                "Measures how often, not how badly."
            ),
            category="Delinquency",
        )
    )

    # Clip at 0 so the undocumented -1 / -2 codes cannot read as negative severity.
    result["max_delinquency"] = (
        status_frame.max(axis=1, skipna=True).clip(lower=0).astype("float64")
    )
    definitions.append(
        FeatureDefinition(
            name="max_delinquency",
            formula="max(pay_status_m1..m6), floored at 0",
            source_columns=tuple(status_columns),
            interpretation=(
                "Worst payment delay observed, in months. Measures severity. Negative "
                "status codes (paid in full / no transaction) are floored to 0 so they "
                "do not read as negative severity."
            ),
            category="Delinquency",
        )
    )

    latest_month = min(statuses)
    result["current_delinquency"] = (
        pd.to_numeric(result[statuses[latest_month]], errors="coerce").clip(lower=0)
    )
    definitions.append(
        FeatureDefinition(
            name="current_delinquency",
            formula=f"max({statuses[latest_month]}, 0)",
            source_columns=(statuses[latest_month],),
            interpretation=(
                "Payment delay in the most recent month, in months. The closest "
                "available signal to present-day status."
            ),
            category="Delinquency",
        )
    )

    result["ever_delinquent"] = delinquent_mask.any(axis=1).fillna(False)
    definitions.append(
        FeatureDefinition(
            name="ever_delinquent",
            formula=f"any(pay_status_m1..m6 >= {DELINQUENCY_THRESHOLD})",
            source_columns=tuple(status_columns),
            interpretation="True when the client was behind on payment in any observed month.",
            category="Delinquency",
        )
    )

    result["is_currently_delinquent"] = (
        result["current_delinquency"] >= DELINQUENCY_THRESHOLD
    ).fillna(False)
    definitions.append(
        FeatureDefinition(
            name="is_currently_delinquent",
            formula=f"current_delinquency >= {DELINQUENCY_THRESHOLD}",
            source_columns=("current_delinquency",),
            interpretation="True when the client was behind on payment in the most recent month.",
            category="Delinquency",
        )
    )

    # Direction of travel: recent half versus older half.
    recent_months = [m for m in sorted(statuses) if m <= 3]
    older_months = [m for m in sorted(statuses) if m > 3]
    if recent_months and older_months:
        recent_count = delinquent_mask[[statuses[m] for m in recent_months]].sum(axis=1)
        older_count = delinquent_mask[[statuses[m] for m in older_months]].sum(axis=1)
        difference = recent_count - older_count

        trend = pd.Series("Stable", index=result.index, dtype="object")
        trend = trend.where(difference >= 0, "Improving")
        trend = trend.where(difference <= 0, "Worsening")
        result["delinquency_trend"] = pd.Categorical(
            trend, categories=list(DELINQUENCY_TREND_LABELS), ordered=True
        )
        definitions.append(
            FeatureDefinition(
                name="delinquency_trend",
                formula=(
                    "compare count of delinquent months in m1-m3 (recent) against "
                    "m4-m6 (older): more recent = Worsening, fewer = Improving, "
                    "equal = Stable"
                ),
                source_columns=tuple(status_columns),
                interpretation=(
                    "Direction of delinquency over the observed window. 'Worsening' "
                    "means the client was late more often in the recent three months "
                    "than in the earlier three."
                ),
                category="Delinquency",
                used_for_ml=False,  # categorical duplicate of the underlying counts
            )
        )

    logger.info("Added %d delinquency feature(s).", len(definitions))
    return result, definitions


def add_balance_trajectory_features(
    df: pd.DataFrame, registry: ColumnRegistry
) -> tuple[pd.DataFrame, list[FeatureDefinition]]:
    """Add balance-trajectory features across the panel.

    These exist only because the dataset is a panel: a single-snapshot dataset
    could not support any of them.

    Args:
        df: Cleaned dataframe.
        registry: Column registry.

    Returns:
        ``(dataframe, definitions)``.
    """
    result = df.copy()
    definitions: list[FeatureDefinition] = []

    bills = registry.panel_columns(Role.BILL_AMOUNT)
    if not bills:
        reason = "Requires monthly statement balance columns, which are not present."
        definitions.append(
            FeatureDefinition(
                name="bill_trend_slope",
                formula="OLS slope of statement balance over months",
                source_columns=("bill_amt_m1..m6",),
                interpretation="Direction and pace of balance change.",
                category="Balance trajectory",
                created=False,
                skip_reason=reason,
            )
        )
        logger.warning("Skipped balance trajectory features: %s", reason)
        return result, definitions

    bill_columns = list(bills.values())
    bill_frame = result[bill_columns].apply(pd.to_numeric, errors="coerce")

    result["avg_bill_6m"] = bill_frame.mean(axis=1, skipna=True)
    definitions.append(
        FeatureDefinition(
            name="avg_bill_6m",
            formula="mean(bill_amt_m1..m6)",
            source_columns=tuple(bill_columns),
            interpretation="Average statement balance across the observed months (NT$).",
            category="Balance trajectory",
        )
    )

    result["bill_volatility"] = bill_frame.std(axis=1, ddof=0, skipna=True)
    definitions.append(
        FeatureDefinition(
            name="bill_volatility",
            formula="std(bill_amt_m1..m6)",
            source_columns=tuple(bill_columns),
            interpretation=(
                "Variability of the statement balance. High values indicate erratic "
                "spending rather than a steady carried balance."
            ),
            category="Balance trajectory",
        )
    )

    # Chronological order so a positive slope genuinely means "growing over time".
    ordered_months = [m for m in CHRONOLOGICAL_MONTHS if m in bills]
    if len(ordered_months) >= 2:
        ordered_columns = [bills[m] for m in ordered_months]
        values = bill_frame[ordered_columns].to_numpy(dtype="float64")
        x = np.arange(1, len(ordered_months) + 1, dtype="float64")
        result["bill_trend_slope"] = _ols_slope(values, x)
        definitions.append(
            FeatureDefinition(
                name="bill_trend_slope",
                formula=(
                    "least-squares slope of statement balance against month position, "
                    f"ordered oldest to newest ({' -> '.join(ordered_columns)})"
                ),
                source_columns=tuple(ordered_columns),
                interpretation=(
                    "Average change in statement balance per month (NT$/month). "
                    "Positive means the balance is growing over time; negative means "
                    "it is being paid down."
                ),
                category="Balance trajectory",
            )
        )

        oldest_column = bills[max(bills)]
        newest_column = bills[min(bills)]
        oldest = pd.to_numeric(result[oldest_column], errors="coerce")
        newest = pd.to_numeric(result[newest_column], errors="coerce")
        # Divide by the absolute base so the sign reflects direction, not the
        # sign of the starting balance.
        result["balance_growth_6m"] = safe_divide(newest - oldest, oldest.abs())
        definitions.append(
            FeatureDefinition(
                name="balance_growth_6m",
                formula=f"({newest_column} - {oldest_column}) / abs({oldest_column})",
                source_columns=(newest_column, oldest_column),
                interpretation=(
                    "Proportional change in balance from the oldest to the most recent "
                    "month. Undefined (null) when the starting balance was zero. "
                    "CAVEAT: a very small starting balance produces an extremely large "
                    "growth figure, so this measure is unbounded and strongly "
                    "right-skewed. Report its median rather than its mean, and prefer "
                    "bill_trend_slope (denominated in NT$/month) for modelling."
                ),
                category="Balance trajectory",
                used_for_ml=False,  # unbounded ratio; slope carries the same signal safely
            )
        )

    logger.info("Added %d balance trajectory feature(s).", len(definitions))
    return result, definitions


def add_band_features(
    df: pd.DataFrame, registry: ColumnRegistry, n_limit_bands: int = N_LIMIT_BANDS
) -> tuple[pd.DataFrame, list[FeatureDefinition]]:
    """Add categorical bands for segment-style reporting.

    ``limit_band`` is a quantile band of the **credit limit**. It stands in for
    the income band the brief suggests, and is named and described as a credit
    limit so it is never mistaken for income.

    Args:
        df: Cleaned dataframe.
        registry: Column registry.
        n_limit_bands: Number of quantile bands for the credit limit.

    Returns:
        ``(dataframe, definitions)``.
    """
    result = df.copy()
    definitions: list[FeatureDefinition] = []

    limit_column = registry.one(Role.CREDIT_LIMIT)
    if limit_column and limit_column in result.columns:
        limits = pd.to_numeric(result[limit_column], errors="coerce")
        try:
            codes, bins = pd.qcut(
                limits, q=n_limit_bands, labels=False, retbins=True, duplicates="drop"
            )
            n_bands = len(bins) - 1
            if n_bands >= 2:
                labels = [
                    f"NT${bins[i] / 1000:,.0f}k-{bins[i + 1] / 1000:,.0f}k"
                    for i in range(n_bands)
                ]
                result["limit_band"] = pd.Categorical.from_codes(
                    codes.fillna(-1).astype("int64"), categories=labels, ordered=True
                )
                definitions.append(
                    FeatureDefinition(
                        name="limit_band",
                        formula=f"quantile band of {limit_column} into {n_bands} groups",
                        source_columns=(limit_column,),
                        interpretation=(
                            "Credit-limit band, used as a proxy for credit capacity. "
                            "This is a CREDIT LIMIT, not income - the dataset contains "
                            "no income column, so no income banding is possible. Bands "
                            "are quantiles, so each holds a similar number of clients."
                        ),
                        category="Bands",
                        used_for_ml=False,  # the continuous credit_limit is the model input
                    )
                )
            else:
                logger.warning(
                    "Credit limit produced fewer than 2 distinct quantile bands; "
                    "limit_band skipped."
                )
        except (ValueError, TypeError) as exc:
            logger.warning("Could not band the credit limit: %s", exc)
            definitions.append(
                FeatureDefinition(
                    name="limit_band",
                    formula=f"quantile band of {limit_column}",
                    source_columns=(limit_column,),
                    interpretation="Credit-limit band used as a credit-capacity proxy.",
                    category="Bands",
                    created=False,
                    skip_reason=f"Quantile banding failed: {exc}",
                )
            )
    else:
        definitions.append(
            FeatureDefinition(
                name="limit_band",
                formula="quantile band of credit_limit",
                source_columns=("credit_limit",),
                interpretation="Credit-limit band used as a credit-capacity proxy.",
                category="Bands",
                created=False,
                skip_reason="Requires a credit_limit column, which is not present.",
            )
        )

    age_column = registry.one(Role.AGE)
    if age_column and age_column in result.columns:
        ages = pd.to_numeric(result[age_column], errors="coerce")
        result["age_band"] = pd.cut(
            ages,
            bins=list(AGE_BAND_EDGES),
            labels=list(AGE_BAND_LABELS),
            right=True,
            include_lowest=True,
            ordered=True,
        )
        definitions.append(
            FeatureDefinition(
                name="age_band",
                formula=f"fixed bins on {age_column}: {', '.join(AGE_BAND_LABELS)}",
                source_columns=(age_column,),
                interpretation=(
                    "Age group. Fixed bands are used rather than quantiles so the "
                    "groups stay comparable across any filtered subset."
                ),
                category="Bands",
                used_for_ml=False,
            )
        )
    else:
        definitions.append(
            FeatureDefinition(
                name="age_band",
                formula="fixed bins on age",
                source_columns=("age",),
                interpretation="Age group.",
                category="Bands",
                created=False,
                skip_reason="Requires an age column, which is not present.",
            )
        )

    logger.info("Added %d band feature(s).", len([d for d in definitions if d.created]))
    return result, definitions


def unavailable_feature_definitions(registry: ColumnRegistry) -> list[FeatureDefinition]:
    """Document the features the brief suggests that this dataset cannot support.

    Recording these explicitly is the point: it turns a silent gap into a stated
    limitation, so a reader can see that debt-to-income was considered and found
    impossible rather than forgotten.

    Args:
        registry: Column registry.

    Returns:
        Definitions marked ``created=False`` with the reason.
    """
    specs = (
        (
            "debt_to_income_ratio",
            "total debt / income",
            ("income",),
            "Debt burden relative to earnings.",
            Role.INCOME,
        ),
        (
            "savings_rate",
            "savings / income",
            ("savings", "income"),
            "Share of income retained as savings.",
            Role.SAVINGS,
        ),
        (
            "expense_ratio",
            "expenses / income",
            ("expenses", "income"),
            "Share of income consumed by expenses.",
            Role.EXPENSES,
        ),
        (
            "loan_to_income_ratio",
            "loan amount / income",
            ("loan_amount", "income"),
            "Loan size relative to earnings.",
            Role.LOAN_AMOUNT,
        ),
        (
            "income_band",
            "quantile band of income",
            ("income",),
            "Income group.",
            Role.INCOME,
        ),
    )

    definitions: list[FeatureDefinition] = []
    for name, formula, sources, interpretation, role in specs:
        missing = [column for column in sources if not _has_column_for(registry, column)]
        if not missing:
            continue
        definitions.append(
            FeatureDefinition(
                name=name,
                formula=formula,
                source_columns=tuple(sources),
                interpretation=interpretation,
                category="Not available in this dataset",
                created=False,
                skip_reason=(
                    f"Required column(s) {missing} do not exist in this dataset. "
                    "This feature is deliberately not computed rather than "
                    "approximated from unrelated fields."
                ),
                used_for_analytics=False,
                used_for_ml=False,
            )
        )
    return definitions


def _has_column_for(registry: ColumnRegistry, logical_name: str) -> bool:
    """Check whether a logical column name is available in the registry."""
    role_lookup = {
        "income": Role.INCOME,
        "savings": Role.SAVINGS,
        "expenses": Role.EXPENSES,
        "loan_amount": Role.LOAN_AMOUNT,
        "credit_score": Role.CREDIT_SCORE,
    }
    role = role_lookup.get(logical_name)
    if role is not None:
        return registry.has(role)
    return logical_name in registry.columns


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def engineer_features(
    df: pd.DataFrame,
    registry: ColumnRegistry | None = None,
    high_utilisation_threshold: float | None = None,
) -> tuple[pd.DataFrame, FeatureReport]:
    """Run every feature group that the dataset supports.

    Args:
        df: Cleaned dataframe.
        registry: Optional pre-built registry.
        high_utilisation_threshold: Override for the high-utilisation cut-off.

    Returns:
        ``(dataframe_with_features, report)``.
    """
    reg = registry or build_registry(df)
    logger.info("Starting feature engineering on %d rows.", len(df))

    working = df.copy()
    definitions: list[FeatureDefinition] = []

    working, group = add_utilisation_features(working, reg, high_utilisation_threshold)
    definitions.extend(group)

    working, group = add_repayment_features(working, reg)
    definitions.extend(group)

    working, group = add_delinquency_features(working, reg)
    definitions.extend(group)

    working, group = add_balance_trajectory_features(working, reg)
    definitions.extend(group)

    working, group = add_band_features(working, reg)
    definitions.extend(group)

    definitions.extend(unavailable_feature_definitions(reg))

    skipped_groups = {
        definition.name: definition.skip_reason or "unspecified"
        for definition in definitions
        if not definition.created
    }

    report = FeatureReport(definitions=tuple(definitions), skipped_groups=skipped_groups)
    logger.info("Feature engineering complete. %s", report.summary())
    return working, report
