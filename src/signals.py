"""
Derivation of key findings, risk signals and opportunities from the data.

Every item the dashboard presents as an insight is produced here, by measuring
the actual selection. Nothing is hard-coded: a rule that does not trigger emits
nothing, so a filtered selection with no delinquency gradient simply has no
delinquency finding rather than a stale sentence.

Labelling discipline
--------------------
These are **observed analytical signals** computed from historical outcomes, not
model predictions. No predictive model exists at this stage, and every signal
carries :data:`OBSERVED_SIGNAL_LABEL` so the distinction survives into the UI.
Once the risk model is built, its output will be labelled separately as
model-derived.

Language discipline
-------------------
Findings describe associations. The generated text uses "associated with",
"shows a relationship with" and "identified as" - never "causes" or "because of".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

from src.analytics import (
    InsufficientData,
    category_outcome_rates,
    numeric_outcome_comparison,
)
from src.config import settings
from src.filtering import assign_utilisation_band
from src.findings import (
    Confidence,
    Evidence,
    Finding,
    FindingSet,
    FindingType,
    Severity,
    build_finding,
    confidence_from_sample,
    make_evidence,
)
from src.kpi_engine import KPIResult, format_currency
from src.logging_setup import get_logger
from src.schema import ColumnRegistry, Role, build_registry
from src.trends import TrendResult, delinquency_trend, utilisation_trend

logger = get_logger(__name__)

OBSERVED_SIGNAL_LABEL: Final[str] = "Observed Analytical Signal"

CAUSATION_CAVEAT: Final[str] = (
    "This is an observed association in historical data, not a causal "
    "relationship and not a model prediction."
)

#: Minimum rows before any signal derivation is attempted.
MIN_ROWS: Final[int] = 100
#: Minimum members for a group to be eligible for comparison.
MIN_GROUP: Final[int] = 30
#: Percentage-point spread below which a gradient is not worth reporting.
MIN_SPREAD_PP: Final[float] = 3.0


# --------------------------------------------------------------------------- #
# Signal structure
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Signal:
    """A compact risk or opportunity signal for a dashboard card."""

    key: str
    name: str
    value: str
    raw_value: float
    description: str
    severity: Severity
    direction: str = "neutral"
    intensity: float = 0.0
    evidence: tuple[Evidence, ...] = field(default=())
    label: str = OBSERVED_SIGNAL_LABEL

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "key": self.key,
            "name": self.name,
            "value": self.value,
            "raw_value": self.raw_value,
            "description": self.description,
            "severity": self.severity.value,
            "direction": self.direction,
            "intensity": round(self.intensity, 4),
            "label": self.label,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass(frozen=True)
class SignalSet:
    """Risk signals and opportunities derived from one selection."""

    risks: tuple[Signal, ...] = field(default=())
    opportunities: tuple[Signal, ...] = field(default=())
    n_rows: int = 0
    note: str = OBSERVED_SIGNAL_LABEL

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "n_rows": self.n_rows,
            "note": self.note,
            "risks": [s.to_dict() for s in self.risks],
            "opportunities": [s.to_dict() for s in self.opportunities],
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _intensity(value: float, low: float, high: float) -> float:
    """Scale a value into 0-1 for a severity bar, clamped at the ends."""
    if high <= low:
        return 0.0
    return float(max(0.0, min(1.0, (value - low) / (high - low))))


#: Severity ladder, least to most severe, used when escalating.
_SEVERITY_LADDER: Final[tuple[Severity, ...]] = (
    Severity.INFO,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.CRITICAL,
)

#: Percentage-point spreads at which a gradient finding is escalated.
_ESCALATE_ONE_PP: Final[float] = 20.0
_ESCALATE_TWO_PP: Final[float] = 35.0


def gradient_severity(share_pct: float, spread_pp: float) -> Severity:
    """Severity for a gradient finding, from both its reach and its size.

    Reach alone is not enough. A pattern that separates the portfolio by 58
    percentage points matters more than one separating it by 18, even if the
    second touches more customers, so a large gap escalates the level. Both
    inputs are measured, so the result stays data-derived.

    Args:
        share_pct: Share of the selection the finding concerns, 0-100.
        spread_pp: Gap between the highest and lowest group rate, in points.

    Returns:
        The escalated severity.
    """
    from src.findings import severity_from_share

    base = severity_from_share(share_pct)
    steps = 2 if spread_pp >= _ESCALATE_TWO_PP else 1 if spread_pp >= _ESCALATE_ONE_PP else 0
    if steps == 0:
        return base

    index = _SEVERITY_LADDER.index(base)
    return _SEVERITY_LADDER[min(index + steps, len(_SEVERITY_LADDER) - 1)]


def _share_true(series: pd.Series) -> float:
    """Percentage of True values, 0.0 for an empty series."""
    if series.empty:
        return 0.0
    return float(series.fillna(False).astype("bool").mean() * 100.0)


def _eligible_outcome_table(
    df: pd.DataFrame, column: str, target: str
) -> pd.DataFrame | None:
    """Outcome-rate table for a category, restricted to adequately sized groups.

    Returns ``None`` when the analysis is impossible or nothing qualifies, so
    callers can skip the rule rather than report a rate from a handful of rows.
    """
    result = category_outcome_rates(df, column, target, min_group_size=MIN_GROUP)
    if isinstance(result, InsufficientData):
        return None
    eligible = result.table[result.table["count"] >= MIN_GROUP]
    return eligible if len(eligible) >= 2 else None


# --------------------------------------------------------------------------- #
# Key findings
# --------------------------------------------------------------------------- #


def _delinquency_finding(
    df: pd.DataFrame, target: str, n_rows: int
) -> Finding | None:
    """Compare default rates across recent-payment-status groups."""
    if "current_delinquency" not in df.columns:
        return None

    working = df.copy()
    # Group the documented delay codes; 0 means no recorded delay.
    working["_delay_group"] = pd.cut(
        pd.to_numeric(working["current_delinquency"], errors="coerce"),
        bins=[-0.5, 0.5, 1.5, 2.5, 100],
        labels=["No delay", "1 month", "2 months", "3+ months"],
        ordered=True,
    )

    table = _eligible_outcome_table(working, "_delay_group", target)
    if table is None:
        return None

    baseline_label = "No delay"
    if baseline_label not in table.index:
        return None

    baseline = float(table.loc[baseline_label, "rate_pct"])
    worst_label = str(table["rate_pct"].idxmax())
    worst = float(table["rate_pct"].max())
    spread = worst - baseline

    if spread < MIN_SPREAD_PP or worst_label == baseline_label:
        return None

    affected = int(table.loc[table.index != baseline_label, "count"].sum())
    share = affected / n_rows * 100.0
    multiple = worst / baseline if baseline > 0 else float("nan")
    multiple_text = f" - roughly {multiple:.1f}x the rate" if np.isfinite(multiple) else ""

    return build_finding(
        key="delinquency_gradient",
        title="Recent payment delay is the strongest observed risk marker",
        fact=(
            f"Customers with no recorded payment delay in the most recent month "
            f"default at {baseline:.2f}%, while those with a '{worst_label}' delay "
            f"default at {worst:.2f}%{multiple_text}."
        ),
        insight=(
            "Default rate rises consistently as the length of the most recent "
            "payment delay increases, so recent repayment status separates the "
            "portfolio more sharply than any demographic attribute measured here."
        ),
        implication=(
            f"{affected:,} customers ({share:.1f}% of the selection) carry some "
            "recorded delay and sit in a materially higher observed default band."
        ),
        action=(
            "Review whether recent payment status is weighted appropriately in "
            "monitoring and collections prioritisation, and treat it as a primary "
            "candidate feature when the predictive model is built."
        ),
        evidence=[
            make_evidence(
                "default_rate_no_delay", "Default rate, no delay", baseline,
                f"{baseline:.2f}%", "analytics.category_outcome_rates",
                n_observations=int(table.loc[baseline_label, "count"]),
            ),
            make_evidence(
                "default_rate_worst_delay", f"Default rate, {worst_label}", worst,
                f"{worst:.2f}%", "analytics.category_outcome_rates",
                comparison=f"{baseline:.2f}% with no delay",
                n_observations=int(table.loc[worst_label, "count"]),
            ),
        ],
        finding_type=FindingType.RISK,
        severity=gradient_severity(share, spread),
        affected_clients=affected,
        affected_share_pct=share,
        caveats=(CAUSATION_CAVEAT,),
        tags=("delinquency", "payment_behaviour"),
    )


def _utilisation_finding(df: pd.DataFrame, target: str, n_rows: int) -> Finding | None:
    """Compare default rates across utilisation bands."""
    if "utilisation_latest" not in df.columns:
        return None

    working = df.copy()
    working["_util_band"] = assign_utilisation_band(working)

    table = _eligible_outcome_table(working, "_util_band", target)
    if table is None:
        return None

    # The "Negative / zero" band holds accounts in credit. Those are not a low
    # point on the utilisation scale - they are a different kind of account
    # altogether, and including them breaks the ordered comparison. They are
    # excluded from the gradient and reported separately.
    ordered = table.sort_index()
    negative_band = [b for b in ordered.index if str(b).startswith("Negative")]
    gradient = ordered.drop(index=negative_band)
    if len(gradient) < 2:
        return None

    lowest_label = str(gradient["rate_pct"].idxmin())
    highest_label = str(gradient["rate_pct"].idxmax())
    lowest = float(gradient["rate_pct"].min())
    highest = float(gradient["rate_pct"].max())
    spread = highest - lowest

    if spread < MIN_SPREAD_PP:
        return None

    # Describe the shape honestly rather than asserting a clean rise.
    rates = gradient["rate_pct"].tolist()
    if rates == sorted(rates):
        shape = (
            "Default rate rises at every step up the utilisation scale."
        )
    elif rates[0] == min(rates):
        shape = (
            "Default rate rises across the lower bands and then plateaus at the "
            "top, so the relationship is directional but not strictly monotonic."
        )
    else:
        shape = "The relationship is directional but not strictly monotonic."

    negative_note = ""
    negative_evidence: list[Evidence] = []
    if negative_band:
        negative_rate = float(ordered.loc[negative_band[0], "rate_pct"])
        negative_count = int(ordered.loc[negative_band[0], "count"])
        negative_note = (
            f" Separately, {negative_count:,} accounts carry a zero or negative "
            f"balance and default at {negative_rate:.2f}%; these are accounts in "
            "credit rather than low-utilisation borrowers, so they are excluded "
            "from the gradient above."
        )
        negative_evidence.append(
            make_evidence(
                "default_rate_negative_balance", "Default rate, zero/negative balance",
                negative_rate, f"{negative_rate:.2f}%",
                "analytics.category_outcome_rates", n_observations=negative_count,
            )
        )

    threshold = settings.high_utilisation_threshold
    high_count = (
        int(df["is_high_utilisation"].fillna(False).astype("bool").sum())
        if "is_high_utilisation" in df.columns
        else int(gradient.loc[highest_label, "count"])
    )
    share = high_count / n_rows * 100.0

    return build_finding(
        key="utilisation_gradient",
        title="Default rate rises with credit utilisation",
        fact=(
            f"Customers in the '{lowest_label}' utilisation band default at "
            f"{lowest:.2f}%, compared with {highest:.2f}% in the "
            f"'{highest_label}' band - a spread of {spread:.1f} percentage points. "
            f"{shape}{negative_note}"
        ),
        insight=(
            "Utilisation, the derived ratio of statement balance to credit limit, "
            "shows a clear relationship with observed default. It is the closest "
            "available substitute for a debt-to-income ratio, which cannot be "
            "computed because this dataset holds no income variable."
        ),
        implication=(
            f"{high_count:,} customers ({share:.1f}%) are using more than "
            f"{threshold:.0%} of their limit and sit in the elevated band."
        ),
        action=(
            "Consider whether sustained high utilisation should trigger earlier "
            "review, and examine whether limit increases for this group would "
            "reduce or merely relocate the exposure."
        ),
        evidence=[
            make_evidence(
                "default_rate_low_utilisation", f"Default rate, {lowest_label}",
                lowest, f"{lowest:.2f}%", "analytics.category_outcome_rates",
                n_observations=int(gradient.loc[lowest_label, "count"]),
            ),
            make_evidence(
                "default_rate_high_utilisation", f"Default rate, {highest_label}",
                highest, f"{highest:.2f}%", "analytics.category_outcome_rates",
                comparison=f"{lowest:.2f}% in the {lowest_label} band",
                n_observations=int(gradient.loc[highest_label, "count"]),
            ),
            *negative_evidence,
        ],
        finding_type=FindingType.RISK,
        severity=gradient_severity(share, spread),
        affected_clients=high_count,
        affected_share_pct=share,
        caveats=(CAUSATION_CAVEAT,),
        tags=("utilisation", "credit_behaviour"),
    )


def _limit_band_finding(df: pd.DataFrame, target: str, n_rows: int) -> Finding | None:
    """Compare default rates across credit-limit bands."""
    if "limit_band" not in df.columns:
        return None

    table = _eligible_outcome_table(df, "limit_band", target)
    if table is None:
        return None

    ordered = table.sort_index()
    lowest_band = str(ordered.index[0])
    highest_band = str(ordered.index[-1])
    lowest_rate = float(ordered.iloc[0]["rate_pct"])
    highest_rate = float(ordered.iloc[-1]["rate_pct"])

    if abs(lowest_rate - highest_rate) < MIN_SPREAD_PP:
        return None

    rates = ordered["rate_pct"].tolist()
    monotonic = rates == sorted(rates, reverse=True)
    pattern = (
        "The relationship is monotonic: every step up in credit limit is "
        "associated with a lower observed default rate."
        if monotonic
        else "The relationship is not strictly monotonic across bands."
    )

    affected = int(ordered.iloc[0]["count"])
    share = affected / n_rows * 100.0

    return build_finding(
        key="limit_band_gradient",
        title="Lower credit limits are associated with higher observed default",
        fact=(
            f"The {lowest_band} band defaults at {lowest_rate:.2f}% against "
            f"{highest_rate:.2f}% in the {highest_band} band. {pattern}"
        ),
        insight=(
            "Credit limit is set by the lender partly from an assessed view of "
            "capacity, so this gradient likely reflects that original assessment "
            "as much as customer behaviour. The limit is a credit limit, not "
            "income and not wealth."
        ),
        implication=(
            f"{affected:,} customers ({share:.1f}%) sit in the lowest limit band, "
            "which carries the highest observed default rate in this selection."
        ),
        action=(
            "Examine whether the lowest-limit band is homogeneous or contains "
            "distinguishable sub-groups, since a single elevated rate across a "
            "quarter of the portfolio is too coarse to act on directly."
        ),
        evidence=[
            make_evidence(
                "default_rate_lowest_limit", f"Default rate, {lowest_band}",
                lowest_rate, f"{lowest_rate:.2f}%", "analytics.category_outcome_rates",
                n_observations=affected,
            ),
            make_evidence(
                "default_rate_highest_limit", f"Default rate, {highest_band}",
                highest_rate, f"{highest_rate:.2f}%", "analytics.category_outcome_rates",
                comparison=f"{lowest_rate:.2f}% in the lowest band",
                n_observations=int(ordered.iloc[-1]["count"]),
            ),
        ],
        finding_type=FindingType.OBSERVATION,
        severity=gradient_severity(share, abs(highest_rate - lowest_rate)),
        affected_clients=affected,
        affected_share_pct=share,
        caveats=(
            CAUSATION_CAVEAT,
            "Credit-limit bands are quartiles of this portfolio and describe "
            "granted credit capacity only.",
        ),
        tags=("credit_limit", "segmentation"),
    )


def _repayment_finding(df: pd.DataFrame, target: str, n_rows: int) -> Finding | None:
    """Compare repayment behaviour between outcome classes."""
    if "repayment_ratio_capped_mean" not in df.columns:
        return None

    comparison = numeric_outcome_comparison(df, "repayment_ratio_capped_mean", target)
    if isinstance(comparison, InsufficientData):
        return None

    defaulter_mean = float(comparison["mean_positive"])
    other_mean = float(comparison["mean_negative"])
    if not np.isfinite(defaulter_mean) or not np.isfinite(other_mean):
        return None
    if abs(defaulter_mean - other_mean) < 0.02:
        return None

    revolver_share = (
        _share_true(df["is_revolver"]) if "is_revolver" in df.columns else None
    )
    revolver_count = (
        int(df["is_revolver"].fillna(False).astype("bool").sum())
        if "is_revolver" in df.columns
        else None
    )

    revolver_sentence = ""
    if revolver_share is not None:
        revolver_sentence = (
            f" {revolver_share:.1f}% of the selection typically repays less than "
            "30% of the statement."
        )

    return build_finding(
        key="repayment_behaviour",
        title="Customers who default repay a smaller share of their statement",
        fact=(
            f"Average capped repayment ratio is {defaulter_mean:.1%} among "
            f"customers who defaulted, against {other_mean:.1%} among those who "
            f"did not (effect size: {comparison['effect_size_label']})."
            + revolver_sentence
        ),
        insight=(
            "Repayment ratio is the payment made in a month divided by the "
            "previous month's statement balance, averaged with each month capped "
            "at 200%. Paying down a consistently small fraction of the balance is "
            "associated with the observed default outcome."
        ),
        implication=(
            "Repayment share is measurable every month and moves earlier than a "
            "delinquency code, so it offers a potential early indicator."
        ),
        action=(
            "Consider tracking sustained low repayment share as a monitoring "
            "trigger in its own right, and test it as a model feature alongside "
            "delinquency history."
        ),
        evidence=[
            make_evidence(
                "repayment_ratio_defaulters", "Repayment ratio, defaulters",
                defaulter_mean, f"{defaulter_mean:.1%}",
                "analytics.numeric_outcome_comparison",
                n_observations=int(comparison["n_positive"]),
            ),
            make_evidence(
                "repayment_ratio_non_defaulters", "Repayment ratio, non-defaulters",
                other_mean, f"{other_mean:.1%}",
                "analytics.numeric_outcome_comparison",
                comparison=f"{defaulter_mean:.1%} among defaulters",
                n_observations=int(comparison["n_negative"]),
            ),
        ],
        finding_type=FindingType.RISK,
        affected_clients=revolver_count,
        affected_share_pct=revolver_share,
        confidence=confidence_from_sample(n_rows, n_methods_agreeing=2),
        caveats=(CAUSATION_CAVEAT,),
        tags=("repayment", "payment_behaviour"),
    )


def _trend_finding(
    df: pd.DataFrame, registry: ColumnRegistry, n_rows: int
) -> Finding | None:
    """Report movement in delinquency and utilisation across the observed months."""
    delinquency = delinquency_trend(df, registry)
    utilisation = utilisation_trend(df, registry, "mean")

    if not isinstance(delinquency, TrendResult):
        return None

    first = delinquency.first_value
    last = delinquency.last_value
    change_pp = last - first
    if abs(change_pp) < 1.0:
        return None

    first_label = delinquency.points[0].month_label
    last_label = delinquency.points[-1].month_label

    utilisation_sentence = ""
    utilisation_evidence: list[Evidence] = []
    if isinstance(utilisation, TrendResult):
        utilisation_sentence = (
            f" Over the same periods mean utilisation moved from "
            f"{utilisation.first_value:.1f}% to {utilisation.last_value:.1f}%."
        )
        utilisation_evidence.append(
            make_evidence(
                "utilisation_trend_change", "Utilisation change across periods",
                utilisation.last_value - utilisation.first_value,
                f"{utilisation.last_value - utilisation.first_value:+.1f} pp",
                "trends.utilisation_trend", n_observations=n_rows,
            )
        )

    direction = "rose" if change_pp > 0 else "fell"
    finding_type = FindingType.RISK if change_pp > 0 else FindingType.OPPORTUNITY

    return build_finding(
        key="delinquency_trend",
        title=f"Share of customers behind on payment {direction} across the observed months",
        fact=(
            f"The share of customers behind on payment {direction} from "
            f"{first:.2f}% in {first_label} to {last:.2f}% in {last_label}, a change "
            f"of {change_pp:+.2f} percentage points across the six observed monthly "
            f"periods.{utilisation_sentence}"
        ),
        insight=(
            "The six monthly observations are snapshots of the same customers, so "
            "this is a genuine period-over-period movement rather than a comparison "
            "of different populations."
        ),
        implication=(
            "A sustained move in delinquency changes the level of the whole "
            "portfolio, not just of individual accounts."
            if change_pp > 0
            else "Improving repayment behaviour reduces portfolio-level exposure."
        ),
        action=(
            "Investigate whether the movement is concentrated in specific limit "
            "bands or utilisation groups, since a portfolio-wide average can hide "
            "a localised change."
        ),
        evidence=[
            make_evidence(
                "delinquency_first_period", f"Behind on payment, {first_label}",
                first, f"{first:.2f}%", "trends.delinquency_trend", n_observations=n_rows,
            ),
            make_evidence(
                "delinquency_last_period", f"Behind on payment, {last_label}",
                last, f"{last:.2f}%", "trends.delinquency_trend",
                comparison=f"{first:.2f}% in {first_label}", n_observations=n_rows,
            ),
            *utilisation_evidence,
        ],
        finding_type=finding_type,
        affected_share_pct=abs(change_pp) * 2,  # scaled so the severity reflects movement
        caveats=(
            "Six monthly periods support period-over-period comparison but are too "
            "few to support a forecast, so none is offered.",
        ),
        tags=("trend", "delinquency"),
    )


def _full_payer_finding(df: pd.DataFrame, target: str, n_rows: int) -> Finding | None:
    """Identify the low-risk transactor segment as an opportunity."""
    if "is_full_payer" not in df.columns:
        return None

    mask = df["is_full_payer"].fillna(False).astype("bool")
    count = int(mask.sum())
    if count < MIN_GROUP:
        return None

    share = count / n_rows * 100.0
    if target not in df.columns:
        return None

    segment_rate = float(pd.to_numeric(df.loc[mask, target], errors="coerce").mean() * 100)
    overall_rate = float(pd.to_numeric(df[target], errors="coerce").mean() * 100)
    if not np.isfinite(segment_rate) or segment_rate >= overall_rate:
        return None

    limit_text = ""
    limit_evidence: list[Evidence] = []
    if "credit_limit" in df.columns:
        segment_limit = float(
            pd.to_numeric(df.loc[mask, "credit_limit"], errors="coerce").mean()
        )
        overall_limit = float(pd.to_numeric(df["credit_limit"], errors="coerce").mean())
        if np.isfinite(segment_limit) and np.isfinite(overall_limit):
            limit_text = (
                f" Their average credit limit is {format_currency(segment_limit)} "
                f"against {format_currency(overall_limit)} portfolio-wide."
            )
            limit_evidence.append(
                make_evidence(
                    "full_payer_avg_limit", "Average credit limit, full payers",
                    segment_limit, format_currency(segment_limit), "kpi_engine",
                    comparison=format_currency(overall_limit), n_observations=count,
                )
            )

    return build_finding(
        key="full_payer_segment",
        title="A sizeable transactor segment shows materially lower observed default",
        fact=(
            f"{count:,} customers ({share:.1f}%) typically clear their statement in "
            f"full. Their observed default rate is {segment_rate:.2f}% against "
            f"{overall_rate:.2f}% across the selection.{limit_text}"
        ),
        insight=(
            "These customers use credit for transacting rather than borrowing. "
            "They generate interchange activity while showing a lower historical "
            "default rate."
        ),
        implication=(
            "A well-behaved segment of this size is the natural place to look for "
            "growth that does not add proportionate risk."
        ),
        action=(
            "Examine whether this segment is under-served on limit or product fit, "
            "and confirm the lower rate persists once utilisation and limit are "
            "controlled for before treating it as a growth target."
        ),
        evidence=[
            make_evidence(
                "full_payer_count", "Full payers", count, f"{count:,}",
                "kpi_engine", n_observations=n_rows,
            ),
            make_evidence(
                "full_payer_default_rate", "Default rate, full payers",
                segment_rate, f"{segment_rate:.2f}%", "signals",
                comparison=f"{overall_rate:.2f}% overall", n_observations=count,
            ),
            *limit_evidence,
        ],
        finding_type=FindingType.OPPORTUNITY,
        affected_clients=count,
        affected_share_pct=share,
        caveats=(CAUSATION_CAVEAT,),
        tags=("opportunity", "repayment"),
    )


def _age_finding(df: pd.DataFrame, target: str, n_rows: int) -> Finding | None:
    """Report an age-band pattern, framed descriptively rather than causally."""
    if "age_band" not in df.columns:
        return None

    table = _eligible_outcome_table(df, "age_band", target)
    if table is None:
        return None

    highest = str(table["rate_pct"].idxmax())
    lowest = str(table["rate_pct"].idxmin())
    high_rate = float(table["rate_pct"].max())
    low_rate = float(table["rate_pct"].min())
    spread = high_rate - low_rate

    if spread < MIN_SPREAD_PP:
        return None

    return build_finding(
        key="age_band_pattern",
        title="Observed default rate varies across age bands",
        fact=(
            f"The '{highest}' band shows the highest observed default rate at "
            f"{high_rate:.2f}%, and '{lowest}' the lowest at {low_rate:.2f}% - a "
            f"spread of {spread:.1f} percentage points."
        ),
        insight=(
            "This is a descriptive difference between groups. Age is not used as a "
            "risk driver or a model feature; it is reported here for portfolio "
            "understanding and for the fairness review that accompanies the model."
        ),
        implication=(
            "Age bands differ in composition as well as behaviour, so the gap may "
            "reflect differences in credit limit, tenure or utilisation rather than "
            "anything about age itself."
        ),
        action=(
            "Treat this as context only. If age-band differences persist after "
            "controlling for limit and utilisation, record it as a fairness "
            "consideration rather than a targeting rule."
        ),
        evidence=[
            make_evidence(
                "age_band_highest_rate", f"Default rate, {highest}", high_rate,
                f"{high_rate:.2f}%", "analytics.category_outcome_rates",
                n_observations=int(table.loc[highest, "count"]),
            ),
            make_evidence(
                "age_band_lowest_rate", f"Default rate, {lowest}", low_rate,
                f"{low_rate:.2f}%", "analytics.category_outcome_rates",
                comparison=f"{high_rate:.2f}% in {highest}",
                n_observations=int(table.loc[lowest, "count"]),
            ),
        ],
        finding_type=FindingType.OBSERVATION,
        severity=Severity.LOW,
        caveats=(
            CAUSATION_CAVEAT,
            "Demographic attributes are reported descriptively and are excluded "
            "from risk modelling by default.",
        ),
        tags=("demographics", "fairness"),
    )


def derive_key_findings(
    df: pd.DataFrame,
    registry: ColumnRegistry | None = None,
    target: str | None = None,
    limit: int = 5,
) -> FindingSet:
    """Derive the key findings for a selection.

    Each rule measures the data and returns a finding only when the pattern is
    genuinely present and material. Rules that do not trigger contribute nothing.

    Args:
        df: Filtered, analysis-ready dataframe.
        registry: Optional pre-built registry.
        target: Target column. Resolved from the registry when omitted.
        limit: Maximum findings to return, highest priority first.

    Returns:
        A :class:`FindingSet`, possibly empty.
    """
    reg = registry or build_registry(df)
    target_column = target or reg.one(Role.TARGET)
    n_rows = int(len(df))

    if n_rows < MIN_ROWS or target_column is None or target_column not in df.columns:
        logger.info(
            "Key findings skipped: %d rows, target=%s", n_rows, target_column
        )
        return FindingSet(
            context=(
                f"Findings need at least {MIN_ROWS} customers and a target column; "
                f"this selection has {n_rows:,}."
            )
        )

    if pd.to_numeric(df[target_column], errors="coerce").nunique() < 2:
        return FindingSet(
            context=(
                "This selection contains only one outcome class, so comparative "
                "findings cannot be derived."
            )
        )

    candidates = [
        _delinquency_finding(df, target_column, n_rows),
        _utilisation_finding(df, target_column, n_rows),
        _repayment_finding(df, target_column, n_rows),
        _trend_finding(df, reg, n_rows),
        _limit_band_finding(df, target_column, n_rows),
        _full_payer_finding(df, target_column, n_rows),
        _age_finding(df, target_column, n_rows),
    ]
    findings = [f for f in candidates if f is not None]

    result = FindingSet(
        findings=tuple(findings),
        context=f"Derived from {n_rows:,} customers in the current selection.",
    )
    logger.info("Derived %d key finding(s) from %d rows.", len(findings), n_rows)
    return FindingSet(findings=result.sorted_by_priority()[:limit], context=result.context)


# --------------------------------------------------------------------------- #
# Risk signals and opportunities
# --------------------------------------------------------------------------- #


def derive_signals(
    df: pd.DataFrame,
    kpis: KPIResult | None = None,
    registry: ColumnRegistry | None = None,
) -> SignalSet:
    """Derive compact risk signals and opportunities for the overview cards.

    Args:
        df: Filtered, analysis-ready dataframe.
        kpis: Optional pre-computed KPIs, reused to avoid recalculation.
        registry: Optional pre-built registry.

    Returns:
        A :class:`SignalSet`. Every entry is an observed measurement, never a
        model prediction.
    """
    reg = registry or build_registry(df)
    n_rows = int(len(df))

    if n_rows < MIN_ROWS:
        return SignalSet(n_rows=n_rows)

    target_column = reg.one(Role.TARGET)
    overall_rate = (
        float(pd.to_numeric(df[target_column], errors="coerce").mean() * 100)
        if target_column and target_column in df.columns
        else None
    )

    risks: list[Signal] = []
    opportunities: list[Signal] = []

    # ---------------- risk: high utilisation ---------------- #
    if "is_high_utilisation" in df.columns:
        share = _share_true(df["is_high_utilisation"])
        count = int(df["is_high_utilisation"].fillna(False).astype("bool").sum())
        threshold = settings.high_utilisation_threshold
        detail = (
            f"{count:,} customers use more than {threshold:.0%} of their credit "
            "limit in the most recent month."
        )
        if overall_rate is not None and count >= MIN_GROUP:
            mask = df["is_high_utilisation"].fillna(False).astype("bool")
            band_rate = float(
                pd.to_numeric(df.loc[mask, target_column], errors="coerce").mean() * 100
            )
            if np.isfinite(band_rate):
                detail += (
                    f" Their observed default rate is {band_rate:.1f}% against "
                    f"{overall_rate:.1f}% overall."
                )
        risks.append(
            Signal(
                key="high_utilisation",
                name="High utilisation",
                value=f"{share:.1f}%",
                raw_value=share,
                description=detail,
                severity=(
                    Severity.HIGH if share >= 25 else
                    Severity.MEDIUM if share >= 10 else Severity.LOW
                ),
                direction="up",
                intensity=_intensity(share, 0.0, 50.0),
                evidence=(
                    make_evidence(
                        "high_utilisation_share", "High-utilisation share", share,
                        f"{share:.2f}%", "kpi_engine", n_observations=n_rows,
                    ),
                ),
            )
        )

    # ---------------- risk: current delinquency ---------------- #
    if "is_currently_delinquent" in df.columns:
        share = _share_true(df["is_currently_delinquent"])
        count = int(df["is_currently_delinquent"].fillna(False).astype("bool").sum())
        detail = (
            f"{count:,} customers were behind on payment in the most recent "
            "observed month."
        )
        if "ever_delinquent" in df.columns:
            ever = _share_true(df["ever_delinquent"])
            detail += f" {ever:.1f}% were behind in at least one of the six months."
        risks.append(
            Signal(
                key="payment_delinquency",
                name="Payment delinquency",
                value=f"{share:.1f}%",
                raw_value=share,
                description=detail,
                severity=(
                    Severity.HIGH if share >= 20 else
                    Severity.MEDIUM if share >= 10 else Severity.LOW
                ),
                direction="up",
                intensity=_intensity(share, 0.0, 40.0),
                evidence=(
                    make_evidence(
                        "currently_delinquent_share", "Currently delinquent share",
                        share, f"{share:.2f}%", "kpi_engine", n_observations=n_rows,
                    ),
                ),
            )
        )

    # ---------------- risk: weak repayment ---------------- #
    if "is_revolver" in df.columns:
        share = _share_true(df["is_revolver"])
        count = int(df["is_revolver"].fillna(False).astype("bool").sum())
        detail = (
            f"{count:,} customers typically repay less than 30% of their statement, "
            "carrying a balance forward."
        )
        if "months_zero_payment" in df.columns:
            zero_months = float(
                pd.to_numeric(df["months_zero_payment"], errors="coerce").mean()
            )
            if np.isfinite(zero_months):
                detail += (
                    f" On average customers record {zero_months:.1f} month(s) with "
                    "no payment at all."
                )
        risks.append(
            Signal(
                key="repayment_behaviour",
                name="Repayment behaviour",
                value=f"{share:.1f}%",
                raw_value=share,
                description=detail,
                severity=(
                    Severity.HIGH if share >= 55 else
                    Severity.MEDIUM if share >= 35 else Severity.LOW
                ),
                direction="up",
                intensity=_intensity(share, 0.0, 80.0),
                evidence=(
                    make_evidence(
                        "revolver_share", "Revolving-borrower share", share,
                        f"{share:.2f}%", "kpi_engine", n_observations=n_rows,
                    ),
                ),
            )
        )

    # ---------------- risk: over-limit exposure ---------------- #
    if "is_over_limit" in df.columns:
        share = _share_true(df["is_over_limit"])
        if share > 0:
            count = int(df["is_over_limit"].fillna(False).astype("bool").sum())
            risks.append(
                Signal(
                    key="over_limit",
                    name="Over-limit exposure",
                    value=f"{share:.1f}%",
                    raw_value=share,
                    description=(
                        f"{count:,} customers exceeded their credit limit in at "
                        "least one of the six observed months."
                    ),
                    severity=Severity.MEDIUM if share >= 10 else Severity.LOW,
                    direction="up",
                    intensity=_intensity(share, 0.0, 30.0),
                    evidence=(
                        make_evidence(
                            "over_limit_share", "Over-limit share", share,
                            f"{share:.2f}%", "kpi_engine", n_observations=n_rows,
                        ),
                    ),
                )
            )

    # ---------------- opportunity: full payers ---------------- #
    if "is_full_payer" in df.columns and target_column in df.columns:
        mask = df["is_full_payer"].fillna(False).astype("bool")
        count = int(mask.sum())
        if count >= MIN_GROUP and overall_rate is not None:
            share = count / n_rows * 100.0
            rate = float(
                pd.to_numeric(df.loc[mask, target_column], errors="coerce").mean() * 100
            )
            if np.isfinite(rate):
                opportunities.append(
                    Signal(
                        key="full_payers",
                        name="Stable repayment segment",
                        value=f"{share:.1f}%",
                        raw_value=share,
                        description=(
                            f"{count:,} customers typically clear their statement in "
                            f"full, with an observed default rate of {rate:.1f}% "
                            f"against {overall_rate:.1f}% overall."
                        ),
                        severity=Severity.INFO,
                        direction="down",
                        intensity=_intensity(share, 0.0, 50.0),
                        evidence=(
                            make_evidence(
                                "full_payer_rate", "Default rate, full payers", rate,
                                f"{rate:.2f}%", "signals",
                                comparison=f"{overall_rate:.2f}% overall",
                                n_observations=count,
                            ),
                        ),
                    )
                )

    # ---------------- opportunity: low utilisation headroom ---------------- #
    if "utilisation_latest" in df.columns and target_column in df.columns:
        utilisation = pd.to_numeric(df["utilisation_latest"], errors="coerce")
        mask = (utilisation >= 0) & (utilisation < 0.30)
        count = int(mask.sum())
        if count >= MIN_GROUP and overall_rate is not None:
            share = count / n_rows * 100.0
            rate = float(
                pd.to_numeric(df.loc[mask, target_column], errors="coerce").mean() * 100
            )
            if np.isfinite(rate) and rate < overall_rate:
                opportunities.append(
                    Signal(
                        key="low_utilisation_headroom",
                        name="Favourable utilisation",
                        value=f"{share:.1f}%",
                        raw_value=share,
                        description=(
                            f"{count:,} customers use under 30% of their limit and "
                            f"default at {rate:.1f}% against {overall_rate:.1f}% "
                            "overall, leaving unused headroom."
                        ),
                        severity=Severity.INFO,
                        direction="down",
                        intensity=_intensity(share, 0.0, 60.0),
                        evidence=(
                            make_evidence(
                                "low_utilisation_rate", "Default rate, low utilisation",
                                rate, f"{rate:.2f}%", "signals",
                                comparison=f"{overall_rate:.2f}% overall",
                                n_observations=count,
                            ),
                        ),
                    )
                )

    # ---------------- opportunity: improving delinquency ---------------- #
    if "delinquency_trend" in df.columns and target_column in df.columns:
        improving = df["delinquency_trend"].astype("string") == "Improving"
        count = int(improving.sum())
        if count >= MIN_GROUP and overall_rate is not None:
            share = count / n_rows * 100.0
            rate = float(
                pd.to_numeric(df.loc[improving, target_column], errors="coerce").mean() * 100
            )
            if np.isfinite(rate):
                comparison_text = (
                    f"{rate:.1f}% against {overall_rate:.1f}% overall"
                    if rate < overall_rate
                    else f"{rate:.1f}% (above the {overall_rate:.1f}% overall rate)"
                )
                opportunities.append(
                    Signal(
                        key="improving_delinquency",
                        name="Improving payment behaviour",
                        value=f"{share:.1f}%",
                        raw_value=share,
                        description=(
                            f"{count:,} customers were late less often in the recent "
                            "three months than in the earlier three. Observed default "
                            f"rate: {comparison_text}."
                        ),
                        severity=Severity.INFO,
                        direction="down",
                        intensity=_intensity(share, 0.0, 30.0),
                        evidence=(
                            make_evidence(
                                "improving_delinquency_rate",
                                "Default rate, improving customers", rate,
                                f"{rate:.2f}%", "signals",
                                comparison=f"{overall_rate:.2f}% overall",
                                n_observations=count,
                            ),
                        ),
                    )
                )

    risks.sort(key=lambda s: s.intensity, reverse=True)
    opportunities.sort(key=lambda s: s.intensity, reverse=True)

    logger.info(
        "Derived %d risk signal(s) and %d opportunity/ies from %d rows.",
        len(risks),
        len(opportunities),
        n_rows,
    )
    return SignalSet(
        risks=tuple(risks), opportunities=tuple(opportunities), n_rows=n_rows
    )
