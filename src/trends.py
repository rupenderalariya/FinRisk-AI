"""
Trend analysis over the monthly panel.

What is genuinely possible here
-------------------------------
This dataset has **no calendar date column**. What it has is six monthly
observations per client (April to September 2005) stored as parallel columns.
That supports real period-over-period analysis: balances, repayments,
utilisation and delinquency can each be tracked month by month across 30,000
clients, and no part of the time axis is invented.

What is deliberately NOT offered
--------------------------------
* **No forecasting.** Six points cannot support a credible projection, and
  producing one would be the kind of fabricated output this project avoids.
* **No daily or weekly analysis.** The data is monthly; finer granularity does
  not exist.
* **No seasonality decomposition.** Six months is less than one annual cycle.

Anomaly detection uses a z-score on the period-to-period changes, and with only
six periods that is a weak test by construction. Findings are labelled as
"worth checking", never as confirmed anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import settings
from src.logging_setup import get_logger
from src.schema import (
    CHRONOLOGICAL_MONTHS,
    DATASET_PERIOD,
    DELINQUENCY_THRESHOLD,
    ColumnRegistry,
    Role,
    build_registry,
    month_label,
)

logger = get_logger(__name__)

#: Absolute percentage change below which a trend is called flat.
FLAT_THRESHOLD_PCT: Final[float] = 2.0
#: Absolute z-score threshold, used only as a fallback when MAD is zero.
ANOMALY_ZSCORE: Final[float] = 2.0
#: Modified z-score (median/MAD) threshold. 3.5 is the Iglewicz-Hoaglin
#: recommendation and is the primary test, because it resists the masking effect
#: a single large spike has on the standard deviation of a short series.
ANOMALY_MODIFIED_ZSCORE: Final[float] = 3.5
#: Minimum periods needed before anomaly detection is attempted at all.
MIN_PERIODS_FOR_ANOMALY: Final[int] = 4

NO_TEMPORAL_DATA_MESSAGE: Final[str] = (
    "Temporal trend analysis is not applicable because the selected dataset does "
    "not contain a suitable temporal variable."
)

PANEL_NOTE: Final[str] = (
    f"This dataset covers {DATASET_PERIOD} as six monthly snapshots per client "
    "rather than dated transactions. Period-over-period comparison is therefore "
    "valid, but no calendar-date resampling (daily/weekly), seasonality "
    "decomposition or forecast is offered, because six points cannot support them."
)


class TrendDirection(str, Enum):
    """Direction of a measured trend."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    FLAT = "flat"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Result structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PeriodPoint:
    """One period's aggregated value."""

    month_index: int
    month_label: str
    month_order: int
    value: float
    n_observations: int
    change_from_previous: float | None = None
    pct_change_from_previous: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "month_index": self.month_index,
            "month_label": self.month_label,
            "month_order": self.month_order,
            "value": self.value,
            "n_observations": self.n_observations,
            "change_from_previous": self.change_from_previous,
            "pct_change_from_previous": self.pct_change_from_previous,
        }


@dataclass(frozen=True)
class TrendResult:
    """Trend analysis for one measure across the panel."""

    measure: str
    label: str
    unit: str
    points: tuple[PeriodPoint, ...]
    direction: TrendDirection
    total_change: float
    total_change_pct: float | None
    average_period_change: float
    slope_per_period: float
    peak_period: str
    trough_period: str
    anomalies: tuple[str, ...] = field(default=())
    note: str = ""

    @property
    def first_value(self) -> float:
        """Value in the oldest period."""
        return self.points[0].value if self.points else float("nan")

    @property
    def last_value(self) -> float:
        """Value in the most recent period."""
        return self.points[-1].value if self.points else float("nan")

    @property
    def n_periods(self) -> int:
        """Number of periods analysed."""
        return len(self.points)

    def to_frame(self) -> pd.DataFrame:
        """Chart-ready dataframe in chronological order."""
        return pd.DataFrame([p.to_dict() for p in self.points])

    def describe(self) -> str:
        """Plain-language summary grounded in the computed numbers."""
        if not self.points:
            return f"No periods available for {self.label}."

        if self.direction is TrendDirection.FLAT:
            movement = "remained broadly flat"
        elif self.direction is TrendDirection.INCREASING:
            movement = "increased"
        else:
            movement = "decreased"

        change = (
            f" by {abs(self.total_change_pct):.1f}%"
            if self.total_change_pct is not None
            else ""
        )
        return (
            f"{self.label} {movement}{change} from {self.points[0].month_label} to "
            f"{self.points[-1].month_label}, peaking in {self.peak_period} and "
            f"bottoming in {self.trough_period}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "label": self.label,
            "unit": self.unit,
            "direction": self.direction.value,
            "n_periods": self.n_periods,
            "first_value": self.first_value,
            "last_value": self.last_value,
            "total_change": self.total_change,
            "total_change_pct": self.total_change_pct,
            "average_period_change": self.average_period_change,
            "slope_per_period": self.slope_per_period,
            "peak_period": self.peak_period,
            "trough_period": self.trough_period,
            "anomalies": list(self.anomalies),
            "description": self.describe(),
            "points": [p.to_dict() for p in self.points],
            "note": self.note,
        }


@dataclass(frozen=True)
class TrendUnavailable:
    """Returned when trend analysis genuinely cannot be performed."""

    reason: str
    measure: str = "all"
    message: str = NO_TEMPORAL_DATA_MESSAGE

    @property
    def is_available(self) -> bool:
        """Always False. Lets callers branch without an isinstance check."""
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": False,
            "measure": self.measure,
            "reason": self.reason,
            "message": self.message,
        }


# --------------------------------------------------------------------------- #
# Capability check
# --------------------------------------------------------------------------- #


def temporal_capability(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> dict[str, Any]:
    """Report exactly what temporal analysis this data supports.

    Called before any trend work so the UI can state the capability honestly
    instead of rendering an empty chart.

    Args:
        df: Dataframe to inspect.
        registry: Optional pre-built registry.

    Returns:
        Capability flags, the periods found, and an explanatory note.
    """
    reg = registry or build_registry(df)

    has_date = reg.has(Role.DATE)
    bills = reg.panel_columns(Role.BILL_AMOUNT)
    payments = reg.panel_columns(Role.PAYMENT_AMOUNT)
    statuses = reg.panel_columns(Role.PAYMENT_STATUS)
    months = sorted(set(bills) | set(payments) | set(statuses))

    return {
        "has_calendar_date": has_date,
        "has_monthly_panel": bool(months),
        "n_periods": len(months),
        "period_labels": [month_label(m) for m in sorted(months, reverse=True)],
        "panel_families": {
            "statement_balance": len(bills),
            "payment_amount": len(payments),
            "repayment_status": len(statuses),
        },
        "supports_period_comparison": len(months) >= 2,
        "supports_daily_weekly": has_date,
        "supports_forecasting": False,
        "forecasting_note": (
            "Forecasting is deliberately not offered: with only "
            f"{len(months)} observed periods, any projection would be unreliable "
            "and would amount to fabricated output."
        ),
        "note": PANEL_NOTE if months else NO_TEMPORAL_DATA_MESSAGE,
    }


# --------------------------------------------------------------------------- #
# Core trend computation
# --------------------------------------------------------------------------- #


def _direction_for(total_change_pct: float | None, total_change: float) -> TrendDirection:
    """Classify a trend direction from its overall change."""
    if total_change_pct is None:
        if total_change > 0:
            return TrendDirection.INCREASING
        if total_change < 0:
            return TrendDirection.DECREASING
        return TrendDirection.FLAT

    if abs(total_change_pct) < FLAT_THRESHOLD_PCT:
        return TrendDirection.FLAT
    return TrendDirection.INCREASING if total_change_pct > 0 else TrendDirection.DECREASING


def _detect_anomalies(points: Sequence[PeriodPoint]) -> tuple[str, ...]:
    """Flag periods whose change from the previous period is unusually large.

    Uses the **modified z-score** (median and MAD) rather than mean and standard
    deviation. With only about five period-to-period changes to work from, a
    single large spike inflates the standard deviation enough to hide itself -
    the classic masking problem. The median and MAD are unaffected by the very
    point being tested, so the spike still stands out.

    This remains a weak test by construction: six periods is very little
    evidence. Callers must present results as "worth checking", never as
    established anomalies.

    Args:
        points: Period points in chronological order.

    Returns:
        Labels of the flagged periods.
    """
    if len(points) < MIN_PERIODS_FOR_ANOMALY:
        return ()

    changes = np.array(
        [p.change_from_previous for p in points if p.change_from_previous is not None],
        dtype="float64",
    )
    if changes.size < 3:
        return ()

    median = float(np.median(changes))
    mad = float(np.median(np.abs(changes - median)))

    if mad > 0:
        # 0.6745 is the standard Iglewicz-Hoaglin constant, making the modified
        # z-score comparable in scale to an ordinary z-score.
        scores = 0.6745 * (changes - median) / mad
        threshold = ANOMALY_MODIFIED_ZSCORE
    else:
        # Every change is identical apart from possibly one; fall back to the
        # standard deviation so a lone deviation is still catchable.
        std = float(changes.std(ddof=0))
        if std <= 0:
            return ()
        scores = (changes - float(changes.mean())) / std
        threshold = ANOMALY_ZSCORE

    flagged: list[str] = []
    index = 0
    for point in points:
        if point.change_from_previous is None:
            continue
        if abs(scores[index]) > threshold:
            flagged.append(point.month_label)
        index += 1
    return tuple(flagged)


def _build_trend(
    measure: str,
    label: str,
    unit: str,
    values_by_month: Mapping[int, tuple[float, int]],
    note: str = "",
) -> TrendResult | TrendUnavailable:
    """Assemble a :class:`TrendResult` from per-month aggregates.

    Args:
        measure: Machine-readable measure key.
        label: Human-readable measure name.
        unit: Unit for display.
        values_by_month: ``{month_index: (value, n_observations)}``.
        note: Optional caveat carried into the result.

    Returns:
        A :class:`TrendResult`, or :class:`TrendUnavailable` with fewer than two
        periods.
    """
    ordered_months = [m for m in CHRONOLOGICAL_MONTHS if m in values_by_month]
    # Include any months outside the known chronology, oldest first.
    ordered_months += [m for m in sorted(values_by_month, reverse=True) if m not in ordered_months]

    if len(ordered_months) < 2:
        return TrendUnavailable(
            measure=measure,
            reason=(
                f"Only {len(ordered_months)} period(s) available for {label}; at least "
                "two are needed to measure a trend."
            ),
        )

    points: list[PeriodPoint] = []
    previous: float | None = None
    for position, month in enumerate(ordered_months, start=1):
        value, n_obs = values_by_month[month]
        change = None if previous is None else value - previous
        pct_change = (
            None
            if previous is None or previous == 0
            else (value - previous) / abs(previous) * 100.0
        )
        points.append(
            PeriodPoint(
                month_index=month,
                month_label=month_label(month),
                month_order=position,
                value=float(value),
                n_observations=int(n_obs),
                change_from_previous=change,
                pct_change_from_previous=pct_change,
            )
        )
        previous = value

    first = points[0].value
    last = points[-1].value
    total_change = last - first
    total_change_pct = (total_change / abs(first) * 100.0) if first != 0 else None

    values = np.array([p.value for p in points], dtype="float64")
    positions = np.arange(1, len(points) + 1, dtype="float64")
    slope = (
        float(np.polyfit(positions, values, 1)[0]) if np.isfinite(values).all() else float("nan")
    )

    peak = max(points, key=lambda p: p.value)
    trough = min(points, key=lambda p: p.value)

    return TrendResult(
        measure=measure,
        label=label,
        unit=unit,
        points=tuple(points),
        direction=_direction_for(total_change_pct, total_change),
        total_change=total_change,
        total_change_pct=total_change_pct,
        average_period_change=total_change / (len(points) - 1),
        slope_per_period=slope,
        peak_period=peak.month_label,
        trough_period=trough.month_label,
        anomalies=_detect_anomalies(points),
        note=note,
    )


# --------------------------------------------------------------------------- #
# Measure-specific trends
# --------------------------------------------------------------------------- #


def balance_trend(
    df: pd.DataFrame,
    registry: ColumnRegistry | None = None,
    statistic: str = "mean",
) -> TrendResult | TrendUnavailable:
    """Trend in statement balance across the panel.

    Args:
        df: Dataframe with the wide monthly columns.
        registry: Optional pre-built registry.
        statistic: ``mean``, ``median`` or ``sum``.

    Returns:
        A trend result, or :class:`TrendUnavailable`.
    """
    reg = registry or build_registry(df)
    panel = reg.panel_columns(Role.BILL_AMOUNT)
    if not panel:
        return TrendUnavailable(
            measure="statement_balance",
            reason="No monthly statement balance columns are present.",
        )

    values: dict[int, tuple[float, int]] = {}
    for month, column in panel.items():
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        aggregate = getattr(series, statistic)()
        values[month] = (float(aggregate), int(series.size))

    return _build_trend(
        measure="statement_balance",
        label=f"{statistic.title()} statement balance",
        unit="NT$",
        values_by_month=values,
        note=(
            "Median is a better guide than mean here, since balances are strongly "
            "right-skewed."
            if statistic == "mean"
            else ""
        ),
    )


def repayment_trend(
    df: pd.DataFrame,
    registry: ColumnRegistry | None = None,
    statistic: str = "mean",
) -> TrendResult | TrendUnavailable:
    """Trend in payment amounts across the panel.

    Args:
        df: Dataframe with the wide monthly columns.
        registry: Optional pre-built registry.
        statistic: ``mean``, ``median`` or ``sum``.

    Returns:
        A trend result, or :class:`TrendUnavailable`.
    """
    reg = registry or build_registry(df)
    panel = reg.panel_columns(Role.PAYMENT_AMOUNT)
    if not panel:
        return TrendUnavailable(
            measure="payment_amount",
            reason="No monthly payment amount columns are present.",
        )

    values: dict[int, tuple[float, int]] = {}
    for month, column in panel.items():
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        values[month] = (float(getattr(series, statistic)()), int(series.size))

    return _build_trend(
        measure="payment_amount",
        label=f"{statistic.title()} payment amount",
        unit="NT$",
        values_by_month=values,
    )


def utilisation_trend(
    df: pd.DataFrame, registry: ColumnRegistry | None = None, statistic: str = "mean"
) -> TrendResult | TrendUnavailable:
    """Trend in credit utilisation across the panel.

    Args:
        df: Dataframe carrying engineered ``utilisation_m*`` columns.
        registry: Optional pre-built registry.
        statistic: ``mean`` or ``median``.

    Returns:
        A trend result, or :class:`TrendUnavailable`.
    """
    reg = registry or build_registry(df)
    panel = reg.panel_columns(Role.UTILISATION)
    if not panel:
        return TrendUnavailable(
            measure="utilisation",
            reason=(
                "No monthly utilisation columns are present. Run feature "
                "engineering first."
            ),
        )

    values: dict[int, tuple[float, int]] = {}
    for month, column in panel.items():
        series = (
            pd.to_numeric(df[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if series.empty:
            continue
        values[month] = (float(getattr(series, statistic)() * 100.0), int(series.size))

    return _build_trend(
        measure="utilisation",
        label=f"{statistic.title()} credit utilisation",
        unit="%",
        values_by_month=values,
    )


def delinquency_trend(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> TrendResult | TrendUnavailable:
    """Trend in the share of clients behind on payment, per period.

    Args:
        df: Dataframe with the wide monthly status columns.
        registry: Optional pre-built registry.

    Returns:
        A trend result, or :class:`TrendUnavailable`.
    """
    reg = registry or build_registry(df)
    panel = reg.panel_columns(Role.PAYMENT_STATUS)
    if not panel:
        return TrendUnavailable(
            measure="delinquency_rate",
            reason="No monthly repayment status columns are present.",
        )

    values: dict[int, tuple[float, int]] = {}
    for month, column in panel.items():
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        rate = float((series >= DELINQUENCY_THRESHOLD).mean() * 100.0)
        values[month] = (rate, int(series.size))

    return _build_trend(
        measure="delinquency_rate",
        label="Share of clients behind on payment",
        unit="%",
        values_by_month=values,
        note=(
            "Counts only documented status codes of 1 or more (months of payment "
            "delay), so the undocumented -2 and 0 codes do not affect this measure."
        ),
    )


def active_account_trend(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> TrendResult | TrendUnavailable:
    """Trend in the share of accounts carrying a positive balance.

    A proxy for account activity, standing in for the transaction-volume trend
    that this dataset cannot provide.

    Args:
        df: Dataframe with the wide monthly columns.
        registry: Optional pre-built registry.

    Returns:
        A trend result, or :class:`TrendUnavailable`.
    """
    reg = registry or build_registry(df)
    panel = reg.panel_columns(Role.BILL_AMOUNT)
    if not panel:
        return TrendUnavailable(
            measure="active_accounts",
            reason="No monthly statement balance columns are present.",
        )

    values: dict[int, tuple[float, int]] = {}
    for month, column in panel.items():
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        values[month] = (float((series > 0).mean() * 100.0), int(series.size))

    return _build_trend(
        measure="active_accounts",
        label="Share of accounts with a positive balance",
        unit="%",
        values_by_month=values,
        note=(
            "Used as an activity proxy. The dataset holds no transaction counts, so "
            "true transaction volume cannot be measured."
        ),
    )


# --------------------------------------------------------------------------- #
# Panel-based aggregation
# --------------------------------------------------------------------------- #


def panel_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a long panel into one row per period.

    Args:
        panel: Long panel from :func:`src.data_processing.build_panel_long`.

    Returns:
        One row per period with the standard measures. Empty when the panel is.
    """
    expected = [
        "month_index", "month_label", "month_order", "n_clients",
        "mean_bill", "median_bill", "total_bill",
        "mean_payment", "total_payment",
        "delinquency_rate_pct", "mean_utilisation_pct", "active_share_pct",
    ]
    if panel.empty:
        logger.warning("panel_summary received an empty panel.")
        return pd.DataFrame({column: pd.Series(dtype="float64") for column in expected})

    grouped = panel.groupby(["month_index", "month_label", "month_order"], observed=True)
    summary = grouped.agg(
        n_clients=("client_id", "nunique"),
        mean_bill=("bill_amt", "mean"),
        median_bill=("bill_amt", "median"),
        total_bill=("bill_amt", "sum"),
        mean_payment=("pay_amt", "mean"),
        total_payment=("pay_amt", "sum"),
        delinquency_rate_pct=("is_delinquent", lambda s: float(s.mean() * 100.0)),
        mean_utilisation_pct=("utilisation", lambda s: float(s.mean() * 100.0)),
        active_share_pct=("bill_amt", lambda s: float((s > 0).mean() * 100.0)),
    ).reset_index()

    return summary.sort_values("month_order").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_trend_analysis(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> dict[str, Any]:
    """Run every applicable trend analysis.

    Args:
        df: Dataframe carrying cleaned data and engineered features.
        registry: Optional pre-built registry.

    Returns:
        Capability info plus one entry per measure. Measures that cannot be
        computed appear as :class:`TrendUnavailable`, never as fabricated data.
    """
    reg = registry or build_registry(df)
    capability = temporal_capability(df, reg)

    if not capability["supports_period_comparison"]:
        logger.warning("Trend analysis unavailable: %s", capability["note"])
        return {
            "capability": capability,
            "available": False,
            "message": NO_TEMPORAL_DATA_MESSAGE,
            "trends": {},
        }

    trends: dict[str, TrendResult | TrendUnavailable] = {
        "balance_mean": balance_trend(df, reg, "mean"),
        "balance_median": balance_trend(df, reg, "median"),
        "balance_total": balance_trend(df, reg, "sum"),
        "payment_mean": repayment_trend(df, reg, "mean"),
        "payment_total": repayment_trend(df, reg, "sum"),
        "utilisation_mean": utilisation_trend(df, reg, "mean"),
        "delinquency_rate": delinquency_trend(df, reg),
        "active_accounts": active_account_trend(df, reg),
    }

    computed = {k: v for k, v in trends.items() if isinstance(v, TrendResult)}
    logger.info(
        "Trend analysis complete: %d of %d measures computed across %d period(s).",
        len(computed),
        len(trends),
        capability["n_periods"],
    )

    return {
        "capability": capability,
        "available": True,
        "message": PANEL_NOTE,
        "trends": trends,
        "n_computed": len(computed),
    }
