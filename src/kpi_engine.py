"""
KPI calculation engine.

A registry of :class:`KPIDefinition` objects, each declaring the semantic roles
it needs. :func:`calculate_kpis` evaluates only the definitions whose
requirements are satisfied, so a KPI that the dataset cannot support is **omitted
entirely** rather than displayed as "N/A". Anything omitted is explained in
:attr:`KPIResult.unavailable`.

Precision policy
----------------
Values are stored as full-precision floats. Rounding happens only in the
``formatted`` string used for display, so no downstream calculation ever
inherits a rounded input.

Skew policy
-----------
Financial distributions are strongly right-skewed - the verified statement
balance has a mean of about NT$51,223 against a median of NT$22,382. Where mean
and median tell materially different stories, both are registered, and the
description says which one to trust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import settings
from src.logging_setup import get_logger
from src.schema import (
    DATASET_CURRENCY,
    UNAVAILABLE_ROLES,
    ColumnRegistry,
    Role,
    build_registry,
)

logger = get_logger(__name__)


class KPIFormat(str, Enum):
    """How a KPI value should be rendered."""

    INTEGER = "integer"
    CURRENCY = "currency"
    CURRENCY_COMPACT = "currency_compact"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    DECIMAL = "decimal"
    MONTHS = "months"


class KPICategory(str, Enum):
    """Grouping used to lay KPIs out in the dashboard."""

    PORTFOLIO = "Portfolio scale"
    RISK = "Risk"
    CREDIT = "Credit capacity"
    UTILISATION = "Utilisation"
    REPAYMENT = "Repayment behaviour"
    DELINQUENCY = "Delinquency"
    DEMOGRAPHICS = "Demographics"
    TREND = "Trend"


# --------------------------------------------------------------------------- #
# Value formatting
# --------------------------------------------------------------------------- #


def format_currency(value: float, compact: bool = False, currency: str = DATASET_CURRENCY) -> str:
    """Format a monetary amount.

    Args:
        value: Amount to format.
        compact: Use K/M/B suffixes, which keeps portfolio totals readable on a
            KPI card.
        currency: Currency prefix.

    Returns:
        A display string.

    Examples:
        >>> format_currency(167484.32)
        'NT$167,484'
        >>> format_currency(5024529900, compact=True)
        'NT$5.02B'
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"

    sign = "-" if value < 0 else ""
    magnitude = abs(float(value))

    if compact:
        for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if magnitude >= threshold:
                return f"{sign}{currency}{magnitude / threshold:,.2f}{suffix}"
    return f"{sign}{currency}{magnitude:,.0f}"


def format_value(value: float | None, fmt: KPIFormat, decimals: int = 2) -> str:
    """Render a KPI value according to its declared format.

    Args:
        value: Raw full-precision value, or None.
        fmt: Target format.
        decimals: Decimal places for the decimal/ratio/percentage formats.

    Returns:
        A display string; ``"n/a"`` for None or a non-finite value.
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"

    if fmt is KPIFormat.INTEGER:
        # Half-up rounding rather than Python's round(), which rounds half to
        # even (round(1234.5) == 1234). Half-up is what a reader expects on a
        # dashboard, and the difference is otherwise a silent surprise.
        rounded = math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)
        return f"{int(rounded):,}"
    if fmt is KPIFormat.CURRENCY:
        return format_currency(value)
    if fmt is KPIFormat.CURRENCY_COMPACT:
        return format_currency(value, compact=True)
    if fmt is KPIFormat.PERCENTAGE:
        return f"{value:.{decimals}f}%"
    if fmt is KPIFormat.RATIO:
        return f"{value:.{decimals}f}x"
    if fmt is KPIFormat.MONTHS:
        return f"{value:.{decimals}f} months"
    return f"{value:,.{decimals}f}"


# --------------------------------------------------------------------------- #
# Definition and value structures
# --------------------------------------------------------------------------- #

#: Signature every KPI computation must satisfy. Returning None means
#: "not computable on this data", and the KPI is omitted.
KPIComputeFn = Callable[[pd.DataFrame, ColumnRegistry], "float | None"]


@dataclass(frozen=True)
class KPIDefinition:
    """Declarative description of one KPI."""

    key: str
    label: str
    description: str
    formula: str
    compute: KPIComputeFn
    fmt: KPIFormat
    category: KPICategory
    required_roles: tuple[Role, ...] = field(default=())
    required_columns: tuple[str, ...] = field(default=())
    decimals: int = 2
    higher_is_better: bool | None = None

    def is_supported(self, df: pd.DataFrame, registry: ColumnRegistry) -> tuple[bool, str]:
        """Check whether this KPI can be computed.

        Args:
            df: Dataframe to check.
            registry: Column registry.

        Returns:
            ``(supported, reason_if_not)``.
        """
        missing_roles = registry.missing(*self.required_roles)
        if missing_roles:
            names = ", ".join(role.value for role in missing_roles)
            return False, f"Requires missing data role(s): {names}."

        missing_columns = [c for c in self.required_columns if c not in df.columns]
        if missing_columns:
            return False, f"Requires missing column(s): {', '.join(missing_columns)}."

        return True, ""


@dataclass(frozen=True)
class KPIValue:
    """A computed KPI, carrying both the raw value and its display string."""

    key: str
    label: str
    value: float
    formatted: str
    description: str
    formula: str
    category: str
    fmt: str
    higher_is_better: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form. ``value`` keeps full precision."""
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "formatted": self.formatted,
            "description": self.description,
            "formula": self.formula,
            "category": self.category,
            "higher_is_better": self.higher_is_better,
        }


@dataclass(frozen=True)
class KPIResult:
    """The full outcome of a KPI run."""

    values: Mapping[str, KPIValue]
    unavailable: Mapping[str, str]
    n_rows: int
    insufficient_data: bool = False

    def __len__(self) -> int:
        return len(self.values)

    def __contains__(self, key: object) -> bool:
        return key in self.values

    def get(self, key: str) -> KPIValue | None:
        """Fetch one KPI by key, or None when unavailable."""
        return self.values.get(key)

    def raw(self, key: str, default: float | None = None) -> float | None:
        """Fetch a KPI's full-precision value."""
        kpi = self.values.get(key)
        return kpi.value if kpi else default

    def formatted(self, key: str, default: str = "n/a") -> str:
        """Fetch a KPI's display string."""
        kpi = self.values.get(key)
        return kpi.formatted if kpi else default

    def by_category(self) -> dict[str, list[KPIValue]]:
        """Group the computed KPIs by category, preserving registration order."""
        grouped: dict[str, list[KPIValue]] = {}
        for kpi in self.values.values():
            grouped.setdefault(kpi.category, []).append(kpi)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        """Flat ``{key: raw_value}`` mapping - the shape the brief asks for."""
        return {key: kpi.value for key, kpi in self.values.items()}

    def to_display_dict(self) -> dict[str, str]:
        """Flat ``{key: formatted_value}`` mapping for display."""
        return {key: kpi.formatted for key, kpi in self.values.items()}

    def to_full_dict(self) -> dict[str, Any]:
        """Complete serialisable payload, safe to hand to the AI layer."""
        return {
            "n_rows": self.n_rows,
            "insufficient_data": self.insufficient_data,
            "kpis": {key: kpi.to_dict() for key, kpi in self.values.items()},
            "unavailable": dict(self.unavailable),
        }


# --------------------------------------------------------------------------- #
# Computation helpers
# --------------------------------------------------------------------------- #


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a column to numeric, or return an empty series when absent."""
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _safe_mean(series: pd.Series) -> float | None:
    """Mean of the finite values, or None when there are none."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if not clean.empty else None


def _safe_median(series: pd.Series) -> float | None:
    """Median of the finite values, or None when there are none."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.median()) if not clean.empty else None


def _safe_sum(series: pd.Series) -> float | None:
    """Sum of the finite values, or None when there are none."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.sum()) if not clean.empty else None


def _share_true(series: pd.Series) -> float | None:
    """Percentage of True values in a boolean-like series."""
    if series.empty:
        return None
    as_bool = series.fillna(False).astype("bool")
    return float(as_bool.mean() * 100.0)


def _role_column(registry: ColumnRegistry, role: Role) -> str | None:
    """First column fulfilling ``role``, or None."""
    return registry.one(role)


# --------------------------------------------------------------------------- #
# KPI registry
# --------------------------------------------------------------------------- #


def _build_registry_definitions() -> tuple[KPIDefinition, ...]:
    """Construct the KPI registry.

    A function rather than a module constant so the closures stay readable and
    the registry can be rebuilt in tests.
    """

    def total_clients(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        id_column = _role_column(reg, Role.CLIENT_ID)
        if id_column:
            return float(df[id_column].nunique(dropna=True))
        return float(len(df))

    def total_records(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return float(len(df))

    def client_month_records(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        months = len(reg.all(Role.BILL_AMOUNT))
        return float(len(df) * months) if months else None

    def observed_default_rate(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        column = reg.require(Role.TARGET)
        values = _numeric(df, column).dropna()
        return float(values.mean() * 100.0) if not values.empty else None

    def default_client_count(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        column = reg.require(Role.TARGET)
        values = _numeric(df, column).dropna()
        return float((values == 1).sum()) if not values.empty else None

    def avg_credit_limit(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        return _safe_mean(_numeric(df, reg.require(Role.CREDIT_LIMIT)))

    def median_credit_limit(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        return _safe_median(_numeric(df, reg.require(Role.CREDIT_LIMIT)))

    def total_credit_limit(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        return _safe_sum(_numeric(df, reg.require(Role.CREDIT_LIMIT)))

    def total_outstanding_balance(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        panel = reg.panel_columns(Role.BILL_AMOUNT)
        if not panel:
            return None
        return _safe_sum(_numeric(df, panel[min(panel)]))

    def avg_outstanding_balance(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        panel = reg.panel_columns(Role.BILL_AMOUNT)
        if not panel:
            return None
        return _safe_mean(_numeric(df, panel[min(panel)]))

    def median_outstanding_balance(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        panel = reg.panel_columns(Role.BILL_AMOUNT)
        if not panel:
            return None
        return _safe_median(_numeric(df, panel[min(panel)]))

    def avg_utilisation(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        mean = _safe_mean(_numeric(df, "utilisation_latest"))
        return mean * 100.0 if mean is not None else None

    def median_utilisation(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        median = _safe_median(_numeric(df, "utilisation_latest"))
        return median * 100.0 if median is not None else None

    def high_utilisation_share(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return _share_true(df["is_high_utilisation"])

    def high_utilisation_count(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return float(df["is_high_utilisation"].fillna(False).astype("bool").sum())

    def over_limit_share(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return _share_true(df["is_over_limit"])

    def total_repayments(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return _safe_sum(_numeric(df, "total_paid_6m"))

    def median_repayment_ratio(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        median = _safe_median(_numeric(df, "repayment_ratio_median"))
        return median * 100.0 if median is not None else None

    def avg_capped_repayment_ratio(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        mean = _safe_mean(_numeric(df, "repayment_ratio_capped_mean"))
        return mean * 100.0 if mean is not None else None

    def full_payer_share(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return _share_true(df["is_full_payer"])

    def revolver_share(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return _share_true(df["is_revolver"])

    def currently_delinquent_share(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return _share_true(df["is_currently_delinquent"])

    def currently_delinquent_count(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return float(df["is_currently_delinquent"].fillna(False).astype("bool").sum())

    def ever_delinquent_share(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return _share_true(df["ever_delinquent"])

    def avg_max_delinquency(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return _safe_mean(_numeric(df, "max_delinquency"))

    def avg_delinquent_months(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return _safe_mean(_numeric(df, "delinquent_months_count"))

    def avg_age(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        return _safe_mean(_numeric(df, reg.require(Role.AGE)))

    def median_age(df: pd.DataFrame, reg: ColumnRegistry) -> float | None:
        return _safe_median(_numeric(df, reg.require(Role.AGE)))

    def median_balance_trend(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        return _safe_median(_numeric(df, "bill_trend_slope"))

    def share_growing_balance(df: pd.DataFrame, _reg: ColumnRegistry) -> float | None:
        values = _numeric(df, "bill_trend_slope").dropna()
        return float((values > 0).mean() * 100.0) if not values.empty else None

    return (
        # ---------------- Portfolio scale ----------------
        KPIDefinition(
            key="total_clients",
            label="Total Clients",
            description="Distinct credit-card clients in the current selection.",
            formula="count of distinct client_id",
            compute=total_clients,
            fmt=KPIFormat.INTEGER,
            category=KPICategory.PORTFOLIO,
        ),
        KPIDefinition(
            key="total_records",
            label="Total Records",
            description="Rows in the current selection. One row is one client.",
            formula="row count",
            compute=total_records,
            fmt=KPIFormat.INTEGER,
            category=KPICategory.PORTFOLIO,
        ),
        KPIDefinition(
            key="client_month_records",
            label="Client-Month Observations",
            description=(
                "Client rows multiplied by observed months - the size of the "
                "underlying monthly panel available for trend analysis."
            ),
            formula="row count x number of monthly statement columns",
            compute=client_month_records,
            fmt=KPIFormat.INTEGER,
            category=KPICategory.PORTFOLIO,
            required_roles=(Role.BILL_AMOUNT,),
        ),
        # ---------------- Risk ----------------
        KPIDefinition(
            key="observed_default_rate",
            label="Observed Default Rate",
            description=(
                "Share of clients who defaulted on the following month's payment. "
                "This is the recorded historical outcome in the data, NOT a model "
                "prediction."
            ),
            formula="mean(default_next_month) x 100",
            compute=observed_default_rate,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.RISK,
            required_roles=(Role.TARGET,),
            higher_is_better=False,
        ),
        KPIDefinition(
            key="default_client_count",
            label="Defaulting Clients",
            description="Number of clients recorded as defaulting next month.",
            formula="count(default_next_month == 1)",
            compute=default_client_count,
            fmt=KPIFormat.INTEGER,
            category=KPICategory.RISK,
            required_roles=(Role.TARGET,),
            higher_is_better=False,
        ),
        # ---------------- Credit capacity ----------------
        KPIDefinition(
            key="avg_credit_limit",
            label="Average Credit Limit",
            description=(
                "Mean credit limit. This is a CREDIT LIMIT, not income - the dataset "
                "has no income column. Compare with the median, since limits are "
                "right-skewed."
            ),
            formula="mean(credit_limit)",
            compute=avg_credit_limit,
            fmt=KPIFormat.CURRENCY,
            category=KPICategory.CREDIT,
            required_roles=(Role.CREDIT_LIMIT,),
        ),
        KPIDefinition(
            key="median_credit_limit",
            label="Median Credit Limit",
            description=(
                "Typical credit limit. More representative than the mean because the "
                "distribution is right-skewed."
            ),
            formula="median(credit_limit)",
            compute=median_credit_limit,
            fmt=KPIFormat.CURRENCY,
            category=KPICategory.CREDIT,
            required_roles=(Role.CREDIT_LIMIT,),
        ),
        KPIDefinition(
            key="total_credit_limit",
            label="Total Credit Exposure Limit",
            description=(
                "Sum of all credit limits: the maximum the portfolio could owe if "
                "every client drew their full limit."
            ),
            formula="sum(credit_limit)",
            compute=total_credit_limit,
            fmt=KPIFormat.CURRENCY_COMPACT,
            category=KPICategory.CREDIT,
            required_roles=(Role.CREDIT_LIMIT,),
        ),
        KPIDefinition(
            key="total_outstanding_balance",
            label="Total Outstanding Balance",
            description="Sum of the most recent monthly statement balances.",
            formula="sum(bill_amt_m1)",
            compute=total_outstanding_balance,
            fmt=KPIFormat.CURRENCY_COMPACT,
            category=KPICategory.CREDIT,
            required_roles=(Role.BILL_AMOUNT,),
        ),
        KPIDefinition(
            key="avg_outstanding_balance",
            label="Average Outstanding Balance",
            description="Mean of the most recent monthly statement balances.",
            formula="mean(bill_amt_m1)",
            compute=avg_outstanding_balance,
            fmt=KPIFormat.CURRENCY,
            category=KPICategory.CREDIT,
            required_roles=(Role.BILL_AMOUNT,),
        ),
        KPIDefinition(
            key="median_outstanding_balance",
            label="Median Outstanding Balance",
            description=(
                "Typical most-recent statement balance. Materially lower than the "
                "mean because balances are right-skewed."
            ),
            formula="median(bill_amt_m1)",
            compute=median_outstanding_balance,
            fmt=KPIFormat.CURRENCY,
            category=KPICategory.CREDIT,
            required_roles=(Role.BILL_AMOUNT,),
        ),
        # ---------------- Utilisation ----------------
        KPIDefinition(
            key="avg_utilisation",
            label="Average Credit Utilisation",
            description=(
                "Mean share of the credit limit in use in the most recent month. "
                "This is the closest available equivalent to a debt-to-income ratio, "
                "which cannot be computed without an income column."
            ),
            formula="mean(bill_amt_m1 / credit_limit) x 100",
            compute=avg_utilisation,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.UTILISATION,
            required_columns=("utilisation_latest",),
            higher_is_better=False,
        ),
        KPIDefinition(
            key="median_utilisation",
            label="Median Credit Utilisation",
            description="Typical utilisation in the most recent month.",
            formula="median(bill_amt_m1 / credit_limit) x 100",
            compute=median_utilisation,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.UTILISATION,
            required_columns=("utilisation_latest",),
            higher_is_better=False,
        ),
        KPIDefinition(
            key="high_utilisation_share",
            label="High-Utilisation Clients",
            description=(
                "Share of clients using more than "
                f"{settings.high_utilisation_threshold:.0%} of their credit limit."
            ),
            formula=f"count(utilisation_latest > {settings.high_utilisation_threshold}) / total x 100",
            compute=high_utilisation_share,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.UTILISATION,
            required_columns=("is_high_utilisation",),
            higher_is_better=False,
        ),
        KPIDefinition(
            key="high_utilisation_count",
            label="High-Utilisation Client Count",
            description=(
                "Number of clients above the high-utilisation threshold in the most "
                "recent month."
            ),
            formula=f"count(utilisation_latest > {settings.high_utilisation_threshold})",
            compute=high_utilisation_count,
            fmt=KPIFormat.INTEGER,
            category=KPICategory.UTILISATION,
            required_columns=("is_high_utilisation",),
            higher_is_better=False,
        ),
        KPIDefinition(
            key="over_limit_share",
            label="Over-Limit Clients",
            description=(
                "Share of clients whose statement balance exceeded their credit limit "
                "in at least one observed month."
            ),
            formula="count(utilisation_max_6m > 1.0) / total x 100",
            compute=over_limit_share,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.UTILISATION,
            required_columns=("is_over_limit",),
            higher_is_better=False,
        ),
        # ---------------- Repayment ----------------
        KPIDefinition(
            key="total_repayments",
            label="Total Repayments (6 months)",
            description="Sum of every payment received across the observed months.",
            formula="sum(total_paid_6m)",
            compute=total_repayments,
            fmt=KPIFormat.CURRENCY_COMPACT,
            category=KPICategory.REPAYMENT,
            required_columns=("total_paid_6m",),
            higher_is_better=True,
        ),
        KPIDefinition(
            key="median_repayment_ratio",
            label="Median Repayment Ratio",
            description=(
                "Typical share of a statement that clients repay. The median is used "
                "because the raw ratio is unbounded: a near-zero prior balance can "
                "produce an enormous value that would distort a mean."
            ),
            formula="median(client median repayment ratio) x 100",
            compute=median_repayment_ratio,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.REPAYMENT,
            required_columns=("repayment_ratio_median",),
            higher_is_better=True,
        ),
        KPIDefinition(
            key="avg_capped_repayment_ratio",
            label="Average Repayment Ratio (capped)",
            description=(
                "Mean repayment share with each month clipped at 200% of the "
                "statement, so pathological denominators cannot dominate the average."
            ),
            formula="mean(repayment_ratio_capped_mean) x 100",
            compute=avg_capped_repayment_ratio,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.REPAYMENT,
            required_columns=("repayment_ratio_capped_mean",),
            higher_is_better=True,
        ),
        KPIDefinition(
            key="full_payer_share",
            label="Full Payers",
            description=(
                "Share of clients who typically clear their statement in full - "
                "transactors rather than borrowers."
            ),
            formula="count(repayment_ratio_median >= 0.95) / total x 100",
            compute=full_payer_share,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.REPAYMENT,
            required_columns=("is_full_payer",),
            higher_is_better=True,
        ),
        KPIDefinition(
            key="revolver_share",
            label="Revolving Borrowers",
            description=(
                "Share of clients who typically repay less than 30% of the statement, "
                "carrying a balance forward."
            ),
            formula="count(repayment_ratio_median < 0.30) / total x 100",
            compute=revolver_share,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.REPAYMENT,
            required_columns=("is_revolver",),
            higher_is_better=False,
        ),
        # ---------------- Delinquency ----------------
        KPIDefinition(
            key="currently_delinquent_share",
            label="Currently Delinquent",
            description=(
                "Share of clients behind on payment in the most recent observed month."
            ),
            formula="count(current_delinquency >= 1) / total x 100",
            compute=currently_delinquent_share,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.DELINQUENCY,
            required_columns=("is_currently_delinquent",),
            higher_is_better=False,
        ),
        KPIDefinition(
            key="currently_delinquent_count",
            label="Currently Delinquent Clients",
            description="Number of clients behind on payment in the most recent month.",
            formula="count(current_delinquency >= 1)",
            compute=currently_delinquent_count,
            fmt=KPIFormat.INTEGER,
            category=KPICategory.DELINQUENCY,
            required_columns=("is_currently_delinquent",),
            higher_is_better=False,
        ),
        KPIDefinition(
            key="ever_delinquent_share",
            label="Ever Delinquent",
            description=(
                "Share of clients behind on payment in at least one observed month."
            ),
            formula="count(any pay_status >= 1) / total x 100",
            compute=ever_delinquent_share,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.DELINQUENCY,
            required_columns=("ever_delinquent",),
            higher_is_better=False,
        ),
        KPIDefinition(
            key="avg_delinquent_months",
            label="Average Delinquent Months",
            description=(
                "Mean number of months, out of those observed, in which a client was "
                "behind on payment. Measures frequency."
            ),
            formula="mean(delinquent_months_count)",
            compute=avg_delinquent_months,
            fmt=KPIFormat.DECIMAL,
            category=KPICategory.DELINQUENCY,
            required_columns=("delinquent_months_count",),
            higher_is_better=False,
        ),
        KPIDefinition(
            key="avg_max_delinquency",
            label="Average Worst Delay",
            description=(
                "Mean of each client's worst payment delay, in months. Measures "
                "severity rather than frequency."
            ),
            formula="mean(max_delinquency)",
            compute=avg_max_delinquency,
            fmt=KPIFormat.MONTHS,
            category=KPICategory.DELINQUENCY,
            required_columns=("max_delinquency",),
            higher_is_better=False,
        ),
        # ---------------- Demographics ----------------
        KPIDefinition(
            key="avg_age",
            label="Average Age",
            description="Mean client age in years.",
            formula="mean(age)",
            compute=avg_age,
            fmt=KPIFormat.DECIMAL,
            category=KPICategory.DEMOGRAPHICS,
            required_roles=(Role.AGE,),
        ),
        KPIDefinition(
            key="median_age",
            label="Median Age",
            description="Typical client age in years.",
            formula="median(age)",
            compute=median_age,
            fmt=KPIFormat.DECIMAL,
            category=KPICategory.DEMOGRAPHICS,
            required_roles=(Role.AGE,),
        ),
        # ---------------- Trend ----------------
        KPIDefinition(
            key="median_balance_trend",
            label="Median Balance Trend",
            description=(
                "Typical month-on-month change in statement balance (NT$ per month). "
                "Positive means balances are growing across the portfolio."
            ),
            formula="median(bill_trend_slope)",
            compute=median_balance_trend,
            fmt=KPIFormat.CURRENCY,
            category=KPICategory.TREND,
            required_columns=("bill_trend_slope",),
        ),
        KPIDefinition(
            key="share_growing_balance",
            label="Clients With Growing Balances",
            description=(
                "Share of clients whose statement balance trended upward across the "
                "observed months."
            ),
            formula="count(bill_trend_slope > 0) / total x 100",
            compute=share_growing_balance,
            fmt=KPIFormat.PERCENTAGE,
            category=KPICategory.TREND,
            required_columns=("bill_trend_slope",),
            higher_is_better=False,
        ),
    )


#: The KPI registry, built once at import.
KPI_REGISTRY: Final[tuple[KPIDefinition, ...]] = _build_registry_definitions()


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def calculate_kpis(
    df: pd.DataFrame,
    registry: ColumnRegistry | None = None,
    definitions: Sequence[KPIDefinition] | None = None,
    min_rows: int | None = None,
) -> KPIResult:
    """Evaluate every supported KPI against the supplied dataframe.

    Unsupported KPIs are omitted and explained. A KPI whose computation raises is
    also omitted, with the error recorded, so one broken metric can never take
    the dashboard down.

    Args:
        df: Dataframe carrying cleaned data and engineered features.
        registry: Optional pre-built registry.
        definitions: Optional KPI subset. Defaults to the full registry.
        min_rows: Minimum rows required. Below this the result is marked
            ``insufficient_data`` but still computed, so the UI can warn rather
            than silently present statistics from a handful of records.

    Returns:
        A :class:`KPIResult`.
    """
    reg = registry or build_registry(df)
    active = tuple(definitions) if definitions is not None else KPI_REGISTRY
    threshold = min_rows if min_rows is not None else settings.min_rows_for_analysis

    values: dict[str, KPIValue] = {}
    unavailable: dict[str, str] = {}

    if df.empty:
        for definition in active:
            unavailable[definition.key] = "No rows in the current selection."
        logger.warning("KPI calculation skipped: the dataframe is empty.")
        return KPIResult(values={}, unavailable=unavailable, n_rows=0, insufficient_data=True)

    for definition in active:
        supported, reason = definition.is_supported(df, reg)
        if not supported:
            unavailable[definition.key] = reason
            continue

        try:
            raw = definition.compute(df, reg)
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            unavailable[definition.key] = f"Calculation failed: {type(exc).__name__}: {exc}"
            logger.warning("KPI '%s' failed to compute: %s", definition.key, exc)
            continue

        if raw is None or (isinstance(raw, float) and not math.isfinite(raw)):
            unavailable[definition.key] = (
                "No finite value could be computed from the available rows."
            )
            continue

        values[definition.key] = KPIValue(
            key=definition.key,
            label=definition.label,
            value=float(raw),
            formatted=format_value(float(raw), definition.fmt, definition.decimals),
            description=definition.description,
            formula=definition.formula,
            category=definition.category.value,
            fmt=definition.fmt.value,
            higher_is_better=definition.higher_is_better,
        )

    # Explain the roles this dataset lacks, so absent KPIs have a visible reason.
    for role, explanation in UNAVAILABLE_ROLES.items():
        if not reg.has(role):
            unavailable.setdefault(f"role:{role.value}", explanation)

    insufficient = len(df) < threshold
    if insufficient:
        logger.warning(
            "Only %d row(s) in the selection, below the minimum of %d. KPIs were "
            "computed but are flagged as based on insufficient data.",
            len(df),
            threshold,
        )

    logger.info(
        "Calculated %d KPI(s); %d unavailable.", len(values), len(unavailable)
    )
    return KPIResult(
        values=values,
        unavailable=unavailable,
        n_rows=int(len(df)),
        insufficient_data=insufficient,
    )


def kpi_catalogue() -> list[dict[str, Any]]:
    """Describe every registered KPI without computing anything.

    Powers the Methodology page's KPI-definition table.

    Returns:
        One dictionary per registered KPI.
    """
    return [
        {
            "key": definition.key,
            "label": definition.label,
            "description": definition.description,
            "formula": definition.formula,
            "category": definition.category.value,
            "format": definition.fmt.value,
            "required_roles": [role.value for role in definition.required_roles],
            "required_columns": list(definition.required_columns),
            "higher_is_better": definition.higher_is_better,
        }
        for definition in KPI_REGISTRY
    ]
