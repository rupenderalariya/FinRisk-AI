"""
Global filter logic.

Pure functions over dataframes, with no Streamlit import, so filtering is unit
testable and the same code path serves the dashboard, the tests and any future
report generator.

The contract that matters: :func:`apply_filters` is called **once** per rerun and
its output feeds every KPI, chart and table on the page. That is what prevents
the failure mode where filters silently affect only some visuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from src.logging_setup import get_logger
from src.schema import ColumnRegistry, Role, build_registry

logger = get_logger(__name__)

#: Delinquency filter options. Keys are stable; values are display labels.
DELINQUENCY_OPTIONS: Final[Mapping[str, str]] = {
    "all": "All customers",
    "current": "Currently behind on payment",
    "ever": "Ever behind on payment",
    "never": "Never behind on payment",
}

#: Utilisation band definitions as ``(label, lower, upper)`` with the upper bound
#: exclusive except for the final band. Thresholds are documented in
#: :func:`utilisation_band_documentation`.
UTILISATION_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("Negative / zero", -np.inf, 0.0001),
    ("Low (0-30%)", 0.0001, 0.30),
    ("Moderate (30-60%)", 0.30, 0.60),
    ("High (60-80%)", 0.60, 0.80),
    ("Very high (80%+)", 0.80, np.inf),
)


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FilterSpec:
    """A complete filter selection.

    ``None`` means "no constraint on this dimension", which keeps the difference
    between "unset" and "everything selected" explicit.
    """

    age_range: tuple[int, int] | None = None
    limit_range: tuple[float, float] | None = None
    utilisation_range: tuple[float, float] | None = None
    education: tuple[str, ...] | None = None
    marriage: tuple[str, ...] | None = None
    sex: tuple[str, ...] | None = None
    delinquency: str = "all"

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, used for cache keys and display."""
        return {
            "age_range": list(self.age_range) if self.age_range else None,
            "limit_range": list(self.limit_range) if self.limit_range else None,
            "utilisation_range": (
                list(self.utilisation_range) if self.utilisation_range else None
            ),
            "education": list(self.education) if self.education else None,
            "marriage": list(self.marriage) if self.marriage else None,
            "sex": list(self.sex) if self.sex else None,
            "delinquency": self.delinquency,
        }

    def reset(self) -> "FilterSpec":
        """Return an unconstrained specification."""
        return FilterSpec()


@dataclass(frozen=True)
class FilterBounds:
    """The selectable range of each filter, read from the data itself.

    Derived rather than hard-coded so the widgets always match reality.
    """

    age_min: int = 0
    age_max: int = 100
    limit_min: float = 0.0
    limit_max: float = 1_000_000.0
    utilisation_min: float = 0.0
    utilisation_max: float = 1.0
    education_options: tuple[str, ...] = field(default=())
    marriage_options: tuple[str, ...] = field(default=())
    sex_options: tuple[str, ...] = field(default=())

    @property
    def has_age(self) -> bool:
        """True when age filtering is possible."""
        return self.age_max > self.age_min

    @property
    def has_limit(self) -> bool:
        """True when credit-limit filtering is possible."""
        return self.limit_max > self.limit_min

    @property
    def has_utilisation(self) -> bool:
        """True when utilisation filtering is possible."""
        return self.utilisation_max > self.utilisation_min


@dataclass(frozen=True)
class FilterResult:
    """The outcome of applying a filter specification."""

    data: pd.DataFrame
    spec: FilterSpec
    n_total: int
    n_selected: int
    applied: tuple[str, ...] = field(default=())
    skipped: tuple[str, ...] = field(default=())

    @property
    def is_filtered(self) -> bool:
        """True when at least one constraint actually narrowed the data."""
        return bool(self.applied)

    @property
    def is_empty(self) -> bool:
        """True when no rows survived."""
        return self.n_selected == 0

    @property
    def share_pct(self) -> float:
        """Selected rows as a percentage of the total."""
        return 0.0 if self.n_total == 0 else self.n_selected / self.n_total * 100.0

    @property
    def n_excluded(self) -> int:
        """Rows removed by the filters."""
        return self.n_total - self.n_selected

    def summary(self) -> str:
        """One-line description of the selection."""
        if not self.is_filtered:
            return f"All {self.n_total:,} customers (no filters applied)"
        return (
            f"{self.n_selected:,} of {self.n_total:,} customers "
            f"({self.share_pct:.1f}%) match {len(self.applied)} filter(s)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for logging and the AI payload."""
        return {
            "n_total": self.n_total,
            "n_selected": self.n_selected,
            "share_pct": round(self.share_pct, 4),
            "is_filtered": self.is_filtered,
            "applied": list(self.applied),
            "skipped": list(self.skipped),
            "spec": self.spec.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Bounds discovery
# --------------------------------------------------------------------------- #


def _category_options(series: pd.Series) -> tuple[str, ...]:
    """Sorted distinct labels present in a categorical column."""
    values = series.dropna().astype("string").unique().tolist()
    return tuple(sorted(str(v) for v in values))


def derive_bounds(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> FilterBounds:
    """Read the selectable range of every filter from the data.

    Args:
        df: Analysis-ready dataframe.
        registry: Optional pre-built registry.

    Returns:
        A :class:`FilterBounds` describing what can be filtered.
    """
    reg = registry or build_registry(df)

    if df.empty:
        return FilterBounds()

    age_column = reg.one(Role.AGE)
    if age_column and age_column in df.columns:
        ages = pd.to_numeric(df[age_column], errors="coerce").dropna()
        age_min = int(ages.min()) if not ages.empty else 0
        age_max = int(ages.max()) if not ages.empty else 100
    else:
        age_min, age_max = 0, 0

    limit_column = reg.one(Role.CREDIT_LIMIT)
    if limit_column and limit_column in df.columns:
        limits = pd.to_numeric(df[limit_column], errors="coerce").dropna()
        limit_min = float(limits.min()) if not limits.empty else 0.0
        limit_max = float(limits.max()) if not limits.empty else 0.0
    else:
        limit_min, limit_max = 0.0, 0.0

    if "utilisation_latest" in df.columns:
        utilisation = (
            pd.to_numeric(df["utilisation_latest"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        # Clip the upper bound at the 99.5th percentile: a handful of extreme
        # over-limit accounts would otherwise stretch the slider so far that the
        # useful 0-100% region becomes impossible to select.
        util_min = float(np.floor(utilisation.min() * 100) / 100) if not utilisation.empty else 0.0
        util_max = (
            float(np.ceil(utilisation.quantile(0.995) * 100) / 100)
            if not utilisation.empty
            else 1.0
        )
        if util_max <= util_min:
            util_max = util_min + 1.0
    else:
        util_min, util_max = 0.0, 0.0

    return FilterBounds(
        age_min=age_min,
        age_max=age_max,
        limit_min=limit_min,
        limit_max=limit_max,
        utilisation_min=util_min,
        utilisation_max=util_max,
        education_options=(
            _category_options(df["education"]) if "education" in df.columns else ()
        ),
        marriage_options=(
            _category_options(df["marriage"]) if "marriage" in df.columns else ()
        ),
        sex_options=_category_options(df["sex"]) if "sex" in df.columns else (),
    )


def default_spec(bounds: FilterBounds) -> FilterSpec:
    """Build the unconstrained specification matching the supplied bounds."""
    return FilterSpec(
        age_range=(bounds.age_min, bounds.age_max) if bounds.has_age else None,
        limit_range=(bounds.limit_min, bounds.limit_max) if bounds.has_limit else None,
        utilisation_range=(
            (bounds.utilisation_min, bounds.utilisation_max)
            if bounds.has_utilisation
            else None
        ),
        education=bounds.education_options or None,
        marriage=bounds.marriage_options or None,
        sex=bounds.sex_options or None,
        delinquency="all",
    )


def is_default(spec: FilterSpec, bounds: FilterBounds) -> bool:
    """True when the specification imposes no real constraint.

    Compares against the data-derived bounds, so selecting every option counts as
    unfiltered rather than as an active filter.
    """
    return spec == default_spec(bounds) or spec == FilterSpec()


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


def _range_covers(
    selected: tuple[float, float] | None, low: float, high: float
) -> bool:
    """True when a selected range spans the full available range."""
    if selected is None:
        return True
    return selected[0] <= low and selected[1] >= high


def apply_filters(
    df: pd.DataFrame,
    spec: FilterSpec,
    registry: ColumnRegistry | None = None,
    bounds: FilterBounds | None = None,
) -> FilterResult:
    """Apply a filter specification to the data.

    A constraint is recorded in ``applied`` only when it genuinely narrows the
    selection, and in ``skipped`` when the required column is absent. That makes
    the UI able to say precisely what is in effect.

    Args:
        df: Analysis-ready dataframe.
        spec: The selection to apply.
        registry: Optional pre-built registry.
        bounds: Optional pre-computed bounds, used to recognise full-range
            selections as unfiltered.

    Returns:
        A :class:`FilterResult`.
    """
    reg = registry or build_registry(df)
    resolved_bounds = bounds or derive_bounds(df, reg)

    n_total = int(len(df))
    if n_total == 0:
        return FilterResult(data=df, spec=spec, n_total=0, n_selected=0)

    mask = pd.Series(True, index=df.index)
    applied: list[str] = []
    skipped: list[str] = []

    # --- age ---
    age_column = reg.one(Role.AGE)
    if spec.age_range is not None:
        if age_column and age_column in df.columns:
            if not _range_covers(
                spec.age_range, resolved_bounds.age_min, resolved_bounds.age_max
            ):
                ages = pd.to_numeric(df[age_column], errors="coerce")
                mask &= ages.between(spec.age_range[0], spec.age_range[1])
                applied.append(f"Age {spec.age_range[0]}-{spec.age_range[1]}")
        else:
            skipped.append("Age (column unavailable)")

    # --- credit limit ---
    limit_column = reg.one(Role.CREDIT_LIMIT)
    if spec.limit_range is not None:
        if limit_column and limit_column in df.columns:
            if not _range_covers(
                spec.limit_range, resolved_bounds.limit_min, resolved_bounds.limit_max
            ):
                limits = pd.to_numeric(df[limit_column], errors="coerce")
                mask &= limits.between(spec.limit_range[0], spec.limit_range[1])
                applied.append(
                    f"Credit limit NT${spec.limit_range[0]:,.0f}-{spec.limit_range[1]:,.0f}"
                )
        else:
            skipped.append("Credit limit (column unavailable)")

    # --- utilisation ---
    if spec.utilisation_range is not None:
        if "utilisation_latest" in df.columns:
            if not _range_covers(
                spec.utilisation_range,
                resolved_bounds.utilisation_min,
                resolved_bounds.utilisation_max,
            ):
                utilisation = pd.to_numeric(
                    df["utilisation_latest"], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)
                # NaN utilisation (no usable credit limit) cannot satisfy a
                # numeric range, so those rows drop out - which is correct.
                mask &= utilisation.between(
                    spec.utilisation_range[0], spec.utilisation_range[1]
                )
                applied.append(
                    f"Utilisation {spec.utilisation_range[0]:.0%}-"
                    f"{spec.utilisation_range[1]:.0%}"
                )
        else:
            skipped.append("Utilisation (feature unavailable)")

    # --- categorical dimensions ---
    for column, selected, label, available in (
        ("education", spec.education, "Education", resolved_bounds.education_options),
        ("marriage", spec.marriage, "Marital status", resolved_bounds.marriage_options),
        ("sex", spec.sex, "Sex", resolved_bounds.sex_options),
    ):
        if selected is None:
            continue
        if column not in df.columns:
            skipped.append(f"{label} (column unavailable)")
            continue
        chosen = set(selected)
        if chosen and chosen != set(available):
            mask &= df[column].astype("string").isin(list(chosen))
            applied.append(f"{label}: {', '.join(sorted(chosen))}")
        elif not chosen:
            # An explicitly empty selection means "nothing", which is a valid
            # (if unhelpful) request and must be honoured rather than ignored.
            mask &= pd.Series(False, index=df.index)
            applied.append(f"{label}: none selected")

    # --- delinquency status ---
    if spec.delinquency and spec.delinquency != "all":
        current_column = "is_currently_delinquent"
        ever_column = "ever_delinquent"
        if spec.delinquency == "current" and current_column in df.columns:
            mask &= df[current_column].fillna(False).astype("bool")
            applied.append("Currently behind on payment")
        elif spec.delinquency == "ever" and ever_column in df.columns:
            mask &= df[ever_column].fillna(False).astype("bool")
            applied.append("Ever behind on payment")
        elif spec.delinquency == "never" and ever_column in df.columns:
            mask &= ~df[ever_column].fillna(False).astype("bool")
            applied.append("Never behind on payment")
        else:
            skipped.append("Delinquency status (feature unavailable)")

    filtered = df.loc[mask]

    result = FilterResult(
        data=filtered,
        spec=spec,
        n_total=n_total,
        n_selected=int(len(filtered)),
        applied=tuple(applied),
        skipped=tuple(skipped),
    )

    if result.is_filtered:
        logger.info(
            "Filters applied: %d of %d rows selected (%.1f%%) via %s",
            result.n_selected,
            result.n_total,
            result.share_pct,
            "; ".join(result.applied),
        )
    if result.skipped:
        logger.warning("Filters skipped: %s", "; ".join(result.skipped))

    return result


# --------------------------------------------------------------------------- #
# Utilisation banding
# --------------------------------------------------------------------------- #


def assign_utilisation_band(
    df: pd.DataFrame,
    column: str = "utilisation_latest",
    high_threshold: float | None = None,
) -> pd.Series:
    """Assign each row to a utilisation band.

    Band edges come from :data:`UTILISATION_BANDS`, whose top boundary is the
    configured ``HIGH_UTILISATION_THRESHOLD`` rather than a number written into
    this function. Changing the setting changes the band.

    Args:
        df: Dataframe containing the utilisation column.
        column: Utilisation column to band.
        high_threshold: Override for the high-utilisation cut-off.

    Returns:
        An ordered categorical Series; all-NaN when the column is absent.
    """
    from src.config import settings

    threshold = (
        high_threshold if high_threshold is not None else settings.high_utilisation_threshold
    )

    if column not in df.columns:
        return pd.Series(
            pd.Categorical([None] * len(df), categories=[b[0] for b in UTILISATION_BANDS]),
            index=df.index,
            name="utilisation_band",
        )

    values = (
        pd.to_numeric(df[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    # Rebuild the edges so the final boundary honours the configured threshold.
    bands = [
        ("Negative / zero", -np.inf, 0.0001),
        ("Low (0-30%)", 0.0001, 0.30),
        ("Moderate (30-60%)", 0.30, 0.60),
        (f"High (60-{threshold:.0%})", 0.60, threshold),
        (f"Very high ({threshold:.0%}+)", threshold, np.inf),
    ]
    labels = [name for name, _, _ in bands]

    assigned = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, lower, upper in bands:
        in_band = (values >= lower) & (values < upper)
        assigned = assigned.mask(in_band.fillna(False), name)

    return pd.Series(
        pd.Categorical(assigned, categories=labels, ordered=True),
        index=df.index,
        name="utilisation_band",
    )


def utilisation_band_documentation(high_threshold: float | None = None) -> str:
    """Explain the utilisation banding rule for display.

    Args:
        high_threshold: Override for the high-utilisation cut-off.

    Returns:
        A short methodology note.
    """
    from src.config import settings

    threshold = (
        high_threshold if high_threshold is not None else settings.high_utilisation_threshold
    )
    return (
        "Utilisation is the most recent statement balance divided by the credit "
        "limit, a derived analytical measure. Bands are fixed rather than "
        "quantile-based so they stay comparable across any filtered selection: "
        "negative or zero, low (below 30%), moderate (30-60%), high "
        f"(60-{threshold:.0%}) and very high ({threshold:.0%} or more). The top "
        "boundary is the configured HIGH_UTILISATION_THRESHOLD "
        f"({threshold:.2f}), not a value hard-coded into the chart. A negative "
        "band exists because an account in credit produces negative utilisation, "
        "which is valid data."
    )


def limit_band_documentation() -> str:
    """Explain the credit-limit banding rule for display."""
    return (
        "Credit-limit bands are quartiles of the observed credit limit, so each "
        "band holds roughly a quarter of customers and the boundaries reflect this "
        "portfolio rather than an external standard. The label shows the actual "
        "NT$ range. These are CREDIT LIMITS, not income and not wealth: the "
        "dataset contains no income variable, so the bands describe granted credit "
        "capacity only and imply nothing about socioeconomic class."
    )
