"""
Data cleaning and reshaping.

Guiding principles
------------------
1. **The raw file is never modified.** Every function returns a new dataframe;
   results are persisted separately under ``data/processed``.
2. **Preserve information.** Rows are not dropped to make the data look tidy.
   Suspect values are *flagged* with an extra boolean column so an analyst can
   include or exclude them deliberately. Original coded columns survive
   alongside their human-readable labels.
3. **Every decision is recorded.** Each step appends a :class:`CleaningStep`
   carrying what changed, how many rows it touched, and the reasoning. The
   resulting :class:`CleaningReport` is what the Methodology page renders, so the
   cleaning narrative is generated rather than hand-written.
4. **Reproducible.** No randomness, no dependence on row order. The same input
   always produces the same output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import settings
from src.logging_setup import get_logger
from src.schema import (
    DELINQUENCY_THRESHOLD,
    EDUCATION_LABELS,
    ID_COLUMN,
    MARRIAGE_LABELS,
    SEX_LABELS,
    UNKNOWN_LABEL,
    ColumnRegistry,
    Role,
    build_registry,
    month_label,
)

logger = get_logger(__name__)

PROCESSED_WIDE_FILENAME: Final[str] = "clients_wide.parquet"
PROCESSED_PANEL_FILENAME: Final[str] = "panel_long.parquet"

MissingNumericStrategy = Literal["median", "mean", "zero", "leave"]
MissingCategoricalStrategy = Literal["unknown", "mode", "leave"]

#: Plausible age range for a credit-card holder. Outside this, values are
#: flagged (never deleted) and excluded from age-band analysis.
MIN_PLAUSIBLE_AGE: Final[int] = 18
MAX_PLAUSIBLE_AGE: Final[int] = 100


# --------------------------------------------------------------------------- #
# Step / report structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CleaningStep:
    """Record of one cleaning operation.

    ``rationale`` exists so the generated methodology explains *why* a decision
    was taken, not merely that it happened.
    """

    name: str
    description: str
    rationale: str
    rows_before: int
    rows_after: int
    affected_rows: int = 0
    columns_added: tuple[str, ...] = field(default=())
    columns_modified: tuple[str, ...] = field(default=())
    columns_dropped: tuple[str, ...] = field(default=())
    action_taken: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        """Rows removed by this step (zero for non-destructive steps)."""
        return self.rows_before - self.rows_after

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the UI and report."""
        return {
            "name": self.name,
            "description": self.description,
            "rationale": self.rationale,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_removed": self.rows_removed,
            "affected_rows": self.affected_rows,
            "columns_added": list(self.columns_added),
            "columns_modified": list(self.columns_modified),
            "columns_dropped": list(self.columns_dropped),
            "action_taken": self.action_taken,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CleaningReport:
    """The full audit trail of a cleaning run."""

    steps: tuple[CleaningStep, ...]
    original_rows: int
    original_columns: int
    final_rows: int
    final_columns: int

    @property
    def rows_removed(self) -> int:
        """Net rows removed across the whole run."""
        return self.original_rows - self.final_rows

    @property
    def columns_added(self) -> tuple[str, ...]:
        """Every column added, in order."""
        added: list[str] = []
        for step in self.steps:
            added.extend(step.columns_added)
        return tuple(added)

    @property
    def steps_with_action(self) -> tuple[CleaningStep, ...]:
        """Only the steps that actually changed something."""
        return tuple(step for step in self.steps if step.action_taken)

    def summary(self) -> str:
        """One-line human-readable outcome."""
        return (
            f"{len(self.steps_with_action)} of {len(self.steps)} cleaning steps made "
            f"changes. Rows: {self.original_rows:,} -> {self.final_rows:,} "
            f"({self.rows_removed:,} removed). "
            f"Columns: {self.original_columns} -> {self.final_columns}."
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the UI, report and AI payload."""
        return {
            "summary": self.summary(),
            "original_shape": [self.original_rows, self.original_columns],
            "final_shape": [self.final_rows, self.final_columns],
            "rows_removed": self.rows_removed,
            "columns_added": list(self.columns_added),
            "steps": [step.to_dict() for step in self.steps],
        }


# --------------------------------------------------------------------------- #
# Individual cleaning operations
# --------------------------------------------------------------------------- #


def convert_dtypes(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> tuple[pd.DataFrame, CleaningStep]:
    """Coerce columns that must be numeric into a numeric dtype.

    Unparseable entries become ``NaN`` rather than causing a failure, so one bad
    cell from a spreadsheet export cannot break the pipeline.

    Args:
        df: Input dataframe.
        registry: Optional pre-built registry.

    Returns:
        ``(dataframe, step)``.
    """
    reg = registry or build_registry(df)
    result = df.copy()
    rows_before = len(result)

    numeric_roles = (
        Role.CLIENT_ID,
        Role.CREDIT_LIMIT,
        Role.AGE,
        Role.BILL_AMOUNT,
        Role.PAYMENT_AMOUNT,
        Role.PAYMENT_STATUS,
        Role.TARGET,
        Role.SEX,
        Role.EDUCATION,
        Role.MARRIAGE,
    )
    targets: list[str] = []
    for role in numeric_roles:
        targets.extend(reg.all(role))

    converted: list[str] = []
    newly_missing = 0

    for column in dict.fromkeys(targets):  # de-duplicate, preserve order
        if column not in result.columns:
            continue
        if pd.api.types.is_numeric_dtype(result[column]):
            continue
        before_missing = int(result[column].isna().sum())
        coerced = pd.to_numeric(result[column], errors="coerce")
        after_missing = int(coerced.isna().sum())
        result[column] = coerced
        converted.append(column)
        newly_missing += after_missing - before_missing

    step = CleaningStep(
        name="convert_dtypes",
        description=(
            f"Converted {len(converted)} column(s) to a numeric dtype."
            if converted
            else "All columns expected to be numeric already had a numeric dtype; no conversion needed."
        ),
        rationale=(
            "Spreadsheet exports often store numbers as text when a stray space or "
            "thousands separator is present, which silently breaks arithmetic. "
            "Unparseable entries are converted to missing values rather than dropped, "
            "so no row is lost to a single bad cell."
        ),
        rows_before=rows_before,
        rows_after=len(result),
        affected_rows=newly_missing,
        columns_modified=tuple(converted),
        action_taken=bool(converted),
        details={"converted_columns": converted, "values_made_missing": newly_missing},
    )
    logger.info("convert_dtypes: %s", step.description)
    return result, step


def remove_duplicates(
    df: pd.DataFrame,
    drop_exact: bool = True,
    drop_excluding_id: bool = False,
) -> tuple[pd.DataFrame, CleaningStep]:
    """Handle duplicate rows.

    Exact duplicates (identical in every column, identifier included) carry no
    information and are removed. Rows that match only once the surrogate
    ``client_id`` is dropped are **kept by default** and flagged instead: with
    23 integer-coded columns, a coincidental match between two genuinely
    different clients is entirely plausible, so deleting them could discard real
    customers.

    Args:
        df: Input dataframe.
        drop_exact: Remove fully identical rows.
        drop_excluding_id: Also remove rows duplicated once ``client_id`` is
            excluded. Off by default.

    Returns:
        ``(dataframe, step)``.
    """
    result = df.copy()
    rows_before = len(result)
    added: list[str] = []
    exact_removed = 0
    id_dupes_removed = 0

    if drop_exact and rows_before:
        exact_mask = result.duplicated(keep="first")
        exact_removed = int(exact_mask.sum())
        if exact_removed:
            result = result.loc[~exact_mask].copy()

    duplicate_flag_count = 0
    if ID_COLUMN in result.columns and len(result.columns) > 1 and len(result):
        without_id = result.drop(columns=[ID_COLUMN])
        dupe_mask = without_id.duplicated(keep="first")
        duplicate_flag_count = int(dupe_mask.sum())

        if drop_excluding_id and duplicate_flag_count:
            result = result.loc[~dupe_mask].copy()
            id_dupes_removed = duplicate_flag_count
        elif duplicate_flag_count:
            # Flag every member of each duplicate group, not just the repeats,
            # so an analyst can inspect the whole group.
            all_members = without_id.duplicated(keep=False)
            result["is_duplicate_excluding_id"] = all_members.to_numpy()
            added.append("is_duplicate_excluding_id")

    if exact_removed or id_dupes_removed:
        description = (
            f"Removed {exact_removed:,} exact duplicate row(s)"
            + (f" and {id_dupes_removed:,} row(s) duplicated excluding the identifier." if id_dupes_removed else ".")
        )
    elif duplicate_flag_count:
        description = (
            f"No exact duplicates found. Flagged {duplicate_flag_count:,} row(s) that "
            "are identical once the identifier is excluded, retaining them for analysis."
        )
    else:
        description = "No duplicate rows found by either definition; no action taken."

    step = CleaningStep(
        name="remove_duplicates",
        description=description,
        rationale=(
            "Exact duplicates are removed because they add no information and would "
            "double-count a client. Rows that match only after dropping the surrogate "
            "key are retained and flagged instead: with 23 integer-coded columns a "
            "coincidental match between two real clients is plausible, and deleting "
            "them risks discarding genuine customers. The flag lets the analyst decide."
        ),
        rows_before=rows_before,
        rows_after=len(result),
        affected_rows=exact_removed + id_dupes_removed + duplicate_flag_count,
        columns_added=tuple(added),
        action_taken=bool(exact_removed or id_dupes_removed or duplicate_flag_count),
        details={
            "exact_duplicates_removed": exact_removed,
            "id_excluded_duplicates_removed": id_dupes_removed,
            "id_excluded_duplicates_flagged": duplicate_flag_count if not drop_excluding_id else 0,
        },
    )
    logger.info("remove_duplicates: %s", step.description)
    return result, step


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: MissingNumericStrategy = "median",
    categorical_strategy: MissingCategoricalStrategy = "unknown",
    add_indicators: bool = True,
) -> tuple[pd.DataFrame, CleaningStep]:
    """Impute missing values, recording where imputation happened.

    Rows are never dropped for being incomplete. When a column is imputed an
    optional ``<column>_was_missing`` indicator is added so no imputed value is
    ever mistaken for an observed one.

    The reference dataset has no missing values, so this step normally reports
    "no action needed". It is implemented properly regardless, because the
    platform must behave correctly on a filtered subset or a similar dataset.

    Args:
        df: Input dataframe.
        numeric_strategy: How to fill numeric columns. ``median`` resists skew,
            which matters for financial amounts.
        categorical_strategy: How to fill categorical columns.
        add_indicators: Add a boolean indicator per imputed column.

    Returns:
        ``(dataframe, step)``.
    """
    result = df.copy()
    rows_before = len(result)

    missing_counts = {
        str(column): int(result[column].isna().sum()) for column in result.columns
    }
    columns_with_missing = {
        column: count for column, count in missing_counts.items() if count > 0
    }

    if not columns_with_missing:
        step = CleaningStep(
            name="handle_missing_values",
            description="No missing values present; no imputation performed.",
            rationale=(
                "The dataset is fully populated, so imputation would introduce "
                "synthetic values with no benefit. The imputation logic remains "
                "available for filtered subsets or a different source dataset."
            ),
            rows_before=rows_before,
            rows_after=len(result),
            action_taken=False,
            details={"columns_with_missing": {}},
        )
        logger.info("handle_missing_values: %s", step.description)
        return result, step

    added: list[str] = []
    modified: list[str] = []
    imputation_detail: dict[str, Any] = {}

    for column, count in columns_with_missing.items():
        series = result[column]

        # A fully empty column cannot be imputed from itself.
        if count == rows_before:
            imputation_detail[column] = {"missing": count, "action": "left as missing (column is entirely empty)"}
            continue

        if add_indicators:
            indicator = f"{column}_was_missing"
            result[indicator] = series.isna().to_numpy()
            added.append(indicator)

        if pd.api.types.is_numeric_dtype(series):
            if numeric_strategy == "leave":
                imputation_detail[column] = {"missing": count, "action": "left as missing"}
                continue
            if numeric_strategy == "median":
                fill_value: Any = float(series.median())
            elif numeric_strategy == "mean":
                fill_value = float(series.mean())
            else:
                fill_value = 0
            result[column] = series.fillna(fill_value)
            modified.append(column)
            imputation_detail[column] = {
                "missing": count,
                "action": f"filled with {numeric_strategy}",
                "fill_value": fill_value,
            }
        else:
            if categorical_strategy == "leave":
                imputation_detail[column] = {"missing": count, "action": "left as missing"}
                continue
            if categorical_strategy == "mode":
                modes = series.mode(dropna=True)
                fill_value = modes.iloc[0] if not modes.empty else UNKNOWN_LABEL
            else:
                fill_value = UNKNOWN_LABEL
            result[column] = series.fillna(fill_value)
            modified.append(column)
            imputation_detail[column] = {
                "missing": count,
                "action": f"filled with {categorical_strategy}",
                "fill_value": str(fill_value),
            }

    total_imputed = sum(columns_with_missing.values())
    step = CleaningStep(
        name="handle_missing_values",
        description=(
            f"Imputed {total_imputed:,} missing value(s) across "
            f"{len(columns_with_missing)} column(s). Numeric columns used the "
            f"'{numeric_strategy}' strategy; categorical columns used "
            f"'{categorical_strategy}'."
        ),
        rationale=(
            "Rows are never dropped for incompleteness, since that would bias the "
            "sample toward clients with fuller records. The median is preferred for "
            "numeric columns because financial amounts are strongly right-skewed and "
            "the mean would be pulled by extreme values. An indicator column marks "
            "every imputed cell so imputed values are never mistaken for observed ones."
        ),
        rows_before=rows_before,
        rows_after=len(result),
        affected_rows=total_imputed,
        columns_added=tuple(added),
        columns_modified=tuple(modified),
        action_taken=True,
        details={"columns_with_missing": imputation_detail},
    )
    logger.info("handle_missing_values: %s", step.description)
    return result, step


def normalise_categoricals(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> tuple[pd.DataFrame, CleaningStep]:
    """Turn integer category codes into readable labels.

    New label columns are added; the original ``*_code`` columns are kept so
    nothing is lost and any mapping decision stays auditable.

    Codes absent from the source documentation are mapped to a single
    ``"Unknown"`` category rather than guessed at or discarded. Guessing would
    fabricate information; discarding would throw away otherwise valid rows.

    Args:
        df: Input dataframe.
        registry: Optional pre-built registry.

    Returns:
        ``(dataframe, step)``.
    """
    reg = registry or build_registry(df)
    result = df.copy()
    rows_before = len(result)
    added: list[str] = []
    mapping_detail: dict[str, Any] = {}

    specs = (
        ("sex_code", "sex", SEX_LABELS, Role.SEX),
        ("education_code", "education", EDUCATION_LABELS, Role.EDUCATION),
        ("marriage_code", "marriage", MARRIAGE_LABELS, Role.MARRIAGE),
    )

    for code_column, label_column, labels, _role in specs:
        if code_column not in result.columns:
            continue

        codes = pd.to_numeric(result[code_column], errors="coerce")
        mapped = codes.map(labels)
        unmapped_mask = mapped.isna() & codes.notna()
        n_unmapped = int(unmapped_mask.sum())

        unmapped_codes = (
            codes[unmapped_mask].astype("int64").value_counts().sort_index().to_dict()
            if n_unmapped
            else {}
        )

        labelled = mapped.fillna(UNKNOWN_LABEL).astype("string")
        # Ordered category keeps charts and tables in a stable, sensible order.
        categories = [labels[key] for key in sorted(labels)] + [UNKNOWN_LABEL]
        result[label_column] = pd.Categorical(labelled, categories=categories, ordered=False)
        added.append(label_column)

        mapping_detail[code_column] = {
            "label_column": label_column,
            "documented_codes": {str(k): v for k, v in sorted(labels.items())},
            "undocumented_codes_mapped_to_unknown": {
                str(k): int(v) for k, v in unmapped_codes.items()
            },
            "rows_marked_unknown": n_unmapped,
        }

    total_unknown = sum(
        int(detail["rows_marked_unknown"]) for detail in mapping_detail.values()
    )

    step = CleaningStep(
        name="normalise_categoricals",
        description=(
            f"Created {len(added)} readable label column(s) from integer codes "
            f"({', '.join(added)}). {total_unknown:,} row(s) carried codes absent from "
            f"the source documentation and were labelled '{UNKNOWN_LABEL}'."
            if added
            else "No coded categorical columns found; no action taken."
        ),
        rationale=(
            "Integer codes are unreadable in a dashboard, so labelled columns are "
            "added while the original codes are retained for auditability. Codes that "
            "the source documentation does not define are grouped into a single "
            f"'{UNKNOWN_LABEL}' category: guessing a meaning would fabricate "
            "information, and dropping the rows would discard otherwise valid clients."
        ),
        rows_before=rows_before,
        rows_after=len(result),
        affected_rows=total_unknown,
        columns_added=tuple(added),
        action_taken=bool(added),
        details=mapping_detail,
    )
    logger.info("normalise_categoricals: %s", step.description)
    return result, step


def handle_invalid_numeric_values(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> tuple[pd.DataFrame, CleaningStep]:
    """Flag implausible numeric values and neutralise unusable denominators.

    Two different treatments, for two different situations:

    * **Implausible but usable** (an age outside 18-100): a boolean flag is
      added and the value is left untouched, so the analyst can exclude it.
    * **Unusable as a denominator** (a credit limit of zero or less): the value
      is set to ``NaN`` so every ratio built on it becomes ``NaN`` rather than
      ``inf`` or a divide-by-zero error. The original is preserved in a
      ``*_original`` column.

    Args:
        df: Input dataframe.
        registry: Optional pre-built registry.

    Returns:
        ``(dataframe, step)``.
    """
    reg = registry or build_registry(df)
    result = df.copy()
    rows_before = len(result)
    added: list[str] = []
    modified: list[str] = []
    detail: dict[str, Any] = {}
    affected = 0

    # --- age: flag, keep ---
    age_column = reg.one(Role.AGE)
    if age_column and age_column in result.columns:
        ages = pd.to_numeric(result[age_column], errors="coerce")
        implausible = (ages < MIN_PLAUSIBLE_AGE) | (ages > MAX_PLAUSIBLE_AGE)
        count = int(implausible.sum())
        if count:
            result["age_is_implausible"] = implausible.fillna(False).to_numpy()
            added.append("age_is_implausible")
            affected += count
        detail["age"] = {
            "plausible_range": [MIN_PLAUSIBLE_AGE, MAX_PLAUSIBLE_AGE],
            "observed_min": float(ages.min()) if ages.notna().any() else None,
            "observed_max": float(ages.max()) if ages.notna().any() else None,
            "rows_flagged": count,
            "action": "flagged only, values retained",
        }

    # --- credit limit: neutralise a zero/negative denominator ---
    limit_column = reg.one(Role.CREDIT_LIMIT)
    if limit_column and limit_column in result.columns:
        limits = pd.to_numeric(result[limit_column], errors="coerce")
        non_positive = limits <= 0
        count = int(non_positive.sum())
        if count:
            result[f"{limit_column}_original"] = limits.to_numpy()
            result[limit_column] = limits.where(~non_positive, other=np.nan)
            added.append(f"{limit_column}_original")
            modified.append(limit_column)
            affected += count
        detail["credit_limit"] = {
            "rows_non_positive": count,
            "action": (
                "set to missing so dependent ratios are undefined rather than "
                "dividing by zero; original value preserved in "
                f"'{limit_column}_original'"
                if count
                else "no non-positive limits found"
            ),
        }

    # --- negative statement balances: flag, keep (they are legitimate) ---
    bill_columns = reg.all(Role.BILL_AMOUNT)
    if bill_columns:
        negative_any = pd.Series(False, index=result.index)
        total_cells = 0
        for column in bill_columns:
            mask = pd.to_numeric(result[column], errors="coerce") < 0
            total_cells += int(mask.sum())
            negative_any |= mask.fillna(False)
        count = int(negative_any.sum())
        if count:
            result["has_negative_balance"] = negative_any.to_numpy()
            added.append("has_negative_balance")
        detail["negative_statement_balance"] = {
            "distinct_clients": count,
            "client_month_cells": total_cells,
            "action": (
                "retained unchanged and flagged; a negative balance means the account "
                "is in credit (overpayment or refund), which is valid data"
            ),
        }

    step = CleaningStep(
        name="handle_invalid_numeric_values",
        description=(
            f"Reviewed numeric ranges and added {len(added)} flag/preservation "
            f"column(s): {', '.join(added)}."
            if added
            else "All numeric values fell within their expected ranges; no action taken."
        ),
        rationale=(
            "Implausible values are flagged rather than deleted so the analyst keeps "
            "control over inclusion. The one exception is a non-positive credit limit, "
            "which is set to missing because it cannot serve as a denominator: this "
            "makes dependent ratios explicitly undefined instead of producing infinity "
            "or a runtime error. Negative statement balances are genuine credit "
            "balances and are preserved untouched."
        ),
        rows_before=rows_before,
        rows_after=len(result),
        affected_rows=affected,
        columns_added=tuple(added),
        columns_modified=tuple(modified),
        action_taken=bool(added or modified),
        details=detail,
    )
    logger.info("handle_invalid_numeric_values: %s", step.description)
    return result, step


def flag_extreme_outliers(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    iqr_multiplier: float = 3.0,
    registry: ColumnRegistry | None = None,
) -> tuple[pd.DataFrame, CleaningStep]:
    """Flag extreme values on monetary columns without altering them.

    Uses a 3.0x IQR fence (the "extreme" convention) rather than the usual 1.5x,
    because credit balances are strongly right-skewed and the tighter fence
    flags a large share of perfectly ordinary accounts.

    No value is winsorised, capped or removed. A NT$1,000,000 credit limit is a
    real premium customer, not a data error, and truncating it would destroy the
    signal that matters most for risk.

    Args:
        df: Input dataframe.
        columns: Columns to assess. Defaults to credit limit plus the statement
            and payment panel columns.
        iqr_multiplier: Fence multiplier.
        registry: Optional pre-built registry.

    Returns:
        ``(dataframe, step)``.
    """
    reg = registry or build_registry(df)
    result = df.copy()
    rows_before = len(result)

    if columns is None:
        targets = list(reg.all(Role.CREDIT_LIMIT)) + list(reg.all(Role.BILL_AMOUNT)) + list(
            reg.all(Role.PAYMENT_AMOUNT)
        )
    else:
        targets = [column for column in columns if column in result.columns]

    extreme_any = pd.Series(False, index=result.index)
    per_column: dict[str, int] = {}

    for column in targets:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        valid = values.dropna()
        if valid.empty or valid.nunique() <= 2:
            continue
        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask = ((values < lower) | (values > upper)).fillna(False)
        count = int(mask.sum())
        if count:
            per_column[column] = count
            extreme_any |= mask

    added: list[str] = []
    n_flagged = int(extreme_any.sum())
    if n_flagged:
        result["has_extreme_value"] = extreme_any.to_numpy()
        added.append("has_extreme_value")

    step = CleaningStep(
        name="flag_extreme_outliers",
        description=(
            f"Flagged {n_flagged:,} client(s) holding at least one extreme value "
            f"across {len(per_column)} monetary column(s), using a "
            f"{iqr_multiplier}x IQR fence. No values were altered."
            if n_flagged
            else "No extreme values found under the configured fence; no action taken."
        ),
        rationale=(
            f"A {iqr_multiplier}x IQR fence is used instead of the conventional 1.5x "
            "because credit balances are strongly right-skewed, and the tighter fence "
            "would flag a large share of ordinary accounts. Nothing is capped or "
            "removed: a very high credit limit is a real premium customer, not an "
            "error, and truncating it would erase exactly the signal that matters for "
            "risk. The flag simply makes these clients easy to isolate."
        ),
        rows_before=rows_before,
        rows_after=len(result),
        affected_rows=n_flagged,
        columns_added=tuple(added),
        action_taken=bool(n_flagged),
        details={"iqr_multiplier": iqr_multiplier, "flagged_by_column": per_column},
    )
    logger.info("flag_extreme_outliers: %s", step.description)
    return result, step


# --------------------------------------------------------------------------- #
# Panel reshape
# --------------------------------------------------------------------------- #


def build_panel_long(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> pd.DataFrame:
    """Reshape the monthly wide columns into a tidy long panel.

    Turns one row per client with 18 monthly columns into one row per
    client-month, which is the natural shape for trend analysis and for plotting
    time on an axis.

    Args:
        df: Cleaned wide dataframe.
        registry: Optional pre-built registry.

    Returns:
        A long dataframe with columns ``client_id``, ``month_index``,
        ``month_label``, ``month_order``, ``bill_amt``, ``pay_amt``,
        ``pay_status``, ``is_delinquent`` and ``utilisation`` where computable.
        Empty (with the right columns) when no panel columns exist.
    """
    reg = registry or build_registry(df)

    bills = reg.panel_columns(Role.BILL_AMOUNT)
    payments = reg.panel_columns(Role.PAYMENT_AMOUNT)
    statuses = reg.panel_columns(Role.PAYMENT_STATUS)

    months = sorted(set(bills) | set(payments) | set(statuses))
    expected_columns = [
        "client_id",
        "month_index",
        "month_label",
        "month_order",
        "bill_amt",
        "pay_amt",
        "pay_status",
        "is_delinquent",
        "utilisation",
    ]

    if not months:
        logger.warning(
            "No monthly panel columns found; returning an empty panel. Trend "
            "analysis will be unavailable."
        )
        return pd.DataFrame({column: pd.Series(dtype="float64") for column in expected_columns})

    id_series = (
        df[ID_COLUMN]
        if ID_COLUMN in df.columns
        else pd.Series(df.index, index=df.index, name=ID_COLUMN)
    )
    limit_column = reg.one(Role.CREDIT_LIMIT)
    limits = (
        pd.to_numeric(df[limit_column], errors="coerce")
        if limit_column and limit_column in df.columns
        else None
    )

    frames: list[pd.DataFrame] = []
    n_months = len(months)

    for month in months:
        block = pd.DataFrame({"client_id": id_series.to_numpy()})
        block["month_index"] = month
        block["month_label"] = month_label(month)
        # month_order ascends with calendar time so charts read left-to-right.
        block["month_order"] = n_months - month + 1

        for name, mapping in (
            ("bill_amt", bills),
            ("pay_amt", payments),
            ("pay_status", statuses),
        ):
            column = mapping.get(month)
            block[name] = (
                pd.to_numeric(df[column], errors="coerce").to_numpy()
                if column
                else np.nan
            )

        block["is_delinquent"] = block["pay_status"] >= DELINQUENCY_THRESHOLD
        block["utilisation"] = (
            (block["bill_amt"] / limits.to_numpy()) if limits is not None else np.nan
        )
        frames.append(block)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["client_id", "month_order"]).reset_index(drop=True)

    logger.info(
        "Built long panel: %d client-month rows across %d month(s) for %d client(s).",
        len(panel),
        n_months,
        int(panel["client_id"].nunique()),
    )
    return panel


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def clean_dataset(
    df: pd.DataFrame,
    drop_exact_duplicates: bool = True,
    drop_id_excluded_duplicates: bool = False,
    numeric_strategy: MissingNumericStrategy = "median",
    categorical_strategy: MissingCategoricalStrategy = "unknown",
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning sequence, returning the result and an audit trail.

    Order matters: dtypes are fixed first so later numeric checks are reliable;
    duplicates go next to avoid imputing into rows about to be removed;
    imputation follows; then labelling, range handling and outlier flagging.

    Args:
        df: Normalised dataframe from the loader.
        drop_exact_duplicates: Remove fully identical rows.
        drop_id_excluded_duplicates: Also remove rows duplicated once the
            identifier is excluded. Off by default.
        numeric_strategy: Numeric imputation strategy.
        categorical_strategy: Categorical imputation strategy.

    Returns:
        ``(cleaned_dataframe, report)``.
    """
    logger.info("Starting cleaning pipeline on %d rows x %d columns.", len(df), len(df.columns))
    original_rows, original_columns = len(df), len(df.columns)

    working = df.copy()
    steps: list[CleaningStep] = []

    working, step = convert_dtypes(working)
    steps.append(step)

    working, step = remove_duplicates(
        working, drop_exact=drop_exact_duplicates, drop_excluding_id=drop_id_excluded_duplicates
    )
    steps.append(step)

    working, step = handle_missing_values(
        working, numeric_strategy=numeric_strategy, categorical_strategy=categorical_strategy
    )
    steps.append(step)

    # Rebuild the registry: label columns from here on should resolve to the new
    # readable columns rather than the original codes.
    working, step = normalise_categoricals(working, build_registry(working))
    steps.append(step)

    working, step = handle_invalid_numeric_values(working, build_registry(working))
    steps.append(step)

    working, step = flag_extreme_outliers(working, registry=build_registry(working))
    steps.append(step)

    report = CleaningReport(
        steps=tuple(steps),
        original_rows=original_rows,
        original_columns=original_columns,
        final_rows=len(working),
        final_columns=len(working.columns),
    )
    logger.info("Cleaning complete. %s", report.summary())
    return working, report


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_processed(
    df: pd.DataFrame,
    filename: str = PROCESSED_WIDE_FILENAME,
    processed_dir: Path | None = None,
) -> Path:
    """Write a processed dataframe to ``data/processed``.

    Parquet is used because it round-trips dtypes (categoricals and booleans
    survive), whereas CSV would silently degrade them.

    Args:
        df: Dataframe to persist.
        filename: Output filename.
        processed_dir: Target directory. Defaults to the configured one.

    Returns:
        The path written.

    Raises:
        OSError: If the file cannot be written.
    """
    directory = processed_dir or settings.processed_dir
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename

    if destination.suffix == ".parquet":
        df.to_parquet(destination, index=False)
    else:
        df.to_csv(destination, index=False)

    size_mb = destination.stat().st_size / (1024**2)
    logger.info(
        "Saved processed data: %s (%d rows x %d columns, %.2f MB)",
        destination.name,
        len(df),
        len(df.columns),
        size_mb,
    )
    return destination


def load_processed(
    filename: str = PROCESSED_WIDE_FILENAME, processed_dir: Path | None = None
) -> pd.DataFrame:
    """Read a previously processed dataframe.

    Args:
        filename: File to read.
        processed_dir: Source directory. Defaults to the configured one.

    Returns:
        The stored dataframe.

    Raises:
        FileNotFoundError: If the file does not exist, with guidance on how to
            regenerate it.
    """
    directory = processed_dir or settings.processed_dir
    source = directory / filename

    if not source.exists():
        raise FileNotFoundError(
            f"Processed file not found: {source}\n"
            "Run the pipeline first: python -m src.pipeline"
        )

    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    logger.info("Loaded processed data: %s (%d rows).", source.name, len(frame))
    return frame
