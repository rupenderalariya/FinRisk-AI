"""
Data quality analysis.

Returns structured results (dataclasses), never printed side effects, so the
same report drives the dashboard, the tests and the written report.

Why this module is not just a null-counter
------------------------------------------
The dataset in use has **zero missing cells and zero exact duplicate rows**. A
quality module that only counted blanks would score it 100/100 and tell the
analyst nothing. The defects in this data are *semantic*:

* category codes that appear in the file but not in the source documentation
  (``education_code`` 0/5/6, ``marriage_code`` 0, ``pay_status`` -2/0)
* rows that duplicate once the surrogate ``client_id`` is dropped
* statement balances exceeding the credit limit (over-limit accounts)
* negative statement balances (genuine credit balances, not errors)
* internal contradictions between repayment status and amount paid

So the engine checks four dimensions - completeness, uniqueness, validity and
consistency - and reports outliers separately without scoring them.
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
    DELINQUENCY_THRESHOLD,
    EDUCATION_LABELS,
    EDUCATION_UNDOCUMENTED_CODES,
    ID_COLUMN,
    MARRIAGE_LABELS,
    MARRIAGE_UNDOCUMENTED_CODES,
    PAY_STATUS_LABELS,
    PAY_STATUS_UNDOCUMENTED_CODES,
    SEX_LABELS,
    TARGET_COLUMN,
    ColumnRegistry,
    Role,
    build_registry,
    month_label,
)

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Scoring configuration - documented, project-defined, not an industry standard
# --------------------------------------------------------------------------- #

#: Weights of the four scored dimensions. They sum to 100.
DIMENSION_WEIGHTS: Final[Mapping[str, float]] = {
    "completeness": 30.0,
    "uniqueness": 20.0,
    "validity": 35.0,
    "consistency": 15.0,
}

SCORE_METHODOLOGY: Final[str] = """\
Data Quality Score - methodology
================================
This is a PROJECT-DEFINED heuristic for tracking data quality in this
application. It is NOT an industry-standard metric such as DAMA or ISO 8000,
and it should not be quoted as one.

Four dimensions are each scored on 0.0-1.0 and combined as a weighted sum,
giving a 0-100 result.

1. COMPLETENESS  (weight 30)
   1 - (missing cells / total cells).
   A fully populated table scores 1.0.

2. UNIQUENESS    (weight 20)
   1 - (duplicate rows excluding the surrogate ID / total rows).
   Rows that are identical once `client_id` is removed are treated as the
   uniqueness defect, because an auto-increment key makes every row trivially
   unique and hides genuine repetition.

3. VALIDITY      (weight 35)
   1 - (invalid cells / cells checked), over the columns that have an explicit
   domain rule: coded categoricals must use a documented code, `age` must fall
   in 18-100, `credit_limit` must be positive, and the target must be 0/1.
   Only columns with a declared rule contribute, so the denominator is honest
   about what was actually checked.

4. CONSISTENCY   (weight 15)
   1 - (rows with a cross-field contradiction / total rows).
   A contradiction is a pair of fields that cannot both be true, for example a
   repayment status of "paid in full" recorded alongside a zero payment against
   a positive outstanding statement.

NOT SCORED: outliers.
   Extreme values are detected and reported, but they do not reduce the score.
   In a credit portfolio a NT$1,000,000 limit or a large statement balance is
   usually genuine, not an error. Penalising a naturally skewed financial
   distribution would mislabel real data as dirty. Outliers are surfaced for
   analyst judgement instead.

Bands: 90-100 excellent, 75-89 good, 60-74 fair, below 60 poor.
"""


class Severity(str, Enum):
    """How much an issue should worry the analyst."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


SEVERITY_ORDER: Final[Mapping[Severity, int]] = {
    Severity.HIGH: 0,
    Severity.MEDIUM: 1,
    Severity.LOW: 2,
    Severity.INFO: 3,
}


# --------------------------------------------------------------------------- #
# Result structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QualityIssue:
    """One finding from one quality check."""

    check: str
    severity: Severity
    description: str
    affected_rows: int
    affected_percentage: float
    column: str | None = None
    recommendation: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the UI and AI payload."""
        return {
            "check": self.check,
            "severity": self.severity.value,
            "column": self.column,
            "description": self.description,
            "affected_rows": self.affected_rows,
            "affected_percentage": round(self.affected_percentage, 4),
            "recommendation": self.recommendation,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class MissingValueReport:
    """Completeness analysis."""

    total_cells: int
    missing_cells: int
    missing_by_column: dict[str, int]
    missing_percentage_by_column: dict[str, float]
    columns_fully_missing: tuple[str, ...]
    columns_with_any_missing: tuple[str, ...]

    @property
    def missing_percentage(self) -> float:
        """Missing cells as a percentage of all cells."""
        return 0.0 if self.total_cells == 0 else self.missing_cells / self.total_cells * 100.0

    @property
    def is_complete(self) -> bool:
        """True when there are no missing cells at all."""
        return self.missing_cells == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cells": self.total_cells,
            "missing_cells": self.missing_cells,
            "missing_percentage": round(self.missing_percentage, 4),
            "columns_with_any_missing": list(self.columns_with_any_missing),
            "columns_fully_missing": list(self.columns_fully_missing),
        }


@dataclass(frozen=True)
class DuplicateReport:
    """Uniqueness analysis."""

    total_rows: int
    exact_duplicate_rows: int
    duplicate_rows_excluding_id: int
    duplicated_id_values: int
    id_column_present: bool
    id_is_unique: bool
    example_duplicate_indices: tuple[int, ...] = field(default=())

    @property
    def duplicate_percentage(self) -> float:
        """Duplicates (ID excluded) as a percentage of rows."""
        if self.total_rows == 0:
            return 0.0
        return self.duplicate_rows_excluding_id / self.total_rows * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "duplicate_rows_excluding_id": self.duplicate_rows_excluding_id,
            "duplicate_percentage": round(self.duplicate_percentage, 4),
            "duplicated_id_values": self.duplicated_id_values,
            "id_is_unique": self.id_is_unique,
        }


@dataclass(frozen=True)
class ColumnOutlierSummary:
    """Outlier counts for a single numeric column."""

    column: str
    n_valid: int
    iqr_outliers: int
    zscore_outliers: int
    lower_bound: float
    upper_bound: float
    min_value: float
    max_value: float
    skewness: float

    @property
    def iqr_percentage(self) -> float:
        """IQR-flagged values as a percentage of non-null values."""
        return 0.0 if self.n_valid == 0 else self.iqr_outliers / self.n_valid * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "n_valid": self.n_valid,
            "iqr_outliers": self.iqr_outliers,
            "iqr_percentage": round(self.iqr_percentage, 3),
            "zscore_outliers": self.zscore_outliers,
            "lower_bound": round(self.lower_bound, 3),
            "upper_bound": round(self.upper_bound, 3),
            "min_value": round(self.min_value, 3),
            "max_value": round(self.max_value, 3),
            "skewness": round(self.skewness, 3),
        }


@dataclass(frozen=True)
class OutlierReport:
    """Outlier analysis across numeric columns. Reported, never auto-removed."""

    method_note: str
    iqr_multiplier: float
    by_column: dict[str, ColumnOutlierSummary]

    @property
    def total_iqr_outliers(self) -> int:
        """Sum of IQR-flagged values across all columns."""
        return sum(summary.iqr_outliers for summary in self.by_column.values())

    def most_affected(self, limit: int = 5) -> list[ColumnOutlierSummary]:
        """Columns with the highest share of IQR-flagged values."""
        return sorted(
            self.by_column.values(), key=lambda s: s.iqr_percentage, reverse=True
        )[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_note": self.method_note,
            "iqr_multiplier": self.iqr_multiplier,
            "total_iqr_outliers": self.total_iqr_outliers,
            "columns_analysed": len(self.by_column),
            "most_affected": [s.to_dict() for s in self.most_affected()],
        }


@dataclass(frozen=True)
class InvalidValueReport:
    """Domain-rule violations and undocumented codes."""

    cells_checked: int
    invalid_cells: int
    issues: tuple[QualityIssue, ...]
    undocumented_codes: dict[str, dict[int, int]]

    @property
    def invalid_percentage(self) -> float:
        """Invalid cells as a percentage of the cells actually checked."""
        return 0.0 if self.cells_checked == 0 else self.invalid_cells / self.cells_checked * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells_checked": self.cells_checked,
            "invalid_cells": self.invalid_cells,
            "invalid_percentage": round(self.invalid_percentage, 4),
            "undocumented_codes": {
                col: {str(code): count for code, count in codes.items()}
                for col, codes in self.undocumented_codes.items()
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class QualityScore:
    """The composite score with a full per-dimension breakdown."""

    score: float
    band: str
    dimension_scores: dict[str, float]
    dimension_weights: dict[str, float]
    methodology: str = SCORE_METHODOLOGY

    @property
    def contributions(self) -> dict[str, float]:
        """Points each dimension contributed to the final score."""
        return {
            name: round(self.dimension_scores.get(name, 0.0) * weight, 2)
            for name, weight in self.dimension_weights.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "band": self.band,
            "dimension_scores": {k: round(v, 4) for k, v in self.dimension_scores.items()},
            "dimension_weights": dict(self.dimension_weights),
            "contributions": self.contributions,
        }


@dataclass(frozen=True)
class DataQualityReport:
    """Complete quality assessment for one dataframe."""

    n_rows: int
    n_columns: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    missing: MissingValueReport
    duplicates: DuplicateReport
    outliers: OutlierReport
    invalid: InvalidValueReport
    dtype_issues: tuple[QualityIssue, ...]
    consistency_issues: tuple[QualityIssue, ...]
    uniqueness_issues: tuple[QualityIssue, ...]
    quality_score: QualityScore

    # ------------------------------------------------------------------ #

    @property
    def all_issues(self) -> tuple[QualityIssue, ...]:
        """Every issue found, sorted by severity then by rows affected."""
        combined = (
            list(self.invalid.issues)
            + list(self.dtype_issues)
            + list(self.consistency_issues)
            + list(self.uniqueness_issues)
        )
        combined.sort(key=lambda i: (SEVERITY_ORDER[i.severity], -i.affected_rows))
        return tuple(combined)

    def issues_by_severity(self, severity: Severity) -> tuple[QualityIssue, ...]:
        """Filter issues to one severity level."""
        return tuple(issue for issue in self.all_issues if issue.severity is severity)

    @property
    def headline_counts(self) -> dict[str, int]:
        """Issue counts per severity, for a summary strip in the UI."""
        return {
            severity.value: len(self.issues_by_severity(severity)) for severity in Severity
        }

    def summary(self) -> dict[str, Any]:
        """The compact dictionary shape requested for the quality summary card."""
        return {
            "rows": self.n_rows,
            "columns": self.n_columns,
            "missing_values": self.missing.missing_cells,
            "duplicates": self.duplicates.duplicate_rows_excluding_id,
            "numeric_columns": len(self.numeric_columns),
            "categorical_columns": len(self.categorical_columns),
            "quality_score": round(self.quality_score.score, 2),
        }

    def to_dict(self) -> dict[str, Any]:
        """Full serialisable report."""
        return {
            "summary": self.summary(),
            "quality_score": self.quality_score.to_dict(),
            "missing": self.missing.to_dict(),
            "duplicates": self.duplicates.to_dict(),
            "outliers": self.outliers.to_dict(),
            "invalid": self.invalid.to_dict(),
            "issue_counts": self.headline_counts,
            "issues": [issue.to_dict() for issue in self.all_issues],
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _percentage(count: int, total: int) -> float:
    """Safe percentage that returns 0.0 rather than dividing by zero."""
    return 0.0 if total == 0 else count / total * 100.0


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, turning unparseable entries into NaN."""
    return pd.to_numeric(series, errors="coerce")


# --------------------------------------------------------------------------- #
# 1. Completeness
# --------------------------------------------------------------------------- #


def analyse_missing_values(df: pd.DataFrame) -> MissingValueReport:
    """Analyse missing values across every column.

    Args:
        df: Dataframe to analyse.

    Returns:
        A :class:`MissingValueReport`.
    """
    n_rows = len(df)
    missing_by_column = {str(col): int(df[col].isna().sum()) for col in df.columns}
    total_cells = n_rows * len(df.columns)

    report = MissingValueReport(
        total_cells=total_cells,
        missing_cells=int(sum(missing_by_column.values())),
        missing_by_column=missing_by_column,
        missing_percentage_by_column={
            col: _percentage(count, n_rows) for col, count in missing_by_column.items()
        },
        columns_fully_missing=tuple(
            col for col, count in missing_by_column.items() if n_rows > 0 and count == n_rows
        ),
        columns_with_any_missing=tuple(
            col for col, count in missing_by_column.items() if count > 0
        ),
    )
    logger.info(
        "Completeness: %d missing cells (%.4f%%) across %d column(s).",
        report.missing_cells,
        report.missing_percentage,
        len(report.columns_with_any_missing),
    )
    return report


# --------------------------------------------------------------------------- #
# 2. Uniqueness
# --------------------------------------------------------------------------- #


def analyse_duplicates(df: pd.DataFrame) -> tuple[DuplicateReport, list[QualityIssue]]:
    """Analyse duplicate rows and identifier uniqueness.

    Both "exact duplicate" and "duplicate once the surrogate key is dropped" are
    measured. The second is the meaningful one: an auto-increment ``client_id``
    makes every row unique by construction and would mask real repetition.

    Args:
        df: Dataframe to analyse.

    Returns:
        ``(report, issues)``.
    """
    issues: list[QualityIssue] = []
    n_rows = len(df)

    if n_rows == 0:
        return (
            DuplicateReport(
                total_rows=0,
                exact_duplicate_rows=0,
                duplicate_rows_excluding_id=0,
                duplicated_id_values=0,
                id_column_present=ID_COLUMN in df.columns,
                id_is_unique=True,
            ),
            issues,
        )

    exact = int(df.duplicated().sum())
    id_present = ID_COLUMN in df.columns

    if id_present and len(df.columns) > 1:
        without_id = df.drop(columns=[ID_COLUMN])
        duplicate_mask = without_id.duplicated(keep="first")
        excluding_id = int(duplicate_mask.sum())
        examples = tuple(int(i) for i in df.index[duplicate_mask][:5])
        duplicated_ids = int(df[ID_COLUMN].duplicated().sum())
    else:
        duplicate_mask = df.duplicated(keep="first")
        excluding_id = exact
        examples = tuple(int(i) for i in df.index[duplicate_mask][:5])
        duplicated_ids = 0

    report = DuplicateReport(
        total_rows=n_rows,
        exact_duplicate_rows=exact,
        duplicate_rows_excluding_id=excluding_id,
        duplicated_id_values=duplicated_ids,
        id_column_present=id_present,
        id_is_unique=(duplicated_ids == 0),
        example_duplicate_indices=examples,
    )

    if exact > 0:
        issues.append(
            QualityIssue(
                check="exact_duplicate_rows",
                severity=Severity.HIGH,
                description=(
                    f"{exact:,} row(s) are exact duplicates across every column, "
                    "including the identifier."
                ),
                affected_rows=exact,
                affected_percentage=_percentage(exact, n_rows),
                recommendation="Remove exact duplicates; they carry no extra information.",
            )
        )

    if excluding_id > 0:
        issues.append(
            QualityIssue(
                check="duplicate_rows_excluding_id",
                severity=Severity.MEDIUM,
                description=(
                    f"{excluding_id:,} row(s) are identical once '{ID_COLUMN}' is "
                    "excluded. These may be genuinely distinct clients with "
                    "coincidentally identical records, or repeated data entry."
                ),
                affected_rows=excluding_id,
                affected_percentage=_percentage(excluding_id, n_rows),
                recommendation=(
                    "Retained by default because coincidental matches are plausible "
                    "with integer-coded fields. Removal is available as an explicit "
                    "cleaning option."
                ),
                details={"example_row_indices": list(examples)},
            )
        )

    if duplicated_ids > 0:
        issues.append(
            QualityIssue(
                check="duplicate_identifier",
                severity=Severity.HIGH,
                description=f"'{ID_COLUMN}' has {duplicated_ids:,} repeated value(s).",
                affected_rows=duplicated_ids,
                affected_percentage=_percentage(duplicated_ids, n_rows),
                column=ID_COLUMN,
                recommendation="Investigate the source extract; the key should be unique.",
            )
        )

    logger.info(
        "Uniqueness: %d exact duplicate(s), %d duplicate(s) excluding ID.",
        exact,
        excluding_id,
    )
    return report, issues


def analyse_uniqueness(df: pd.DataFrame) -> list[QualityIssue]:
    """Flag constant columns and near-constant low-information columns.

    Args:
        df: Dataframe to analyse.

    Returns:
        A list of issues.
    """
    issues: list[QualityIssue] = []
    n_rows = len(df)
    if n_rows <= 1:
        return issues

    for column in df.columns:
        distinct = int(df[column].nunique(dropna=True))
        if distinct == 0:
            issues.append(
                QualityIssue(
                    check="empty_column",
                    severity=Severity.HIGH,
                    description=f"Column '{column}' contains no values at all.",
                    affected_rows=n_rows,
                    affected_percentage=100.0,
                    column=str(column),
                    recommendation="Drop this column; it cannot contribute to analysis.",
                )
            )
        elif distinct == 1:
            issues.append(
                QualityIssue(
                    check="constant_column",
                    severity=Severity.LOW,
                    description=(
                        f"Column '{column}' holds a single repeated value, so it has "
                        "no discriminating power."
                    ),
                    affected_rows=n_rows,
                    affected_percentage=100.0,
                    column=str(column),
                    recommendation="Exclude from correlation and modelling inputs.",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 3. Data types
# --------------------------------------------------------------------------- #


def validate_dtypes(df: pd.DataFrame, registry: ColumnRegistry | None = None) -> list[QualityIssue]:
    """Check that columns expected to be numeric really are.

    Spreadsheet exports commonly turn a numeric column into text via a stray
    space or a thousands separator, which silently breaks every downstream
    calculation. This catches that.

    Args:
        df: Dataframe to check.
        registry: Optional pre-built registry; rebuilt when omitted.

    Returns:
        A list of issues.
    """
    issues: list[QualityIssue] = []
    reg = registry or build_registry(df)
    n_rows = len(df)

    numeric_roles = (
        Role.CREDIT_LIMIT,
        Role.AGE,
        Role.BILL_AMOUNT,
        Role.PAYMENT_AMOUNT,
        Role.PAYMENT_STATUS,
        Role.TARGET,
    )
    expected_numeric: list[str] = []
    for role in numeric_roles:
        expected_numeric.extend(reg.all(role))

    for column in expected_numeric:
        if column not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        coerced = _numeric(df[column])
        unparseable = int(coerced.isna().sum() - df[column].isna().sum())
        issues.append(
            QualityIssue(
                check="unexpected_dtype",
                severity=Severity.HIGH if unparseable else Severity.MEDIUM,
                description=(
                    f"Column '{column}' is expected to be numeric but has dtype "
                    f"'{df[column].dtype}'."
                    + (
                        f" {unparseable:,} value(s) cannot be parsed as numbers."
                        if unparseable
                        else " All values are still parseable as numbers."
                    )
                ),
                affected_rows=unparseable if unparseable else n_rows,
                affected_percentage=_percentage(unparseable if unparseable else n_rows, n_rows),
                column=str(column),
                recommendation=(
                    "Converted to numeric during cleaning; unparseable entries become "
                    "missing values rather than being dropped."
                ),
            )
        )

    if issues:
        logger.warning("Dtype validation raised %d issue(s).", len(issues))
    return issues


# --------------------------------------------------------------------------- #
# 4. Validity - domain rules and undocumented codes
# --------------------------------------------------------------------------- #


def _check_coded_column(
    df: pd.DataFrame,
    column: str,
    documented: Mapping[int, str],
    undocumented: Sequence[int],
    human_name: str,
) -> tuple[int, dict[int, int], list[QualityIssue]]:
    """Compare a coded categorical column against its documented code list.

    Args:
        df: Dataframe containing the column.
        column: Column to check.
        documented: Code -> label mapping from the source documentation.
        undocumented: Codes known to occur but absent from the documentation.
        human_name: Readable field name for messages.

    Returns:
        ``(invalid_cell_count, {code: count}, issues)``.
    """
    issues: list[QualityIssue] = []
    if column not in df.columns:
        return 0, {}, issues

    n_rows = len(df)
    values = _numeric(df[column]).dropna().astype("int64")
    counts = values.value_counts().to_dict()

    known_undocumented = {
        int(code): int(count) for code, count in counts.items() if code in set(undocumented)
    }
    fully_unknown = {
        int(code): int(count)
        for code, count in counts.items()
        if code not in documented and code not in set(undocumented)
    }

    if known_undocumented:
        affected = sum(known_undocumented.values())
        issues.append(
            QualityIssue(
                check="undocumented_category_code",
                severity=Severity.MEDIUM,
                description=(
                    f"'{column}' ({human_name}) contains code(s) "
                    f"{sorted(known_undocumented)} that appear in the data but are "
                    f"not defined in the source documentation "
                    f"(documented codes: {sorted(documented)})."
                ),
                affected_rows=affected,
                affected_percentage=_percentage(affected, n_rows),
                column=column,
                recommendation=(
                    f"Grouped into a single 'Unknown' category during cleaning rather "
                    "than guessing an interpretation or discarding the rows."
                ),
                details={"code_counts": {str(k): v for k, v in sorted(known_undocumented.items())}},
            )
        )

    if fully_unknown:
        affected = sum(fully_unknown.values())
        issues.append(
            QualityIssue(
                check="unexpected_category_code",
                severity=Severity.HIGH,
                description=(
                    f"'{column}' ({human_name}) contains entirely unexpected code(s) "
                    f"{sorted(fully_unknown)}."
                ),
                affected_rows=affected,
                affected_percentage=_percentage(affected, n_rows),
                column=column,
                recommendation="Treated as 'Unknown'. Verify the source extract.",
                details={"code_counts": {str(k): v for k, v in sorted(fully_unknown.items())}},
            )
        )

    all_invalid = {**known_undocumented, **fully_unknown}
    return sum(all_invalid.values()), all_invalid, issues


def detect_invalid_values(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> InvalidValueReport:
    """Apply domain rules and report violations.

    Only columns with an explicit rule contribute to ``cells_checked``, so the
    validity denominator honestly reflects what was actually examined.

    Args:
        df: Dataframe to check.
        registry: Optional pre-built registry.

    Returns:
        An :class:`InvalidValueReport`.
    """
    reg = registry or build_registry(df)
    n_rows = len(df)
    issues: list[QualityIssue] = []
    undocumented: dict[str, dict[int, int]] = {}
    invalid_cells = 0
    cells_checked = 0

    # --- coded categoricals ---
    coded_checks = (
        (reg.one(Role.SEX), SEX_LABELS, (), "biological sex"),
        (reg.one(Role.EDUCATION), EDUCATION_LABELS, EDUCATION_UNDOCUMENTED_CODES, "education level"),
        (reg.one(Role.MARRIAGE), MARRIAGE_LABELS, MARRIAGE_UNDOCUMENTED_CODES, "marital status"),
    )
    for column, documented, undoc, human_name in coded_checks:
        if column is None or column not in df.columns:
            continue
        # Only the raw *_code columns carry integer codes; skip already-labelled text.
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        cells_checked += n_rows
        count, codes, found = _check_coded_column(df, column, documented, undoc, human_name)
        invalid_cells += count
        if codes:
            undocumented[column] = codes
        issues.extend(found)

    # --- repayment status codes, one column per month ---
    for column in reg.all(Role.PAYMENT_STATUS):
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        cells_checked += n_rows
        values = _numeric(df[column]).dropna().astype("int64")
        counts = values.value_counts().to_dict()
        undoc_found = {
            int(code): int(count)
            for code, count in counts.items()
            if code in set(PAY_STATUS_UNDOCUMENTED_CODES)
        }
        unknown_found = {
            int(code): int(count)
            for code, count in counts.items()
            if code not in PAY_STATUS_LABELS
        }
        if undoc_found or unknown_found:
            merged = {**undoc_found, **unknown_found}
            undocumented[column] = merged
            invalid_cells += sum(unknown_found.values())  # only truly unknown codes count
        if unknown_found:
            affected = sum(unknown_found.values())
            issues.append(
                QualityIssue(
                    check="unexpected_category_code",
                    severity=Severity.HIGH,
                    description=(
                        f"'{column}' contains repayment-status code(s) "
                        f"{sorted(unknown_found)} outside the documented range -2 to 9."
                    ),
                    affected_rows=affected,
                    affected_percentage=_percentage(affected, n_rows),
                    column=column,
                    recommendation="Verify the source extract before relying on this month.",
                )
            )

    # Aggregate the -2 / 0 finding into one issue instead of six near-identical ones.
    status_columns = [c for c in reg.all(Role.PAYMENT_STATUS) if c in undocumented]
    if status_columns:
        undocumented_codes = set(PAY_STATUS_UNDOCUMENTED_CODES)
        cell_occurrences = sum(
            count
            for column in status_columns
            for code, count in undocumented[column].items()
            if code in undocumented_codes
        )
        if cell_occurrences:
            # Distinct clients touched, so the row count never exceeds the row total.
            affected_mask = pd.Series(False, index=df.index)
            for column in status_columns:
                affected_mask |= _numeric(df[column]).isin(list(undocumented_codes))
            affected = int(affected_mask.sum())
            issues.append(
                QualityIssue(
                    check="undocumented_category_code",
                    severity=Severity.LOW,
                    description=(
                        f"Repayment-status codes {list(PAY_STATUS_UNDOCUMENTED_CODES)} occur in "
                        f"{cell_occurrences:,} client-month cell(s) across "
                        f"{len(status_columns)} monthly column(s), affecting {affected:,} "
                        "distinct client(s). These codes are not defined in the source "
                        "paper; the common reading is -2 = no transaction and "
                        "0 = revolving credit used."
                    ),
                    affected_rows=affected,
                    affected_percentage=_percentage(affected, n_rows),
                    recommendation=(
                        "Retained as distinct categories and labelled as undocumented. "
                        "Delinquency features use only codes >= 1, which are documented, "
                        "so this ambiguity does not affect the delinquency measures."
                    ),
                    details={
                        "columns": status_columns,
                        "client_month_cells": cell_occurrences,
                        "distinct_clients": affected,
                    },
                )
            )

    # --- numeric range rules ---
    age_column = reg.one(Role.AGE)
    if age_column and age_column in df.columns:
        cells_checked += n_rows
        ages = _numeric(df[age_column])
        implausible = int(((ages < 18) | (ages > 100)).sum())
        if implausible:
            invalid_cells += implausible
            issues.append(
                QualityIssue(
                    check="value_out_of_range",
                    severity=Severity.MEDIUM,
                    description=(
                        f"{implausible:,} row(s) have an age outside the plausible "
                        "18-100 range for a credit-card holder."
                    ),
                    affected_rows=implausible,
                    affected_percentage=_percentage(implausible, n_rows),
                    column=age_column,
                    recommendation="Flagged, not deleted. Excluded from age-band analysis.",
                    details={
                        "observed_min": float(ages.min()) if len(ages.dropna()) else None,
                        "observed_max": float(ages.max()) if len(ages.dropna()) else None,
                    },
                )
            )

    limit_column = reg.one(Role.CREDIT_LIMIT)
    if limit_column and limit_column in df.columns:
        cells_checked += n_rows
        limits = _numeric(df[limit_column])
        non_positive = int((limits <= 0).sum())
        if non_positive:
            invalid_cells += non_positive
            issues.append(
                QualityIssue(
                    check="value_out_of_range",
                    severity=Severity.HIGH,
                    description=(
                        f"{non_positive:,} row(s) have a credit limit of zero or less, "
                        "which is not a valid limit."
                    ),
                    affected_rows=non_positive,
                    affected_percentage=_percentage(non_positive, n_rows),
                    column=limit_column,
                    recommendation=(
                        "Utilisation is left undefined (null) for these rows rather "
                        "than dividing by zero."
                    ),
                )
            )

    target_column = reg.one(Role.TARGET) or (TARGET_COLUMN if TARGET_COLUMN in df.columns else None)
    if target_column and target_column in df.columns:
        cells_checked += n_rows
        target_values = _numeric(df[target_column])
        bad_target = int((~target_values.isin([0, 1]) & target_values.notna()).sum())
        if bad_target:
            invalid_cells += bad_target
            issues.append(
                QualityIssue(
                    check="invalid_target_value",
                    severity=Severity.HIGH,
                    description=(
                        f"{bad_target:,} row(s) have a target value outside {{0, 1}}."
                    ),
                    affected_rows=bad_target,
                    affected_percentage=_percentage(bad_target, n_rows),
                    column=target_column,
                    recommendation="Exclude these rows from supervised modelling.",
                )
            )

    report = InvalidValueReport(
        cells_checked=cells_checked,
        invalid_cells=invalid_cells,
        issues=tuple(issues),
        undocumented_codes=undocumented,
    )
    logger.info(
        "Validity: %d invalid cell(s) out of %d checked (%.4f%%), %d issue(s).",
        invalid_cells,
        cells_checked,
        report.invalid_percentage,
        len(issues),
    )
    return report


# --------------------------------------------------------------------------- #
# 5. Consistency - cross-field contradictions
# --------------------------------------------------------------------------- #


def detect_consistency_issues(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> tuple[list[QualityIssue], int]:
    """Look for combinations of fields that cannot both be correct.

    Genuine contradictions reduce the consistency score. Patterns that are
    unusual but legitimate - a negative statement balance, an over-limit
    account - are reported at INFO severity and do not reduce it, because
    penalising real behaviour would misrepresent the data.

    Args:
        df: Dataframe to check.
        registry: Optional pre-built registry.

    Returns:
        ``(issues, n_contradictory_rows)``.
    """
    reg = registry or build_registry(df)
    issues: list[QualityIssue] = []
    n_rows = len(df)
    if n_rows == 0:
        return issues, 0

    contradiction_mask = pd.Series(False, index=df.index)

    bills = reg.panel_columns(Role.BILL_AMOUNT)
    payments = reg.panel_columns(Role.PAYMENT_AMOUNT)
    statuses = reg.panel_columns(Role.PAYMENT_STATUS)
    limit_column = reg.one(Role.CREDIT_LIMIT)

    # --- INFO: negative statement balances (credit balances / overpayments) ---
    if bills:
        negative_counts: dict[str, int] = {}
        for month, column in bills.items():
            count = int((_numeric(df[column]) < 0).sum())
            if count:
                negative_counts[f"{column} ({month_label(month)})"] = count
        if negative_counts:
            worst = max(negative_counts.values())
            issues.append(
                QualityIssue(
                    check="negative_statement_balance",
                    severity=Severity.INFO,
                    description=(
                        "Negative statement balances occur in "
                        f"{len(negative_counts)} monthly column(s), up to {worst:,} rows "
                        "in a single month. A negative balance means the account is in "
                        "credit, usually from an overpayment or a refund."
                    ),
                    affected_rows=worst,
                    affected_percentage=_percentage(worst, n_rows),
                    recommendation=(
                        "Legitimate values, retained unchanged. Utilisation can be "
                        "negative for these rows, which is meaningful and not an error."
                    ),
                    details={"counts_by_month": negative_counts},
                )
            )

    # --- INFO: over-limit accounts ---
    if bills and limit_column and limit_column in df.columns:
        limits = _numeric(df[limit_column])
        over_limit_any = pd.Series(False, index=df.index)
        for column in bills.values():
            over_limit_any |= _numeric(df[column]) > limits
        count = int(over_limit_any.sum())
        if count:
            issues.append(
                QualityIssue(
                    check="over_limit_balance",
                    severity=Severity.INFO,
                    description=(
                        f"{count:,} client(s) had a statement balance above their credit "
                        "limit in at least one month, i.e. utilisation above 100%."
                    ),
                    affected_rows=count,
                    affected_percentage=_percentage(count, n_rows),
                    recommendation=(
                        "Plausible in a real portfolio (fees, interest, temporary limit "
                        "changes). Retained and treated as a genuine high-risk signal "
                        "rather than an error."
                    ),
                )
            )

    # --- MEDIUM: status says paid in full, but nothing was paid ---
    # A payment recorded in month m settles the statement from month m+1.
    if statuses and payments and bills:
        check_mask = pd.Series(False, index=df.index)
        record_total = 0
        per_month: dict[str, int] = {}
        for month, status_column in statuses.items():
            payment_column = payments.get(month)
            prior_bill_column = bills.get(month + 1)
            if payment_column is None or prior_bill_column is None:
                continue
            status = _numeric(df[status_column])
            paid = _numeric(df[payment_column])
            prior_bill = _numeric(df[prior_bill_column])
            mask = (status == -1) & (paid == 0) & (prior_bill > 0)
            count = int(mask.sum())
            if count:
                per_month[f"{status_column} ({month_label(month)})"] = count
                record_total += count
                check_mask |= mask
        if per_month:
            affected = int(check_mask.sum())
            issues.append(
                QualityIssue(
                    check="status_payment_contradiction",
                    severity=Severity.MEDIUM,
                    description=(
                        f"{record_total:,} client-month record(s), spanning {affected:,} "
                        "distinct client(s), are marked 'paid in full' (-1) yet record a "
                        "zero payment against a positive prior statement balance."
                    ),
                    affected_rows=affected,
                    affected_percentage=_percentage(affected, n_rows),
                    recommendation=(
                        "Retained. Likely a settlement-timing effect between the "
                        "statement and payment cut-off dates, but it is a genuine "
                        "internal inconsistency and is reported as one."
                    ),
                    details={
                        "client_month_records": record_total,
                        "distinct_clients": affected,
                        "counts_by_month": per_month,
                    },
                )
            )
            contradiction_mask |= check_mask

    # --- MEDIUM: delinquent status but no outstanding balance ---
    if statuses and bills:
        check_mask = pd.Series(False, index=df.index)
        per_month_delinquent: dict[str, int] = {}
        record_total = 0
        for month, status_column in statuses.items():
            bill_column = bills.get(month)
            if bill_column is None:
                continue
            mask = (_numeric(df[status_column]) >= DELINQUENCY_THRESHOLD) & (
                _numeric(df[bill_column]) <= 0
            )
            count = int(mask.sum())
            if count:
                per_month_delinquent[f"{status_column} ({month_label(month)})"] = count
                record_total += count
                check_mask |= mask
        if per_month_delinquent:
            affected = int(check_mask.sum())
            issues.append(
                QualityIssue(
                    check="delinquent_without_balance",
                    severity=Severity.MEDIUM,
                    description=(
                        f"{record_total:,} client-month record(s), spanning {affected:,} "
                        "distinct client(s), show a payment delay of one month or more "
                        "while the statement balance is zero or negative."
                    ),
                    affected_rows=affected,
                    affected_percentage=_percentage(affected, n_rows),
                    recommendation=(
                        "Retained and reported. Can arise when an arrear is cleared "
                        "mid-cycle, but the two fields still disagree."
                    ),
                    details={
                        "client_month_records": record_total,
                        "distinct_clients": affected,
                        "counts_by_month": per_month_delinquent,
                    },
                )
            )
            contradiction_mask |= check_mask

    n_contradictory = int(contradiction_mask.sum())
    logger.info(
        "Consistency: %d row(s) carry at least one contradiction; %d issue(s) raised.",
        n_contradictory,
        len(issues),
    )
    return issues, n_contradictory


# --------------------------------------------------------------------------- #
# 6. Outliers
# --------------------------------------------------------------------------- #


def detect_outliers(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    iqr_multiplier: float | None = None,
    zscore_threshold: float = 3.0,
) -> OutlierReport:
    """Detect outliers in numeric columns using both IQR and z-score.

    Nothing is removed. Financial distributions are legitimately skewed, so IQR
    flags a large share of values on money columns; that is a property of the
    distribution, not evidence of dirty data.

    Args:
        df: Dataframe to analyse.
        columns: Columns to examine. Defaults to all numeric columns except the
            identifier and the binary target.
        iqr_multiplier: IQR fence multiplier. Defaults to the configured value.
        zscore_threshold: Absolute z-score above which a value is flagged.

    Returns:
        An :class:`OutlierReport`.
    """
    multiplier = iqr_multiplier if iqr_multiplier is not None else settings.outlier_iqr_multiplier

    if columns is None:
        candidates = [
            str(column)
            for column in df.select_dtypes(include="number").columns
            if str(column) not in {ID_COLUMN, TARGET_COLUMN}
        ]
    else:
        candidates = [str(column) for column in columns if column in df.columns]

    by_column: dict[str, ColumnOutlierSummary] = {}

    for column in candidates:
        values = _numeric(df[column]).dropna()
        if values.empty:
            continue
        # A binary or near-constant column has no meaningful outliers.
        if values.nunique() <= 2:
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        iqr_count = int(((values < lower) | (values > upper)).sum())

        std = float(values.std(ddof=0))
        if std > 0:
            z_scores = (values - float(values.mean())) / std
            z_count = int((z_scores.abs() > zscore_threshold).sum())
        else:
            z_count = 0

        by_column[column] = ColumnOutlierSummary(
            column=column,
            n_valid=int(values.size),
            iqr_outliers=iqr_count,
            zscore_outliers=z_count,
            lower_bound=lower,
            upper_bound=upper,
            min_value=float(values.min()),
            max_value=float(values.max()),
            skewness=float(values.skew()) if values.size > 2 else 0.0,
        )

    report = OutlierReport(
        method_note=(
            f"IQR fences at {multiplier}x the interquartile range, plus a z-score "
            f"threshold of {zscore_threshold}. Both are reported; neither removes data. "
            "Financial amount columns are strongly right-skewed, so a high IQR flag "
            "rate is expected and does not by itself indicate an error."
        ),
        iqr_multiplier=multiplier,
        by_column=by_column,
    )
    logger.info(
        "Outliers: analysed %d numeric column(s); %d IQR-flagged value(s) in total.",
        len(by_column),
        report.total_iqr_outliers,
    )
    return report


# --------------------------------------------------------------------------- #
# 7. Scoring
# --------------------------------------------------------------------------- #


def _band_for(score: float) -> str:
    """Map a 0-100 score to its descriptive band."""
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Poor"


def compute_quality_score(
    missing: MissingValueReport,
    duplicates: DuplicateReport,
    invalid: InvalidValueReport,
    n_contradictory_rows: int,
    n_rows: int,
) -> QualityScore:
    """Combine the four dimensions into a 0-100 score.

    See :data:`SCORE_METHODOLOGY` for the full definition. Each dimension is
    scored on 0.0-1.0 and weighted; a dimension that could not be assessed
    scores 1.0 so an unmeasurable aspect is not penalised as a defect.

    Args:
        missing: Completeness report.
        duplicates: Uniqueness report.
        invalid: Validity report.
        n_contradictory_rows: Rows with at least one cross-field contradiction.
        n_rows: Total rows.

    Returns:
        A :class:`QualityScore`.
    """
    completeness = (
        1.0 - missing.missing_cells / missing.total_cells if missing.total_cells else 1.0
    )
    uniqueness = 1.0 - duplicates.duplicate_rows_excluding_id / n_rows if n_rows else 1.0
    validity = (
        1.0 - invalid.invalid_cells / invalid.cells_checked if invalid.cells_checked else 1.0
    )
    consistency = 1.0 - n_contradictory_rows / n_rows if n_rows else 1.0

    dimension_scores = {
        "completeness": max(0.0, min(1.0, completeness)),
        "uniqueness": max(0.0, min(1.0, uniqueness)),
        "validity": max(0.0, min(1.0, validity)),
        "consistency": max(0.0, min(1.0, consistency)),
    }

    score = sum(
        dimension_scores[name] * weight for name, weight in DIMENSION_WEIGHTS.items()
    )
    score = max(0.0, min(100.0, score))

    result = QualityScore(
        score=score,
        band=_band_for(score),
        dimension_scores=dimension_scores,
        dimension_weights=dict(DIMENSION_WEIGHTS),
    )
    logger.info(
        "Quality score: %.2f/100 (%s) | completeness=%.3f uniqueness=%.3f "
        "validity=%.3f consistency=%.3f",
        score,
        result.band,
        dimension_scores["completeness"],
        dimension_scores["uniqueness"],
        dimension_scores["validity"],
        dimension_scores["consistency"],
    )
    return result


# --------------------------------------------------------------------------- #
# 8. Orchestration
# --------------------------------------------------------------------------- #


def run_quality_report(
    df: pd.DataFrame, registry: ColumnRegistry | None = None
) -> DataQualityReport:
    """Run every quality check and assemble the full report.

    Args:
        df: Dataframe to assess, with canonical column names.
        registry: Optional pre-built registry.

    Returns:
        A :class:`DataQualityReport`.
    """
    reg = registry or build_registry(df)
    logger.info("Running data quality analysis on %d rows x %d columns.", len(df), len(df.columns))

    missing = analyse_missing_values(df)
    duplicates, duplicate_issues = analyse_duplicates(df)
    uniqueness_issues = analyse_uniqueness(df) + duplicate_issues
    dtype_issues = validate_dtypes(df, reg)
    invalid = detect_invalid_values(df, reg)
    consistency_issues, n_contradictory = detect_consistency_issues(df, reg)
    outliers = detect_outliers(df)

    score = compute_quality_score(
        missing=missing,
        duplicates=duplicates,
        invalid=invalid,
        n_contradictory_rows=n_contradictory,
        n_rows=len(df),
    )

    report = DataQualityReport(
        n_rows=int(len(df)),
        n_columns=int(len(df.columns)),
        numeric_columns=tuple(str(c) for c in df.select_dtypes(include="number").columns),
        categorical_columns=tuple(
            str(c) for c in df.select_dtypes(include=["object", "string", "category"]).columns
        ),
        datetime_columns=tuple(
            str(c) for c in df.select_dtypes(include=["datetime", "datetimetz"]).columns
        ),
        missing=missing,
        duplicates=duplicates,
        outliers=outliers,
        invalid=invalid,
        dtype_issues=tuple(dtype_issues),
        consistency_issues=tuple(consistency_issues),
        uniqueness_issues=tuple(uniqueness_issues),
        quality_score=score,
    )

    logger.info(
        "Quality analysis complete: score %.2f (%s), %d issue(s) - %s",
        score.score,
        score.band,
        len(report.all_issues),
        report.headline_counts,
    )
    return report
