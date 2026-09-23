"""
Dataset contract and column-role registry.

Downstream modules must never hard-code a physical column name such as
``BILL_AMT1``. They ask the registry a *semantic* question instead::

    if registry.has(Role.BILL_AMOUNT):
        ...

That indirection is what makes the KPI engine, feature engineering and
analytics adaptive: a KPI whose required roles are absent is omitted rather
than rendered as "N/A", and a feature whose source columns are missing is
skipped with a logged reason instead of fabricating a value.

Dataset in use
--------------
UCI ML Repository, dataset 350 - "Default of Credit Card Clients"
Taiwan, April to September 2005, 30,000 clients, 25 columns, amounts in NT$.

Panel layout (the property that makes real trend analysis possible)
------------------------------------------------------------------
The source file stores six monthly observations per client across three column
families. Month index 1 is the **most recent** month::

    m1 = September 2005   ...   m6 = April 2005

A quirk worth knowing: the source names the repayment-status columns
``PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6`` - there is no ``PAY_1``.
``PAY_0`` is the September status, so it maps to month index 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Iterable, Mapping

import pandas as pd

# --------------------------------------------------------------------------- #
# Provenance - single source of truth, reused by the docs and the UI
# --------------------------------------------------------------------------- #

DATASET_NAME: Final[str] = "Default of Credit Card Clients"
DATASET_SOURCE: Final[str] = "UCI Machine Learning Repository (dataset 350)"
DATASET_LANDING_URL: Final[str] = (
    "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
)
DATASET_DOWNLOAD_URL: Final[str] = (
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
)
DATASET_CITATION: Final[str] = (
    "Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques "
    "for the predictive accuracy of probability of default of credit card clients. "
    "Expert Systems with Applications, 36(2), 2473-2480."
)
DATASET_LICENCE: Final[str] = "CC BY 4.0 (UCI ML Repository)"
DATASET_CURRENCY: Final[str] = "NT$"
DATASET_PERIOD: Final[str] = "April 2005 - September 2005"

# Expected shape of the raw file. Used to validate, never to invent.
EXPECTED_RAW_ROWS: Final[int] = 30_000
EXPECTED_RAW_COLUMNS: Final[int] = 25

N_PANEL_MONTHS: Final[int] = 6

#: Month index -> calendar label. Month 1 is the most recent observation.
MONTH_LABELS: Final[Mapping[int, str]] = {
    1: "Sep 2005",
    2: "Aug 2005",
    3: "Jul 2005",
    4: "Jun 2005",
    5: "May 2005",
    6: "Apr 2005",
}

#: Chronological order (oldest first) for plotting time on a left-to-right axis.
CHRONOLOGICAL_MONTHS: Final[tuple[int, ...]] = (6, 5, 4, 3, 2, 1)


# --------------------------------------------------------------------------- #
# Raw -> canonical column names
# --------------------------------------------------------------------------- #

#: Maps the raw header exactly as it appears in the source file to the
#: canonical snake_case name used everywhere else in the project.
RAW_TO_CANONICAL: Final[Mapping[str, str]] = {
    "ID": "client_id",
    "LIMIT_BAL": "credit_limit",
    "SEX": "sex_code",
    "EDUCATION": "education_code",
    "MARRIAGE": "marriage_code",
    "AGE": "age",
    # Repayment status. Note PAY_0 -> month 1; there is no PAY_1 in the source.
    "PAY_0": "pay_status_m1",
    "PAY_2": "pay_status_m2",
    "PAY_3": "pay_status_m3",
    "PAY_4": "pay_status_m4",
    "PAY_5": "pay_status_m5",
    "PAY_6": "pay_status_m6",
    # Statement balance per month.
    "BILL_AMT1": "bill_amt_m1",
    "BILL_AMT2": "bill_amt_m2",
    "BILL_AMT3": "bill_amt_m3",
    "BILL_AMT4": "bill_amt_m4",
    "BILL_AMT5": "bill_amt_m5",
    "BILL_AMT6": "bill_amt_m6",
    # Amount actually paid in each month.
    "PAY_AMT1": "pay_amt_m1",
    "PAY_AMT2": "pay_amt_m2",
    "PAY_AMT3": "pay_amt_m3",
    "PAY_AMT4": "pay_amt_m4",
    "PAY_AMT5": "pay_amt_m5",
    "PAY_AMT6": "pay_amt_m6",
    # Target.
    "default payment next month": "default_next_month",
    # Seen in some redistributions of this dataset (e.g. the Kaggle mirror).
    "default.payment.next.month": "default_next_month",
    "PAY_1": "pay_status_m1",
}

TARGET_COLUMN: Final[str] = "default_next_month"
ID_COLUMN: Final[str] = "client_id"


# --------------------------------------------------------------------------- #
# Coded categorical values, straight from the UCI documentation
# --------------------------------------------------------------------------- #

SEX_LABELS: Final[Mapping[int, str]] = {1: "Male", 2: "Female"}

#: The documentation defines 1-4 only. Codes 0, 5 and 6 are present in the real
#: file but undocumented, so they are folded into "Unknown" rather than guessed.
EDUCATION_LABELS: Final[Mapping[int, str]] = {
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Others",
}
EDUCATION_UNDOCUMENTED_CODES: Final[tuple[int, ...]] = (0, 5, 6)

#: The documentation defines 1-3. Code 0 is present but undocumented.
MARRIAGE_LABELS: Final[Mapping[int, str]] = {1: "Married", 2: "Single", 3: "Others"}
MARRIAGE_UNDOCUMENTED_CODES: Final[tuple[int, ...]] = (0,)

UNKNOWN_LABEL: Final[str] = "Unknown"

#: Repayment status codes. -1 and >=1 are documented; -2 and 0 are not defined
#: in the source paper but occur frequently, and the community reading (no
#: transaction / revolving credit used) is recorded here as an interpretation,
#: explicitly flagged as such rather than presented as source truth.
PAY_STATUS_LABELS: Final[Mapping[int, str]] = {
    -2: "No transaction (undocumented code)",
    -1: "Paid in full",
    0: "Revolving credit used (undocumented code)",
    1: "1 month delay",
    2: "2 months delay",
    3: "3 months delay",
    4: "4 months delay",
    5: "5 months delay",
    6: "6 months delay",
    7: "7 months delay",
    8: "8 months delay",
    9: "9+ months delay",
}
PAY_STATUS_UNDOCUMENTED_CODES: Final[tuple[int, ...]] = (-2, 0)

#: A status code at or above this value means the client was in payment delay.
DELINQUENCY_THRESHOLD: Final[int] = 1

TARGET_LABELS: Final[Mapping[int, str]] = {0: "No default", 1: "Default"}


# --------------------------------------------------------------------------- #
# Column meanings - the single source of truth for the data dictionary
# --------------------------------------------------------------------------- #

#: Plain-language meaning of every source column, keyed by canonical name.
#: ``data_dictionary.md`` is generated from this, so the documentation cannot
#: drift away from the code.
COLUMN_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "client_id": (
        "Surrogate identifier for the credit-card client. An auto-increment key "
        "carrying no information about the client, used only to join and count."
    ),
    "credit_limit": (
        "Amount of revolving credit granted to the client, in NT$. This is a "
        "CREDIT LIMIT, not income: the dataset has no income column, so this is "
        "used as a coarse proxy for credit capacity and is always labelled as a "
        "limit."
    ),
    "sex_code": (
        "Biological sex as an integer code (1 = male, 2 = female). Retained for "
        "the fairness audit only; excluded from risk modelling because pricing "
        "credit on sex is discriminatory and unlawful in many jurisdictions."
    ),
    "sex": "Readable label derived from sex_code.",
    "education_code": (
        "Highest education level as an integer code. The source documents 1 = "
        "graduate school, 2 = university, 3 = high school, 4 = others. Codes 0, 5 "
        "and 6 also occur in the data but are undocumented."
    ),
    "education": (
        "Readable label derived from education_code. Undocumented codes (0, 5, 6) "
        "are grouped as 'Unknown' rather than guessed at."
    ),
    "marriage_code": (
        "Marital status as an integer code. The source documents 1 = married, "
        "2 = single, 3 = others. Code 0 also occurs but is undocumented. Retained "
        "for the fairness audit only; excluded from risk modelling."
    ),
    "marriage": (
        "Readable label derived from marriage_code. Undocumented code 0 is grouped "
        "as 'Unknown'."
    ),
    "age": "Client age in years at the time of the extract.",
    "default_next_month": (
        "TARGET. Whether the client defaulted on the following month's payment "
        "(1 = default, 0 = no default). This is the recorded historical outcome, "
        "never a model prediction."
    ),
}

#: Descriptions for the monthly panel families, formatted with the month label.
PANEL_COLUMN_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "bill_amt_m": (
        "Statement balance for {month} in NT$. Negative values are legitimate and "
        "mean the account is in credit, usually after an overpayment or refund."
    ),
    "pay_amt_m": (
        "Amount paid by the client during {month}, in NT$. This payment settles "
        "the statement issued in the PREVIOUS month, which is why repayment "
        "ratios divide by the month+1 balance."
    ),
    "pay_status_m": (
        "Repayment status for {month}. Documented codes: -1 = paid in full, and "
        "1 to 9 = that many months of payment delay. Codes -2 and 0 occur "
        "frequently but are not defined in the source paper; they are commonly "
        "read as 'no transaction' and 'revolving credit used'. Only codes of 1 or "
        "more are treated as delinquency, so this ambiguity does not affect the "
        "delinquency measures."
    ),
}

#: Descriptions for the flag and bookkeeping columns added during cleaning.
CLEANING_COLUMN_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "is_duplicate_excluding_id": (
        "Flag: this row is identical to another once client_id is excluded. Such "
        "rows are retained by default, because a coincidental match between two "
        "genuinely different clients is plausible with integer-coded fields."
    ),
    "has_negative_balance": (
        "Flag: the client had a negative statement balance in at least one month, "
        "meaning the account was in credit. Valid data, retained unchanged."
    ),
    "has_extreme_value": (
        "Flag: at least one monetary value for this client lies beyond a 3x IQR "
        "fence. Nothing is capped or removed; the flag exists so these clients can "
        "be isolated on request."
    ),
    "age_is_implausible": (
        "Flag: the recorded age falls outside 18-100. The value itself is retained."
    ),
    "credit_limit_original": (
        "The original credit limit before a non-positive value was set to missing, "
        "preserved for auditability."
    ),
}


def describe_column(column: str) -> str:
    """Return the plain-language meaning of a column.

    Args:
        column: Canonical column name.

    Returns:
        A description, or a generic note when the column is unrecognised.
    """
    if column in COLUMN_DESCRIPTIONS:
        return COLUMN_DESCRIPTIONS[column]
    if column in CLEANING_COLUMN_DESCRIPTIONS:
        return CLEANING_COLUMN_DESCRIPTIONS[column]

    for prefix, template in PANEL_COLUMN_DESCRIPTIONS.items():
        if column.startswith(prefix):
            suffix = column[len(prefix) :]
            if suffix.isdigit():
                return template.format(month=month_label(int(suffix)))

    if column.endswith("_was_missing"):
        source = column.removesuffix("_was_missing")
        return (
            f"Flag: the value of '{source}' was missing in the source data and has "
            "been imputed. Ensures an imputed value is never mistaken for an "
            "observed one."
        )

    return "No description registered for this column."


# --------------------------------------------------------------------------- #
# Semantic roles
# --------------------------------------------------------------------------- #


class Role(str, Enum):
    """Semantic role a column plays, independent of its physical name.

    Downstream code depends on these rather than on literal column names, so
    swapping in a different credit dataset means extending the detection rules
    in one place instead of editing every analytics module.
    """

    # Identity and demographics
    CLIENT_ID = "client_id"
    AGE = "age"
    SEX = "sex"
    EDUCATION = "education"
    MARRIAGE = "marriage"

    # Credit capacity
    CREDIT_LIMIT = "credit_limit"

    # Monthly panel families (multi-column roles)
    BILL_AMOUNT = "bill_amount"
    PAYMENT_AMOUNT = "payment_amount"
    PAYMENT_STATUS = "payment_status"

    # Target
    TARGET = "target"

    # Engineered, registered once feature engineering has run
    UTILISATION = "utilisation"
    REPAYMENT_RATIO = "repayment_ratio"
    DELINQUENCY = "delinquency"
    LIMIT_BAND = "limit_band"
    AGE_BAND = "age_band"

    # Roles this dataset genuinely does NOT have. Declared so the platform can
    # state a capability is unavailable instead of silently omitting it.
    INCOME = "income"
    SAVINGS = "savings"
    EXPENSES = "expenses"
    CREDIT_SCORE = "credit_score"
    EMPLOYMENT = "employment"
    LOAN_AMOUNT = "loan_amount"
    DATE = "date"


#: Roles that map to exactly one column.
SINGLE_COLUMN_ROLES: Final[frozenset[Role]] = frozenset(
    {
        Role.CLIENT_ID,
        Role.AGE,
        Role.SEX,
        Role.EDUCATION,
        Role.MARRIAGE,
        Role.CREDIT_LIMIT,
        Role.TARGET,
        Role.LIMIT_BAND,
        Role.AGE_BAND,
    }
)

#: Roles the brief mentions that this dataset cannot supply. Surfaced in the UI
#: and docs so the absence is explicit rather than a silent gap.
UNAVAILABLE_ROLES: Final[Mapping[Role, str]] = {
    Role.INCOME: (
        "No income column exists in this dataset. Debt-to-income and "
        "income-band analyses are therefore not computed. `credit_limit` is "
        "used as a credit-capacity proxy and is always labelled as a credit "
        "limit, never as income."
    ),
    Role.SAVINGS: "No savings column exists, so savings rate is not computed.",
    Role.EXPENSES: "No expense column exists, so expense ratio is not computed.",
    Role.CREDIT_SCORE: (
        "No bureau credit score exists. The risk model's predicted probability "
        "serves as an internal risk score and is always labelled as model output."
    ),
    Role.EMPLOYMENT: "No employment-status column exists.",
    Role.LOAN_AMOUNT: (
        "This is a revolving credit-card portfolio, not an instalment loan book, "
        "so there is no loan amount or loan term."
    ),
    Role.DATE: (
        "There is no calendar date column. The six monthly columns form a panel, "
        "which supports period-over-period comparison but not date-indexed "
        "forecasting."
    ),
}

#: Exact canonical names for the single-column roles.
_EXACT_ROLE_COLUMNS: Final[Mapping[Role, tuple[str, ...]]] = {
    Role.CLIENT_ID: ("client_id",),
    Role.AGE: ("age",),
    Role.SEX: ("sex", "sex_code"),
    Role.EDUCATION: ("education", "education_code"),
    Role.MARRIAGE: ("marriage", "marriage_code"),
    Role.CREDIT_LIMIT: ("credit_limit",),
    Role.TARGET: ("default_next_month",),
    Role.LIMIT_BAND: ("limit_band",),
    Role.AGE_BAND: ("age_band",),
}

#: Prefixes for the multi-column panel and engineered-family roles. Order
#: matters: the longest prefix is tested first so ``bill_amt_m1`` is not
#: mistaken for a utilisation column and vice versa.
_PREFIX_ROLE_COLUMNS: Final[tuple[tuple[Role, str], ...]] = (
    (Role.BILL_AMOUNT, "bill_amt_m"),
    (Role.PAYMENT_AMOUNT, "pay_amt_m"),
    (Role.PAYMENT_STATUS, "pay_status_m"),
    (Role.UTILISATION, "utilisation_m"),
    (Role.REPAYMENT_RATIO, "repayment_ratio_m"),
)

#: Engineered summary columns that register a role when present.
_ENGINEERED_ROLE_COLUMNS: Final[Mapping[Role, tuple[str, ...]]] = {
    Role.UTILISATION: ("utilisation_latest", "utilisation_mean_6m", "utilisation_max_6m"),
    Role.REPAYMENT_RATIO: ("repayment_ratio_mean",),
    Role.DELINQUENCY: (
        "delinquent_months_count",
        "max_delinquency",
        "current_delinquency",
        "ever_delinquent",
    ),
}


@dataclass(frozen=True)
class ColumnRegistry:
    """Immutable record of which semantic roles a dataframe actually provides.

    Attributes:
        columns: Every column present, in dataframe order.
        roles: Role -> the columns fulfilling it. Roles with no matching column
            are absent from this mapping entirely.
    """

    columns: tuple[str, ...]
    roles: Mapping[Role, tuple[str, ...]]

    # -------------------------- queries -------------------------- #

    def has(self, *roles: Role) -> bool:
        """True only if *every* supplied role is present."""
        return all(role in self.roles and self.roles[role] for role in roles)

    def has_any(self, *roles: Role) -> bool:
        """True if at least one supplied role is present."""
        return any(role in self.roles and self.roles[role] for role in roles)

    def one(self, role: Role) -> str | None:
        """Return the single column for ``role``, or None when absent.

        For multi-column roles this returns the first column, which for the
        panel families is month 1 - the most recent observation.
        """
        found = self.roles.get(role)
        return found[0] if found else None

    def require(self, role: Role) -> str:
        """Return the single column for ``role`` or raise.

        Raises:
            KeyError: If the role is not present. Use only where the caller has
                already checked with :meth:`has`.
        """
        column = self.one(role)
        if column is None:
            raise KeyError(
                f"Role '{role.value}' is not available in this dataset. "
                f"Present roles: {sorted(r.value for r in self.roles)}"
            )
        return column

    def all(self, role: Role) -> tuple[str, ...]:
        """Return every column fulfilling ``role`` (empty tuple when absent)."""
        return tuple(self.roles.get(role, ()))

    def missing(self, *roles: Role) -> tuple[Role, ...]:
        """Return the subset of ``roles`` that this dataset does not provide."""
        return tuple(role for role in roles if not self.has(role))

    def panel_columns(self, role: Role) -> dict[int, str]:
        """Map month index -> column name for a monthly panel role.

        Args:
            role: One of BILL_AMOUNT, PAYMENT_AMOUNT or PAYMENT_STATUS.

        Returns:
            ``{1: 'bill_amt_m1', ...}``, sorted by month index. Empty if absent.
        """
        prefix = dict((r, p) for r, p in _PREFIX_ROLE_COLUMNS).get(role)
        if prefix is None:
            return {}
        result: dict[int, str] = {}
        for column in self.all(role):
            if column.startswith(prefix):
                suffix = column[len(prefix) :]
                if suffix.isdigit():
                    result[int(suffix)] = column
        return dict(sorted(result.items()))

    def available_role_names(self) -> tuple[str, ...]:
        """Sorted role names present, for display and logging."""
        return tuple(sorted(role.value for role in self.roles))

    def unavailable_explanations(self) -> dict[str, str]:
        """Explanations for the documented roles this dataset lacks.

        Powers the honest "this analysis is not possible, and here is why"
        messaging in the UI and the report.
        """
        return {
            role.value: reason
            for role, reason in UNAVAILABLE_ROLES.items()
            if not self.has(role)
        }


def build_registry(df: pd.DataFrame) -> ColumnRegistry:
    """Inspect a dataframe and record which semantic roles it supplies.

    Detection is purely name-based against the canonical naming scheme, so it
    runs after :func:`src.data_loader.normalise_columns`. Calling it again after
    feature engineering picks up the engineered roles too.

    Args:
        df: Dataframe with canonical (snake_case) column names.

    Returns:
        A populated :class:`ColumnRegistry`.
    """
    present = list(df.columns)
    present_set = set(present)
    found: dict[Role, list[str]] = {}

    # Exact single-column matches. First candidate wins so a labelled column
    # ("education") is preferred over its raw code ("education_code").
    for role, candidates in _EXACT_ROLE_COLUMNS.items():
        for candidate in candidates:
            if candidate in present_set:
                found.setdefault(role, []).append(candidate)
                break

    # Prefixed panel families, kept in month order.
    for role, prefix in _PREFIX_ROLE_COLUMNS:
        matches = sorted(
            (c for c in present if c.startswith(prefix) and c[len(prefix) :].isdigit()),
            key=lambda c: int(c[len(prefix) :]),
        )
        if matches:
            found.setdefault(role, []).extend(matches)

    # Engineered summary columns.
    for role, candidates in _ENGINEERED_ROLE_COLUMNS.items():
        for candidate in candidates:
            if candidate in present_set:
                found.setdefault(role, []).append(candidate)

    return ColumnRegistry(
        columns=tuple(present),
        roles={role: tuple(cols) for role, cols in found.items()},
    )


def month_label(month_index: int) -> str:
    """Human-readable calendar label for a month index (1 = most recent)."""
    return MONTH_LABELS.get(month_index, f"Month {month_index}")


def describe_panel_alignment() -> str:
    """Explain the panel indexing and the payment/statement offset.

    Surfaced in the Methodology page because the alignment is the single most
    misread aspect of this dataset.
    """
    return (
        "Month index 1 is the most recent observation (September 2005) and index 6 "
        "is the oldest (April 2005). A payment recorded in month m settles the "
        "statement issued in month m+1, so repayment ratios divide "
        "pay_amt_m<m> by bill_amt_m<m+1>. The source file's repayment-status "
        "columns are named PAY_0, PAY_2..PAY_6 with no PAY_1; PAY_0 is the "
        "September status and is mapped to month index 1."
    )


def canonical_columns_for(raw_columns: Iterable[str]) -> dict[str, str]:
    """Build the rename mapping for whichever raw columns are actually present.

    Args:
        raw_columns: Column names as read from the source file.

    Returns:
        ``{raw_name: canonical_name}`` for recognised columns only. Unrecognised
        columns are left out so the caller can report them rather than dropping
        them silently.
    """
    return {raw: RAW_TO_CANONICAL[raw] for raw in raw_columns if raw in RAW_TO_CANONICAL}
