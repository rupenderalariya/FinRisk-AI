"""
Dataset discovery, loading, validation and profiling.

Responsibilities, in order:

1. **Locate** the dataset in ``data/raw`` without the caller knowing the filename.
2. **Load** it safely across ``.xls`` / ``.xlsx`` / ``.csv``, coping with the
   banner row and encoding variation found in real source files.
3. **Normalise** headers to the canonical snake_case names in :mod:`src.schema`.
4. **Validate** the result against the expected contract, loudly and clearly.
5. **Profile** it: dimensions, dtype groupings, missing counts, duplicates,
   uniqueness.

The raw file is opened read-only and never written to. Every cleaning or
engineering step happens on an in-memory copy and is persisted separately under
``data/processed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

import pandas as pd

from src.config import settings
from src.logging_setup import get_logger
from src.schema import (
    DATASET_CURRENCY,
    DATASET_NAME,
    DATASET_SOURCE,
    EXPECTED_RAW_COLUMNS,
    EXPECTED_RAW_ROWS,
    ID_COLUMN,
    RAW_TO_CANONICAL,
    TARGET_COLUMN,
    canonical_columns_for,
)

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: Final[tuple[str, ...]] = (".xls", ".xlsx", ".csv")

#: Encodings tried in order for CSV input. cp1252/latin-1 never fail outright,
#: so they act as the terminal fallback.
_CSV_ENCODINGS: Final[tuple[str, ...]] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

#: Header rows to try for spreadsheets. The UCI file carries a merged banner in
#: row 0 with the real header in row 1, so index 1 is attempted first.
_HEADER_CANDIDATES: Final[tuple[int, ...]] = (1, 0, 2)

#: A candidate header must resolve at least this many recognised columns to be
#: accepted. Guards against locking on to a junk row.
_MIN_RECOGNISED_COLUMNS: Final[int] = 5


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class DataLoadError(Exception):
    """Raised when the dataset cannot be found, read or parsed.

    Messages are written for a dashboard user, not just a developer: they say
    what was looked for, what was found, and what to do next.
    """


class DataValidationError(Exception):
    """Raised when a dataset loads but violates the expected contract."""


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetProfile:
    """Structural description of a loaded dataframe.

    Purely factual - counts and types, no judgement. Quality scoring lives in
    :mod:`src.data_quality`.
    """

    n_rows: int
    n_columns: int
    column_names: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    boolean_columns: tuple[str, ...]
    total_missing_values: int
    missing_by_column: dict[str, int]
    duplicate_rows: int
    duplicate_rows_excluding_id: int
    unique_counts: dict[str, int]
    constant_columns: tuple[str, ...]
    empty_columns: tuple[str, ...]
    memory_usage_mb: float
    source_path: str | None = None
    source_format: str | None = None
    unrecognised_columns: tuple[str, ...] = field(default=())

    @property
    def missing_percentage(self) -> float:
        """Missing cells as a percentage of all cells (0.0 for an empty frame)."""
        total_cells = self.n_rows * self.n_columns
        if total_cells == 0:
            return 0.0
        return self.total_missing_values / total_cells * 100.0

    @property
    def duplicate_percentage(self) -> float:
        """Fully duplicated rows as a percentage of all rows."""
        if self.n_rows == 0:
            return 0.0
        return self.duplicate_rows / self.n_rows * 100.0

    def columns_with_missing(self) -> dict[str, int]:
        """Only the columns that actually have missing values."""
        return {col: count for col, count in self.missing_by_column.items() if count > 0}

    def to_dict(self) -> dict[str, Any]:
        """Flat serialisable summary for the UI, logs and the AI payload."""
        return {
            "rows": self.n_rows,
            "columns": self.n_columns,
            "numeric_columns": len(self.numeric_columns),
            "categorical_columns": len(self.categorical_columns),
            "datetime_columns": len(self.datetime_columns),
            "boolean_columns": len(self.boolean_columns),
            "total_missing_values": self.total_missing_values,
            "missing_percentage": round(self.missing_percentage, 4),
            "duplicate_rows": self.duplicate_rows,
            "duplicate_rows_excluding_id": self.duplicate_rows_excluding_id,
            "constant_columns": list(self.constant_columns),
            "empty_columns": list(self.empty_columns),
            "memory_usage_mb": round(self.memory_usage_mb, 3),
            "source_format": self.source_format,
            "unrecognised_columns": list(self.unrecognised_columns),
        }


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def find_dataset(
    raw_dir: Path | None = None,
    filename: str | None = None,
) -> Path:
    """Locate the raw dataset file.

    Resolution order: an explicit ``filename`` argument, then
    ``RAW_DATASET_FILENAME`` from configuration, then auto-discovery of the
    largest supported file in ``raw_dir`` (the dataset is reliably the biggest
    file next to any stray notes or archives).

    Args:
        raw_dir: Directory to search. Defaults to the configured raw directory.
        filename: Exact filename to use, bypassing discovery.

    Returns:
        Path to the dataset file.

    Raises:
        DataLoadError: If the directory or a suitable file is missing.
    """
    directory = raw_dir or settings.raw_dir
    explicit = filename or settings.raw_dataset_filename

    if not directory.exists():
        raise DataLoadError(
            f"The raw data directory does not exist: {directory}\n"
            "Run `python scripts/download_data.py` to fetch the dataset."
        )

    if explicit:
        candidate = directory / explicit
        if not candidate.exists():
            raise DataLoadError(
                f"The configured dataset file was not found: {candidate}\n"
                f"Files present in {directory}: "
                f"{sorted(p.name for p in directory.iterdir() if p.is_file()) or 'none'}"
            )
        logger.info("Using explicitly configured dataset file: %s", candidate.name)
        return candidate

    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith("~$")  # Excel lock files
    ]

    if not candidates:
        raise DataLoadError(
            f"No dataset file found in {directory}.\n"
            f"Expected one of: {', '.join(SUPPORTED_EXTENSIONS)}\n"
            "Run `python scripts/download_data.py` to fetch the dataset."
        )

    chosen = max(candidates, key=lambda p: p.stat().st_size)
    if len(candidates) > 1:
        logger.info(
            "Found %d candidate files; selected the largest: %s",
            len(candidates),
            chosen.name,
        )
    return chosen


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _count_recognised(columns: Sequence[Any]) -> int:
    """Count how many columns map to a known canonical name."""
    return sum(1 for col in columns if str(col).strip() in RAW_TO_CANONICAL)


def _looks_like_placeholder_header(columns: Sequence[Any]) -> bool:
    """True if a header row is mostly pandas-generated placeholders.

    A merged banner row produces ``Unnamed: 1``, ``Unnamed: 2`` and so on, which
    is the signal that the real header sits one row lower.
    """
    if len(columns) == 0:
        return True
    unnamed = sum(1 for col in columns if str(col).startswith("Unnamed"))
    return unnamed > len(columns) / 2


def _read_excel_with_header_detection(path: Path) -> tuple[pd.DataFrame, int]:
    """Read a spreadsheet, discovering which row holds the real header.

    Each candidate row is scored by how many recognised columns it yields, so
    the choice is evidence-based rather than a hard-coded ``header=1``.

    Args:
        path: Spreadsheet path.

    Returns:
        ``(dataframe, header_row_index)``.

    Raises:
        DataLoadError: If no candidate row produces a usable header.
    """
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    best: tuple[int, pd.DataFrame, int] | None = None
    errors: list[str] = []

    for header_row in _HEADER_CANDIDATES:
        try:
            frame = pd.read_excel(path, header=header_row, engine=engine)
        except Exception as exc:  # noqa: BLE001 - surface any reader failure as context
            errors.append(f"header={header_row}: {type(exc).__name__}: {exc}")
            continue

        if frame.empty or _looks_like_placeholder_header(frame.columns):
            continue

        score = _count_recognised(frame.columns)
        logger.debug("header=%d yielded %d recognised columns", header_row, score)

        if best is None or score > best[0]:
            best = (score, frame, header_row)

        # A full match means there is nothing better to find.
        if score >= len(RAW_TO_CANONICAL) - 2:
            break

    if best is None:
        detail = "; ".join(errors) if errors else "no candidate row produced a usable header"
        raise DataLoadError(
            f"Could not determine the header row for {path.name}. Details: {detail}"
        )

    score, frame, header_row = best
    if score < _MIN_RECOGNISED_COLUMNS:
        logger.warning(
            "Header row %d matched only %d known columns. The file may not be the "
            "expected dataset; validation will report specifics.",
            header_row,
            score,
        )
    logger.info(
        "Read %s with header row %d (%d of %d columns recognised).",
        path.name,
        header_row,
        score,
        len(frame.columns),
    )
    return frame, header_row


def _read_csv_with_encoding_detection(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a CSV, trying encodings in turn until one succeeds.

    Args:
        path: CSV path.

    Returns:
        ``(dataframe, encoding_used)``.

    Raises:
        DataLoadError: If every candidate encoding fails.
    """
    errors: list[str] = []
    for encoding in _CSV_ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError(f"{path.name} contains no data: {exc}") from exc
        except pd.errors.ParserError as exc:
            raise DataLoadError(
                f"{path.name} is not valid CSV - the delimiter or quoting may be "
                f"malformed. Parser said: {exc}"
            ) from exc

        # A banner row shows up in CSV too; retry one row down if so.
        if _looks_like_placeholder_header(frame.columns) or (
            _count_recognised(frame.columns) == 0 and len(frame) > 1
        ):
            retry = pd.read_csv(path, encoding=encoding, header=1)
            if _count_recognised(retry.columns) > _count_recognised(frame.columns):
                frame = retry
                logger.info("Used header row 1 for %s (row 0 looked like a banner).", path.name)

        logger.info("Read %s as CSV using encoding '%s'.", path.name, encoding)
        return frame, encoding

    raise DataLoadError(
        f"Could not decode {path.name} with any of: {', '.join(_CSV_ENCODINGS)}. "
        f"Details: {'; '.join(errors)}"
    )


def load_raw_dataframe(path: Path | str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the raw dataset exactly as stored, with no renaming or cleaning.

    Args:
        path: Dataset path. When omitted, :func:`find_dataset` discovers it.

    Returns:
        ``(dataframe, read_metadata)`` where the metadata records the resolved
        path, format, and how the header/encoding were determined.

    Raises:
        DataLoadError: On a missing, unreadable, unsupported or empty file.
    """
    resolved = Path(path) if path is not None else find_dataset()

    if not resolved.exists():
        raise DataLoadError(
            f"Dataset file not found: {resolved}\n"
            "Run `python scripts/download_data.py` to fetch it."
        )
    if resolved.is_dir():
        raise DataLoadError(f"Expected a file but got a directory: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file format '{suffix}' for {resolved.name}. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if resolved.stat().st_size == 0:
        raise DataLoadError(f"{resolved.name} is empty (0 bytes).")

    metadata: dict[str, Any] = {
        "source_path": str(resolved),
        "source_format": suffix.lstrip("."),
        "file_size_bytes": resolved.stat().st_size,
    }

    if suffix in {".xls", ".xlsx"}:
        frame, header_row = _read_excel_with_header_detection(resolved)
        metadata["header_row"] = header_row
        metadata["engine"] = "xlrd" if suffix == ".xls" else "openpyxl"
    else:
        frame, encoding = _read_csv_with_encoding_detection(resolved)
        metadata["encoding"] = encoding

    if frame.empty:
        raise DataLoadError(
            f"{resolved.name} parsed successfully but contains no data rows."
        )

    logger.info(
        "Loaded raw dataset: %d rows x %d columns from %s",
        len(frame),
        len(frame.columns),
        resolved.name,
    )
    return frame, metadata


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def _fallback_snake_case(name: str) -> str:
    """Convert an unrecognised header to a safe snake_case identifier."""
    cleaned = str(name).strip().lower()
    for char in (" ", "-", ".", "/", "\\", "(", ")", "[", "]", ":", ";", ","):
        cleaned = cleaned.replace(char, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unnamed_column"


def normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Rename raw headers to canonical names, reporting anything unrecognised.

    Recognised headers are mapped via :data:`src.schema.RAW_TO_CANONICAL`.
    Unrecognised ones are snake_cased and **kept** - dropping columns silently
    would hide information the analyst may need.

    Args:
        df: Raw dataframe straight from the reader.

    Returns:
        ``(renamed_dataframe, unrecognised_original_names)``.
    """
    stripped = df.rename(columns=lambda c: str(c).strip())
    mapping = canonical_columns_for(stripped.columns)
    unrecognised = tuple(str(c) for c in stripped.columns if str(c) not in mapping)

    full_mapping = dict(mapping)
    for original in unrecognised:
        full_mapping[original] = _fallback_snake_case(original)

    renamed = stripped.rename(columns=full_mapping)

    if unrecognised:
        logger.warning(
            "%d column(s) were not recognised and have been kept with snake_case "
            "names: %s",
            len(unrecognised),
            ", ".join(unrecognised[:10]),
        )
    logger.info("Normalised %d column names to canonical form.", len(mapping))
    return renamed, unrecognised


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a loaded dataset against the expected contract."""

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checks_run: int

    def summary(self) -> str:
        """One-line human-readable verdict."""
        if self.is_valid and not self.warnings:
            return f"All {self.checks_run} validation checks passed."
        if self.is_valid:
            return (
                f"{self.checks_run} checks run, passed with "
                f"{len(self.warnings)} warning(s)."
            )
        return f"Validation failed: {len(self.errors)} error(s) across {self.checks_run} checks."

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the UI and logs."""
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks_run": self.checks_run,
            "summary": self.summary(),
        }


def validate_dataset(df: pd.DataFrame, strict: bool = False) -> ValidationResult:
    """Check a canonically named dataframe against the expected contract.

    The split between error and warning is deliberate. Errors mean the data
    cannot be analysed meaningfully (no rows, no target). Warnings mean it
    differs from the reference dataset but remains analysable - which keeps the
    platform usable on a filtered subset or a similar dataset.

    Args:
        df: Dataframe with canonical column names.
        strict: Treat warnings as errors. Used by the download verification step.

    Returns:
        A :class:`ValidationResult`.
    """
    errors: list[str] = []
    warnings: list[str] = []
    checks = 0

    # --- structure ---
    checks += 1
    if df.empty:
        errors.append("The dataset contains no rows.")

    checks += 1
    if len(df.columns) == 0:
        errors.append("The dataset contains no columns.")

    checks += 1
    duplicated_names = df.columns[df.columns.duplicated()].tolist()
    if duplicated_names:
        errors.append(f"Duplicate column names present: {duplicated_names}")

    # --- expected shape (reference dataset) ---
    checks += 1
    if not df.empty and len(df) != EXPECTED_RAW_ROWS:
        warnings.append(
            f"Row count is {len(df):,}, but the reference dataset has "
            f"{EXPECTED_RAW_ROWS:,}. This is expected for a filtered or partial extract."
        )

    checks += 1
    if len(df.columns) != EXPECTED_RAW_COLUMNS:
        warnings.append(
            f"Column count is {len(df.columns)}, but the reference dataset has "
            f"{EXPECTED_RAW_COLUMNS}."
        )

    # --- target ---
    checks += 1
    if TARGET_COLUMN not in df.columns:
        errors.append(
            f"The target column '{TARGET_COLUMN}' is missing. Credit-risk "
            "analysis cannot proceed without it."
        )
    else:
        checks += 1
        target_values = set(df[TARGET_COLUMN].dropna().unique().tolist())
        unexpected = target_values - {0, 1}
        if unexpected:
            errors.append(
                f"The target '{TARGET_COLUMN}' must be binary 0/1 but also contains: "
                f"{sorted(unexpected)}"
            )
        checks += 1
        if len(target_values) < 2:
            warnings.append(
                f"The target contains only one class ({target_values}). Predictive "
                "modelling requires both classes; descriptive analytics still work."
            )

    # --- identifier ---
    checks += 1
    if ID_COLUMN not in df.columns:
        warnings.append(
            f"No '{ID_COLUMN}' column found. Per-client views will fall back to "
            "the row index."
        )
    elif df[ID_COLUMN].duplicated().any():
        count = int(df[ID_COLUMN].duplicated().sum())
        warnings.append(
            f"'{ID_COLUMN}' is not unique: {count:,} repeated value(s). It will not "
            "be treated as a reliable key."
        )

    # --- core analytical columns ---
    essential = ["credit_limit", "age"]
    for column in essential:
        checks += 1
        if column not in df.columns:
            warnings.append(
                f"Expected analytical column '{column}' is absent. Dependent KPIs "
                "and features will be skipped."
            )

    # --- panel completeness ---
    for prefix, label in (
        ("bill_amt_m", "statement balance"),
        ("pay_amt_m", "payment amount"),
        ("pay_status_m", "repayment status"),
    ):
        checks += 1
        found = [c for c in df.columns if c.startswith(prefix)]
        if not found:
            warnings.append(
                f"No {label} columns found (prefix '{prefix}'). Trend and "
                "behavioural analysis for this family will be unavailable."
            )
        elif len(found) < 6:
            warnings.append(
                f"Only {len(found)} of 6 {label} months present. Panel analysis "
                "will use the months available."
            )

    # --- value sanity ---
    if "age" in df.columns:
        checks += 1
        numeric_age = pd.to_numeric(df["age"], errors="coerce")
        implausible = int(((numeric_age < 18) | (numeric_age > 100)).sum())
        if implausible:
            warnings.append(
                f"{implausible:,} row(s) have an age outside 18-100. These are "
                "flagged by the data-quality report rather than removed."
            )

    if "credit_limit" in df.columns:
        checks += 1
        numeric_limit = pd.to_numeric(df["credit_limit"], errors="coerce")
        non_positive = int((numeric_limit <= 0).sum())
        if non_positive:
            warnings.append(
                f"{non_positive:,} row(s) have a non-positive credit limit. "
                "Utilisation cannot be computed for these and will be null."
            )

    if strict:
        errors.extend(warnings)
        warnings = []

    result = ValidationResult(
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks_run=checks,
    )

    if result.is_valid:
        logger.info("Validation passed (%d checks, %d warnings).", checks, len(result.warnings))
    else:
        logger.error("Validation failed: %s", " | ".join(result.errors))
    for warning in result.warnings:
        logger.warning("Validation: %s", warning)

    return result


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #


def profile_dataset(
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    unrecognised_columns: Sequence[str] = (),
) -> DatasetProfile:
    """Compute the structural profile of a dataframe.

    Args:
        df: Dataframe to profile.
        metadata: Optional read metadata from :func:`load_raw_dataframe`.
        unrecognised_columns: Original names that had no canonical mapping.

    Returns:
        A populated :class:`DatasetProfile`.
    """
    meta = metadata or {}

    numeric = tuple(df.select_dtypes(include="number").columns)
    boolean = tuple(df.select_dtypes(include="bool").columns)
    datetime_cols = tuple(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    categorical = tuple(
        c for c in df.select_dtypes(include=["object", "string", "category"]).columns
    )

    missing_by_column = {str(col): int(df[col].isna().sum()) for col in df.columns}

    duplicate_rows = int(df.duplicated().sum()) if len(df) else 0

    # Duplicates ignoring the surrogate key are the analytically interesting
    # ones: identical clients carrying different IDs.
    if ID_COLUMN in df.columns and len(df.columns) > 1 and len(df):
        duplicate_excluding_id = int(df.drop(columns=[ID_COLUMN]).duplicated().sum())
    else:
        duplicate_excluding_id = duplicate_rows

    unique_counts = {str(col): int(df[col].nunique(dropna=True)) for col in df.columns}

    constant = tuple(col for col, count in unique_counts.items() if count == 1 and len(df) > 1)
    empty = tuple(
        col for col in df.columns if len(df) > 0 and missing_by_column[str(col)] == len(df)
    )

    profile = DatasetProfile(
        n_rows=int(len(df)),
        n_columns=int(len(df.columns)),
        column_names=tuple(str(c) for c in df.columns),
        numeric_columns=tuple(str(c) for c in numeric),
        categorical_columns=tuple(str(c) for c in categorical),
        datetime_columns=tuple(str(c) for c in datetime_cols),
        boolean_columns=tuple(str(c) for c in boolean),
        total_missing_values=int(sum(missing_by_column.values())),
        missing_by_column=missing_by_column,
        duplicate_rows=duplicate_rows,
        duplicate_rows_excluding_id=duplicate_excluding_id,
        unique_counts=unique_counts,
        constant_columns=constant,
        empty_columns=empty,
        memory_usage_mb=float(df.memory_usage(deep=True).sum()) / (1024**2),
        source_path=meta.get("source_path"),
        source_format=meta.get("source_format"),
        unrecognised_columns=tuple(str(c) for c in unrecognised_columns),
    )

    logger.info(
        "Profiled dataset: %d rows, %d cols, %d numeric, %d categorical, "
        "%d missing cells, %d duplicate rows.",
        profile.n_rows,
        profile.n_columns,
        len(profile.numeric_columns),
        len(profile.categorical_columns),
        profile.total_missing_values,
        profile.duplicate_rows,
    )
    return profile


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoadResult:
    """Everything the load stage produces, bundled for the pipeline."""

    data: pd.DataFrame
    profile: DatasetProfile
    validation: ValidationResult
    metadata: dict[str, Any]

    @property
    def dataset_label(self) -> str:
        """Display label naming the dataset and its source."""
        return f"{DATASET_NAME} ({DATASET_SOURCE})"

    @property
    def currency(self) -> str:
        """Currency symbol for monetary formatting."""
        return DATASET_CURRENCY


def load_dataset(
    path: Path | str | None = None,
    validate: bool = True,
    raise_on_invalid: bool = True,
) -> LoadResult:
    """Load, normalise, validate and profile the dataset in one call.

    This is the single entry point the pipeline uses.

    Args:
        path: Dataset path. Auto-discovered when omitted.
        validate: Run contract validation.
        raise_on_invalid: Raise :class:`DataValidationError` when validation
            fails. Set False to inspect the failure yourself.

    Returns:
        A :class:`LoadResult`.

    Raises:
        DataLoadError: If the file cannot be found or read.
        DataValidationError: If validation fails and ``raise_on_invalid`` is set.
    """
    frame, metadata = load_raw_dataframe(path)
    normalised, unrecognised = normalise_columns(frame)

    if validate:
        validation = validate_dataset(normalised)
        if not validation.is_valid and raise_on_invalid:
            raise DataValidationError(
                "The dataset failed validation and cannot be analysed:\n  - "
                + "\n  - ".join(validation.errors)
            )
    else:
        validation = ValidationResult(is_valid=True, errors=(), warnings=(), checks_run=0)

    profile = profile_dataset(normalised, metadata, unrecognised)
    return LoadResult(
        data=normalised,
        profile=profile,
        validation=validation,
        metadata=metadata,
    )
