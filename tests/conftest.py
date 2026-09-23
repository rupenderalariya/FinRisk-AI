"""
Shared pytest fixtures.

Two tiers of fixture, deliberately:

* **Synthetic** frames built in code. Small, deterministic, and containing
  hand-placed edge cases (zero credit limit, negative balance, undocumented
  category codes). Expected values can be calculated by hand, so a test failure
  points at a real bug rather than at drifting data. These need no data file.
* **Real dataset** fixtures, marked ``integration`` and skipped automatically
  when ``data/raw`` is empty, so the suite still passes on a fresh clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `src` importable when pytest is invoked from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Output isolation
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", autouse=True)
def _isolate_output_directories(tmp_path_factory: pytest.TempPathFactory):
    """Redirect every writable output directory into a temp folder for the session.

    Without this, a test that runs the pipeline with ``persist=True`` on a small
    fixture would overwrite the real 30,000-row artefacts in ``data/processed``.
    That happened once; this fixture makes it impossible to happen again.

    ``raw_dir`` is deliberately left pointing at the real directory so the
    integration tests can still read the source dataset.
    """
    import dataclasses

    from _pytest.monkeypatch import MonkeyPatch

    import src.config as config_module

    session_root = tmp_path_factory.mktemp("finrisk_test_outputs")
    redirected = dataclasses.replace(
        config_module.settings,
        processed_dir=session_root / "processed",
        models_dir=session_root / "models",
        log_dir=session_root / "logs",
    )

    patcher = MonkeyPatch()
    patcher.setattr(config_module, "settings", redirected)
    # Each module binds `settings` at import time, so patch every alias.
    for module_name in ("src.data_processing", "src.pipeline", "src.data_loader"):
        try:
            patcher.setattr(f"{module_name}.settings", redirected)
        except AttributeError:  # pragma: no cover - module without the alias
            pass

    yield redirected
    patcher.undo()


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #


@pytest.fixture
def minimal_credit_df() -> pd.DataFrame:
    """A tiny, fully valid credit frame with canonical column names.

    Six clients, six months, no edge cases. Small enough that every expected
    value in a test can be worked out by hand.
    """
    n = 6
    data: dict[str, list] = {
        "client_id": list(range(1, n + 1)),
        "credit_limit": [10_000, 20_000, 50_000, 100_000, 200_000, 500_000],
        "sex_code": [1, 2, 1, 2, 1, 2],
        "education_code": [1, 2, 3, 4, 1, 2],
        "marriage_code": [1, 2, 3, 1, 2, 1],
        "age": [25, 32, 41, 55, 63, 29],
        "default_next_month": [1, 0, 1, 0, 0, 0],
    }
    # Balances descend from month 1 (most recent) to month 6 (oldest), so every
    # client has a rising balance over calendar time.
    for month in range(1, 7):
        factor = 7 - month
        data[f"bill_amt_m{month}"] = [1_000 * factor * (i + 1) for i in range(n)]
        data[f"pay_amt_m{month}"] = [500 * factor for _ in range(n)]
        data[f"pay_status_m{month}"] = [0, 0, 1, -1, 2, 0]

    return pd.DataFrame(data)


@pytest.fixture
def edge_case_credit_df() -> pd.DataFrame:
    """A frame packed with the awkward cases the pipeline must survive.

    Row-by-row:

    0. Zero credit limit           -> utilisation must be NaN, never inf.
    1. Negative statement balances -> valid credit balance, must be preserved.
    2. Undocumented education (5) and marriage (0) codes -> become "Unknown".
    3. Zero prior balance          -> repayment ratio must be NaN.
    4. Over-limit balance          -> utilisation above 1.0, must be kept.
    5. Duplicate of row 4 but for the client_id -> uniqueness check.
    6. Negative credit limit       -> must be neutralised to NaN.
    7. Implausible age (150)       -> flagged, not deleted.
    """
    rows = [
        {"client_id": 1, "credit_limit": 0, "age": 30, "sex_code": 1,
         "education_code": 1, "marriage_code": 1, "default_next_month": 0,
         "bills": [5_000] * 6, "pays": [100] * 6, "status": [0] * 6},
        {"client_id": 2, "credit_limit": 50_000, "age": 45, "sex_code": 2,
         "education_code": 2, "marriage_code": 2, "default_next_month": 0,
         "bills": [-2_000] * 6, "pays": [0] * 6, "status": [-1] * 6},
        {"client_id": 3, "credit_limit": 80_000, "age": 28, "sex_code": 1,
         "education_code": 5, "marriage_code": 0, "default_next_month": 1,
         "bills": [40_000] * 6, "pays": [2_000] * 6, "status": [2] * 6},
        {"client_id": 4, "credit_limit": 30_000, "age": 52, "sex_code": 2,
         "education_code": 3, "marriage_code": 1, "default_next_month": 0,
         "bills": [0] * 6, "pays": [500] * 6, "status": [0] * 6},
        {"client_id": 5, "credit_limit": 10_000, "age": 38, "sex_code": 1,
         "education_code": 1, "marriage_code": 2, "default_next_month": 1,
         "bills": [15_000] * 6, "pays": [1_000] * 6, "status": [3] * 6},
        {"client_id": 6, "credit_limit": 10_000, "age": 38, "sex_code": 1,
         "education_code": 1, "marriage_code": 2, "default_next_month": 1,
         "bills": [15_000] * 6, "pays": [1_000] * 6, "status": [3] * 6},
        {"client_id": 7, "credit_limit": -5_000, "age": 41, "sex_code": 2,
         "education_code": 2, "marriage_code": 3, "default_next_month": 0,
         "bills": [7_000] * 6, "pays": [700] * 6, "status": [1] * 6},
        {"client_id": 8, "credit_limit": 60_000, "age": 150, "sex_code": 1,
         "education_code": 6, "marriage_code": 1, "default_next_month": 0,
         "bills": [20_000] * 6, "pays": [3_000] * 6, "status": [0] * 6},
    ]

    records: list[dict] = []
    for row in rows:
        record = {
            key: row[key]
            for key in (
                "client_id", "credit_limit", "age", "sex_code",
                "education_code", "marriage_code", "default_next_month",
            )
        }
        for month in range(1, 7):
            record[f"bill_amt_m{month}"] = row["bills"][month - 1]
            record[f"pay_amt_m{month}"] = row["pays"][month - 1]
            record[f"pay_status_m{month}"] = row["status"][month - 1]
        records.append(record)

    return pd.DataFrame(records)


@pytest.fixture
def analysis_df() -> pd.DataFrame:
    """A 400-row synthetic frame large enough for statistical tests.

    Built with a fixed seed and a deliberate signal: utilisation and delinquency
    both rise with the target, so driver analysis has something real to find and
    the tests can assert on a known direction rather than on noise.
    """
    rng = np.random.default_rng(42)
    n = 400

    default_flag = rng.binomial(1, 0.25, n)
    credit_limit = rng.choice([20_000, 50_000, 100_000, 200_000, 400_000], n)

    # Defaulters get systematically higher utilisation.
    base_utilisation = np.where(default_flag == 1, 0.75, 0.35)
    utilisation = np.clip(base_utilisation + rng.normal(0, 0.15, n), 0.01, 1.4)

    data: dict[str, np.ndarray | list] = {
        "client_id": np.arange(1, n + 1),
        "credit_limit": credit_limit,
        "age": rng.integers(21, 70, n),
        "sex_code": rng.choice([1, 2], n),
        "education_code": rng.choice([1, 2, 3, 4], n),
        "marriage_code": rng.choice([1, 2, 3], n),
        "default_next_month": default_flag,
    }

    for month in range(1, 7):
        # Older months carry slightly smaller balances, mirroring the real data.
        decay = 1.0 - 0.04 * (month - 1)
        data[f"bill_amt_m{month}"] = np.round(credit_limit * utilisation * decay).astype("int64")
        data[f"pay_amt_m{month}"] = np.round(
            credit_limit * utilisation * decay * rng.uniform(0.05, 0.4, n)
        ).astype("int64")
        delay = np.where(default_flag == 1, rng.integers(0, 4, n), rng.integers(-1, 1, n))
        data[f"pay_status_m{month}"] = delay.astype("int64")

    return pd.DataFrame(data)


@pytest.fixture
def raw_style_df() -> pd.DataFrame:
    """A frame using the ORIGINAL source headers, for normalisation tests."""
    return pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "LIMIT_BAL": [20_000, 50_000, 100_000],
            "SEX": [1, 2, 1],
            "EDUCATION": [2, 2, 3],
            "MARRIAGE": [1, 2, 1],
            "AGE": [24, 35, 48],
            "PAY_0": [2, 0, -1],
            "PAY_2": [2, 0, -1],
            "PAY_3": [0, 0, 0],
            "PAY_4": [0, 0, 0],
            "PAY_5": [0, 0, 0],
            "PAY_6": [0, 0, 0],
            **{f"BILL_AMT{m}": [1_000 * m, 2_000 * m, 3_000 * m] for m in range(1, 7)},
            **{f"PAY_AMT{m}": [100 * m, 200 * m, 300 * m] for m in range(1, 7)},
            "default payment next month": [1, 0, 0],
        }
    )


@pytest.fixture
def featured_df(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """The 400-row frame after cleaning and feature engineering."""
    from src.data_processing import clean_dataset
    from src.feature_engineering import engineer_features

    clean, _ = clean_dataset(analysis_df)
    features, _ = engineer_features(clean)
    return features


@pytest.fixture
def small_featured_df(minimal_credit_df: pd.DataFrame) -> pd.DataFrame:
    """The 6-row frame after cleaning and feature engineering."""
    from src.data_processing import clean_dataset
    from src.feature_engineering import engineer_features

    clean, _ = clean_dataset(minimal_credit_df)
    features, _ = engineer_features(clean)
    return features


# --------------------------------------------------------------------------- #
# Real dataset
# --------------------------------------------------------------------------- #


def _real_dataset_path() -> Path | None:
    """Locate the real dataset, or None when it has not been downloaded."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        return None
    candidates = [
        p
        for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".xls", ".xlsx", ".csv"}
    ]
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


@pytest.fixture(scope="session")
def real_dataset_path() -> Path:
    """Path to the real dataset, skipping the test when it is absent."""
    path = _real_dataset_path()
    if path is None:
        pytest.skip(
            "Real dataset not present. Run `python scripts/download_data.py` to "
            "enable integration tests."
        )
    return path


@pytest.fixture(scope="session")
def real_pipeline(real_dataset_path: Path):
    """The full pipeline run once against the real dataset.

    Session-scoped so 30,000 rows are processed a single time for the whole suite.
    ``persist=False`` keeps the test run from overwriting real artefacts.
    """
    from src.pipeline import run_pipeline

    return run_pipeline(path=real_dataset_path, persist=False)


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_csv(tmp_path: Path, raw_style_df: pd.DataFrame) -> Path:
    """A valid CSV written with the original source headers."""
    destination = tmp_path / "credit_data.csv"
    raw_style_df.to_csv(destination, index=False)
    return destination


@pytest.fixture
def tmp_banner_csv(tmp_path: Path, raw_style_df: pd.DataFrame) -> Path:
    """A CSV with a merged-banner first row, as the real ``.xls`` file has."""
    destination = tmp_path / "banner_data.csv"
    header = ",".join(raw_style_df.columns)
    banner = "X1,X2" + "," * (len(raw_style_df.columns) - 2)
    body = raw_style_df.to_csv(index=False, header=False)
    destination.write_text(f"{banner}\n{header}\n{body}", encoding="utf-8")
    return destination


@pytest.fixture
def tmp_empty_dir(tmp_path: Path) -> Path:
    """An existing but empty directory."""
    target = tmp_path / "empty_raw"
    target.mkdir()
    return target
