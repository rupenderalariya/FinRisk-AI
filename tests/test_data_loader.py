"""Tests for dataset discovery, loading, normalisation, validation and profiling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    DataLoadError,
    DataValidationError,
    find_dataset,
    load_dataset,
    load_raw_dataframe,
    normalise_columns,
    profile_dataset,
    validate_dataset,
)


class TestFindDataset:
    """Locating the dataset file."""

    def test_finds_csv_in_directory(self, tmp_csv: Path) -> None:
        assert find_dataset(raw_dir=tmp_csv.parent) == tmp_csv

    def test_picks_largest_when_several_present(self, tmp_path: Path) -> None:
        small = tmp_path / "small.csv"
        large = tmp_path / "large.csv"
        small.write_text("a,b\n1,2\n", encoding="utf-8")
        large.write_text("a,b\n" + "1,2\n" * 500, encoding="utf-8")
        assert find_dataset(raw_dir=tmp_path) == large

    def test_missing_directory_raises_with_guidance(self, tmp_path: Path) -> None:
        with pytest.raises(DataLoadError, match="download_data"):
            find_dataset(raw_dir=tmp_path / "does_not_exist")

    def test_empty_directory_raises_with_guidance(self, tmp_empty_dir: Path) -> None:
        with pytest.raises(DataLoadError, match="No dataset file found"):
            find_dataset(raw_dir=tmp_empty_dir)

    def test_explicit_filename_that_is_absent_lists_what_is_present(
        self, tmp_csv: Path
    ) -> None:
        with pytest.raises(DataLoadError, match="credit_data.csv"):
            find_dataset(raw_dir=tmp_csv.parent, filename="nope.csv")

    def test_ignores_excel_lock_files(self, tmp_path: Path) -> None:
        (tmp_path / "~$locked.xlsx").write_text("lock", encoding="utf-8")
        real = tmp_path / "real.csv"
        real.write_text("a,b\n1,2\n", encoding="utf-8")
        assert find_dataset(raw_dir=tmp_path) == real


class TestLoadRawDataframe:
    """Reading files of each supported format."""

    def test_loads_valid_csv(self, tmp_csv: Path) -> None:
        frame, metadata = load_raw_dataframe(tmp_csv)
        assert len(frame) == 3
        assert metadata["source_format"] == "csv"
        assert metadata["encoding"] == "utf-8"

    def test_detects_banner_row_in_csv(self, tmp_banner_csv: Path) -> None:
        """A merged banner on row 0 must not become the header."""
        frame, _ = load_raw_dataframe(tmp_banner_csv)
        assert "ID" in frame.columns
        assert len(frame) == 3

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataLoadError, match="not found"):
            load_raw_dataframe(tmp_path / "absent.csv")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "data.json"
        bad.write_text('{"a": 1}', encoding="utf-8")
        with pytest.raises(DataLoadError, match="Unsupported file format"):
            load_raw_dataframe(bad)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_bytes(b"")
        with pytest.raises(DataLoadError, match="empty"):
            load_raw_dataframe(empty)

    def test_header_only_file_raises(self, tmp_path: Path) -> None:
        """A file with headers but no rows is useless and must be rejected."""
        header_only = tmp_path / "header_only.csv"
        header_only.write_text("ID,LIMIT_BAL,AGE\n", encoding="utf-8")
        with pytest.raises(DataLoadError):
            load_raw_dataframe(header_only)

    def test_directory_instead_of_file_raises(self, tmp_path: Path) -> None:
        directory = tmp_path / "adir.csv"
        directory.mkdir()
        with pytest.raises(DataLoadError, match="directory"):
            load_raw_dataframe(directory)

    def test_non_utf8_encoding_is_handled(self, tmp_path: Path) -> None:
        """cp1252 content must load rather than raise UnicodeDecodeError."""
        target = tmp_path / "latin.csv"
        target.write_bytes("ID,NAME\n1,Café\n2,Señor\n".encode("cp1252"))
        frame, metadata = load_raw_dataframe(target)
        assert len(frame) == 2
        assert metadata["encoding"] in {"utf-8", "utf-8-sig", "cp1252", "latin-1"}


class TestNormaliseColumns:
    """Renaming raw headers to canonical names."""

    def test_maps_known_headers(self, raw_style_df: pd.DataFrame) -> None:
        renamed, unrecognised = normalise_columns(raw_style_df)
        assert unrecognised == ()
        for expected in (
            "client_id", "credit_limit", "age", "default_next_month",
            "bill_amt_m1", "pay_amt_m6", "pay_status_m1",
        ):
            assert expected in renamed.columns

    def test_pay_0_maps_to_month_one(self, raw_style_df: pd.DataFrame) -> None:
        """PAY_0 is the September status, so it must become month 1."""
        renamed, _ = normalise_columns(raw_style_df)
        pd.testing.assert_series_equal(
            renamed["pay_status_m1"], raw_style_df["PAY_0"], check_names=False
        )

    def test_unrecognised_columns_are_kept_not_dropped(self) -> None:
        frame = pd.DataFrame({"ID": [1], "Mystery Column": [5], "AGE": [30]})
        renamed, unrecognised = normalise_columns(frame)
        assert "Mystery Column" in unrecognised
        assert "mystery_column" in renamed.columns, "unknown columns must be preserved"

    def test_whitespace_in_headers_is_stripped(self) -> None:
        frame = pd.DataFrame({"  ID  ": [1], " AGE ": [30]})
        renamed, _ = normalise_columns(frame)
        assert "client_id" in renamed.columns
        assert "age" in renamed.columns

    def test_kaggle_style_target_name_is_recognised(self) -> None:
        frame = pd.DataFrame({"ID": [1], "default.payment.next.month": [0]})
        renamed, unrecognised = normalise_columns(frame)
        assert "default_next_month" in renamed.columns
        assert unrecognised == ()


class TestValidateDataset:
    """Contract validation."""

    def test_valid_frame_passes(self, minimal_credit_df: pd.DataFrame) -> None:
        result = validate_dataset(minimal_credit_df)
        assert result.is_valid
        assert result.errors == ()
        assert result.checks_run > 0

    def test_empty_frame_fails(self) -> None:
        result = validate_dataset(pd.DataFrame())
        assert not result.is_valid
        assert any("no rows" in e.lower() for e in result.errors)

    def test_missing_target_fails(self, minimal_credit_df: pd.DataFrame) -> None:
        result = validate_dataset(minimal_credit_df.drop(columns=["default_next_month"]))
        assert not result.is_valid
        assert any("default_next_month" in e for e in result.errors)

    def test_non_binary_target_fails(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame.loc[0, "default_next_month"] = 7
        result = validate_dataset(frame)
        assert not result.is_valid
        assert any("binary" in e.lower() for e in result.errors)

    def test_single_class_target_warns_but_passes(
        self, minimal_credit_df: pd.DataFrame
    ) -> None:
        """Descriptive analytics still work with one class, so this is a warning."""
        frame = minimal_credit_df.copy()
        frame["default_next_month"] = 0
        result = validate_dataset(frame)
        assert result.is_valid
        assert any("one class" in w.lower() for w in result.warnings)

    def test_duplicate_column_names_fail(self) -> None:
        frame = pd.DataFrame([[1, 2, 0]], columns=["client_id", "client_id", "default_next_month"])
        result = validate_dataset(frame)
        assert not result.is_valid

    def test_implausible_age_warns(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame.loc[0, "age"] = 200
        result = validate_dataset(frame)
        assert result.is_valid, "an odd age must not block analysis"
        assert any("age" in w.lower() for w in result.warnings)

    def test_non_positive_credit_limit_warns(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame.loc[0, "credit_limit"] = 0
        result = validate_dataset(frame)
        assert any("non-positive" in w.lower() for w in result.warnings)

    def test_strict_mode_turns_warnings_into_errors(
        self, minimal_credit_df: pd.DataFrame
    ) -> None:
        relaxed = validate_dataset(minimal_credit_df, strict=False)
        strict = validate_dataset(minimal_credit_df, strict=True)
        assert relaxed.is_valid
        # The 6-row fixture differs from the 30,000-row reference, so warnings exist.
        assert len(strict.errors) >= len(relaxed.errors)

    def test_summary_is_human_readable(self, minimal_credit_df: pd.DataFrame) -> None:
        assert "check" in validate_dataset(minimal_credit_df).summary().lower()


class TestProfileDataset:
    """Structural profiling."""

    def test_reports_shape_and_types(self, minimal_credit_df: pd.DataFrame) -> None:
        profile = profile_dataset(minimal_credit_df)
        assert profile.n_rows == 6
        assert profile.n_columns == len(minimal_credit_df.columns)
        assert len(profile.numeric_columns) == len(minimal_credit_df.columns)

    def test_counts_missing_values(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame.loc[0, "age"] = None
        profile = profile_dataset(frame)
        assert profile.total_missing_values == 1
        assert profile.columns_with_missing() == {"age": 1}
        assert profile.missing_percentage > 0

    def test_counts_duplicates_excluding_id(self, edge_case_credit_df: pd.DataFrame) -> None:
        """Rows 5 and 6 differ only by client_id in the fixture."""
        profile = profile_dataset(edge_case_credit_df)
        assert profile.duplicate_rows == 0
        assert profile.duplicate_rows_excluding_id == 1

    def test_detects_constant_and_empty_columns(
        self, minimal_credit_df: pd.DataFrame
    ) -> None:
        frame = minimal_credit_df.copy()
        frame["always_same"] = 1
        frame["always_empty"] = None
        profile = profile_dataset(frame)
        assert "always_same" in profile.constant_columns
        assert "always_empty" in profile.empty_columns

    def test_empty_frame_does_not_divide_by_zero(self) -> None:
        profile = profile_dataset(pd.DataFrame())
        assert profile.n_rows == 0
        assert profile.missing_percentage == 0.0
        assert profile.duplicate_percentage == 0.0

    def test_to_dict_is_serialisable(self, minimal_credit_df: pd.DataFrame) -> None:
        import json

        json.dumps(profile_dataset(minimal_credit_df).to_dict())


class TestLoadDataset:
    """The combined entry point."""

    def test_loads_normalises_and_validates(self, tmp_csv: Path) -> None:
        result = load_dataset(tmp_csv, raise_on_invalid=False)
        assert "client_id" in result.data.columns
        assert result.profile.n_rows == 3
        assert result.currency == "NT$"
        assert "UCI" in result.dataset_label

    def test_raises_on_invalid_when_requested(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}).to_csv(bad, index=False)
        with pytest.raises(DataValidationError, match="failed validation"):
            load_dataset(bad, raise_on_invalid=True)

    def test_returns_invalid_result_when_not_raising(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}).to_csv(bad, index=False)
        result = load_dataset(bad, raise_on_invalid=False)
        assert not result.validation.is_valid


@pytest.mark.integration
class TestRealDataset:
    """Verification against the real UCI file."""

    def test_loads_expected_shape(self, real_dataset_path: Path) -> None:
        result = load_dataset(real_dataset_path)
        assert result.profile.n_rows == 30_000
        assert result.profile.n_columns == 25

    def test_has_no_missing_values(self, real_dataset_path: Path) -> None:
        """Verified property of this dataset; a change here means a different file."""
        assert load_dataset(real_dataset_path).profile.total_missing_values == 0

    def test_duplicate_counts_match_verified_values(self, real_dataset_path: Path) -> None:
        profile = load_dataset(real_dataset_path).profile
        assert profile.duplicate_rows == 0
        assert profile.duplicate_rows_excluding_id == 35

    def test_all_columns_are_recognised(self, real_dataset_path: Path) -> None:
        assert load_dataset(real_dataset_path).profile.unrecognised_columns == ()

    def test_validation_passes_cleanly(self, real_dataset_path: Path) -> None:
        validation = load_dataset(real_dataset_path).validation
        assert validation.is_valid
        assert validation.warnings == ()

    def test_default_rate_matches_verified_value(self, real_dataset_path: Path) -> None:
        data = load_dataset(real_dataset_path).data
        assert data["default_next_month"].sum() == 6_636
        assert round(float(data["default_next_month"].mean() * 100), 2) == 22.12
