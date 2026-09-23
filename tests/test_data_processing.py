"""Tests for the cleaning pipeline, panel reshape and persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    build_panel_long,
    clean_dataset,
    convert_dtypes,
    flag_extreme_outliers,
    handle_invalid_numeric_values,
    handle_missing_values,
    load_processed,
    normalise_categoricals,
    remove_duplicates,
    save_processed,
)
from src.schema import UNKNOWN_LABEL


class TestConvertDtypes:
    """Numeric coercion."""

    def test_no_action_when_already_numeric(self, minimal_credit_df: pd.DataFrame) -> None:
        _result, step = convert_dtypes(minimal_credit_df)
        assert not step.action_taken

    def test_converts_text_numbers(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame["credit_limit"] = frame["credit_limit"].astype("string")
        result, step = convert_dtypes(frame)
        assert pd.api.types.is_numeric_dtype(result["credit_limit"])
        assert step.action_taken

    def test_unparseable_becomes_missing_not_an_error(
        self, minimal_credit_df: pd.DataFrame
    ) -> None:
        frame = minimal_credit_df.copy()
        frame["age"] = frame["age"].astype("object")
        frame.loc[0, "age"] = "unknown"
        result, step = convert_dtypes(frame)
        assert pd.isna(result.loc[0, "age"])
        assert step.details["values_made_missing"] == 1
        assert len(result) == len(frame), "no row may be dropped for one bad cell"


class TestRemoveDuplicates:
    """Duplicate handling."""

    def test_removes_exact_duplicates(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = pd.concat([minimal_credit_df, minimal_credit_df.head(2)], ignore_index=True)
        result, step = remove_duplicates(frame)
        assert len(result) == len(minimal_credit_df)
        assert step.details["exact_duplicates_removed"] == 2

    def test_keeps_and_flags_id_excluded_duplicates_by_default(
        self, edge_case_credit_df: pd.DataFrame
    ) -> None:
        result, step = remove_duplicates(edge_case_credit_df)
        assert len(result) == len(edge_case_credit_df), "rows must be retained by default"
        assert "is_duplicate_excluding_id" in result.columns
        assert step.details["id_excluded_duplicates_flagged"] == 1

    def test_flags_all_members_of_a_duplicate_group(
        self, edge_case_credit_df: pd.DataFrame
    ) -> None:
        """Both rows of the pair are flagged so the analyst sees the whole group."""
        result, _ = remove_duplicates(edge_case_credit_df)
        assert int(result["is_duplicate_excluding_id"].sum()) == 2

    def test_can_drop_id_excluded_duplicates_when_requested(
        self, edge_case_credit_df: pd.DataFrame
    ) -> None:
        result, step = remove_duplicates(edge_case_credit_df, drop_excluding_id=True)
        assert len(result) == len(edge_case_credit_df) - 1
        assert step.details["id_excluded_duplicates_removed"] == 1

    def test_no_duplicates_means_no_action(self, minimal_credit_df: pd.DataFrame) -> None:
        _result, step = remove_duplicates(minimal_credit_df)
        assert not step.action_taken


class TestHandleMissingValues:
    """Imputation."""

    def test_no_action_on_complete_data(self, minimal_credit_df: pd.DataFrame) -> None:
        _result, step = handle_missing_values(minimal_credit_df)
        assert not step.action_taken
        assert "No missing values" in step.description

    def test_median_imputation_with_indicator(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame.loc[0, "age"] = None
        expected_median = frame["age"].median()
        result, step = handle_missing_values(frame, numeric_strategy="median")
        assert result.loc[0, "age"] == expected_median
        assert "age_was_missing" in result.columns
        assert bool(result.loc[0, "age_was_missing"]) is True
        assert step.affected_rows == 1

    def test_no_rows_are_dropped(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame.loc[0, "age"] = None
        result, _ = handle_missing_values(frame)
        assert len(result) == len(frame)

    def test_leave_strategy_preserves_nulls(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame.loc[0, "age"] = None
        result, _ = handle_missing_values(frame, numeric_strategy="leave")
        assert pd.isna(result.loc[0, "age"])

    def test_fully_empty_column_is_left_alone(self, minimal_credit_df: pd.DataFrame) -> None:
        """There is nothing to impute from, so it must not invent a value."""
        frame = minimal_credit_df.copy()
        frame["blank"] = np.nan
        result, step = handle_missing_values(frame)
        assert result["blank"].isna().all()
        assert "entirely empty" in step.details["columns_with_missing"]["blank"]["action"]

    def test_categorical_filled_with_unknown(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame["label"] = ["a", "b", None, "a", "b", "a"]
        result, _ = handle_missing_values(frame, categorical_strategy="unknown")
        assert result.loc[2, "label"] == UNKNOWN_LABEL


class TestNormaliseCategoricals:
    """Code-to-label mapping."""

    def test_creates_label_columns(self, minimal_credit_df: pd.DataFrame) -> None:
        result, step = normalise_categoricals(minimal_credit_df)
        for column in ("sex", "education", "marriage"):
            assert column in result.columns
        assert set(step.columns_added) == {"sex", "education", "marriage"}

    def test_original_codes_are_preserved(self, minimal_credit_df: pd.DataFrame) -> None:
        result, _ = normalise_categoricals(minimal_credit_df)
        for column in ("sex_code", "education_code", "marriage_code"):
            assert column in result.columns, "raw codes must survive for auditability"

    def test_maps_documented_codes_correctly(self, minimal_credit_df: pd.DataFrame) -> None:
        result, _ = normalise_categoricals(minimal_credit_df)
        assert result.loc[0, "sex"] == "Male"      # code 1
        assert result.loc[1, "sex"] == "Female"    # code 2
        assert result.loc[0, "education"] == "Graduate school"  # code 1
        assert result.loc[2, "education"] == "High school"      # code 3
        assert result.loc[0, "marriage"] == "Married"           # code 1

    def test_undocumented_codes_become_unknown(self, edge_case_credit_df: pd.DataFrame) -> None:
        result, step = normalise_categoricals(edge_case_credit_df)
        # Education codes 5 and 6 are undocumented.
        assert result.loc[2, "education"] == UNKNOWN_LABEL
        assert result.loc[7, "education"] == UNKNOWN_LABEL
        # Marriage code 0 is undocumented.
        assert result.loc[2, "marriage"] == UNKNOWN_LABEL
        detail = step.details["education_code"]["undocumented_codes_mapped_to_unknown"]
        assert set(detail) == {"5", "6"}

    def test_no_rows_lost_to_unknown_codes(self, edge_case_credit_df: pd.DataFrame) -> None:
        result, _ = normalise_categoricals(edge_case_credit_df)
        assert len(result) == len(edge_case_credit_df)

    def test_labels_are_ordered_categoricals(self, minimal_credit_df: pd.DataFrame) -> None:
        result, _ = normalise_categoricals(minimal_credit_df)
        assert isinstance(result["education"].dtype, pd.CategoricalDtype)


class TestHandleInvalidNumericValues:
    """Range handling."""

    def test_non_positive_limit_becomes_nan_with_original_kept(
        self, edge_case_credit_df: pd.DataFrame
    ) -> None:
        result, _ = handle_invalid_numeric_values(edge_case_credit_df)
        assert pd.isna(result.loc[0, "credit_limit"]), "zero limit must not stay a denominator"
        assert pd.isna(result.loc[6, "credit_limit"]), "negative limit must be neutralised"
        assert result.loc[0, "credit_limit_original"] == 0

    def test_implausible_age_is_flagged_not_removed(
        self, edge_case_credit_df: pd.DataFrame
    ) -> None:
        result, _ = handle_invalid_numeric_values(edge_case_credit_df)
        assert "age_is_implausible" in result.columns
        assert bool(result.loc[7, "age_is_implausible"]) is True
        assert result.loc[7, "age"] == 150, "the value itself must be preserved"
        assert len(result) == len(edge_case_credit_df)

    def test_negative_balance_is_flagged_and_preserved(
        self, edge_case_credit_df: pd.DataFrame
    ) -> None:
        result, _ = handle_invalid_numeric_values(edge_case_credit_df)
        assert bool(result.loc[1, "has_negative_balance"]) is True
        assert result.loc[1, "bill_amt_m1"] == -2_000

    def test_clean_frame_takes_no_action(self, minimal_credit_df: pd.DataFrame) -> None:
        _result, step = handle_invalid_numeric_values(minimal_credit_df)
        assert not step.action_taken


class TestFlagExtremeOutliers:
    """Extreme-value flagging."""

    def test_flags_without_altering_values(self, analysis_df: pd.DataFrame) -> None:
        frame = analysis_df.copy()
        frame.loc[0, "bill_amt_m1"] = 99_000_000
        result, step = flag_extreme_outliers(frame)
        assert result.loc[0, "bill_amt_m1"] == 99_000_000, "values must never be capped"
        if step.action_taken:
            assert bool(result.loc[0, "has_extreme_value"]) is True

    def test_uses_a_three_times_iqr_fence(self, analysis_df: pd.DataFrame) -> None:
        _result, step = flag_extreme_outliers(analysis_df)
        assert step.details["iqr_multiplier"] == 3.0

    def test_row_count_is_unchanged(self, analysis_df: pd.DataFrame) -> None:
        result, _ = flag_extreme_outliers(analysis_df)
        assert len(result) == len(analysis_df)


class TestCleanDataset:
    """The orchestrated cleaning run."""

    def test_preserves_all_rows_by_default(self, edge_case_credit_df: pd.DataFrame) -> None:
        result, report = clean_dataset(edge_case_credit_df)
        assert len(result) == len(edge_case_credit_df)
        assert report.rows_removed == 0

    def test_adds_columns_without_dropping_any(self, minimal_credit_df: pd.DataFrame) -> None:
        result, _ = clean_dataset(minimal_credit_df)
        for column in minimal_credit_df.columns:
            assert column in result.columns
        assert len(result.columns) > len(minimal_credit_df.columns)

    def test_every_step_records_a_rationale(self, minimal_credit_df: pd.DataFrame) -> None:
        _result, report = clean_dataset(minimal_credit_df)
        for step in report.steps:
            assert step.rationale.strip(), f"step '{step.name}' has no rationale"
            assert len(step.rationale) > 40, "the rationale must actually explain the decision"

    def test_report_is_serialisable(self, minimal_credit_df: pd.DataFrame) -> None:
        import json

        _result, report = clean_dataset(minimal_credit_df)
        json.dumps(report.to_dict())

    def test_is_deterministic(self, analysis_df: pd.DataFrame) -> None:
        first, _ = clean_dataset(analysis_df)
        second, _ = clean_dataset(analysis_df)
        pd.testing.assert_frame_equal(first, second)

    def test_does_not_mutate_the_input(self, minimal_credit_df: pd.DataFrame) -> None:
        before = minimal_credit_df.copy()
        clean_dataset(minimal_credit_df)
        pd.testing.assert_frame_equal(minimal_credit_df, before)

    def test_summary_mentions_row_and_column_movement(
        self, minimal_credit_df: pd.DataFrame
    ) -> None:
        _result, report = clean_dataset(minimal_credit_df)
        summary = report.summary()
        assert "Rows" in summary and "Columns" in summary


class TestBuildPanelLong:
    """Wide-to-long panel reshape."""

    def test_produces_one_row_per_client_month(self, minimal_credit_df: pd.DataFrame) -> None:
        panel = build_panel_long(minimal_credit_df)
        assert len(panel) == len(minimal_credit_df) * 6

    def test_has_expected_columns(self, minimal_credit_df: pd.DataFrame) -> None:
        panel = build_panel_long(minimal_credit_df)
        for column in (
            "client_id", "month_index", "month_label", "month_order",
            "bill_amt", "pay_amt", "pay_status", "is_delinquent", "utilisation",
        ):
            assert column in panel.columns

    def test_month_order_follows_calendar_time(self, minimal_credit_df: pd.DataFrame) -> None:
        """Month 6 is the oldest (April), so it must sort first."""
        panel = build_panel_long(minimal_credit_df)
        first = panel.loc[panel["client_id"] == 1].sort_values("month_order").iloc[0]
        assert first["month_index"] == 6
        assert first["month_label"] == "Apr 2005"

    def test_values_match_the_wide_source(self, minimal_credit_df: pd.DataFrame) -> None:
        panel = build_panel_long(minimal_credit_df)
        row = panel[(panel["client_id"] == 1) & (panel["month_index"] == 1)].iloc[0]
        assert row["bill_amt"] == minimal_credit_df.loc[0, "bill_amt_m1"]
        assert row["pay_amt"] == minimal_credit_df.loc[0, "pay_amt_m1"]
        assert row["pay_status"] == minimal_credit_df.loc[0, "pay_status_m1"]

    def test_delinquency_flag_uses_documented_threshold(
        self, minimal_credit_df: pd.DataFrame
    ) -> None:
        panel = build_panel_long(minimal_credit_df)
        assert (panel.loc[panel["pay_status"] >= 1, "is_delinquent"]).all()
        assert not (panel.loc[panel["pay_status"] <= 0, "is_delinquent"]).any()

    def test_empty_panel_when_no_monthly_columns(self) -> None:
        frame = pd.DataFrame({"client_id": [1, 2], "age": [30, 40]})
        panel = build_panel_long(frame)
        assert panel.empty
        assert "month_index" in panel.columns, "columns must exist even when empty"

    def test_utilisation_is_nan_for_zero_limit(self, edge_case_credit_df: pd.DataFrame) -> None:
        cleaned, _ = clean_dataset(edge_case_credit_df)
        panel = build_panel_long(cleaned)
        client_one = panel.loc[panel["client_id"] == 1, "utilisation"]
        assert client_one.isna().all(), "must be NaN, never inf"
        assert not np.isinf(panel["utilisation"].dropna()).any()


class TestPersistence:
    """Reading and writing processed artefacts."""

    def test_round_trips_through_parquet(
        self, tmp_path: Path, small_featured_df: pd.DataFrame
    ) -> None:
        path = save_processed(small_featured_df, "test.parquet", processed_dir=tmp_path)
        assert path.exists()
        reloaded = load_processed("test.parquet", processed_dir=tmp_path)
        assert len(reloaded) == len(small_featured_df)
        assert list(reloaded.columns) == list(small_featured_df.columns)

    def test_preserves_categorical_dtype(
        self, tmp_path: Path, small_featured_df: pd.DataFrame
    ) -> None:
        """Parquet is chosen over CSV precisely so dtypes survive."""
        save_processed(small_featured_df, "test.parquet", processed_dir=tmp_path)
        reloaded = load_processed("test.parquet", processed_dir=tmp_path)
        assert isinstance(reloaded["education"].dtype, pd.CategoricalDtype)

    def test_creates_the_target_directory(
        self, tmp_path: Path, small_featured_df: pd.DataFrame
    ) -> None:
        nested = tmp_path / "a" / "b"
        assert save_processed(small_featured_df, "t.parquet", processed_dir=nested).exists()

    def test_missing_file_raises_with_guidance(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="pipeline"):
            load_processed("absent.parquet", processed_dir=tmp_path)


@pytest.mark.integration
class TestRealCleaning:
    """Cleaning behaviour on the real dataset."""

    def test_no_rows_removed(self, real_pipeline) -> None:
        assert real_pipeline.cleaning.rows_removed == 0
        assert len(real_pipeline.clean) == 30_000

    def test_raw_file_is_untouched(self, real_pipeline, real_dataset_path) -> None:
        """The pipeline must never write back to the source file."""
        import hashlib

        digest = hashlib.sha256(real_dataset_path.read_bytes()).hexdigest()
        assert digest == (
            "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
        )

    def test_unknown_label_count_matches_quality_report(self, real_pipeline) -> None:
        """345 education + 54 marriage = 399, as measured."""
        clean = real_pipeline.clean
        assert int((clean["education"] == UNKNOWN_LABEL).sum()) == 345
        assert int((clean["marriage"] == UNKNOWN_LABEL).sum()) == 54

    def test_panel_has_expected_size(self, real_pipeline) -> None:
        assert len(real_pipeline.panel) == 180_000
        assert real_pipeline.panel["client_id"].nunique() == 30_000
