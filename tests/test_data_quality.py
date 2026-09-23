"""Tests for the data quality engine."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_quality import (
    DIMENSION_WEIGHTS,
    Severity,
    analyse_duplicates,
    analyse_missing_values,
    analyse_uniqueness,
    compute_quality_score,
    detect_consistency_issues,
    detect_invalid_values,
    detect_outliers,
    run_quality_report,
    validate_dtypes,
)


class TestMissingValues:
    """Completeness analysis."""

    def test_complete_frame_reports_zero(self, minimal_credit_df: pd.DataFrame) -> None:
        report = analyse_missing_values(minimal_credit_df)
        assert report.missing_cells == 0
        assert report.is_complete
        assert report.missing_percentage == 0.0

    def test_counts_missing_per_column(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame.loc[0, "age"] = None
        frame.loc[[1, 2], "credit_limit"] = None
        report = analyse_missing_values(frame)
        assert report.missing_cells == 3
        assert report.missing_by_column["age"] == 1
        assert report.missing_by_column["credit_limit"] == 2
        assert set(report.columns_with_any_missing) == {"age", "credit_limit"}

    def test_detects_fully_empty_column(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame["blank"] = None
        assert "blank" in analyse_missing_values(frame).columns_fully_missing

    def test_empty_frame_does_not_divide_by_zero(self) -> None:
        report = analyse_missing_values(pd.DataFrame())
        assert report.missing_percentage == 0.0


class TestDuplicates:
    """Uniqueness analysis."""

    def test_no_duplicates_in_clean_frame(self, minimal_credit_df: pd.DataFrame) -> None:
        report, issues = analyse_duplicates(minimal_credit_df)
        assert report.exact_duplicate_rows == 0
        assert report.duplicate_rows_excluding_id == 0
        assert issues == []

    def test_detects_duplicates_excluding_id(self, edge_case_credit_df: pd.DataFrame) -> None:
        report, issues = analyse_duplicates(edge_case_credit_df)
        assert report.exact_duplicate_rows == 0
        assert report.duplicate_rows_excluding_id == 1
        checks = {i.check for i in issues}
        assert "duplicate_rows_excluding_id" in checks

    def test_exact_duplicate_is_high_severity(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = pd.concat([minimal_credit_df, minimal_credit_df.head(1)], ignore_index=True)
        _report, issues = analyse_duplicates(frame)
        exact = [i for i in issues if i.check == "exact_duplicate_rows"]
        assert exact and exact[0].severity is Severity.HIGH

    def test_repeated_identifier_is_flagged(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame.loc[1, "client_id"] = 1
        report, issues = analyse_duplicates(frame)
        assert not report.id_is_unique
        assert any(i.check == "duplicate_identifier" for i in issues)

    def test_empty_frame_is_handled(self) -> None:
        report, issues = analyse_duplicates(pd.DataFrame())
        assert report.total_rows == 0
        assert report.duplicate_percentage == 0.0
        assert issues == []


class TestUniqueness:
    """Constant and empty column detection."""

    def test_constant_column_is_low_severity(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame["same"] = 5
        issues = analyse_uniqueness(frame)
        constant = [i for i in issues if i.check == "constant_column"]
        assert constant and constant[0].severity is Severity.LOW

    def test_empty_column_is_high_severity(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame["blank"] = np.nan
        issues = analyse_uniqueness(frame)
        empty = [i for i in issues if i.check == "empty_column"]
        assert empty and empty[0].severity is Severity.HIGH


class TestDtypeValidation:
    """Detecting columns that should be numeric but are not."""

    def test_numeric_frame_has_no_issues(self, minimal_credit_df: pd.DataFrame) -> None:
        assert validate_dtypes(minimal_credit_df) == []

    def test_text_in_numeric_column_is_flagged(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame["credit_limit"] = frame["credit_limit"].astype("object")
        frame.loc[0, "credit_limit"] = "20,000"
        issues = validate_dtypes(frame)
        assert any(i.check == "unexpected_dtype" and i.column == "credit_limit" for i in issues)


class TestInvalidValues:
    """Domain-rule violations."""

    def test_clean_frame_has_no_invalid_cells(self, minimal_credit_df: pd.DataFrame) -> None:
        report = detect_invalid_values(minimal_credit_df)
        assert report.invalid_cells == 0
        assert report.cells_checked > 0

    def test_detects_undocumented_education_code(self, edge_case_credit_df: pd.DataFrame) -> None:
        report = detect_invalid_values(edge_case_credit_df)
        assert "education_code" in report.undocumented_codes
        assert set(report.undocumented_codes["education_code"]) >= {5, 6}

    def test_detects_undocumented_marriage_code(self, edge_case_credit_df: pd.DataFrame) -> None:
        report = detect_invalid_values(edge_case_credit_df)
        assert 0 in report.undocumented_codes.get("marriage_code", {})

    def test_detects_implausible_age(self, edge_case_credit_df: pd.DataFrame) -> None:
        report = detect_invalid_values(edge_case_credit_df)
        assert any(
            i.check == "value_out_of_range" and i.column == "age" for i in report.issues
        )

    def test_detects_non_positive_credit_limit(self, edge_case_credit_df: pd.DataFrame) -> None:
        report = detect_invalid_values(edge_case_credit_df)
        limit_issues = [
            i for i in report.issues
            if i.check == "value_out_of_range" and i.column == "credit_limit"
        ]
        assert limit_issues
        # Rows 1 (zero) and 7 (negative) in the fixture.
        assert limit_issues[0].affected_rows == 2

    def test_undocumented_pay_status_is_low_severity_not_invalid(self) -> None:
        """-2 and 0 are undocumented but real, so they are reported without
        being counted as invalid cells."""
        frame = pd.DataFrame(
            {
                "client_id": [1, 2],
                "credit_limit": [10_000, 20_000],
                "age": [30, 40],
                "default_next_month": [0, 1],
                **{f"pay_status_m{m}": [-2, 0] for m in range(1, 7)},
                **{f"bill_amt_m{m}": [100, 200] for m in range(1, 7)},
                **{f"pay_amt_m{m}": [10, 20] for m in range(1, 7)},
            }
        )
        report = detect_invalid_values(frame)
        undocumented = [
            i for i in report.issues if i.check == "undocumented_category_code"
        ]
        assert any(i.severity is Severity.LOW for i in undocumented)
        assert report.invalid_cells == 0, "undocumented-but-real codes are not invalid"

    def test_affected_rows_never_exceeds_row_count(
        self, edge_case_credit_df: pd.DataFrame
    ) -> None:
        report = detect_invalid_values(edge_case_credit_df)
        for issue in report.issues:
            assert issue.affected_rows <= len(edge_case_credit_df)

    def test_invalid_target_value_is_flagged(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame.loc[0, "default_next_month"] = 5
        report = detect_invalid_values(frame)
        assert any(i.check == "invalid_target_value" for i in report.issues)


class TestConsistency:
    """Cross-field contradiction detection."""

    def test_negative_balance_is_info_only(self, edge_case_credit_df: pd.DataFrame) -> None:
        """A credit balance is legitimate, so it must not be penalised."""
        issues, _ = detect_consistency_issues(edge_case_credit_df)
        negative = [i for i in issues if i.check == "negative_statement_balance"]
        assert negative and negative[0].severity is Severity.INFO

    def test_over_limit_is_info_only(self, edge_case_credit_df: pd.DataFrame) -> None:
        issues, _ = detect_consistency_issues(edge_case_credit_df)
        over = [i for i in issues if i.check == "over_limit_balance"]
        assert over and over[0].severity is Severity.INFO

    def test_delinquent_without_balance_is_a_contradiction(self) -> None:
        frame = pd.DataFrame(
            {
                "client_id": [1],
                "credit_limit": [10_000],
                "age": [30],
                "default_next_month": [0],
                **{f"bill_amt_m{m}": [0] for m in range(1, 7)},
                **{f"pay_amt_m{m}": [0] for m in range(1, 7)},
                **{f"pay_status_m{m}": [3] for m in range(1, 7)},
            }
        )
        issues, n_contradictory = detect_consistency_issues(frame)
        assert any(i.check == "delinquent_without_balance" for i in issues)
        assert n_contradictory == 1

    def test_contradiction_count_is_distinct_rows(self, analysis_df: pd.DataFrame) -> None:
        """The count is rows, not client-months, so it cannot exceed the row total."""
        _issues, n_contradictory = detect_consistency_issues(analysis_df)
        assert 0 <= n_contradictory <= len(analysis_df)

    def test_empty_frame_is_handled(self) -> None:
        issues, count = detect_consistency_issues(pd.DataFrame())
        assert issues == []
        assert count == 0


class TestOutliers:
    """Outlier detection, which reports without removing."""

    def test_detects_an_extreme_value(self) -> None:
        frame = pd.DataFrame({"amount": [10, 11, 12, 13, 14, 15, 10_000]})
        report = detect_outliers(frame, columns=["amount"])
        assert report.by_column["amount"].iqr_outliers >= 1

    def test_skips_binary_columns(self, minimal_credit_df: pd.DataFrame) -> None:
        """A 0/1 column has no meaningful outliers."""
        report = detect_outliers(minimal_credit_df)
        assert "default_next_month" not in report.by_column

    def test_method_note_explains_skew(self, analysis_df: pd.DataFrame) -> None:
        report = detect_outliers(analysis_df)
        assert "skew" in report.method_note.lower()
        assert "remove" in report.method_note.lower()

    def test_most_affected_is_sorted(self, analysis_df: pd.DataFrame) -> None:
        affected = detect_outliers(analysis_df).most_affected(5)
        percentages = [s.iqr_percentage for s in affected]
        assert percentages == sorted(percentages, reverse=True)

    def test_constant_column_is_skipped(self) -> None:
        frame = pd.DataFrame({"same": [5] * 20})
        assert detect_outliers(frame, columns=["same"]).by_column == {}


class TestQualityScore:
    """The composite score."""

    def test_weights_sum_to_one_hundred(self) -> None:
        assert sum(DIMENSION_WEIGHTS.values()) == 100.0

    def test_perfect_data_scores_one_hundred(self, minimal_credit_df: pd.DataFrame) -> None:
        score = compute_quality_score(
            missing=analyse_missing_values(minimal_credit_df),
            duplicates=analyse_duplicates(minimal_credit_df)[0],
            invalid=detect_invalid_values(minimal_credit_df),
            n_contradictory_rows=0,
            n_rows=len(minimal_credit_df),
        )
        assert score.score == pytest.approx(100.0)
        assert score.band == "Excellent"

    def test_score_is_bounded(self, minimal_credit_df: pd.DataFrame) -> None:
        """Even with everything wrong, the score must stay within 0-100."""
        missing = analyse_missing_values(minimal_credit_df)
        score = compute_quality_score(
            missing=missing,
            duplicates=analyse_duplicates(minimal_credit_df)[0],
            invalid=detect_invalid_values(minimal_credit_df),
            n_contradictory_rows=len(minimal_credit_df) * 10,  # deliberately absurd
            n_rows=len(minimal_credit_df),
        )
        assert 0.0 <= score.score <= 100.0

    def test_missing_values_reduce_completeness(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        frame.loc[0:2, "age"] = None
        score = compute_quality_score(
            missing=analyse_missing_values(frame),
            duplicates=analyse_duplicates(frame)[0],
            invalid=detect_invalid_values(frame),
            n_contradictory_rows=0,
            n_rows=len(frame),
        )
        assert score.dimension_scores["completeness"] < 1.0

    def test_contributions_sum_to_score(self, minimal_credit_df: pd.DataFrame) -> None:
        score = compute_quality_score(
            missing=analyse_missing_values(minimal_credit_df),
            duplicates=analyse_duplicates(minimal_credit_df)[0],
            invalid=detect_invalid_values(minimal_credit_df),
            n_contradictory_rows=1,
            n_rows=len(minimal_credit_df),
        )
        assert sum(score.contributions.values()) == pytest.approx(score.score, abs=0.05)

    def test_methodology_is_documented_as_project_defined(self) -> None:
        score = compute_quality_score(
            missing=analyse_missing_values(pd.DataFrame({"a": [1]})),
            duplicates=analyse_duplicates(pd.DataFrame({"a": [1]}))[0],
            invalid=detect_invalid_values(pd.DataFrame({"a": [1]})),
            n_contradictory_rows=0,
            n_rows=1,
        )
        assert "NOT an industry-standard" in score.methodology

    def test_empty_frame_scores_without_error(self) -> None:
        empty = pd.DataFrame()
        score = compute_quality_score(
            missing=analyse_missing_values(empty),
            duplicates=analyse_duplicates(empty)[0],
            invalid=detect_invalid_values(empty),
            n_contradictory_rows=0,
            n_rows=0,
        )
        assert score.score == pytest.approx(100.0)


class TestFullReport:
    """The orchestrated report."""

    def test_summary_has_requested_shape(self, analysis_df: pd.DataFrame) -> None:
        summary = run_quality_report(analysis_df).summary()
        for key in (
            "rows", "columns", "missing_values", "duplicates",
            "numeric_columns", "categorical_columns", "quality_score",
        ):
            assert key in summary

    def test_report_is_json_serialisable(self, analysis_df: pd.DataFrame) -> None:
        json.dumps(run_quality_report(analysis_df).to_dict())

    def test_issues_sorted_by_severity(self, edge_case_credit_df: pd.DataFrame) -> None:
        from src.data_quality import SEVERITY_ORDER

        issues = run_quality_report(edge_case_credit_df).all_issues
        ranks = [SEVERITY_ORDER[i.severity] for i in issues]
        assert ranks == sorted(ranks)

    def test_headline_counts_cover_every_severity(self, analysis_df: pd.DataFrame) -> None:
        counts = run_quality_report(analysis_df).headline_counts
        assert set(counts) == {s.value for s in Severity}

    def test_all_issue_row_counts_are_within_bounds(
        self, edge_case_credit_df: pd.DataFrame
    ) -> None:
        report = run_quality_report(edge_case_credit_df)
        for issue in report.all_issues:
            assert 0 <= issue.affected_rows <= report.n_rows
            assert 0.0 <= issue.affected_percentage <= 100.0


@pytest.mark.integration
class TestRealDataQuality:
    """Quality findings on the real dataset."""

    def test_verified_summary_values(self, real_pipeline) -> None:
        summary = real_pipeline.quality.summary()
        assert summary["rows"] == 30_000
        assert summary["columns"] == 25
        assert summary["missing_values"] == 0
        assert summary["duplicates"] == 35

    def test_completeness_is_perfect(self, real_pipeline) -> None:
        assert real_pipeline.quality.quality_score.dimension_scores["completeness"] == 1.0

    def test_consistency_is_the_weakest_dimension(self, real_pipeline) -> None:
        """A null-counter would score this data 100; the real defects are semantic."""
        dimensions = real_pipeline.quality.quality_score.dimension_scores
        assert dimensions["consistency"] == min(dimensions.values())
        assert dimensions["consistency"] < 1.0

    def test_finds_the_undocumented_codes(self, real_pipeline) -> None:
        undocumented = real_pipeline.quality.invalid.undocumented_codes
        assert set(undocumented.get("education_code", {})) == {0, 5, 6}
        assert set(undocumented.get("marriage_code", {})) == {0}

    def test_no_high_severity_issues(self, real_pipeline) -> None:
        assert real_pipeline.quality.issues_by_severity(Severity.HIGH) == ()
