"""Tests for the EDA engine, including insufficient-data guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    InsufficientData,
    _derivation_family,
    _same_family,
    category_outcome_rates,
    compare_groups,
    correlation_matrix,
    cross_tabulate,
    distribution_bins,
    frequency_table,
    numeric_outcome_comparison,
    outlier_analysis,
    run_eda,
    segment_profile,
    summarise_numeric,
    summary_frame,
)


class TestSummariseNumeric:
    """Descriptive statistics."""

    def test_computes_expected_statistics(self) -> None:
        frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        summary = summarise_numeric(frame, columns=["value"])[0]
        assert summary.mean == pytest.approx(3.0)
        assert summary.median == pytest.approx(3.0)
        assert summary.minimum == 1.0
        assert summary.maximum == 5.0
        assert summary.count == 5

    def test_iqr_is_derived_correctly(self) -> None:
        frame = pd.DataFrame({"value": list(range(1, 101))})
        summary = summarise_numeric(frame, columns=["value"])[0]
        assert summary.iqr == pytest.approx(summary.q3 - summary.q1)

    def test_detects_right_skew(self) -> None:
        frame = pd.DataFrame({"value": [1] * 95 + [1_000] * 5})
        summary = summarise_numeric(frame, columns=["value"])[0]
        assert summary.skewness > 1
        assert "right-skewed" in summary.skew_description

    def test_symmetric_distribution_is_described_as_such(self) -> None:
        frame = pd.DataFrame({"value": list(range(-50, 51))})
        assert "symmetric" in summarise_numeric(frame, columns=["value"])[0].skew_description

    def test_counts_missing_values(self) -> None:
        frame = pd.DataFrame({"value": [1.0, 2.0, None, 4.0]})
        summary = summarise_numeric(frame, columns=["value"])[0]
        assert summary.missing == 1
        assert summary.count == 3

    def test_excludes_identifier_by_default(self, featured_df: pd.DataFrame) -> None:
        columns = {s.column for s in summarise_numeric(featured_df)}
        assert "client_id" not in columns

    def test_coefficient_of_variation_undefined_at_zero_mean(self) -> None:
        frame = pd.DataFrame({"value": [-1.0, 1.0]})
        assert summarise_numeric(frame, columns=["value"])[0].coefficient_of_variation is None

    def test_infinities_are_excluded(self) -> None:
        frame = pd.DataFrame({"value": [1.0, 2.0, np.inf, -np.inf, 3.0]})
        summary = summarise_numeric(frame, columns=["value"])[0]
        assert summary.count == 3
        assert np.isfinite(summary.mean)

    def test_empty_frame_returns_nothing(self) -> None:
        assert summarise_numeric(pd.DataFrame({"a": []}), columns=["a"]) == []

    def test_summary_frame_is_tabular(self, featured_df: pd.DataFrame) -> None:
        frame = summary_frame(summarise_numeric(featured_df))
        assert "column" in frame.columns and "skew_description" in frame.columns

    def test_summary_frame_handles_empty_input(self) -> None:
        assert summary_frame([]).empty


class TestFrequencyTable:
    """Categorical frequencies."""

    def test_counts_and_percentages(self) -> None:
        frame = pd.DataFrame({"cat": ["a", "a", "b", "c"]})
        table = frequency_table(frame, "cat")
        assert table.counts["a"] == 2
        assert table.percentages["a"] == pytest.approx(50.0)
        assert table.n_categories == 3

    def test_identifies_top_category(self) -> None:
        frame = pd.DataFrame({"cat": ["x"] * 7 + ["y"] * 3})
        table = frequency_table(frame, "cat")
        assert table.top_category == "x"
        assert table.top_share == pytest.approx(70.0)

    def test_top_n_limits_output(self) -> None:
        frame = pd.DataFrame({"cat": list("abcdefghij")})
        assert len(frequency_table(frame, "cat", top_n=3).counts) == 3

    def test_missing_column_is_reported(self, featured_df: pd.DataFrame) -> None:
        result = frequency_table(featured_df, "no_such_column")
        assert isinstance(result, InsufficientData)
        assert "not present" in result.message

    def test_empty_frame_is_reported(self, featured_df: pd.DataFrame) -> None:
        assert isinstance(frequency_table(featured_df.iloc[0:0], "education"), InsufficientData)

    def test_to_frame_is_tabular(self, featured_df: pd.DataFrame) -> None:
        frame = frequency_table(featured_df, "education").to_frame()
        assert list(frame.columns) == ["category", "count", "percentage"]


class TestDistributionBins:
    """Histogram binning."""

    def test_returns_requested_bin_count(self, featured_df: pd.DataFrame) -> None:
        bins = distribution_bins(featured_df, "credit_limit", bins=10)
        assert len(bins) == 10

    def test_counts_sum_to_row_count(self, featured_df: pd.DataFrame) -> None:
        bins = distribution_bins(featured_df, "credit_limit", bins=15)
        assert int(bins["count"].sum()) == len(featured_df)

    def test_constant_column_is_reported(self) -> None:
        frame = pd.DataFrame({"value": [5.0] * 20})
        result = distribution_bins(frame, "value")
        assert isinstance(result, InsufficientData)
        assert "identical" in result.message

    def test_single_value_is_reported(self) -> None:
        assert isinstance(distribution_bins(pd.DataFrame({"v": [1.0]}), "v"), InsufficientData)

    def test_missing_column_is_reported(self, featured_df: pd.DataFrame) -> None:
        assert isinstance(distribution_bins(featured_df, "nope"), InsufficientData)


class TestCorrelation:
    """Correlation analysis."""

    def test_defaults_to_spearman(self, featured_df: pd.DataFrame) -> None:
        """Pearson is distorted by the heavy skew in monetary columns."""
        assert correlation_matrix(featured_df).method == "spearman"

    def test_matrix_is_square_and_symmetric(self, featured_df: pd.DataFrame) -> None:
        matrix = correlation_matrix(featured_df).matrix
        assert matrix.shape[0] == matrix.shape[1]
        assert matrix.columns.tolist() == matrix.index.tolist()

    def test_perfect_correlation_is_detected(self) -> None:
        frame = pd.DataFrame({"a": list(range(50)), "b": [x * 2 for x in range(50)]})
        result = correlation_matrix(frame, columns=["a", "b"])
        assert result.matrix.loc["a", "b"] == pytest.approx(1.0)

    def test_too_few_rows_is_reported(self, featured_df: pd.DataFrame) -> None:
        result = correlation_matrix(featured_df.head(5))
        assert isinstance(result, InsufficientData)
        assert "unstable" in result.message

    def test_single_column_is_reported(self) -> None:
        frame = pd.DataFrame({"only": list(range(50))})
        result = correlation_matrix(frame, columns=["only"])
        assert isinstance(result, InsufficientData)

    def test_note_states_association_not_causation(self, featured_df: pd.DataFrame) -> None:
        assert "causation" in correlation_matrix(featured_df).note.lower()

    def test_pair_descriptions_avoid_causal_language(self, featured_df: pd.DataFrame) -> None:
        result = correlation_matrix(featured_df)
        for pair in result.top_pairs(10):
            text = pair.describe().lower()
            assert "association" in text
            for banned in ("causes", "because", "drives"):
                assert banned not in text

    def test_strength_labels(self) -> None:
        from src.analytics import CorrelationPair

        assert CorrelationPair("a", "b", 0.9).strength == "strong"
        assert CorrelationPair("a", "b", 0.4).strength == "moderate"
        assert CorrelationPair("a", "b", 0.15).strength == "weak"
        assert CorrelationPair("a", "b", 0.02).strength == "negligible"
        assert CorrelationPair("a", "b", -0.9).direction == "negative"


class TestDerivationFamilies:
    """Filtering out structurally redundant correlation pairs."""

    def test_same_month_family_is_redundant(self) -> None:
        assert _same_family("bill_amt_m1", "bill_amt_m2")

    def test_alias_is_redundant(self) -> None:
        """utilisation_latest IS utilisation_m1."""
        assert _same_family("utilisation_m1", "utilisation_latest")

    def test_column_versus_its_own_aggregate_is_redundant(self) -> None:
        assert _same_family("bill_amt_m3", "avg_bill_6m")
        assert _same_family("total_paid_6m", "avg_payment_6m")

    def test_derived_from_relationship_is_redundant(self) -> None:
        """Utilisation is balance / limit, so both pairings are circular."""
        assert _same_family("utilisation_m1", "bill_amt_m1")
        assert _same_family("utilisation_latest", "credit_limit")
        assert _same_family("repayment_ratio_mean", "pay_amt_m1")

    def test_genuinely_different_measures_are_kept(self) -> None:
        assert not _same_family("utilisation_latest", "max_delinquency")
        assert not _same_family("credit_limit", "bill_amt_m1")
        assert not _same_family("age", "utilisation_latest")
        assert not _same_family("repayment_ratio_mean", "pay_status_m1")

    def test_unknown_columns_are_not_filtered(self) -> None:
        assert _derivation_family("age") is None
        assert not _same_family("age", "some_other_column")

    def test_filter_removes_tautological_top_pairs(self, featured_df: pd.DataFrame) -> None:
        result = correlation_matrix(featured_df)
        for pair in result.top_pairs(15, exclude_same_family=True):
            assert not _same_family(pair.left, pair.right)


class TestGroupComparison:
    """Comparing a measure across categories."""

    def test_computes_per_group_statistics(self, featured_df: pd.DataFrame) -> None:
        result = compare_groups(featured_df, "limit_band", "utilisation_latest")
        assert not isinstance(result, InsufficientData)
        assert "mean" in result.table.columns
        assert result.highest_group is not None

    def test_drops_undersized_groups(self) -> None:
        frame = pd.DataFrame(
            {"grp": ["a"] * 20 + ["b"] * 2, "value": list(range(22))}
        )
        result = compare_groups(frame, "grp", "value", min_group_size=5)
        assert "b" not in result.table.index

    def test_reports_when_no_group_qualifies(self) -> None:
        frame = pd.DataFrame({"grp": ["a", "b"], "value": [1.0, 2.0]})
        result = compare_groups(frame, "grp", "value", min_group_size=10)
        assert isinstance(result, InsufficientData)

    def test_missing_column_is_reported(self, featured_df: pd.DataFrame) -> None:
        result = compare_groups(featured_df, "nope", "utilisation_latest")
        assert isinstance(result, InsufficientData)

    def test_spread_ratio_handles_zero_minimum(self) -> None:
        frame = pd.DataFrame({"grp": ["a"] * 10 + ["b"] * 10, "value": [0.0] * 10 + [5.0] * 10})
        result = compare_groups(frame, "grp", "value", min_group_size=5)
        assert result.spread_ratio is None


class TestCategoryOutcomeRates:
    """Outcome rates per category."""

    def test_computes_rates_and_lift(self, featured_df: pd.DataFrame) -> None:
        result = category_outcome_rates(featured_df, "limit_band", "default_next_month")
        assert not isinstance(result, InsufficientData)
        assert "rate_pct" in result.table.columns
        assert "lift_vs_overall_pp" in result.table.columns

    def test_rate_matches_manual_calculation(self, featured_df: pd.DataFrame) -> None:
        result = category_outcome_rates(featured_df, "age_band", "default_next_month")
        for band in result.table.index:
            subset = featured_df[featured_df["age_band"] == band]
            if len(subset):
                expected = float(subset["default_next_month"].mean() * 100)
                assert result.table.loc[band, "rate_pct"] == pytest.approx(expected)

    def test_undersized_groups_excluded_from_ranking(self) -> None:
        """A tiny group must not be announced as the riskiest segment."""
        frame = pd.DataFrame(
            {
                "grp": ["big"] * 100 + ["tiny"] * 3,
                "target": [0] * 100 + [1] * 3,  # tiny group is 100% default
            }
        )
        result = category_outcome_rates(frame, "grp", "target", min_group_size=30)
        assert result.highest_category == "big", "the 3-row group must not be ranked"
        assert bool(result.table.loc["tiny", "meets_min_size"]) is False

    def test_overall_rate_is_computed(self, featured_df: pd.DataFrame) -> None:
        result = category_outcome_rates(featured_df, "education", "default_next_month")
        expected = float(featured_df["default_next_month"].mean() * 100)
        assert result.overall_rate == pytest.approx(expected)

    def test_missing_column_is_reported(self, featured_df: pd.DataFrame) -> None:
        result = category_outcome_rates(featured_df, "nope", "default_next_month")
        assert isinstance(result, InsufficientData)


class TestNumericOutcomeComparison:
    """Comparing a numeric measure between outcome classes."""

    def test_reports_means_and_effect_size(self, featured_df: pd.DataFrame) -> None:
        result = numeric_outcome_comparison(
            featured_df, "utilisation_latest", "default_next_month"
        )
        assert not isinstance(result, InsufficientData)
        assert "cohens_d" in result
        assert result["effect_size_label"] in {"negligible", "small", "medium", "large"}

    def test_detects_the_planted_signal(self, featured_df: pd.DataFrame) -> None:
        """The fixture gives defaulters higher utilisation by construction."""
        result = numeric_outcome_comparison(
            featured_df, "utilisation_latest", "default_next_month"
        )
        assert result["mean_positive"] > result["mean_negative"]
        assert result["cohens_d"] > 0

    def test_single_class_is_reported(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.copy()
        frame["default_next_month"] = 0
        result = numeric_outcome_comparison(frame, "credit_limit", "default_next_month")
        assert isinstance(result, InsufficientData)
        assert "both outcome classes" in result.message.lower()

    def test_too_few_rows_is_reported(self, featured_df: pd.DataFrame) -> None:
        result = numeric_outcome_comparison(
            featured_df.head(5), "credit_limit", "default_next_month"
        )
        assert isinstance(result, InsufficientData)

    def test_note_disclaims_causation(self, featured_df: pd.DataFrame) -> None:
        result = numeric_outcome_comparison(
            featured_df, "credit_limit", "default_next_month"
        )
        assert "causation" in result["interpretation_note"].lower()


class TestMultivariate:
    """Cross-tabulation and segment profiling."""

    def test_cross_tabulate_counts(self, featured_df: pd.DataFrame) -> None:
        table = cross_tabulate(featured_df, "limit_band", "age_band")
        assert not isinstance(table, InsufficientData)
        assert int(table.to_numpy().sum()) == len(featured_df)

    def test_cross_tabulate_with_aggregation(self, featured_df: pd.DataFrame) -> None:
        table = cross_tabulate(
            featured_df, "limit_band", "age_band", "default_next_month", "mean"
        )
        assert not isinstance(table, InsufficientData)
        values = table.to_numpy()
        assert np.nanmin(values) >= 0.0 and np.nanmax(values) <= 1.0

    def test_cross_tabulate_missing_column_reported(self, featured_df: pd.DataFrame) -> None:
        assert isinstance(cross_tabulate(featured_df, "nope", "age_band"), InsufficientData)

    def test_segment_profile_includes_size_and_share(self, featured_df: pd.DataFrame) -> None:
        profile = segment_profile(
            featured_df, "limit_band", ["utilisation_latest"], "default_next_month"
        )
        assert not isinstance(profile, InsufficientData)
        assert "n_clients" in profile.columns
        assert "share_of_portfolio_pct" in profile.columns
        assert profile["n_clients"].sum() == len(featured_df)

    def test_segment_profile_adds_outcome_rate(self, featured_df: pd.DataFrame) -> None:
        profile = segment_profile(
            featured_df, "limit_band", ["utilisation_latest"], "default_next_month"
        )
        assert "default_next_month_rate_pct" in profile.columns

    def test_segment_profile_missing_metrics_reported(self, featured_df: pd.DataFrame) -> None:
        result = segment_profile(featured_df, "limit_band", ["nope"])
        assert isinstance(result, InsufficientData)


class TestOutlierAnalysis:
    """Outlier quantification for interpretation."""

    def test_reports_bounds_and_counts(self, featured_df: pd.DataFrame) -> None:
        frame = outlier_analysis(featured_df, columns=["credit_limit", "bill_amt_m1"])
        assert not isinstance(frame, InsufficientData)
        for column in ("lower_bound", "upper_bound", "n_outliers", "outlier_pct"):
            assert column in frame.columns

    def test_detects_a_planted_extreme(self) -> None:
        frame = pd.DataFrame({"amount": [10.0] * 50 + [100_000.0]})
        result = outlier_analysis(frame, columns=["amount"])
        assert int(result.loc[result["column"] == "amount", "n_outliers"].iloc[0]) >= 1

    def test_empty_frame_is_reported(self) -> None:
        assert isinstance(outlier_analysis(pd.DataFrame()), InsufficientData)

    def test_constant_column_is_skipped(self) -> None:
        frame = pd.DataFrame({"same": [1.0] * 30})
        assert isinstance(outlier_analysis(frame, columns=["same"]), InsufficientData)


class TestRunEda:
    """The bundled EDA run."""

    def test_returns_all_sections(self, featured_df: pd.DataFrame) -> None:
        eda = run_eda(featured_df)
        for key in (
            "numeric_summaries", "numeric_summary_frame", "correlations",
            "outliers", "frequencies", "category_outcomes", "n_rows",
        ):
            assert key in eda

    def test_handles_a_tiny_frame_without_raising(self, featured_df: pd.DataFrame) -> None:
        eda = run_eda(featured_df.head(3))
        assert isinstance(eda["correlations"], InsufficientData)


class TestInsufficientData:
    """The insufficient-data sentinel."""

    def test_message_states_the_requirement(self) -> None:
        result = InsufficientData(analysis="Test analysis", n_rows=5, required_rows=30)
        assert "at least 30" in result.message
        assert "has 5" in result.message

    def test_dict_flags_the_condition(self) -> None:
        result = InsufficientData(analysis="T", n_rows=1, required_rows=10)
        assert result.to_dict()["insufficient_data"] is True


@pytest.mark.integration
class TestRealAnalytics:
    """Analytics on the real dataset."""

    def test_balance_is_strongly_right_skewed(self, real_pipeline) -> None:
        summary = summarise_numeric(real_pipeline.data, columns=["bill_amt_m1"])[0]
        assert summary.mean == pytest.approx(51_223.33, abs=0.01)
        assert summary.median == pytest.approx(22_381.5)
        assert summary.skewness > 2
        assert "strongly right-skewed" == summary.skew_description

    def test_default_rate_falls_as_credit_limit_rises(self, real_pipeline) -> None:
        """A real, measured monotonic gradient across limit bands."""
        result = category_outcome_rates(
            real_pipeline.data, "limit_band", "default_next_month"
        )
        rates = result.table.sort_index()["rate_pct"].tolist()
        assert rates == sorted(rates, reverse=True), f"expected monotonic decline, got {rates}"

    def test_top_associations_are_not_tautological(self, real_pipeline) -> None:
        result = correlation_matrix(real_pipeline.data, registry=real_pipeline.registry)
        for pair in result.top_pairs(10):
            assert abs(pair.correlation) < 0.999, "a near-perfect pair is a restatement"

    def test_delinquency_has_a_large_effect_size(self, real_pipeline) -> None:
        result = numeric_outcome_comparison(
            real_pipeline.data, "max_delinquency", "default_next_month"
        )
        assert result["effect_size_label"] == "large"
        assert result["mean_positive"] > result["mean_negative"]
