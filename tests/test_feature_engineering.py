"""Tests for feature engineering, with emphasis on division-by-zero safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FULL_PAYER_RATIO,
    REPAYMENT_RATIO_CAP,
    REVOLVER_RATIO,
    _ols_slope,
    add_balance_trajectory_features,
    add_band_features,
    add_delinquency_features,
    add_repayment_features,
    add_utilisation_features,
    engineer_features,
    safe_divide,
    unavailable_feature_definitions,
)
from src.schema import build_registry


class TestSafeDivide:
    """The guard that keeps `inf` out of every downstream calculation."""

    def test_normal_division(self) -> None:
        result = safe_divide(pd.Series([10.0, 20.0]), pd.Series([2.0, 4.0]))
        assert result.tolist() == [5.0, 5.0]

    def test_zero_denominator_returns_nan_not_inf(self) -> None:
        result = safe_divide(pd.Series([10.0]), pd.Series([0.0]))
        assert pd.isna(result.iloc[0])
        assert not np.isinf(result.iloc[0]) if pd.notna(result.iloc[0]) else True

    def test_zero_over_zero_returns_nan(self) -> None:
        assert pd.isna(safe_divide(pd.Series([0.0]), pd.Series([0.0])).iloc[0])

    def test_nan_denominator_returns_nan(self) -> None:
        assert pd.isna(safe_divide(pd.Series([10.0]), pd.Series([np.nan])).iloc[0])

    def test_negative_denominator_allowed_by_default(self) -> None:
        assert safe_divide(pd.Series([10.0]), pd.Series([-5.0])).iloc[0] == -2.0

    def test_negative_denominator_rejected_when_positive_required(self) -> None:
        result = safe_divide(
            pd.Series([10.0]), pd.Series([-5.0]), require_positive_denominator=True
        )
        assert pd.isna(result.iloc[0])

    def test_scalar_denominator(self) -> None:
        assert safe_divide(pd.Series([10.0, 20.0]), 2.0).tolist() == [5.0, 10.0]

    def test_never_produces_infinity(self) -> None:
        result = safe_divide(
            pd.Series([1.0, -1.0, 0.0, 1e300]), pd.Series([0.0, 0.0, 0.0, 1e-300])
        )
        assert not np.isinf(result.dropna()).any()

    def test_non_numeric_input_coerced_to_nan(self) -> None:
        result = safe_divide(pd.Series(["x"]), pd.Series([2.0]))
        assert pd.isna(result.iloc[0])


class TestOlsSlope:
    """The vectorised trend slope."""

    def test_matches_polyfit(self) -> None:
        values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        x = np.arange(1, 7, dtype="float64")
        expected = np.polyfit(x, values[0], 1)[0]
        assert _ols_slope(values, x)[0] == pytest.approx(expected)

    def test_flat_series_has_zero_slope(self) -> None:
        values = np.array([[5.0] * 6])
        assert _ols_slope(values, np.arange(1, 7, dtype="float64"))[0] == pytest.approx(0.0)

    def test_declining_series_has_negative_slope(self) -> None:
        values = np.array([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
        assert _ols_slope(values, np.arange(1, 7, dtype="float64"))[0] < 0

    def test_insufficient_points_returns_nan(self) -> None:
        values = np.array([[1.0, np.nan, np.nan, np.nan, np.nan, np.nan]])
        assert np.isnan(_ols_slope(values, np.arange(1, 7, dtype="float64"))[0])

    def test_handles_partial_nans(self) -> None:
        values = np.array([[1.0, 2.0, np.nan, 4.0, 5.0, 6.0]])
        assert np.isfinite(_ols_slope(values, np.arange(1, 7, dtype="float64"))[0])


class TestUtilisationFeatures:
    """Credit utilisation."""

    def test_computes_expected_value(self, minimal_credit_df: pd.DataFrame) -> None:
        registry = build_registry(minimal_credit_df)
        result, _ = add_utilisation_features(minimal_credit_df, registry)
        expected = (
            minimal_credit_df.loc[0, "bill_amt_m1"] / minimal_credit_df.loc[0, "credit_limit"]
        )
        assert result.loc[0, "utilisation_m1"] == pytest.approx(expected)
        assert result.loc[0, "utilisation_latest"] == pytest.approx(expected)

    def test_zero_limit_gives_nan_not_inf(self, edge_case_credit_df: pd.DataFrame) -> None:
        registry = build_registry(edge_case_credit_df)
        result, _ = add_utilisation_features(edge_case_credit_df, registry)
        assert pd.isna(result.loc[0, "utilisation_m1"])
        numeric = result.select_dtypes(include="number")
        assert not np.isinf(numeric).any().any()

    def test_negative_limit_gives_nan(self, edge_case_credit_df: pd.DataFrame) -> None:
        registry = build_registry(edge_case_credit_df)
        result, _ = add_utilisation_features(edge_case_credit_df, registry)
        assert pd.isna(result.loc[6, "utilisation_m1"])

    def test_over_limit_is_detected(self, edge_case_credit_df: pd.DataFrame) -> None:
        registry = build_registry(edge_case_credit_df)
        result, _ = add_utilisation_features(edge_case_credit_df, registry)
        assert bool(result.loc[4, "is_over_limit"]) is True  # 15,000 on a 10,000 limit

    def test_negative_balance_yields_negative_utilisation(
        self, edge_case_credit_df: pd.DataFrame
    ) -> None:
        """A credit balance is meaningful, so the sign must be preserved."""
        registry = build_registry(edge_case_credit_df)
        result, _ = add_utilisation_features(edge_case_credit_df, registry)
        assert result.loc[1, "utilisation_m1"] < 0

    def test_skipped_when_limit_absent(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.drop(columns=["credit_limit"])
        registry = build_registry(frame)
        result, definitions = add_utilisation_features(frame, registry)
        assert "utilisation_latest" not in result.columns
        assert definitions[0].created is False
        assert "credit limit" in definitions[0].skip_reason.lower()

    def test_skipped_when_bills_absent(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.drop(columns=[f"bill_amt_m{m}" for m in range(1, 7)])
        registry = build_registry(frame)
        _result, definitions = add_utilisation_features(frame, registry)
        assert definitions[0].created is False


class TestRepaymentFeatures:
    """Repayment behaviour, including the statement/payment offset."""

    def test_uses_previous_month_statement_as_denominator(
        self, minimal_credit_df: pd.DataFrame
    ) -> None:
        """A payment in month m settles the statement from month m+1."""
        registry = build_registry(minimal_credit_df)
        result, _ = add_repayment_features(minimal_credit_df, registry)
        expected = (
            minimal_credit_df.loc[0, "pay_amt_m1"] / minimal_credit_df.loc[0, "bill_amt_m2"]
        )
        assert result.loc[0, "repayment_ratio_m1"] == pytest.approx(expected)

    def test_zero_prior_balance_gives_nan(self, edge_case_credit_df: pd.DataFrame) -> None:
        registry = build_registry(edge_case_credit_df)
        result, _ = add_repayment_features(edge_case_credit_df, registry)
        assert pd.isna(result.loc[3, "repayment_ratio_m1"])

    def test_negative_prior_balance_gives_nan(self, edge_case_credit_df: pd.DataFrame) -> None:
        registry = build_registry(edge_case_credit_df)
        result, _ = add_repayment_features(edge_case_credit_df, registry)
        assert pd.isna(result.loc[1, "repayment_ratio_m1"])

    def test_oldest_month_has_no_ratio(self, minimal_credit_df: pd.DataFrame) -> None:
        """Month 6 has no earlier statement to compare against."""
        registry = build_registry(minimal_credit_df)
        result, _ = add_repayment_features(minimal_credit_df, registry)
        assert "repayment_ratio_m6" not in result.columns
        assert "repayment_ratio_m5" in result.columns

    def test_capped_mean_is_bounded(self, analysis_df: pd.DataFrame) -> None:
        registry = build_registry(analysis_df)
        result, _ = add_repayment_features(analysis_df, registry)
        capped = result["repayment_ratio_capped_mean"].dropna()
        assert capped.max() <= REPAYMENT_RATIO_CAP
        assert capped.min() >= 0.0

    def test_capped_mean_excluded_from_nothing_but_uncapped_is_not_ml_safe(
        self, analysis_df: pd.DataFrame
    ) -> None:
        """Unbounded ratios must be kept out of model inputs."""
        _result, definitions = add_repayment_features(analysis_df, build_registry(analysis_df))
        by_name = {d.name: d for d in definitions}
        assert by_name["repayment_ratio_mean"].used_for_ml is False
        assert by_name["repayment_ratio_capped_mean"].used_for_ml is True

    def test_full_payer_and_revolver_thresholds(self) -> None:
        """Construct a clear full payer and a clear revolver and check the flags."""
        frame = pd.DataFrame(
            {
                "client_id": [1, 2],
                "credit_limit": [10_000, 10_000],
                "age": [30, 30],
                "default_next_month": [0, 1],
                # Client 1 repays in full; client 2 repays a tenth.
                **{f"bill_amt_m{m}": [1_000, 1_000] for m in range(1, 7)},
                **{f"pay_amt_m{m}": [1_000, 100] for m in range(1, 7)},
                **{f"pay_status_m{m}": [-1, 2] for m in range(1, 7)},
            }
        )
        result, _ = add_repayment_features(frame, build_registry(frame))
        assert result.loc[0, "repayment_ratio_median"] == pytest.approx(1.0)
        assert bool(result.loc[0, "is_full_payer"]) is True
        assert bool(result.loc[0, "is_revolver"]) is False
        assert result.loc[1, "repayment_ratio_median"] == pytest.approx(0.1)
        assert bool(result.loc[1, "is_revolver"]) is True

    def test_counts_zero_payment_months(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.copy()
        for month in (1, 2, 3):
            frame[f"pay_amt_m{month}"] = 0
        result, _ = add_repayment_features(frame, build_registry(frame))
        assert result.loc[0, "months_zero_payment"] == 3

    def test_skipped_when_payments_absent(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.drop(columns=[f"pay_amt_m{m}" for m in range(1, 7)])
        _result, definitions = add_repayment_features(frame, build_registry(frame))
        assert definitions[0].created is False


class TestDelinquencyFeatures:
    """Delinquency counts and severity."""

    def test_counts_delinquent_months(self, minimal_credit_df: pd.DataFrame) -> None:
        result, _ = add_delinquency_features(minimal_credit_df, build_registry(minimal_credit_df))
        # Fixture row 2 has status 1 in every month; row 0 has status 0 throughout.
        assert result.loc[2, "delinquent_months_count"] == 6
        assert result.loc[0, "delinquent_months_count"] == 0

    def test_max_delinquency_is_floored_at_zero(self, minimal_credit_df: pd.DataFrame) -> None:
        """Status -1 (paid in full) must not read as negative severity."""
        result, _ = add_delinquency_features(minimal_credit_df, build_registry(minimal_credit_df))
        assert result.loc[3, "max_delinquency"] == 0.0
        assert (result["max_delinquency"] >= 0).all()

    def test_only_documented_codes_count_as_delinquent(self) -> None:
        """Undocumented -2 and 0 must not be treated as payment delay."""
        frame = pd.DataFrame(
            {
                "client_id": [1],
                "credit_limit": [10_000],
                "age": [30],
                "default_next_month": [0],
                **{f"bill_amt_m{m}": [100] for m in range(1, 7)},
                **{f"pay_amt_m{m}": [50] for m in range(1, 7)},
                **{f"pay_status_m{m}": [-2 if m % 2 else 0] for m in range(1, 7)},
            }
        )
        result, _ = add_delinquency_features(frame, build_registry(frame))
        assert result.loc[0, "delinquent_months_count"] == 0
        assert bool(result.loc[0, "ever_delinquent"]) is False

    def test_current_delinquency_uses_most_recent_month(
        self, minimal_credit_df: pd.DataFrame
    ) -> None:
        frame = minimal_credit_df.copy()
        frame["pay_status_m1"] = [4, 0, 0, 0, 0, 0]
        result, _ = add_delinquency_features(frame, build_registry(frame))
        assert result.loc[0, "current_delinquency"] == 4
        assert bool(result.loc[0, "is_currently_delinquent"]) is True

    def test_trend_detects_worsening(self) -> None:
        """Late in recent months, clean earlier -> Worsening."""
        frame = pd.DataFrame(
            {
                "client_id": [1],
                "credit_limit": [10_000],
                "age": [30],
                "default_next_month": [1],
                **{f"bill_amt_m{m}": [100] for m in range(1, 7)},
                **{f"pay_amt_m{m}": [50] for m in range(1, 7)},
                # m1-m3 recent (delinquent), m4-m6 older (clean)
                **{f"pay_status_m{m}": [2 if m <= 3 else 0] for m in range(1, 7)},
            }
        )
        result, _ = add_delinquency_features(frame, build_registry(frame))
        assert result.loc[0, "delinquency_trend"] == "Worsening"

    def test_trend_detects_improving(self) -> None:
        frame = pd.DataFrame(
            {
                "client_id": [1],
                "credit_limit": [10_000],
                "age": [30],
                "default_next_month": [0],
                **{f"bill_amt_m{m}": [100] for m in range(1, 7)},
                **{f"pay_amt_m{m}": [50] for m in range(1, 7)},
                **{f"pay_status_m{m}": [0 if m <= 3 else 2] for m in range(1, 7)},
            }
        )
        result, _ = add_delinquency_features(frame, build_registry(frame))
        assert result.loc[0, "delinquency_trend"] == "Improving"

    def test_skipped_when_status_absent(self, minimal_credit_df: pd.DataFrame) -> None:
        frame = minimal_credit_df.drop(columns=[f"pay_status_m{m}" for m in range(1, 7)])
        _result, definitions = add_delinquency_features(frame, build_registry(frame))
        assert definitions[0].created is False


class TestBalanceTrajectoryFeatures:
    """Panel-only trajectory features."""

    def test_rising_balance_gives_positive_slope(self, minimal_credit_df: pd.DataFrame) -> None:
        """The fixture's balances rise from April to September."""
        result, _ = add_balance_trajectory_features(
            minimal_credit_df, build_registry(minimal_credit_df)
        )
        assert result.loc[0, "bill_trend_slope"] > 0

    def test_slope_matches_manual_polyfit(self, minimal_credit_df: pd.DataFrame) -> None:
        result, _ = add_balance_trajectory_features(
            minimal_credit_df, build_registry(minimal_credit_df)
        )
        y = np.array(
            [minimal_credit_df.loc[0, f"bill_amt_m{m}"] for m in (6, 5, 4, 3, 2, 1)],
            dtype=float,
        )
        expected = np.polyfit(np.arange(1, 7, dtype=float), y, 1)[0]
        assert result.loc[0, "bill_trend_slope"] == pytest.approx(expected)

    def test_flat_balance_gives_zero_slope(self, edge_case_credit_df: pd.DataFrame) -> None:
        result, _ = add_balance_trajectory_features(
            edge_case_credit_df, build_registry(edge_case_credit_df)
        )
        assert result.loc[0, "bill_trend_slope"] == pytest.approx(0.0)

    def test_growth_is_nan_when_starting_balance_is_zero(
        self, edge_case_credit_df: pd.DataFrame
    ) -> None:
        result, _ = add_balance_trajectory_features(
            edge_case_credit_df, build_registry(edge_case_credit_df)
        )
        assert pd.isna(result.loc[3, "balance_growth_6m"])

    def test_growth_is_excluded_from_ml_features(self, analysis_df: pd.DataFrame) -> None:
        """Unbounded ratio; bill_trend_slope carries the same signal safely."""
        _result, definitions = add_balance_trajectory_features(
            analysis_df, build_registry(analysis_df)
        )
        by_name = {d.name: d for d in definitions}
        assert by_name["balance_growth_6m"].used_for_ml is False
        assert by_name["bill_trend_slope"].used_for_ml is True


class TestBandFeatures:
    """Categorical bands."""

    def test_creates_limit_and_age_bands(self, analysis_df: pd.DataFrame) -> None:
        result, _ = add_band_features(analysis_df, build_registry(analysis_df))
        assert "limit_band" in result.columns
        assert "age_band" in result.columns

    def test_limit_band_is_described_as_a_limit_not_income(
        self, analysis_df: pd.DataFrame
    ) -> None:
        """Guards the single most important honesty point about this dataset."""
        _result, definitions = add_band_features(analysis_df, build_registry(analysis_df))
        limit_band = next(d for d in definitions if d.name == "limit_band")
        interpretation = limit_band.interpretation.lower()
        assert "not income" in interpretation
        assert "credit limit" in interpretation

    def test_age_bands_use_fixed_edges(self, analysis_df: pd.DataFrame) -> None:
        result, _ = add_band_features(analysis_df, build_registry(analysis_df))
        categories = set(result["age_band"].cat.categories)
        assert categories == {"Under 30", "30-39", "40-49", "50-59", "60+"}

    def test_age_band_assignment_is_correct(self, minimal_credit_df: pd.DataFrame) -> None:
        result, _ = add_band_features(minimal_credit_df, build_registry(minimal_credit_df))
        ages = minimal_credit_df["age"].tolist()  # [25, 32, 41, 55, 63, 29]
        bands = result["age_band"].astype("string").tolist()
        assert bands[0] == "Under 30"  # 25
        assert bands[1] == "30-39"     # 32
        assert bands[2] == "40-49"     # 41
        assert bands[3] == "50-59"     # 55
        assert bands[4] == "60+"       # 63
        assert bands[5] == "Under 30"  # 29

    def test_skipped_when_columns_absent(self) -> None:
        frame = pd.DataFrame({"client_id": [1, 2, 3]})
        _result, definitions = add_band_features(frame, build_registry(frame))
        assert all(d.created is False for d in definitions)


class TestUnavailableFeatures:
    """Features the dataset genuinely cannot support."""

    def test_income_based_features_are_declared_unavailable(
        self, minimal_credit_df: pd.DataFrame
    ) -> None:
        definitions = unavailable_feature_definitions(build_registry(minimal_credit_df))
        names = {d.name for d in definitions}
        assert {"debt_to_income_ratio", "savings_rate", "expense_ratio"} <= names
        for definition in definitions:
            assert definition.created is False
            assert "do not exist" in definition.skip_reason

    def test_they_are_not_used_anywhere(self, minimal_credit_df: pd.DataFrame) -> None:
        for definition in unavailable_feature_definitions(build_registry(minimal_credit_df)):
            assert definition.used_for_analytics is False
            assert definition.used_for_ml is False


class TestEngineerFeatures:
    """The orchestrated feature run."""

    def test_creates_features_and_reports_skips(self, analysis_df: pd.DataFrame) -> None:
        result, report = engineer_features(analysis_df)
        assert len(report.created) > 20
        assert len(report.skipped) >= 5
        assert len(result.columns) > len(analysis_df.columns)

    def test_every_created_feature_is_documented(self, analysis_df: pd.DataFrame) -> None:
        _result, report = engineer_features(analysis_df)
        for definition in report.created:
            assert definition.formula.strip()
            assert definition.source_columns
            assert len(definition.interpretation) > 20

    def test_no_infinities_are_produced(self, edge_case_credit_df: pd.DataFrame) -> None:
        result, _ = engineer_features(edge_case_credit_df)
        numeric = result.select_dtypes(include="number")
        assert not np.isinf(numeric).any().any()

    def test_row_count_is_preserved(self, analysis_df: pd.DataFrame) -> None:
        result, _ = engineer_features(analysis_df)
        assert len(result) == len(analysis_df)

    def test_is_deterministic(self, analysis_df: pd.DataFrame) -> None:
        first, _ = engineer_features(analysis_df)
        second, _ = engineer_features(analysis_df)
        pd.testing.assert_frame_equal(first, second)

    def test_ml_feature_list_excludes_unbounded_ratios(
        self, analysis_df: pd.DataFrame
    ) -> None:
        _result, report = engineer_features(analysis_df)
        ml_features = set(report.ml_feature_names())
        assert "repayment_ratio_mean" not in ml_features
        assert "balance_growth_6m" not in ml_features
        assert "repayment_ratio_capped_mean" in ml_features

    def test_works_on_a_bare_frame_without_crashing(self) -> None:
        """A frame with almost nothing must skip everything, not raise."""
        frame = pd.DataFrame({"client_id": [1, 2, 3], "age": [30, 40, 50]})
        result, report = engineer_features(frame)
        assert len(result) == 3
        assert len(report.skipped) > 0

    def test_report_is_serialisable(self, analysis_df: pd.DataFrame) -> None:
        import json

        _result, report = engineer_features(analysis_df)
        json.dumps(report.to_dict())


@pytest.mark.integration
class TestRealFeatures:
    """Feature engineering on the real dataset."""

    def test_creates_thirty_seven_features(self, real_pipeline) -> None:
        assert len(real_pipeline.feature_report.created) == 37

    def test_skips_the_five_income_based_features(self, real_pipeline) -> None:
        skipped = {d.name for d in real_pipeline.feature_report.skipped}
        assert skipped == {
            "debt_to_income_ratio", "savings_rate", "expense_ratio",
            "loan_to_income_ratio", "income_band",
        }

    def test_no_infinities_in_the_output(self, real_pipeline) -> None:
        numeric = real_pipeline.data.select_dtypes(include="number")
        assert not np.isinf(numeric).any().any()

    def test_utilisation_matches_manual_calculation(self, real_pipeline) -> None:
        data = real_pipeline.data
        expected = data["bill_amt_m1"] / data["credit_limit"]
        pd.testing.assert_series_equal(
            data["utilisation_latest"], expected, check_names=False
        )
