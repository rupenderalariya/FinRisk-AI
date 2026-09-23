"""Tests for panel trend analysis, including the no-fabrication guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.trends import (
    NO_TEMPORAL_DATA_MESSAGE,
    TrendDirection,
    TrendResult,
    TrendUnavailable,
    active_account_trend,
    balance_trend,
    delinquency_trend,
    panel_summary,
    repayment_trend,
    run_trend_analysis,
    temporal_capability,
    utilisation_trend,
)


class TestTemporalCapability:
    """Honest reporting of what temporal analysis is possible."""

    def test_detects_the_monthly_panel(self, featured_df: pd.DataFrame) -> None:
        capability = temporal_capability(featured_df)
        assert capability["has_monthly_panel"] is True
        assert capability["n_periods"] == 6
        assert capability["supports_period_comparison"] is True

    def test_reports_no_calendar_date(self, featured_df: pd.DataFrame) -> None:
        assert temporal_capability(featured_df)["has_calendar_date"] is False

    def test_daily_and_weekly_are_unsupported(self, featured_df: pd.DataFrame) -> None:
        assert temporal_capability(featured_df)["supports_daily_weekly"] is False

    def test_forecasting_is_refused_with_a_reason(self, featured_df: pd.DataFrame) -> None:
        """Six points cannot support a projection, and we say so."""
        capability = temporal_capability(featured_df)
        assert capability["supports_forecasting"] is False
        assert "unreliable" in capability["forecasting_note"].lower()

    def test_period_labels_run_oldest_to_newest(self, featured_df: pd.DataFrame) -> None:
        labels = temporal_capability(featured_df)["period_labels"]
        assert labels[0] == "Apr 2005"
        assert labels[-1] == "Sep 2005"

    def test_no_panel_gives_the_standard_message(self) -> None:
        frame = pd.DataFrame({"client_id": [1, 2], "age": [30, 40]})
        capability = temporal_capability(frame)
        assert capability["has_monthly_panel"] is False
        assert capability["note"] == NO_TEMPORAL_DATA_MESSAGE


class TestBalanceTrend:
    """Statement balance across periods."""

    def test_detects_a_rising_balance(self, featured_df: pd.DataFrame) -> None:
        """The fixture's balances rise from April to September by construction."""
        result = balance_trend(featured_df)
        assert isinstance(result, TrendResult)
        assert result.direction is TrendDirection.INCREASING
        assert result.total_change > 0

    def test_points_are_in_chronological_order(self, featured_df: pd.DataFrame) -> None:
        result = balance_trend(featured_df)
        assert result.points[0].month_label == "Apr 2005"
        assert result.points[-1].month_label == "Sep 2005"
        orders = [p.month_order for p in result.points]
        assert orders == sorted(orders)

    def test_values_match_manual_aggregation(self, featured_df: pd.DataFrame) -> None:
        result = balance_trend(featured_df, statistic="mean")
        for point in result.points:
            expected = float(featured_df[f"bill_amt_m{point.month_index}"].mean())
            assert point.value == pytest.approx(expected)

    def test_period_changes_are_computed(self, featured_df: pd.DataFrame) -> None:
        result = balance_trend(featured_df)
        assert result.points[0].change_from_previous is None, "first period has no predecessor"
        for point in result.points[1:]:
            assert point.change_from_previous is not None

    def test_median_statistic_is_supported(self, featured_df: pd.DataFrame) -> None:
        result = balance_trend(featured_df, statistic="median")
        assert "Median" in result.label

    def test_flat_series_is_called_flat(self, edge_case_credit_df: pd.DataFrame) -> None:
        """The edge-case fixture holds every balance constant across months."""
        result = balance_trend(edge_case_credit_df)
        assert result.direction is TrendDirection.FLAT

    def test_unavailable_without_balance_columns(self) -> None:
        frame = pd.DataFrame({"client_id": [1], "age": [30]})
        result = balance_trend(frame)
        assert isinstance(result, TrendUnavailable)
        assert result.is_available is False

    def test_peak_and_trough_are_identified(self, featured_df: pd.DataFrame) -> None:
        result = balance_trend(featured_df)
        values = {p.month_label: p.value for p in result.points}
        assert values[result.peak_period] == max(values.values())
        assert values[result.trough_period] == min(values.values())

    def test_describe_is_grounded_in_the_numbers(self, featured_df: pd.DataFrame) -> None:
        description = balance_trend(featured_df).describe()
        assert "Apr 2005" in description and "Sep 2005" in description


class TestOtherTrends:
    """The remaining measures."""

    def test_repayment_trend(self, featured_df: pd.DataFrame) -> None:
        result = repayment_trend(featured_df)
        assert isinstance(result, TrendResult)
        assert result.unit == "NT$"

    def test_utilisation_trend_is_a_percentage(self, featured_df: pd.DataFrame) -> None:
        result = utilisation_trend(featured_df)
        assert isinstance(result, TrendResult)
        assert result.unit == "%"

    def test_utilisation_trend_needs_engineered_columns(
        self, minimal_credit_df: pd.DataFrame
    ) -> None:
        result = utilisation_trend(minimal_credit_df)
        assert isinstance(result, TrendUnavailable)
        assert "feature" in result.reason.lower()

    def test_delinquency_trend_uses_documented_codes_only(
        self, featured_df: pd.DataFrame
    ) -> None:
        result = delinquency_trend(featured_df)
        assert isinstance(result, TrendResult)
        assert "documented" in result.note.lower()
        for point in result.points:
            expected = float(
                (featured_df[f"pay_status_m{point.month_index}"] >= 1).mean() * 100
            )
            assert point.value == pytest.approx(expected)

    def test_active_account_trend_states_it_is_a_proxy(
        self, featured_df: pd.DataFrame
    ) -> None:
        result = active_account_trend(featured_df)
        assert isinstance(result, TrendResult)
        assert "proxy" in result.note.lower()
        assert "transaction" in result.note.lower()

    def test_all_percentages_are_in_range(self, featured_df: pd.DataFrame) -> None:
        for result in (delinquency_trend(featured_df), active_account_trend(featured_df)):
            for point in result.points:
                assert 0.0 <= point.value <= 100.0


class TestPanelSummary:
    """Aggregating the long panel per period."""

    def test_one_row_per_period(self, real_or_synthetic_panel: pd.DataFrame) -> None:
        summary = panel_summary(real_or_synthetic_panel)
        assert len(summary) == 6

    def test_has_expected_measures(self, real_or_synthetic_panel: pd.DataFrame) -> None:
        summary = panel_summary(real_or_synthetic_panel)
        for column in (
            "n_clients", "mean_bill", "median_bill", "total_bill",
            "mean_payment", "delinquency_rate_pct", "active_share_pct",
        ):
            assert column in summary.columns

    def test_sorted_chronologically(self, real_or_synthetic_panel: pd.DataFrame) -> None:
        summary = panel_summary(real_or_synthetic_panel)
        assert summary["month_order"].tolist() == sorted(summary["month_order"].tolist())
        assert summary.iloc[0]["month_label"] == "Apr 2005"

    def test_empty_panel_returns_empty_frame_with_columns(self) -> None:
        summary = panel_summary(pd.DataFrame())
        assert summary.empty
        assert "month_index" in summary.columns


@pytest.fixture
def real_or_synthetic_panel(featured_df: pd.DataFrame) -> pd.DataFrame:
    """A long panel built from the synthetic fixture."""
    from src.data_processing import build_panel_long

    return build_panel_long(featured_df)


class TestRunTrendAnalysis:
    """The orchestrated trend run."""

    def test_computes_every_measure(self, featured_df: pd.DataFrame) -> None:
        output = run_trend_analysis(featured_df)
        assert output["available"] is True
        assert output["n_computed"] == 8

    def test_note_explains_the_panel_limitation(self, featured_df: pd.DataFrame) -> None:
        message = run_trend_analysis(featured_df)["message"]
        assert "forecast" in message.lower()
        assert "seasonality" in message.lower()

    def test_unavailable_without_a_panel(self) -> None:
        frame = pd.DataFrame({"client_id": [1, 2], "age": [30, 40]})
        output = run_trend_analysis(frame)
        assert output["available"] is False
        assert output["message"] == NO_TEMPORAL_DATA_MESSAGE
        assert output["trends"] == {}

    def test_results_are_serialisable(self, featured_df: pd.DataFrame) -> None:
        import json

        output = run_trend_analysis(featured_df)
        for result in output["trends"].values():
            json.dumps(result.to_dict())

    def test_to_frame_is_chart_ready(self, featured_df: pd.DataFrame) -> None:
        result = run_trend_analysis(featured_df)["trends"]["balance_mean"]
        frame = result.to_frame()
        assert "month_label" in frame.columns and "value" in frame.columns
        assert len(frame) == 6


class TestNoFabrication:
    """The no-invented-data guarantees."""

    def test_single_period_is_refused(self, featured_df: pd.DataFrame) -> None:
        """One period cannot make a trend, so nothing is invented to fill the gap."""
        frame = featured_df.drop(columns=[f"bill_amt_m{m}" for m in range(2, 7)])
        result = balance_trend(frame)
        assert isinstance(result, TrendUnavailable)
        assert "at least two" in result.reason.lower()

    def test_point_count_never_exceeds_available_months(
        self, featured_df: pd.DataFrame
    ) -> None:
        frame = featured_df.drop(columns=["bill_amt_m5", "bill_amt_m6"])
        result = balance_trend(frame)
        assert isinstance(result, TrendResult)
        assert result.n_periods == 4, "must not invent the dropped months"

    def test_observation_counts_are_real(self, featured_df: pd.DataFrame) -> None:
        result = balance_trend(featured_df)
        for point in result.points:
            assert point.n_observations == len(featured_df)


class TestAnomalyDetection:
    """Anomaly flagging, presented as weak evidence."""

    def test_no_anomalies_in_a_smooth_series(self, featured_df: pd.DataFrame) -> None:
        assert balance_trend(featured_df).anomalies == ()

    def test_flags_a_planted_spike(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.copy()
        frame["bill_amt_m3"] = frame["bill_amt_m3"] * 50
        result = balance_trend(frame)
        assert isinstance(result, TrendResult)
        assert len(result.anomalies) >= 1


@pytest.mark.integration
class TestRealTrends:
    """Trends measured on the real dataset."""

    def test_balances_rose_over_the_window(self, real_pipeline) -> None:
        result = balance_trend(real_pipeline.data, real_pipeline.registry, "mean")
        assert result.direction is TrendDirection.INCREASING
        assert result.first_value == pytest.approx(38_871.76, abs=0.01)  # Apr 2005
        assert result.last_value == pytest.approx(51_223.33, abs=0.01)   # Sep 2005
        assert result.total_change_pct == pytest.approx(31.8, abs=0.2)

    def test_delinquency_rose_over_the_window(self, real_pipeline) -> None:
        result = delinquency_trend(real_pipeline.data, real_pipeline.registry)
        assert result.direction is TrendDirection.INCREASING
        assert result.first_value == pytest.approx(10.263, abs=0.01)  # Apr 2005
        assert result.last_value == pytest.approx(22.727, abs=0.01)   # Sep 2005

    def test_utilisation_rose_over_the_window(self, real_pipeline) -> None:
        result = utilisation_trend(real_pipeline.data, real_pipeline.registry)
        assert result.direction is TrendDirection.INCREASING
        assert result.last_value == pytest.approx(42.377, abs=0.01)

    def test_panel_summary_covers_all_clients(self, real_pipeline) -> None:
        summary = panel_summary(real_pipeline.panel)
        assert len(summary) == 6
        assert (summary["n_clients"] == 30_000).all()

    def test_all_eight_measures_compute(self, real_pipeline) -> None:
        output = run_trend_analysis(real_pipeline.data, real_pipeline.registry)
        assert output["n_computed"] == 8
