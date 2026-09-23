"""Tests for the KPI engine, including adaptive omission and edge cases."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.kpi_engine import (
    KPI_REGISTRY,
    KPIFormat,
    calculate_kpis,
    format_currency,
    format_value,
    kpi_catalogue,
)


class TestFormatting:
    """Display formatting. Rounding happens here and nowhere else."""

    def test_currency_basic(self) -> None:
        assert format_currency(167_484.32) == "NT$167,484"

    def test_currency_compact(self) -> None:
        assert format_currency(5_024_529_680, compact=True) == "NT$5.02B"
        assert format_currency(949_541_777, compact=True) == "NT$949.54M"
        assert format_currency(51_223, compact=True) == "NT$51.22K"

    def test_currency_negative(self) -> None:
        assert format_currency(-1_500).startswith("-NT$")

    def test_currency_handles_non_finite(self) -> None:
        assert format_currency(float("inf")) == "n/a"
        assert format_currency(float("nan")) == "n/a"

    def test_integer_uses_half_up_rounding(self) -> None:
        """Python's round() rounds half to even; a dashboard should not."""
        assert format_value(1234.5, KPIFormat.INTEGER) == "1,235"
        assert format_value(1233.5, KPIFormat.INTEGER) == "1,234"
        assert format_value(-1234.5, KPIFormat.INTEGER) == "-1,235"

    def test_each_format_type(self) -> None:
        assert format_value(1234.4, KPIFormat.INTEGER) == "1,234"
        assert format_value(22.12, KPIFormat.PERCENTAGE) == "22.12%"
        assert format_value(1.5, KPIFormat.RATIO) == "1.50x"
        assert format_value(0.68, KPIFormat.MONTHS) == "0.68 months"
        assert format_value(35.4855, KPIFormat.DECIMAL) == "35.49"

    def test_none_and_nan_render_as_na(self) -> None:
        assert format_value(None, KPIFormat.INTEGER) == "n/a"
        assert format_value(float("nan"), KPIFormat.PERCENTAGE) == "n/a"


class TestKpiCalculation:
    """Core KPI computation."""

    def test_computes_kpis_on_a_full_frame(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df)
        assert len(result) > 20
        assert "total_clients" in result

    def test_total_clients_counts_distinct_ids(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df)
        assert result.raw("total_clients") == featured_df["client_id"].nunique()

    def test_default_rate_matches_manual_calculation(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df)
        expected = float(featured_df["default_next_month"].mean() * 100)
        assert result.raw("observed_default_rate") == pytest.approx(expected)

    def test_credit_limit_statistics(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df)
        assert result.raw("avg_credit_limit") == pytest.approx(
            float(featured_df["credit_limit"].mean())
        )
        assert result.raw("median_credit_limit") == pytest.approx(
            float(featured_df["credit_limit"].median())
        )
        assert result.raw("total_credit_limit") == pytest.approx(
            float(featured_df["credit_limit"].sum())
        )

    def test_outstanding_balance_uses_most_recent_month(
        self, featured_df: pd.DataFrame
    ) -> None:
        result = calculate_kpis(featured_df)
        assert result.raw("total_outstanding_balance") == pytest.approx(
            float(featured_df["bill_amt_m1"].sum())
        )

    def test_delinquency_kpis_match_manual_calculation(
        self, featured_df: pd.DataFrame
    ) -> None:
        result = calculate_kpis(featured_df)
        expected_count = int((featured_df["pay_status_m1"] >= 1).sum())
        assert result.raw("currently_delinquent_count") == expected_count

    def test_full_precision_is_retained(self, featured_df: pd.DataFrame) -> None:
        """Rounding must only affect the display string."""
        result = calculate_kpis(featured_df)
        raw = result.raw("avg_credit_limit")
        assert raw == float(featured_df["credit_limit"].mean())
        assert isinstance(raw, float)

    def test_formatted_and_raw_are_both_available(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df)
        kpi = result.get("observed_default_rate")
        assert kpi.value != kpi.formatted
        assert "%" in kpi.formatted

    def test_every_kpi_has_a_formula_and_description(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df)
        for kpi in result.values.values():
            assert kpi.formula.strip()
            assert len(kpi.description) > 15


class TestAdaptiveBehaviour:
    """KPIs must be omitted, not faked, when the data cannot support them."""

    def test_omits_unsupported_kpis(self) -> None:
        frame = pd.DataFrame({"client_id": [1, 2, 3], "age": [30, 40, 50]})
        result = calculate_kpis(frame)
        computed = set(result.to_dict())
        assert computed == {"total_clients", "total_records", "avg_age", "median_age"}

    def test_every_omission_carries_a_reason(self) -> None:
        frame = pd.DataFrame({"client_id": [1, 2, 3], "age": [30, 40, 50]})
        result = calculate_kpis(frame)
        assert len(result.unavailable) > 20
        for key, reason in result.unavailable.items():
            assert reason.strip(), f"'{key}' was omitted without explanation"

    def test_explains_the_absent_income_role(self, featured_df: pd.DataFrame) -> None:
        """The single most important honesty point must be surfaced."""
        result = calculate_kpis(featured_df)
        assert "role:income" in result.unavailable
        assert "no income column" in result.unavailable["role:income"].lower()

    def test_no_income_kpi_is_ever_produced(self, featured_df: pd.DataFrame) -> None:
        computed = set(calculate_kpis(featured_df).to_dict())
        for forbidden in ("avg_income", "median_income", "avg_dti", "savings_rate"):
            assert forbidden not in computed

    def test_missing_target_omits_risk_kpis(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.drop(columns=["default_next_month"])
        result = calculate_kpis(frame)
        assert "observed_default_rate" not in result
        assert "observed_default_rate" in result.unavailable


class TestEdgeCases:
    """Behaviour at the boundaries."""

    def test_empty_frame_returns_no_kpis(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df.iloc[0:0])
        assert len(result) == 0
        assert result.insufficient_data is True
        assert result.n_rows == 0

    def test_empty_frame_explains_every_kpi(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df.iloc[0:0])
        assert len(result.unavailable) == len(KPI_REGISTRY)

    def test_small_selection_is_flagged_but_computed(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df.head(3))
        assert result.insufficient_data is True
        assert len(result) > 0, "values are still computed so the UI can warn, not hide"

    def test_single_row(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df.head(1))
        assert result.raw("total_clients") == 1

    def test_single_class_target(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.copy()
        frame["default_next_month"] = 0
        result = calculate_kpis(frame)
        assert result.raw("observed_default_rate") == pytest.approx(0.0)
        assert result.raw("default_client_count") == pytest.approx(0.0)

    def test_all_null_column_omits_its_kpi(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.copy()
        frame["credit_limit"] = np.nan
        result = calculate_kpis(frame)
        assert "avg_credit_limit" in result.unavailable

    def test_infinities_are_excluded_not_propagated(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.copy()
        frame.loc[frame.index[0], "utilisation_latest"] = np.inf
        result = calculate_kpis(frame)
        value = result.raw("avg_utilisation")
        assert value is None or math.isfinite(value)

    def test_zero_denominators_do_not_raise(self) -> None:
        frame = pd.DataFrame(
            {
                "client_id": [1, 2],
                "credit_limit": [0, 0],
                "age": [30, 40],
                "default_next_month": [0, 1],
                **{f"bill_amt_m{m}": [100, 200] for m in range(1, 7)},
                **{f"pay_amt_m{m}": [0, 0] for m in range(1, 7)},
                **{f"pay_status_m{m}": [0, 0] for m in range(1, 7)},
            }
        )
        result = calculate_kpis(frame)
        assert result.raw("total_clients") == 2

    def test_all_values_are_finite(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df)
        for key, kpi in result.values.items():
            assert math.isfinite(kpi.value), f"KPI '{key}' is not finite"


class TestResultInterface:
    """The KPIResult container."""

    def test_to_dict_is_flat_key_to_value(self, featured_df: pd.DataFrame) -> None:
        flat = calculate_kpis(featured_df).to_dict()
        assert all(isinstance(v, float) for v in flat.values())

    def test_to_display_dict_is_strings(self, featured_df: pd.DataFrame) -> None:
        display = calculate_kpis(featured_df).to_display_dict()
        assert all(isinstance(v, str) for v in display.values())

    def test_full_dict_is_serialisable(self, featured_df: pd.DataFrame) -> None:
        import json

        json.dumps(calculate_kpis(featured_df).to_full_dict())

    def test_grouping_by_category(self, featured_df: pd.DataFrame) -> None:
        grouped = calculate_kpis(featured_df).by_category()
        assert len(grouped) >= 5
        assert all(isinstance(v, list) and v for v in grouped.values())

    def test_missing_key_returns_defaults(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df)
        assert result.get("does_not_exist") is None
        assert result.raw("does_not_exist") is None
        assert result.formatted("does_not_exist") == "n/a"

    def test_contains_operator(self, featured_df: pd.DataFrame) -> None:
        result = calculate_kpis(featured_df)
        assert "total_clients" in result
        assert "nonsense" not in result


class TestCatalogue:
    """The definition catalogue used by the Methodology page."""

    def test_describes_every_registered_kpi(self) -> None:
        assert len(kpi_catalogue()) == len(KPI_REGISTRY)

    def test_entries_are_complete(self) -> None:
        for entry in kpi_catalogue():
            assert entry["key"] and entry["label"] and entry["formula"]
            assert entry["description"]

    def test_keys_are_unique(self) -> None:
        keys = [definition.key for definition in KPI_REGISTRY]
        assert len(keys) == len(set(keys))


@pytest.mark.integration
class TestRealKpis:
    """KPI values verified against the real dataset."""

    def test_portfolio_scale(self, real_pipeline) -> None:
        result = calculate_kpis(real_pipeline.data, real_pipeline.registry)
        assert result.raw("total_clients") == 30_000
        assert result.raw("client_month_records") == 180_000

    def test_default_rate_is_the_verified_value(self, real_pipeline) -> None:
        result = calculate_kpis(real_pipeline.data, real_pipeline.registry)
        assert result.raw("observed_default_rate") == pytest.approx(22.12, abs=0.01)
        assert result.raw("default_client_count") == 6_636

    def test_credit_limits_match_verified_values(self, real_pipeline) -> None:
        result = calculate_kpis(real_pipeline.data, real_pipeline.registry)
        assert result.raw("avg_credit_limit") == pytest.approx(167_484.32, abs=0.01)
        assert result.raw("median_credit_limit") == pytest.approx(140_000.0)

    def test_balance_statistics_match_verified_values(self, real_pipeline) -> None:
        result = calculate_kpis(real_pipeline.data, real_pipeline.registry)
        assert result.raw("avg_outstanding_balance") == pytest.approx(51_223.33, abs=0.01)
        assert result.raw("median_outstanding_balance") == pytest.approx(22_381.5)

    def test_utilisation_and_delinquency(self, real_pipeline) -> None:
        result = calculate_kpis(real_pipeline.data, real_pipeline.registry)
        assert result.raw("avg_utilisation") == pytest.approx(42.38, abs=0.01)
        assert result.raw("high_utilisation_count") == 7_980
        assert result.raw("currently_delinquent_count") == 6_818

    def test_age_statistics(self, real_pipeline) -> None:
        result = calculate_kpis(real_pipeline.data, real_pipeline.registry)
        assert result.raw("avg_age") == pytest.approx(35.4855, abs=0.001)
        assert result.raw("median_age") == pytest.approx(34.0)

    def test_mean_exceeds_median_for_balances(self, real_pipeline) -> None:
        """Confirms the right skew that motivates reporting both."""
        result = calculate_kpis(real_pipeline.data, real_pipeline.registry)
        assert result.raw("avg_outstanding_balance") > result.raw(
            "median_outstanding_balance"
        )
