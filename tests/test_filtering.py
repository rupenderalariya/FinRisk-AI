"""Tests for the global filter logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.filtering import (
    DELINQUENCY_OPTIONS,
    FilterBounds,
    FilterSpec,
    apply_filters,
    assign_utilisation_band,
    default_spec,
    derive_bounds,
    is_default,
    limit_band_documentation,
    utilisation_band_documentation,
)


class TestDeriveBounds:
    """Reading selectable ranges from the data."""

    def test_reads_age_range(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        assert bounds.age_min == int(featured_df["age"].min())
        assert bounds.age_max == int(featured_df["age"].max())
        assert bounds.has_age

    def test_reads_limit_range(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        assert bounds.limit_min == pytest.approx(float(featured_df["credit_limit"].min()))
        assert bounds.limit_max == pytest.approx(float(featured_df["credit_limit"].max()))

    def test_lists_category_options(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        assert set(bounds.education_options) <= set(
            featured_df["education"].astype("string").unique()
        )
        assert bounds.education_options == tuple(sorted(bounds.education_options))

    def test_clips_utilisation_upper_bound(self, featured_df: pd.DataFrame) -> None:
        """A handful of extreme over-limit accounts must not stretch the slider."""
        frame = featured_df.copy()
        frame.loc[frame.index[0], "utilisation_latest"] = 50.0
        bounds = derive_bounds(frame)
        assert bounds.utilisation_max < 50.0

    def test_empty_frame_returns_safe_defaults(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df.iloc[0:0])
        assert bounds.age_min == 0 and bounds.age_max == 100

    def test_missing_columns_disable_those_filters(self) -> None:
        bounds = derive_bounds(pd.DataFrame({"client_id": [1, 2, 3]}))
        assert not bounds.has_age
        assert not bounds.has_limit
        assert bounds.education_options == ()


class TestDefaultSpec:
    """The unconstrained specification."""

    def test_covers_full_ranges(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        spec = default_spec(bounds)
        assert spec.age_range == (bounds.age_min, bounds.age_max)
        assert spec.delinquency == "all"

    def test_recognised_as_default(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        assert is_default(default_spec(bounds), bounds)
        assert is_default(FilterSpec(), bounds)

    def test_narrowed_spec_is_not_default(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        assert not is_default(FilterSpec(age_range=(30, 35)), bounds)

    def test_reset_returns_unconstrained(self) -> None:
        assert FilterSpec(age_range=(30, 40)).reset() == FilterSpec()


class TestApplyFilters:
    """Applying a specification."""

    def test_default_spec_selects_everything(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        result = apply_filters(featured_df, default_spec(bounds), bounds=bounds)
        assert result.n_selected == len(featured_df)
        assert not result.is_filtered
        assert result.applied == ()

    def test_age_range_narrows(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        result = apply_filters(featured_df, FilterSpec(age_range=(30, 40)), bounds=bounds)
        assert result.n_selected < len(featured_df)
        assert result.data["age"].between(30, 40).all()
        assert result.is_filtered

    def test_limit_range_narrows(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        low, high = 20_000.0, 100_000.0
        result = apply_filters(
            featured_df, FilterSpec(limit_range=(low, high)), bounds=bounds
        )
        assert result.data["credit_limit"].between(low, high).all()

    def test_category_selection_narrows(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        chosen = bounds.education_options[:1]
        result = apply_filters(featured_df, FilterSpec(education=chosen), bounds=bounds)
        assert set(result.data["education"].astype("string").unique()) <= set(chosen)

    def test_selecting_all_categories_is_not_a_filter(
        self, featured_df: pd.DataFrame
    ) -> None:
        bounds = derive_bounds(featured_df)
        result = apply_filters(
            featured_df, FilterSpec(education=bounds.education_options), bounds=bounds
        )
        assert result.n_selected == len(featured_df)
        assert not any("Education" in a for a in result.applied)

    def test_empty_category_selection_yields_nothing(
        self, featured_df: pd.DataFrame
    ) -> None:
        """An explicitly empty selection is honoured, not ignored."""
        bounds = derive_bounds(featured_df)
        result = apply_filters(featured_df, FilterSpec(education=()), bounds=bounds)
        assert result.is_empty
        assert result.n_selected == 0

    @pytest.mark.parametrize("status", ["current", "ever", "never"])
    def test_delinquency_options(self, featured_df: pd.DataFrame, status: str) -> None:
        bounds = derive_bounds(featured_df)
        result = apply_filters(
            featured_df, FilterSpec(delinquency=status), bounds=bounds
        )
        if status == "current":
            assert result.data["is_currently_delinquent"].all()
        elif status == "ever":
            assert result.data["ever_delinquent"].all()
        else:
            assert not result.data["ever_delinquent"].any()

    def test_delinquency_all_is_not_a_filter(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        result = apply_filters(featured_df, FilterSpec(delinquency="all"), bounds=bounds)
        assert result.n_selected == len(featured_df)

    def test_utilisation_range_narrows(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        result = apply_filters(
            featured_df, FilterSpec(utilisation_range=(0.8, 5.0)), bounds=bounds
        )
        values = pd.to_numeric(result.data["utilisation_latest"], errors="coerce")
        assert (values >= 0.8).all()

    def test_combined_filters_intersect(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        combined = apply_filters(
            featured_df,
            FilterSpec(age_range=(30, 45), delinquency="current"),
            bounds=bounds,
        )
        age_only = apply_filters(
            featured_df, FilterSpec(age_range=(30, 45)), bounds=bounds
        )
        assert combined.n_selected <= age_only.n_selected
        assert len(combined.applied) == 2

    def test_impossible_combination_is_empty_not_an_error(
        self, featured_df: pd.DataFrame
    ) -> None:
        bounds = derive_bounds(featured_df)
        result = apply_filters(
            featured_df,
            FilterSpec(age_range=(bounds.age_min, bounds.age_min), limit_range=(1e9, 2e9)),
            bounds=bounds,
        )
        assert result.is_empty
        assert result.share_pct == 0.0

    def test_missing_column_is_skipped_with_a_reason(self) -> None:
        frame = pd.DataFrame({"client_id": [1, 2, 3], "age": [30, 40, 50]})
        result = apply_filters(frame, FilterSpec(utilisation_range=(0.0, 1.0)))
        assert any("Utilisation" in s for s in result.skipped)

    def test_empty_input_frame(self, featured_df: pd.DataFrame) -> None:
        result = apply_filters(featured_df.iloc[0:0], FilterSpec(age_range=(30, 40)))
        assert result.n_total == 0
        assert result.is_empty
        assert result.share_pct == 0.0

    def test_result_is_serialisable(self, featured_df: pd.DataFrame) -> None:
        import json

        bounds = derive_bounds(featured_df)
        json.dumps(apply_filters(featured_df, FilterSpec(age_range=(30, 40)), bounds=bounds).to_dict())

    def test_summary_describes_the_selection(self, featured_df: pd.DataFrame) -> None:
        bounds = derive_bounds(featured_df)
        unfiltered = apply_filters(featured_df, default_spec(bounds), bounds=bounds)
        filtered = apply_filters(featured_df, FilterSpec(age_range=(30, 40)), bounds=bounds)
        assert "no filters" in unfiltered.summary()
        assert "match" in filtered.summary()

    def test_does_not_mutate_the_input(self, featured_df: pd.DataFrame) -> None:
        before = featured_df.copy()
        apply_filters(featured_df, FilterSpec(age_range=(30, 40)))
        pd.testing.assert_frame_equal(featured_df, before)

    def test_excluded_count_adds_up(self, featured_df: pd.DataFrame) -> None:
        result = apply_filters(featured_df, FilterSpec(age_range=(30, 40)))
        assert result.n_selected + result.n_excluded == result.n_total


class TestUtilisationBanding:
    """Utilisation band assignment."""

    def test_assigns_expected_bands(self) -> None:
        frame = pd.DataFrame({"utilisation_latest": [-0.5, 0.0, 0.15, 0.45, 0.70, 0.95]})
        bands = assign_utilisation_band(frame).astype("string").tolist()
        assert bands[0].startswith("Negative")
        assert bands[1].startswith("Negative")  # exactly zero is not "in use"
        assert bands[2].startswith("Low")
        assert bands[3].startswith("Moderate")
        assert bands[4].startswith("High")
        assert bands[5].startswith("Very high")

    def test_uses_the_configured_threshold(self) -> None:
        """The top boundary must come from configuration, not a literal."""
        frame = pd.DataFrame({"utilisation_latest": [0.85]})
        default_band = assign_utilisation_band(frame).astype("string").iloc[0]
        raised_band = (
            assign_utilisation_band(frame, high_threshold=0.90).astype("string").iloc[0]
        )
        assert default_band.startswith("Very high")
        assert raised_band.startswith("High"), "raising the threshold must move the row"

    def test_threshold_appears_in_labels(self) -> None:
        frame = pd.DataFrame({"utilisation_latest": [0.5]})
        categories = list(assign_utilisation_band(frame, high_threshold=0.75).cat.categories)
        assert any("75%" in c for c in categories)

    def test_is_ordered(self, featured_df: pd.DataFrame) -> None:
        assert assign_utilisation_band(featured_df).cat.ordered

    def test_missing_column_yields_nulls(self) -> None:
        bands = assign_utilisation_band(pd.DataFrame({"a": [1, 2]}))
        assert bands.isna().all()

    def test_nan_utilisation_is_unassigned(self) -> None:
        frame = pd.DataFrame({"utilisation_latest": [np.nan, 0.5]})
        bands = assign_utilisation_band(frame)
        assert pd.isna(bands.iloc[0])
        assert pd.notna(bands.iloc[1])

    def test_every_row_is_banded_when_finite(self, featured_df: pd.DataFrame) -> None:
        values = pd.to_numeric(featured_df["utilisation_latest"], errors="coerce")
        bands = assign_utilisation_band(featured_df)
        assert bands[values.notna()].notna().all()


class TestDocumentation:
    """Banding methodology text."""

    def test_utilisation_note_states_the_threshold(self) -> None:
        text = utilisation_band_documentation(0.80)
        assert "0.80" in text
        assert "HIGH_UTILISATION_THRESHOLD" in text

    def test_utilisation_note_explains_negative_band(self) -> None:
        assert "in credit" in utilisation_band_documentation()

    def test_limit_note_denies_income_interpretation(self) -> None:
        """The single most important labelling guarantee for this dataset."""
        text = limit_band_documentation().lower()
        assert "not income" in text
        assert "not wealth" in text
        assert "socioeconomic" in text

    def test_limit_note_explains_quartiles(self) -> None:
        assert "quartile" in limit_band_documentation().lower()

    def test_delinquency_options_are_labelled(self) -> None:
        assert set(DELINQUENCY_OPTIONS) == {"all", "current", "ever", "never"}
        for label in DELINQUENCY_OPTIONS.values():
            assert label and label[0].isupper()


@pytest.mark.integration
class TestRealFiltering:
    """Filtering against the real dataset."""

    def test_unfiltered_selects_all(self, real_pipeline) -> None:
        bounds = derive_bounds(real_pipeline.data, real_pipeline.registry)
        result = apply_filters(
            real_pipeline.data, default_spec(bounds), real_pipeline.registry, bounds
        )
        assert result.n_selected == 30_000
        assert not result.is_filtered

    def test_delinquent_subset_matches_the_kpi(self, real_pipeline) -> None:
        bounds = derive_bounds(real_pipeline.data, real_pipeline.registry)
        result = apply_filters(
            real_pipeline.data,
            FilterSpec(delinquency="current"),
            real_pipeline.registry,
            bounds,
        )
        assert result.n_selected == 6_818

    def test_filtering_changes_the_default_rate(self, real_pipeline) -> None:
        from src.kpi_engine import calculate_kpis

        bounds = derive_bounds(real_pipeline.data, real_pipeline.registry)
        result = apply_filters(
            real_pipeline.data,
            FilterSpec(delinquency="current"),
            real_pipeline.registry,
            bounds,
        )
        full = calculate_kpis(real_pipeline.data, real_pipeline.registry)
        subset = calculate_kpis(result.data, real_pipeline.registry)
        assert subset.raw("observed_default_rate") > full.raw("observed_default_rate")

    def test_band_counts_are_stable(self, real_pipeline) -> None:
        counts = assign_utilisation_band(real_pipeline.data).value_counts()
        assert int(counts.sum()) == 30_000
        # The very-high band is the verified high-utilisation population.
        very_high = [c for c in counts.index if str(c).startswith("Very high")]
        assert int(counts[very_high[0]]) == 7_980
