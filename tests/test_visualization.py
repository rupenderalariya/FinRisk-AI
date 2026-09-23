"""Tests for the Plotly chart builders, focusing on empty states and labelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src import visualization as viz
from src.analytics import (
    category_outcome_rates,
    correlation_matrix,
    distribution_bins,
    frequency_table,
)
from src.trends import balance_trend, delinquency_trend


class TestTheme:
    """Shared styling."""

    def test_applies_titles_and_labels(self) -> None:
        figure = viz.apply_theme(go.Figure(), "My title", "X label", "Y label")
        assert figure.layout.title.text == "My title"
        assert figure.layout.xaxis.title.text == "X label"
        assert figure.layout.yaxis.title.text == "Y label"

    def test_subtitle_is_embedded(self) -> None:
        figure = viz.apply_theme(go.Figure(), "Title", subtitle="A caveat")
        assert "A caveat" in figure.layout.title.text

    def test_palette_has_semantic_risk_colours(self) -> None:
        for key in ("risk_low", "risk_medium", "risk_high"):
            assert key in viz.COLOURS
        assert viz.RISK_COLOURS["High"] == viz.COLOURS["risk_high"]

    def test_categorical_sequence_is_distinct(self) -> None:
        assert len(set(viz.CATEGORICAL_SEQUENCE)) == len(viz.CATEGORICAL_SEQUENCE)


class TestEmptyFigure:
    """The placeholder used whenever there is nothing to plot."""

    def test_carries_the_message(self) -> None:
        figure = viz.empty_figure("Nothing to show here.")
        assert len(figure.layout.annotations) == 1
        assert figure.layout.annotations[0].text == "Nothing to show here."

    def test_hides_the_axes(self) -> None:
        figure = viz.empty_figure("msg")
        assert figure.layout.xaxis.visible is False
        assert figure.layout.yaxis.visible is False

    def test_has_no_data_traces(self) -> None:
        assert len(viz.empty_figure("msg").data) == 0


class TestDistributionChart:
    """Binned distributions."""

    def test_builds_from_bins(self, featured_df: pd.DataFrame) -> None:
        bins = distribution_bins(featured_df, "credit_limit", bins=10)
        figure = viz.distribution_chart(bins, "Title", "X")
        assert len(figure.data) == 1
        assert figure.data[0].type == "bar"

    def test_marks_mean_and_median(self, featured_df: pd.DataFrame) -> None:
        bins = distribution_bins(featured_df, "credit_limit", bins=10)
        figure = viz.distribution_chart(
            bins, "Title", "X", median_value=140_000, mean_value=167_484
        )
        assert len(figure.layout.shapes) == 2, "one vertical line for each"

    def test_explains_a_large_mean_median_gap(self, featured_df: pd.DataFrame) -> None:
        """The skew should be stated, not left for the reader to infer."""
        bins = distribution_bins(featured_df, "credit_limit", bins=10)
        figure = viz.distribution_chart(
            bins, "Title", "X", median_value=100.0, mean_value=200.0
        )
        assert "skew" in figure.layout.title.text.lower()

    def test_empty_input_gives_a_placeholder(self) -> None:
        figure = viz.distribution_chart(pd.DataFrame(), "Title", "X")
        assert len(figure.data) == 0
        assert len(figure.layout.annotations) == 1


class TestCategoryBarChart:
    """Per-category values."""

    def test_builds_from_a_frequency_table(self, featured_df: pd.DataFrame) -> None:
        frame = frequency_table(featured_df, "education").to_frame()
        figure = viz.category_bar_chart(frame, "category", "count", "Title")
        assert len(figure.data) == 1
        assert len(figure.data[0].x) == len(frame)

    def test_reference_line_is_drawn(self, featured_df: pd.DataFrame) -> None:
        frame = frequency_table(featured_df, "education").to_frame()
        figure = viz.category_bar_chart(
            frame, "category", "count", "Title", reference_line=100
        )
        assert len(figure.layout.shapes) == 1

    def test_horizontal_orientation(self, featured_df: pd.DataFrame) -> None:
        frame = frequency_table(featured_df, "education").to_frame()
        figure = viz.category_bar_chart(
            frame, "category", "count", "Title", horizontal=True
        )
        assert figure.data[0].orientation == "h"

    def test_colour_map_is_applied(self) -> None:
        frame = pd.DataFrame({"category": ["Low", "High"], "value": [1, 2]})
        figure = viz.category_bar_chart(
            frame, "category", "value", "Title", colour_map=viz.RISK_COLOURS
        )
        assert viz.COLOURS["risk_low"] in figure.data[0].marker.color

    def test_missing_columns_give_a_placeholder(self) -> None:
        figure = viz.category_bar_chart(pd.DataFrame({"a": [1]}), "nope", "also", "T")
        assert len(figure.layout.annotations) == 1


class TestTrendCharts:
    """Panel trend plots."""

    def test_line_chart_from_a_trend_result(self, featured_df: pd.DataFrame) -> None:
        frame = balance_trend(featured_df).to_frame()
        figure = viz.trend_line_chart(frame, "Title", "NT$", is_currency=True)
        assert len(figure.data) == 1
        assert len(figure.data[0].x) == 6

    def test_x_axis_is_chronological(self, featured_df: pd.DataFrame) -> None:
        frame = balance_trend(featured_df).to_frame()
        figure = viz.trend_line_chart(frame, "Title", "NT$")
        assert figure.data[0].x[0] == "Apr 2005"
        assert figure.data[0].x[-1] == "Sep 2005"

    def test_axis_label_states_direction_of_time(self, featured_df: pd.DataFrame) -> None:
        frame = balance_trend(featured_df).to_frame()
        figure = viz.trend_line_chart(frame, "Title", "NT$")
        assert "oldest" in figure.layout.xaxis.title.text.lower()

    def test_anomaly_markers_add_a_trace(self, featured_df: pd.DataFrame) -> None:
        frame = balance_trend(featured_df).to_frame()
        label = frame.iloc[2]["month_label"]
        figure = viz.trend_line_chart(frame, "T", "Y", anomaly_labels=[label])
        assert len(figure.data) == 2

    def test_multi_trend_overlays_series(self, featured_df: pd.DataFrame) -> None:
        figure = viz.multi_trend_chart(
            {
                "Delinquency": delinquency_trend(featured_df).to_frame(),
                "Balance": balance_trend(featured_df).to_frame(),
            },
            "Title",
            "Value",
        )
        assert len(figure.data) == 2

    def test_multi_trend_skips_empty_series(self, featured_df: pd.DataFrame) -> None:
        figure = viz.multi_trend_chart(
            {"Good": balance_trend(featured_df).to_frame(), "Bad": pd.DataFrame()},
            "Title",
            "Value",
        )
        assert len(figure.data) == 1

    def test_all_empty_gives_a_placeholder(self) -> None:
        figure = viz.multi_trend_chart({"a": pd.DataFrame()}, "T", "Y")
        assert len(figure.layout.annotations) == 1


class TestCorrelationHeatmap:
    """Correlation matrices."""

    def test_builds_a_heatmap(self, featured_df: pd.DataFrame) -> None:
        matrix = correlation_matrix(featured_df).matrix
        figure = viz.correlation_heatmap(matrix)
        assert figure.data[0].type == "heatmap"

    def test_scale_is_centred_on_zero(self, featured_df: pd.DataFrame) -> None:
        figure = viz.correlation_heatmap(correlation_matrix(featured_df).matrix)
        assert figure.data[0].zmid == 0
        assert figure.data[0].zmin == -1 and figure.data[0].zmax == 1

    def test_caveat_is_in_the_subtitle(self, featured_df: pd.DataFrame) -> None:
        figure = viz.correlation_heatmap(correlation_matrix(featured_df).matrix)
        assert "causation" in figure.layout.title.text.lower()

    def test_large_matrix_is_truncated_and_disclosed(self, featured_df: pd.DataFrame) -> None:
        matrix = correlation_matrix(featured_df).matrix
        figure = viz.correlation_heatmap(matrix, max_columns=8)
        assert len(figure.data[0].x) <= 8
        if matrix.shape[0] > 8:
            assert "most" in figure.layout.title.text.lower()

    def test_too_small_gives_a_placeholder(self) -> None:
        figure = viz.correlation_heatmap(pd.DataFrame({"a": [1.0]}))
        assert len(figure.layout.annotations) == 1


class TestScatterChart:
    """Scatter plots."""

    def test_builds_a_scatter(self, featured_df: pd.DataFrame) -> None:
        figure = viz.scatter_chart(featured_df, "credit_limit", "bill_amt_m1", "Title")
        assert len(figure.data) == 1

    def test_samples_large_frames_and_says_so(self, featured_df: pd.DataFrame) -> None:
        figure = viz.scatter_chart(
            featured_df, "credit_limit", "bill_amt_m1", "Title", max_points=50
        )
        assert len(figure.data[0].x) == 50
        assert "sample" in figure.layout.title.text.lower()

    def test_colour_column_splits_traces(self, featured_df: pd.DataFrame) -> None:
        figure = viz.scatter_chart(
            featured_df, "credit_limit", "bill_amt_m1", "Title",
            colour_column="limit_band",
        )
        assert len(figure.data) > 1

    def test_missing_column_gives_a_placeholder(self, featured_df: pd.DataFrame) -> None:
        figure = viz.scatter_chart(featured_df, "nope", "bill_amt_m1", "Title")
        assert len(figure.layout.annotations) == 1

    def test_infinities_are_dropped(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.copy()
        frame.loc[frame.index[0], "utilisation_latest"] = np.inf
        figure = viz.scatter_chart(frame, "utilisation_latest", "credit_limit", "T")
        assert np.isfinite(np.asarray(figure.data[0].x, dtype=float)).all()


class TestBoxComparison:
    """Box plots across categories."""

    def test_one_trace_per_category(self, featured_df: pd.DataFrame) -> None:
        figure = viz.box_comparison_chart(
            featured_df, "limit_band", "utilisation_latest", "Title"
        )
        assert len(figure.data) == featured_df["limit_band"].nunique()

    def test_explains_the_box(self, featured_df: pd.DataFrame) -> None:
        figure = viz.box_comparison_chart(
            featured_df, "limit_band", "utilisation_latest", "Title"
        )
        assert "interquartile" in figure.layout.title.text.lower()

    def test_caps_category_count(self, featured_df: pd.DataFrame) -> None:
        figure = viz.box_comparison_chart(
            featured_df, "limit_band", "utilisation_latest", "T", max_categories=2
        )
        assert len(figure.data) <= 2


class TestCompositionCharts:
    """Donut and stacked bar."""

    def test_donut_builds(self) -> None:
        figure = viz.donut_chart(["A", "B"], [60, 40], "Title", centre_label="100")
        assert figure.data[0].type == "pie"
        assert figure.data[0].hole == 0.58

    def test_donut_centre_label(self) -> None:
        figure = viz.donut_chart(["A"], [1], "Title", centre_label="Total: 1")
        assert any(a.text == "Total: 1" for a in figure.layout.annotations)

    def test_donut_with_zero_total_gives_a_placeholder(self) -> None:
        figure = viz.donut_chart(["A", "B"], [0, 0], "Title")
        assert len(figure.data) == 0

    def test_stacked_bar_normalises_to_percent(self, featured_df: pd.DataFrame) -> None:
        table = pd.crosstab(featured_df["limit_band"], featured_df["education"])
        figure = viz.stacked_bar_chart(table, "Title", normalise=True)
        assert figure.layout.barmode == "stack"
        totals = np.nansum([np.asarray(t.y, dtype=float) for t in figure.data], axis=0)
        assert np.allclose(totals[~np.isnan(totals)], 100.0, atol=0.01)

    def test_stacked_bar_raw_counts(self, featured_df: pd.DataFrame) -> None:
        table = pd.crosstab(featured_df["limit_band"], featured_df["education"])
        figure = viz.stacked_bar_chart(table, "Title", normalise=False)
        total = np.nansum([np.asarray(t.y, dtype=float) for t in figure.data])
        assert total == pytest.approx(len(featured_df))

    def test_empty_gives_a_placeholder(self) -> None:
        assert len(viz.stacked_bar_chart(pd.DataFrame(), "T").layout.annotations) == 1


class TestImportanceChart:
    """Driver importance bars."""

    def test_builds_horizontal_bars(self) -> None:
        figure = viz.importance_chart(["a", "b", "c"], [0.1, 0.5, 0.3])
        assert figure.data[0].orientation == "h"

    def test_sorted_with_largest_on_top(self) -> None:
        figure = viz.importance_chart(["small", "big"], [0.1, 0.9])
        assert figure.data[0].y[-1] == "big"

    def test_default_subtitle_carries_the_caveat(self) -> None:
        """The disclaimer must travel with the chart."""
        figure = viz.importance_chart(["a"], [0.5])
        assert "causation" in figure.layout.title.text.lower()

    def test_empty_gives_a_placeholder(self) -> None:
        assert len(viz.importance_chart([], []).layout.annotations) == 1


class TestOutcomeRateChart:
    """Outcome rates per group."""

    def test_builds_from_a_category_outcome(self, featured_df: pd.DataFrame) -> None:
        result = category_outcome_rates(featured_df, "limit_band", "default_next_month")
        figure = viz.outcome_rate_chart(result.table, "limit_band")
        assert len(figure.data) == 1

    def test_overall_reference_line(self, featured_df: pd.DataFrame) -> None:
        result = category_outcome_rates(featured_df, "limit_band", "default_next_month")
        figure = viz.outcome_rate_chart(
            result.table, "limit_band", overall_rate=result.overall_rate
        )
        assert len(figure.layout.shapes) == 1

    def test_undersized_groups_greyed_and_disclosed(self) -> None:
        table = pd.DataFrame(
            {"count": [100, 5], "events": [20, 4], "rate_pct": [20.0, 80.0]},
            index=pd.Index(["big", "tiny"], name="grp"),
        )
        figure = viz.outcome_rate_chart(table, "grp", min_group_size=30)
        assert viz.COLOURS["neutral_light"] in figure.data[0].marker.color
        assert "grey" in figure.layout.title.text.lower()

    def test_empty_gives_a_placeholder(self) -> None:
        assert len(viz.outcome_rate_chart(pd.DataFrame(), "grp").layout.annotations) == 1


class TestQualityCharts:
    """Data-quality visuals."""

    def test_gauge_builds(self) -> None:
        figure = viz.quality_score_gauge(98.18)
        assert figure.data[0].value == pytest.approx(98.18)

    def test_gauge_discloses_it_is_project_defined(self) -> None:
        figure = viz.quality_score_gauge(98.18)
        assert "project-defined" in figure.layout.title.text.lower()

    def test_gauge_colour_reflects_band(self) -> None:
        assert viz.quality_score_gauge(95).data[0].gauge.bar.color == viz.COLOURS["risk_low"]
        assert viz.quality_score_gauge(40).data[0].gauge.bar.color == viz.COLOURS["risk_high"]

    def test_gauge_handles_nan(self) -> None:
        assert len(viz.quality_score_gauge(float("nan")).layout.annotations) == 1

    def test_missing_chart_positive_empty_state(self) -> None:
        """No missing values is good news and should read that way."""
        figure = viz.missing_values_chart({"a": 0, "b": 0}, 100)
        assert len(figure.data) == 0
        assert "fully populated" in figure.layout.annotations[0].text.lower()

    def test_missing_chart_plots_when_present(self) -> None:
        figure = viz.missing_values_chart({"a": 10, "b": 5, "c": 0}, 100)
        assert len(figure.data) == 1
        assert len(figure.data[0].y) == 2, "only columns with missing values"


class TestUniversalRequirements:
    """Properties every chart must satisfy."""

    @pytest.fixture
    def all_figures(self, featured_df: pd.DataFrame) -> dict[str, go.Figure]:
        bins = distribution_bins(featured_df, "credit_limit", bins=10)
        freq = frequency_table(featured_df, "education").to_frame()
        trend = balance_trend(featured_df).to_frame()
        matrix = correlation_matrix(featured_df).matrix
        outcome = category_outcome_rates(
            featured_df, "limit_band", "default_next_month"
        ).table
        return {
            "distribution": viz.distribution_chart(bins, "T", "X", is_currency=True),
            "category_bar": viz.category_bar_chart(freq, "category", "count", "T", "X", "Y"),
            "trend_line": viz.trend_line_chart(trend, "T", "NT$", is_currency=True),
            "heatmap": viz.correlation_heatmap(matrix),
            "scatter": viz.scatter_chart(featured_df, "credit_limit", "age", "T", "X", "Y"),
            "box": viz.box_comparison_chart(featured_df, "limit_band", "age", "T"),
            "donut": viz.donut_chart(["A", "B"], [1, 2], "T"),
            "importance": viz.importance_chart(["a", "b"], [0.2, 0.4]),
            "outcome_rate": viz.outcome_rate_chart(outcome, "limit_band"),
            "gauge": viz.quality_score_gauge(90.0),
        }

    def test_every_chart_has_a_title(self, all_figures) -> None:
        for name, figure in all_figures.items():
            assert figure.layout.title.text, f"{name} has no title"

    def test_every_chart_has_a_height(self, all_figures) -> None:
        for name, figure in all_figures.items():
            assert figure.layout.height, f"{name} has no height"

    def test_data_charts_define_hover_templates(self, all_figures) -> None:
        """Tooltips are a stated UI requirement."""
        for name, figure in all_figures.items():
            if name == "gauge":
                continue  # an indicator has no hover surface
            for trace in figure.data:
                if hasattr(trace, "hovertemplate"):
                    assert trace.hovertemplate, f"{name} trace lacks a hover template"

    def test_every_chart_is_json_serialisable(self, all_figures) -> None:
        for name, figure in all_figures.items():
            assert figure.to_json(), f"{name} failed to serialise"

    def test_no_chart_raises_on_an_empty_frame(self, featured_df: pd.DataFrame) -> None:
        empty = featured_df.iloc[0:0]
        builders = (
            lambda: viz.distribution_chart(pd.DataFrame(), "T", "X"),
            lambda: viz.category_bar_chart(pd.DataFrame(), "a", "b", "T"),
            lambda: viz.trend_line_chart(pd.DataFrame(), "T", "Y"),
            lambda: viz.correlation_heatmap(pd.DataFrame()),
            lambda: viz.scatter_chart(empty, "credit_limit", "age", "T"),
            lambda: viz.box_comparison_chart(empty, "limit_band", "age", "T"),
            lambda: viz.donut_chart([], [], "T"),
            lambda: viz.importance_chart([], []),
            lambda: viz.outcome_rate_chart(pd.DataFrame(), "g"),
            lambda: viz.stacked_bar_chart(pd.DataFrame(), "T"),
            lambda: viz.missing_values_chart({}, 0),
        )
        for index, builder in enumerate(builders):
            figure = builder()
            assert isinstance(figure, go.Figure), f"builder {index} returned a non-figure"
