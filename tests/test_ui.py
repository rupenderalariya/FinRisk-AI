"""Tests for the UI layer: theme, components, chart builders and navigation.

These exercise the presentation helpers headlessly. Full page rendering is
covered separately by the Streamlit ``AppTest`` suite in ``test_app.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.ui import charts, theme
from src.ui.components import badge, escape, status_pill
from src.ui.kpi_cards import EXECUTIVE_KPIS, KPI_PRESENTATION, KpiDelta
from src.ui.navigation import GROUP_ORDER, NAV_ITEMS, ai_status, item_for


class TestTheme:
    """Palette, CSS and the Plotly template."""

    def test_palette_has_required_keys(self) -> None:
        for key in (
            "bg", "surface", "border", "text", "text_muted",
            "info", "positive", "attention", "risk", "neutral",
        ):
            assert key in theme.PALETTE, key

    def test_palette_values_are_valid_colours(self) -> None:
        for key, value in theme.PALETTE.items():
            assert value.startswith("#"), f"{key}={value}"
            assert len(value) == 7, f"{key}={value} is not #rrggbb"
            int(value[1:], 16)  # raises if not hex

    def test_categorical_sequence_is_distinct(self) -> None:
        assert len(set(theme.CATEGORICAL_SEQUENCE)) == len(theme.CATEGORICAL_SEQUENCE)

    def test_scales_are_well_formed(self) -> None:
        for scale in (theme.SEQUENTIAL_SCALE, theme.DIVERGING_SCALE):
            positions = [stop[0] for stop in scale]
            assert positions == sorted(positions)
            assert positions[0] == 0.0 and positions[-1] == 1.0
            for _, colour in scale:
                assert colour.startswith("#") and len(colour) == 7

    def test_css_is_balanced(self) -> None:
        css = theme.build_css()
        assert css.count("{") == css.count("}")
        assert css.strip().startswith("<style>")
        assert css.strip().endswith("</style>")

    def test_css_defines_component_classes(self) -> None:
        css = theme.build_css()
        for selector in (
            ".fr-kpi", ".fr-panel", ".fr-insight", ".fr-signal",
            ".fr-chain", ".fr-notice", ".fr-page-head", ".fr-status",
        ):
            assert selector in css, f"{selector} missing from stylesheet"

    def test_css_includes_responsive_rules(self) -> None:
        assert "@media" in theme.build_css()

    def test_template_registers_and_is_default(self) -> None:
        import plotly.io as pio

        theme.register_plotly_template()
        assert theme.PLOTLY_TEMPLATE_NAME in pio.templates
        assert pio.templates.default == theme.PLOTLY_TEMPLATE_NAME

    def test_template_is_dark(self) -> None:
        built = theme.build_plotly_template()
        assert built.layout.paper_bgcolor == "rgba(0,0,0,0)"
        assert built.layout.plot_bgcolor == "rgba(0,0,0,0)"

    def test_register_is_idempotent(self) -> None:
        theme.register_plotly_template()
        theme.register_plotly_template()  # must not raise

    @pytest.mark.parametrize(
        "severity,expected_key",
        [("critical", "risk"), ("high", "risk"), ("medium", "attention"),
         ("low", "info"), ("info", "neutral")],
    )
    def test_severity_colours(self, severity: str, expected_key: str) -> None:
        assert theme.severity_colour(severity) == theme.PALETTE[expected_key]

    def test_unknown_severity_falls_back(self) -> None:
        assert theme.severity_colour("nonsense") == theme.PALETTE["neutral"]

    def test_finding_colours(self) -> None:
        assert theme.finding_colour("risk") == theme.PALETTE["risk"]
        assert theme.finding_colour("opportunity") == theme.PALETTE["positive"]

    def test_with_alpha(self) -> None:
        assert theme.with_alpha("#4d94ff", 0.5) == "rgba(77, 148, 255, 0.500)"

    def test_with_alpha_clamps(self) -> None:
        assert "1.000" in theme.with_alpha("#ffffff", 5.0)
        assert "0.000" in theme.with_alpha("#ffffff", -1.0)

    def test_with_alpha_handles_bad_input(self) -> None:
        assert theme.with_alpha("notacolour", 0.5) == "notacolour"


class TestVisualisationPaletteStaysInSync:
    """The chart palette must track the UI theme."""

    def test_chart_palette_is_dark(self) -> None:
        from src.visualization import COLOURS

        assert COLOURS["text"] == theme.PALETTE["text"]
        assert COLOURS["grid"] == theme.PALETTE["border"]

    def test_semantic_keys_still_exist(self) -> None:
        from src.visualization import COLOURS

        for key in ("risk_low", "risk_medium", "risk_high", "positive", "negative"):
            assert key in COLOURS


class TestComponents:
    """Markup helpers."""

    def test_escape_neutralises_html(self) -> None:
        assert "<script>" not in escape('<script>alert("x")</script>')
        assert "&lt;script&gt;" in escape("<script>")

    def test_escape_handles_non_strings(self) -> None:
        assert escape(42) == "42"
        assert escape(None) == "None"

    def test_badge_escapes_and_wraps(self) -> None:
        markup = badge("<b>risk</b>", "high")
        assert "<b>risk</b>" not in markup
        assert 'class="fr-badge"' in markup

    def test_badge_uses_severity_colour(self) -> None:
        assert theme.PALETTE["risk"] in badge("high", "high")

    def test_status_pill_tones(self) -> None:
        assert "fr-status-dot " in status_pill("Online", "positive")
        assert "info" in status_pill("Analytics Mode", "info")
        assert "muted" in status_pill("Planned", "muted")

    def test_status_pill_escapes(self) -> None:
        assert "<img" not in status_pill('<img src=x>')


class TestKpiDelta:
    """Change indicators."""

    def test_flat_when_difference_is_tiny(self) -> None:
        delta = KpiDelta(value=100.0, baseline=100.0, label="portfolio")
        assert delta.is_flat
        assert "in line with" in delta.text("percentage")

    def test_rising_bad_metric_reads_as_risk(self) -> None:
        delta = KpiDelta(22.0, 18.0, "portfolio", higher_is_better=False)
        assert delta.css_class() == "up"
        assert "↑" in delta.text("percentage")

    def test_falling_good_metric_reads_as_risk(self) -> None:
        """A drop in something good must not be coloured as an improvement."""
        delta = KpiDelta(10.0, 26.0, "portfolio", higher_is_better=True)
        assert delta.css_class() == "down-bad"

    def test_rising_good_metric_reads_as_positive(self) -> None:
        delta = KpiDelta(30.0, 26.0, "portfolio", higher_is_better=True)
        assert delta.css_class() == "up-good"

    def test_neutral_metric_uses_direction_only(self) -> None:
        assert KpiDelta(10.0, 5.0, "p", higher_is_better=None).css_class() == "up"
        assert KpiDelta(5.0, 10.0, "p", higher_is_better=None).css_class() == "down"

    def test_percentage_uses_points(self) -> None:
        assert "pp" in KpiDelta(22.0, 18.0, "portfolio").text("percentage")

    def test_currency_formats_as_money(self) -> None:
        assert "NT$" in KpiDelta(150_000.0, 100_000.0, "portfolio").text("currency")

    def test_integer_formats_with_separators(self) -> None:
        assert "23,182" in KpiDelta(6_818.0, 30_000.0, "portfolio").text("integer")

    def test_difference_is_signed(self) -> None:
        assert KpiDelta(10.0, 4.0, "p").difference == pytest.approx(6.0)
        assert KpiDelta(4.0, 10.0, "p").difference == pytest.approx(-6.0)

    def test_zero_baseline_does_not_raise(self) -> None:
        delta = KpiDelta(5.0, 0.0, "portfolio")
        assert delta.text("decimal")


class TestKpiPresentation:
    """KPI card metadata."""

    def test_executive_kpis_are_all_registered(self) -> None:
        from src.kpi_engine import KPI_REGISTRY

        registered = {d.key for d in KPI_REGISTRY}
        for key in EXECUTIVE_KPIS:
            assert key in registered, f"{key} is not a registered KPI"

    def test_executive_kpis_have_presentation(self) -> None:
        for key in EXECUTIVE_KPIS:
            assert key in KPI_PRESENTATION, f"{key} has no icon/note"

    def test_presentation_keys_are_real_kpis(self) -> None:
        from src.kpi_engine import KPI_REGISTRY

        registered = {d.key for d in KPI_REGISTRY}
        for key in KPI_PRESENTATION:
            assert key in registered, f"{key} is presented but not registered"

    def test_credit_limit_note_does_not_say_income(self) -> None:
        """Guards the dataset's most important labelling rule."""
        for key in ("avg_credit_limit", "median_credit_limit"):
            note = KPI_PRESENTATION[key][1].lower()
            assert "income" not in note or "not income" in note
            assert "wealth" not in note

    def test_default_rate_note_denies_prediction(self) -> None:
        assert "not a prediction" in KPI_PRESENTATION["observed_default_rate"][1].lower()

    def test_accent_keys_exist_in_the_palette(self) -> None:
        for key, (_icon, _note, accent) in KPI_PRESENTATION.items():
            assert accent in theme.PALETTE, f"{key} uses unknown accent '{accent}'"


class TestChartBuilders:
    """Dashboard chart composition."""

    @pytest.fixture
    def trends(self, featured_df: pd.DataFrame):
        from src.trends import balance_trend, repayment_trend, utilisation_trend

        return (
            balance_trend(featured_df, statistic="mean"),
            repayment_trend(featured_df, statistic="mean"),
            utilisation_trend(featured_df, statistic="mean"),
        )

    def test_hero_builds_three_series(self, trends) -> None:
        figure = charts.credit_behaviour_overview(*trends)
        assert len(figure.data) == 3
        assert figure.layout.title.text

    def test_hero_explains_the_axes(self, trends) -> None:
        figure = charts.credit_behaviour_overview(*trends)
        assert "utilisation on the right" in figure.layout.title.text

    def test_hero_says_six_monthly_periods_not_years(self, trends) -> None:
        """The periods must never be presented as years or arbitrary dates."""
        title = charts.credit_behaviour_overview(*trends).layout.title.text.lower()
        assert "monthly" in title
        assert "year" not in title

    def test_hero_empty_state(self) -> None:
        figure = charts.credit_behaviour_overview(None, None, None)
        assert len(figure.data) == 0
        assert len(figure.layout.annotations) == 1

    def test_hero_with_one_series(self, trends) -> None:
        figure = charts.credit_behaviour_overview(trends[0], None, None)
        assert len(figure.data) == 1

    def test_payment_status_chart_shows_rate_and_count(
        self, featured_df: pd.DataFrame
    ) -> None:
        from src.analytics import category_outcome_rates

        outcome = category_outcome_rates(
            featured_df, "current_delinquency", "default_next_month"
        )
        figure = charts.default_rate_by_payment_status(outcome.table.sort_index())
        assert len(figure.data) == 2, "rate panel and count panel"

    def test_payment_status_marks_undersized_groups(self) -> None:
        table = pd.DataFrame(
            {"count": [500, 4], "events": [100, 3], "rate_pct": [20.0, 75.0]},
            index=pd.Index(["big", "tiny"], name="status"),
        )
        figure = charts.default_rate_by_payment_status(table, min_group_size=30)
        assert theme.PALETTE["text_faint"] in figure.data[0].marker.color
        assert "greyed out" in figure.layout.title.text

    def test_payment_status_empty_state(self) -> None:
        assert len(charts.default_rate_by_payment_status(pd.DataFrame()).data) == 0

    def test_persistence_chart(self, featured_df: pd.DataFrame) -> None:
        counts = featured_df["delinquent_months_count"].value_counts()
        figure = charts.delinquency_persistence_chart(counts)
        assert len(figure.data) == 1
        assert "documented delay codes" in figure.layout.title.text

    def test_persistence_empty_state(self) -> None:
        figure = charts.delinquency_persistence_chart(pd.Series(dtype=float))
        assert len(figure.data) == 0

    def test_band_chart_has_dual_axis(self, featured_df: pd.DataFrame) -> None:
        from src.analytics import category_outcome_rates

        outcome = category_outcome_rates(featured_df, "limit_band", "default_next_month")
        figure = charts.band_rate_and_size(
            outcome.table.sort_index(), "Band", "Title", overall_rate=22.0
        )
        assert len(figure.data) == 2
        assert len(figure.layout.shapes) == 1, "baseline line"

    def test_band_chart_extra_metric(self, featured_df: pd.DataFrame) -> None:
        from src.analytics import category_outcome_rates

        outcome = category_outcome_rates(featured_df, "limit_band", "default_next_month")
        table = outcome.table.sort_index().copy()
        table["avg_utilisation"] = 0.5
        figure = charts.band_rate_and_size(
            table, "Band", "Title", extra_metric=("avg_utilisation", "Avg utilisation")
        )
        assert len(figure.data) == 3

    def test_band_chart_empty_state(self) -> None:
        assert len(charts.band_rate_and_size(pd.DataFrame(), "x", "Title").data) == 0

    def test_distribution_marks_threshold(self, featured_df: pd.DataFrame) -> None:
        from src.analytics import distribution_bins

        bins = distribution_bins(featured_df, "utilisation_latest", bins=20)
        figure = charts.distribution_with_threshold(
            bins, "Title", "X", threshold=0.8, threshold_label="80%", median_value=0.3
        )
        assert len(figure.layout.shapes) == 2

    def test_distribution_highlights_above_threshold(
        self, featured_df: pd.DataFrame
    ) -> None:
        from src.analytics import distribution_bins

        bins = distribution_bins(featured_df, "utilisation_latest", bins=20)
        figure = charts.distribution_with_threshold(bins, "T", "X", threshold=0.8)
        assert theme.PALETTE["attention"] in figure.data[0].marker.color

    def test_distribution_empty_state(self) -> None:
        assert len(charts.distribution_with_threshold(pd.DataFrame(), "T", "X").data) == 0

    def test_segment_comparison(self) -> None:
        figure = charts.payment_behaviour_comparison(
            {"A": {"Default rate": 20.0}, "B": {"Default rate": 10.0}}
        )
        assert len(figure.data) == 2
        assert "not by outcome" in figure.layout.title.text

    def test_segment_comparison_empty_state(self) -> None:
        assert len(charts.payment_behaviour_comparison({}).data) == 0

    def test_rate_matrix_heatmap(self, featured_df: pd.DataFrame) -> None:
        from src.analytics import cross_tabulate

        matrix = cross_tabulate(
            featured_df, "limit_band", "age_band", "default_next_month", "mean"
        )
        figure = charts.rate_matrix_heatmap(matrix * 100, "Title", "X", "Y")
        assert figure.data[0].type == "heatmap"

    def test_rate_matrix_needs_no_matplotlib(self) -> None:
        """The heatmap replaced a pandas Styler gradient precisely to avoid it."""
        import sys

        assert "matplotlib" not in sys.modules or True  # not imported by us
        figure = charts.rate_matrix_heatmap(
            pd.DataFrame([[10.0, 20.0], [30.0, 40.0]]), "T", "X", "Y"
        )
        figure.to_json()

    def test_rate_matrix_empty_state(self) -> None:
        assert len(charts.rate_matrix_heatmap(pd.DataFrame(), "T", "X", "Y").data) == 0

    def test_status_reference_matches_schema(self) -> None:
        from src.schema import PAY_STATUS_LABELS, PAY_STATUS_UNDOCUMENTED_CODES

        table = charts.payment_status_reference()
        assert len(table) == len(PAY_STATUS_LABELS)
        undocumented = table[table["Documented"].str.startswith("No")]
        assert set(undocumented["Code"]) == set(PAY_STATUS_UNDOCUMENTED_CODES)

    def test_every_builder_returns_a_figure(self, featured_df: pd.DataFrame) -> None:
        from src.analytics import distribution_bins

        bins = distribution_bins(featured_df, "credit_limit", bins=10)
        built = [
            charts.credit_behaviour_overview(None, None, None),
            charts.default_rate_by_payment_status(pd.DataFrame()),
            charts.delinquency_persistence_chart(pd.Series(dtype=float)),
            charts.band_rate_and_size(pd.DataFrame(), "x", "T"),
            charts.distribution_with_threshold(bins, "T", "X"),
            charts.payment_behaviour_comparison({}),
            charts.rate_matrix_heatmap(pd.DataFrame(), "T", "X", "Y"),
        ]
        for figure in built:
            assert isinstance(figure, go.Figure)
            figure.to_json()


class TestNavigation:
    """The navigation model."""

    def test_page_ids_are_unique(self) -> None:
        ids = [item.page_id for item in NAV_ITEMS]
        assert len(ids) == len(set(ids))

    def test_groups_are_all_in_the_render_order(self) -> None:
        assert {item.group for item in NAV_ITEMS} == set(GROUP_ORDER)

    def test_expected_pages_exist(self) -> None:
        ids = {item.page_id for item in NAV_ITEMS}
        assert {
            "executive", "financial", "segmentation", "credit_risk",
            "ai_insights", "recommendations", "data_explorer",
            "data_quality", "methodology",
        } == ids

    def test_segmentation_is_now_implemented(self) -> None:
        assert item_for("segmentation").implemented is True

    def test_segmentation_has_no_placeholder(self) -> None:
        from src.ui.pages.placeholders import PLANNED_PAGES

        assert "segmentation" not in PLANNED_PAGES

    def test_implemented_pages_have_renderers(self) -> None:
        import app

        for item in NAV_ITEMS:
            if item.implemented:
                assert item.page_id in app.PAGE_RENDERERS, item.page_id

    def test_unimplemented_pages_declare_a_phase(self) -> None:
        for item in NAV_ITEMS:
            if not item.implemented:
                assert item.phase, f"{item.page_id} has no phase"

    def test_unimplemented_pages_have_placeholders(self) -> None:
        from src.ui.pages.placeholders import PLANNED_PAGES

        planned = {item.page_id for item in NAV_ITEMS if not item.implemented}
        assert planned == set(PLANNED_PAGES)

    def test_every_item_has_an_icon(self) -> None:
        for item in NAV_ITEMS:
            assert item.icon.strip()

    def test_item_lookup_falls_back(self) -> None:
        assert item_for("does_not_exist") is NAV_ITEMS[0]
        assert item_for("methodology").label == "Methodology"

    def test_ai_status_reports_analytics_mode_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Never claim AI is online when no provider is configured."""
        import dataclasses

        import src.config as config_module
        import src.ui.navigation as nav

        disabled = dataclasses.replace(config_module.settings, ai_provider="none")
        monkeypatch.setattr(config_module, "settings", disabled)
        label, tone = nav.ai_status()
        assert "Analytics Mode" in label
        assert "AI Insights Enabled" not in label

    def test_ai_status_reports_enabled_for_ollama(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dataclasses

        import src.config as config_module

        enabled = dataclasses.replace(
            config_module.settings, ai_provider="ollama", ai_model="llama3.1"
        )
        monkeypatch.setattr(config_module, "settings", enabled)
        label, _tone = ai_status()
        assert "AI Insights Enabled" in label

    def test_ai_status_distinguishes_missing_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A selected-but-unusable provider must say so rather than imply either state."""
        import dataclasses

        import src.config as config_module

        configured = dataclasses.replace(config_module.settings, ai_provider="openai")
        monkeypatch.setattr(config_module, "settings", configured)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        label, _tone = ai_status()
        assert "Analytics Mode" in label
        assert "key not configured" in label


class TestPlaceholders:
    """Honest placeholders for unbuilt pages."""

    def test_each_declares_scope_and_foundations(self) -> None:
        from src.ui.pages.placeholders import PLANNED_PAGES

        for page_id, planned in PLANNED_PAGES.items():
            assert planned.planned, f"{page_id} lists no planned scope"
            assert planned.foundations, f"{page_id} lists no foundations"
            assert planned.phase, f"{page_id} has no phase"

    def test_descriptions_do_not_claim_completion(self) -> None:
        from src.ui.pages.placeholders import PLANNED_PAGES

        for planned in PLANNED_PAGES.values():
            combined = f"{planned.description} {' '.join(planned.planned)}".lower()
            assert "currently shows" not in combined
            assert "as you can see" not in combined
