"""
End-to-end dashboard tests using Streamlit's AppTest harness.

These render the real application against the real dataset, which is the only way
to catch integration failures that unit tests miss - a missing optional
dependency inside a Streamlit call, for example, or a filter that fails to
propagate. Skipped automatically when the dataset is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

#: Absolute path to the app entry point. AppTest resolves a relative path against
#: the current working directory, which is not reliable under pytest.
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

PAGE_IDS = (
    "executive",
    "financial",
    "segmentation",
    "data_explorer",
    "data_quality",
    "methodology",
    "credit_risk",
    "ai_insights",
    "recommendations",
)

IMPLEMENTED_PAGES = (
    "executive",
    "financial",
    "segmentation",
    "data_explorer",
    "data_quality",
    "methodology",
)


@pytest.fixture(scope="module")
def app_factory(real_dataset_path):
    """Return a factory producing a fresh AppTest instance.

    Depends on ``real_dataset_path`` so the whole module skips without data.
    """
    from streamlit.testing.v1 import AppTest

    def factory(page_id: str | None = None, **session_state):
        app = AppTest.from_file(APP_PATH, default_timeout=300)
        if page_id is not None:
            app.session_state["fr_active_page"] = page_id
        for key, value in session_state.items():
            app.session_state[key] = value
        return app

    return factory


def _body_text(app) -> str:
    """Concatenate everything the page rendered as text.

    Includes dataframe contents as well as markdown and captions, since much of
    the reference material (the provenance table, KPI definitions) is rendered as
    a table rather than prose.
    """
    parts: list[str] = [m.value for m in app.markdown]
    parts += [c.value for c in app.caption]
    for element in app.dataframe:
        try:
            parts.append(element.value.to_string())
        except Exception:  # noqa: BLE001 - a styler or odd payload is not fatal here
            parts.append(str(element.value))
    return " ".join(parts)


class TestAllPagesRender:
    """Every page must render without raising."""

    @pytest.mark.parametrize("page_id", PAGE_IDS)
    def test_page_renders_without_exception(self, app_factory, page_id: str) -> None:
        app = app_factory(page_id).run()
        assert not app.exception, [str(e.value) for e in app.exception]

    @pytest.mark.parametrize("page_id", IMPLEMENTED_PAGES)
    def test_implemented_page_has_no_failure_notice(
        self, app_factory, page_id: str
    ) -> None:
        """Catches a section that silently degraded to an error panel."""
        app = app_factory(page_id).run()
        body = _body_text(app)
        assert "unavailable</div>" not in body or "not implemented" in body, (
            f"{page_id} rendered a failure notice"
        )

    @pytest.mark.parametrize("page_id", IMPLEMENTED_PAGES)
    def test_implemented_page_produces_content(self, app_factory, page_id: str) -> None:
        app = app_factory(page_id).run()
        assert len(app.markdown) > 15, f"{page_id} rendered very little"


class TestExecutiveOverview:
    """The primary page."""

    def test_shows_the_headline_kpis(self, app_factory) -> None:
        app = app_factory("executive").run()
        body = _body_text(app)
        for label in ("Total Clients", "Observed Default Rate", "Average Credit Utilisation"):
            assert label in body, f"missing KPI '{label}'"

    def test_shows_the_verified_default_rate(self, app_factory) -> None:
        app = app_factory("executive").run()
        assert "22.12%" in _body_text(app)

    def test_shows_the_customer_count(self, app_factory) -> None:
        app = app_factory("executive").run()
        assert "30,000" in _body_text(app)

    def test_shows_the_narrative_sections(self, app_factory) -> None:
        app = app_factory("executive").run()
        body = _body_text(app)
        for heading in (
            "Portfolio Snapshot",
            "What changed across the observed periods?",
            "Key Intelligence",
            "Risk Signals",
            "Opportunities",
            "Recommended Next Steps",
        ):
            assert heading in body, f"missing section '{heading}'"

    def test_renders_the_action_chain(self, app_factory) -> None:
        app = app_factory("executive").run()
        body = _body_text(app)
        for tag in ("FACT", "INSIGHT", "IMPLICATION", "ACTION"):
            assert tag in body

    def test_labels_signals_as_observed_not_predicted(self, app_factory) -> None:
        app = app_factory("executive").run()
        body = _body_text(app)
        assert "Observed Analytical Signal" in body
        assert "not model predictions" in body

    def test_states_why_there_is_no_change_indicator(self, app_factory) -> None:
        """Unfiltered, there is no prior period, and the page must say so."""
        app = app_factory("executive").run()
        assert "no prior period" in _body_text(app)

    def test_declares_no_model_is_active(self, app_factory) -> None:
        app = app_factory("executive").run()
        assert "no predictive model is active" in _body_text(app).lower()


class TestFinancialAnalytics:
    """The analytical workspace."""

    def test_has_four_tabs(self, app_factory) -> None:
        app = app_factory("financial").run()
        assert len(app.tabs) >= 4

    def test_exposes_the_filter_controls(self, app_factory) -> None:
        app = app_factory("financial").run()
        assert len(app.slider) >= 3
        assert len(app.multiselect) >= 2
        assert len(app.selectbox) >= 1

    def test_documents_the_utilisation_threshold(self, app_factory) -> None:
        app = app_factory("financial").run()
        assert "HIGH_UTILISATION_THRESHOLD" in _body_text(app)

    def test_denies_the_income_interpretation(self, app_factory) -> None:
        app = app_factory("financial").run()
        assert "not income" in _body_text(app).lower()

    def test_includes_the_causation_caveat(self, app_factory) -> None:
        app = app_factory("financial").run()
        body = _body_text(app).lower()
        assert "not causal" in body or "not a causal" in body

    def test_explains_zero_payment_is_not_default(self, app_factory) -> None:
        app = app_factory("financial").run()
        assert "does not by itself mean default" in _body_text(app)

    def test_renders_dataframes(self, app_factory) -> None:
        app = app_factory("financial").run()
        assert len(app.dataframe) >= 2


class TestFilterBehaviour:
    """Filters must move the whole page."""

    def test_default_state_shows_everything(self, app_factory) -> None:
        app = app_factory("financial").run()
        assert "30,000" in _body_text(app)

    def test_narrowing_a_slider_changes_the_count(self, app_factory) -> None:
        app = app_factory("financial").run()
        before = _body_text(app)
        app.slider[0].set_range(30, 35).run()
        assert not app.exception
        after = _body_text(app)
        assert before != after
        assert "of 30,000 customers selected" in after

    def test_delinquency_filter_changes_the_default_rate(self, app_factory) -> None:
        """Proves the filter reaches the KPI layer, not only the row count."""
        app = app_factory("financial").run()
        assert "22.12%" in _body_text(app)

        app.selectbox[0].select("current").run()
        assert not app.exception
        body = _body_text(app)
        assert "22.12%" not in body, "the default rate must change with the selection"
        assert "6,818" in body

    def test_empty_selection_shows_the_empty_state(self, app_factory) -> None:
        app = app_factory("financial", flt_education=[]).run()
        assert not app.exception
        body = _body_text(app)
        assert "No customers match the selected filters." in body
        assert any("Reset filters" in str(b.label) for b in app.button)

    def test_reset_restores_the_full_dataset(self, app_factory) -> None:
        app = app_factory("financial").run()
        app.slider[0].set_range(30, 32).run()
        assert "of 30,000 customers selected" in _body_text(app)

        reset = next(b for b in app.button if b.key == "flt_reset")
        reset.click().run()
        assert not app.exception
        assert "30,000</b> of 30,000" in _body_text(app)

    def test_filters_reach_the_data_explorer(self, app_factory) -> None:
        app = app_factory("data_explorer").run()
        assert len(app.slider) >= 3
        app.selectbox[0].select("current").run()
        assert not app.exception
        assert "6,818" in _body_text(app)


class TestDataQuality:
    """Quality page honesty."""

    def test_shows_the_verified_score(self, app_factory) -> None:
        app = app_factory("data_quality").run()
        assert "98.18" in _body_text(app)

    def test_states_the_data_is_fully_populated(self, app_factory) -> None:
        """No manufactured missing-value problem."""
        app = app_factory("data_quality").run()
        assert "fully populated" in _body_text(app)

    def test_declares_the_score_is_not_an_industry_standard(self, app_factory) -> None:
        app = app_factory("data_quality").run()
        assert "not an industry standard" in _body_text(app).lower()

    def test_explains_why_outliers_are_unscored(self, app_factory) -> None:
        app = app_factory("data_quality").run()
        assert "do not reduce the quality score" in _body_text(app)

    def test_shows_the_real_semantic_issues(self, app_factory) -> None:
        app = app_factory("data_quality").run()
        body = _body_text(app)
        assert "Undocumented Category Code" in body or "undocumented" in body.lower()

    def test_is_not_affected_by_filters(self, app_factory) -> None:
        """The assessment describes the raw source, so a filter must not change it."""
        app = app_factory("data_quality", flt_delinquency="current").run()
        assert "98.18" in _body_text(app)


class TestMethodology:
    """Methodology completeness."""

    def test_shows_the_pipeline_stages(self, app_factory) -> None:
        app = app_factory("methodology").run()
        body = _body_text(app)
        for stage in ("Raw data", "Validation", "Cleaning", "Feature engineering", "Action"):
            assert stage in body

    def test_cites_the_dataset_source(self, app_factory) -> None:
        app = app_factory("methodology").run()
        assert "archive.ics.uci.edu" in _body_text(app)

    def test_states_the_missing_variables(self, app_factory) -> None:
        app = app_factory("methodology").run()
        assert "no income, savings, expense, employment or credit-score" in _body_text(app)

    def test_states_forecasting_is_not_offered(self, app_factory) -> None:
        app = app_factory("methodology").run()
        assert "cannot support forecasting" in _body_text(app)

    def test_lists_kpi_definitions(self, app_factory) -> None:
        app = app_factory("methodology").run()
        assert len(app.dataframe) >= 4


class TestSegmentationPage:
    """The Customer Segmentation page."""

    def test_shows_the_header(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        body = _body_text(app)
        assert "Customer Segmentation" in body
        assert "Behaviour-based clustering" in body

    def test_shows_the_segmentation_kpis(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        body = _body_text(app)
        for label in (
            "Customers Segmented", "Behavioural Segments", "Selected K",
            "Silhouette Score", "Features Used",
        ):
            assert label in body, f"missing KPI '{label}'"

    def test_shows_every_required_section(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        body = _body_text(app)
        for heading in (
            "Segmentation Overview",
            "How Segments Were Created",
            "K Selection",
            "Portfolio Segments",
            "Behaviour Map",
            "Segment Risk Profile",
            "Analytical Interpretation",
        ):
            assert heading in body, f"missing section '{heading}'"

    def test_renders_the_four_part_interpretation(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        body = _body_text(app)
        for tag in ("FACT", "BEHAVIOUR", "OBSERVED RISK", "IMPLICATION"):
            assert tag in body

    def test_states_the_outcome_was_excluded(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        body = _body_text(app).lower()
        assert "no outcome used" in body or "excluded from the feature matrix" in body

    def test_states_it_is_not_a_prediction(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        assert "not a prediction" in _body_text(app).lower()

    def test_discloses_the_pca_limitation(self, app_factory) -> None:
        """The projection must not be mistaken for the model."""
        app = app_factory("segmentation").run()
        body = _body_text(app)
        assert "navigational aid" in body
        assert "is not visible on this plot" in body

    def test_discloses_the_scaler_choice(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        assert "robust" in _body_text(app).lower()

    def test_discloses_imputation(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        assert "median" in _body_text(app).lower()

    def test_discloses_the_majority_segment_limitation(self, app_factory) -> None:
        """The inconvenient result must be stated, not hidden."""
        app = app_factory("segmentation").run()
        assert "undifferentiated" in _body_text(app).lower()

    def test_shows_derived_segment_names(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        body = _body_text(app)
        assert "Segment 1" in body
        assert "Delinquent" in body or "Mid-Range" in body

    def test_uses_no_judgemental_labels(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        body = _body_text(app).lower()
        for banned in ("good customer", "bad customer", "safe customer"):
            assert banned not in body

    def test_uses_no_causal_or_prediction_language(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        body = _body_text(app).lower()
        for banned in ("causes", "will default", "guaranteed", "model predicts"):
            assert banned not in body

    def test_renders_profile_tables(self, app_factory) -> None:
        app = app_factory("segmentation").run()
        assert len(app.dataframe) >= 4

    def test_survives_an_active_filter(self, app_factory) -> None:
        """Segments are fitted on the portfolio, so a filter must not break them."""
        app = app_factory("segmentation", flt_delinquency="current").run()
        assert not app.exception
        body = _body_text(app)
        assert "filtered selection across these segments" in body

    def test_survives_an_empty_filter_selection(self, app_factory) -> None:
        app = app_factory("segmentation", flt_education=[]).run()
        assert not app.exception

    def test_segments_stay_fitted_on_the_portfolio(self, app_factory) -> None:
        app = app_factory("segmentation", flt_delinquency="current").run()
        assert "30,000" in _body_text(app)


class TestPlaceholderPages:
    """Unbuilt pages must be honest."""

    @pytest.mark.parametrize(
        "page_id", ["credit_risk", "ai_insights", "recommendations"]
    )
    def test_states_it_is_not_implemented(self, app_factory, page_id: str) -> None:
        app = app_factory(page_id).run()
        assert "not implemented yet" in _body_text(app)

    @pytest.mark.parametrize(
        "page_id", ["credit_risk", "ai_insights", "recommendations"]
    )
    def test_renders_no_charts(self, app_factory, page_id: str) -> None:
        """A placeholder must not show mock visuals."""
        app = app_factory(page_id).run()
        assert len(app.dataframe) == 0

    def test_ai_page_reports_analytics_mode(self, app_factory) -> None:
        app = app_factory("ai_insights").run()
        assert "Analytics Mode" in _body_text(app)


class TestNavigationAndStatus:
    """Sidebar behaviour."""

    def test_default_page_is_the_overview(self, app_factory) -> None:
        app = app_factory().run()
        assert app.session_state["fr_active_page"] == "executive"

    def test_nav_buttons_exist_for_every_page(self, app_factory) -> None:
        app = app_factory().run()
        nav_keys = {str(b.key) for b in app.button if str(b.key).startswith("nav_")}
        assert len(nav_keys) == len(PAGE_IDS)

    def test_clicking_navigates(self, app_factory) -> None:
        app = app_factory().run()
        target = next(b for b in app.button if b.key == "nav_methodology")
        target.click().run()
        assert app.session_state["fr_active_page"] == "methodology"
        assert not app.exception

    def test_reports_the_analytics_engine_as_online(self, app_factory) -> None:
        app = app_factory().run()
        assert "Analytics Engine Online" in _body_text(app)

    def test_does_not_claim_ai_is_enabled(self, app_factory) -> None:
        """With AI_PROVIDER=none the shell must say Analytics Mode."""
        app = app_factory().run()
        body = _body_text(app)
        assert "Analytics Mode" in body
        assert "AI Insights Enabled" not in body
