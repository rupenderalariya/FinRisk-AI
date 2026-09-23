"""
Methodology.

The page that makes the rest of the product auditable: where the data came from,
what was done to it, how each KPI is defined, and - importantly - what this
dataset cannot support. Every table is generated from the same objects the
analytics use, so the documentation cannot drift from the implementation.
"""

from __future__ import annotations

import pandas as pd

import streamlit as st

from src.data_quality import SCORE_METHODOLOGY
from src.filtering import limit_band_documentation, utilisation_band_documentation
from src.kpi_engine import kpi_catalogue
from src.schema import (
    DATASET_CITATION,
    DATASET_CURRENCY,
    DATASET_LANDING_URL,
    DATASET_LICENCE,
    DATASET_NAME,
    DATASET_PERIOD,
    DATASET_SOURCE,
    describe_panel_alignment,
)
from src.trends import PANEL_NOTE
from src.ui import components
from src.ui.context import PageContext
from src.ui.navigation import NAV_ITEMS
from src.ui.theme import PALETTE

#: Pipeline stages with their explanations. Mirrors the real execution order in
#: :func:`src.pipeline.run_pipeline`.
PIPELINE_STAGES: tuple[tuple[str, str], ...] = (
    (
        "Raw data",
        "The source .xls is read with the header row detected by evidence rather "
        "than assumption: candidate rows are scored by how many recognised columns "
        "they yield. The source file is opened read-only and never modified.",
    ),
    (
        "Validation",
        "Sixteen contract checks run before anything else. Errors that make "
        "analysis impossible (no rows, no target, a non-binary target) stop the "
        "pipeline; differences that remain analysable (an unexpected row count) "
        "become warnings so a filtered extract still works.",
    ),
    (
        "Quality assessment",
        "Completeness, uniqueness, validity and consistency are scored on the raw "
        "data, before cleaning, so the report describes the source as it arrived "
        "rather than the state after our own fixes.",
    ),
    (
        "Cleaning",
        "Six steps, each recording its rationale. Integer codes gain readable "
        "labels while the originals are kept; undocumented codes become 'Unknown' "
        "rather than being guessed; suspect values are flagged rather than deleted. "
        "No row is removed.",
    ),
    (
        "Feature engineering",
        "37 behavioural features are derived from the monthly panel - utilisation, "
        "repayment ratio, delinquency counts and balance trajectory. Every ratio "
        "passes through a guard that returns 'undefined' instead of infinity when a "
        "denominator is zero.",
    ),
    (
        "Analytics",
        "Univariate, bivariate and multivariate analysis, period-over-period trends "
        "across the six observed months, and KPI computation. Analyses on too few "
        "records are suppressed with an explanation instead of being computed on a "
        "handful of rows.",
    ),
    (
        "Risk analysis",
        "Four independent driver methods - rank correlation, distribution "
        "comparison, categorical independence testing and tree importance - are "
        "triangulated. Observed signals are labelled as measurements, never as "
        "predictions.",
    ),
    (
        "AI insights",
        "Optional narration over already-computed results. The layer receives "
        "aggregate statistics only, never customer rows, and the platform is fully "
        "functional with no AI provider configured. Not yet implemented.",
    ),
    (
        "Action",
        "Findings are expressed as FACT → INSIGHT → IMPLICATION → ACTION, and a "
        "finding cannot be constructed without at least one computed metric "
        "attached as evidence.",
    ),
)


def _render_provenance(context: PageContext) -> None:
    """Dataset source, licence and citation."""
    provenance = context.pipeline.dataset_provenance()
    rows = [
        {"Property": "Dataset", "Value": DATASET_NAME},
        {"Property": "Source", "Value": DATASET_SOURCE},
        {"Property": "URL", "Value": DATASET_LANDING_URL},
        {"Property": "Licence", "Value": DATASET_LICENCE},
        {"Property": "Period covered", "Value": DATASET_PERIOD},
        {"Property": "Currency", "Value": f"{DATASET_CURRENCY} (New Taiwan Dollar)"},
        {"Property": "Rows", "Value": f"{len(context.pipeline.raw):,}"},
        {"Property": "Source columns", "Value": f"{len(context.pipeline.raw.columns)}"},
        {
            "Property": "Columns after processing",
            "Value": f"{len(context.pipeline.data.columns)}",
        },
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"Citation: {DATASET_CITATION}")


def _render_panel(context: PageContext) -> None:
    """Explain the monthly panel structure and its limits."""
    st.markdown(
        f'<div class="fr-panel"><div class="fr-insight-body">'
        f"{components.escape(describe_panel_alignment())}</div></div>",
        unsafe_allow_html=True,
    )
    components.spacer(0.5)
    st.markdown(
        f'<div class="fr-notice" style="text-align:left;padding:0.85rem 1rem">'
        f'<div class="fr-notice-body" style="max-width:none;margin:0">'
        f"<b>What this supports, and what it does not.</b> "
        f"{components.escape(PANEL_NOTE)}</div></div>",
        unsafe_allow_html=True,
    )


def _render_kpi_definitions() -> None:
    """The full KPI catalogue with formulas."""
    catalogue = kpi_catalogue()
    frame = pd.DataFrame(
        [
            {
                "KPI": entry["label"],
                "Category": entry["category"],
                "Formula": entry["formula"],
                "Definition": entry["description"],
            }
            for entry in catalogue
        ]
    )
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        height=460,
        column_config={
            "Definition": st.column_config.TextColumn(width="large"),
            "Formula": st.column_config.TextColumn(width="medium"),
        },
    )
    st.caption(
        f"{len(catalogue)} KPIs are registered. A KPI whose required columns are "
        "absent is omitted entirely rather than displayed as 'N/A', and the reason "
        "is recorded."
    )


def _render_features(context: PageContext) -> None:
    """Engineered features grouped by category."""
    report = context.pipeline.feature_report
    for category, definitions in report.by_category().items():
        with st.expander(f"{category} ({len(definitions)} features)", expanded=False):
            frame = pd.DataFrame(
                [
                    {
                        "Feature": d.name,
                        "Formula": d.formula,
                        "Interpretation": d.interpretation,
                        "Model input": "Yes" if d.used_for_ml else "No",
                    }
                    for d in definitions
                ]
            )
            st.dataframe(
                frame,
                width="stretch",
                hide_index=True,
                column_config={
                    "Interpretation": st.column_config.TextColumn(width="large")
                },
            )


def _render_limitations(context: PageContext) -> None:
    """What this dataset cannot support - stated plainly."""
    report = context.pipeline.feature_report

    st.markdown(
        f'<div class="fr-notice" style="text-align:left;padding:1rem 1.1rem;'
        f'border-color:{PALETTE["attention_dim"]}">'
        f'<div class="fr-notice-title" style="text-align:left">'
        f"This dataset contains no income, savings, expense, employment or credit-score "
        f"variable</div>"
        f'<div class="fr-notice-body" style="max-width:none;margin:0.35rem 0 0 0">'
        f"Debt-to-income, savings rate and expense ratio therefore cannot be computed. "
        f"They are <b>not</b> approximated from unrelated fields. Credit utilisation "
        f"is used as the closest valid substitute, and the credit limit is described "
        f"as a credit limit throughout - never as income or wealth."
        f"</div></div>",
        unsafe_allow_html=True,
    )
    components.spacer(0.6)

    if report.skipped:
        rows = [
            {
                "Measure": d.name,
                "Would require": ", ".join(d.source_columns),
                "Status": d.skip_reason or "unspecified",
            }
            for d in report.skipped
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    unavailable = context.registry.unavailable_explanations()
    if unavailable:
        components.spacer(0.5)
        st.markdown("**Analytical capabilities unavailable in this dataset**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Capability": name, "Explanation": explanation}
                    for name, explanation in unavailable.items()
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={"Explanation": st.column_config.TextColumn(width="large")},
        )

    components.spacer(0.6)
    st.markdown("**Other limitations to keep in view**")
    for item in (
        "The data covers April to September 2005 in Taiwan. Findings are historical "
        "and region-specific, and do not transfer to a present-day portfolio.",
        "The target is narrow: default on the single following month's payment, not "
        "a broader definition of credit loss.",
        "Six monthly periods support period-over-period comparison but cannot "
        "support forecasting, seasonality decomposition, or daily and weekly "
        "resampling. None of these is offered.",
        "All relationships shown are observed associations. This is observational "
        "data, so causal claims are unavailable at any sample size.",
        "Risk signals in this build are measured historical outcomes. No predictive "
        "model is active yet, so nothing shown is a prediction.",
        "Sex and marital status are excluded from driver analysis and from planned "
        "model features. They appear only in descriptive and fairness views.",
    ):
        st.markdown(
            f'<div style="display:flex;gap:0.6rem;padding:0.3rem 0;'
            f'font-size:0.815rem;color:{PALETTE["text_secondary"]};line-height:1.55">'
            f'<span style="color:{PALETTE["attention"]}">•</span>'
            f"<span>{components.escape(item)}</span></div>",
            unsafe_allow_html=True,
        )


def _render_roadmap() -> None:
    """What is built and what is still to come."""
    rows = [
        {
            "Page": item.label,
            "Status": "Implemented" if item.implemented else "Planned",
            "Notes": "" if item.implemented else f"Arrives in {item.phase}.",
        }
        for item in NAV_ITEMS
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render(context: PageContext) -> None:
    """Render the Methodology page.

    Args:
        context: The per-run page context.
    """
    components.page_header(
        title="Methodology",
        description=(
            "How the data is sourced, processed and analysed - and what it cannot "
            "support. Every table here is generated from the same definitions the "
            "analytics use."
        ),
        meta=[
            ("Pipeline stages", f"{len(PIPELINE_STAGES)}"),
            ("Features engineered", f"{len(context.pipeline.feature_report.created)}"),
            ("KPIs registered", f"{len(kpi_catalogue())}"),
        ],
    )

    components.section(
        "Analytical Pipeline",
        "Each stage runs in this order on every load. The whole sequence completes "
        f"in about {context.pipeline.duration_seconds:.1f} seconds for "
        f"{len(context.pipeline.raw):,} records.",
    )
    components.pipeline_diagram(PIPELINE_STAGES)

    components.section("Dataset & Provenance", "Where the data comes from.")
    _render_provenance(context)

    components.section(
        "Panel Structure",
        "The single most misread aspect of this dataset, so it is spelled out.",
    )
    _render_panel(context)

    components.section(
        "KPI Definitions",
        "Exact formula and definition for every registered KPI.",
    )
    _render_kpi_definitions()

    components.section(
        "Engineered Features",
        "Derived measures, their formulas, and whether each is suitable as a model "
        "input.",
    )
    _render_features(context)

    components.section(
        "Banding Rules",
        "How the analytical bands used across the dashboard are defined.",
    )
    with st.expander("Utilisation bands", expanded=False):
        st.caption(utilisation_band_documentation())
    with st.expander("Credit-limit bands", expanded=False):
        st.caption(limit_band_documentation())

    components.section(
        "Data Quality Scoring",
        "A project-defined heuristic, documented in full so it is not mistaken for "
        "an industry standard.",
    )
    with st.expander("Scoring methodology", expanded=False):
        st.code(SCORE_METHODOLOGY, language="text")

    components.section(
        "Limitations",
        "What this dataset and this build genuinely cannot do.",
    )
    _render_limitations(context)

    components.section("Build Status", "Which pages are implemented.")
    _render_roadmap()
