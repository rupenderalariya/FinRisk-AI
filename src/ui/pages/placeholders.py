"""
Placeholders for pages whose engines are not yet built.

These pages state plainly what they will contain, which phase delivers them, and
what already exists to support them. No mock charts, no sample numbers and no
simulated AI text: a placeholder that fakes content is worse than an empty page,
because a reader cannot tell which parts of the product are real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import streamlit as st

from src.config import settings
from src.ui import components
from src.ui.context import PageContext
from src.ui.navigation import item_for
from src.ui.theme import PALETTE


@dataclass(frozen=True)
class PlannedPage:
    """Description of a page that is not yet implemented."""

    page_id: str
    title: str
    description: str
    planned: tuple[str, ...]
    foundations: tuple[str, ...]
    phase: str


PLANNED_PAGES: dict[str, PlannedPage] = {
    "credit_risk": PlannedPage(
        page_id="credit_risk",
        title="Credit Risk",
        description=(
            "A predictive default model with honest evaluation, explainability and a "
            "fairness audit."
        ),
        planned=(
            "A dummy baseline first, then logistic regression, random forest and "
            "gradient boosting, compared on the same stratified split.",
            "ROC-AUC and PR-AUC as headline metrics with recall on the default class, "
            "because predicting 'no default' for everyone already scores 77.88% "
            "accuracy on this data.",
            "A decision threshold tuned against a stated cost assumption rather than "
            "left at 0.5.",
            "Global explainability through permutation importance, with SHAP attempted "
            "and a graceful fallback if it is unavailable.",
            "Risk tiers derived from predicted probability, labelled 'model-derived' "
            "to keep them distinct from the observed historical outcome.",
            "A fairness audit comparing error rates across sex, age band and education.",
            "A single-customer prediction interface with a per-customer contribution "
            "breakdown.",
        ),
        foundations=(
            "Driver analysis already ranks candidate features using four triangulated "
            "methods, which is the preparation step for model feature selection.",
            "Sex and marital status are already excluded from candidate features on "
            "fairness grounds.",
            "The leakage question is already documented: the most recent repayment "
            "status is legitimately prior information, not leakage.",
        ),
        phase="the risk-model phase",
    ),
    "ai_insights": PlannedPage(
        page_id="ai_insights",
        title="AI Insights",
        description=(
            "Natural-language narration over results that have already been computed."
        ),
        planned=(
            "A provider abstraction covering OpenAI, Gemini and Ollama, selected by "
            "the AI_PROVIDER environment variable.",
            "A strict contract: the model receives a compact JSON of computed KPIs, "
            "trends, driver rankings and segment profiles - never customer rows.",
            "Post-validation of generated text, checking numerals in the output "
            "against the supplied payload so a fabricated figure is caught rather "
            "than displayed.",
            "Deterministic template-generated summaries as a first-class path, so the "
            "page is useful with no API key at all.",
        ),
        foundations=(
            "Every analytical result already serialises to a compact aggregate payload "
            "through `to_dict()`, with dataframes deliberately excluded.",
            "Configuration and key handling are already in place, with keys read on "
            "demand and a redaction filter on all logging.",
        ),
        phase="the AI insights phase",
    ),
    "recommendations": PlannedPage(
        page_id="recommendations",
        title="Recommendations",
        description=(
            "Rule-driven recommendations, each traceable to the metric that triggered "
            "it."
        ),
        planned=(
            "Rule objects that declare a trigger condition on computed metrics and "
            "emit the FACT → INSIGHT → RISK/OPPORTUNITY → ACTION chain.",
            "A rule that does not trigger produces nothing, so there is no filler.",
            "Each recommendation rendered with its supporting number visible.",
            "Framing as options for review rather than instructions, and never as "
            "financial advice.",
        ),
        foundations=(
            "The four-part chain structure already exists and is enforced: a finding "
            "cannot be constructed without at least one computed metric as evidence.",
            "The Executive Overview already renders derived findings through this "
            "structure; this page will expand it into a full rule library.",
        ),
        phase="the recommendation-engine phase",
    ),
}


def _bullets(items: Sequence[str], colour: str, glyph: str = "•") -> None:
    """Render a styled bullet list."""
    for item in items:
        st.markdown(
            f'<div style="display:flex;gap:0.6rem;padding:0.3rem 0;'
            f'font-size:0.815rem;color:{PALETTE["text_secondary"]};line-height:1.55">'
            f'<span style="color:{colour};flex-shrink:0">{glyph}</span>'
            f"<span>{components.escape(item)}</span></div>",
            unsafe_allow_html=True,
        )


def render(context: PageContext, page_id: str) -> None:
    """Render a placeholder page.

    Args:
        context: The per-run page context.
        page_id: Which planned page to describe.
    """
    planned = PLANNED_PAGES.get(page_id)
    if planned is None:
        item = item_for(page_id)
        components.page_header(title=item.label)
        components.unavailable(item.label, "This page is not available.", item.phase)
        return

    components.page_header(
        title=planned.title,
        description=planned.description,
        meta=[("Status", "Not yet implemented"), ("Arrives in", planned.phase.title())],
        status=("Planned", "muted"),
    )

    components.notice(
        "This page is not implemented yet",
        f"It is scheduled for {planned.phase}. Nothing is shown here rather than "
        "sample charts or placeholder numbers, so that everything visible elsewhere "
        "in the application can be trusted as real.",
        icon="◷",
        tone="attention",
    )

    if page_id == "ai_insights" and not settings.ai_enabled:
        components.spacer(0.6)
        components.notice(
            "Currently running in Analytics Mode",
            f"AI_PROVIDER is set to '{settings.ai_provider}', so no AI narration is "
            "generated. The platform is fully functional in this mode: every insight "
            "elsewhere in the application is computed from the data, not written by a "
            "language model.",
            icon="◑",
        )

    components.section(
        "What this page will contain",
        "The planned scope, so the gap is explicit.",
    )
    _bullets(planned.planned, PALETTE["info"], "▸")

    components.section(
        "What already exists to support it",
        "Foundations delivered in earlier phases.",
    )
    _bullets(planned.foundations, PALETTE["positive"], "✓")

    components.spacer(0.5)
    st.caption(
        "In the meantime, the Executive Overview, Financial Analytics, Data Explorer, "
        "Data Quality and Methodology pages are fully implemented and operate on the "
        f"complete {len(context.pipeline.raw):,}-record dataset."
    )
