"""
Custom grouped sidebar navigation.

Streamlit's native multipage navigation cannot be grouped or restyled, so pages
are registered here and routed through session state. Each entry declares whether
it is implemented, which lets the sidebar mark forthcoming pages honestly rather
than linking to empty screens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import streamlit as st

from src.ui.components import escape, render_brand, status_pill
from src.ui.theme import PALETTE

#: Session-state key holding the active page id.
ACTIVE_PAGE_KEY: Final[str] = "fr_active_page"


@dataclass(frozen=True)
class NavItem:
    """One navigation entry."""

    page_id: str
    label: str
    icon: str
    group: str
    implemented: bool = True
    phase: str = ""


#: The navigation model. Order here is the order rendered.
NAV_ITEMS: Final[tuple[NavItem, ...]] = (
    NavItem("executive", "Executive Overview", "◧", "Overview"),
    NavItem("financial", "Financial Analytics", "◫", "Analytics"),
    NavItem("segmentation", "Customer Segmentation", "◈", "Analytics"),
    NavItem(
        "credit_risk", "Credit Risk", "▲", "Analytics",
        implemented=False, phase="the risk-model phase",
    ),
    NavItem(
        "ai_insights", "AI Insights", "✦", "Intelligence",
        implemented=False, phase="the AI insights phase",
    ),
    NavItem(
        "recommendations", "Recommendations", "➔", "Intelligence",
        implemented=False, phase="the recommendation-engine phase",
    ),
    NavItem("data_explorer", "Data Explorer", "▦", "Data"),
    NavItem("data_quality", "Data Quality", "◉", "Data"),
    NavItem("methodology", "Methodology", "❖", "System"),
)

#: Group render order.
GROUP_ORDER: Final[tuple[str, ...]] = (
    "Overview",
    "Analytics",
    "Intelligence",
    "Data",
    "System",
)


def item_for(page_id: str) -> NavItem:
    """Look up a navigation item, falling back to the first entry."""
    for item in NAV_ITEMS:
        if item.page_id == page_id:
            return item
    return NAV_ITEMS[0]


def active_page() -> str:
    """Return the active page id, initialising it on first run."""
    if ACTIVE_PAGE_KEY not in st.session_state:
        st.session_state[ACTIVE_PAGE_KEY] = NAV_ITEMS[0].page_id
    return str(st.session_state[ACTIVE_PAGE_KEY])


def set_active_page(page_id: str) -> None:
    """Set the active page."""
    st.session_state[ACTIVE_PAGE_KEY] = page_id


def ai_status() -> tuple[str, str]:
    """Describe AI availability for the status pill.

    Never claims AI is active when no provider is configured, which is the whole
    point of surfacing it.

    Returns:
        ``(label, tone)``.
    """
    from src.config import settings

    if settings.ai_enabled:
        return f"AI Insights Enabled · {settings.ai_provider}", "positive"
    if settings.ai_provider != "none":
        # A provider was selected but is unusable, so say so rather than implying
        # either state.
        return "Analytics Mode · AI key not configured", "info"
    return "Analytics Mode", "info"


def render_sidebar(
    footer_note: str = "",
    dataset_label: str = "",
) -> str:
    """Render the sidebar and return the selected page id.

    Args:
        footer_note: Small print shown at the bottom, e.g. the dataset period.
        dataset_label: Dataset name shown under the engine status.

    Returns:
        The active page id after handling any click.
    """
    current = active_page()

    with st.sidebar:
        render_brand()

        st.markdown(
            f'<div style="padding:0 0.35rem 0.55rem 0.35rem">'
            f'{status_pill("Analytics Engine Online", "positive")}</div>',
            unsafe_allow_html=True,
        )
        label, tone = ai_status()
        st.markdown(
            f'<div style="padding:0 0.35rem 0.9rem 0.35rem">'
            f"{status_pill(label, tone)}</div>",
            unsafe_allow_html=True,
        )

        for group in GROUP_ORDER:
            items = [item for item in NAV_ITEMS if item.group == group]
            if not items:
                continue
            st.markdown(
                f'<div class="fr-nav-group">{escape(group)}</div>', unsafe_allow_html=True
            )
            for item in items:
                suffix = "" if item.implemented else "  ·  soon"
                if st.button(
                    f"{item.icon}   {item.label}{suffix}",
                    key=f"nav_{item.page_id}",
                    width="stretch",
                    type="primary" if item.page_id == current else "secondary",
                ):
                    set_active_page(item.page_id)
                    st.rerun()

        st.markdown(
            f'<div style="height:1px;background:{PALETTE["border"]};'
            f'margin:1.2rem 0 0.7rem 0"></div>',
            unsafe_allow_html=True,
        )
        if dataset_label:
            st.markdown(
                f'<div style="font-size:0.68rem;color:{PALETTE["text_muted"]};'
                f'padding:0 0.35rem;line-height:1.5">{escape(dataset_label)}</div>',
                unsafe_allow_html=True,
            )
        if footer_note:
            st.markdown(
                f'<div style="font-size:0.66rem;color:{PALETTE["text_faint"]};'
                f'padding:0.4rem 0.35rem 0 0.35rem;line-height:1.5">'
                f"{escape(footer_note)}</div>",
                unsafe_allow_html=True,
            )

    return active_page()
