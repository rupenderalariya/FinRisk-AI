"""
Layout primitives shared by every page.

Each helper renders one piece of chrome and nothing else, so markup exists in a
single place. All interpolated text passes through :func:`escape` first, which
keeps a stray ``<`` in a data label from breaking the page.
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Mapping, Sequence

import streamlit as st

from src.ui.theme import PALETTE, severity_colour, with_alpha


def escape(value: Any) -> str:
    """HTML-escape a value for safe interpolation into markup."""
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------- #
# Brand and status
# --------------------------------------------------------------------------- #


def status_pill(label: str, tone: str = "positive") -> str:
    """Build a small status pill.

    Args:
        label: Text to display.
        tone: ``positive``, ``info`` or ``muted``.

    Returns:
        HTML for the pill.
    """
    dot_class = {"positive": "", "info": "info", "muted": "muted"}.get(tone, "")
    return (
        f'<span class="fr-status">'
        f'<span class="fr-status-dot {dot_class}"></span>{escape(label)}</span>'
    )


def render_brand() -> None:
    """Render the sidebar brand block."""
    st.markdown(
        '<div class="fr-brand"><div class="fr-brand-row">'
        '<div class="fr-brand-mark">FR</div>'
        '<div><div class="fr-brand-name">FinRisk AI</div>'
        '<div class="fr-brand-sub">Credit Risk Intelligence</div></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Headers
# --------------------------------------------------------------------------- #


def page_header(
    title: str,
    description: str = "",
    meta: Sequence[tuple[str, str]] = (),
    status: tuple[str, str] | None = None,
) -> None:
    """Render a page header with optional metadata and a status pill.

    Args:
        title: Page title.
        description: Supporting sentence.
        meta: ``(label, value)`` pairs shown right-aligned.
        status: ``(label, tone)`` for a status pill beside the title.
    """
    status_html = ""
    if status is not None:
        status_html = f'<div style="margin-top:0.55rem">{status_pill(*status)}</div>'

    meta_html = ""
    if meta:
        items = "".join(
            f'<div class="fr-meta-item">'
            f'<div class="fr-meta-label">{escape(label)}</div>'
            f'<div class="fr-meta-value">{escape(value)}</div></div>'
            for label, value in meta
        )
        meta_html = f'<div class="fr-head-meta">{items}</div>'

    description_html = (
        f'<div class="fr-page-desc">{escape(description)}</div>' if description else ""
    )

    st.markdown(
        f'<div class="fr-page-head"><div>'
        f'<h1 class="fr-page-title">{escape(title)}</h1>'
        f"{description_html}{status_html}"
        f"</div>{meta_html}</div>",
        unsafe_allow_html=True,
    )


def section(title: str, description: str = "", rule: bool = True) -> None:
    """Render a section heading.

    Args:
        title: Section title.
        description: Supporting sentence.
        rule: Draw a hairline beneath the heading.
    """
    description_html = (
        f'<div class="fr-section-desc">{escape(description)}</div>' if description else ""
    )
    rule_html = '<div class="fr-section-rule"></div>' if rule else ""
    st.markdown(
        f'<div class="fr-section">'
        f'<div class="fr-section-title">{escape(title)}</div>'
        f"{description_html}{rule_html}</div>",
        unsafe_allow_html=True,
    )


def badge(label: str, tone: str = "neutral") -> str:
    """Build a small badge.

    Args:
        label: Badge text.
        tone: A severity name, a palette key, or ``neutral``.

    Returns:
        HTML for the badge.
    """
    colour = (
        severity_colour(tone)
        if tone.lower() in {"critical", "high", "medium", "low", "info"}
        else PALETTE.get(tone, PALETTE["neutral"])
    )
    return (
        f'<span class="fr-badge" style="--fr-badge-fg:{colour};'
        f"--fr-badge-border:{with_alpha(colour, 0.35)};"
        f'--fr-badge-bg:{with_alpha(colour, 0.10)}">{escape(label)}</span>'
    )


# --------------------------------------------------------------------------- #
# Notices
# --------------------------------------------------------------------------- #


def notice(
    title: str, body: str = "", icon: str = "", tone: str = "neutral"
) -> None:
    """Render a product-style notice panel.

    Used for empty states and unavailable analyses, so the user never sees a
    blank area or a raw exception.

    Args:
        title: Headline.
        body: Explanation.
        icon: Optional glyph.
        tone: ``neutral``, ``attention`` or ``risk``.
    """
    tone_class = f" {tone}" if tone in {"attention", "risk"} else ""
    icon_html = f'<div class="fr-notice-icon">{escape(icon)}</div>' if icon else ""
    body_html = f'<div class="fr-notice-body">{escape(body)}</div>' if body else ""
    st.markdown(
        f'<div class="fr-notice{tone_class}">{icon_html}'
        f'<div class="fr-notice-title">{escape(title)}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )


def empty_state(
    message: str = "No customers match the selected filters.",
    hint: str = "Widen or clear the filters to bring data back into view.",
    on_reset: str | None = None,
) -> None:
    """Render the standard empty state, optionally with a reset button.

    Args:
        message: Headline explaining the empty result.
        hint: Guidance on what to do next.
        on_reset: Session-state key to set True when the reset button is pressed.
    """
    notice(message, hint, icon="◍", tone="attention")
    if on_reset:
        left, _ = st.columns([1, 4])
        with left:
            if st.button("Reset filters", key=f"reset_{on_reset}", width="stretch"):
                st.session_state[on_reset] = True
                st.rerun()


def unavailable(
    analysis: str,
    reason: str = "",
    phase: str = "",
) -> None:
    """Render a panel explaining why an analysis cannot be shown.

    Args:
        analysis: What was unavailable.
        reason: Why.
        phase: Optional delivery phase for not-yet-built features.
    """
    body = reason or "The required variables are not available in this selection."
    if phase:
        body = f"{body} Planned for {phase}."
    notice(f"{analysis} unavailable", body, icon="⊘")


def caption(text: str) -> None:
    """Render a muted caption line."""
    st.caption(text)


def methodology_note(title: str, body: str) -> None:
    """Render a collapsed expander holding a methodology explanation.

    Kept collapsed so the page stays clean while the definition remains one
    click away.
    """
    with st.expander(title, expanded=False):
        st.markdown(
            f'<div style="font-size:0.82rem;color:{PALETTE["text_secondary"]};'
            f'line-height:1.6">{escape(body)}</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #


def panel_open(tight: bool = False) -> None:
    """Open a surface panel ``div``. Pair with :func:`panel_close`."""
    st.markdown(
        f'<div class="fr-panel{" tight" if tight else ""}">', unsafe_allow_html=True
    )


def panel_close() -> None:
    """Close a surface panel ``div``."""
    st.markdown("</div>", unsafe_allow_html=True)


def stat_row(items: Sequence[tuple[str, str]], columns: int | None = None) -> None:
    """Render a compact row of label/value statistics.

    Args:
        items: ``(label, value)`` pairs.
        columns: Column count. Defaults to the number of items.
    """
    if not items:
        return
    count = columns or len(items)
    for index, column in enumerate(st.columns(count)):
        if index >= len(items):
            continue
        label, value = items[index]
        with column:
            st.markdown(
                f'<div class="fr-meta-label">{escape(label)}</div>'
                f'<div class="fr-meta-value" style="font-size:1.05rem">{escape(value)}</div>',
                unsafe_allow_html=True,
            )


def spacer(height: float = 0.8) -> None:
    """Insert vertical space in rem."""
    st.markdown(f'<div style="height:{height}rem"></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Chart rendering
# --------------------------------------------------------------------------- #


def chart(figure: Any, key: str | None = None) -> None:
    """Render a Plotly figure with the product's standard configuration.

    Args:
        figure: The figure to render.
        key: Optional Streamlit element key.
    """
    st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        config={
            "displayModeBar": False,
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": False,
        },
    )


def safe_render(label: str, render_fn: Any, *args: Any, **kwargs: Any) -> bool:
    """Run a render function, converting a failure into a local notice.

    Wrapping each section individually means one failing analysis degrades to a
    single notice instead of blanking everything below it on the page.

    Args:
        label: What is being rendered, used in the message and the log.
        render_fn: The callable to invoke.
        *args: Positional arguments for ``render_fn``.
        **kwargs: Keyword arguments for ``render_fn``.

    Returns:
        True when the section rendered, False when it failed.
    """
    from src.logging_setup import get_logger

    logger = get_logger(__name__)
    try:
        render_fn(*args, **kwargs)
        return True
    except Exception as exc:  # noqa: BLE001 - keep tracebacks out of the UI
        logger.exception("Section '%s' failed to render.", label)
        notice(
            f"{label} unavailable",
            "This section could not be displayed for the current selection. "
            f"Technical detail has been logged ({type(exc).__name__}). "
            "Other sections on this page are unaffected.",
            icon="⊘",
            tone="attention",
        )
        return False


def pipeline_diagram(stages: Sequence[tuple[str, str]]) -> None:
    """Render a vertical pipeline diagram.

    Args:
        stages: ``(name, description)`` pairs in order.
    """
    blocks: list[str] = []
    for index, (name, description) in enumerate(stages, start=1):
        blocks.append(
            f'<div class="fr-pipe-stage">'
            f'<div class="fr-pipe-num">{index}</div>'
            f'<div><div class="fr-pipe-name">{escape(name)}</div>'
            f'<div class="fr-pipe-desc">{escape(description)}</div></div></div>'
        )
        if index < len(stages):
            blocks.append('<div class="fr-pipe-arrow">↓</div>')
    st.markdown(f'<div class="fr-pipe">{"".join(blocks)}</div>', unsafe_allow_html=True)
