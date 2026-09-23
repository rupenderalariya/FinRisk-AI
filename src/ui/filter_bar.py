"""
Filter widgets.

Renders controls bound to :class:`src.filtering.FilterSpec`. This module owns the
widgets; :mod:`src.filtering` owns the logic. Keeping them apart means the filter
behaviour is unit tested without a browser.

Filters live in session state so a selection survives page navigation, and a
single reset clears everything without restarting the app.
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from src.filtering import (
    DELINQUENCY_OPTIONS,
    FilterBounds,
    FilterResult,
    FilterSpec,
    default_spec,
)
from src.ui.components import escape
from src.ui.theme import PALETTE

#: Session-state keys.
SPEC_KEY: Final[str] = "fr_filter_spec"
RESET_FLAG: Final[str] = "fr_filter_reset"
_WIDGET_KEYS: Final[tuple[str, ...]] = (
    "flt_age",
    "flt_limit",
    "flt_util",
    "flt_education",
    "flt_marriage",
    "flt_delinquency",
)


def current_spec(bounds: FilterBounds) -> FilterSpec:
    """Reconstruct the active specification from widget state.

    Reading the individual **widget keys** rather than a separately stored spec
    matters for ordering. Streamlit reruns the whole script when a control
    changes, and the page context is assembled *before* the filter bar renders.
    Widget state is already updated at that point, so reading it here means a
    filter change takes effect on the same rerun instead of lagging by one.

    Args:
        bounds: Data-derived selectable ranges, used for the defaults.

    Returns:
        The active :class:`FilterSpec`.
    """
    fallback = default_spec(bounds)

    if st.session_state.get(RESET_FLAG):
        _clear_widget_state()
        st.session_state[RESET_FLAG] = False
        return fallback

    def stored(key: str, default):
        value = st.session_state.get(key)
        return default if value is None else value

    age = stored("flt_age", fallback.age_range)
    limit = stored("flt_limit", fallback.limit_range)
    utilisation = stored("flt_util", fallback.utilisation_range)
    education = stored("flt_education", fallback.education)
    marriage = stored("flt_marriage", fallback.marriage)
    delinquency = stored("flt_delinquency", fallback.delinquency)

    return FilterSpec(
        age_range=tuple(age) if age else None,
        limit_range=tuple(limit) if limit else None,
        utilisation_range=tuple(utilisation) if utilisation else None,
        # An empty multiselect is a deliberate "none", so it is preserved rather
        # than being treated as unset.
        education=tuple(education) if education is not None else None,
        marriage=tuple(marriage) if marriage is not None else None,
        sex=None,
        delinquency=str(delinquency or "all"),
    )


def _clear_widget_state() -> None:
    """Drop widget state so controls re-initialise at their defaults."""
    for key in _WIDGET_KEYS:
        st.session_state.pop(key, None)


def request_reset() -> None:
    """Flag a reset to be applied on the next run."""
    st.session_state[RESET_FLAG] = True


def render_filter_bar(
    bounds: FilterBounds,
    result: FilterResult | None = None,
    key_prefix: str = "",
) -> FilterSpec:
    """Render the filter controls and return the resulting specification.

    Controls for dimensions the data does not provide are omitted rather than
    disabled, so the bar only ever offers filters that work.

    Args:
        bounds: Data-derived selectable ranges.
        result: The previous run's result, used for the selection count.
        key_prefix: Prefix for widget keys when reused on several pages.

    Returns:
        The specification reflecting the current widget state.
    """
    spec = current_spec(bounds)

    count_html = ""
    if result is not None:
        count_html = (
            f'<div class="fr-filter-count"><b>{result.n_selected:,}</b> of '
            f"{result.n_total:,} customers selected"
            + (f" · {result.share_pct:.1f}%" if result.is_filtered else "")
            + "</div>"
        )
    st.markdown(
        f'<div class="fr-filter-head">'
        f'<div class="fr-section-title" style="font-size:0.95rem">Filters</div>'
        f"{count_html}</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        row1 = st.columns([1.1, 1.3, 1.2], gap="medium")
        row2 = st.columns([1.2, 1.2, 1.0, 0.55], gap="medium")

        age_range = spec.age_range
        with row1[0]:
            if bounds.has_age:
                age_range = st.slider(
                    "Age",
                    min_value=bounds.age_min,
                    max_value=bounds.age_max,
                    value=spec.age_range or (bounds.age_min, bounds.age_max),
                    key=f"{key_prefix}flt_age",
                    help="Customer age in years.",
                )

        limit_range = spec.limit_range
        with row1[1]:
            if bounds.has_limit:
                limit_range = st.slider(
                    "Credit limit (NT$)",
                    min_value=float(bounds.limit_min),
                    max_value=float(bounds.limit_max),
                    value=spec.limit_range or (bounds.limit_min, bounds.limit_max),
                    step=10_000.0,
                    format="%.0f",
                    key=f"{key_prefix}flt_limit",
                    help="Granted credit limit. This is a credit limit, not income.",
                )

        utilisation_range = spec.utilisation_range
        with row1[2]:
            if bounds.has_utilisation:
                utilisation_range = st.slider(
                    "Utilisation (balance ÷ limit)",
                    min_value=float(bounds.utilisation_min),
                    max_value=float(bounds.utilisation_max),
                    value=spec.utilisation_range
                    or (bounds.utilisation_min, bounds.utilisation_max),
                    step=0.05,
                    format="%.2f",
                    key=f"{key_prefix}flt_util",
                    help=(
                        "Derived measure. Negative values mean the account is in "
                        "credit; values above 1.0 mean it is over its limit."
                    ),
                )

        education = spec.education
        with row2[0]:
            if bounds.education_options:
                education = tuple(
                    st.multiselect(
                        "Education",
                        options=list(bounds.education_options),
                        default=list(spec.education or bounds.education_options),
                        key=f"{key_prefix}flt_education",
                        help=(
                            "Undocumented source codes are grouped as 'Unknown'. "
                            "Used descriptively, not as a risk driver."
                        ),
                    )
                )

        marriage = spec.marriage
        with row2[1]:
            if bounds.marriage_options:
                marriage = tuple(
                    st.multiselect(
                        "Marital status",
                        options=list(bounds.marriage_options),
                        default=list(spec.marriage or bounds.marriage_options),
                        key=f"{key_prefix}flt_marriage",
                        help="Used descriptively, not as a risk driver.",
                    )
                )

        delinquency = spec.delinquency
        with row2[2]:
            options = list(DELINQUENCY_OPTIONS)
            delinquency = st.selectbox(
                "Delinquency status",
                options=options,
                index=options.index(spec.delinquency)
                if spec.delinquency in options
                else 0,
                format_func=lambda key: DELINQUENCY_OPTIONS[key],
                key=f"{key_prefix}flt_delinquency",
                help=(
                    "'Currently' uses the most recent observed month; 'ever' uses "
                    "any of the six months. Counts documented delay codes only."
                ),
            )

        with row2[3]:
            st.markdown(
                '<div style="height:1.55rem"></div>', unsafe_allow_html=True
            )
            if st.button(
                "Reset",
                key=f"{key_prefix}flt_reset",
                width="stretch",
                help="Return to the full dataset.",
            ):
                request_reset()
                st.rerun()

    # Widget state is the source of truth, so this return value simply mirrors
    # what the controls now hold. No separate copy is stored.
    return FilterSpec(
        age_range=tuple(age_range) if age_range else None,
        limit_range=tuple(limit_range) if limit_range else None,
        utilisation_range=tuple(utilisation_range) if utilisation_range else None,
        education=education,
        marriage=marriage,
        sex=spec.sex,
        delinquency=delinquency,
    )


def render_active_filters(result: FilterResult) -> None:
    """Show which constraints are in effect, if any.

    Args:
        result: The applied filter result.
    """
    if not result.is_filtered:
        return

    chips = "".join(
        f'<span class="fr-insight-metric" style="font-family:inherit;'
        f'font-weight:520">{escape(label)}</span>'
        for label in result.applied
    )
    st.markdown(
        f'<div style="display:flex;gap:0.4rem;flex-wrap:wrap;align-items:center;'
        f'margin:0.55rem 0 0.2rem 0">'
        f'<span style="font-size:0.7rem;color:{PALETTE["text_muted"]};'
        f'text-transform:uppercase;letter-spacing:0.7px">Active</span>'
        f"{chips}</div>",
        unsafe_allow_html=True,
    )

    if result.skipped:
        st.caption(
            "Not applied (data unavailable): " + "; ".join(result.skipped)
        )
