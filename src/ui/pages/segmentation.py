"""
Customer Segmentation.

Page order follows the analytical argument: what was produced, how it was
produced, why this K, what the segments are, where they sit in behaviour space,
what outcome each experienced, and what that means.

Two deliberate design choices
-----------------------------
**The model is fitted on the whole portfolio, not on the filtered selection.**
Segments are a property of the portfolio. Refitting on every filter change would
make them move underfoot and mean nothing could be compared between views. When
filters are active the page keeps the fixed segments and shows how the selection
distributes across them, which is the more useful question anyway.

**Nothing here is a prediction.** Clustering used no outcome column. The observed
default rate per segment is measured afterwards and is descriptive only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import streamlit as st

from src.kpi_engine import format_currency
from src.segmentation import (
    CANDIDATE_FEATURES,
    FEATURE_FORMATS,
    FEATURE_LABELS,
    FEATURE_SET_RATIONALE,
    FORBIDDEN_FEATURES,
    SegmentationResult,
    SegmentationUnavailable,
    methodology_note,
    segment_interpretation,
)
from src.ui import charts, components
from src.ui.context import PageContext
from src.ui.theme import CATEGORICAL_SEQUENCE, PALETTE, with_alpha

#: Session-state key holding the user's K override.
K_OVERRIDE_KEY = "seg_k_override"


def _format_value(feature: str, value: float | None) -> str:
    """Format a profile metric according to its declared style."""
    if value is None or not np.isfinite(value):
        return "n/a"
    style = FEATURE_FORMATS.get(feature, "decimal")
    if style == "currency":
        return format_currency(value)
    if style == "currency_signed":
        sign = "+" if value >= 0 else "-"
        return f"{sign}{format_currency(abs(value))}/mo"
    if style == "percent":
        return f"{value:.1%}"
    if style == "decimal3":
        return f"{value:.3f}"
    return f"{value:.2f}"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _render_kpis(result: SegmentationResult, context: PageContext) -> None:
    """Executive segmentation KPIs."""
    selection = result.k_selection
    score = selection.score_for(result.k)

    cards = [
        (
            "◫",
            "Customers Segmented",
            f"{result.n_segmented:,}",
            "Every customer receives a segment",
            "info",
        ),
        (
            "◈",
            "Behavioural Segments",
            f"{len(result.profiles)}",
            "Distinct groups discovered",
            "accent",
        ),
        (
            "❖",
            "Selected K",
            f"{result.k}",
            (
                "Measured recommendation"
                if result.k == selection.recommended_k
                else f"Manual override · recommended {selection.recommended_k}"
            ),
            "attention" if result.k != selection.recommended_k else "positive",
        ),
        (
            "◉",
            "Silhouette Score",
            f"{result.silhouette:.4f}",
            (
                "Sampled" if selection.was_sampled else "All rows"
            )
            + " · higher is better",
            "positive" if result.silhouette >= 0.35 else "attention",
        ),
        (
            "▦",
            "Features Used",
            f"{result.matrix.n_features}",
            "Behavioural only · no outcome, no demographics",
            "info",
        ),
    ]

    for column, (icon, label, value, note, accent_key) in zip(
        st.columns(len(cards), gap="small"), cards
    ):
        accent = PALETTE.get(accent_key, PALETTE["info"])
        with column:
            st.markdown(
                f'<div class="fr-kpi" style="--fr-accent-bar:{accent}">'
                f'<div class="fr-kpi-head">'
                f'<span class="fr-kpi-icon" style="color:{accent}">{icon}</span>'
                f'<span class="fr-kpi-label">{components.escape(label)}</span></div>'
                f'<div class="fr-kpi-value">{components.escape(value)}</div>'
                f'<div class="fr-kpi-note">{components.escape(note)}</div></div>',
                unsafe_allow_html=True,
            )


def _render_methodology(result: SegmentationResult) -> None:
    """How the segments were created."""
    st.markdown(
        f'<div class="fr-panel"><div class="fr-insight-body">'
        f"{components.escape(methodology_note(result))}</div></div>",
        unsafe_allow_html=True,
    )
    components.spacer(0.5)

    left, right = st.columns(2, gap="medium")

    with left:
        with st.expander("Features used, and why these", expanded=False):
            st.caption(FEATURE_SET_RATIONALE)
            rows = [
                {
                    "Feature": FEATURE_LABELS.get(f, f).replace("Avg ", ""),
                    "Column": f,
                    "In model": "Yes" if f in result.matrix.feature_names else "No",
                }
                for f in CANDIDATE_FEATURES
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            if result.matrix.excluded:
                st.caption(
                    "Excluded at runtime: "
                    + "; ".join(f"{k} — {v}" for k, v in result.matrix.excluded.items())
                )

    with right:
        with st.expander("Excluded by policy, and why", expanded=False):
            st.caption(
                "Two groups of column are blocked from the feature matrix outright. "
                "The outcome and anything derived from it would make the segments a "
                "restatement of the answer rather than a description of behaviour. "
                "Demographic attributes are excluded on fairness grounds, "
                "consistently with the rest of this project. Attempting to pass "
                "either raises an error rather than silently succeeding."
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "Blocked column": sorted(FORBIDDEN_FEATURES),
                        "Reason": [
                            (
                                "Outcome leakage"
                                if "default" in c
                                else "Identifier / bookkeeping"
                                if c in {
                                    "client_id",
                                    "is_duplicate_excluding_id",
                                    "has_extreme_value",
                                }
                                else "Protected or demographic attribute"
                            )
                            for c in sorted(FORBIDDEN_FEATURES)
                        ],
                    }
                ),
                width="stretch",
                hide_index=True,
                height=260,
            )

    if result.matrix.imputations:
        components.spacer(0.4)
        with st.expander("Missing-value treatment", expanded=False):
            for info in result.matrix.imputations:
                st.markdown(
                    f"**`{info.column}`** — {info.n_missing:,} customers "
                    f"({info.pct_missing:.2f}%) had no value. "
                    f"Filled with the {info.method} ({info.fill_value:.4f})."
                )
                st.caption(info.note)


def _render_k_selection(result: SegmentationResult) -> None:
    """K selection diagnostics with the measured rationale."""
    selection = result.k_selection

    components.chart(
        charts.k_selection_chart(
            selection.to_frame(),
            recommended_k=result.k,
            elbow_k=selection.elbow_k,
            was_sampled=selection.was_sampled,
            sample_size=selection.silhouette_sample_size,
        ),
        key="seg_k_selection",
    )

    st.markdown(
        f'<div class="fr-panel tight"><div class="fr-insight-body">'
        f"<b>Measured rationale.</b> {components.escape(selection.rationale)}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    components.spacer(0.5)
    left, right = st.columns([1.1, 1], gap="medium")

    with left:
        frame = selection.to_frame()
        display = frame[
            ["k", "silhouette", "davies_bouldin", "inertia", "smallest_cluster_share_pct"]
        ].rename(
            columns={
                "k": "K",
                "silhouette": "Silhouette ↑",
                "davies_bouldin": "Davies-Bouldin ↓",
                "inertia": "Inertia",
                "smallest_cluster_share_pct": "Smallest segment %",
            }
        )
        st.dataframe(display.round(4), width="stretch", hide_index=True)
        st.caption(
            "↑ higher is better, ↓ lower is better. A very small 'smallest segment' "
            "means K-Means isolated a handful of outliers rather than finding a "
            "usable group."
        )

    with right:
        if result.scaler_comparison is not None:
            components.chart(
                charts.scaler_comparison_chart(
                    result.scaler_comparison.to_frame(),
                    result.scaler_comparison.chosen,
                ),
                key="seg_scaler_cmp",
            )

    if result.scaler_comparison is not None:
        with st.expander("Scaler comparison detail", expanded=False):
            st.dataframe(
                result.scaler_comparison.best_row_per_scaler(),
                width="stretch",
                hide_index=True,
            )
            st.caption(result.scaler_comparison.rationale)


def _render_k_control(result: SegmentationResult) -> None:
    """Let the analyst explore other K values."""
    selection = result.k_selection
    available = [s.k for s in selection.scores]
    if len(available) < 2:
        return

    with st.expander("Explore a different number of segments", expanded=False):
        st.caption(
            "The default is the measured recommendation. Changing K here refits the "
            "model on the same feature matrix so alternatives can be inspected "
            "directly. The diagnostics above show what each choice costs."
        )
        chosen = st.select_slider(
            "Number of segments (K)",
            options=available,
            value=result.k if result.k in available else selection.recommended_k,
            key=K_OVERRIDE_KEY,
        )
        if chosen != selection.recommended_k:
            st.caption(
                f"K={chosen} is a manual override. The measured recommendation is "
                f"K={selection.recommended_k}."
            )


def _render_segments(result: SegmentationResult, context: PageContext) -> None:
    """Segment distribution and the profile table."""
    left, right = st.columns([1, 1.15], gap="medium")

    with left:
        components.chart(
            charts.segment_distribution_chart(result.profiles), key="seg_distribution"
        )

    with right:
        components.chart(
            charts.segment_profile_comparison(
                result.profiles, result.matrix.feature_names, FEATURE_LABELS
            ),
            key="seg_profile_cmp",
        )

    if result.has_degenerate_segment:
        components.notice(
            "One segment is very small",
            "At least one segment holds under 1% of customers, which usually means "
            "K-Means isolated a cluster of outliers rather than finding a genuine "
            "behavioural group. Treat it as a curiosity rather than a population to "
            "act on, and consider a lower K.",
            icon="◍",
            tone="attention",
        )

    largest = max(result.profiles, key=lambda p: p.size)
    if not largest.traits and largest.share_pct > 50:
        components.notice(
            f"The largest segment holds {largest.share_pct:.1f}% of customers and has "
            "no distinguishing behaviour",
            "This is a genuine limitation of the measured result, not a display "
            "problem. Silhouette favoured this K, but it leaves a majority of the "
            "portfolio in one undifferentiated group that sits near the average on "
            "every clustering feature. Credit behaviour here is largely continuous "
            "rather than naturally clustered, so a clean separation of the whole "
            "book does not exist. Higher K values split this group further but score "
            "worse and produce thin segments; the K selector above lets you compare.",
            icon="◍",
            tone="attention",
        )

    components.spacer(0.6)
    components.section(
        "Segment profile table",
        "Measured averages per segment. The observed default rate is the rightmost "
        "column and was calculated after clustering.",
        rule=False,
    )
    st.dataframe(
        result.profile_frame(),
        width="stretch",
        hide_index=True,
        column_config={"Segment": st.column_config.TextColumn(width="large")},
    )


def _render_behaviour_map(result: SegmentationResult) -> None:
    """The PCA projection."""
    components.chart(
        charts.segment_pca_map(result.projection, result.pca_explained_variance),
        key="seg_pca",
    )

    variance_rows = [
        {
            "Component": f"PC{i + 1}",
            "Variance explained": f"{v * 100:.2f}%",
            "Cumulative": f"{sum(result.pca_explained_variance[: i + 1]) * 100:.2f}%",
        }
        for i, v in enumerate(result.pca_explained_variance)
    ]
    left, right = st.columns([1, 2], gap="medium")
    with left:
        st.dataframe(pd.DataFrame(variance_rows), width="stretch", hide_index=True)
    with right:
        st.caption(
            f"The clustering ran in {result.matrix.n_features}-dimensional space. "
            f"These two components retain {result.pca_total_variance_pct:.1f}% of the "
            f"variance, so roughly "
            f"{100 - result.pca_total_variance_pct:.1f}% of the structure K-Means "
            "used is not visible on this plot. Points that appear to overlap here may "
            "be well separated on the dimensions the projection discards, so the map "
            "is a navigational aid rather than evidence about cluster quality — the "
            "silhouette and Davies-Bouldin scores above serve that purpose."
        )


def _render_observed_risk(result: SegmentationResult, context: PageContext) -> None:
    """Observed default rate by segment."""
    overall = context.portfolio_kpis.raw("observed_default_rate")
    components.chart(
        charts.segment_observed_risk_chart(result.profiles, overall), key="seg_risk"
    )

    st.markdown(
        f'<div class="fr-notice" style="text-align:left;padding:0.85rem 1rem;'
        f'border-color:{with_alpha(PALETTE["info"], 0.35)}">'
        f'<div class="fr-notice-body" style="max-width:none;margin:0">'
        f"<b>How to read this.</b> The outcome column was excluded from the "
        f"clustering features, so these rates are measured properties of the "
        f"customers who happened to group together, not effects of segment "
        f"membership and not predictions. A segment with an elevated observed rate "
        f"is an <b>observed-risk segment</b>: the behaviour came first, and the rate "
        f"is what those customers went on to experience historically."
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _render_interpretation(result: SegmentationResult, context: PageContext) -> None:
    """Four-part interpretation per segment."""
    overall = context.portfolio_kpis.raw("observed_default_rate")
    ordered = sorted(
        result.profiles,
        key=lambda p: (p.observed_default_rate_pct or 0.0),
        reverse=True,
    )

    for profile in ordered:
        interpretation = segment_interpretation(profile, overall)
        colour = CATEGORICAL_SEQUENCE[profile.cluster_id % len(CATEGORICAL_SEQUENCE)]

        traits = "".join(
            f'<span class="fr-insight-metric">{components.escape(t)}</span>'
            for t in profile.traits
        ) or (
            '<span class="fr-insight-metric">no distinguishing trait</span>'
        )

        steps = [
            ("FACT", interpretation["fact"], PALETTE["info"]),
            ("BEHAVIOUR", interpretation["behavior"], PALETTE["accent"]),
            ("OBSERVED RISK", interpretation["observed_risk"], PALETTE["attention"]),
            ("IMPLICATION", interpretation["implication"], PALETTE["positive"]),
        ]
        rows = []
        for index, (tag, text, tag_colour) in enumerate(steps):
            rows.append(
                f'<div class="fr-chain-step">'
                f'<div class="fr-chain-tag" style="color:{tag_colour};flex-basis:106px">'
                f"{tag}</div>"
                f'<div class="fr-chain-text">{components.escape(text)}</div></div>'
            )
            if index < len(steps) - 1:
                rows.append('<div class="fr-chain-sep"></div>')

        st.markdown(
            f'<div class="fr-chain" style="border-left:2px solid {colour}">'
            f'<div class="fr-insight-head" style="margin-bottom:0.55rem">'
            f'<div class="fr-insight-title">'
            f"{components.escape(profile.display_name)}</div>"
            f'<div style="display:flex;gap:0.3rem;flex-shrink:0">'
            f"{components.badge(f'{profile.share_pct:.1f}% of portfolio', 'neutral')}"
            f"</div></div>"
            f'<div style="display:flex;gap:0.35rem;flex-wrap:wrap;'
            f'margin-bottom:0.6rem">{traits}</div>'
            f'{"".join(rows)}</div>',
            unsafe_allow_html=True,
        )


def _render_filtered_distribution(
    result: SegmentationResult, context: PageContext
) -> None:
    """Show how the filtered selection distributes across the fixed segments."""
    if not context.is_filtered:
        return

    annotated = result.assign_to(context.data)
    counts = annotated["segment"].value_counts().sort_index()
    if counts.sum() == 0:
        return

    portfolio = {p.display_name: p.share_pct for p in result.profiles}
    rows = []
    for name, count in counts.items():
        share = count / len(annotated) * 100.0 if len(annotated) else 0.0
        base = portfolio.get(str(name), 0.0)
        rows.append(
            {
                "Segment": str(name),
                "Customers in selection": int(count),
                "% of selection": round(share, 2),
                "% of portfolio": round(base, 2),
                "Difference (pp)": round(share - base, 2),
            }
        )

    components.section(
        "Your filtered selection across these segments",
        "The model stays fitted on the whole portfolio so segments do not shift as "
        "filters change. This table shows where the current selection sits within "
        "those fixed segments.",
    )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        f"{len(annotated):,} customers in the current selection. A positive "
        "difference means the selection is over-represented in that segment relative "
        "to the portfolio."
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def render(context: PageContext, segmentation=None) -> None:
    """Render the Customer Segmentation page.

    Args:
        context: The per-run page context.
        segmentation: A fitted :class:`SegmentationResult` or
            :class:`SegmentationUnavailable`. Supplied by the caller so the
            expensive fit is cached across reruns.
    """
    components.page_header(
        title="Customer Segmentation",
        description="Behaviour-based clustering of the credit portfolio.",
        meta=[
            ("Portfolio", f"{len(context.full_data):,} customers"),
            (
                "Method",
                f"K-Means · K={segmentation.k}"
                if isinstance(segmentation, SegmentationResult)
                else "K-Means",
            ),
        ],
        status=("Unsupervised · no outcome used", "info"),
    )

    if segmentation is None:
        components.unavailable(
            "Segmentation",
            "The clustering model has not been produced for this session.",
        )
        return

    if isinstance(segmentation, SegmentationUnavailable):
        components.notice(
            "Segmentation unavailable",
            segmentation.reason,
            icon="⊘",
            tone="attention",
        )
        if segmentation.missing_features:
            st.caption(
                "Missing features: " + ", ".join(segmentation.missing_features)
            )
        return

    result = segmentation

    components.section(
        "Segmentation Overview",
        "Measured properties of the fitted model. Clustering ran on the full "
        f"{len(context.full_data):,}-customer portfolio, so segments are stable "
        "regardless of the filters applied elsewhere in the dashboard.",
    )
    components.safe_render("Segmentation KPIs", _render_kpis, result, context)

    components.section(
        "How Segments Were Created",
        "The method, the features, and what was deliberately kept out.",
    )
    components.safe_render("Methodology", _render_methodology, result)

    components.section(
        "K Selection",
        "How many segments the data actually supports, measured rather than assumed.",
    )
    components.safe_render("K selection", _render_k_selection, result)
    components.safe_render("K control", _render_k_control, result)

    components.section(
        "Portfolio Segments",
        "Size and behavioural composition of each segment.",
    )
    components.safe_render("Portfolio segments", _render_segments, result, context)
    components.safe_render(
        "Filtered distribution", _render_filtered_distribution, result, context
    )

    components.section(
        "Behaviour Map",
        "Where segments sit relative to one another in the clustering space.",
    )
    components.safe_render("Behaviour map", _render_behaviour_map, result)

    components.section(
        "Segment Risk Profile",
        "What outcome each behavioural segment historically experienced.",
    )
    components.safe_render("Observed risk", _render_observed_risk, result, context)

    components.section(
        "Analytical Interpretation",
        "Each segment as a chain: what the data shows, what characterises the group, "
        "what outcome it recorded, and why it matters analytically. Ordered by "
        "observed default rate, highest first.",
    )
    components.safe_render("Interpretation", _render_interpretation, result, context)

    components.spacer(0.4)
    st.caption(
        "Segmentation is unsupervised and descriptive. It assigns every customer to "
        "a behavioural group; it does not score, rank or predict any individual. "
        "Observed default rates are historical measurements attached to segments "
        "after clustering, and no causal relationship between segment membership and "
        "outcome is claimed."
    )
