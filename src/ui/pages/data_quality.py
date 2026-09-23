"""
Data Quality.

Reports the quality assessment produced during the pipeline run. The important
editorial point, stated on the page itself: this dataset has **no missing values
and no exact duplicate rows**, so no missing-value problem is manufactured to
fill the screen. What is surfaced instead are the semantic defects that genuinely
exist - undocumented category codes, internal contradictions, and rows that
duplicate once the surrogate key is dropped.

The assessment always describes the **source data as it arrived**, which is why
it is computed before cleaning and is not affected by the dashboard filters.
"""

from __future__ import annotations

import pandas as pd

import streamlit as st

from src.data_quality import SCORE_METHODOLOGY, Severity
from src.ui import components
from src.ui.context import PageContext
from src.ui.theme import PALETTE, severity_colour, with_alpha
from src.visualization import missing_values_chart, quality_score_gauge

#: Severity display order, most serious first.
SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
)


def _render_score(context: PageContext) -> None:
    """Score gauge with the per-dimension breakdown."""
    quality = context.pipeline.quality
    score = quality.quality_score

    left, right = st.columns([1, 1.5], gap="medium")

    with left:
        components.chart(quality_score_gauge(score.score), key="dq_gauge")

    with right:
        rows: list[str] = []
        for name, weight in score.dimension_weights.items():
            value = score.dimension_scores.get(name, 0.0)
            contribution = score.contributions.get(name, 0.0)
            width = max(2.0, min(100.0, value * 100.0))
            colour = (
                PALETTE["positive"]
                if value >= 0.98
                else PALETTE["attention"]
                if value >= 0.90
                else PALETTE["risk"]
            )
            rows.append(
                f'<div style="padding:0.42rem 0">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.78rem;margin-bottom:0.28rem">'
                f'<span style="color:{PALETTE["text"]};font-weight:560">'
                f'{name.title()} <span style="color:{PALETTE["text_muted"]};'
                f'font-weight:400">(weight {weight:.0f})</span></span>'
                f'<span style="color:{PALETTE["text_secondary"]};'
                f'font-variant-numeric:tabular-nums">{value:.4f} → '
                f"{contribution:.2f} pts</span></div>"
                f'<div style="height:5px;background:{PALETTE["bg_alt"]};'
                f'border-radius:3px;overflow:hidden">'
                f'<span style="display:block;height:100%;width:{width:.2f}%;'
                f'background:{colour};border-radius:3px"></span></div></div>'
            )
        st.markdown(
            f'<div class="fr-panel">'
            f'<div style="font-size:0.85rem;font-weight:600;color:{PALETTE["text"]};'
            f'margin-bottom:0.5rem">Dimension breakdown</div>'
            f'{"".join(rows)}</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"Overall score {score.score:.2f} / 100 ({score.band}). This is a "
        "project-defined heuristic for tracking quality within this application, "
        "not an industry standard such as DAMA or ISO 8000, and it should not be "
        "quoted as one."
    )


def _render_issues(context: PageContext) -> None:
    """Issue cards grouped by severity."""
    issues = context.pipeline.quality.all_issues
    if not issues:
        components.notice(
            "No quality issues detected",
            "Every check passed on the source data.",
            icon="✓",
        )
        return

    counts = context.pipeline.quality.headline_counts
    chips = "".join(
        components.badge(f"{count} {name}", name)
        for name, count in counts.items()
        if count
    )
    st.markdown(
        f'<div style="display:flex;gap:0.4rem;margin-bottom:0.9rem">{chips}</div>',
        unsafe_allow_html=True,
    )

    for severity in SEVERITY_ORDER:
        group = [issue for issue in issues if issue.severity is severity]
        if not group:
            continue

        colour = severity_colour(severity.value)
        for issue in group:
            recommendation = ""
            if issue.recommendation:
                recommendation = (
                    f'<div class="fr-insight-why"><b>Treatment:</b> '
                    f"{components.escape(issue.recommendation)}</div>"
                )
            column_chip = (
                f'<span class="fr-insight-metric">{components.escape(issue.column)}</span>'
                if issue.column
                else ""
            )
            st.markdown(
                f'<div class="fr-insight" style="--fr-insight-accent:{colour}">'
                f'<div class="fr-insight-head">'
                f'<div class="fr-insight-title">'
                f'{components.escape(issue.check.replace("_", " ").title())}</div>'
                f'<div style="display:flex;gap:0.3rem;flex-shrink:0">'
                f"{components.badge(severity.value, severity.value)}</div></div>"
                f'<div class="fr-insight-body">{components.escape(issue.description)}</div>'
                f'<div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin-top:0.55rem">'
                f"{column_chip}"
                f'<span class="fr-insight-metric">{issue.affected_rows:,} rows '
                f"({issue.affected_percentage:.2f}%)</span></div>"
                f"{recommendation}</div>",
                unsafe_allow_html=True,
            )


def _render_completeness(context: PageContext) -> None:
    """Completeness panel - honest about there being nothing wrong."""
    quality = context.pipeline.quality
    missing = quality.missing

    if missing.is_complete:
        components.notice(
            "The source dataset is fully populated",
            f"All {missing.total_cells:,} cells across {quality.n_columns} columns "
            "contain a value. No imputation was performed, because inventing values "
            "where none are missing would add noise for no benefit. This is why the "
            "quality engine checks semantic validity rather than only counting blanks.",
            icon="✓",
        )
        return

    components.chart(
        missing_values_chart(missing.missing_by_column, quality.n_rows), key="dq_missing"
    )


def _render_outliers(context: PageContext) -> None:
    """Outlier panel, with the reason they are not scored."""
    outliers = context.pipeline.quality.outliers
    affected = outliers.most_affected(8)

    if not affected:
        components.unavailable("Outlier analysis", "No numeric column qualified.")
        return

    rows = [
        {
            "Column": summary.column,
            "Values": f"{summary.n_valid:,}",
            "IQR-flagged": f"{summary.iqr_outliers:,}",
            "Share": f"{summary.iqr_percentage:.2f}%",
            "Z-score flagged": f"{summary.zscore_outliers:,}",
            "Max": f"{summary.max_value:,.0f}",
            "Skew": f"{summary.skewness:.2f}",
        }
        for summary in affected
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(outliers.method_note)

    st.markdown(
        f'<div class="fr-notice" style="text-align:left;padding:0.85rem 1rem;'
        f'border-color:{with_alpha(PALETTE["info"], 0.35)}">'
        f'<div class="fr-notice-body" style="max-width:none;margin:0">'
        f"<b>Outliers do not reduce the quality score.</b> In a credit portfolio a "
        f"NT$1,000,000 limit or a very large statement balance is usually a real "
        f"premium customer, not an error. Penalising a naturally skewed financial "
        f"distribution would mislabel genuine data as dirty, so extreme values are "
        f"reported for analyst judgement and left out of the score."
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _render_cleaning(context: PageContext) -> None:
    """The cleaning audit trail."""
    cleaning = context.pipeline.cleaning
    st.caption(cleaning.summary())

    for step in cleaning.steps:
        acted = step.action_taken
        colour = PALETTE["positive"] if acted else PALETTE["text_faint"]
        added = (
            f'<div style="display:flex;gap:0.35rem;flex-wrap:wrap;margin-top:0.5rem">'
            + "".join(
                f'<span class="fr-insight-metric">+ {components.escape(c)}</span>'
                for c in step.columns_added
            )
            + "</div>"
            if step.columns_added
            else ""
        )
        st.markdown(
            f'<div class="fr-insight" style="--fr-insight-accent:{colour}">'
            f'<div class="fr-insight-head">'
            f'<div class="fr-insight-title">'
            f'{components.escape(step.name.replace("_", " ").title())}</div>'
            f"{components.badge('applied' if acted else 'no action needed', 'positive' if acted else 'neutral')}"
            f"</div>"
            f'<div class="fr-insight-body">{components.escape(step.description)}</div>'
            f"{added}"
            f'<div class="fr-insight-why"><b>Why:</b> '
            f"{components.escape(step.rationale)}</div></div>",
            unsafe_allow_html=True,
        )


def render(context: PageContext) -> None:
    """Render the Data Quality page.

    Args:
        context: The per-run page context.
    """
    quality = context.pipeline.quality
    components.page_header(
        title="Data Quality",
        description=(
            "Assessment of the source dataset as it arrived, before any cleaning. "
            "Not affected by the dashboard filters."
        ),
        meta=[
            ("Quality score", f"{quality.quality_score.score:.2f} / 100"),
            ("Band", quality.quality_score.band),
            ("Issues found", f"{len(quality.all_issues)}"),
        ],
    )

    components.section(
        "Quality Score",
        "Four weighted dimensions. Outliers are assessed and reported but "
        "deliberately excluded from the score.",
    )
    _render_score(context)

    components.section("Completeness", "Missing-value assessment.")
    _render_completeness(context)

    components.section(
        "Findings",
        "The defects that genuinely exist in this dataset, with the treatment "
        "applied to each.",
    )
    _render_issues(context)

    components.section(
        "Extreme Values",
        "Reported for judgement, not penalised.",
    )
    _render_outliers(context)

    components.section(
        "Cleaning Audit Trail",
        "Every transformation applied to the source data, with the reasoning behind "
        "it. No row was removed.",
    )
    _render_cleaning(context)

    with st.expander("Scoring methodology in full", expanded=False):
        st.code(SCORE_METHODOLOGY, language="text")
