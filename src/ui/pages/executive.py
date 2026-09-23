"""
Executive Overview.

Ordered as an analytical narrative rather than a wall of charts:

1. Portfolio snapshot - what is in front of us
2. Headline KPIs - what is happening
3. Credit Behaviour Overview - what changed across the observed periods
4. Key Intelligence - why it might be happening
5. Risk Signals - which signals matter
6. Opportunities - where the upside is
7. Recommended Next Steps - what to investigate

A reader should be able to answer "what is happening and what should I look at"
within about thirty seconds of the page loading.
"""

from __future__ import annotations

import pandas as pd

import streamlit as st

from src.analytics import InsufficientData, category_outcome_rates
from src.schema import DATASET_NAME, DATASET_PERIOD, Role
from src.trends import TrendResult, balance_trend, repayment_trend, utilisation_trend
from src.ui import charts, components, insight_cards, kpi_cards
from src.ui.context import PageContext
from src.ui.navigation import ai_status


def _snapshot_meta(context: PageContext) -> list[tuple[str, str]]:
    """Build the header metadata items."""
    meta = [
        ("Dataset", DATASET_NAME.replace("Default of ", "")),
        ("Observation period", DATASET_PERIOD),
        ("Customers", f"{context.n_selected:,}"),
    ]
    if context.is_filtered:
        meta.append(
            ("Of portfolio", f"{context.filters.share_pct:.1f}% of {context.filters.n_total:,}")
        )
    return meta


def _render_hero(context: PageContext) -> None:
    """Render the primary credit-behaviour visualisation."""
    registry = context.registry
    data = context.data

    bill = balance_trend(data, registry, "mean")
    payment = repayment_trend(data, registry, "mean")
    utilisation = utilisation_trend(data, registry, "mean")

    figure = charts.credit_behaviour_overview(
        bill_trend=bill if isinstance(bill, TrendResult) else None,
        payment_trend=payment if isinstance(payment, TrendResult) else None,
        utilisation_trend=utilisation if isinstance(utilisation, TrendResult) else None,
    )
    components.chart(figure, key="exec_hero")

    if isinstance(bill, TrendResult):
        st.caption(
            f"{bill.describe()} "
            "The six columns are the six monthly observations held for each "
            "customer, not six years. No forecast is shown: six periods cannot "
            "support one."
        )


def _render_movement(context: PageContext) -> None:
    """Show what changed across the observed periods, as a compact table."""
    registry = context.registry
    data = context.data

    rows: list[dict[str, str]] = []
    series = (
        ("Avg statement balance", balance_trend(data, registry, "mean"), "NT$"),
        ("Avg payment received", repayment_trend(data, registry, "mean"), "NT$"),
        ("Avg utilisation", utilisation_trend(data, registry, "mean"), "%"),
    )

    from src.trends import delinquency_trend

    delinquency = delinquency_trend(data, registry)
    all_series = list(series) + [
        ("Share behind on payment", delinquency, "%")
    ]

    for label, trend, unit in all_series:
        if not isinstance(trend, TrendResult):
            continue
        first = trend.first_value
        last = trend.last_value
        change = last - first
        if unit == "NT$":
            first_text, last_text = f"{first:,.0f}", f"{last:,.0f}"
            change_text = f"{change:+,.0f}"
        else:
            first_text, last_text = f"{first:.2f}%", f"{last:.2f}%"
            change_text = f"{change:+.2f} pp"
        pct = (
            f"{trend.total_change_pct:+.1f}%"
            if trend.total_change_pct is not None
            else "n/a"
        )
        rows.append(
            {
                "Measure": label,
                f"{trend.points[0].month_label} (oldest)": first_text,
                f"{trend.points[-1].month_label} (latest)": last_text,
                "Change": change_text,
                "Relative change": pct,
                "Direction": trend.direction.value.title(),
            }
        )

    if not rows:
        components.unavailable(
            "Period movement",
            "The monthly panel columns are not available in this selection.",
        )
        return

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Measure": st.column_config.TextColumn(width="medium"),
            "Direction": st.column_config.TextColumn(width="small"),
        },
    )
    st.caption(
        "Movement is measured between the oldest and most recent observed month "
        "for the customers in the current selection. Every value is computed from "
        "the data; none is projected."
    )


def _render_delinquency_snapshot(context: PageContext) -> None:
    """Default rate by most recent payment status - the sharpest single split."""
    target = context.target_column
    data = context.data

    if target is None or "current_delinquency" not in data.columns:
        components.unavailable(
            "Delinquency breakdown",
            "Requires the target and the monthly repayment-status columns.",
        )
        return

    working = data.copy()
    working["_delay"] = pd.cut(
        pd.to_numeric(working["current_delinquency"], errors="coerce"),
        bins=[-0.5, 0.5, 1.5, 2.5, 100],
        labels=["No delay", "1 month", "2 months", "3+ months"],
        ordered=True,
    )
    outcome = category_outcome_rates(working, "_delay", target, min_group_size=30)

    if isinstance(outcome, InsufficientData):
        components.unavailable("Delinquency breakdown", outcome.message)
        return

    figure = charts.band_rate_and_size(
        outcome.table.sort_index(),
        band_label="Payment delay in the most recent observed month",
        title="Default rate by recent payment delay",
        subtitle=(
            "Delay length uses the documented status codes. Bars show the customer "
            "count; the line shows the observed default rate."
        ),
        overall_rate=outcome.overall_rate,
    )
    components.chart(figure, key="exec_delinquency")


def render(context: PageContext) -> None:
    """Render the Executive Overview page.

    Args:
        context: The per-run page context.
    """
    label, tone = ai_status()

    components.page_header(
        title="Financial Risk Intelligence",
        description=(
            "Monitor credit behaviour, identify risk signals, and convert financial "
            "data into actionable insights."
        ),
        meta=_snapshot_meta(context),
        status=(label, tone),
    )

    if context.is_empty:
        from src.ui.filter_bar import RESET_FLAG

        components.empty_state(on_reset=RESET_FLAG)
        return

    if context.filters.n_selected < 100 and context.is_filtered:
        components.notice(
            "Small selection",
            f"Only {context.n_selected:,} customers match the current filters. "
            "Comparative findings need at least 100 to be meaningful, so some "
            "sections below may be unavailable.",
            icon="◍",
            tone="attention",
        )

    # ---------------- 2. Headline KPIs ---------------- #
    components.section(
        "Portfolio Snapshot",
        "Headline measures for the current selection."
        + (
            " Change indicators compare this selection with the full portfolio."
            if context.is_filtered
            else " No change indicators are shown: this dataset is a single extract "
            "with no prior period to compare against."
        ),
    )
    kpi_cards.render_kpi_row(
        context.kpis,
        kpi_cards.EXECUTIVE_KPIS,
        baseline=context.baseline_kpis,
        baseline_label="full portfolio",
    )

    # ---------------- 3. Hero visualisation ---------------- #
    components.section(
        "What changed across the observed periods?",
        "Statement balances, payments received and utilisation over the six "
        "monthly observations held for each customer.",
    )
    _render_hero(context)

    with st.expander("Period-by-period movement", expanded=False):
        _render_movement(context)

    # ---------------- 4. Key intelligence ---------------- #
    components.section(
        "Key Intelligence",
        "Findings derived automatically from the current selection. Each is "
        "measured, not written in advance, and carries the values it came from.",
    )
    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        insight_cards.render_key_intelligence(context.findings, limit=3)
    with right:
        _render_delinquency_snapshot(context)

    # ---------------- 5 & 6. Signals and opportunities ---------------- #
    insight_cards.render_signal_set(context.signals)

    # ---------------- 7. Next steps ---------------- #
    components.section(
        "Recommended Next Steps",
        "Each finding expanded into the full chain: what the data shows, what it "
        "indicates, why it matters, and what an analyst could examine next. These "
        "are suggestions for investigation, not financial advice.",
    )
    insight_cards.render_action_chains(context.findings, limit=3)

    components.spacer(0.4)
    st.caption(
        "All figures on this page are computed from the selected records. Risk "
        "signals are observed historical measurements, not model predictions: no "
        "predictive model is active at this stage of the project."
    )
