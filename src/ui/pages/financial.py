"""
Financial Analytics workspace.

Four tabs, each answering a distinct question:

* **Credit Behaviour** - how is credit used, and how does that relate to outcome?
* **Payment Behaviour & Delinquency** - how do customers repay, and who falls behind?
* **Driver Analysis** - which measures are most strongly associated with default?
* **Customer Profile** - descriptive composition, for fairness review not targeting.

Every visual reads from the same filtered selection supplied in the context, so a
filter change moves the whole page at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import streamlit as st

from src.analytics import (
    InsufficientData,
    category_outcome_rates,
    compare_groups,
    correlation_matrix,
    cross_tabulate,
    distribution_bins,
    numeric_outcome_comparison,
    segment_profile,
    summarise_numeric,
)
from src.config import settings
from src.drivers import (
    CAUSATION_CAVEAT,
    DriverAnalysisResult,
    describe_driver,
    run_driver_analysis,
)
from src.filtering import (
    assign_utilisation_band,
    limit_band_documentation,
    utilisation_band_documentation,
)
from src.schema import DATASET_CURRENCY, Role
from src.ui import charts, components, insight_cards
from src.ui.context import PageContext
from src.visualization import correlation_heatmap


# --------------------------------------------------------------------------- #
# Tab 1: credit behaviour
# --------------------------------------------------------------------------- #


def _utilisation_distribution(context: PageContext) -> None:
    """Distribution of utilisation with the configured threshold marked."""
    data = context.data
    if "utilisation_latest" not in data.columns:
        components.unavailable(
            "Utilisation distribution", "The utilisation feature is not available."
        )
        return

    # Trim the extreme right tail for display only; the underlying data is intact.
    values = pd.to_numeric(data["utilisation_latest"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    upper = float(values.quantile(0.995)) if values.notna().any() else 1.0
    display = data.loc[values <= upper]

    bins = distribution_bins(display, "utilisation_latest", bins=40)
    if isinstance(bins, InsufficientData):
        components.unavailable("Utilisation distribution", bins.message)
        return

    threshold = settings.high_utilisation_threshold
    median = float(values.median()) if values.notna().any() else None

    figure = charts.distribution_with_threshold(
        bins,
        title="How is credit utilisation distributed?",
        x_title="Utilisation (statement balance ÷ credit limit)",
        threshold=threshold,
        threshold_label=f"High-utilisation threshold {threshold:.0%}",
        median_value=median,
        is_percentage=True,
    )
    components.chart(figure, key="fin_util_dist")
    st.caption(
        f"Display trimmed at the 99.5th percentile ({upper:.0%}) so the bulk of the "
        "distribution stays readable; no records were removed from any calculation. "
        "Bars at or above the threshold are highlighted."
    )


def _default_by_utilisation(context: PageContext) -> None:
    """Default rate across utilisation bands."""
    target = context.target_column
    data = context.data
    if target is None or "utilisation_latest" not in data.columns:
        components.unavailable(
            "Default rate by utilisation", "Requires the target and utilisation feature."
        )
        return

    working = data.copy()
    working["_util_band"] = assign_utilisation_band(working)
    outcome = category_outcome_rates(working, "_util_band", target, min_group_size=30)

    if isinstance(outcome, InsufficientData):
        components.unavailable("Default rate by utilisation", outcome.message)
        return

    figure = charts.band_rate_and_size(
        outcome.table.sort_index(),
        band_label="Utilisation band",
        title="Does the default rate rise with utilisation?",
        subtitle=(
            "The top band boundary is the configured HIGH_UTILISATION_THRESHOLD "
            f"({settings.high_utilisation_threshold:.0%}), not a hard-coded value."
        ),
        overall_rate=outcome.overall_rate,
    )
    components.chart(figure, key="fin_util_rate")
    components.methodology_note(
        "How utilisation bands are defined", utilisation_band_documentation()
    )


def _limit_distribution(context: PageContext) -> None:
    """Distribution of credit limits."""
    data = context.data
    limit_column = context.registry.one(Role.CREDIT_LIMIT)
    if limit_column is None or limit_column not in data.columns:
        components.unavailable("Credit-limit distribution", "No credit-limit column.")
        return

    bins = distribution_bins(data, limit_column, bins=35)
    if isinstance(bins, InsufficientData):
        components.unavailable("Credit-limit distribution", bins.message)
        return

    values = pd.to_numeric(data[limit_column], errors="coerce")
    figure = charts.distribution_with_threshold(
        bins,
        title="How are credit limits distributed?",
        x_title=f"Credit limit ({DATASET_CURRENCY})",
        median_value=float(values.median()) if values.notna().any() else None,
    )
    components.chart(figure, key="fin_limit_dist")

    summaries = summarise_numeric(data, columns=[limit_column])
    if summaries:
        summary = summaries[0]
        st.caption(
            f"Mean {DATASET_CURRENCY}{summary.mean:,.0f} against median "
            f"{DATASET_CURRENCY}{summary.median:,.0f} - the distribution is "
            f"{summary.skew_description}, so the median is the better guide to a "
            "typical customer. This is a credit limit, not income or wealth."
        )


def _default_by_limit_band(context: PageContext) -> None:
    """Default rate and average utilisation across credit-limit bands."""
    target = context.target_column
    data = context.data
    if target is None or "limit_band" not in data.columns:
        components.unavailable(
            "Default rate by credit-limit band", "Requires the target and limit bands."
        )
        return

    outcome = category_outcome_rates(data, "limit_band", target, min_group_size=30)
    if isinstance(outcome, InsufficientData):
        components.unavailable("Default rate by credit-limit band", outcome.message)
        return

    table = outcome.table.sort_index().copy()

    # Attach mean utilisation per band so the two measures can be read together.
    if "utilisation_latest" in data.columns:
        utilisation = (
            data.groupby("limit_band", observed=True)["utilisation_latest"]
            .mean()
            .reindex(table.index)
        )
        table["avg_utilisation"] = utilisation

    figure = charts.band_rate_and_size(
        table,
        band_label="Credit-limit band (quartiles of this portfolio)",
        title="How does outcome vary across credit-limit bands?",
        subtitle=(
            "Bands are quartiles of the observed credit limit. They describe "
            "granted credit capacity only, not income or social class."
        ),
        overall_rate=outcome.overall_rate,
        extra_metric=("avg_utilisation", "Avg utilisation"),
    )
    components.chart(figure, key="fin_limit_rate")
    components.methodology_note(
        "How credit-limit bands are defined", limit_band_documentation()
    )


def _render_credit_behaviour(context: PageContext) -> None:
    """Render the Credit Behaviour tab."""
    components.section(
        "Credit Behaviour",
        "How much credit customers use, how much they were granted, and how each "
        "relates to the observed default outcome.",
    )
    left, right = st.columns(2, gap="medium")
    with left:
        _utilisation_distribution(context)
        components.spacer(0.5)
        _limit_distribution(context)
    with right:
        _default_by_utilisation(context)
        components.spacer(0.5)
        _default_by_limit_band(context)

    components.section(
        "Two-way view",
        "Default rate across credit-limit band and age band together, which shows "
        "whether the limit gradient holds within every age group.",
    )
    target = context.target_column
    if target and {"limit_band", "age_band"} <= set(context.data.columns):
        table = cross_tabulate(context.data, "limit_band", "age_band", target, "mean")
        if isinstance(table, InsufficientData):
            components.unavailable("Two-way breakdown", table.message)
        else:
            components.chart(
                charts.rate_matrix_heatmap(
                    (table * 100).round(2),
                    title="Observed default rate by credit-limit band and age band",
                    x_title="Age band",
                    y_title="Credit-limit band",
                    subtitle=(
                        "Cells with no customers in the current selection are left "
                        "blank. Read across a row to see whether the limit gradient "
                        "holds within each age group."
                    ),
                ),
                key="fin_twoway",
            )
    else:
        components.unavailable("Two-way breakdown", "Requires limit and age bands.")


# --------------------------------------------------------------------------- #
# Tab 2: payment behaviour and delinquency
# --------------------------------------------------------------------------- #


def _delinquency_by_status(context: PageContext) -> None:
    """Default rate by the most recent repayment-status code."""
    target = context.target_column
    data = context.data
    status_column = "pay_status_m1"

    if target is None or status_column not in data.columns:
        components.unavailable(
            "Default rate by payment status",
            "Requires the target and the most recent repayment-status column.",
        )
        return

    from src.schema import PAY_STATUS_LABELS

    working = data.copy()
    codes = pd.to_numeric(working[status_column], errors="coerce")
    # Label with the documented meaning so a reader never has to decode integers.
    working["_status"] = codes.map(
        lambda c: f"{int(c)} · {PAY_STATUS_LABELS.get(int(c), 'Undocumented code')}"
        if pd.notna(c)
        else "Unknown"
    )
    order = [
        f"{code} · {PAY_STATUS_LABELS.get(code, 'Undocumented code')}"
        for code in sorted(PAY_STATUS_LABELS)
    ]
    present = [label for label in order if label in set(working["_status"])]
    working["_status"] = pd.Categorical(working["_status"], categories=present, ordered=True)

    outcome = category_outcome_rates(working, "_status", target, min_group_size=30)
    if isinstance(outcome, InsufficientData):
        components.unavailable("Default rate by payment status", outcome.message)
        return

    figure = charts.default_rate_by_payment_status(
        outcome.table.sort_index(),
        overall_rate=outcome.overall_rate,
        min_group_size=30,
    )
    components.chart(figure, key="fin_status_rate")

    with st.expander("Repayment status code reference", expanded=False):
        st.caption(
            "Codes as defined by the source documentation. Codes -2 and 0 occur "
            "frequently in the data but are not defined in the source paper; the "
            "meanings shown for them are the common community interpretation and "
            "are labelled as undocumented. Delinquency measures in this application "
            "count only codes of 1 or more, so this ambiguity does not affect them."
        )
        st.dataframe(
            charts.payment_status_reference(), width="stretch", hide_index=True
        )


def _persistence(context: PageContext) -> None:
    """How many months customers spent behind on payment."""
    data = context.data
    if "delinquent_months_count" not in data.columns:
        components.unavailable(
            "Delinquency persistence", "Requires the delinquency count feature."
        )
        return
    counts = pd.to_numeric(data["delinquent_months_count"], errors="coerce").value_counts()
    components.chart(charts.delinquency_persistence_chart(counts), key="fin_persistence")


def _repayment_distribution(context: PageContext) -> None:
    """Distribution of the capped repayment ratio."""
    data = context.data
    column = "repayment_ratio_capped_mean"
    if column not in data.columns:
        components.unavailable("Repayment ratio distribution", "Feature unavailable.")
        return

    bins = distribution_bins(data, column, bins=40)
    if isinstance(bins, InsufficientData):
        components.unavailable("Repayment ratio distribution", bins.message)
        return

    values = pd.to_numeric(data[column], errors="coerce")
    figure = charts.distribution_with_threshold(
        bins,
        title="What share of their statement do customers repay?",
        x_title="Average repayment ratio (each month capped at 200%)",
        median_value=float(values.median()) if values.notna().any() else None,
        is_percentage=True,
    )
    components.chart(figure, key="fin_repay_dist")
    st.caption(
        "Repayment ratio is the payment made in a month divided by the previous "
        "month's statement balance, since a payment settles the prior statement. "
        "Each month is capped at 200% before averaging, because a near-zero prior "
        "balance can otherwise inflate the raw ratio into the thousands. A value of "
        "1.0 means the statement was typically cleared in full."
    )


def _behaviour_segments(context: PageContext) -> None:
    """Compare measured behaviour across payment segments."""
    data = context.data
    target = context.target_column
    needed = {"is_full_payer", "is_revolver"}
    if not needed <= set(data.columns) or target is None:
        components.unavailable(
            "Payment segment comparison", "Requires the repayment-behaviour features."
        )
        return

    full = data["is_full_payer"].fillna(False).astype("bool")
    revolver = data["is_revolver"].fillna(False).astype("bool")
    middle = ~full & ~revolver

    segments: dict[str, dict[str, float]] = {}
    for name, mask in (
        ("Full payers", full),
        ("Partial payers", middle),
        ("Revolvers", revolver),
    ):
        subset = data.loc[mask]
        if len(subset) < 30:
            continue
        metrics: dict[str, float] = {
            "Default rate": float(
                pd.to_numeric(subset[target], errors="coerce").mean() * 100
            )
        }
        if "is_currently_delinquent" in subset.columns:
            metrics["Currently behind"] = float(
                subset["is_currently_delinquent"].fillna(False).astype("bool").mean() * 100
            )
        if "is_high_utilisation" in subset.columns:
            metrics["High utilisation"] = float(
                subset["is_high_utilisation"].fillna(False).astype("bool").mean() * 100
            )
        if "is_over_limit" in subset.columns:
            metrics["Over limit"] = float(
                subset["is_over_limit"].fillna(False).astype("bool").mean() * 100
            )
        segments[f"{name} (n={len(subset):,})"] = metrics

    if not segments:
        components.unavailable(
            "Payment segment comparison",
            "No behavioural segment reached the 30-customer minimum in this selection.",
        )
        return

    components.chart(
        charts.payment_behaviour_comparison(segments), key="fin_segments"
    )
    st.caption(
        "Segments are defined purely by measured repayment behaviour. The default "
        "rate is shown as an outcome of each segment, not as part of its definition, "
        "so the comparison is not circular."
    )


def _zero_payment_note(context: PageContext) -> None:
    """Report zero-payment frequency without implying it means default."""
    data = context.data
    target = context.target_column
    if "months_zero_payment" not in data.columns or target is None:
        return

    zero_months = pd.to_numeric(data["months_zero_payment"], errors="coerce")
    outcome = pd.to_numeric(data[target], errors="coerce")

    rows: list[dict[str, str]] = []
    for count in sorted(zero_months.dropna().unique()):
        mask = zero_months == count
        n = int(mask.sum())
        if n < 30:
            continue
        rows.append(
            {
                "Months with no payment": f"{int(count)}",
                "Customers": f"{n:,}",
                "Share of selection": f"{n / len(data) * 100:.1f}%",
                "Observed default rate": f"{outcome[mask].mean() * 100:.2f}%",
            }
        )

    if not rows:
        return

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "A month with no payment does not by itself mean default. Many customers "
        "record zero-payment months and do not default; the table shows the "
        "observed rate for each group so the relationship can be judged from the "
        "data rather than assumed."
    )


def _render_payment_behaviour(context: PageContext) -> None:
    """Render the Payment Behaviour & Delinquency tab."""
    components.section(
        "Payment Behaviour & Delinquency",
        "How customers repay, how persistent any delinquency is, and how each "
        "pattern relates to the observed outcome.",
    )
    _delinquency_by_status(context)

    components.spacer(0.6)
    left, right = st.columns(2, gap="medium")
    with left:
        _persistence(context)
    with right:
        _repayment_distribution(context)

    components.spacer(0.4)
    components.section("Behaviour by payment segment", rule=False)
    _behaviour_segments(context)

    components.spacer(0.4)
    components.section(
        "Zero-payment months",
        "Frequency of months with no payment, and the observed outcome for each group.",
        rule=False,
    )
    _zero_payment_note(context)


# --------------------------------------------------------------------------- #
# Tab 3: driver analysis
# --------------------------------------------------------------------------- #


def _render_drivers(context: PageContext) -> None:
    """Render the Driver Analysis tab."""
    components.section(
        "What Drives Observed Default Risk?",
        "Four independent methods applied to the same selection and triangulated. "
        "Agreement across methods is stronger evidence than a single large "
        "statistic from one of them.",
    )

    with st.spinner("Analysing feature relationships..."):
        result = run_driver_analysis(context.data, registry=context.registry)

    if not isinstance(result, DriverAnalysisResult):
        components.unavailable("Driver analysis", result.reason)
        return

    st.markdown(
        f'<div class="fr-notice" style="text-align:left;padding:0.8rem 1rem">'
        f'<div class="fr-notice-body" style="max-width:none;margin:0">'
        f"<b>Interpretation.</b> {CAUSATION_CAVEAT}</div></div>",
        unsafe_allow_html=True,
    )
    components.spacer(0.5)

    left, right = st.columns([1.15, 1], gap="medium")

    with left:
        components.section("Consensus ranking", rule=False)
        insight_cards.render_driver_rows(result.consensus, limit=10)
        st.caption(
            "Bar length is the mean absolute effect size across methods. "
            "'agree' counts how many methods found a non-negligible relationship. "
            "Red indicates higher values accompany more default; green indicates "
            "the reverse."
        )

    with right:
        components.section("Strongest single measures", rule=False)
        sentences = [
            describe_driver(item, "default")
            for item in list(result.numeric_comparisons)[:4]
        ] + [
            describe_driver(item, "default")
            for item in list(result.categorical_tests)[:2]
        ]
        for sentence in sentences:
            st.markdown(
                f'<div class="fr-insight" style="--fr-insight-accent:'
                f'var(--fr-info);padding:0.62rem 0.8rem;margin-bottom:0.45rem">'
                f'<div class="fr-insight-body">{components.escape(sentence)}</div></div>',
                unsafe_allow_html=True,
            )

    components.section(
        "Group comparison detail",
        "Median values for each outcome class, with the effect size that says "
        "whether the gap is large enough to matter.",
    )
    rows: list[dict[str, str]] = []
    for item in list(result.numeric_comparisons)[:12]:
        details = item.details
        rows.append(
            {
                "Measure": item.feature,
                "Median (defaulted)": f"{details.get('median_when_outcome_1', float('nan')):,.3f}",
                "Median (did not)": f"{details.get('median_when_outcome_0', float('nan')):,.3f}",
                "Effect size": f"{item.effect_size:+.3f}",
                "Strength": item.strength.value,
                "Direction": item.direction,
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Effect size is the rank-biserial correlation from a Mann-Whitney U "
            "test, chosen because the monetary measures are heavily skewed and a "
            "t-test would be unreliable. " + result.significance_note
        )

    components.section(
        "Measure relationships",
        "Spearman rank correlation between measures. Pairs that restate one another "
        "by construction are excluded, so what remains is behavioural.",
    )
    correlation = correlation_matrix(context.data, registry=context.registry)
    if isinstance(correlation, InsufficientData):
        components.unavailable("Correlation analysis", correlation.message)
    else:
        components.chart(
            correlation_heatmap(correlation.matrix, max_columns=16), key="fin_corr"
        )
        top = correlation.top_pairs(6)
        if top:
            st.caption(
                "Strongest non-redundant associations: "
                + " · ".join(
                    f"{p.left} ~ {p.right} (r={p.correlation:+.2f})" for p in top[:4]
                )
            )


# --------------------------------------------------------------------------- #
# Tab 4: customer profile
# --------------------------------------------------------------------------- #


def _render_profile(context: PageContext) -> None:
    """Render the Customer Profile tab."""
    components.section(
        "Customer Profile",
        "Descriptive composition of the selection. These attributes are reported "
        "for portfolio understanding and fairness review only. They are excluded "
        "from risk modelling by default and are never presented as risk drivers.",
    )

    st.markdown(
        f'<div class="fr-notice" style="text-align:left;padding:0.8rem 1rem">'
        f'<div class="fr-notice-body" style="max-width:none;margin:0">'
        f"<b>Fairness note.</b> Differences between demographic groups reflect "
        f"differences in group composition - credit limit, utilisation and tenure - "
        f"as much as anything else. Sex and marital status are excluded from driver "
        f"analysis and from model features by design; they appear here so the "
        f"portfolio can be understood and audited.</div></div>",
        unsafe_allow_html=True,
    )
    components.spacer(0.5)

    target = context.target_column
    data = context.data
    dimensions = [c for c in ("age_band", "education", "marriage", "sex") if c in data.columns]

    if not dimensions or target is None:
        components.unavailable(
            "Profile analysis", "Requires the demographic columns and the target."
        )
        return

    pairs = [dimensions[i : i + 2] for i in range(0, len(dimensions), 2)]
    for pair in pairs:
        columns = st.columns(len(pair), gap="medium")
        for column, dimension in zip(columns, pair):
            with column:
                outcome = category_outcome_rates(data, dimension, target, min_group_size=30)
                if isinstance(outcome, InsufficientData):
                    components.unavailable(f"Breakdown by {dimension}", outcome.message)
                    continue
                figure = charts.band_rate_and_size(
                    outcome.table.sort_index(),
                    band_label=dimension.replace("_", " ").title(),
                    title=f"Composition and observed outcome by {dimension.replace('_', ' ')}",
                    subtitle="Descriptive only; not used as a risk driver.",
                    overall_rate=outcome.overall_rate,
                )
                components.chart(figure, key=f"fin_profile_{dimension}")

    components.section(
        "Segment profile table",
        "Average measured behaviour per credit-limit band, with the observed "
        "outcome rate alongside.",
    )
    metrics = [
        c
        for c in (
            "utilisation_latest",
            "repayment_ratio_capped_mean",
            "max_delinquency",
            "total_paid_6m",
            "age",
        )
        if c in data.columns
    ]
    if "limit_band" in data.columns and metrics:
        profile = segment_profile(data, "limit_band", metrics, target)
        if isinstance(profile, InsufficientData):
            components.unavailable("Segment profile", profile.message)
        else:
            st.dataframe(profile.round(3), width="stretch")
    else:
        components.unavailable("Segment profile", "Requires limit bands and metrics.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def render(context: PageContext, show_filters: bool = True) -> None:
    """Render the Financial Analytics page.

    Args:
        context: The per-run page context.
        show_filters: Render the global filter bar beneath the header.
    """
    components.page_header(
        title="Financial Analytics",
        description=(
            "An analytical workspace for credit behaviour, payment patterns and the "
            "measures associated with the observed default outcome."
        ),
        meta=[
            ("Customers selected", f"{context.n_selected:,}"),
            (
                "Observed default rate",
                context.kpis.formatted("observed_default_rate"),
            ),
        ],
    )

    if show_filters:
        from src.ui import filter_bar

        filter_bar.render_filter_bar(context.bounds, context.filters)
        filter_bar.render_active_filters(context.filters)
        components.spacer(0.6)

    if context.is_empty:
        from src.ui.filter_bar import RESET_FLAG

        components.empty_state(on_reset=RESET_FLAG)
        return

    credit_tab, payment_tab, driver_tab, profile_tab = st.tabs(
        [
            "Credit Behaviour",
            "Payment Behaviour & Delinquency",
            "Driver Analysis",
            "Customer Profile",
        ]
    )

    # Each tab is wrapped independently so a failure in one does not blank the
    # others.
    with credit_tab:
        components.safe_render("Credit behaviour", _render_credit_behaviour, context)
    with payment_tab:
        components.safe_render("Payment behaviour", _render_payment_behaviour, context)
    with driver_tab:
        components.safe_render("Driver analysis", _render_drivers, context)
    with profile_tab:
        components.safe_render("Customer profile", _render_profile, context)
