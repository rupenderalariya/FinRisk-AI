"""
Dashboard chart builders.

Composed over :mod:`src.visualization` and the shared Plotly template, so every
figure in the product inherits one visual language. Builders here are the
*page-specific compositions* - the hero multi-axis view, the delinquency
breakdown - while the generic primitives stay in ``src.visualization``.

Each builder returns a figure and never renders, so charts can be produced and
asserted on headlessly in tests.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.schema import DATASET_CURRENCY, PAY_STATUS_LABELS
from src.trends import TrendResult
from src.ui.theme import CATEGORICAL_SEQUENCE, PALETTE, with_alpha
from src.visualization import empty_figure


def _axis_title(text: str) -> dict[str, Any]:
    """Consistent axis-title styling."""
    return {"text": text, "font": {"size": 12, "color": PALETTE["text_muted"]}}


# --------------------------------------------------------------------------- #
# Hero: credit behaviour across the observed months
# --------------------------------------------------------------------------- #


def credit_behaviour_overview(
    bill_trend: TrendResult | None,
    payment_trend: TrendResult | None,
    utilisation_trend: TrendResult | None,
    height: int = 420,
) -> go.Figure:
    """Build the primary Executive Overview visualisation.

    Combines the three portfolio measures that share a story - what customers
    owe, what they repay, and how much of their limit is committed - on a dual
    axis so the money series and the percentage series stay readable together.

    Bars carry the money values and a line carries utilisation, which keeps the
    chart to three series and avoids the overcrowding a five-series chart
    produces.

    Args:
        bill_trend: Mean statement balance per period.
        payment_trend: Mean payment per period.
        utilisation_trend: Mean utilisation per period.
        height: Figure height in pixels.

    Returns:
        The figure, or an explanatory placeholder when no series is available.
    """
    available = [t for t in (bill_trend, payment_trend, utilisation_trend) if t is not None]
    if not available:
        return empty_figure(
            "Monthly trend data is unavailable for the current selection.",
            height,
            "Credit Behaviour Overview",
        )

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    labels = [p.month_label for p in available[0].points]

    if bill_trend is not None:
        figure.add_trace(
            go.Bar(
                x=[p.month_label for p in bill_trend.points],
                y=[p.value for p in bill_trend.points],
                name="Avg statement balance",
                marker_color=with_alpha(PALETTE["info"], 0.72),
                marker_line_width=0,
                hovertemplate=(
                    f"<b>%{{x}}</b><br>Avg statement balance: {DATASET_CURRENCY}"
                    "%{y:,.0f}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    if payment_trend is not None:
        figure.add_trace(
            go.Bar(
                x=[p.month_label for p in payment_trend.points],
                y=[p.value for p in payment_trend.points],
                name="Avg payment received",
                marker_color=with_alpha(PALETTE["positive"], 0.78),
                marker_line_width=0,
                hovertemplate=(
                    f"<b>%{{x}}</b><br>Avg payment: {DATASET_CURRENCY}"
                    "%{y:,.0f}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    if utilisation_trend is not None:
        figure.add_trace(
            go.Scatter(
                x=[p.month_label for p in utilisation_trend.points],
                y=[p.value for p in utilisation_trend.points],
                name="Avg utilisation",
                mode="lines+markers",
                line={"color": PALETTE["attention"], "width": 2.6},
                marker={"size": 8, "color": PALETTE["attention"]},
                hovertemplate="<b>%{x}</b><br>Avg utilisation: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=True,
        )

    figure.update_layout(
        title={
            "text": "Credit Behaviour Overview"
            "<br><span style='font-size:11.5px;color:"
            f"{PALETTE['text_muted']}'>Six observed monthly periods, oldest to most "
            "recent. Bars read on the left axis, utilisation on the right.</span>",
            "x": 0,
            "xanchor": "left",
        },
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        height=height,
        hovermode="x unified",
        margin={"l": 68, "r": 62, "t": 74, "b": 48},
    )
    figure.update_xaxes(title=_axis_title("Observed monthly period"), showgrid=False)
    figure.update_yaxes(
        title=_axis_title(f"Amount ({DATASET_CURRENCY})"),
        secondary_y=False,
        showgrid=True,
        gridcolor=PALETTE["border"],
        tickformat=",.0f",
    )
    figure.update_yaxes(
        title=_axis_title("Utilisation (%)"),
        secondary_y=True,
        showgrid=False,
        ticksuffix="%",
    )
    return figure


# --------------------------------------------------------------------------- #
# Delinquency
# --------------------------------------------------------------------------- #


def default_rate_by_payment_status(
    table: pd.DataFrame,
    overall_rate: float | None = None,
    min_group_size: int = 30,
    height: int = 420,
) -> go.Figure:
    """Plot default rate and customer count per repayment-status code.

    Two panels sharing an x-axis: the rate above, the population below. Showing
    the count matters because a high rate over 30 customers means something very
    different from the same rate over 3,000, and a single chart cannot convey
    both without one hiding the other.

    Args:
        table: Indexed by status label, with ``rate_pct`` and ``count`` columns.
        overall_rate: Portfolio baseline to mark.
        min_group_size: Groups below this are de-emphasised.
        height: Figure height in pixels.

    Returns:
        The figure.
    """
    if table is None or len(table) == 0:
        return empty_figure(
            "No repayment-status breakdown available for this selection.",
            height,
            "Default rate by repayment status",
        )

    labels = [str(i) for i in table.index]
    rates = pd.to_numeric(table["rate_pct"], errors="coerce").tolist()
    counts = pd.to_numeric(table["count"], errors="coerce").tolist()

    colours = [
        PALETTE["risk"] if (c or 0) >= min_group_size else PALETTE["text_faint"]
        for c in counts
    ]
    n_small = sum(1 for c in counts if (c or 0) < min_group_size)

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.09,
    )

    figure.add_trace(
        go.Bar(
            x=labels,
            y=rates,
            marker_color=colours,
            marker_line_width=0,
            text=[f"{r:.1f}%" if pd.notna(r) else "" for r in rates],
            textposition="outside",
            textfont={"size": 11, "color": PALETTE["text_secondary"]},
            customdata=counts,
            name="Default rate",
            hovertemplate=(
                "<b>%{x}</b><br>Default rate: %{y:.2f}%<br>"
                "Customers: %{customdata:,}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Bar(
            x=labels,
            y=counts,
            marker_color=with_alpha(PALETTE["info"], 0.5),
            marker_line_width=0,
            name="Customers",
            hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if overall_rate is not None and np.isfinite(overall_rate):
        figure.add_hline(
            y=overall_rate,
            line_width=1.6,
            line_dash="dash",
            line_color=PALETTE["attention"],
            annotation_text=f"Selection average {overall_rate:.2f}%",
            annotation_position="top right",
            annotation_font_size=10.5,
            annotation_font_color=PALETTE["attention"],
            row=1,
            col=1,
        )

    subtitle = (
        "Status codes follow the source documentation. "
        + (
            f"{n_small} group(s) below {min_group_size} customers are greyed out; "
            "their rates are too noisy to rank."
            if n_small
            else "All groups shown meet the minimum size for comparison."
        )
    )

    figure.update_layout(
        title={
            "text": "Default rate by most recent repayment status"
            f"<br><span style='font-size:11.5px;color:{PALETTE['text_muted']}'>"
            f"{subtitle}</span>",
            "x": 0,
            "xanchor": "left",
        },
        height=height,
        showlegend=False,
        margin={"l": 62, "r": 24, "t": 76, "b": 78},
    )
    figure.update_yaxes(
        title=_axis_title("Default rate (%)"), ticksuffix="%", row=1, col=1
    )
    figure.update_yaxes(title=_axis_title("Customers"), tickformat=",.0f", row=2, col=1)
    figure.update_xaxes(
        title=_axis_title("Repayment status in the most recent observed month"),
        tickangle=-18,
        row=2,
        col=1,
    )
    return figure


def delinquency_persistence_chart(
    counts: pd.Series, height: int = 340
) -> go.Figure:
    """Plot how many months customers spent behind on payment.

    Args:
        counts: Value counts of months-delinquent, indexed 0-6.
        height: Figure height in pixels.

    Returns:
        The figure.
    """
    if counts is None or counts.empty:
        return empty_figure(
            "Delinquency persistence is unavailable for this selection.",
            height,
            "Delinquency persistence",
        )

    ordered = counts.sort_index()
    total = float(ordered.sum())
    labels = [f"{int(i)}" for i in ordered.index]
    values = ordered.tolist()
    shares = [v / total * 100.0 if total else 0.0 for v in values]

    # Deepen the colour as persistence rises, so severity is readable at a glance.
    max_index = max((int(i) for i in ordered.index), default=1) or 1
    colours = [
        with_alpha(PALETTE["risk"], 0.25 + 0.72 * (int(i) / max_index))
        if int(i) > 0
        else with_alpha(PALETTE["positive"], 0.6)
        for i in ordered.index
    ]

    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colours,
            marker_line_width=0,
            customdata=shares,
            text=[f"{s:.1f}%" for s in shares],
            textposition="outside",
            textfont={"size": 11, "color": PALETTE["text_secondary"]},
            hovertemplate=(
                "<b>%{x} month(s) behind</b><br>Customers: %{y:,}<br>"
                "Share: %{customdata:.2f}%<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={
            "text": "Delinquency persistence across the six observed months"
            f"<br><span style='font-size:11.5px;color:{PALETTE['text_muted']}'>"
            "Counts only documented delay codes of one month or more.</span>",
            "x": 0,
            "xanchor": "left",
        },
        height=height,
        showlegend=False,
        margin={"l": 62, "r": 24, "t": 74, "b": 52},
    )
    figure.update_xaxes(title=_axis_title("Number of months behind on payment"))
    figure.update_yaxes(title=_axis_title("Customers"), tickformat=",.0f")
    return figure


# --------------------------------------------------------------------------- #
# Band comparisons
# --------------------------------------------------------------------------- #


def band_rate_and_size(
    table: pd.DataFrame,
    band_label: str,
    title: str,
    subtitle: str = "",
    overall_rate: float | None = None,
    extra_metric: tuple[str, str] | None = None,
    height: int = 380,
) -> go.Figure:
    """Plot default rate per band with customer count on a secondary axis.

    Args:
        table: Indexed by band, with ``rate_pct`` and ``count`` columns.
        band_label: X-axis label.
        title: Chart title.
        subtitle: Caveat or methodology line.
        overall_rate: Baseline to mark.
        extra_metric: ``(column, label)`` for an optional third series such as
            average utilisation.
        height: Figure height in pixels.

    Returns:
        The figure.
    """
    if table is None or len(table) == 0:
        return empty_figure("No band breakdown available for this selection.", height, title)

    labels = [str(i) for i in table.index]
    rates = pd.to_numeric(table["rate_pct"], errors="coerce").tolist()
    counts = pd.to_numeric(table["count"], errors="coerce").tolist()

    figure = make_subplots(specs=[[{"secondary_y": True}]])

    figure.add_trace(
        go.Bar(
            x=labels,
            y=counts,
            name="Customers",
            marker_color=with_alpha(PALETTE["info"], 0.34),
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<extra></extra>",
        ),
        secondary_y=True,
    )

    figure.add_trace(
        go.Scatter(
            x=labels,
            y=rates,
            name="Default rate",
            mode="lines+markers",
            line={"color": PALETTE["risk"], "width": 2.6},
            marker={"size": 9, "color": PALETTE["risk"]},
            hovertemplate="<b>%{x}</b><br>Default rate: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )

    if extra_metric is not None:
        column, label = extra_metric
        if column in table.columns:
            values = pd.to_numeric(table[column], errors="coerce") * 100.0
            figure.add_trace(
                go.Scatter(
                    x=labels,
                    y=values.tolist(),
                    name=label,
                    mode="lines+markers",
                    line={"color": PALETTE["attention"], "width": 2, "dash": "dot"},
                    marker={"size": 7, "color": PALETTE["attention"]},
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>",
                ),
                secondary_y=False,
            )

    if overall_rate is not None and np.isfinite(overall_rate):
        figure.add_hline(
            y=overall_rate,
            line_width=1.5,
            line_dash="dash",
            line_color=PALETTE["text_muted"],
            annotation_text=f"Selection average {overall_rate:.2f}%",
            annotation_position="top left",
            annotation_font_size=10.5,
            annotation_font_color=PALETTE["text_muted"],
            secondary_y=False,
        )

    title_html = title
    if subtitle:
        title_html = (
            f"{title}<br><span style='font-size:11.5px;color:"
            f"{PALETTE['text_muted']}'>{subtitle}</span>"
        )

    figure.update_layout(
        title={"text": title_html, "x": 0, "xanchor": "left"},
        height=height,
        hovermode="x unified",
        margin={"l": 62, "r": 62, "t": 76 if subtitle else 56, "b": 58},
    )
    figure.update_xaxes(title=_axis_title(band_label), tickangle=-15)
    figure.update_yaxes(
        title=_axis_title("Rate (%)"), ticksuffix="%", secondary_y=False
    )
    figure.update_yaxes(
        title=_axis_title("Customers"), showgrid=False, tickformat=",.0f", secondary_y=True
    )
    return figure


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #


def distribution_with_threshold(
    bins: pd.DataFrame,
    title: str,
    x_title: str,
    threshold: float | None = None,
    threshold_label: str = "",
    median_value: float | None = None,
    is_percentage: bool = False,
    height: int = 360,
) -> go.Figure:
    """Plot a binned distribution, optionally marking a threshold and median.

    Args:
        bins: Output of :func:`src.analytics.distribution_bins`.
        title: Chart title.
        x_title: X-axis label with units.
        threshold: Value to mark with a vertical rule.
        threshold_label: Annotation for the threshold.
        median_value: Median to mark.
        is_percentage: Format the axis as a percentage.
        height: Figure height in pixels.

    Returns:
        The figure.
    """
    if bins is None or len(bins) == 0 or "bin_center" not in getattr(bins, "columns", []):
        return empty_figure("No distribution available for this selection.", height, title)

    centres = bins["bin_center"]
    counts = bins["count"]

    # Bars past the threshold take the attention colour, so the elevated region
    # is visible without needing to read the axis.
    if threshold is not None:
        colours = [
            PALETTE["attention"] if c >= threshold else with_alpha(PALETTE["info"], 0.7)
            for c in centres
        ]
    else:
        colours = [with_alpha(PALETTE["info"], 0.75)] * len(centres)

    hover_x = "%{customdata[0]:.1%} to %{customdata[1]:.1%}" if is_percentage else (
        "%{customdata[0]:,.0f} to %{customdata[1]:,.0f}"
    )

    figure = go.Figure(
        go.Bar(
            x=centres,
            y=counts,
            marker_color=colours,
            marker_line_width=0,
            customdata=np.stack(
                [bins["bin_left"], bins["bin_right"], bins["percentage"]], axis=-1
            ),
            hovertemplate=(
                f"Range: {hover_x}<br>Customers: %{{y:,}}<br>"
                "Share: %{customdata[2]:.2f}%<extra></extra>"
            ),
        )
    )

    for value, label, colour in (
        (threshold, threshold_label, PALETTE["attention"]),
        (median_value, "Median", PALETTE["text_secondary"]),
    ):
        if value is not None and np.isfinite(value):
            figure.add_vline(
                x=value,
                line_width=1.8,
                line_dash="dash",
                line_color=colour,
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=10.5,
                annotation_font_color=colour,
            )

    figure.update_layout(
        title={"text": title, "x": 0, "xanchor": "left"},
        height=height,
        showlegend=False,
        margin={"l": 62, "r": 24, "t": 56, "b": 52},
    )
    figure.update_xaxes(
        title=_axis_title(x_title), tickformat=".0%" if is_percentage else ",.0f"
    )
    figure.update_yaxes(title=_axis_title("Customers"), tickformat=",.0f")
    return figure


# --------------------------------------------------------------------------- #
# Payment behaviour
# --------------------------------------------------------------------------- #


def payment_behaviour_comparison(
    segments: Mapping[str, Mapping[str, float]],
    height: int = 360,
) -> go.Figure:
    """Compare behaviour metrics across named behavioural segments.

    Args:
        segments: ``{segment: {metric: value}}`` with values already in percent.
        height: Figure height in pixels.

    Returns:
        The figure.
    """
    if not segments:
        return empty_figure(
            "No behavioural segments available for this selection.",
            height,
            "Payment behaviour comparison",
        )

    metrics = sorted({metric for values in segments.values() for metric in values})
    figure = go.Figure()

    for index, (segment, values) in enumerate(segments.items()):
        figure.add_trace(
            go.Bar(
                name=segment,
                x=metrics,
                y=[values.get(metric, np.nan) for metric in metrics],
                marker_color=CATEGORICAL_SEQUENCE[index % len(CATEGORICAL_SEQUENCE)],
                marker_line_width=0,
                hovertemplate=(
                    f"<b>{segment}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title={
            "text": "Behaviour comparison across payment segments"
            f"<br><span style='font-size:11.5px;color:{PALETTE['text_muted']}'>"
            "Segments are defined by measured repayment behaviour, not by outcome."
            "</span>",
            "x": 0,
            "xanchor": "left",
        },
        barmode="group",
        bargap=0.3,
        height=height,
        margin={"l": 62, "r": 24, "t": 76, "b": 58},
    )
    figure.update_xaxes(title=_axis_title("Measure"))
    figure.update_yaxes(title=_axis_title("Percent of segment (%)"), ticksuffix="%")
    return figure


def rate_matrix_heatmap(
    matrix: pd.DataFrame,
    title: str,
    x_title: str,
    y_title: str,
    subtitle: str = "",
    height: int = 360,
) -> go.Figure:
    """Render a two-way rate table as a heatmap.

    Used instead of a pandas ``Styler`` gradient, which would pull in matplotlib
    as a dependency and would not match the product theme. A Plotly heatmap keeps
    the visual language consistent and shows empty combinations explicitly.

    Args:
        matrix: Rates as percentages, rows and columns already ordered.
        title: Chart title.
        x_title: X-axis label.
        y_title: Y-axis label.
        subtitle: Caveat line.
        height: Figure height in pixels.

    Returns:
        The figure.
    """
    if matrix is None or matrix.empty:
        return empty_figure("No two-way breakdown available for this selection.", height, title)

    values = matrix.to_numpy(dtype="float64")
    text = [
        [f"{v:.1f}%" if np.isfinite(v) else "-" for v in row] for row in values
    ]

    title_html = title
    if subtitle:
        title_html = (
            f"{title}<br><span style='font-size:11.5px;color:"
            f"{PALETTE['text_muted']}'>{subtitle}</span>"
        )

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            # Blue (lower) through amber to red (higher) reads as increasing risk.
            colorscale=[
                [0.0, "#1b3a52"],
                [0.45, "#2c4a5e"],
                [0.7, "#8f6320"],
                [1.0, "#ef5f5f"],
            ],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11.5, "color": PALETTE["text"]},
            colorbar={"title": "Rate %", "thickness": 12, "len": 0.85},
            hovertemplate=(
                f"<b>%{{y}}</b> · <b>%{{x}}</b><br>Default rate: %{{z:.2f}}%"
                "<extra></extra>"
            ),
            hoverongaps=False,
        )
    )
    figure.update_layout(
        title={"text": title_html, "x": 0, "xanchor": "left"},
        height=height,
        margin={"l": 130, "r": 24, "t": 76 if subtitle else 56, "b": 54},
    )
    figure.update_xaxes(title=_axis_title(x_title), showgrid=False)
    figure.update_yaxes(title=_axis_title(y_title), showgrid=False, autorange="reversed")
    return figure


def payment_status_reference() -> pd.DataFrame:
    """Build the repayment-status code reference table.

    Codes come straight from :data:`src.schema.PAY_STATUS_LABELS`, so the
    documentation shown in the UI cannot drift from the codes the analytics use.

    Returns:
        Columns ``Code``, ``Meaning`` and ``Documented``.
    """
    from src.schema import PAY_STATUS_UNDOCUMENTED_CODES

    rows = []
    for code in sorted(PAY_STATUS_LABELS):
        documented = code not in PAY_STATUS_UNDOCUMENTED_CODES
        rows.append(
            {
                "Code": code,
                "Meaning": PAY_STATUS_LABELS[code],
                "Documented": "Yes" if documented else "No - interpretation only",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


def k_selection_chart(
    scores: pd.DataFrame,
    recommended_k: int,
    elbow_k: int | None = None,
    height: int = 380,
    was_sampled: bool = False,
    sample_size: int = 0,
) -> go.Figure:
    """Plot the three K-selection diagnostics on shared axes.

    Silhouette (higher is better) and Davies-Bouldin (lower is better) share the
    left axis since both are dimensionless; inertia uses the right axis because it
    is on a completely different scale. Showing all three together is the point:
    where they disagree, the disagreement is visible rather than hidden behind a
    single asserted number.

    Args:
        scores: Output of :meth:`KSelectionResult.to_frame`.
        recommended_k: The K that was adopted.
        elbow_k: The inertia elbow, marked separately when it differs.
        height: Figure height in pixels.
        was_sampled: Whether silhouette used a sample.
        sample_size: Size of that sample.

    Returns:
        The figure.
    """
    if scores is None or len(scores) == 0:
        return empty_figure(
            "No K values could be evaluated for this selection.",
            height,
            "Choosing the number of segments",
        )

    figure = make_subplots(specs=[[{"secondary_y": True}]])

    figure.add_trace(
        go.Bar(
            x=scores["k"],
            y=scores["inertia"],
            name="Inertia (elbow)",
            marker_color=with_alpha(PALETTE["info"], 0.26),
            marker_line_width=0,
            hovertemplate="<b>K = %{x}</b><br>Inertia: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )

    figure.add_trace(
        go.Scatter(
            x=scores["k"],
            y=scores["silhouette"],
            name="Silhouette (higher better)",
            mode="lines+markers",
            line={"color": PALETTE["positive"], "width": 2.6},
            marker={"size": 9, "color": PALETTE["positive"]},
            hovertemplate="<b>K = %{x}</b><br>Silhouette: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=scores["k"],
            y=scores["davies_bouldin"],
            name="Davies-Bouldin (lower better)",
            mode="lines+markers",
            line={"color": PALETTE["attention"], "width": 2, "dash": "dot"},
            marker={"size": 7, "color": PALETTE["attention"]},
            hovertemplate="<b>K = %{x}</b><br>Davies-Bouldin: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    figure.add_vline(
        x=recommended_k,
        line_width=2,
        line_dash="dash",
        line_color=PALETTE["text_secondary"],
        annotation_text=f"Adopted K = {recommended_k}",
        annotation_position="top",
        annotation_font_size=11,
        annotation_font_color=PALETTE["text_secondary"],
    )

    if elbow_k is not None and elbow_k != recommended_k:
        figure.add_vline(
            x=elbow_k,
            line_width=1.4,
            line_dash="dot",
            line_color=PALETTE["text_faint"],
            annotation_text=f"Elbow K = {elbow_k}",
            annotation_position="bottom",
            annotation_font_size=10,
            annotation_font_color=PALETTE["text_faint"],
        )

    subtitle = (
        "Silhouette is the primary criterion. Inertia and Davies-Bouldin are shown "
        "alongside so any disagreement between them is visible."
    )
    if was_sampled and sample_size:
        subtitle += f" Silhouette computed on a fixed-seed sample of {sample_size:,}."

    figure.update_layout(
        title={
            "text": "Choosing the number of segments"
            f"<br><span style='font-size:11.5px;color:{PALETTE['text_muted']}'>"
            f"{subtitle}</span>",
            "x": 0,
            "xanchor": "left",
        },
        height=height,
        hovermode="x unified",
        margin={"l": 62, "r": 66, "t": 78, "b": 52},
    )
    figure.update_xaxes(
        title=_axis_title("Number of segments (K)"),
        dtick=1,
        showgrid=False,
    )
    figure.update_yaxes(
        title=_axis_title("Score"), secondary_y=False, showgrid=True,
        gridcolor=PALETTE["border"],
    )
    figure.update_yaxes(
        title=_axis_title("Inertia"), secondary_y=True, showgrid=False,
        tickformat=",.0f",
    )
    return figure


def segment_distribution_chart(
    profiles: Sequence[Any], height: int = 340
) -> go.Figure:
    """Plot how customers are distributed across segments.

    Args:
        profiles: :class:`src.segmentation.SegmentProfile` objects.
        height: Figure height in pixels.

    Returns:
        The figure.
    """
    if not profiles:
        return empty_figure(
            "No segments available.", height, "Portfolio distribution across segments"
        )

    ordered = sorted(profiles, key=lambda p: p.size, reverse=True)
    labels = [p.display_name for p in ordered]
    sizes = [p.size for p in ordered]
    shares = [p.share_pct for p in ordered]

    figure = go.Figure(
        go.Bar(
            x=sizes,
            y=labels,
            orientation="h",
            marker_color=[
                CATEGORICAL_SEQUENCE[p.cluster_id % len(CATEGORICAL_SEQUENCE)]
                for p in ordered
            ],
            marker_line_width=0,
            customdata=shares,
            text=[f"{s:,}  ({pct:.1f}%)" for s, pct in zip(sizes, shares)],
            textposition="auto",
            textfont={"size": 11.5},
            hovertemplate=(
                "<b>%{y}</b><br>Customers: %{x:,}<br>"
                "Share of portfolio: %{customdata:.2f}%<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={
            "text": "Portfolio distribution across segments",
            "x": 0,
            "xanchor": "left",
        },
        height=height,
        showlegend=False,
        margin={"l": 240, "r": 24, "t": 54, "b": 48},
    )
    figure.update_xaxes(title=_axis_title("Customers"), tickformat=",.0f")
    figure.update_yaxes(title=_axis_title(""), showgrid=False, autorange="reversed")
    return figure


def segment_observed_risk_chart(
    profiles: Sequence[Any],
    overall_rate: float | None = None,
    height: int = 360,
) -> go.Figure:
    """Plot the observed default rate per segment.

    Labelled as observed and descriptive throughout: clustering used no outcome
    information, so a segment's rate is a property of the customers who happen to
    sit in it, not an effect of membership.

    Args:
        profiles: :class:`src.segmentation.SegmentProfile` objects.
        overall_rate: Portfolio rate to mark as a baseline.
        height: Figure height in pixels.

    Returns:
        The figure.
    """
    scored = [p for p in profiles if p.observed_default_rate_pct is not None]
    if not scored:
        return empty_figure(
            "No outcome column is available, so no observed rate can be shown.",
            height,
            "Observed default rate by segment",
        )

    ordered = sorted(scored, key=lambda p: p.observed_default_rate_pct, reverse=True)
    labels = [p.display_name for p in ordered]
    rates = [p.observed_default_rate_pct for p in ordered]
    sizes = [p.size for p in ordered]

    # Colour by distance from the baseline so elevated segments stand out without
    # implying a predicted risk tier.
    if overall_rate is not None and np.isfinite(overall_rate):
        colours = [
            PALETTE["risk"]
            if r > overall_rate * 1.25
            else PALETTE["attention"]
            if r > overall_rate
            else PALETTE["positive"]
            for r in rates
        ]
    else:
        colours = [PALETTE["info"]] * len(rates)

    figure = go.Figure(
        go.Bar(
            x=labels,
            y=rates,
            marker_color=colours,
            marker_line_width=0,
            customdata=sizes,
            text=[f"{r:.2f}%" for r in rates],
            textposition="outside",
            textfont={"size": 11.5, "color": PALETTE["text_secondary"]},
            hovertemplate=(
                "<b>%{x}</b><br>Observed default rate: %{y:.2f}%<br>"
                "Customers: %{customdata:,}<extra></extra>"
            ),
        )
    )

    if overall_rate is not None and np.isfinite(overall_rate):
        figure.add_hline(
            y=overall_rate,
            line_width=1.6,
            line_dash="dash",
            line_color=PALETTE["text_muted"],
            annotation_text=f"Portfolio {overall_rate:.2f}%",
            annotation_position="top right",
            annotation_font_size=10.5,
            annotation_font_color=PALETTE["text_muted"],
        )

    figure.update_layout(
        title={
            "text": "Observed default rate by segment"
            f"<br><span style='font-size:11.5px;color:{PALETTE['text_muted']}'>"
            "Measured historical outcome, calculated after clustering. The outcome "
            "was not used to form the segments and this is not a prediction.</span>",
            "x": 0,
            "xanchor": "left",
        },
        height=height,
        showlegend=False,
        margin={"l": 62, "r": 24, "t": 78, "b": 92},
    )
    figure.update_xaxes(title=_axis_title(""), tickangle=-15)
    figure.update_yaxes(title=_axis_title("Observed default rate (%)"), ticksuffix="%")
    return figure


def segment_pca_map(
    projection: pd.DataFrame,
    explained_variance: Sequence[float],
    max_points: int = 6_000,
    height: int = 460,
) -> go.Figure:
    """Plot the 2D PCA projection of the clustering space.

    **This is a display device, not the model.** K-Means ran on the full scaled
    feature matrix; PCA compresses that space to two axes so it can be drawn. The
    subtitle states how much variance the two components retain, so a reader can
    see how lossy the view is and should not read overlap on this plot as overlap
    in the model.

    Args:
        projection: Frame with ``pc1``, ``pc2`` and ``segment`` columns.
        explained_variance: Explained variance ratio per component.
        max_points: Maximum markers before sampling.
        height: Figure height in pixels.

    Returns:
        The figure.
    """
    if projection is None or projection.empty:
        return empty_figure(
            "No projection available for this selection.", height, "Behaviour map"
        )

    working = projection
    note = ""
    if len(working) > max_points:
        original = len(working)
        working = working.sample(max_points, random_state=42)
        note = (
            f" Showing a random sample of {max_points:,} of {original:,} customers "
            "so individual points stay visible."
        )

    total_variance = float(sum(explained_variance) * 100.0) if explained_variance else 0.0
    pc1_pct = float(explained_variance[0] * 100.0) if len(explained_variance) > 0 else 0.0
    pc2_pct = float(explained_variance[1] * 100.0) if len(explained_variance) > 1 else 0.0

    figure = go.Figure()
    for segment, group in working.groupby("segment", observed=True):
        cluster_ids = group["cluster"].unique()
        colour_index = int(cluster_ids[0]) if len(cluster_ids) else 0
        figure.add_trace(
            go.Scattergl(
                x=group["pc1"],
                y=group["pc2"],
                mode="markers",
                name=str(segment),
                marker={
                    "size": 4.5,
                    "opacity": 0.45,
                    "color": CATEGORICAL_SEQUENCE[colour_index % len(CATEGORICAL_SEQUENCE)],
                    "line": {"width": 0},
                },
                hovertemplate=(
                    f"<b>{segment}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title={
            "text": "Behaviour map · 2D projection of the clustering space"
            f"<br><span style='font-size:11.5px;color:{PALETTE['text_muted']}'>"
            f"PCA for visualisation only. These two components retain "
            f"{total_variance:.1f}% of the variance in the feature space "
            f"({len(explained_variance)} of {len(explained_variance)} shown), so the "
            f"plot is a compressed view of the model, not the model itself.{note}"
            "</span>",
            "x": 0,
            "xanchor": "left",
        },
        height=height,
        legend={"itemsizing": "constant"},
        margin={"l": 62, "r": 24, "t": 86, "b": 56},
    )
    figure.update_xaxes(
        title=_axis_title(f"Principal component 1 ({pc1_pct:.1f}% of variance)"),
        showgrid=True,
        gridcolor=PALETTE["border"],
        zeroline=False,
    )
    figure.update_yaxes(
        title=_axis_title(f"Principal component 2 ({pc2_pct:.1f}% of variance)"),
        zeroline=False,
    )
    return figure


def segment_profile_comparison(
    profiles: Sequence[Any],
    feature_names: Sequence[str],
    feature_labels: Mapping[str, str] | None = None,
    height: int = 420,
) -> go.Figure:
    """Compare segments on a normalised scale.

    Uses a grouped bar chart rather than a radar. A radar with seven axes and
    several overlapping polygons becomes unreadable, and the brief allows radar
    only when it stays legible. Values are min-max normalised across segments so
    features on completely different units (NT$ versus a 0-1 ratio) can sit on one
    axis; the underlying values are in the profile table.

    Args:
        profiles: :class:`src.segmentation.SegmentProfile` objects.
        feature_names: Features to compare.
        feature_labels: Optional display names per feature.
        height: Figure height in pixels.

    Returns:
        The figure.
    """
    if not profiles or not feature_names:
        return empty_figure(
            "No segment profiles available to compare.", height, "Segment comparison"
        )

    labels_map = dict(feature_labels or {})
    axis_labels = [
        labels_map.get(f, f).replace("Avg ", "") for f in feature_names
    ]

    matrix = np.array(
        [[float(p.metrics.get(f, np.nan)) for f in feature_names] for p in profiles],
        dtype="float64",
    )

    # Min-max per feature across segments, so each axis spans 0-1.
    minimum = np.nanmin(matrix, axis=0)
    maximum = np.nanmax(matrix, axis=0)
    span = np.where((maximum - minimum) == 0, np.nan, maximum - minimum)
    normalised = (matrix - minimum) / span
    normalised = np.where(np.isnan(normalised), 0.5, normalised)

    figure = go.Figure()
    for row, profile in enumerate(profiles):
        figure.add_trace(
            go.Bar(
                name=profile.display_name,
                x=axis_labels,
                y=normalised[row],
                marker_color=CATEGORICAL_SEQUENCE[
                    profile.cluster_id % len(CATEGORICAL_SEQUENCE)
                ],
                marker_line_width=0,
                customdata=matrix[row],
                hovertemplate=(
                    f"<b>{profile.display_name}</b><br>%{{x}}<br>"
                    "Actual value: %{customdata:,.3f}<br>"
                    "Normalised: %{y:.2f}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title={
            "text": "Segment comparison across clustering features"
            f"<br><span style='font-size:11.5px;color:{PALETTE['text_muted']}'>"
            "Each feature is min-max normalised across segments so units as "
            "different as NT$ and a ratio share one axis. Hover for actual values."
            "</span>",
            "x": 0,
            "xanchor": "left",
        },
        barmode="group",
        bargap=0.26,
        height=height,
        margin={"l": 62, "r": 24, "t": 82, "b": 96},
    )
    figure.update_xaxes(title=_axis_title(""), tickangle=-22)
    figure.update_yaxes(
        title=_axis_title("Normalised value (0 = lowest segment, 1 = highest)"),
        range=[0, 1.08],
    )
    return figure


def scaler_comparison_chart(
    comparison_frame: pd.DataFrame,
    chosen: str,
    height: int = 340,
) -> go.Figure:
    """Plot silhouette by K for each scaler that was compared.

    Makes the scaling decision auditable: the reader sees both curves and can
    confirm the adopted scaler genuinely scored better rather than taking it on
    trust.

    Args:
        comparison_frame: Output of :meth:`ScalerComparison.to_frame`.
        chosen: Name of the adopted scaler.
        height: Figure height in pixels.

    Returns:
        The figure.
    """
    if comparison_frame is None or comparison_frame.empty:
        return empty_figure(
            "No scaler comparison available.", height, "Scaler comparison"
        )

    figure = go.Figure()
    for index, (name, group) in enumerate(comparison_frame.groupby("scaler", observed=True)):
        is_chosen = name == chosen
        figure.add_trace(
            go.Scatter(
                x=group["k"],
                y=group["silhouette"],
                name=f"{name}{' (adopted)' if is_chosen else ''}",
                mode="lines+markers",
                line={
                    "color": PALETTE["positive"] if is_chosen else PALETTE["neutral"],
                    "width": 2.8 if is_chosen else 1.8,
                    "dash": "solid" if is_chosen else "dot",
                },
                marker={
                    "size": 9 if is_chosen else 6,
                    "color": PALETTE["positive"] if is_chosen else PALETTE["neutral"],
                },
                hovertemplate=(
                    f"<b>{name}</b><br>K = %{{x}}<br>Silhouette: %{{y:.4f}}"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title={
            "text": "Scaler comparison · silhouette by K"
            f"<br><span style='font-size:11.5px;color:{PALETTE['text_muted']}'>"
            "K-Means is distance-based, so scaling decides which features dominate. "
            "Both options were measured on the same feature space.</span>",
            "x": 0,
            "xanchor": "left",
        },
        height=height,
        hovermode="x unified",
        margin={"l": 62, "r": 24, "t": 80, "b": 50},
    )
    figure.update_xaxes(title=_axis_title("Number of segments (K)"), dtick=1, showgrid=False)
    figure.update_yaxes(title=_axis_title("Silhouette score"))
    return figure
