"""
Reusable Plotly chart builders.

Returns ``plotly.graph_objects.Figure`` objects and imports no Streamlit, so
charts can be built and tested headlessly.

Design rules applied to every chart
-----------------------------------
* A title that states the question the chart answers, not just the column name.
* Axis labels with units (NT$, %, months).
* A hover template showing formatted values rather than raw floats.
* A colour-blind-safe palette. Semantic red/amber/green is reserved for risk, so
  colour never implies "bad" where none is meant.
* An empty-state placeholder with an explanation whenever there is nothing to
  plot, so a filtered-out selection never produces a blank rectangle.

No chart is produced for decoration. Each builder maps to a specific analytical
question asked elsewhere in the platform.
"""

from __future__ import annotations

from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.logging_setup import get_logger
from src.schema import DATASET_CURRENCY

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

#: Dark financial-intelligence palette, kept in step with :mod:`src.ui.theme`.
#: Keys are stable; only the hex values changed when the product moved to dark
#: mode, so callers and tests that reference keys are unaffected.
COLOURS: Final[Mapping[str, str]] = {
    "primary": "#4d94ff",
    "primary_light": "#7fb2ff",
    "accent": "#e8a33d",
    "neutral": "#64748b",
    "neutral_light": "#39485c",
    "surface": "rgba(0,0,0,0)",
    "grid": "#222d3b",
    "text": "#e8eef6",
    "text_muted": "#6f7f92",
    # Semantic, reserved for risk and outcome only.
    "risk_low": "#16b47f",
    "risk_medium": "#e8a33d",
    "risk_high": "#ef5f5f",
    "positive": "#16b47f",
    "negative": "#ef5f5f",
}

#: Qualitative sequence chosen to remain distinguishable with common forms of
#: colour-vision deficiency on a dark surface.
CATEGORICAL_SEQUENCE: Final[tuple[str, ...]] = (
    "#4d94ff",
    "#e8a33d",
    "#16b47f",
    "#7c6cf0",
    "#ef5f5f",
    "#3fbfc4",
    "#b47fd6",
    "#8b98a9",
)

#: Single-hue sequential scale; safe for ordered magnitude.
SEQUENTIAL_SCALE: Final[tuple[str, ...]] = (
    "#111a24",
    "#1b3a52",
    "#255a7d",
    "#3480ae",
    "#4d94ff",
    "#7fb2ff",
)

#: Diverging scale for correlations. Red-neutral-blue reads as
#: negative-none-positive and survives greyscale printing.
DIVERGING_SCALE: Final[tuple[tuple[float, str], ...]] = (
    (0.0, "#ef5f5f"),
    (0.5, "#1b2430"),
    (1.0, "#4d94ff"),
)

RISK_COLOURS: Final[Mapping[str, str]] = {
    "Low": COLOURS["risk_low"],
    "Medium": COLOURS["risk_medium"],
    "High": COLOURS["risk_high"],
    "Low Risk": COLOURS["risk_low"],
    "Medium Risk": COLOURS["risk_medium"],
    "High Risk": COLOURS["risk_high"],
}

BASE_FONT: Final[str] = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
)


def apply_theme(
    figure: go.Figure,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 380,
    show_legend: bool = True,
    subtitle: str = "",
) -> go.Figure:
    """Apply the shared visual theme to a figure.

    Args:
        figure: Figure to style.
        title: Chart title, ideally phrased as the question answered.
        x_title: X-axis label, including units.
        y_title: Y-axis label, including units.
        height: Height in pixels.
        show_legend: Whether to show the legend.
        subtitle: Optional smaller line under the title, for caveats.

    Returns:
        The same figure, styled.
    """
    full_title = title
    if subtitle:
        full_title = (
            f"{title}<br><span style='font-size:12px;color:{COLOURS['text_muted']}'>"
            f"{subtitle}</span>"
        )

    figure.update_layout(
        title={
            "text": full_title,
            "font": {"size": 15, "color": COLOURS["text"], "family": BASE_FONT},
            "x": 0,
            "xanchor": "left",
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        showlegend=show_legend,
        # Transparent so the figure sits on the themed panel behind it.
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": BASE_FONT, "size": 12, "color": COLOURS["text_muted"]},
        margin={"l": 60, "r": 24, "t": 70 if subtitle else 54, "b": 52},
        hoverlabel={
            "bgcolor": "#18212c",
            "bordercolor": "#2e3c4d",
            "font_size": 12,
            "font_family": BASE_FONT,
            "font_color": COLOURS["text"],
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11.5, "color": COLOURS["text_muted"]},
        },
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor=COLOURS["grid"],
        ticks="outside",
        tickcolor=COLOURS["grid"],
        tickfont={"size": 11.5, "color": COLOURS["text_muted"]},
        title_font={"size": 12, "color": COLOURS["text_muted"]},
        automargin=True,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=COLOURS["grid"],
        zeroline=True,
        zerolinecolor=COLOURS["neutral_light"],
        linecolor=COLOURS["grid"],
        tickfont={"size": 11.5, "color": COLOURS["text_muted"]},
        title_font={"size": 12, "color": COLOURS["text_muted"]},
        automargin=True,
    )
    return figure


def empty_figure(message: str, height: int = 380, title: str = "") -> go.Figure:
    """Build a placeholder figure carrying an explanation.

    Used wherever data is missing or a filter matched nothing, so the user is
    told why a chart is absent rather than shown an empty axis.

    Args:
        message: Explanation to display.
        height: Height in pixels.
        title: Optional title retained above the placeholder.

    Returns:
        A figure containing only the message.
    """
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 13, "color": COLOURS["text_muted"], "family": BASE_FONT},
        align="center",
    )
    figure.update_layout(
        title={"text": title, "font": {"size": 16, "color": COLOURS["text"]}, "x": 0},
        height=height,
        plot_bgcolor=COLOURS["surface"],
        paper_bgcolor=COLOURS["surface"],
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 40, "r": 24, "t": 56, "b": 40},
    )
    return figure


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def _currency_hover(label: str, value_ref: str = "y") -> str:
    """Hover template for a monetary axis."""
    return f"<b>%{{x}}</b><br>{label}: {DATASET_CURRENCY}%{{{value_ref}:,.0f}}<extra></extra>"


def _percent_hover(label: str, value_ref: str = "y") -> str:
    """Hover template for a percentage axis."""
    return f"<b>%{{x}}</b><br>{label}: %{{{value_ref}:.2f}}%<extra></extra>"


def _is_empty(data: pd.DataFrame | pd.Series | None) -> bool:
    """True when there is nothing plottable."""
    return data is None or len(data) == 0


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #


def distribution_chart(
    bins: pd.DataFrame,
    title: str,
    x_title: str,
    is_currency: bool = False,
    median_value: float | None = None,
    mean_value: float | None = None,
    height: int = 380,
) -> go.Figure:
    """Plot a binned distribution, optionally marking the mean and median.

    Marking both matters here: the monetary columns are strongly right-skewed, so
    showing the gap between mean and median makes the skew visible rather than
    something the reader has to infer.

    Args:
        bins: Output of :func:`src.analytics.distribution_bins`.
        title: Chart title.
        x_title: X-axis label with units.
        is_currency: Format values as currency.
        median_value: Median to mark with a vertical line.
        mean_value: Mean to mark with a vertical line.
        height: Height in pixels.

    Returns:
        The figure.
    """
    if _is_empty(bins) or "bin_center" not in getattr(bins, "columns", []):
        return empty_figure(
            "No distribution to display for the current selection.", height, title
        )

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=bins["bin_center"],
            y=bins["count"],
            marker_color=COLOURS["primary"],
            marker_line_width=0,
            name="Clients",
            hovertemplate=(
                f"Range: {DATASET_CURRENCY}%{{customdata[0]:,.0f}} to "
                f"{DATASET_CURRENCY}%{{customdata[1]:,.0f}}"
                if is_currency
                else "Range: %{customdata[0]:,.2f} to %{customdata[1]:,.2f}"
            )
            + "<br>Clients: %{y:,}<br>Share: %{customdata[2]:.2f}%<extra></extra>",
            customdata=np.stack(
                [bins["bin_left"], bins["bin_right"], bins["percentage"]], axis=-1
            ),
        )
    )

    for value, label, colour, dash in (
        (median_value, "Median", COLOURS["accent"], "solid"),
        (mean_value, "Mean", COLOURS["neutral"], "dash"),
    ):
        if value is not None and np.isfinite(value):
            formatted = (
                f"{DATASET_CURRENCY}{value:,.0f}" if is_currency else f"{value:,.2f}"
            )
            figure.add_vline(
                x=value,
                line_width=2,
                line_dash=dash,
                line_color=colour,
                annotation_text=f"{label}: {formatted}",
                annotation_position="top",
                annotation_font_size=11,
                annotation_font_color=colour,
            )

    subtitle = ""
    if median_value is not None and mean_value is not None and median_value != 0:
        gap = (mean_value - median_value) / abs(median_value) * 100.0
        if abs(gap) > 10:
            subtitle = (
                f"Mean exceeds median by {gap:.0f}%, indicating a right-skewed "
                "distribution - the median is the better guide to a typical client."
            )

    return apply_theme(
        figure, title, x_title, "Number of clients", height, show_legend=False, subtitle=subtitle
    )


def category_bar_chart(
    data: pd.DataFrame,
    category_column: str,
    value_column: str,
    title: str,
    x_title: str = "",
    y_title: str = "",
    is_percentage: bool = False,
    is_currency: bool = False,
    reference_line: float | None = None,
    reference_label: str = "Portfolio average",
    horizontal: bool = False,
    height: int = 380,
    colour_map: Mapping[str, str] | None = None,
) -> go.Figure:
    """Plot a value per category, optionally against a reference line.

    The reference line is what turns a bar chart into an answer: it shows which
    categories sit above or below the portfolio baseline instead of leaving the
    reader to eyeball it.

    Args:
        data: Dataframe holding the category and value columns.
        category_column: Column of category labels.
        value_column: Column of values.
        title: Chart title.
        x_title: X-axis label.
        y_title: Y-axis label.
        is_percentage: Format values as percentages.
        is_currency: Format values as currency.
        reference_line: Baseline value to draw.
        reference_label: Label for the baseline.
        horizontal: Draw horizontal bars, better for long category names.
        height: Height in pixels.
        colour_map: Optional per-category colour overrides, e.g. risk tiers.

    Returns:
        The figure.
    """
    if _is_empty(data) or category_column not in data.columns or value_column not in data.columns:
        return empty_figure("No category data available for the current selection.", height, title)

    categories = data[category_column].astype("string").tolist()
    values = pd.to_numeric(data[value_column], errors="coerce").tolist()

    if colour_map:
        colours = [colour_map.get(str(c), COLOURS["primary"]) for c in categories]
    else:
        colours = [COLOURS["primary"]] * len(categories)

    if is_currency:
        text = [f"{DATASET_CURRENCY}{v:,.0f}" if pd.notna(v) else "" for v in values]
        hover_value = f"{DATASET_CURRENCY}%{{value:,.0f}}"
    elif is_percentage:
        text = [f"{v:.1f}%" if pd.notna(v) else "" for v in values]
        hover_value = "%{value:.2f}%"
    else:
        text = [f"{v:,.2f}" if pd.notna(v) else "" for v in values]
        hover_value = "%{value:,.2f}"

    figure = go.Figure()
    if horizontal:
        figure.add_trace(
            go.Bar(
                y=categories,
                x=values,
                orientation="h",
                marker_color=colours,
                marker_line_width=0,
                text=text,
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>"
                + hover_value.replace("value", "x")
                + "<extra></extra>",
            )
        )
    else:
        figure.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color=colours,
                marker_line_width=0,
                text=text,
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>"
                + hover_value.replace("value", "y")
                + "<extra></extra>",
            )
        )

    if reference_line is not None and np.isfinite(reference_line):
        formatted = (
            f"{DATASET_CURRENCY}{reference_line:,.0f}"
            if is_currency
            else (f"{reference_line:.2f}%" if is_percentage else f"{reference_line:,.2f}")
        )
        if horizontal:
            figure.add_vline(
                x=reference_line,
                line_width=2,
                line_dash="dash",
                line_color=COLOURS["accent"],
                annotation_text=f"{reference_label}: {formatted}",
                annotation_position="top",
                annotation_font_size=11,
            )
        else:
            figure.add_hline(
                y=reference_line,
                line_width=2,
                line_dash="dash",
                line_color=COLOURS["accent"],
                annotation_text=f"{reference_label}: {formatted}",
                annotation_position="top right",
                annotation_font_size=11,
            )

    return apply_theme(figure, title, x_title, y_title, height, show_legend=False)


# --------------------------------------------------------------------------- #
# Trends
# --------------------------------------------------------------------------- #


def trend_line_chart(
    points: pd.DataFrame,
    title: str,
    y_title: str,
    is_currency: bool = False,
    is_percentage: bool = False,
    subtitle: str = "",
    anomaly_labels: Sequence[str] = (),
    height: int = 380,
) -> go.Figure:
    """Plot a measure across the monthly panel in chronological order.

    Args:
        points: Output of :meth:`src.trends.TrendResult.to_frame`.
        title: Chart title.
        y_title: Y-axis label with units.
        is_currency: Format values as currency.
        is_percentage: Format values as percentages.
        subtitle: Caveat line under the title.
        anomaly_labels: Period labels to highlight.
        height: Height in pixels.

    Returns:
        The figure.
    """
    if _is_empty(points) or "month_label" not in getattr(points, "columns", []):
        return empty_figure(
            "Trend analysis is unavailable for the current selection.", height, title
        )

    ordered = points.sort_values("month_order")
    labels = ordered["month_label"].astype("string").tolist()
    values = pd.to_numeric(ordered["value"], errors="coerce").tolist()

    if is_currency:
        hover = f"<b>%{{x}}</b><br>{y_title}: {DATASET_CURRENCY}%{{y:,.0f}}<extra></extra>"
    elif is_percentage:
        hover = f"<b>%{{x}}</b><br>{y_title}: %{{y:.2f}}%<extra></extra>"
    else:
        hover = f"<b>%{{x}}</b><br>{y_title}: %{{y:,.2f}}<extra></extra>"

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            line={"color": COLOURS["primary"], "width": 2.5},
            marker={"size": 8, "color": COLOURS["primary"]},
            name=y_title,
            hovertemplate=hover,
        )
    )

    if anomaly_labels:
        flagged = [(l, v) for l, v in zip(labels, values) if l in set(anomaly_labels)]
        if flagged:
            figure.add_trace(
                go.Scatter(
                    x=[l for l, _ in flagged],
                    y=[v for _, v in flagged],
                    mode="markers",
                    marker={
                        "size": 14,
                        "color": "rgba(0,0,0,0)",
                        "line": {"color": COLOURS["accent"], "width": 2.5},
                    },
                    name="Worth checking",
                    hovertemplate=(
                        "<b>%{x}</b><br>Unusually large change versus the other "
                        "periods<extra></extra>"
                    ),
                )
            )

    return apply_theme(
        figure,
        title,
        "Month (oldest to most recent)",
        y_title,
        height,
        show_legend=bool(anomaly_labels),
        subtitle=subtitle,
    )


def multi_trend_chart(
    series: Mapping[str, pd.DataFrame],
    title: str,
    y_title: str,
    is_percentage: bool = False,
    height: int = 400,
    subtitle: str = "",
) -> go.Figure:
    """Overlay several measures that share a unit on one axis.

    Args:
        series: ``{label: points_dataframe}``.
        title: Chart title.
        y_title: Shared Y-axis label.
        is_percentage: Format values as percentages.
        height: Height in pixels.
        subtitle: Caveat line under the title.

    Returns:
        The figure.
    """
    plottable = {
        label: frame
        for label, frame in series.items()
        if not _is_empty(frame) and "month_label" in getattr(frame, "columns", [])
    }
    if not plottable:
        return empty_figure("No trend series available to compare.", height, title)

    figure = go.Figure()
    for index, (label, frame) in enumerate(plottable.items()):
        ordered = frame.sort_values("month_order")
        colour = CATEGORICAL_SEQUENCE[index % len(CATEGORICAL_SEQUENCE)]
        suffix = "%{y:.2f}%" if is_percentage else "%{y:,.2f}"
        figure.add_trace(
            go.Scatter(
                x=ordered["month_label"].astype("string").tolist(),
                y=pd.to_numeric(ordered["value"], errors="coerce").tolist(),
                mode="lines+markers",
                name=label,
                line={"color": colour, "width": 2.5},
                marker={"size": 7, "color": colour},
                hovertemplate=f"<b>%{{x}}</b><br>{label}: {suffix}<extra></extra>",
            )
        )

    return apply_theme(
        figure,
        title,
        "Month (oldest to most recent)",
        y_title,
        height,
        show_legend=True,
        subtitle=subtitle,
    )


# --------------------------------------------------------------------------- #
# Relationships
# --------------------------------------------------------------------------- #


def correlation_heatmap(
    matrix: pd.DataFrame,
    title: str = "How do the financial measures relate to one another?",
    height: int = 560,
    method_label: str = "Spearman",
    max_columns: int = 20,
) -> go.Figure:
    """Render a correlation matrix as a heatmap.

    Args:
        matrix: Square correlation matrix.
        title: Chart title.
        height: Height in pixels.
        method_label: Correlation method, shown in the subtitle.
        max_columns: Cap on displayed columns; a matrix beyond this is unreadable.

    Returns:
        The figure.
    """
    if _is_empty(matrix) or matrix.shape[0] < 2:
        return empty_figure(
            "At least two numeric columns are needed to show correlations.", height, title
        )

    working = matrix
    subtitle = (
        f"{method_label} rank correlation. Association only - this does not imply causation."
    )
    if working.shape[0] > max_columns:
        # Keep the columns with the strongest average association; showing 60
        # labels would be illegible.
        ranked = working.abs().mean().sort_values(ascending=False).head(max_columns).index
        working = working.loc[ranked, ranked]
        subtitle = (
            f"{method_label} rank correlation, showing the {max_columns} most "
            "strongly associated measures. Association only - not causation."
        )

    figure = go.Figure(
        data=go.Heatmap(
            z=working.to_numpy(),
            x=[str(c) for c in working.columns],
            y=[str(i) for i in working.index],
            colorscale=[list(stop) for stop in DIVERGING_SCALE],
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar={"title": "r", "thickness": 14, "len": 0.8},
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(yaxis={"autorange": "reversed"})
    themed = apply_theme(figure, title, "", "", height, show_legend=False, subtitle=subtitle)
    themed.update_xaxes(tickangle=-45, showgrid=False)
    themed.update_yaxes(showgrid=False)
    return themed


def scatter_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_title: str = "",
    y_title: str = "",
    colour_column: str | None = None,
    max_points: int = 5_000,
    height: int = 420,
) -> go.Figure:
    """Plot the relationship between two numeric columns.

    Large frames are randomly sampled with a fixed seed: 30,000 overlapping
    markers hide the pattern they are meant to reveal, and the sampling is
    disclosed in the subtitle rather than done silently.

    Args:
        data: Dataframe to plot.
        x_column: X-axis column.
        y_column: Y-axis column.
        title: Chart title.
        x_title: X-axis label with units.
        y_title: Y-axis label with units.
        colour_column: Optional categorical column for colouring.
        max_points: Maximum markers before sampling.
        height: Height in pixels.

    Returns:
        The figure.
    """
    needed = [c for c in (x_column, y_column) if c not in getattr(data, "columns", [])]
    if _is_empty(data) or needed:
        return empty_figure(
            f"Cannot plot: missing column(s) {', '.join(needed) or 'no data'}.", height, title
        )

    working = data[[x_column, y_column] + ([colour_column] if colour_column and colour_column in data.columns else [])].copy()
    working[x_column] = pd.to_numeric(working[x_column], errors="coerce")
    working[y_column] = pd.to_numeric(working[y_column], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_column, y_column])

    if working.empty:
        return empty_figure("No complete pairs of values available to plot.", height, title)

    subtitle = ""
    if len(working) > max_points:
        original = len(working)
        working = working.sample(max_points, random_state=42)
        subtitle = (
            f"Showing a random sample of {max_points:,} of {original:,} clients to "
            "keep the pattern legible."
        )

    figure = go.Figure()
    if colour_column and colour_column in working.columns:
        for index, (label, group) in enumerate(working.groupby(colour_column, observed=True)):
            figure.add_trace(
                go.Scattergl(
                    x=group[x_column],
                    y=group[y_column],
                    mode="markers",
                    name=str(label),
                    marker={
                        "size": 5,
                        "opacity": 0.55,
                        "color": CATEGORICAL_SEQUENCE[index % len(CATEGORICAL_SEQUENCE)],
                    },
                    hovertemplate=(
                        f"{x_title or x_column}: %{{x:,.2f}}<br>"
                        f"{y_title or y_column}: %{{y:,.2f}}<extra>{label}</extra>"
                    ),
                )
            )
    else:
        figure.add_trace(
            go.Scattergl(
                x=working[x_column],
                y=working[y_column],
                mode="markers",
                marker={"size": 5, "opacity": 0.5, "color": COLOURS["primary"]},
                hovertemplate=(
                    f"{x_title or x_column}: %{{x:,.2f}}<br>"
                    f"{y_title or y_column}: %{{y:,.2f}}<extra></extra>"
                ),
            )
        )

    return apply_theme(
        figure,
        title,
        x_title or x_column,
        y_title or y_column,
        height,
        show_legend=bool(colour_column),
        subtitle=subtitle,
    )


def box_comparison_chart(
    data: pd.DataFrame,
    category_column: str,
    value_column: str,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 400,
    max_categories: int = 12,
) -> go.Figure:
    """Compare a numeric distribution across categories with box plots.

    Shows the whole distribution rather than only the mean, which matters for
    skewed financial measures where two groups can share a mean and look nothing
    alike.

    Args:
        data: Dataframe to plot.
        category_column: Categorical column.
        value_column: Numeric column.
        title: Chart title.
        x_title: X-axis label.
        y_title: Y-axis label with units.
        height: Height in pixels.
        max_categories: Cap on categories displayed.

    Returns:
        The figure.
    """
    needed = [
        c for c in (category_column, value_column) if c not in getattr(data, "columns", [])
    ]
    if _is_empty(data) or needed:
        return empty_figure(
            f"Cannot plot: missing column(s) {', '.join(needed) or 'no data'}.", height, title
        )

    working = data[[category_column, value_column]].copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan).dropna()

    if working.empty:
        return empty_figure("No values available to compare.", height, title)

    categories = list(working[category_column].astype("string").unique())[:max_categories]

    figure = go.Figure()
    for index, category in enumerate(categories):
        subset = working.loc[working[category_column].astype("string") == category, value_column]
        figure.add_trace(
            go.Box(
                y=subset,
                name=str(category),
                marker_color=CATEGORICAL_SEQUENCE[index % len(CATEGORICAL_SEQUENCE)],
                boxmean=True,
                hovertemplate=f"<b>{category}</b><br>%{{y:,.2f}}<extra></extra>",
            )
        )

    return apply_theme(
        figure,
        title,
        x_title or category_column,
        y_title or value_column,
        height,
        show_legend=False,
        subtitle="Box shows the interquartile range; the dashed line marks the mean.",
    )


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


def donut_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    centre_label: str = "",
    height: int = 340,
    colour_map: Mapping[str, str] | None = None,
) -> go.Figure:
    """Show composition as a donut.

    Args:
        labels: Segment labels.
        values: Segment values.
        title: Chart title.
        centre_label: Text for the hole, typically the total.
        height: Height in pixels.
        colour_map: Optional per-label colour overrides.

    Returns:
        The figure.
    """
    if not len(labels) or not len(values) or float(np.nansum(values)) == 0:
        return empty_figure("No composition data available.", height, title)

    if colour_map:
        colours = [colour_map.get(str(l), COLOURS["primary"]) for l in labels]
    else:
        colours = [CATEGORICAL_SEQUENCE[i % len(CATEGORICAL_SEQUENCE)] for i in range(len(labels))]

    figure = go.Figure(
        data=go.Pie(
            labels=[str(l) for l in labels],
            values=list(values),
            hole=0.58,
            # Solid dark separator so adjacent wedges stay distinguishable.
            marker={"colors": colours, "line": {"color": "#0a0e14", "width": 2}},
            textinfo="percent",
            texttemplate="%{percent:.1%}",
            hovertemplate="<b>%{label}</b><br>%{value:,} clients<br>%{percent:.2%}<extra></extra>",
            sort=False,
        )
    )
    if centre_label:
        figure.add_annotation(
            text=centre_label,
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 15, "color": COLOURS["text"], "family": BASE_FONT},
        )
    return apply_theme(figure, title, "", "", height, show_legend=True)


def stacked_bar_chart(
    data: pd.DataFrame,
    title: str,
    x_title: str = "",
    y_title: str = "Share of clients (%)",
    normalise: bool = True,
    height: int = 400,
    colour_map: Mapping[str, str] | None = None,
) -> go.Figure:
    """Plot a cross-tabulation as a stacked bar chart.

    Args:
        data: Cross-tab with categories as the index and series as columns.
        title: Chart title.
        x_title: X-axis label.
        y_title: Y-axis label.
        normalise: Convert each row to percentages, which makes composition
            comparable across groups of very different sizes.
        height: Height in pixels.
        colour_map: Optional per-series colour overrides.

    Returns:
        The figure.
    """
    if _is_empty(data) or data.shape[1] == 0:
        return empty_figure("No data available for this breakdown.", height, title)

    working = data.copy()
    if normalise:
        totals = working.sum(axis=1)
        working = working.div(totals.where(totals != 0, np.nan), axis=0) * 100.0

    figure = go.Figure()
    for index, column in enumerate(working.columns):
        label = str(column)
        colour = (
            colour_map.get(label)
            if colour_map and label in colour_map
            else CATEGORICAL_SEQUENCE[index % len(CATEGORICAL_SEQUENCE)]
        )
        suffix = "%{y:.2f}%" if normalise else "%{y:,.0f}"
        figure.add_trace(
            go.Bar(
                x=[str(i) for i in working.index],
                y=working[column],
                name=label,
                marker_color=colour,
                marker_line_width=0,
                hovertemplate=f"<b>%{{x}}</b><br>{label}: {suffix}<extra></extra>",
            )
        )

    figure.update_layout(barmode="stack")
    return apply_theme(figure, title, x_title, y_title, height, show_legend=True)


# --------------------------------------------------------------------------- #
# Driver / importance
# --------------------------------------------------------------------------- #


def importance_chart(
    features: Sequence[str],
    values: Sequence[float],
    title: str = "Which measures are most strongly associated with default?",
    x_title: str = "Effect size (absolute)",
    height: int = 420,
    subtitle: str = (
        "Observed statistical association only. This does not demonstrate causation."
    ),
) -> go.Figure:
    """Plot ranked feature importances or effect sizes as horizontal bars.

    The default subtitle carries the causation caveat, so the disclaimer travels
    with the chart rather than living only in the surrounding text.

    Args:
        features: Feature names.
        values: Corresponding magnitudes.
        title: Chart title.
        x_title: X-axis label.
        height: Height in pixels.
        subtitle: Caveat line under the title.

    Returns:
        The figure.
    """
    if not len(features) or not len(values):
        return empty_figure("No driver analysis results available.", height, title)

    # Ascending so the largest bar sits at the top of a horizontal chart.
    order = np.argsort(values)
    ordered_features = [str(features[i]) for i in order]
    ordered_values = [float(values[i]) for i in order]

    figure = go.Figure(
        data=go.Bar(
            x=ordered_values,
            y=ordered_features,
            orientation="h",
            marker_color=COLOURS["primary"],
            marker_line_width=0,
            text=[f"{v:.3f}" for v in ordered_values],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Effect size: %{x:.4f}<extra></extra>",
        )
    )
    themed = apply_theme(figure, title, x_title, "", height, show_legend=False, subtitle=subtitle)
    themed.update_yaxes(showgrid=False)
    themed.update_xaxes(showgrid=True, gridcolor=COLOURS["grid"])
    return themed


def outcome_rate_chart(
    data: pd.DataFrame,
    category_column: str,
    rate_column: str = "rate_pct",
    count_column: str = "count",
    overall_rate: float | None = None,
    title: str = "How does the default rate vary across groups?",
    x_title: str = "",
    height: int = 400,
    min_group_size: int = 30,
) -> go.Figure:
    """Plot an outcome rate per category, greying out undersized groups.

    Undersized groups are shown but visually de-emphasised, so a 12-client
    category cannot be misread as the riskiest segment while still remaining
    visible for completeness.

    Args:
        data: Table from :func:`src.analytics.category_outcome_rates`.
        category_column: Column or index name holding the categories.
        rate_column: Column holding the rate as a percentage.
        count_column: Column holding the group size.
        overall_rate: Portfolio baseline to draw.
        title: Chart title.
        x_title: X-axis label.
        height: Height in pixels.
        min_group_size: Threshold below which a group is de-emphasised.

    Returns:
        The figure.
    """
    if _is_empty(data):
        return empty_figure("No group comparison available.", height, title)

    working = data.reset_index() if category_column not in data.columns else data.copy()
    if category_column not in working.columns or rate_column not in working.columns:
        return empty_figure(
            f"Expected columns '{category_column}' and '{rate_column}' are missing.",
            height,
            title,
        )

    categories = working[category_column].astype("string").tolist()
    rates = pd.to_numeric(working[rate_column], errors="coerce").tolist()
    counts = (
        pd.to_numeric(working[count_column], errors="coerce").tolist()
        if count_column in working.columns
        else [min_group_size] * len(categories)
    )

    colours = [
        COLOURS["primary"] if (c or 0) >= min_group_size else COLOURS["neutral_light"]
        for c in counts
    ]
    n_small = sum(1 for c in counts if (c or 0) < min_group_size)

    figure = go.Figure(
        data=go.Bar(
            x=categories,
            y=rates,
            marker_color=colours,
            marker_line_width=0,
            text=[f"{r:.1f}%" if pd.notna(r) else "" for r in rates],
            textposition="auto",
            customdata=counts,
            hovertemplate=(
                "<b>%{x}</b><br>Rate: %{y:.2f}%<br>Clients: %{customdata:,}<extra></extra>"
            ),
        )
    )

    if overall_rate is not None and np.isfinite(overall_rate):
        figure.add_hline(
            y=overall_rate,
            line_width=2,
            line_dash="dash",
            line_color=COLOURS["accent"],
            annotation_text=f"Overall: {overall_rate:.2f}%",
            annotation_position="top right",
            annotation_font_size=11,
        )

    subtitle = ""
    if n_small:
        subtitle = (
            f"{n_small} group(s) below {min_group_size} clients are shown in grey; "
            "their rates are too noisy to rank."
        )

    return apply_theme(
        figure, title, x_title or category_column, "Default rate (%)", height,
        show_legend=False, subtitle=subtitle,
    )


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #


def quality_score_gauge(
    score: float,
    title: str = "Data quality score",
    height: int = 260,
    subtitle: str = "Project-defined heuristic, not an industry-standard metric.",
) -> go.Figure:
    """Render the data-quality score as a gauge.

    Args:
        score: Score in 0-100.
        title: Chart title.
        height: Height in pixels.
        subtitle: Caveat line; defaults to stating the score is project-defined.

    Returns:
        The figure.
    """
    if score is None or not np.isfinite(score):
        return empty_figure("Data quality score unavailable.", height, title)

    if score >= 90:
        bar_colour = COLOURS["risk_low"]
    elif score >= 75:
        bar_colour = COLOURS["primary"]
    elif score >= 60:
        bar_colour = COLOURS["risk_medium"]
    else:
        bar_colour = COLOURS["risk_high"]

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            number={"suffix": " / 100", "font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLOURS["grid"]},
                "bar": {"color": bar_colour, "thickness": 0.7},
                "bgcolor": "#111a24",
                "borderwidth": 0,
                # Dark band backgrounds: poor -> excellent, left to right.
                "steps": [
                    {"range": [0, 60], "color": "#2a1618"},
                    {"range": [60, 75], "color": "#2a2216"},
                    {"range": [75, 90], "color": "#161f2a"},
                    {"range": [90, 100], "color": "#122420"},
                ],
            },
        )
    )
    figure.update_layout(
        title={
            "text": (
                f"{title}<br><span style='font-size:11px;color:{COLOURS['text_muted']}'>"
                f"{subtitle}</span>"
            ),
            "font": {"size": 15, "color": COLOURS["text"], "family": BASE_FONT},
            "x": 0,
        },
        height=height,
        paper_bgcolor=COLOURS["surface"],
        font={"family": BASE_FONT, "color": COLOURS["text"]},
        margin={"l": 24, "r": 24, "t": 72, "b": 16},
    )
    return figure


def missing_values_chart(
    missing_by_column: Mapping[str, int],
    n_rows: int,
    title: str = "Where are values missing?",
    height: int = 360,
    top_n: int = 20,
) -> go.Figure:
    """Plot missing-value counts per column.

    Args:
        missing_by_column: ``{column: missing_count}``.
        n_rows: Total rows, used to compute percentages.
        title: Chart title.
        height: Height in pixels.
        top_n: Maximum columns to display.

    Returns:
        The figure, or a positive empty state when nothing is missing.
    """
    present = {k: v for k, v in missing_by_column.items() if v > 0}
    if not present:
        return empty_figure(
            "No missing values in any column - the dataset is fully populated.",
            height,
            title,
        )

    ordered = sorted(present.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    columns = [k for k, _ in ordered]
    counts = [v for _, v in ordered]
    percentages = [(v / n_rows * 100.0) if n_rows else 0.0 for v in counts]

    figure = go.Figure(
        data=go.Bar(
            x=counts,
            y=columns,
            orientation="h",
            marker_color=COLOURS["risk_medium"],
            marker_line_width=0,
            customdata=percentages,
            text=[f"{p:.1f}%" for p in percentages],
            textposition="auto",
            hovertemplate=(
                "<b>%{y}</b><br>Missing: %{x:,} rows<br>%{customdata:.2f}% of rows"
                "<extra></extra>"
            ),
        )
    )
    themed = apply_theme(
        figure, title, "Rows with a missing value", "", height, show_legend=False
    )
    themed.update_yaxes(autorange="reversed", showgrid=False)
    return themed
