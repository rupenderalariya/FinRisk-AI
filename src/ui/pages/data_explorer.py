"""
Data Explorer.

Deliberately the least visually dominant page: its job is to let an analyst
inspect records and verify numbers, not to impress. Structure, statistics and
column documentation sit in collapsed expanders so the table itself stays the
focus.

Performance note: the table is paginated and only the requested slice is handed
to Streamlit, so selecting 30,000 rows never ships 30,000 rows to the browser.
"""

from __future__ import annotations

import pandas as pd

import streamlit as st

from src.analytics import summarise_numeric, summary_frame
from src.schema import describe_column
from src.ui import components
from src.ui.context import PageContext

#: Rows shown per page in the record table.
PAGE_SIZES: tuple[int, ...] = (25, 50, 100, 250)

#: Columns shown by default: identity, capacity, the outcome, and the headline
#: engineered behaviour measures. The full set is available on request.
DEFAULT_COLUMNS: tuple[str, ...] = (
    "client_id",
    "age",
    "credit_limit",
    "education",
    "marriage",
    "utilisation_latest",
    "repayment_ratio_capped_mean",
    "delinquent_months_count",
    "max_delinquency",
    "default_next_month",
)


def _render_overview(context: PageContext) -> None:
    """Dataset dimensions and composition."""
    data = context.data
    profile = context.pipeline.processed_profile

    numeric = data.select_dtypes(include="number").shape[1]
    categorical = data.select_dtypes(include=["object", "string", "category"]).shape[1]
    boolean = data.select_dtypes(include="bool").shape[1]

    components.stat_row(
        [
            ("Rows in selection", f"{len(data):,}"),
            ("Total columns", f"{len(data.columns)}"),
            ("Numeric", f"{numeric}"),
            ("Categorical", f"{categorical}"),
            ("Boolean flags", f"{boolean}"),
        ]
    )
    components.spacer(0.6)
    components.stat_row(
        [
            ("Missing cells", f"{int(data.isna().sum().sum()):,}"),
            ("Duplicate rows", f"{int(data.duplicated().sum()):,}"),
            (
                "Duplicates excluding ID",
                f"{context.pipeline.quality.duplicates.duplicate_rows_excluding_id:,}",
            ),
            ("Source columns", f"{len(context.pipeline.raw.columns)}"),
            ("Engineered features", f"{len(context.pipeline.feature_report.created)}"),
        ]
    )
    st.caption(
        "Missing cells in engineered columns are expected and meaningful: a ratio "
        "is left undefined rather than fabricated when its denominator is zero or "
        "negative. Source columns themselves contain no missing values."
    )


def _render_column_info(context: PageContext) -> None:
    """Per-column documentation table."""
    data = context.data
    rows: list[dict[str, str]] = []
    for column in data.columns:
        series = data[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "Column": str(column),
                "Type": str(series.dtype),
                "Distinct": f"{int(series.nunique(dropna=True)):,}",
                "Missing": f"{missing:,}",
                "Missing %": f"{(missing / len(data) * 100 if len(data) else 0):.2f}%",
                "Meaning": describe_column(str(column)),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        height=420,
        column_config={
            "Meaning": st.column_config.TextColumn(width="large"),
            "Column": st.column_config.TextColumn(width="medium"),
        },
    )


def _render_statistics(context: PageContext) -> None:
    """Summary statistics for numeric columns."""
    summaries = summarise_numeric(context.data, registry=context.registry)
    if not summaries:
        components.unavailable("Summary statistics", "No analysable numeric columns.")
        return

    frame = summary_frame(summaries)
    display = frame[
        [
            "column", "count", "missing", "mean", "median", "std",
            "min", "q1", "q3", "max", "skewness", "skew_description",
        ]
    ].rename(
        columns={
            "column": "Column", "count": "Count", "missing": "Missing",
            "mean": "Mean", "median": "Median", "std": "Std dev",
            "min": "Min", "q1": "Q1", "q3": "Q3", "max": "Max",
            "skewness": "Skew", "skew_description": "Shape",
        }
    )
    st.dataframe(
        display.round(3), width="stretch", hide_index=True, height=420
    )
    st.caption(
        "Mean and median are shown together because the monetary columns are "
        "strongly right-skewed; where they diverge, the median describes a typical "
        "customer more faithfully."
    )


def _render_table(context: PageContext) -> None:
    """The searchable, paginated record table."""
    data = context.data
    all_columns = list(data.columns)
    default_selection = [c for c in DEFAULT_COLUMNS if c in all_columns] or all_columns[:10]

    controls = st.columns([2.2, 1, 1], gap="medium")
    with controls[0]:
        selected = st.multiselect(
            "Columns to display",
            options=all_columns,
            default=default_selection,
            key="explorer_columns",
            help="Choose which columns appear in the table below.",
        )
    with controls[1]:
        page_size = st.selectbox(
            "Rows per page", options=PAGE_SIZES, index=1, key="explorer_page_size"
        )
    with controls[2]:
        search = st.text_input(
            "Search",
            key="explorer_search",
            placeholder="value or customer id",
            help="Matches any displayed column, case-insensitive.",
        )

    if not selected:
        components.notice(
            "No columns selected",
            "Choose at least one column to display the record table.",
            icon="◌",
        )
        return

    working = data[selected]

    if search:
        needle = search.strip().lower()
        # Search across the displayed columns only, so the cost stays predictable.
        matches = pd.Series(False, index=working.index)
        for column in selected:
            matches |= (
                working[column].astype("string").str.lower().str.contains(needle, na=False)
            )
        working = working.loc[matches]

    total = len(working)
    if total == 0:
        components.notice(
            "No records match",
            f"No rows in the current selection contain '{search}'. Clear the search "
            "to see all selected records.",
            icon="◌",
            tone="attention",
        )
        return

    n_pages = max(1, (total + page_size - 1) // page_size)
    page = 1
    if n_pages > 1:
        page = st.number_input(
            f"Page (1-{n_pages})",
            min_value=1,
            max_value=n_pages,
            value=1,
            step=1,
            key="explorer_page",
        )

    start = int((page - 1) * page_size)
    end = min(start + page_size, total)

    st.markdown(
        f'<div class="fr-filter-count">Showing rows <b>{start + 1:,}</b> to '
        f"<b>{end:,}</b> of <b>{total:,}</b>"
        + (f" matching '{components.escape(search)}'" if search else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        working.iloc[start:end],
        width="stretch",
        hide_index=True,
        height=min(620, 80 + page_size * 35),
    )

    # Export the filtered selection, not the whole dataset.
    st.download_button(
        "Download current selection as CSV",
        data=working.to_csv(index=False).encode("utf-8"),
        file_name=f"finrisk_selection_{total}_rows.csv",
        mime="text/csv",
        help="Exports every row matching the current filters and search, not just this page.",
    )


def render(context: PageContext, show_filters: bool = True) -> None:
    """Render the Data Explorer page.

    Args:
        context: The per-run page context.
        show_filters: Render the global filter bar beneath the header.
    """
    components.page_header(
        title="Data Explorer",
        description=(
            "Inspect the underlying records, verify any figure shown elsewhere, and "
            "export the current selection."
        ),
        meta=[
            ("Rows in selection", f"{context.n_selected:,}"),
            ("Columns", f"{len(context.data.columns)}"),
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

    with st.expander("Dataset overview", expanded=True):
        _render_overview(context)

    with st.expander("Column information", expanded=False):
        _render_column_info(context)

    with st.expander("Summary statistics", expanded=False):
        _render_statistics(context)

    components.section(
        "Records",
        "Filtered by the global filters, then by any search term entered below.",
    )
    _render_table(context)
