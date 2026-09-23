"""
FinRisk AI - dashboard entry point.

    streamlit run app.py

This module orchestrates only: it loads the pipeline, applies filters, builds the
page context and routes to a page renderer. No analytical logic lives here. Every
number the dashboard shows is computed by a module under ``src/`` that runs
perfectly well without Streamlit.

Caching strategy
----------------
``load_pipeline`` is cached for the session, so the 30,000-row pipeline runs once
rather than on every widget interaction. Filtering and signal derivation run per
rerun but operate on already-processed data and complete in milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Make `src` importable when Streamlit executes this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="FinRisk AI · Credit Risk Intelligence",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.filtering import apply_filters, derive_bounds  # noqa: E402
from src.kpi_engine import calculate_kpis  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402
from src.pipeline import PipelineResult, run_pipeline  # noqa: E402
from src.schema import DATASET_NAME, DATASET_PERIOD, DATASET_SOURCE  # noqa: E402
from src.signals import derive_key_findings, derive_signals  # noqa: E402
from src.ui import components, filter_bar, navigation, theme  # noqa: E402
from src.ui.context import PageContext  # noqa: E402
from src.ui.pages import (  # noqa: E402
    data_explorer,
    data_quality,
    executive,
    financial,
    methodology,
    placeholders,
    segmentation as segmentation_page,
)

logger = get_logger(__name__)

#: Pages that expose the global filter bar. The rest either ignore filters
#: (Methodology), describe the raw source (Data Quality), or model the whole
#: portfolio by design (Customer Segmentation).
FILTERABLE_PAGES: frozenset[str] = frozenset({"financial", "data_explorer"})


# --------------------------------------------------------------------------- #
# Cached data loading
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def load_segmentation(k: int | None = None):
    """Fit the segmentation model once per session, per K.

    Deliberately fitted on the **whole portfolio** rather than the filtered
    selection: segments are a property of the book, and refitting on every filter
    change would make them shift underfoot and prevent any comparison between
    views. The Segmentation page instead shows how a filtered selection
    distributes across these fixed segments.

    Cached on ``k`` so exploring an alternative K reuses everything else. The
    scaler comparison and K sweep are the expensive part and run once.

    Args:
        k: Optional K override. ``None`` uses the measured recommendation.

    Returns:
        A ``SegmentationResult`` or ``SegmentationUnavailable``; ``None`` if the
        pipeline itself is unavailable.
    """
    from src.segmentation import SegmentationUnavailable, run_segmentation

    pipeline, error = load_pipeline()
    if pipeline is None:
        return None

    try:
        return run_segmentation(
            pipeline.data, pipeline.registry, k=k, compare_scaling=True
        )
    except Exception as exc:  # noqa: BLE001 - keep the page usable
        logger.exception("Segmentation failed.")
        return SegmentationUnavailable(
            reason=(
                "The clustering model could not be fitted for this dataset "
                f"({type(exc).__name__}). Technical detail has been logged."
            ),
            n_rows=len(pipeline.data),
        )


@st.cache_resource(show_spinner=False)
def load_pipeline() -> tuple[PipelineResult | None, str | None]:
    """Run the data pipeline once per session.

    ``cache_resource`` rather than ``cache_data`` because the result holds
    dataframes and report objects that should be shared by reference rather than
    copied on every access.

    Returns:
        ``(result, None)`` on success or ``(None, message)`` on failure.
    """
    from src.data_loader import DataLoadError, DataValidationError

    try:
        return run_pipeline(persist=True), None
    except DataLoadError as exc:
        logger.error("Pipeline load failed: %s", exc)
        return None, (
            "The dataset could not be found or read.\n\n"
            f"{exc}\n\n"
            "Run `python scripts/download_data.py` from the project root to fetch it."
        )
    except DataValidationError as exc:
        logger.error("Pipeline validation failed: %s", exc)
        return None, (
            "The dataset was found but did not pass validation.\n\n"
            f"{exc}\n\n"
            "The file may be a different dataset or an incomplete extract."
        )
    except Exception as exc:  # noqa: BLE001 - last line of defence for the UI
        logger.exception("Unexpected failure while preparing data.")
        return None, (
            "An unexpected problem occurred while preparing the data.\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            "Details have been written to the application log."
        )


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #


def build_context(pipeline: PipelineResult, page_id: str) -> PageContext:
    """Assemble the page context for this rerun.

    Built once and shared by every visual on the page, which is what guarantees
    filters apply consistently rather than to some charts only.

    Args:
        pipeline: The cached pipeline result.
        page_id: The active page, used to decide whether filters are offered.

    Returns:
        A fully populated :class:`PageContext`.
    """
    registry = pipeline.registry
    bounds = derive_bounds(pipeline.data, registry)

    # Data Quality describes the raw source, so it must not be filtered.
    if page_id == "data_quality":
        spec = filter_bar.current_spec(bounds).reset()
    else:
        spec = filter_bar.current_spec(bounds)

    filters = apply_filters(pipeline.data, spec, registry, bounds)

    portfolio_kpis = calculate_kpis(pipeline.data, registry)
    kpis = (
        portfolio_kpis
        if not filters.is_filtered
        else calculate_kpis(filters.data, registry)
    )

    if filters.is_empty:
        from src.findings import FindingSet
        from src.signals import SignalSet

        findings = FindingSet(context="No customers match the current filters.")
        signals = SignalSet(n_rows=0)
    else:
        findings = derive_key_findings(filters.data, registry, limit=6)
        signals = derive_signals(filters.data, kpis, registry)

    return PageContext(
        pipeline=pipeline,
        filters=filters,
        bounds=bounds,
        kpis=kpis,
        portfolio_kpis=portfolio_kpis,
        findings=findings,
        signals=signals,
    )


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #

#: Page id -> renderer. Placeholders share one renderer parameterised by id.
PAGE_RENDERERS: dict[str, Callable[..., None]] = {
    "executive": executive.render,
    "financial": financial.render,
    "segmentation": segmentation_page.render,
    "data_explorer": data_explorer.render,
    "data_quality": data_quality.render,
    "methodology": methodology.render,
}


def render_page(page_id: str, context: PageContext, show_filters: bool = False) -> None:
    """Render the requested page, converting any failure into a notice.

    A page that raises must not take the application down or show a traceback, so
    the exception is logged and a product-style message is shown instead.

    Args:
        page_id: Active page id.
        context: The page context.
        show_filters: Pass through to the page so it can place the filter bar
            itself, directly beneath its own header.
    """
    renderer = PAGE_RENDERERS.get(page_id)
    try:
        if renderer is None:
            placeholders.render(context, page_id)
        elif page_id == "segmentation":
            # Fitting is expensive, so it is cached and injected rather than run
            # inside the page.
            override = st.session_state.get(segmentation_page.K_OVERRIDE_KEY)
            with st.spinner("Clustering customer behaviour..."):
                fitted = load_segmentation(override)
            renderer(context, segmentation=fitted)
        elif page_id in FILTERABLE_PAGES:
            renderer(context, show_filters=show_filters)
        else:
            renderer(context)
    except Exception as exc:  # noqa: BLE001 - keep tracebacks out of the UI
        logger.exception("Page '%s' failed to render.", page_id)
        components.notice(
            "Analysis unavailable",
            "This section could not be displayed for the current selection. "
            "Try resetting the filters or widening the selection. "
            f"Technical detail has been logged ({type(exc).__name__}).",
            icon="⊘",
            tone="risk",
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    """Compose and run the dashboard."""
    theme.inject_theme()

    with st.spinner("Analysing financial behaviour..."):
        pipeline, error = load_pipeline()

    if pipeline is None:
        components.page_header(
            title="FinRisk AI",
            description="Financial Analytics & Credit Risk Intelligence",
        )
        components.notice(
            "Unable to load the dataset",
            error or "An unknown problem occurred.",
            icon="⊘",
            tone="risk",
        )
        st.code("python scripts/download_data.py", language="bash")
        return

    page_id = navigation.render_sidebar(
        dataset_label=f"{DATASET_NAME} · {DATASET_SOURCE}",
        footer_note=(
            f"{DATASET_PERIOD} · {len(pipeline.raw):,} customers · "
            f"{len(pipeline.feature_report.created)} engineered features"
        ),
    )

    # The context reads filter state from the widgets, which Streamlit has already
    # updated by this point, so one build per rerun is always current.
    context = build_context(pipeline, page_id)

    show_filters = page_id in FILTERABLE_PAGES
    render_page(page_id, context, show_filters=show_filters)

    _render_footer(pipeline)


def _render_footer(pipeline: PipelineResult) -> None:
    """Render the page footer."""
    from src.ui.theme import PALETTE

    components.spacer(1.4)
    st.markdown(
        f'<div style="border-top:1px solid {PALETTE["border"]};padding-top:0.8rem;'
        f'display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;'
        f'font-size:0.7rem;color:{PALETTE["text_faint"]}">'
        f"<span>FinRisk AI · From Financial Data to Intelligent Decisions</span>"
        f"<span>{DATASET_NAME} · {DATASET_SOURCE} · "
        f"CC BY 4.0 · amounts in NT$</span></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
