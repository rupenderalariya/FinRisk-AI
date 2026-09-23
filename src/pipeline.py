"""
Pipeline orchestration.

Runs the full sequence in order::

    load -> validate -> profile -> quality -> clean -> engineer -> panel -> persist

and returns one :class:`PipelineResult` carrying every artefact and audit trail
the rest of the application needs.

This is the only module permitted to touch Streamlit, and only for its caching
decorators. The import is optional: if Streamlit is absent (running under pytest,
in a notebook, or from the CLI) the decorators degrade to no-ops, so the pipeline
stays importable everywhere.

Run it directly to regenerate the processed artefacts::

    python -m src.pipeline
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

from src.config import settings
from src.data_loader import (
    DataLoadError,
    DataValidationError,
    DatasetProfile,
    LoadResult,
    ValidationResult,
    load_dataset,
    profile_dataset,
)
from src.data_processing import (
    PROCESSED_PANEL_FILENAME,
    PROCESSED_WIDE_FILENAME,
    CleaningReport,
    build_panel_long,
    clean_dataset,
    save_processed,
)
from src.data_quality import DataQualityReport, run_quality_report
from src.feature_engineering import FeatureReport, engineer_features
from src.logging_setup import get_logger
from src.schema import (
    DATASET_CITATION,
    DATASET_CURRENCY,
    DATASET_LANDING_URL,
    DATASET_LICENCE,
    DATASET_NAME,
    DATASET_PERIOD,
    DATASET_SOURCE,
    ColumnRegistry,
    build_registry,
    describe_panel_alignment,
)

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# --------------------------------------------------------------------------- #
# Optional Streamlit caching
# --------------------------------------------------------------------------- #


def _streamlit_runtime_active() -> bool:
    """True only when running inside a live Streamlit server.

    Checked before applying any cache decorator. Streamlit emits a
    "No runtime found" warning if its cache is exercised outside a server, which
    would pollute CLI and pytest output, so caching is applied only when it can
    genuinely work.
    """
    try:
        from streamlit.runtime import exists
    except ImportError:
        return False

    try:
        return bool(exists())
    except Exception:  # noqa: BLE001 - treat any probe failure as "no runtime"
        return False


def _cache_data(func: F) -> F:
    """Wrap a function in ``st.cache_data`` when a Streamlit runtime is active.

    Decides lazily, at call time, because at import time the runtime may not yet
    be up even inside a real Streamlit app. Outside Streamlit the wrapper simply
    calls through, so this module stays importable from pytest and the CLI.
    """
    cached: list[Callable[..., Any]] = []

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _streamlit_runtime_active():
            return func(*args, **kwargs)
        if not cached:
            import streamlit as st

            cached.append(st.cache_data(show_spinner=False)(func))
        return cached[0](*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Result structure
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PipelineResult:
    """Everything the pipeline produces."""

    raw: pd.DataFrame
    clean: pd.DataFrame
    features: pd.DataFrame
    panel: pd.DataFrame
    registry: ColumnRegistry
    raw_profile: DatasetProfile
    processed_profile: DatasetProfile
    validation: ValidationResult
    quality: DataQualityReport
    cleaning: CleaningReport
    feature_report: FeatureReport
    metadata: dict[str, Any]
    duration_seconds: float
    written_files: tuple[Path, ...] = field(default=())

    # ------------------------------------------------------------------ #

    @property
    def data(self) -> pd.DataFrame:
        """The analysis-ready frame: cleaned plus engineered features."""
        return self.features

    @property
    def n_clients(self) -> int:
        """Number of client rows available for analysis."""
        return int(len(self.features))

    @property
    def currency(self) -> str:
        """Currency symbol for monetary formatting."""
        return DATASET_CURRENCY

    def dataset_provenance(self) -> dict[str, str]:
        """Source, licence and citation, for the UI and the report."""
        return {
            "name": DATASET_NAME,
            "source": DATASET_SOURCE,
            "url": DATASET_LANDING_URL,
            "licence": DATASET_LICENCE,
            "citation": DATASET_CITATION,
            "period": DATASET_PERIOD,
            "currency": DATASET_CURRENCY,
            "panel_alignment": describe_panel_alignment(),
        }

    def summary(self) -> dict[str, Any]:
        """Compact summary of the run."""
        return {
            "rows_raw": int(len(self.raw)),
            "rows_final": self.n_clients,
            "columns_raw": int(len(self.raw.columns)),
            "columns_final": int(len(self.features.columns)),
            "panel_rows": int(len(self.panel)),
            "features_created": len(self.feature_report.created),
            "features_skipped": len(self.feature_report.skipped),
            "quality_score": round(self.quality.quality_score.score, 2),
            "quality_band": self.quality.quality_score.band,
            "validation": self.validation.summary(),
            "cleaning": self.cleaning.summary(),
            "duration_seconds": round(self.duration_seconds, 2),
            "available_roles": list(self.registry.available_role_names()),
            "written_files": [p.name for p in self.written_files],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialisable payload. Excludes the dataframes deliberately.

        This is what the AI layer receives: aggregate results only, never
        client-level rows.
        """
        return {
            "summary": self.summary(),
            "provenance": self.dataset_provenance(),
            "validation": self.validation.to_dict(),
            "quality": self.quality.to_dict(),
            "cleaning": self.cleaning.to_dict(),
            # Named "feature_report" rather than "features" so it is unmistakably
            # the documentation of the features, not the feature data itself.
            "feature_report": self.feature_report.to_dict(),
            "raw_profile": self.raw_profile.to_dict(),
            "processed_profile": self.processed_profile.to_dict(),
            "unavailable_capabilities": self.registry.unavailable_explanations(),
        }


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run_pipeline(
    path: Path | str | None = None,
    persist: bool = True,
    drop_id_excluded_duplicates: bool = False,
    raise_on_invalid: bool = True,
) -> PipelineResult:
    """Execute the full data pipeline.

    Args:
        path: Dataset path. Auto-discovered when omitted.
        persist: Write the processed artefacts to ``data/processed``.
        drop_id_excluded_duplicates: Remove rows that duplicate once the
            identifier is excluded. Off by default; they are flagged instead.
        raise_on_invalid: Raise when contract validation fails.

    Returns:
        A :class:`PipelineResult`.

    Raises:
        DataLoadError: If the dataset cannot be found or read.
        DataValidationError: If validation fails and ``raise_on_invalid`` is set.
    """
    started = time.perf_counter()
    logger.info("=" * 70)
    logger.info("FinRisk AI pipeline starting")
    logger.info("=" * 70)

    settings.ensure_directories()

    # --- 1. Load, normalise, validate, profile ---
    load_result: LoadResult = load_dataset(path, validate=True, raise_on_invalid=raise_on_invalid)
    raw = load_result.data
    raw_registry = build_registry(raw)
    logger.info(
        "Stage 1/6 load complete: %d rows x %d columns; roles found: %s",
        len(raw),
        len(raw.columns),
        ", ".join(raw_registry.available_role_names()),
    )

    # --- 2. Quality assessment on the RAW data ---
    # Deliberately before cleaning, so the report describes the source as it
    # arrived rather than the state after our own fixes.
    quality = run_quality_report(raw, raw_registry)
    logger.info(
        "Stage 2/6 quality complete: score %.2f (%s), %d issue(s).",
        quality.quality_score.score,
        quality.quality_score.band,
        len(quality.all_issues),
    )

    # --- 3. Clean ---
    clean, cleaning = clean_dataset(
        raw, drop_id_excluded_duplicates=drop_id_excluded_duplicates
    )
    logger.info("Stage 3/6 cleaning complete: %s", cleaning.summary())

    # --- 4. Engineer features ---
    clean_registry = build_registry(clean)
    features, feature_report = engineer_features(clean, clean_registry)
    logger.info("Stage 4/6 features complete: %s", feature_report.summary())

    # --- 5. Reshape to a long panel ---
    final_registry = build_registry(features)
    panel = build_panel_long(features, final_registry)
    logger.info("Stage 5/6 panel complete: %d client-month rows.", len(panel))

    # --- 6. Persist ---
    written: list[Path] = []
    if persist:
        try:
            written.append(save_processed(features, PROCESSED_WIDE_FILENAME))
            if not panel.empty:
                written.append(save_processed(panel, PROCESSED_PANEL_FILENAME))
        except (OSError, ValueError) as exc:
            # A write failure must not lose the in-memory results.
            logger.error(
                "Could not write processed artefacts (%s). The in-memory results "
                "remain usable.",
                exc,
            )
    logger.info("Stage 6/6 persistence complete: %d file(s) written.", len(written))

    processed_profile = profile_dataset(features, load_result.metadata)
    duration = time.perf_counter() - started

    result = PipelineResult(
        raw=raw,
        clean=clean,
        features=features,
        panel=panel,
        registry=final_registry,
        raw_profile=load_result.profile,
        processed_profile=processed_profile,
        validation=load_result.validation,
        quality=quality,
        cleaning=cleaning,
        feature_report=feature_report,
        metadata=load_result.metadata,
        duration_seconds=duration,
        written_files=tuple(written),
    )

    logger.info("=" * 70)
    logger.info("Pipeline complete in %.2fs", duration)
    logger.info("=" * 70)
    return result


@_cache_data
def load_pipeline_data(path: str | None = None) -> PipelineResult:
    """Cached pipeline entry point for the dashboard.

    Streamlit reruns the whole script on every interaction, so without caching
    the 30,000-row pipeline would re-execute on each click.

    Args:
        path: Optional dataset path.

    Returns:
        A :class:`PipelineResult`.
    """
    return run_pipeline(path=path, persist=True)


def safe_load_pipeline_data(
    path: str | None = None,
) -> tuple[PipelineResult | None, str | None]:
    """Run the pipeline and convert any failure into a readable message.

    The UI uses this so a missing dataset produces a clear instruction rather
    than a traceback in the browser.

    Args:
        path: Optional dataset path.

    Returns:
        ``(result, None)`` on success, ``(None, message)`` on failure.
    """
    try:
        return load_pipeline_data(path), None
    except DataLoadError as exc:
        logger.error("Pipeline load failed: %s", exc)
        return None, (
            f"The dataset could not be loaded.\n\n{exc}\n\n"
            "Fix: run `python scripts/download_data.py` from the project root."
        )
    except DataValidationError as exc:
        logger.error("Pipeline validation failed: %s", exc)
        return None, (
            f"The dataset loaded but failed validation.\n\n{exc}\n\n"
            "The file may be a different dataset or an incomplete extract."
        )
    except (OSError, ValueError, KeyError) as exc:
        logger.exception("Unexpected pipeline failure.")
        return None, (
            f"An unexpected error occurred while preparing the data: "
            f"{type(exc).__name__}: {exc}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    """Run the pipeline from the command line and print a summary."""
    try:
        result = run_pipeline()
    except (DataLoadError, DataValidationError) as exc:
        print(f"\nPipeline failed:\n{exc}\n")
        return 1

    summary = result.summary()
    provenance = result.dataset_provenance()

    print("\n" + "=" * 72)
    print(f"  FinRisk AI - pipeline summary")
    print("=" * 72)
    print(f"  Dataset          : {provenance['name']}")
    print(f"  Source           : {provenance['source']}")
    print(f"  Period           : {provenance['period']}  (amounts in {provenance['currency']})")
    print("-" * 72)
    print(f"  Raw shape        : {summary['rows_raw']:,} rows x {summary['columns_raw']} columns")
    print(f"  Final shape      : {summary['rows_final']:,} rows x {summary['columns_final']} columns")
    print(f"  Long panel       : {summary['panel_rows']:,} client-month rows")
    print(f"  Features created : {summary['features_created']}")
    print(f"  Features skipped : {summary['features_skipped']} (source columns absent)")
    print(f"  Quality score    : {summary['quality_score']} / 100 ({summary['quality_band']})")
    print(f"  Validation       : {summary['validation']}")
    print(f"  Cleaning         : {summary['cleaning']}")
    print(f"  Duration         : {summary['duration_seconds']}s")
    print(f"  Written files    : {', '.join(summary['written_files']) or 'none'}")
    print("-" * 72)
    print(f"  Roles available  : {', '.join(summary['available_roles'])}")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
