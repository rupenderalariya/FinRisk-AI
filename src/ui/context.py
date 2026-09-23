"""
Per-run page context.

Bundles everything a page needs - the pipeline result, the filtered selection,
KPIs for both the selection and the full portfolio, derived findings and signals
- into one object built **once** per rerun.

That single-construction rule is what guarantees every KPI, chart and table on a
page reflects the same filtered selection. It removes the failure mode where a
filter changes some visuals and silently misses others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.filtering import FilterBounds, FilterResult, FilterSpec
from src.findings import FindingSet
from src.kpi_engine import KPIResult
from src.pipeline import PipelineResult
from src.schema import ColumnRegistry
from src.signals import SignalSet


@dataclass(frozen=True)
class PageContext:
    """Everything a page renders from."""

    pipeline: PipelineResult
    filters: FilterResult
    bounds: FilterBounds
    kpis: KPIResult
    portfolio_kpis: KPIResult
    findings: FindingSet
    signals: SignalSet

    # ------------------------------------------------------------------ #

    @property
    def data(self) -> pd.DataFrame:
        """The filtered selection every visual must use."""
        return self.filters.data

    @property
    def full_data(self) -> pd.DataFrame:
        """The unfiltered analysis-ready frame."""
        return self.pipeline.data

    @property
    def registry(self) -> ColumnRegistry:
        """Semantic column registry."""
        return self.pipeline.registry

    @property
    def spec(self) -> FilterSpec:
        """The active filter specification."""
        return self.filters.spec

    @property
    def is_filtered(self) -> bool:
        """True when filters are narrowing the data."""
        return self.filters.is_filtered

    @property
    def is_empty(self) -> bool:
        """True when the selection contains no rows."""
        return self.filters.is_empty

    @property
    def n_selected(self) -> int:
        """Rows in the current selection."""
        return self.filters.n_selected

    @property
    def target_column(self) -> str | None:
        """The target column name, or None when absent."""
        from src.schema import Role

        return self.registry.one(Role.TARGET)

    @property
    def baseline_kpis(self) -> KPIResult | None:
        """Portfolio KPIs, but only when a comparison is meaningful.

        Returns ``None`` when nothing is filtered, because comparing the
        portfolio with itself would produce a delta of zero that implies a
        comparison the data cannot support.
        """
        return self.portfolio_kpis if self.is_filtered else None

    def overall_rate(self) -> float | None:
        """Observed default rate for the current selection, as a percentage."""
        return self.kpis.raw("observed_default_rate")

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary, suitable as AI grounding later on."""
        return {
            "pipeline": self.pipeline.summary(),
            "filters": self.filters.to_dict(),
            "kpis": self.kpis.to_dict(),
            "findings": self.findings.to_dict(),
            "signals": self.signals.to_dict(),
        }
