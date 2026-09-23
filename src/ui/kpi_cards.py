"""
Compact KPI panels.

Honesty rule for the change indicator
-------------------------------------
A delta is shown **only** when a genuine comparison exists. This dataset is a
single six-month extract with no prior period, so there is no "vs last month"
to report and none is invented. The one real comparison available is
*filtered selection versus the full portfolio*, so a delta appears only when
filters are active and a baseline was supplied.

Direction colouring respects each KPI's ``higher_is_better`` flag, so a rising
default rate reads as red while rising repayment reads as green.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import streamlit as st

from src.kpi_engine import KPIResult, KPIValue
from src.ui.components import escape
from src.ui.theme import PALETTE

#: Per-KPI presentation: icon, short interpretation, and the accent bar colour.
KPI_PRESENTATION: Final[Mapping[str, tuple[str, str, str]]] = {
    "total_clients": ("◫", "Customers in the current selection", "info"),
    "total_records": ("◫", "One row per customer", "info"),
    "client_month_records": ("▦", "Customer-month observations available", "info"),
    "observed_default_rate": ("▲", "Observed historical outcome, not a prediction", "risk"),
    "default_client_count": ("▲", "Customers recorded as defaulting", "risk"),
    "avg_credit_limit": ("◈", "Mean granted credit limit, not income", "info"),
    "median_credit_limit": ("◈", "Typical granted credit limit", "info"),
    "total_credit_limit": ("◈", "Maximum possible exposure if fully drawn", "info"),
    "total_outstanding_balance": ("▣", "Sum of the most recent statements", "info"),
    "avg_outstanding_balance": ("▣", "Mean most-recent statement balance", "info"),
    "median_outstanding_balance": ("▣", "Typical most-recent statement balance", "info"),
    "avg_utilisation": ("◐", "Derived: balance ÷ credit limit", "attention"),
    "median_utilisation": ("◐", "Typical derived utilisation", "attention"),
    "high_utilisation_share": ("◕", "Above the configured utilisation threshold", "attention"),
    "high_utilisation_count": ("◕", "Customers above the threshold", "attention"),
    "over_limit_share": ("◉", "Exceeded the limit in at least one month", "attention"),
    "total_repayments": ("▼", "Total repaid across six months", "positive"),
    "median_repayment_ratio": ("▽", "Typical share of statement repaid", "positive"),
    "avg_capped_repayment_ratio": ("▽", "Mean repaid share, each month capped at 200%", "positive"),
    "full_payer_share": ("✓", "Typically clear the statement in full", "positive"),
    "revolver_share": ("↻", "Typically repay under 30% of the statement", "attention"),
    "currently_delinquent_share": ("!", "Behind on payment in the latest month", "risk"),
    "currently_delinquent_count": ("!", "Customers behind in the latest month", "risk"),
    "ever_delinquent_share": ("!", "Behind in at least one of six months", "attention"),
    "avg_delinquent_months": ("▤", "Mean months behind, out of six", "attention"),
    "avg_max_delinquency": ("▤", "Mean worst delay recorded", "attention"),
    "avg_age": ("◎", "Mean customer age", "info"),
    "median_age": ("◎", "Typical customer age", "info"),
    "median_balance_trend": ("↗", "Typical monthly change in balance", "attention"),
    "share_growing_balance": ("↗", "Balances trending upward", "attention"),
}

#: The default headline set for the Executive Overview.
EXECUTIVE_KPIS: Final[tuple[str, ...]] = (
    "total_clients",
    "observed_default_rate",
    "avg_utilisation",
    "currently_delinquent_share",
    "median_credit_limit",
    "full_payer_share",
)


@dataclass(frozen=True)
class KpiDelta:
    """A measured comparison between the current selection and a baseline."""

    value: float
    baseline: float
    label: str
    higher_is_better: bool | None = None

    @property
    def difference(self) -> float:
        """Signed difference from the baseline."""
        return self.value - self.baseline

    @property
    def is_flat(self) -> bool:
        """True when the difference is negligible relative to the baseline."""
        scale = max(abs(self.baseline), 1e-9)
        return abs(self.difference) / scale < 0.005

    def css_class(self) -> str:
        """CSS class encoding both direction and whether it is good news."""
        if self.is_flat:
            return "flat"
        rising = self.difference > 0
        if self.higher_is_better is None:
            return "up" if rising else "down"
        if self.higher_is_better:
            return "up-good" if rising else "down-bad"
        return "up" if rising else "down"

    def text(self, fmt: str) -> str:
        """Human-readable delta text.

        Args:
            fmt: The parent KPI's format name, used to choose units.

        Returns:
            A short string such as ``↑ 3.4 pp vs full portfolio``.
        """
        if self.is_flat:
            return f"≈ in line with {self.label}"

        arrow = "↑" if self.difference > 0 else "↓"
        magnitude = abs(self.difference)

        if fmt == "percentage":
            amount = f"{magnitude:.2f} pp"
        elif fmt in {"currency", "currency_compact"}:
            from src.kpi_engine import format_currency

            amount = format_currency(magnitude, compact=(fmt == "currency_compact"))
        elif fmt == "integer":
            amount = f"{magnitude:,.0f}"
        else:
            amount = f"{magnitude:,.2f}"

        return f"{arrow} {amount} vs {self.label}"


def _card_html(
    label: str,
    value: str,
    note: str,
    icon: str,
    accent: str,
    delta: KpiDelta | None,
    fmt: str,
) -> str:
    """Build the markup for one KPI card."""
    delta_html = ""
    if delta is not None:
        delta_html = (
            f'<div class="fr-kpi-delta {delta.css_class()}">'
            f"{escape(delta.text(fmt))}</div>"
        )
    return (
        f'<div class="fr-kpi" style="--fr-accent-bar:{accent}">'
        f'<div class="fr-kpi-head">'
        f'<span class="fr-kpi-icon" style="color:{accent}">{escape(icon)}</span>'
        f'<span class="fr-kpi-label">{escape(label)}</span></div>'
        f'<div class="fr-kpi-value">{escape(value)}</div>'
        f"{delta_html}"
        f'<div class="fr-kpi-note">{escape(note)}</div>'
        f"</div>"
    )


def render_kpi_row(
    kpis: KPIResult,
    keys: Sequence[str] = EXECUTIVE_KPIS,
    baseline: KPIResult | None = None,
    baseline_label: str = "full portfolio",
    columns: int | None = None,
) -> None:
    """Render a row of KPI cards.

    Keys that the KPI engine could not compute are skipped silently here; the
    engine already records why, and the Methodology page surfaces it. That keeps
    the overview free of "n/a" tiles.

    Args:
        kpis: KPIs for the current selection.
        keys: Which KPIs to show, in order.
        baseline: Optional full-portfolio KPIs for a real comparison.
        baseline_label: What the baseline represents, shown in the delta text.
        columns: Column count. Defaults to the number of rendered cards.
    """
    available = [key for key in keys if kpis.get(key) is not None]
    if not available:
        return

    count = columns or len(available)
    for index, column in enumerate(st.columns(count, gap="small")):
        if index >= len(available):
            continue
        key = available[index]
        kpi: KPIValue = kpis.values[key]
        icon, note, accent_key = KPI_PRESENTATION.get(key, ("•", kpi.description, "info"))

        delta: KpiDelta | None = None
        if baseline is not None:
            baseline_value = baseline.raw(key)
            if baseline_value is not None:
                delta = KpiDelta(
                    value=kpi.value,
                    baseline=baseline_value,
                    label=baseline_label,
                    higher_is_better=kpi.higher_is_better,
                )

        with column:
            st.markdown(
                _card_html(
                    label=kpi.label,
                    value=kpi.formatted,
                    note=note,
                    icon=icon,
                    accent=PALETTE.get(accent_key, PALETTE["info"]),
                    delta=delta,
                    fmt=kpi.fmt,
                ),
                unsafe_allow_html=True,
            )
            # The formula lives in a tooltip rather than on the card face.
            st.markdown(
                f'<div style="font-size:0.66rem;color:{PALETTE["text_faint"]};'
                f'margin:0.3rem 0 0.2rem 0.1rem" title="{escape(kpi.description)}">'
                f"ⓘ {escape(kpi.formula)}</div>",
                unsafe_allow_html=True,
            )


def render_kpi_group(
    kpis: KPIResult,
    groups: Mapping[str, Sequence[str]],
    baseline: KPIResult | None = None,
) -> None:
    """Render several labelled KPI rows.

    Args:
        kpis: KPIs for the current selection.
        groups: ``{group_title: kpi_keys}``.
        baseline: Optional baseline for comparisons.
    """
    from src.ui.components import section

    for title, keys in groups.items():
        if not any(kpis.get(key) is not None for key in keys):
            continue
        section(title, rule=False)
        render_kpi_row(kpis, keys, baseline=baseline)


def kpi_definition_table(kpis: KPIResult) -> list[dict[str, str]]:
    """Build rows describing each computed KPI, for the Methodology page.

    Args:
        kpis: Computed KPIs.

    Returns:
        One dictionary per KPI with its label, value, formula and definition.
    """
    return [
        {
            "KPI": kpi.label,
            "Value": kpi.formatted,
            "Formula": kpi.formula,
            "Definition": kpi.description,
            "Category": kpi.category,
        }
        for kpi in kpis.values.values()
    ]
