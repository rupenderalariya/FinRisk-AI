"""
Insight, signal and action-chain cards.

Renders the output of :mod:`src.signals` and :mod:`src.findings`. Every card is
driven by a computed object, so nothing here can display a claim that the data
does not support: if a rule did not fire there is simply no card.

Signals are labelled as observed analytical measurements. No card implies a
model prediction, because no model exists yet.
"""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from src.findings import Finding, FindingSet
from src.signals import OBSERVED_SIGNAL_LABEL, Signal, SignalSet
from src.ui.components import badge, escape, notice
from src.ui.theme import PALETTE, finding_colour, severity_colour, with_alpha


# --------------------------------------------------------------------------- #
# Key intelligence
# --------------------------------------------------------------------------- #


def render_finding_card(finding: Finding, show_why: bool = True) -> None:
    """Render one finding as an insight card.

    Args:
        finding: The finding to render.
        show_why: Include the "why it matters" footer.
    """
    accent = finding_colour(finding.finding_type.value)
    severity_badge = badge(finding.severity.value, finding.severity.value)
    type_badge = badge(finding.finding_type.value.replace("_", " "), finding.finding_type.value)

    metric_html = ""
    if finding.evidence:
        chips = "".join(
            f'<span class="fr-insight-metric" title="{escape(item.label)}">'
            f"{escape(item.formatted)}</span>"
            for item in finding.evidence[:3]
        )
        metric_html = (
            f'<div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin-top:0.55rem">'
            f"{chips}</div>"
        )

    why_html = ""
    if show_why:
        why_html = (
            f'<div class="fr-insight-why"><b>Why it matters:</b> '
            f"{escape(finding.implication)}</div>"
        )

    st.markdown(
        f'<div class="fr-insight" style="--fr-insight-accent:{accent}">'
        f'<div class="fr-insight-head">'
        f'<div class="fr-insight-title">{escape(finding.title)}</div>'
        f'<div style="display:flex;gap:0.3rem;flex-shrink:0">{type_badge}{severity_badge}</div>'
        f"</div>"
        f'<div class="fr-insight-body">{escape(finding.fact)}</div>'
        f"{metric_html}{why_html}</div>",
        unsafe_allow_html=True,
    )


def render_key_intelligence(
    findings: FindingSet, limit: int = 4, columns: int = 1
) -> None:
    """Render the Key Intelligence block.

    Args:
        findings: Derived findings.
        limit: Maximum cards to show.
        columns: Lay cards out across this many columns.
    """
    if not findings:
        notice(
            "No findings for this selection",
            findings.context
            or "Widen the filters so the comparison groups are large enough to analyse.",
            icon="◌",
        )
        return

    selected = findings.sorted_by_priority()[:limit]

    if columns <= 1:
        for finding in selected:
            render_finding_card(finding)
        return

    for start in range(0, len(selected), columns):
        row = selected[start : start + columns]
        for column, finding in zip(st.columns(len(row), gap="small"), row):
            with column:
                render_finding_card(finding)


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #


def render_signal_card(signal: Signal) -> None:
    """Render one risk or opportunity signal.

    Args:
        signal: The signal to render.
    """
    colour = severity_colour(signal.severity.value)
    bar_width = max(4.0, min(100.0, signal.intensity * 100.0))

    st.markdown(
        f'<div class="fr-signal">'
        f'<div class="fr-signal-top">'
        f'<span class="fr-signal-name">{escape(signal.name)}</span>'
        f"{badge(signal.severity.value, signal.severity.value)}</div>"
        f'<div class="fr-signal-value" style="color:{colour}">{escape(signal.value)}</div>'
        f'<div class="fr-signal-desc">{escape(signal.description)}</div>'
        f'<div class="fr-signal-bar"><span style="width:{bar_width:.1f}%;'
        f'background:{colour}"></span></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_signals(
    signals: Sequence[Signal],
    empty_message: str,
    columns: int = 3,
) -> None:
    """Render a grid of signal cards.

    Args:
        signals: Signals to render.
        empty_message: Shown when there are none.
        columns: Cards per row.
    """
    if not signals:
        notice("Nothing to report", empty_message, icon="◌")
        return

    for start in range(0, len(signals), columns):
        row = signals[start : start + columns]
        for column, signal in zip(st.columns(len(row), gap="small"), row):
            with column:
                render_signal_card(signal)


def render_signal_set(signal_set: SignalSet) -> None:
    """Render both risk signals and opportunities with their section headings.

    Args:
        signal_set: The derived signals.
    """
    from src.ui.components import section, spacer

    section(
        "Risk Signals",
        f"{OBSERVED_SIGNAL_LABEL}s measured from historical outcomes in the current "
        "selection. These are not model predictions.",
    )
    render_signals(
        signal_set.risks,
        "No risk signals crossed their reporting threshold in this selection.",
    )

    spacer()
    section(
        "Opportunities",
        "Segments whose measured behaviour or observed outcome is more favourable "
        "than the selection average.",
    )
    render_signals(
        signal_set.opportunities,
        "No segment in this selection showed a materially more favourable observed "
        "outcome than the average.",
    )


# --------------------------------------------------------------------------- #
# Action chain
# --------------------------------------------------------------------------- #

_CHAIN_STEPS: tuple[tuple[str, str, str], ...] = (
    ("FACT", "fact", "info"),
    ("INSIGHT", "insight", "accent"),
    ("IMPLICATION", "implication", "attention"),
    ("ACTION", "action", "positive"),
)


def render_action_chain(finding: Finding) -> None:
    """Render a finding as the full four-part chain.

    Args:
        finding: The finding to render.
    """
    rows: list[str] = []
    for index, (label, attribute, colour_key) in enumerate(_CHAIN_STEPS):
        colour = PALETTE.get(colour_key, PALETTE["info"])
        rows.append(
            f'<div class="fr-chain-step">'
            f'<div class="fr-chain-tag" style="color:{colour}">{label}</div>'
            f'<div class="fr-chain-text">{escape(getattr(finding, attribute))}</div></div>'
        )
        if index < len(_CHAIN_STEPS) - 1:
            rows.append('<div class="fr-chain-sep"></div>')

    caveat_html = ""
    if finding.caveats:
        caveat_html = (
            f'<div class="fr-insight-why"><b>Caveats:</b> '
            f'{escape(" ".join(finding.caveats))}</div>'
        )

    evidence_html = ""
    if finding.evidence:
        chips = "".join(
            f'<span class="fr-insight-metric">{escape(item.label)}: '
            f"{escape(item.formatted)}</span>"
            for item in finding.evidence
        )
        evidence_html = (
            f'<div style="display:flex;gap:0.4rem;flex-wrap:wrap;'
            f'margin-top:0.7rem;padding-top:0.6rem;'
            f'border-top:1px solid {PALETTE["border"]}">{chips}</div>'
        )

    header = (
        f'<div class="fr-insight-head" style="margin-bottom:0.55rem">'
        f'<div class="fr-insight-title">{escape(finding.title)}</div>'
        f'<div style="display:flex;gap:0.3rem;flex-shrink:0">'
        f'{badge(finding.finding_type.value.replace("_", " "), finding.finding_type.value)}'
        f'{badge(f"{finding.confidence.value} confidence", "neutral")}</div></div>'
    )

    st.markdown(
        f'<div class="fr-chain">{header}{"".join(rows)}{evidence_html}{caveat_html}</div>',
        unsafe_allow_html=True,
    )


def render_action_chains(findings: FindingSet, limit: int = 5) -> None:
    """Render several findings as action chains.

    Args:
        findings: Derived findings.
        limit: Maximum chains to render.
    """
    if not findings:
        notice(
            "No recommended next steps yet",
            findings.context
            or "Findings drive the suggested next steps, and none were derived for "
            "this selection.",
            icon="◌",
        )
        return
    for finding in findings.sorted_by_priority()[:limit]:
        render_action_chain(finding)


# --------------------------------------------------------------------------- #
# Driver evidence
# --------------------------------------------------------------------------- #


def render_driver_rows(consensus: Sequence[dict], limit: int = 8) -> None:
    """Render ranked driver-analysis results as compact rows.

    Args:
        consensus: Records from :func:`src.drivers.build_consensus`.
        limit: Maximum rows to render.
    """
    if not consensus:
        notice(
            "Driver analysis unavailable",
            "Not enough records in this selection to test feature relationships.",
            icon="⊘",
        )
        return

    rows: list[str] = []
    for record in list(consensus)[:limit]:
        effect = float(record.get("mean_effect_size", 0.0))
        width = max(3.0, min(100.0, effect * 100.0))
        agreement = f'{record.get("n_methods_nonneglible", 0)}/{record.get("n_methods", 0)}'
        direction = str(record.get("direction", "n/a"))
        colour = (
            PALETTE["risk"]
            if direction == "positive"
            else PALETTE["positive"]
            if direction == "negative"
            else PALETTE["info"]
        )
        rows.append(
            f'<div style="display:flex;align-items:center;gap:0.7rem;padding:0.4rem 0;'
            f'border-bottom:1px solid {PALETTE["border"]}">'
            f'<div style="flex:0 0 200px;font-size:0.8rem;color:{PALETTE["text"]};'
            f'font-weight:560">{escape(record["feature"])}</div>'
            f'<div style="flex:1;height:5px;background:{PALETTE["bg_alt"]};'
            f'border-radius:3px;overflow:hidden">'
            f'<span style="display:block;height:100%;width:{width:.1f}%;'
            f'background:{colour};border-radius:3px"></span></div>'
            f'<div style="flex:0 0 58px;text-align:right;font-size:0.75rem;'
            f'color:{PALETTE["text_secondary"]};font-variant-numeric:tabular-nums">'
            f"{effect:.3f}</div>"
            f'<div style="flex:0 0 64px;text-align:right;font-size:0.7rem;'
            f'color:{PALETTE["text_muted"]}">{escape(agreement)} agree</div>'
            f"</div>"
        )

    st.markdown(
        f'<div class="fr-panel tight">{"".join(rows)}</div>', unsafe_allow_html=True
    )
