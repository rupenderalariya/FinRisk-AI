"""
Reusable structure for analytical findings.

Every finding follows one chain::

    FACT  ->  INSIGHT  ->  IMPLICATION  ->  ACTION

The four parts are separate fields, not one blob of prose, which forces each
link to be stated explicitly and makes them individually testable.

Evidence traceability
---------------------
A :class:`Finding` cannot be constructed without at least one :class:`Evidence`
item naming the metric, its value and where it came from. That is enforced in
``__post_init__``, so an unsupported claim cannot enter the system even by
accident. This is the mechanism behind the "every insight traces to real data"
requirement.

This module supplies only the *structure* and generic builders. No specific
finding is hard-coded here: the rules that generate findings from computed
results live in the recommendations layer, which is built in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class FindingType(str, Enum):
    """Whether a finding represents exposure or upside."""

    RISK = "risk"
    OPPORTUNITY = "opportunity"
    OBSERVATION = "observation"
    DATA_QUALITY = "data_quality"


class Severity(str, Enum):
    """How much attention a finding warrants."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Confidence(str, Enum):
    """How much weight the underlying evidence can bear.

    Tied to sample size and method agreement, not to how convincing the
    narrative sounds.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


SEVERITY_ORDER: Final[dict[Severity, int]] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}

CONFIDENCE_ORDER: Final[dict[Confidence, int]] = {
    Confidence.HIGH: 0,
    Confidence.MEDIUM: 1,
    Confidence.LOW: 2,
}


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Evidence:
    """One computed number supporting a finding.

    Attributes:
        metric: Machine-readable metric key, e.g. ``observed_default_rate``.
        label: Human-readable metric name.
        value: The computed value, at full precision.
        formatted: Display string for the value.
        source: Which module or analysis produced it, e.g. ``kpi_engine``.
        comparison: Optional baseline the value is measured against.
        n_observations: Rows behind the value, where meaningful.
    """

    metric: str
    label: str
    value: float | int | str
    formatted: str
    source: str
    comparison: str | None = None
    n_observations: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "metric": self.metric,
            "label": self.label,
            "value": self.value,
            "formatted": self.formatted,
            "source": self.source,
            "comparison": self.comparison,
            "n_observations": self.n_observations,
        }

    def describe(self) -> str:
        """One-line rendering of the evidence."""
        base = f"{self.label}: {self.formatted}"
        if self.comparison:
            base += f" (vs {self.comparison})"
        if self.n_observations is not None:
            base += f" [n = {self.n_observations:,}]"
        return base


# --------------------------------------------------------------------------- #
# Finding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Finding:
    """A single analytical finding as a four-part chain.

    Attributes:
        key: Stable identifier for the finding.
        title: Short headline.
        fact: What the data shows. Must be a statement of measured fact.
        insight: What the fact means analytically.
        implication: The business or financial consequence.
        action: What could be considered. Phrased as an option for review, never
            as an instruction or as financial advice.
        finding_type: Risk, opportunity, observation or data quality.
        severity: How much attention it warrants.
        confidence: How much weight the evidence bears.
        evidence: Computed values supporting the chain. At least one required.
        affected_clients: Clients concerned, where applicable.
        affected_share_pct: Share of the selection concerned.
        caveats: Limitations a reader must know.
        tags: Free-form labels for filtering.
    """

    key: str
    title: str
    fact: str
    insight: str
    implication: str
    action: str
    finding_type: FindingType
    severity: Severity
    confidence: Confidence
    evidence: tuple[Evidence, ...]
    affected_clients: int | None = None
    affected_share_pct: float | None = None
    caveats: tuple[str, ...] = field(default=())
    tags: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        """Validate the finding.

        Raises:
            ValueError: If no evidence is attached, or a required part of the
                chain is blank. Both would allow an unsupported claim to reach
                the user, which this project treats as a defect.
        """
        if not self.evidence:
            raise ValueError(
                f"Finding '{self.key}' has no supporting evidence. Every finding must "
                "cite at least one computed metric so the claim is traceable to data."
            )

        for part_name in ("fact", "insight", "implication", "action"):
            value = getattr(self, part_name)
            if not value or not str(value).strip():
                raise ValueError(
                    f"Finding '{self.key}' is missing its '{part_name}'. All four "
                    "parts of FACT -> INSIGHT -> IMPLICATION -> ACTION are required."
                )

    # ------------------------------------------------------------------ #

    @property
    def is_actionable(self) -> bool:
        """True for findings above the informational level."""
        return self.severity in {Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM}

    @property
    def evidence_summary(self) -> str:
        """Semicolon-joined one-line rendering of all evidence."""
        return "; ".join(item.describe() for item in self.evidence)

    def to_chain(self) -> str:
        """Render the finding as the labelled four-part chain."""
        lines = [
            f"FACT: {self.fact}",
            f"INSIGHT: {self.insight}",
            f"IMPLICATION: {self.implication}",
            f"ACTION: {self.action}",
        ]
        if self.caveats:
            lines.append("CAVEATS: " + " ".join(self.caveats))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, safe to hand to the AI layer as grounding."""
        return {
            "key": self.key,
            "title": self.title,
            "fact": self.fact,
            "insight": self.insight,
            "implication": self.implication,
            "action": self.action,
            "type": self.finding_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "affected_clients": self.affected_clients,
            "affected_share_pct": self.affected_share_pct,
            "evidence": [item.to_dict() for item in self.evidence],
            "caveats": list(self.caveats),
            "tags": list(self.tags),
        }


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FindingSet:
    """An ordered collection of findings with convenient filters."""

    findings: tuple[Finding, ...] = field(default=())
    context: str = ""

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self):
        return iter(self.findings)

    def __bool__(self) -> bool:
        return bool(self.findings)

    # ------------------------------ filters ------------------------------ #

    def of_type(self, finding_type: FindingType) -> tuple[Finding, ...]:
        """Findings of one type."""
        return tuple(f for f in self.findings if f.finding_type is finding_type)

    @property
    def risks(self) -> tuple[Finding, ...]:
        """Risk findings."""
        return self.of_type(FindingType.RISK)

    @property
    def opportunities(self) -> tuple[Finding, ...]:
        """Opportunity findings."""
        return self.of_type(FindingType.OPPORTUNITY)

    @property
    def observations(self) -> tuple[Finding, ...]:
        """Neutral observations."""
        return self.of_type(FindingType.OBSERVATION)

    @property
    def data_quality(self) -> tuple[Finding, ...]:
        """Data-quality findings."""
        return self.of_type(FindingType.DATA_QUALITY)

    def of_severity(self, *severities: Severity) -> tuple[Finding, ...]:
        """Findings at any of the supplied severities."""
        wanted = set(severities)
        return tuple(f for f in self.findings if f.severity in wanted)

    def actionable(self) -> tuple[Finding, ...]:
        """Findings above the informational level."""
        return tuple(f for f in self.findings if f.is_actionable)

    def sorted_by_priority(self) -> tuple[Finding, ...]:
        """Findings ranked by severity, then confidence, then reach."""
        return tuple(
            sorted(
                self.findings,
                key=lambda f: (
                    SEVERITY_ORDER[f.severity],
                    CONFIDENCE_ORDER[f.confidence],
                    -(f.affected_share_pct or 0.0),
                ),
            )
        )

    def top(self, limit: int = 5) -> tuple[Finding, ...]:
        """Highest-priority findings."""
        return self.sorted_by_priority()[:limit]

    # ------------------------------ output ------------------------------ #

    @property
    def counts_by_type(self) -> dict[str, int]:
        """Number of findings per type."""
        return {t.value: len(self.of_type(t)) for t in FindingType}

    @property
    def counts_by_severity(self) -> dict[str, int]:
        """Number of findings per severity."""
        return {s.value: len(self.of_severity(s)) for s in Severity}

    def summary(self) -> str:
        """One-line human-readable summary."""
        if not self.findings:
            return "No findings were generated from the current selection."
        return (
            f"{len(self.findings)} finding(s): {len(self.risks)} risk, "
            f"{len(self.opportunities)} opportunity, {len(self.observations)} "
            f"observation, {len(self.data_quality)} data quality."
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the UI, report and AI grounding payload."""
        return {
            "summary": self.summary(),
            "context": self.context,
            "n_findings": len(self.findings),
            "counts_by_type": self.counts_by_type,
            "counts_by_severity": self.counts_by_severity,
            "findings": [f.to_dict() for f in self.sorted_by_priority()],
        }

    def extend(self, more: Iterable[Finding]) -> "FindingSet":
        """Return a new set with additional findings appended."""
        return FindingSet(findings=self.findings + tuple(more), context=self.context)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def make_evidence(
    metric: str,
    label: str,
    value: float | int | str,
    formatted: str | None = None,
    source: str = "analytics",
    comparison: str | None = None,
    n_observations: int | None = None,
) -> Evidence:
    """Construct an :class:`Evidence` item, formatting the value if needed.

    Args:
        metric: Metric key.
        label: Human-readable name.
        value: Computed value.
        formatted: Display string. Derived from ``value`` when omitted.
        source: Producing module or analysis.
        comparison: Optional baseline.
        n_observations: Rows behind the value.

    Returns:
        An :class:`Evidence` item.
    """
    if formatted is None:
        if isinstance(value, float):
            formatted = f"{value:,.2f}"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = str(value)

    return Evidence(
        metric=metric,
        label=label,
        value=value,
        formatted=formatted,
        source=source,
        comparison=comparison,
        n_observations=n_observations,
    )


def severity_from_share(
    share_pct: float,
    critical_at: float = 40.0,
    high_at: float = 25.0,
    medium_at: float = 10.0,
) -> Severity:
    """Derive a severity from how much of the portfolio a finding touches.

    Keeps severity proportional to measured reach rather than assigned by
    judgement, so it stays consistent between findings.

    Args:
        share_pct: Share of the selection affected, 0-100.
        critical_at: Threshold for critical.
        high_at: Threshold for high.
        medium_at: Threshold for medium.

    Returns:
        The derived severity.
    """
    if share_pct >= critical_at:
        return Severity.CRITICAL
    if share_pct >= high_at:
        return Severity.HIGH
    if share_pct >= medium_at:
        return Severity.MEDIUM
    if share_pct > 0:
        return Severity.LOW
    return Severity.INFO


def confidence_from_sample(
    n_observations: int,
    n_methods_agreeing: int = 1,
    high_n: int = 1_000,
    medium_n: int = 100,
) -> Confidence:
    """Derive a confidence level from sample size and method agreement.

    Args:
        n_observations: Rows behind the finding.
        n_methods_agreeing: Independent methods pointing the same way.
        high_n: Sample size qualifying for high confidence.
        medium_n: Sample size qualifying for medium confidence.

    Returns:
        The derived confidence.
    """
    if n_observations >= high_n and n_methods_agreeing >= 2:
        return Confidence.HIGH
    if n_observations >= medium_n:
        return Confidence.MEDIUM
    return Confidence.LOW


def build_finding(
    key: str,
    title: str,
    fact: str,
    insight: str,
    implication: str,
    action: str,
    evidence: Sequence[Evidence],
    finding_type: FindingType = FindingType.OBSERVATION,
    severity: Severity | None = None,
    confidence: Confidence | None = None,
    affected_clients: int | None = None,
    affected_share_pct: float | None = None,
    caveats: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> Finding:
    """Construct a :class:`Finding`, deriving severity and confidence if absent.

    Args:
        key: Stable identifier.
        title: Short headline.
        fact: Measured fact.
        insight: Analytical meaning.
        implication: Business consequence.
        action: Suggested consideration.
        evidence: Supporting computed values. Must not be empty.
        finding_type: Finding type.
        severity: Explicit severity. Derived from reach when omitted.
        confidence: Explicit confidence. Derived from sample size when omitted.
        affected_clients: Clients concerned.
        affected_share_pct: Share of the selection concerned.
        caveats: Limitations to display.
        tags: Filtering labels.

    Returns:
        A validated :class:`Finding`.

    Raises:
        ValueError: If evidence is missing or a chain part is blank.
    """
    resolved_severity = severity or (
        severity_from_share(affected_share_pct)
        if affected_share_pct is not None
        else Severity.INFO
    )
    resolved_confidence = confidence or confidence_from_sample(
        max((item.n_observations or 0) for item in evidence) if evidence else 0
    )

    return Finding(
        key=key,
        title=title,
        fact=fact,
        insight=insight,
        implication=implication,
        action=action,
        finding_type=finding_type,
        severity=resolved_severity,
        confidence=resolved_confidence,
        evidence=tuple(evidence),
        affected_clients=affected_clients,
        affected_share_pct=affected_share_pct,
        caveats=tuple(caveats),
        tags=tuple(tags),
    )
