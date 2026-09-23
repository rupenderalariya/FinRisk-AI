"""Tests for the analytical findings structure and its evidence guarantees."""

from __future__ import annotations

import json

import pytest

from src.findings import (
    Confidence,
    Evidence,
    Finding,
    FindingSet,
    FindingType,
    Severity,
    build_finding,
    confidence_from_sample,
    make_evidence,
    severity_from_share,
)


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    """Two evidence items referencing computed metrics."""
    return [
        make_evidence(
            metric="observed_default_rate",
            label="Observed default rate",
            value=22.12,
            formatted="22.12%",
            source="kpi_engine",
            n_observations=30_000,
        ),
        make_evidence(
            metric="high_utilisation_share",
            label="High-utilisation clients",
            value=26.6,
            formatted="26.60%",
            source="kpi_engine",
            comparison="portfolio average",
            n_observations=30_000,
        ),
    ]


class TestEvidence:
    """Evidence items."""

    def test_describe_includes_label_and_value(self, sample_evidence) -> None:
        text = sample_evidence[0].describe()
        assert "Observed default rate" in text
        assert "22.12%" in text
        assert "30,000" in text

    def test_comparison_is_included(self, sample_evidence) -> None:
        assert "vs portfolio average" in sample_evidence[1].describe()

    def test_auto_formats_a_float(self) -> None:
        assert make_evidence("m", "L", 1234.5678, source="t").formatted == "1,234.57"

    def test_auto_formats_an_int(self) -> None:
        assert make_evidence("m", "L", 30000, source="t").formatted == "30,000"

    def test_auto_formats_a_string(self) -> None:
        assert make_evidence("m", "L", "high", source="t").formatted == "high"

    def test_is_serialisable(self, sample_evidence) -> None:
        json.dumps([e.to_dict() for e in sample_evidence])


class TestFindingValidation:
    """The guarantees that stop unsupported claims entering the system."""

    def test_evidence_is_mandatory(self) -> None:
        with pytest.raises(ValueError, match="no supporting evidence"):
            Finding(
                key="k", title="t", fact="f", insight="i", implication="im", action="a",
                finding_type=FindingType.RISK, severity=Severity.HIGH,
                confidence=Confidence.HIGH, evidence=(),
            )

    def test_builder_also_requires_evidence(self) -> None:
        with pytest.raises(ValueError, match="no supporting evidence"):
            build_finding("k", "t", "f", "i", "im", "a", evidence=[])

    @pytest.mark.parametrize("blank_part", ["fact", "insight", "implication", "action"])
    def test_all_four_chain_parts_are_required(
        self, sample_evidence, blank_part: str
    ) -> None:
        parts = {"fact": "f", "insight": "i", "implication": "im", "action": "a"}
        parts[blank_part] = ""
        with pytest.raises(ValueError, match=blank_part):
            build_finding("k", "t", evidence=sample_evidence, **parts)

    def test_whitespace_only_part_is_rejected(self, sample_evidence) -> None:
        with pytest.raises(ValueError, match="fact"):
            build_finding("k", "t", "   ", "i", "im", "a", evidence=sample_evidence)


class TestFindingChain:
    """The FACT -> INSIGHT -> IMPLICATION -> ACTION chain."""

    def test_chain_contains_all_four_labels(self, sample_evidence) -> None:
        finding = build_finding(
            "k", "Title", "A fact.", "An insight.", "An implication.", "An action.",
            evidence=sample_evidence,
        )
        chain = finding.to_chain()
        for label in ("FACT:", "INSIGHT:", "IMPLICATION:", "ACTION:"):
            assert label in chain

    def test_caveats_appear_in_the_chain(self, sample_evidence) -> None:
        finding = build_finding(
            "k", "T", "f", "i", "im", "a", evidence=sample_evidence,
            caveats=["Historical data from 2005."],
        )
        assert "CAVEATS:" in finding.to_chain()
        assert "2005" in finding.to_chain()

    def test_evidence_summary_joins_all_items(self, sample_evidence) -> None:
        finding = build_finding("k", "T", "f", "i", "im", "a", evidence=sample_evidence)
        assert finding.evidence_summary.count(";") == 1

    def test_is_serialisable(self, sample_evidence) -> None:
        finding = build_finding("k", "T", "f", "i", "im", "a", evidence=sample_evidence)
        json.dumps(finding.to_dict())


class TestSeverityDerivation:
    """Severity proportional to measured reach."""

    @pytest.mark.parametrize(
        "share,expected",
        [
            (45.0, Severity.CRITICAL),
            (40.0, Severity.CRITICAL),
            (30.0, Severity.HIGH),
            (25.0, Severity.HIGH),
            (15.0, Severity.MEDIUM),
            (10.0, Severity.MEDIUM),
            (5.0, Severity.LOW),
            (0.0, Severity.INFO),
        ],
    )
    def test_thresholds(self, share: float, expected: Severity) -> None:
        assert severity_from_share(share) is expected

    def test_custom_thresholds(self) -> None:
        assert severity_from_share(20.0, critical_at=15.0) is Severity.CRITICAL

    def test_derived_when_not_supplied(self, sample_evidence) -> None:
        finding = build_finding(
            "k", "T", "f", "i", "im", "a", evidence=sample_evidence,
            affected_share_pct=45.0,
        )
        assert finding.severity is Severity.CRITICAL

    def test_explicit_severity_wins(self, sample_evidence) -> None:
        finding = build_finding(
            "k", "T", "f", "i", "im", "a", evidence=sample_evidence,
            severity=Severity.LOW, affected_share_pct=90.0,
        )
        assert finding.severity is Severity.LOW


class TestConfidenceDerivation:
    """Confidence from sample size and method agreement."""

    def test_large_sample_with_agreement_is_high(self) -> None:
        assert confidence_from_sample(5_000, n_methods_agreeing=2) is Confidence.HIGH

    def test_large_sample_without_agreement_is_medium(self) -> None:
        assert confidence_from_sample(5_000, n_methods_agreeing=1) is Confidence.MEDIUM

    def test_medium_sample(self) -> None:
        assert confidence_from_sample(500) is Confidence.MEDIUM

    def test_small_sample_is_low(self) -> None:
        assert confidence_from_sample(20) is Confidence.LOW

    def test_derived_from_evidence_observations(self, sample_evidence) -> None:
        finding = build_finding("k", "T", "f", "i", "im", "a", evidence=sample_evidence)
        assert finding.confidence is Confidence.MEDIUM  # 30,000 rows, one method


class TestFindingSet:
    """The findings collection."""

    @pytest.fixture
    def populated_set(self, sample_evidence) -> FindingSet:
        return FindingSet(
            findings=(
                build_finding(
                    "risk_high", "R1", "f", "i", "im", "a", evidence=sample_evidence,
                    finding_type=FindingType.RISK, severity=Severity.CRITICAL,
                    affected_share_pct=50.0,
                ),
                build_finding(
                    "risk_low", "R2", "f", "i", "im", "a", evidence=sample_evidence,
                    finding_type=FindingType.RISK, severity=Severity.LOW,
                    affected_share_pct=3.0,
                ),
                build_finding(
                    "opp", "O1", "f", "i", "im", "a", evidence=sample_evidence,
                    finding_type=FindingType.OPPORTUNITY, severity=Severity.MEDIUM,
                    affected_share_pct=20.0,
                ),
                build_finding(
                    "obs", "B1", "f", "i", "im", "a", evidence=sample_evidence,
                    finding_type=FindingType.OBSERVATION, severity=Severity.INFO,
                ),
                build_finding(
                    "dq", "D1", "f", "i", "im", "a", evidence=sample_evidence,
                    finding_type=FindingType.DATA_QUALITY, severity=Severity.HIGH,
                    affected_share_pct=30.0,
                ),
            ),
            context="test",
        )

    def test_length_and_truthiness(self, populated_set: FindingSet) -> None:
        assert len(populated_set) == 5
        assert bool(populated_set) is True
        assert bool(FindingSet()) is False

    def test_type_filters(self, populated_set: FindingSet) -> None:
        assert len(populated_set.risks) == 2
        assert len(populated_set.opportunities) == 1
        assert len(populated_set.observations) == 1
        assert len(populated_set.data_quality) == 1

    def test_severity_filter(self, populated_set: FindingSet) -> None:
        assert len(populated_set.of_severity(Severity.CRITICAL, Severity.HIGH)) == 2

    def test_actionable_excludes_info_and_low(self, populated_set: FindingSet) -> None:
        keys = {f.key for f in populated_set.actionable()}
        assert "obs" not in keys and "risk_low" not in keys
        assert "risk_high" in keys

    def test_sorted_by_priority(self, populated_set: FindingSet) -> None:
        ordered = populated_set.sorted_by_priority()
        assert ordered[0].severity is Severity.CRITICAL
        assert ordered[-1].severity is Severity.INFO

    def test_top_respects_limit(self, populated_set: FindingSet) -> None:
        assert len(populated_set.top(2)) == 2

    def test_counts(self, populated_set: FindingSet) -> None:
        assert populated_set.counts_by_type["risk"] == 2
        assert populated_set.counts_by_severity["critical"] == 1

    def test_summary_mentions_each_type(self, populated_set: FindingSet) -> None:
        summary = populated_set.summary()
        for word in ("risk", "opportunity", "observation", "data quality"):
            assert word in summary

    def test_empty_set_summary(self) -> None:
        assert "No findings" in FindingSet().summary()

    def test_extend_returns_a_new_set(self, populated_set, sample_evidence) -> None:
        extra = build_finding("new", "N", "f", "i", "im", "a", evidence=sample_evidence)
        extended = populated_set.extend([extra])
        assert len(extended) == 6
        assert len(populated_set) == 5, "the original must be unchanged"

    def test_is_iterable(self, populated_set: FindingSet) -> None:
        assert len(list(populated_set)) == 5

    def test_is_serialisable(self, populated_set: FindingSet) -> None:
        json.dumps(populated_set.to_dict())

    def test_dict_findings_are_priority_sorted(self, populated_set: FindingSet) -> None:
        payload = populated_set.to_dict()
        assert payload["findings"][0]["severity"] == "critical"
