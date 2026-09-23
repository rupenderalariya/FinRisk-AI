"""Tests for finding and signal derivation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.findings import FindingType, Severity
from src.signals import (
    CAUSATION_CAVEAT,
    MIN_ROWS,
    OBSERVED_SIGNAL_LABEL,
    derive_key_findings,
    derive_signals,
    gradient_severity,
)

#: Language that must never appear in generated findings or signals.
BANNED_CAUSAL = (
    "causes", "caused by", "because of", "drives", "leads to",
    "results in", "will default", "guaranteed",
)

#: Language that must never appear before a predictive model exists.
BANNED_PREDICTION = ("predicted risk", "model predicts", "forecast")


class TestGradientSeverity:
    """Severity combining reach with the size of the gap."""

    def test_small_gap_uses_reach_only(self) -> None:
        from src.findings import severity_from_share

        assert gradient_severity(30.0, spread_pp=5.0) is severity_from_share(30.0)

    def test_large_gap_escalates_one_level(self) -> None:
        assert gradient_severity(15.0, spread_pp=25.0) is Severity.HIGH

    def test_very_large_gap_escalates_two_levels(self) -> None:
        assert gradient_severity(15.0, spread_pp=50.0) is Severity.CRITICAL

    def test_escalation_is_capped(self) -> None:
        assert gradient_severity(90.0, spread_pp=90.0) is Severity.CRITICAL

    def test_zero_reach_with_large_gap_still_escalates(self) -> None:
        assert gradient_severity(0.0, spread_pp=50.0) is not Severity.INFO


class TestDeriveKeyFindings:
    """Automatic finding derivation."""

    def test_produces_findings_on_a_healthy_selection(
        self, featured_df: pd.DataFrame
    ) -> None:
        findings = derive_key_findings(featured_df)
        assert len(findings) > 0

    def test_every_finding_carries_evidence(self, featured_df: pd.DataFrame) -> None:
        """The core traceability guarantee."""
        for finding in derive_key_findings(featured_df):
            assert finding.evidence, f"'{finding.key}' has no evidence"
            for item in finding.evidence:
                assert item.metric and item.label and item.source

    def test_every_finding_has_a_complete_chain(self, featured_df: pd.DataFrame) -> None:
        for finding in derive_key_findings(featured_df):
            for part in (finding.fact, finding.insight, finding.implication, finding.action):
                assert part.strip()
                assert len(part) > 20

    def test_no_causal_language(self, featured_df: pd.DataFrame) -> None:
        findings = derive_key_findings(featured_df, limit=10)
        text = " ".join(
            f"{f.title} {f.fact} {f.insight} {f.implication} {f.action}" for f in findings
        ).lower()
        found = [w for w in BANNED_CAUSAL if w in text]
        assert found == [], f"causal language leaked into findings: {found}"

    def test_no_prediction_language(self, featured_df: pd.DataFrame) -> None:
        """No model exists yet, so nothing may imply a prediction."""
        findings = derive_key_findings(featured_df, limit=10)
        text = " ".join(
            f"{f.title} {f.fact} {f.insight} {f.implication} {f.action}" for f in findings
        ).lower()
        found = [w for w in BANNED_PREDICTION if w in text]
        assert found == [], f"prediction language leaked into findings: {found}"

    def test_risk_findings_carry_the_caveat(self, featured_df: pd.DataFrame) -> None:
        findings = derive_key_findings(featured_df, limit=10)
        risk_findings = [f for f in findings if f.finding_type is FindingType.RISK]
        for finding in risk_findings:
            assert any(
                "not causal" in c.lower() or "not a causal" in c.lower()
                for c in finding.caveats
            ), f"'{finding.key}' lacks a causation caveat"

    def test_respects_the_limit(self, featured_df: pd.DataFrame) -> None:
        assert len(derive_key_findings(featured_df, limit=2)) <= 2

    def test_sorted_by_priority(self, featured_df: pd.DataFrame) -> None:
        from src.findings import SEVERITY_ORDER

        findings = list(derive_key_findings(featured_df, limit=10))
        ranks = [SEVERITY_ORDER[f.severity] for f in findings]
        assert ranks == sorted(ranks)

    def test_too_few_rows_returns_empty_with_a_reason(
        self, featured_df: pd.DataFrame
    ) -> None:
        findings = derive_key_findings(featured_df.head(20))
        assert len(findings) == 0
        assert str(MIN_ROWS) in findings.context

    def test_single_class_target_returns_empty_with_a_reason(
        self, featured_df: pd.DataFrame
    ) -> None:
        frame = featured_df.copy()
        frame["default_next_month"] = 0
        findings = derive_key_findings(frame)
        assert len(findings) == 0
        assert "one outcome class" in findings.context

    def test_missing_target_returns_empty(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.drop(columns=["default_next_month"])
        assert len(derive_key_findings(frame)) == 0

    def test_is_deterministic(self, featured_df: pd.DataFrame) -> None:
        first = derive_key_findings(featured_df, limit=6)
        second = derive_key_findings(featured_df, limit=6)
        assert [f.key for f in first] == [f.key for f in second]
        assert [f.fact for f in first] == [f.fact for f in second]

    def test_serialisable(self, featured_df: pd.DataFrame) -> None:
        json.dumps(derive_key_findings(featured_df).to_dict())

    def test_bare_frame_produces_nothing_rather_than_raising(self) -> None:
        frame = pd.DataFrame(
            {"client_id": range(200), "default_next_month": [0, 1] * 100}
        )
        assert len(derive_key_findings(frame)) >= 0  # must not raise


class TestDeriveSignals:
    """Risk signal and opportunity derivation."""

    def test_produces_signals(self, featured_df: pd.DataFrame) -> None:
        signals = derive_signals(featured_df)
        assert signals.risks or signals.opportunities

    def test_labelled_as_observed_not_predicted(self, featured_df: pd.DataFrame) -> None:
        signals = derive_signals(featured_df)
        assert signals.note == OBSERVED_SIGNAL_LABEL
        for signal in signals.risks + signals.opportunities:
            assert signal.label == OBSERVED_SIGNAL_LABEL

    def test_no_prediction_language(self, featured_df: pd.DataFrame) -> None:
        signals = derive_signals(featured_df)
        text = " ".join(
            f"{s.name} {s.description}" for s in signals.risks + signals.opportunities
        ).lower()
        found = [w for w in BANNED_PREDICTION if w in text]
        assert found == [], f"prediction language leaked into signals: {found}"

    def test_no_causal_language(self, featured_df: pd.DataFrame) -> None:
        signals = derive_signals(featured_df)
        text = " ".join(
            f"{s.name} {s.description}" for s in signals.risks + signals.opportunities
        ).lower()
        found = [w for w in BANNED_CAUSAL if w in text]
        assert found == [], f"causal language leaked into signals: {found}"

    def test_every_signal_carries_evidence(self, featured_df: pd.DataFrame) -> None:
        signals = derive_signals(featured_df)
        for signal in signals.risks + signals.opportunities:
            assert signal.evidence, f"signal '{signal.key}' has no evidence"

    def test_intensity_is_bounded(self, featured_df: pd.DataFrame) -> None:
        signals = derive_signals(featured_df)
        for signal in signals.risks + signals.opportunities:
            assert 0.0 <= signal.intensity <= 1.0

    def test_risks_sorted_by_intensity(self, featured_df: pd.DataFrame) -> None:
        intensities = [s.intensity for s in derive_signals(featured_df).risks]
        assert intensities == sorted(intensities, reverse=True)

    def test_too_few_rows_returns_nothing(self, featured_df: pd.DataFrame) -> None:
        signals = derive_signals(featured_df.head(20))
        assert signals.risks == ()
        assert signals.opportunities == ()

    def test_empty_frame_returns_nothing(self, featured_df: pd.DataFrame) -> None:
        signals = derive_signals(featured_df.iloc[0:0])
        assert signals.n_rows == 0
        assert signals.risks == ()

    def test_opportunities_compare_against_the_average(
        self, featured_df: pd.DataFrame
    ) -> None:
        """An opportunity must state the baseline it beats."""
        for signal in derive_signals(featured_df).opportunities:
            assert "overall" in signal.description.lower()

    def test_is_deterministic(self, featured_df: pd.DataFrame) -> None:
        first = derive_signals(featured_df)
        second = derive_signals(featured_df)
        assert [s.key for s in first.risks] == [s.key for s in second.risks]
        assert [s.value for s in first.risks] == [s.value for s in second.risks]

    def test_serialisable(self, featured_df: pd.DataFrame) -> None:
        json.dumps(derive_signals(featured_df).to_dict())

    def test_bare_frame_does_not_raise(self) -> None:
        frame = pd.DataFrame({"client_id": range(200)})
        signals = derive_signals(frame)
        assert signals.risks == () and signals.opportunities == ()


class TestFindingsReflectFilters:
    """Findings must describe the selection handed to them."""

    def test_different_selections_give_different_findings(
        self, featured_df: pd.DataFrame
    ) -> None:
        full = derive_key_findings(featured_df, limit=6)
        subset = derive_key_findings(
            featured_df[featured_df["is_currently_delinquent"]], limit=6
        )
        if len(subset) and len(full):
            assert {f.key for f in full} != {f.key for f in subset} or [
                f.fact for f in full
            ] != [f.fact for f in subset]

    def test_evidence_counts_match_the_selection(self, featured_df: pd.DataFrame) -> None:
        subset = featured_df.head(300)
        for finding in derive_key_findings(subset, limit=6):
            for item in finding.evidence:
                if item.n_observations is not None:
                    assert item.n_observations <= len(subset)


@pytest.mark.integration
class TestRealSignals:
    """Signals derived from the real dataset."""

    def test_finds_the_delinquency_gradient(self, real_pipeline) -> None:
        findings = derive_key_findings(real_pipeline.data, real_pipeline.registry, limit=8)
        assert any(f.key == "delinquency_gradient" for f in findings)

    def test_delinquency_gradient_is_top_severity(self, real_pipeline) -> None:
        """A 58-point separation must not be ranked below a wider but weaker pattern."""
        findings = derive_key_findings(real_pipeline.data, real_pipeline.registry, limit=8)
        gradient = next(f for f in findings if f.key == "delinquency_gradient")
        assert gradient.severity is Severity.CRITICAL

    def test_utilisation_gradient_excludes_the_negative_band(self, real_pipeline) -> None:
        """Accounts in credit are not a low point on the utilisation scale."""
        findings = derive_key_findings(real_pipeline.data, real_pipeline.registry, limit=8)
        utilisation = next(
            (f for f in findings if f.key == "utilisation_gradient"), None
        )
        assert utilisation is not None
        assert "excluded from the gradient" in utilisation.fact

    def test_finds_the_full_payer_opportunity(self, real_pipeline) -> None:
        findings = derive_key_findings(real_pipeline.data, real_pipeline.registry, limit=8)
        opportunities = [f for f in findings if f.finding_type is FindingType.OPPORTUNITY]
        assert opportunities

    def test_expected_signals_are_present(self, real_pipeline) -> None:
        from src.kpi_engine import calculate_kpis

        kpis = calculate_kpis(real_pipeline.data, real_pipeline.registry)
        signals = derive_signals(real_pipeline.data, kpis, real_pipeline.registry)
        risk_keys = {s.key for s in signals.risks}
        assert {"high_utilisation", "payment_delinquency", "repayment_behaviour"} <= risk_keys
        assert signals.opportunities

    def test_signal_values_match_the_kpis(self, real_pipeline) -> None:
        from src.kpi_engine import calculate_kpis

        kpis = calculate_kpis(real_pipeline.data, real_pipeline.registry)
        signals = derive_signals(real_pipeline.data, kpis, real_pipeline.registry)
        by_key = {s.key: s for s in signals.risks}
        assert by_key["high_utilisation"].raw_value == pytest.approx(
            kpis.raw("high_utilisation_share"), abs=0.01
        )
        assert by_key["payment_delinquency"].raw_value == pytest.approx(
            kpis.raw("currently_delinquent_share"), abs=0.01
        )

    def test_no_causal_or_prediction_language_on_real_data(self, real_pipeline) -> None:
        findings = derive_key_findings(real_pipeline.data, real_pipeline.registry, limit=10)
        signals = derive_signals(real_pipeline.data, registry=real_pipeline.registry)
        text = " ".join(
            [f"{f.title} {f.fact} {f.insight} {f.implication} {f.action}" for f in findings]
            + [f"{s.name} {s.description}" for s in signals.risks + signals.opportunities]
        ).lower()
        assert [w for w in BANNED_CAUSAL if w in text] == []
        assert [w for w in BANNED_PREDICTION if w in text] == []
