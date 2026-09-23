"""Tests for driver analysis, with strict enforcement of non-causal language."""

from __future__ import annotations

import pandas as pd
import pytest

from src.drivers import (
    CAUSATION_CAVEAT,
    DEFAULT_EXCLUDED_COLUMNS,
    DriverAnalysisResult,
    DriverAnalysisUnavailable,
    EvidenceStrength,
    build_consensus,
    candidate_features,
    categorical_tests,
    correlation_drivers,
    describe_driver,
    numeric_group_tests,
    run_driver_analysis,
    tree_importances,
)

#: Words that must never appear in generated driver text. This is observational
#: data, so causal claims are unavailable at any sample size.
BANNED_CAUSAL_WORDS = (
    "causes", "caused", "causing", "because of", "drives", "driven by",
    "leads to", "results in", "due to", "makes",
)


class TestCandidateFeatures:
    """Feature selection for analysis."""

    def test_splits_numeric_and_categorical(self, featured_df: pd.DataFrame) -> None:
        numeric, categorical = candidate_features(featured_df, "default_next_month")
        assert len(numeric) > 10
        assert "limit_band" in categorical or "education" in categorical

    def test_excludes_the_target(self, featured_df: pd.DataFrame) -> None:
        numeric, categorical = candidate_features(featured_df, "default_next_month")
        assert "default_next_month" not in numeric + categorical

    def test_excludes_the_identifier(self, featured_df: pd.DataFrame) -> None:
        numeric, _ = candidate_features(featured_df, "default_next_month")
        assert "client_id" not in numeric

    def test_excludes_protected_attributes(self, featured_df: pd.DataFrame) -> None:
        """Pricing credit risk on sex or marital status is discriminatory."""
        numeric, categorical = candidate_features(featured_df, "default_next_month")
        combined = set(numeric + categorical)
        for protected in ("sex", "sex_code", "marriage", "marriage_code"):
            assert protected not in combined

    def test_prefers_label_over_raw_code(self, featured_df: pd.DataFrame) -> None:
        _numeric, categorical = candidate_features(featured_df, "default_next_month")
        if "education" in categorical:
            assert "education_code" not in categorical

    def test_excludes_constant_columns(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.copy()
        frame["always_one"] = 1
        numeric, _ = candidate_features(frame, "default_next_month")
        assert "always_one" not in numeric

    def test_extra_exclusions_are_honoured(self, featured_df: pd.DataFrame) -> None:
        numeric, _ = candidate_features(
            featured_df, "default_next_month", exclude=["credit_limit"]
        )
        assert "credit_limit" not in numeric

    def test_protected_list_is_documented(self) -> None:
        assert "sex" in DEFAULT_EXCLUDED_COLUMNS
        assert "marriage" in DEFAULT_EXCLUDED_COLUMNS


class TestCorrelationDrivers:
    """Rank correlation with the outcome."""

    def test_produces_evidence(self, featured_df: pd.DataFrame) -> None:
        numeric, _ = candidate_features(featured_df, "default_next_month")
        evidence = correlation_drivers(featured_df, "default_next_month", numeric)
        assert evidence
        assert all(-1.0 <= e.effect_size <= 1.0 for e in evidence)

    def test_sorted_by_absolute_effect(self, featured_df: pd.DataFrame) -> None:
        numeric, _ = candidate_features(featured_df, "default_next_month")
        evidence = correlation_drivers(featured_df, "default_next_month", numeric)
        magnitudes = [abs(e.effect_size) for e in evidence]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_finds_the_planted_signal(self, featured_df: pd.DataFrame) -> None:
        evidence = correlation_drivers(
            featured_df, "default_next_month", ["utilisation_latest"]
        )
        assert evidence[0].direction == "positive"

    def test_missing_target_yields_nothing(self, featured_df: pd.DataFrame) -> None:
        assert correlation_drivers(featured_df, "absent", ["credit_limit"]) == []


class TestNumericGroupTests:
    """Mann-Whitney U between outcome classes."""

    def test_produces_evidence_with_medians(self, featured_df: pd.DataFrame) -> None:
        evidence = numeric_group_tests(
            featured_df, "default_next_month", ["utilisation_latest", "credit_limit"]
        )
        assert evidence
        assert "median_when_outcome_1" in evidence[0].details

    def test_effect_size_is_bounded(self, featured_df: pd.DataFrame) -> None:
        numeric, _ = candidate_features(featured_df, "default_next_month")
        for item in numeric_group_tests(featured_df, "default_next_month", numeric):
            assert -1.0 <= item.effect_size <= 1.0

    def test_reports_a_p_value(self, featured_df: pd.DataFrame) -> None:
        evidence = numeric_group_tests(
            featured_df, "default_next_month", ["utilisation_latest"]
        )
        assert evidence[0].p_value is not None

    def test_skips_when_a_class_is_too_small(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.copy()
        frame["default_next_month"] = 0
        frame.loc[frame.index[:2], "default_next_month"] = 1
        assert numeric_group_tests(frame, "default_next_month", ["credit_limit"]) == []


class TestCategoricalTests:
    """Chi-square with Cramer's V."""

    def test_produces_evidence(self, featured_df: pd.DataFrame) -> None:
        _numeric, categorical = candidate_features(featured_df, "default_next_month")
        evidence = categorical_tests(featured_df, "default_next_month", categorical)
        assert evidence
        assert all(e.effect_size_name == "Cramer's V" for e in evidence)

    def test_cramers_v_is_bounded(self, featured_df: pd.DataFrame) -> None:
        _numeric, categorical = candidate_features(featured_df, "default_next_month")
        for item in categorical_tests(featured_df, "default_next_month", categorical):
            assert 0.0 <= item.effect_size <= 1.0

    def test_includes_per_category_rates(self, featured_df: pd.DataFrame) -> None:
        evidence = categorical_tests(featured_df, "default_next_month", ["limit_band"])
        assert evidence[0].details["outcome_rate_by_category_pct"]

    def test_flags_violated_test_assumptions(self) -> None:
        """A tiny expected cell count makes the p-value untrustworthy; say so."""
        frame = pd.DataFrame(
            {
                "grp": ["a"] * 50 + ["b"] * 3,
                "target": [0] * 50 + [1, 0, 1],
            }
        )
        evidence = categorical_tests(frame, "target", ["grp"], min_expected=5)
        assert evidence
        assert evidence[0].details["test_assumption_met"] is False
        assert "unreliable" in evidence[0].details["assumption_note"].lower()

    def test_single_category_is_skipped(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.copy()
        frame["one_value"] = "same"
        assert categorical_tests(frame, "default_next_month", ["one_value"]) == []


class TestTreeImportances:
    """Shallow decision-tree importance."""

    def test_produces_importances(self, featured_df: pd.DataFrame) -> None:
        numeric, _ = candidate_features(featured_df, "default_next_month")
        evidence = tree_importances(featured_df, "default_next_month", numeric)
        assert evidence
        assert all(e.effect_size > 0 for e in evidence)

    def test_documents_the_model_and_its_bias(self, featured_df: pd.DataFrame) -> None:
        numeric, _ = candidate_features(featured_df, "default_next_month")
        evidence = tree_importances(featured_df, "default_next_month", numeric)
        details = evidence[0].details
        assert "DecisionTreeClassifier" in details["model"]
        assert "biased" in details["note"].lower()

    def test_is_deterministic(self, featured_df: pd.DataFrame) -> None:
        numeric, _ = candidate_features(featured_df, "default_next_month")
        first = tree_importances(featured_df, "default_next_month", numeric, random_state=42)
        second = tree_importances(featured_df, "default_next_month", numeric, random_state=42)
        assert [e.feature for e in first] == [e.feature for e in second]
        assert [e.effect_size for e in first] == [e.effect_size for e in second]

    def test_skips_tiny_samples(self, featured_df: pd.DataFrame) -> None:
        assert tree_importances(featured_df.head(10), "default_next_month", ["credit_limit"]) == []

    def test_skips_single_class_target(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.copy()
        frame["default_next_month"] = 0
        assert tree_importances(frame, "default_next_month", ["credit_limit"]) == []


class TestConsensus:
    """Triangulating across methods."""

    def test_ranks_by_agreement_then_effect(self, featured_df: pd.DataFrame) -> None:
        numeric, categorical = candidate_features(featured_df, "default_next_month")
        consensus = build_consensus(
            correlation_drivers(featured_df, "default_next_month", numeric),
            numeric_group_tests(featured_df, "default_next_month", numeric),
            categorical_tests(featured_df, "default_next_month", categorical),
        )
        assert consensus
        keys = [(r["n_methods_nonneglible"], r["mean_effect_size"]) for r in consensus]
        assert keys == sorted(keys, reverse=True)

    def test_records_every_contributing_method(self, featured_df: pd.DataFrame) -> None:
        evidence = correlation_drivers(
            featured_df, "default_next_month", ["utilisation_latest"]
        )
        consensus = build_consensus(evidence)
        assert consensus[0]["n_methods"] == 1
        assert "spearman_correlation" in consensus[0]["methods"]

    def test_conflicting_directions_reported_as_mixed(self) -> None:
        from src.drivers import DriverEvidence

        positive = DriverEvidence(
            feature="x", method="a", statistic=0.5, effect_size=0.5,
            effect_size_name="r", strength=EvidenceStrength.MODERATE, direction="positive",
        )
        negative = DriverEvidence(
            feature="x", method="b", statistic=-0.5, effect_size=-0.5,
            effect_size_name="r", strength=EvidenceStrength.MODERATE, direction="negative",
        )
        assert build_consensus([positive], [negative])[0]["direction"] == "mixed"

    def test_respects_the_limit(self, featured_df: pd.DataFrame) -> None:
        numeric, _ = candidate_features(featured_df, "default_next_month")
        evidence = correlation_drivers(featured_df, "default_next_month", numeric)
        assert len(build_consensus(evidence, limit=5)) <= 5

    def test_empty_input_is_handled(self) -> None:
        assert build_consensus([]) == ()


class TestLanguageDiscipline:
    """No causal claims, anywhere."""

    def test_caveat_states_no_causation(self) -> None:
        assert "not causal" in CAUSATION_CAVEAT.lower()
        assert "no causal claim" in CAUSATION_CAVEAT.lower()

    def test_every_sentence_avoids_causal_words(self, featured_df: pd.DataFrame) -> None:
        result = run_driver_analysis(featured_df)
        assert isinstance(result, DriverAnalysisResult)

        sentences: list[str] = []
        for group in (
            result.correlations, result.numeric_comparisons,
            result.categorical_tests, result.tree_importances,
        ):
            sentences.extend(describe_driver(item, "default") for item in group)

        assert sentences, "there should be sentences to check"
        combined = " ".join(sentences).lower()
        found = [word for word in BANNED_CAUSAL_WORDS if word in combined]
        assert found == [], f"causal language leaked into driver text: {found}"

    def test_sentences_use_association_language(self, featured_df: pd.DataFrame) -> None:
        result = run_driver_analysis(featured_df)
        sentences = [describe_driver(e, "default") for e in result.numeric_comparisons[:5]]
        for sentence in sentences:
            assert any(
                phrase in sentence.lower()
                for phrase in ("associated", "association", "predictive feature")
            ), sentence

    def test_negligible_evidence_is_stated_plainly(self) -> None:
        from src.drivers import DriverEvidence

        evidence = DriverEvidence(
            feature="noise", method="spearman_correlation", statistic=0.01,
            effect_size=0.01, effect_size_name="r_s",
            strength=EvidenceStrength.NEGLIGIBLE, direction="positive",
        )
        assert "no meaningful association" in describe_driver(evidence).lower()

    def test_tree_wording_says_predictive_feature(self) -> None:
        from src.drivers import DriverEvidence

        evidence = DriverEvidence(
            feature="x", method="tree_importance", statistic=0.6, effect_size=0.6,
            effect_size_name="relative importance",
            strength=EvidenceStrength.STRONG, direction="n/a",
        )
        text = describe_driver(evidence, "default").lower()
        assert "predictive feature" in text
        assert "causes" not in text


class TestRunDriverAnalysis:
    """The orchestrated run."""

    def test_runs_all_four_methods(self, featured_df: pd.DataFrame) -> None:
        result = run_driver_analysis(featured_df)
        assert isinstance(result, DriverAnalysisResult)
        assert result.correlations and result.numeric_comparisons
        assert result.categorical_tests and result.tree_importances

    def test_reports_the_outcome_rate(self, featured_df: pd.DataFrame) -> None:
        result = run_driver_analysis(featured_df)
        expected = float(featured_df["default_next_month"].mean() * 100)
        assert result.outcome_rate_pct == pytest.approx(expected)

    def test_attaches_both_caveats(self, featured_df: pd.DataFrame) -> None:
        result = run_driver_analysis(featured_df)
        assert result.caveat == CAUSATION_CAVEAT
        assert "effect size" in result.significance_note.lower()

    def test_unavailable_without_a_target(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.drop(columns=["default_next_month"])
        result = run_driver_analysis(frame)
        assert isinstance(result, DriverAnalysisUnavailable)
        assert "no target" in result.reason.lower()

    def test_unavailable_with_too_few_rows(self, featured_df: pd.DataFrame) -> None:
        result = run_driver_analysis(featured_df.head(10))
        assert isinstance(result, DriverAnalysisUnavailable)
        assert "at least" in result.reason.lower()

    def test_unavailable_with_a_single_class(self, featured_df: pd.DataFrame) -> None:
        frame = featured_df.copy()
        frame["default_next_month"] = 0
        result = run_driver_analysis(frame)
        assert isinstance(result, DriverAnalysisUnavailable)
        assert "one class" in result.reason.lower()

    def test_result_is_serialisable(self, featured_df: pd.DataFrame) -> None:
        import json

        json.dumps(run_driver_analysis(featured_df).to_dict())

    def test_top_features_respects_limit(self, featured_df: pd.DataFrame) -> None:
        assert len(run_driver_analysis(featured_df).top_features(5)) <= 5

    def test_can_skip_the_tree(self, featured_df: pd.DataFrame) -> None:
        result = run_driver_analysis(featured_df, include_tree=False)
        assert result.tree_importances == ()


@pytest.mark.integration
class TestRealDrivers:
    """Driver analysis on the real dataset."""

    def test_delinquency_leads_the_consensus(self, real_pipeline) -> None:
        """A measured result: payment history dominates on this data."""
        result = run_driver_analysis(real_pipeline.data, registry=real_pipeline.registry)
        top_three = [r["feature"] for r in result.top_features(3)]
        assert any("delinquen" in f or "pay_status" in f for f in top_three)

    def test_outcome_rate_is_the_verified_value(self, real_pipeline) -> None:
        result = run_driver_analysis(real_pipeline.data, registry=real_pipeline.registry)
        assert result.outcome_rate_pct == pytest.approx(22.12, abs=0.01)

    def test_no_causal_language_on_real_data(self, real_pipeline) -> None:
        result = run_driver_analysis(real_pipeline.data, registry=real_pipeline.registry)
        combined = " ".join(
            describe_driver(item, "default")
            for group in (
                result.correlations, result.numeric_comparisons,
                result.categorical_tests, result.tree_importances,
            )
            for item in group
        ).lower()
        assert [w for w in BANNED_CAUSAL_WORDS if w in combined] == []

    def test_protected_attributes_are_absent_from_results(self, real_pipeline) -> None:
        result = run_driver_analysis(real_pipeline.data, registry=real_pipeline.registry)
        features = {r["feature"] for r in result.consensus}
        for protected in ("sex", "sex_code", "marriage", "marriage_code"):
            assert protected not in features
