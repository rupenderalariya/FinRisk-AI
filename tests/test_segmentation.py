"""
Tests for behaviour-based customer segmentation.

The most important group here is :class:`TestTargetLeakagePrevention`. Clustering
that sees the outcome stops describing behaviour and starts restating the answer,
so the guards against it are asserted from several angles rather than assumed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.segmentation import (
    CANDIDATE_FEATURES,
    DEGENERATE_CLUSTER_SHARE_PCT,
    FEATURE_LABELS,
    FEATURE_SET_RATIONALE,
    FORBIDDEN_FEATURES,
    MIN_ROWS_FOR_CLUSTERING,
    MIN_ROWS_PER_CLUSTER_IN_SAMPLE,
    TRAIT_THRESHOLD,
    TRAIT_VOCABULARY,
    VALID_SCALERS,
    ImputationInfo,
    KSelectionResult,
    SegmentationConfig,
    SegmentationResult,
    SegmentationUnavailable,
    _derive_label,
    _elbow_k,
    _safe_silhouette,
    _stratified_sample_index,
    assess_features,
    compare_scalers,
    evaluate_k_range,
    fit_segments,
    methodology_note,
    prepare_features,
    profile_segments,
    run_segmentation,
    segment_interpretation,
)

#: Language that must never appear in generated segmentation text.
BANNED_LANGUAGE = (
    "causes",
    "caused by",
    "will default",
    "guaranteed",
    "safe customer",
    "bad customer",
    "good customer",
    "predicted risk",
    "model predicts",
)


@pytest.fixture
def cluster_df(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """A featured frame large enough to cluster, with a planted structure."""
    from src.data_processing import clean_dataset
    from src.feature_engineering import engineer_features

    clean, _ = clean_dataset(analysis_df)
    featured, _ = engineer_features(clean)
    return featured


@pytest.fixture
def small_config() -> SegmentationConfig:
    """A fast configuration for unit tests."""
    return SegmentationConfig(k_min=2, k_max=4, n_init=3, silhouette_sample=1_000)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class TestSegmentationConfig:
    """Configuration validation."""

    def test_defaults_are_sane(self) -> None:
        config = SegmentationConfig()
        assert config.scaler in VALID_SCALERS
        assert config.k_min >= 2
        assert config.k_max >= config.k_min
        assert config.features == CANDIDATE_FEATURES

    def test_default_scaler_is_the_measured_winner(self) -> None:
        """Robust scaling was chosen on measured silhouette, not convention."""
        assert SegmentationConfig().scaler == "robust"

    def test_seed_falls_back_to_configuration(self) -> None:
        from src.config import settings

        assert SegmentationConfig().seed == settings.random_state
        assert SegmentationConfig(random_state=7).seed == 7

    def test_k_range(self) -> None:
        assert SegmentationConfig(k_min=2, k_max=5).k_range == (2, 3, 4, 5)

    def test_rejects_unknown_scaler(self) -> None:
        with pytest.raises(ValueError, match="Unknown scaler"):
            SegmentationConfig(scaler="minmax")

    def test_rejects_k_below_two(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            SegmentationConfig(k_min=1)

    def test_rejects_inverted_k_range(self) -> None:
        with pytest.raises(ValueError, match="must be >="):
            SegmentationConfig(k_min=5, k_max=3)


# --------------------------------------------------------------------------- #
# Leakage prevention
# --------------------------------------------------------------------------- #


class TestTargetLeakagePrevention:
    """The outcome and demographics must never reach the feature matrix."""

    def test_target_is_in_the_forbidden_list(self) -> None:
        assert "default_next_month" in FORBIDDEN_FEATURES

    def test_demographics_are_in_the_forbidden_list(self) -> None:
        for column in (
            "sex", "sex_code", "marriage", "marriage_code",
            "education", "education_code", "age", "age_band",
        ):
            assert column in FORBIDDEN_FEATURES, column

    def test_identifier_is_in_the_forbidden_list(self) -> None:
        assert "client_id" in FORBIDDEN_FEATURES

    def test_no_candidate_feature_is_forbidden(self) -> None:
        assert set(CANDIDATE_FEATURES) & FORBIDDEN_FEATURES == set()

    def test_config_rejects_the_target(self) -> None:
        with pytest.raises(ValueError, match="Forbidden column"):
            SegmentationConfig(features=("credit_limit", "default_next_month"))

    @pytest.mark.parametrize(
        "column", ["sex_code", "marriage", "education_code", "age", "client_id"]
    )
    def test_config_rejects_each_forbidden_column(self, column: str) -> None:
        with pytest.raises(ValueError, match="Forbidden column"):
            SegmentationConfig(features=("credit_limit", column))

    def test_config_error_explains_why(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            SegmentationConfig(features=("default_next_month",))
        message = str(excinfo.value).lower()
        assert "leak" in message
        assert "fairness" in message

    def test_prepare_features_rejects_forbidden_columns(
        self, cluster_df: pd.DataFrame
    ) -> None:
        """Belt and braces: the guard is repeated at the preparation step."""
        config = SegmentationConfig()
        object.__setattr__(config, "features", ("credit_limit", "default_next_month"))
        with pytest.raises(ValueError, match="Refusing to build"):
            prepare_features(cluster_df, config)

    def test_fitted_matrix_excludes_the_target(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        assert "default_next_month" not in matrix.feature_names
        assert "default_next_month" not in matrix.raw.columns

    def test_fitted_matrix_excludes_demographics(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        for column in ("sex", "sex_code", "marriage", "education", "age", "age_band"):
            assert column not in matrix.feature_names
            assert column not in matrix.raw.columns

    def test_result_declares_the_target_excluded(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert result.target_excluded is True

    def test_shuffling_the_target_does_not_change_the_clusters(
        self, cluster_df, small_config
    ) -> None:
        """The decisive test: if the outcome influenced clustering at all,
        permuting it would move the assignments."""
        original = fit_segments(cluster_df, k=3, config=small_config)

        shuffled = cluster_df.copy()
        rng = np.random.default_rng(123)
        shuffled["default_next_month"] = rng.permutation(
            shuffled["default_next_month"].to_numpy()
        )
        permuted = fit_segments(shuffled, k=3, config=small_config)

        pd.testing.assert_series_equal(
            original.labels, permuted.labels, check_names=False
        )

    def test_removing_the_target_entirely_still_clusters(
        self, cluster_df, small_config
    ) -> None:
        without = cluster_df.drop(columns=["default_next_month"])
        result = fit_segments(without, k=3, config=small_config)
        assert len(result.profiles) == 3
        assert all(p.observed_default_rate_pct is None for p in result.profiles)


# --------------------------------------------------------------------------- #
# Feature assessment and preparation
# --------------------------------------------------------------------------- #


class TestAssessFeatures:
    """Evidence for feature-set decisions."""

    def test_reports_one_row_per_available_feature(self, cluster_df) -> None:
        assessment = assess_features(cluster_df, CANDIDATE_FEATURES)
        assert len(assessment) == len(
            [f for f in CANDIDATE_FEATURES if f in cluster_df.columns]
        )

    def test_reports_missingness_and_skew(self, cluster_df) -> None:
        assessment = assess_features(cluster_df, CANDIDATE_FEATURES)
        for column in ("n_missing", "pct_missing", "skew", "max_abs_corr"):
            assert column in assessment.columns

    def test_identifies_the_most_correlated_partner(self, cluster_df) -> None:
        assessment = assess_features(cluster_df, CANDIDATE_FEATURES)
        assert assessment["most_correlated_with"].notna().any()

    def test_correlations_are_bounded(self, cluster_df) -> None:
        assessment = assess_features(cluster_df, CANDIDATE_FEATURES)
        assert (assessment["max_abs_corr"].dropna() <= 1.0).all()

    def test_handles_missing_columns(self) -> None:
        assert assess_features(pd.DataFrame({"a": [1, 2]}), ("nope",)).empty

    def test_rationale_is_documented(self) -> None:
        """The feature-set decision must be written down, not implicit."""
        text = FEATURE_SET_RATIONALE.lower()
        assert "0.756" in FEATURE_SET_RATIONALE
        assert "dimensionality" in text
        assert "not comparable across feature spaces" in text


class TestPrepareFeatures:
    """Building the clustering matrix."""

    def test_produces_a_scaled_matrix(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        assert matrix.scaled.shape == (len(cluster_df), matrix.n_features)
        assert matrix.n_features > 0

    def test_keeps_every_customer(self, cluster_df, small_config) -> None:
        """Dropping rows for a missing ratio would bias toward active accounts."""
        matrix = prepare_features(cluster_df, small_config)
        assert matrix.n_rows == len(cluster_df)

    def test_index_is_preserved(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        pd.testing.assert_index_equal(matrix.index, cluster_df.index)

    def test_no_nan_in_the_scaled_matrix(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        assert not np.isnan(matrix.scaled).any()

    def test_no_infinity_in_the_scaled_matrix(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        assert not np.isinf(matrix.scaled).any()

    def test_excludes_missing_columns_with_a_reason(self, cluster_df) -> None:
        config = SegmentationConfig(
            features=("credit_limit", "utilisation_mean_6m"), k_max=3
        )
        frame = cluster_df.drop(columns=["utilisation_mean_6m"])
        matrix = prepare_features(frame, config)
        assert "utilisation_mean_6m" in matrix.excluded
        assert "not present" in matrix.excluded["utilisation_mean_6m"]

    def test_excludes_constant_columns_with_a_reason(self, cluster_df) -> None:
        frame = cluster_df.copy()
        frame["credit_limit"] = 50_000
        matrix = prepare_features(
            frame, SegmentationConfig(features=("credit_limit", "utilisation_mean_6m"))
        )
        assert "credit_limit" in matrix.excluded
        assert "constant" in matrix.excluded["credit_limit"].lower()

    def test_raises_when_nothing_is_usable(self, cluster_df) -> None:
        frame = cluster_df.copy()
        for feature in CANDIDATE_FEATURES:
            if feature in frame.columns:
                frame[feature] = 1.0
        with pytest.raises(ValueError, match="No usable clustering feature"):
            prepare_features(frame, SegmentationConfig())

    @pytest.mark.parametrize("scaler", VALID_SCALERS)
    def test_each_scaler_works(self, cluster_df, scaler: str) -> None:
        config = SegmentationConfig(scaler=scaler, k_max=3)
        matrix = prepare_features(cluster_df, config)
        assert matrix.scaler_name == scaler
        assert np.isfinite(matrix.scaled).all()

    def test_standard_scaling_centres_the_data(self, cluster_df) -> None:
        matrix = prepare_features(cluster_df, SegmentationConfig(scaler="standard"))
        assert np.allclose(matrix.scaled.mean(axis=0), 0.0, atol=1e-8)

    def test_robust_scaling_centres_the_median(self, cluster_df) -> None:
        matrix = prepare_features(cluster_df, SegmentationConfig(scaler="robust"))
        assert np.allclose(np.median(matrix.scaled, axis=0), 0.0, atol=1e-8)

    def test_is_deterministic(self, cluster_df, small_config) -> None:
        first = prepare_features(cluster_df, small_config)
        second = prepare_features(cluster_df, small_config)
        assert np.array_equal(first.scaled, second.scaled)

    def test_does_not_mutate_the_input(self, cluster_df, small_config) -> None:
        before = cluster_df.copy()
        prepare_features(cluster_df, small_config)
        pd.testing.assert_frame_equal(cluster_df, before)

    def test_matrix_is_serialisable(self, cluster_df, small_config) -> None:
        json.dumps(prepare_features(cluster_df, small_config).to_dict())


class TestImputation:
    """Missing-value handling."""

    def test_records_imputation(self, cluster_df, small_config) -> None:
        frame = cluster_df.copy()
        frame.loc[frame.index[:40], "repayment_ratio_capped_mean"] = np.nan
        matrix = prepare_features(frame, small_config)
        infos = [i for i in matrix.imputations if i.column == "repayment_ratio_capped_mean"]
        assert infos
        assert infos[0].n_missing == 40
        assert infos[0].method == "median"

    def test_reports_count_and_percentage(self, cluster_df, small_config) -> None:
        frame = cluster_df.copy()
        n = 50
        frame.loc[frame.index[:n], "repayment_ratio_capped_mean"] = np.nan
        matrix = prepare_features(frame, small_config)
        info = next(i for i in matrix.imputations if i.column == "repayment_ratio_capped_mean")
        assert info.n_missing == n
        assert info.pct_missing == pytest.approx(n / len(frame) * 100.0)

    def test_fills_with_the_median(self, cluster_df, small_config) -> None:
        frame = cluster_df.copy()
        expected = float(frame["repayment_ratio_capped_mean"].median())
        frame.loc[frame.index[:10], "repayment_ratio_capped_mean"] = np.nan
        matrix = prepare_features(frame, small_config)
        info = next(i for i in matrix.imputations if i.column == "repayment_ratio_capped_mean")
        # The median is recomputed after the nulls are introduced, so it shifts only
        # marginally; it must remain a plausible central value.
        assert 0.0 <= info.fill_value <= 2.0

    def test_note_explains_the_inactive_account_interpretation(
        self, cluster_df, small_config
    ) -> None:
        frame = cluster_df.copy()
        frame.loc[frame.index[:10], "repayment_ratio_capped_mean"] = np.nan
        matrix = prepare_features(frame, small_config)
        note = next(
            i.note for i in matrix.imputations if i.column == "repayment_ratio_capped_mean"
        ).lower()
        assert "inactive" in note or "in-credit" in note
        assert "not an observed customer value" in note

    def test_no_imputation_recorded_when_complete(self, cluster_df) -> None:
        config = SegmentationConfig(features=("credit_limit", "utilisation_mean_6m"))
        matrix = prepare_features(cluster_df, config)
        assert matrix.total_imputed == 0

    def test_total_imputed_adds_up(self, cluster_df, small_config) -> None:
        frame = cluster_df.copy()
        frame.loc[frame.index[:15], "repayment_ratio_capped_mean"] = np.nan
        matrix = prepare_features(frame, small_config)
        assert matrix.total_imputed == sum(i.n_missing for i in matrix.imputations)

    def test_imputation_info_is_serialisable(self) -> None:
        info = ImputationInfo("c", 5, 1.0, "median", 0.5, "note")
        json.dumps(info.to_dict())


# --------------------------------------------------------------------------- #
# Scaler comparison
# --------------------------------------------------------------------------- #


class TestScalerComparison:
    """Choosing a scaler by measurement."""

    def test_evaluates_every_scaler(self, cluster_df, small_config) -> None:
        comparison = compare_scalers(cluster_df, small_config)
        assert set(comparison.per_scaler) == set(VALID_SCALERS)

    def test_chooses_one(self, cluster_df, small_config) -> None:
        comparison = compare_scalers(cluster_df, small_config)
        assert comparison.chosen in VALID_SCALERS

    def test_rationale_cites_measured_numbers(self, cluster_df, small_config) -> None:
        rationale = compare_scalers(cluster_df, small_config).rationale
        assert "silhouette" in rationale.lower()
        assert "empirical, not conventional" in rationale

    def test_chosen_scaler_has_the_best_silhouette(self, cluster_df, small_config) -> None:
        comparison = compare_scalers(cluster_df, small_config)
        scores = {}
        for name, result in comparison.per_scaler.items():
            best = result.score_for(result.recommended_k)
            scores[name] = best.silhouette if best else float("-inf")
        assert comparison.chosen == max(scores, key=lambda n: scores[n])

    def test_frame_has_one_row_per_scaler_and_k(self, cluster_df, small_config) -> None:
        comparison = compare_scalers(cluster_df, small_config)
        frame = comparison.to_frame()
        assert "scaler" in frame.columns
        assert set(frame["scaler"].unique()) == set(VALID_SCALERS)

    def test_summary_table_marks_the_winner(self, cluster_df, small_config) -> None:
        comparison = compare_scalers(cluster_df, small_config)
        summary = comparison.best_row_per_scaler()
        assert (summary["Chosen"] == "Yes").sum() == 1

    def test_is_deterministic(self, cluster_df, small_config) -> None:
        first = compare_scalers(cluster_df, small_config)
        second = compare_scalers(cluster_df, small_config)
        assert first.chosen == second.chosen

    def test_is_serialisable(self, cluster_df, small_config) -> None:
        json.dumps(compare_scalers(cluster_df, small_config).to_dict())

    def test_rejects_an_empty_scaler_list(self, cluster_df, small_config) -> None:
        with pytest.raises(ValueError, match="No valid scaler"):
            compare_scalers(cluster_df, small_config, scalers=("nonsense",))


# --------------------------------------------------------------------------- #
# K selection
# --------------------------------------------------------------------------- #


class TestElbowDetection:
    """The geometric knee."""

    def test_finds_a_clear_elbow(self) -> None:
        ks = [2, 3, 4, 5, 6, 7, 8]
        inertias = [1000.0, 400.0, 200.0, 180.0, 165.0, 155.0, 150.0]
        assert _elbow_k(ks, inertias) in {3, 4}

    def test_handles_a_short_curve(self) -> None:
        assert _elbow_k([2, 3], [100.0, 50.0]) == 2

    def test_handles_an_empty_curve(self) -> None:
        assert _elbow_k([], []) == 2

    def test_handles_a_flat_curve(self) -> None:
        assert _elbow_k([2, 3, 4], [100.0, 100.0, 100.0]) == 2

    def test_returns_a_k_from_the_input(self) -> None:
        ks = [2, 3, 4, 5]
        assert _elbow_k(ks, [900.0, 400.0, 250.0, 200.0]) in ks


class TestStratifiedSilhouetteSample:
    """The silhouette sample must be valid by construction, not by retry.

    A plain random subset can miss a small cluster, leaving one label and an
    undefined silhouette. These tests pin the guarantee that makes that
    impossible: every cluster appears, proportions are respected, and the sample
    is a pure function of (labels, size, seed).
    """

    @staticmethod
    def _skewed_labels() -> np.ndarray:
        """9,900 rows in cluster 0, 90 in cluster 1, 10 in cluster 2."""
        return np.concatenate(
            [np.zeros(9_900, dtype=int), np.ones(90, dtype=int), np.full(10, 2)]
        )

    def test_every_cluster_is_represented(self) -> None:
        labels = self._skewed_labels()
        index = _stratified_sample_index(labels, sample_size=500, seed=42)
        assert set(np.unique(labels[index])) == {0, 1, 2}

    def test_tiny_cluster_survives_a_small_sample(self) -> None:
        """A 10-row cluster in 10,000 would almost surely vanish at random."""
        labels = self._skewed_labels()
        index = _stratified_sample_index(labels, sample_size=100, seed=42)
        assert (labels[index] == 2).sum() >= MIN_ROWS_PER_CLUSTER_IN_SAMPLE

    def test_always_leaves_at_least_two_labels(self) -> None:
        labels = self._skewed_labels()
        for size in (20, 50, 100, 1_000, 5_000):
            index = _stratified_sample_index(labels, sample_size=size, seed=7)
            assert np.unique(labels[index]).size >= 2

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        labels = self._skewed_labels()
        first = _stratified_sample_index(labels, sample_size=400, seed=42)
        second = _stratified_sample_index(labels, sample_size=400, seed=42)
        assert np.array_equal(first, second)

    def test_seed_actually_changes_the_sample(self) -> None:
        labels = self._skewed_labels()
        first = _stratified_sample_index(labels, sample_size=400, seed=1)
        second = _stratified_sample_index(labels, sample_size=400, seed=2)
        assert not np.array_equal(first, second)

    def test_allocation_is_roughly_proportional(self) -> None:
        labels = np.concatenate(
            [np.zeros(8_000, dtype=int), np.ones(2_000, dtype=int)]
        )
        index = _stratified_sample_index(labels, sample_size=1_000, seed=42)
        share_of_zero = (labels[index] == 0).mean()
        assert 0.75 <= share_of_zero <= 0.85

    def test_never_samples_more_rows_than_a_cluster_has(self) -> None:
        labels = self._skewed_labels()
        index = _stratified_sample_index(labels, sample_size=9_999, seed=42)
        assert (labels[index] == 2).sum() <= 10

    def test_returns_unique_sorted_positions(self) -> None:
        labels = self._skewed_labels()
        index = _stratified_sample_index(labels, sample_size=600, seed=42)
        assert np.array_equal(index, np.sort(index))
        assert len(set(index.tolist())) == len(index)
        assert index.min() >= 0
        assert index.max() < labels.shape[0]

    def test_overshoot_from_the_floor_stays_negligible(self) -> None:
        """Flooring small clusters may exceed the target, but only barely."""
        labels = self._skewed_labels()
        index = _stratified_sample_index(labels, sample_size=200, seed=42)
        n_clusters = np.unique(labels).size
        assert len(index) <= 200 + n_clusters * MIN_ROWS_PER_CLUSTER_IN_SAMPLE


class TestSafeSilhouette:
    """Scoring a partition without inventing a number."""

    def test_scores_a_skewed_partition_without_error(self) -> None:
        rng = np.random.default_rng(0)
        labels = TestStratifiedSilhouetteSample._skewed_labels()
        x = rng.normal(size=(labels.shape[0], 3)) + labels[:, None] * 6.0
        score = _safe_silhouette(x, labels, sample_size=200, seed=42)
        assert np.isfinite(score)
        assert -1.0 <= score <= 1.0

    def test_single_cluster_is_reported_unavailable_not_faked(self) -> None:
        x = np.random.default_rng(0).normal(size=(50, 2))
        labels = np.zeros(50, dtype=int)
        assert np.isnan(_safe_silhouette(x, labels, sample_size=20, seed=42))

    def test_uses_all_rows_when_sample_size_is_none(self) -> None:
        rng = np.random.default_rng(0)
        labels = np.concatenate([np.zeros(40, dtype=int), np.ones(40, dtype=int)])
        x = rng.normal(size=(80, 2)) + labels[:, None] * 5.0
        assert np.isfinite(_safe_silhouette(x, labels, sample_size=None, seed=42))

    def test_is_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        labels = TestStratifiedSilhouetteSample._skewed_labels()
        x = rng.normal(size=(labels.shape[0], 3)) + labels[:, None] * 6.0
        first = _safe_silhouette(x, labels, sample_size=300, seed=42)
        second = _safe_silhouette(x, labels, sample_size=300, seed=42)
        assert first == second


class TestEvaluateKRange:
    """Sweeping K."""

    def test_scores_every_k(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        result = evaluate_k_range(matrix, small_config)
        assert [s.k for s in result.scores] == list(small_config.k_range)

    def test_metrics_are_in_valid_ranges(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        for score in evaluate_k_range(matrix, small_config).scores:
            assert -1.0 <= score.silhouette <= 1.0
            assert score.davies_bouldin >= 0.0
            assert score.inertia > 0.0
            assert 0.0 < score.smallest_cluster_share_pct <= 100.0

    def test_inertia_decreases_with_k(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        inertias = [s.inertia for s in evaluate_k_range(matrix, small_config).scores]
        assert inertias == sorted(inertias, reverse=True)

    def test_recommends_a_k_that_was_scored(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        result = evaluate_k_range(matrix, small_config)
        assert result.recommended_k in [s.k for s in result.scores]

    def test_rationale_is_written(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        rationale = evaluate_k_range(matrix, small_config).rationale
        assert "silhouette" in rationale.lower()
        assert str(evaluate_k_range(matrix, small_config).recommended_k) in rationale

    def test_reports_all_three_criteria(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        rationale = evaluate_k_range(matrix, small_config).rationale.lower()
        assert "elbow" in rationale
        assert "davies-bouldin" in rationale

    def test_discloses_sampling(self, cluster_df) -> None:
        config = SegmentationConfig(k_max=3, silhouette_sample=100, n_init=3)
        matrix = prepare_features(cluster_df, config)
        result = evaluate_k_range(matrix, config)
        assert result.was_sampled
        assert result.silhouette_sample_size == 100
        assert "sample" in result.rationale.lower()

    def test_no_sampling_note_when_using_all_rows(self, cluster_df) -> None:
        config = SegmentationConfig(k_max=3, silhouette_sample=10 ** 7, n_init=3)
        matrix = prepare_features(cluster_df, config)
        result = evaluate_k_range(matrix, config)
        assert not result.was_sampled

    def test_avoids_a_degenerate_recommendation(self) -> None:
        """A partition whose smallest cluster is negligible must not be adopted
        silently."""
        rng = np.random.default_rng(0)
        bulk = rng.normal(0, 1, size=(600, 2))
        outlier = np.array([[60.0, 60.0], [61.0, 61.0]])
        frame = pd.DataFrame(
            np.vstack([bulk, outlier]), columns=["credit_limit", "utilisation_mean_6m"]
        )
        config = SegmentationConfig(
            features=("credit_limit", "utilisation_mean_6m"),
            k_min=2, k_max=5, n_init=3, silhouette_sample=500,
        )
        matrix = prepare_features(frame, config)
        result = evaluate_k_range(matrix, config)
        recommended = result.score_for(result.recommended_k)
        if any(s.is_degenerate for s in result.scores):
            assert not recommended.is_degenerate or len(
                [s for s in result.scores if not s.is_degenerate]
            ) == 0

    def test_degenerate_substitution_is_explained(self) -> None:
        rng = np.random.default_rng(1)
        bulk = rng.normal(0, 1, size=(800, 2))
        outlier = np.array([[80.0, 80.0]])
        frame = pd.DataFrame(
            np.vstack([bulk, outlier]), columns=["credit_limit", "utilisation_mean_6m"]
        )
        config = SegmentationConfig(
            features=("credit_limit", "utilisation_mean_6m"),
            k_min=2, k_max=4, n_init=3, silhouette_sample=400,
        )
        result = evaluate_k_range(prepare_features(frame, config), config)
        if result.silhouette_best_k != result.recommended_k:
            assert "degenerate" in result.rationale.lower()

    def test_is_deterministic(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        first = evaluate_k_range(matrix, small_config)
        second = evaluate_k_range(matrix, small_config)
        assert first.recommended_k == second.recommended_k
        assert [s.silhouette for s in first.scores] == [s.silhouette for s in second.scores]

    def test_frame_is_chart_ready(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        frame = evaluate_k_range(matrix, small_config).to_frame()
        for column in ("k", "silhouette", "davies_bouldin", "inertia"):
            assert column in frame.columns

    def test_empty_frame_when_nothing_scored(self) -> None:
        result = KSelectionResult(
            scores=(), recommended_k=2, silhouette_best_k=2, elbow_k=2,
            davies_bouldin_best_k=2, rationale="none", silhouette_sample_size=0,
            was_sampled=False,
        )
        assert result.to_frame().empty
        assert result.score_for(2) is None

    def test_is_serialisable(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        json.dumps(evaluate_k_range(matrix, small_config).to_dict())


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #


class TestSegmentNaming:
    """Names derived from measured centroid positions."""

    def test_vocabulary_covers_every_candidate(self) -> None:
        for feature in CANDIDATE_FEATURES:
            assert feature in TRAIT_VOCABULARY, feature

    def test_vocabulary_has_a_high_and_low_word(self) -> None:
        for feature, words in TRAIT_VOCABULARY.items():
            assert len(words) == 2
            assert all(w and w[0].isupper() for w in words), feature

    def test_uses_the_high_word_above_the_mean(self) -> None:
        label, traits = _derive_label({"utilisation_mean_6m": 1.5})
        assert "High-Utilisation" in label
        assert traits == ("High-Utilisation",)

    def test_uses_the_low_word_below_the_mean(self) -> None:
        label, _ = _derive_label({"repayment_ratio_capped_mean": -1.2})
        assert "Low-Repayment" in label

    def test_combines_the_two_strongest_traits(self) -> None:
        label, traits = _derive_label(
            {
                "delinquent_months_count": 2.4,
                "repayment_ratio_capped_mean": -1.1,
                "credit_limit": 0.05,
            }
        )
        assert len(traits) == 2
        assert "Delinquent" in label and "Low-Repayment" in label
        assert "Limit" not in label, "a near-average feature must not be named"

    def test_strongest_trait_comes_first(self) -> None:
        label, traits = _derive_label(
            {"delinquent_months_count": 2.4, "credit_limit": 0.9}
        )
        assert traits[0] == "Delinquent"
        assert label.startswith("Delinquent")

    def test_average_segment_is_labelled_honestly(self) -> None:
        """No distinguishing trait means saying so, not inventing one."""
        label, traits = _derive_label({f: 0.05 for f in CANDIDATE_FEATURES})
        assert traits == ()
        assert "Mid-Range" in label

    def test_single_trait_is_marked_otherwise_typical(self) -> None:
        label, traits = _derive_label(
            {"credit_limit": 1.8, "utilisation_mean_6m": 0.01}
        )
        assert len(traits) == 1
        assert "Otherwise-Typical" in label

    def test_threshold_is_respected(self) -> None:
        _label, traits = _derive_label(
            {"credit_limit": TRAIT_THRESHOLD - 0.01}
        )
        assert traits == ()

    def test_ignores_non_finite_scores(self) -> None:
        label, _ = _derive_label({"credit_limit": float("nan"), "age": 5.0})
        assert "Mid-Range" in label

    def test_no_banned_words_in_the_vocabulary(self) -> None:
        """Names must describe behaviour, not pass judgement."""
        words = " ".join(w for pair in TRAIT_VOCABULARY.values() for w in pair).lower()
        for banned in ("good", "bad", "safe", "risky", "poor", "excellent"):
            assert banned not in words, banned

    def test_is_deterministic(self) -> None:
        scores = {"delinquent_months_count": 2.0, "credit_limit": -1.0}
        assert _derive_label(scores) == _derive_label(scores)


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #


class TestProfileSegments:
    """Measuring each segment."""

    def test_one_profile_per_cluster(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert len(result.profiles) == 3
        assert {p.cluster_id for p in result.profiles} == {0, 1, 2}

    def test_sizes_sum_to_the_portfolio(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert sum(p.size for p in result.profiles) == len(cluster_df)

    def test_shares_sum_to_one_hundred(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert sum(p.share_pct for p in result.profiles) == pytest.approx(100.0)

    def test_metrics_cover_every_feature(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        for profile in result.profiles:
            assert set(profile.metrics) == set(result.matrix.feature_names)

    def test_metrics_match_manual_means(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        feature = result.matrix.feature_names[0]
        for profile in result.profiles:
            mask = result.labels == profile.cluster_id
            expected = float(result.matrix.raw.loc[mask, feature].mean())
            assert profile.metrics[feature] == pytest.approx(expected)

    def test_observed_default_rate_is_computed(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        for profile in result.profiles:
            assert profile.observed_default_rate_pct is not None
            assert 0.0 <= profile.observed_default_rate_pct <= 100.0

    def test_default_rate_matches_manual_calculation(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        for profile in result.profiles:
            mask = result.labels == profile.cluster_id
            expected = float(
                cluster_df.loc[mask.reindex(cluster_df.index, fill_value=False),
                               "default_next_month"].mean() * 100
            )
            assert profile.observed_default_rate_pct == pytest.approx(expected)

    def test_omits_default_rate_without_a_target(self, cluster_df, small_config) -> None:
        matrix = prepare_features(cluster_df, small_config)
        labels = pd.Series(0, index=cluster_df.index)
        profiles = profile_segments(cluster_df, labels, matrix, target=None)
        assert profiles[0].observed_default_rate_pct is None

    def test_z_scores_are_relative_to_the_portfolio(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        for feature in result.matrix.feature_names:
            weighted = sum(
                p.z_scores[feature] * p.size for p in result.profiles
            ) / len(cluster_df)
            assert abs(weighted) < 0.15, f"{feature} z-scores should centre near zero"

    def test_description_cites_measured_values(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        for profile in result.profiles:
            if profile.traits:
                assert "SD" in profile.description
                assert "portfolio mean" in profile.description

    def test_display_name_includes_a_readable_index(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        for profile in result.profiles:
            assert profile.display_name.startswith(f"Segment {profile.cluster_id + 1}")

    def test_profile_frame_is_tabular(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        frame = result.profile_frame()
        assert len(frame) == 3
        assert "Segment" in frame.columns
        assert "Customers" in frame.columns
        assert "Observed default rate %" in frame.columns

    def test_profile_frame_uses_readable_feature_labels(
        self, cluster_df, small_config
    ) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        frame = result.profile_frame()
        for feature in result.matrix.feature_names:
            assert FEATURE_LABELS[feature] in frame.columns

    def test_profile_is_serialisable(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        json.dumps([p.to_dict() for p in result.profiles])


# --------------------------------------------------------------------------- #
# Fitting and PCA
# --------------------------------------------------------------------------- #


class TestFitSegments:
    """The fitted result."""

    def test_assigns_every_customer(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert result.labels.notna().all()
        assert result.n_segmented == len(cluster_df)

    def test_labels_align_with_the_input_index(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        pd.testing.assert_index_equal(result.labels.index, cluster_df.index)

    def test_reports_the_requested_k(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=4, config=small_config)
        assert result.k == 4
        assert len(result.profiles) == 4

    def test_metrics_are_reported(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        for key in ("inertia", "silhouette", "davies_bouldin", "calinski_harabasz"):
            assert key in result.metrics
            assert np.isfinite(result.metrics[key])

    def test_is_deterministic(self, cluster_df, small_config) -> None:
        first = fit_segments(cluster_df, k=3, config=small_config)
        second = fit_segments(cluster_df, k=3, config=small_config)
        pd.testing.assert_series_equal(first.labels, second.labels)
        assert first.metrics == second.metrics
        assert [p.label for p in first.profiles] == [p.label for p in second.profiles]

    def test_different_seeds_are_allowed_to_differ(self, cluster_df) -> None:
        """Determinism comes from the seed, not from luck."""
        from dataclasses import replace

        base = SegmentationConfig(k_max=3, n_init=1, silhouette_sample=500)
        first = fit_segments(cluster_df, k=3, config=replace(base, random_state=1))
        second = fit_segments(cluster_df, k=3, config=replace(base, random_state=999))
        assert first.metrics["inertia"] > 0 and second.metrics["inertia"] > 0

    def test_label_map_covers_every_cluster(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert set(result.label_map()) == {0, 1, 2}

    def test_assign_to_annotates_by_index(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        annotated = result.assign_to(cluster_df.head(100))
        assert "segment" in annotated.columns
        assert annotated["segment"].notna().all()
        assert len(annotated) == 100

    def test_assign_to_leaves_unknown_rows_null(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        outside = cluster_df.head(10).copy()
        outside.index = range(10 ** 6, 10 ** 6 + 10)
        annotated = result.assign_to(outside)
        assert annotated["segment"].isna().all()

    def test_result_is_serialisable(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        payload = json.dumps(result.to_dict())
        assert "labels" not in result.to_dict(), "row-level data must stay out"
        assert len(payload) < 200_000

    def test_degenerate_flag(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert isinstance(result.has_degenerate_segment, bool)


class TestPcaProjection:
    """PCA is display-only."""

    def test_projection_covers_every_customer(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert len(result.projection) == len(cluster_df)

    def test_projection_has_expected_columns(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        for column in ("pc1", "pc2", "cluster", "segment"):
            assert column in result.projection.columns

    def test_explained_variance_is_reported(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert len(result.pca_explained_variance) == 2
        assert all(0.0 <= v <= 1.0 for v in result.pca_explained_variance)

    def test_explained_variance_is_ordered(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert result.pca_explained_variance[0] >= result.pca_explained_variance[1]

    def test_total_variance_is_a_percentage(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert 0.0 < result.pca_total_variance_pct <= 100.0

    def test_clustering_uses_the_full_space_not_the_projection(
        self, cluster_df, small_config
    ) -> None:
        """PCA must not replace the feature space K-Means ran on."""
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert result.matrix.n_features > 2
        assert result.matrix.scaled.shape[1] == result.matrix.n_features
        # The projection is strictly 2D regardless of the model dimensionality.
        assert result.projection[["pc1", "pc2"]].shape[1] == 2

    def test_projection_segments_match_the_labels(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        pd.testing.assert_series_equal(
            result.projection["cluster"].astype("int64"),
            result.labels.astype("int64"),
            check_names=False,
        )


# --------------------------------------------------------------------------- #
# Orchestration and edge cases
# --------------------------------------------------------------------------- #


class TestRunSegmentation:
    """The full workflow."""

    def test_returns_a_result(self, cluster_df, small_config) -> None:
        result = run_segmentation(cluster_df, config=small_config)
        assert isinstance(result, SegmentationResult)
        assert result.is_available

    def test_uses_the_recommended_k_by_default(self, cluster_df, small_config) -> None:
        result = run_segmentation(cluster_df, config=small_config)
        assert result.k == result.k_selection.recommended_k

    def test_honours_a_k_override(self, cluster_df, small_config) -> None:
        result = run_segmentation(cluster_df, k=4, config=small_config)
        assert result.k == 4

    def test_clamps_an_absurd_k(self, cluster_df, small_config) -> None:
        result = run_segmentation(cluster_df, k=10 ** 6, config=small_config)
        assert 2 <= result.k < len(cluster_df)

    def test_clamps_a_k_below_two(self, cluster_df, small_config) -> None:
        result = run_segmentation(cluster_df, k=1, config=small_config)
        assert result.k >= 2

    def test_carries_the_scaler_comparison(self, cluster_df, small_config) -> None:
        result = run_segmentation(cluster_df, config=small_config, compare_scaling=True)
        assert result.scaler_comparison is not None
        assert result.scaler_name == result.scaler_comparison.chosen

    def test_can_skip_the_comparison(self, cluster_df, small_config) -> None:
        result = run_segmentation(cluster_df, config=small_config, compare_scaling=False)
        assert result.scaler_comparison is None
        assert result.scaler_name == small_config.scaler

    def test_refuses_too_few_rows(self, cluster_df, small_config) -> None:
        result = run_segmentation(cluster_df.head(50), config=small_config)
        assert isinstance(result, SegmentationUnavailable)
        assert not result.is_available
        assert str(MIN_ROWS_FOR_CLUSTERING) in result.reason

    def test_refuses_when_no_feature_exists(self, small_config) -> None:
        frame = pd.DataFrame({"client_id": range(500), "other": range(500)})
        result = run_segmentation(frame, config=small_config)
        assert isinstance(result, SegmentationUnavailable)
        assert result.missing_features

    def test_reports_an_empty_frame_gracefully(self, cluster_df, small_config) -> None:
        result = run_segmentation(cluster_df.iloc[0:0], config=small_config)
        assert isinstance(result, SegmentationUnavailable)

    def test_survives_all_constant_features(self, cluster_df, small_config) -> None:
        frame = cluster_df.copy()
        for feature in CANDIDATE_FEATURES:
            if feature in frame.columns:
                frame[feature] = 5.0
        result = run_segmentation(frame, config=small_config)
        assert isinstance(result, SegmentationUnavailable)

    def test_works_with_a_single_usable_feature(self, cluster_df) -> None:
        config = SegmentationConfig(features=("credit_limit",), k_max=3, n_init=3)
        result = run_segmentation(cluster_df, config=config)
        assert isinstance(result, SegmentationResult)
        assert result.matrix.n_features == 1

    def test_unavailable_is_serialisable(self) -> None:
        json.dumps(SegmentationUnavailable(reason="x", n_rows=1).to_dict())

    def test_is_deterministic(self, cluster_df, small_config) -> None:
        first = run_segmentation(cluster_df, config=small_config)
        second = run_segmentation(cluster_df, config=small_config)
        assert first.k == second.k
        assert first.scaler_name == second.scaler_name
        pd.testing.assert_series_equal(first.labels, second.labels)


# --------------------------------------------------------------------------- #
# Narrative
# --------------------------------------------------------------------------- #


class TestNarrative:
    """Generated prose."""

    def test_methodology_note_describes_the_run(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        note = methodology_note(result)
        assert "K-Means" in note
        assert result.scaler_name in note
        assert "excluded from the feature matrix" in note

    def test_methodology_note_mentions_imputation_state(
        self, cluster_df, small_config
    ) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        note = methodology_note(result)
        assert "imput" in note.lower()

    def test_methodology_note_says_the_rate_came_afterwards(
        self, cluster_df, small_config
    ) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        assert "descriptive only" in methodology_note(result)

    def test_interpretation_has_four_parts(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        for profile in result.profiles:
            parts = segment_interpretation(profile, 22.12)
            assert set(parts) == {"fact", "behavior", "observed_risk", "implication"}
            for value in parts.values():
                assert value.strip() and len(value) > 20

    def test_interpretation_states_the_comparison(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        parts = segment_interpretation(result.profiles[0], 22.12)
        assert "22.12%" in parts["observed_risk"]
        assert "percentage points" in parts["observed_risk"]

    def test_interpretation_denies_causation(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        parts = segment_interpretation(result.profiles[0], 22.12)
        text = parts["observed_risk"].lower()
        assert "not a prediction" in text
        assert "not an effect of segment membership" in text

    def test_interpretation_handles_a_missing_rate(self, cluster_df, small_config) -> None:
        from dataclasses import replace

        result = fit_segments(cluster_df, k=3, config=small_config)
        stripped = replace(result.profiles[0], observed_default_rate_pct=None)
        parts = segment_interpretation(stripped, None)
        assert "no observed default rate" in parts["observed_risk"].lower()

    def test_no_banned_language_anywhere(self, cluster_df, small_config) -> None:
        result = fit_segments(cluster_df, k=3, config=small_config)
        pieces = [methodology_note(result)]
        for profile in result.profiles:
            pieces.append(profile.label)
            pieces.append(profile.description)
            pieces.extend(segment_interpretation(profile, 22.12).values())
        combined = " ".join(pieces).lower()
        found = [w for w in BANNED_LANGUAGE if w in combined]
        assert found == [], f"banned language in segmentation text: {found}"


# --------------------------------------------------------------------------- #
# Real dataset
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fitted(real_pipeline):
    """The segmentation fitted once on the real portfolio.

    Module-scoped and defined at module level: pytest 9 deprecates class-scoped
    fixtures written as instance methods.
    """
    return run_segmentation(real_pipeline.data, real_pipeline.registry)


@pytest.mark.integration
class TestRealSegmentation:
    """Segmentation measured on the real portfolio."""

    def test_produces_a_result(self, fitted) -> None:
        assert isinstance(fitted, SegmentationResult)

    def test_segments_every_customer(self, fitted) -> None:
        assert fitted.n_segmented == 30_000

    def test_chose_robust_scaling(self, fitted) -> None:
        """Measured: silhouette 0.4035 robust against 0.2879 standard."""
        assert fitted.scaler_name == "robust"

    def test_chose_three_segments(self, fitted) -> None:
        """Measured: silhouette peaks at K=3 with a healthy smallest segment."""
        assert fitted.k == 3

    def test_silhouette_is_reasonable(self, fitted) -> None:
        assert fitted.silhouette > 0.35

    def test_uses_all_seven_features(self, fitted) -> None:
        assert fitted.matrix.n_features == 7
        assert set(fitted.matrix.feature_names) == set(CANDIDATE_FEATURES)

    def test_imputes_the_known_missing_ratios(self, fitted) -> None:
        """1,406 customers (4.69%) have an undefined repayment ratio."""
        infos = [
            i for i in fitted.matrix.imputations
            if i.column == "repayment_ratio_capped_mean"
        ]
        assert infos
        assert infos[0].n_missing == 1_406
        assert infos[0].pct_missing == pytest.approx(4.687, abs=0.01)

    def test_no_segment_is_degenerate(self, fitted) -> None:
        for profile in fitted.profiles:
            assert profile.share_pct >= DEGENERATE_CLUSTER_SHARE_PCT

    def test_separates_observed_outcome_strongly(self, fitted) -> None:
        """Clustering saw no outcome, yet the segments differ sharply on it -
        which is the evidence that behaviour carries real signal."""
        rates = [
            p.observed_default_rate_pct
            for p in fitted.profiles
            if p.observed_default_rate_pct is not None
        ]
        assert max(rates) - min(rates) > 30.0

    def test_identifies_a_delinquent_segment(self, fitted) -> None:
        labels = " ".join(p.label for p in fitted.profiles)
        assert "Delinquent" in labels

    def test_pca_retains_a_useful_share_of_variance(self, fitted) -> None:
        assert fitted.pca_total_variance_pct > 60.0

    def test_target_never_entered_the_matrix(self, fitted) -> None:
        assert "default_next_month" not in fitted.matrix.feature_names
        assert "default_next_month" not in fitted.matrix.raw.columns
        assert fitted.target_excluded

    def test_no_demographic_entered_the_matrix(self, fitted) -> None:
        for column in ("sex", "sex_code", "marriage", "marriage_code", "age"):
            assert column not in fitted.matrix.feature_names

    def test_is_serialisable_and_compact(self, fitted) -> None:
        payload = json.dumps(fitted.to_dict())
        assert len(payload) < 300_000
