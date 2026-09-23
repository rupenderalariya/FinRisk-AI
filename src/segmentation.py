"""
Behaviour-based customer segmentation with K-Means.

What this module is, and is not
-------------------------------
It is an **unsupervised description** of how customers behave. It groups
customers by measured credit behaviour and then reports, separately, what
outcome each group happened to experience.

It is **not** a prediction model. Clustering has no target, produces no
probability, and makes no forecast. The observed default rate attached to each
segment is computed *after* clustering, purely for profiling, and never
influences how the clusters form.

Leakage prevention
------------------
Two categories of column are hard-blocked from the feature matrix by
:data:`FORBIDDEN_FEATURES`, and :func:`prepare_features` raises if one is
requested:

* **The target and anything derived from it.** Including the outcome would make
  the segments a restatement of the answer rather than a description of behaviour.
* **Demographic attributes** (sex, marital status, education, age). Segmenting a
  credit portfolio on protected attributes is discriminatory, and this project
  excludes them from every modelling path by default.

Methodological choices, all resolved by measurement
---------------------------------------------------
* **Scaler**: K-Means is distance-based, so scaling decides the outcome. Standard
  and robust scaling are compared on real metrics by :func:`compare_scalers`
  rather than one being assumed.
* **K**: swept across a range, with silhouette as the primary criterion supported
  by the inertia elbow and Davies-Bouldin. The measured result is reported even
  when it is inconvenient.
* **Silhouette sampling**: silhouette is O(n^2), so on large frames it is computed
  on a fixed-seed sample and every result records whether it was sampled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import settings
from src.logging_setup import get_logger
from src.schema import TARGET_COLUMN, ColumnRegistry, Role, build_registry

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Feature policy
# --------------------------------------------------------------------------- #

#: Candidate behavioural features. Each is bounded or money-denominated, and none
#: derives from the outcome.
CANDIDATE_FEATURES: Final[tuple[str, ...]] = (
    "credit_limit",
    "utilisation_mean_6m",
    "repayment_ratio_capped_mean",
    "delinquent_months_count",
    "bill_trend_slope",
    "utilisation_volatility",
    "months_zero_payment",
)

#: Columns that must never reach the feature matrix.
#:
#: The target group would leak the answer into the clustering. The demographic
#: group is excluded on fairness grounds, consistent with the rest of the project.
FORBIDDEN_FEATURES: Final[frozenset[str]] = frozenset(
    {
        # --- target and target-derived ---
        TARGET_COLUMN,
        "default_next_month",
        "default payment next month",
        "default.payment.next.month",
        # --- protected / demographic attributes ---
        "sex",
        "sex_code",
        "marriage",
        "marriage_code",
        "education",
        "education_code",
        "age",
        "age_band",
        # --- identifiers and bookkeeping ---
        "client_id",
        "is_duplicate_excluding_id",
        "has_extreme_value",
    }
)

#: Human-readable names for the features, used in profile tables and charts.
FEATURE_LABELS: Final[Mapping[str, str]] = {
    "credit_limit": "Avg credit limit",
    "utilisation_mean_6m": "Avg utilisation",
    "repayment_ratio_capped_mean": "Avg repayment ratio",
    "delinquent_months_count": "Avg delinquent months",
    "bill_trend_slope": "Avg balance trend",
    "utilisation_volatility": "Avg utilisation volatility",
    "months_zero_payment": "Avg zero-payment months",
}

#: How each feature should be formatted for display.
FEATURE_FORMATS: Final[Mapping[str, str]] = {
    "credit_limit": "currency",
    "utilisation_mean_6m": "percent",
    "repayment_ratio_capped_mean": "percent",
    "delinquent_months_count": "decimal",
    "bill_trend_slope": "currency_signed",
    "utilisation_volatility": "decimal3",
    "months_zero_payment": "decimal",
}

#: Trait wording for a feature sitting notably above / below the portfolio mean.
#: Segment names are composed from these, so every word in a label traces back to
#: a measured centroid position.
TRAIT_VOCABULARY: Final[Mapping[str, tuple[str, str]]] = {
    "credit_limit": ("High-Limit", "Low-Limit"),
    "utilisation_mean_6m": ("High-Utilisation", "Low-Utilisation"),
    "repayment_ratio_capped_mean": ("High-Repayment", "Low-Repayment"),
    "delinquent_months_count": ("Delinquent", "Consistently-Current"),
    "bill_trend_slope": ("Growing-Balance", "Shrinking-Balance"),
    "utilisation_volatility": ("Volatile-Usage", "Steady-Usage"),
    "months_zero_payment": ("Frequent-Non-Payment", "Regular-Payment"),
}

#: Absolute standardised distance from the portfolio mean before a feature counts
#: as a distinguishing trait of a segment.
TRAIT_THRESHOLD: Final[float] = 0.35

#: The reduced feature set used as the comparison baseline, named explicitly so the
#: figures quoted in :data:`FEATURE_SET_RATIONALE` can be reproduced. It drops the
#: two trajectory features - balance slope and utilisation volatility - which is
#: the most defensible reduction, since those are the only derived-shape measures
#: among the seven.
COMPARISON_FEATURE_SUBSET: Final[tuple[str, ...]] = (
    "credit_limit",
    "utilisation_mean_6m",
    "repayment_ratio_capped_mean",
    "delinquent_months_count",
    "months_zero_payment",
)

#: Why all seven candidate features are retained.
#:
#: This decision is deliberately **not** made on silhouette score. Silhouette
#: rises mechanically as dimensionality falls, because distances become more
#: uniform in higher-dimensional space, so it is only comparable across values of
#: K *within a fixed feature space* - never across feature spaces of different
#: size. Measured on the real portfolio with robust scaling, the five-feature
#: subset in :data:`COMPARISON_FEATURE_SUBSET` scored 0.5672 at its own recommended
#: K=2, against 0.4035 for all seven at K=3 - yet it concentrated 87.5% of the book
#: in a single segment. The higher score describes a coarser partition, not a more
#: useful one, which is exactly why the metric is not used to choose here.
FEATURE_SET_RATIONALE: Final[str] = (
    "All seven candidate features are retained. No pair is redundant: the "
    "strongest Spearman correlation between any two is 0.756 (average utilisation "
    "against average repayment ratio), below the 0.80 threshold at which one would "
    "be dropped, and the two measure genuinely different things - how much credit "
    "is drawn versus how much of it is repaid. Each remaining feature covers a "
    "distinct behavioural facet: granted capacity, sustained usage, repayment "
    "share, delinquency frequency, balance trajectory, usage stability and "
    "non-payment frequency. Feature sets were also compared on partition balance. "
    "Dropping the two trajectory features leaves five, which scores a higher "
    "silhouette (0.5672 at its own recommended K=2, against 0.4035 for seven at "
    "K=3) but collapses the portfolio into a coarser split, with its largest "
    "segment holding 87.5% of customers against 72.6% for the seven-feature set. "
    "Silhouette was deliberately not used to choose between feature sets, because "
    "the metric improves as dimensionality falls and is therefore not comparable "
    "across feature spaces of different size."
)

#: Supported scalers.
VALID_SCALERS: Final[tuple[str, ...]] = ("standard", "robust")

#: Minimum rows before clustering is attempted at all.
MIN_ROWS_FOR_CLUSTERING: Final[int] = 200

#: A partition where the smallest cluster falls below this share is flagged as
#: degenerate: it usually means K-Means isolated a handful of outliers.
DEGENERATE_CLUSTER_SHARE_PCT: Final[float] = 1.0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SegmentationConfig:
    """Settings for one segmentation run."""

    features: tuple[str, ...] = CANDIDATE_FEATURES
    k_min: int = 2
    k_max: int = 8
    scaler: str = "robust"
    random_state: int | None = None
    silhouette_sample: int = 5_000
    n_init: int = 10
    max_iter: int = 300
    pca_components: int = 2

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On an unknown scaler, an invalid K range, or a requested
                feature that is forbidden.
        """
        if self.scaler not in VALID_SCALERS:
            raise ValueError(
                f"Unknown scaler '{self.scaler}'. Expected one of {VALID_SCALERS}."
            )
        if self.k_min < 2:
            raise ValueError("k_min must be at least 2; a single cluster is not a partition.")
        if self.k_max < self.k_min:
            raise ValueError(f"k_max ({self.k_max}) must be >= k_min ({self.k_min}).")

        forbidden = sorted(set(self.features) & FORBIDDEN_FEATURES)
        if forbidden:
            raise ValueError(
                f"Forbidden column(s) requested as clustering features: {forbidden}. "
                "The target and demographic attributes must never enter the feature "
                "matrix - the target would leak the outcome into the segments, and "
                "demographic attributes are excluded on fairness grounds."
            )

    @property
    def seed(self) -> int:
        """Resolved random seed."""
        return self.random_state if self.random_state is not None else settings.random_state

    @property
    def k_range(self) -> tuple[int, ...]:
        """The K values to evaluate."""
        return tuple(range(self.k_min, self.k_max + 1))


# --------------------------------------------------------------------------- #
# Result structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImputationInfo:
    """Record of how missing values in one feature were filled."""

    column: str
    n_missing: int
    pct_missing: float
    method: str
    fill_value: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "column": self.column,
            "n_missing": self.n_missing,
            "pct_missing": round(self.pct_missing, 4),
            "method": self.method,
            "fill_value": self.fill_value,
            "note": self.note,
        }


@dataclass(frozen=True)
class FeatureMatrix:
    """The prepared clustering input."""

    scaled: np.ndarray
    raw: pd.DataFrame
    feature_names: tuple[str, ...]
    scaler_name: str
    imputations: tuple[ImputationInfo, ...] = field(default=())
    excluded: Mapping[str, str] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        """Number of customers in the matrix."""
        return int(self.scaled.shape[0])

    @property
    def n_features(self) -> int:
        """Number of features used."""
        return int(self.scaled.shape[1])

    @property
    def index(self) -> pd.Index:
        """Index of the source rows, for aligning labels back."""
        return self.raw.index

    @property
    def total_imputed(self) -> int:
        """Total cells filled across all features."""
        return sum(info.n_missing for info in self.imputations)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary."""
        return {
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names),
            "scaler": self.scaler_name,
            "imputations": [info.to_dict() for info in self.imputations],
            "excluded": dict(self.excluded),
        }


@dataclass(frozen=True)
class KScore:
    """Clustering quality metrics for one value of K."""

    k: int
    inertia: float
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    smallest_cluster_share_pct: float

    @property
    def is_degenerate(self) -> bool:
        """True when one cluster is so small the partition is not useful."""
        return self.smallest_cluster_share_pct < DEGENERATE_CLUSTER_SHARE_PCT

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "k": self.k,
            "inertia": round(self.inertia, 3),
            "silhouette": round(self.silhouette, 5),
            "davies_bouldin": round(self.davies_bouldin, 5),
            "calinski_harabasz": round(self.calinski_harabasz, 3),
            "smallest_cluster_share_pct": round(self.smallest_cluster_share_pct, 3),
            "is_degenerate": self.is_degenerate,
        }


@dataclass(frozen=True)
class KSelectionResult:
    """Outcome of sweeping K across a range."""

    scores: tuple[KScore, ...]
    recommended_k: int
    silhouette_best_k: int
    elbow_k: int
    davies_bouldin_best_k: int
    rationale: str
    silhouette_sample_size: int
    was_sampled: bool
    scaler_name: str = ""

    def score_for(self, k: int) -> KScore | None:
        """Look up the metrics for one K."""
        return next((s for s in self.scores if s.k == k), None)

    def to_frame(self) -> pd.DataFrame:
        """Chart-ready table of metrics by K."""
        if not self.scores:
            return pd.DataFrame(
                columns=[
                    "k", "inertia", "silhouette", "davies_bouldin",
                    "calinski_harabasz", "smallest_cluster_share_pct", "is_degenerate",
                ]
            )
        return pd.DataFrame([s.to_dict() for s in self.scores])

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "scores": [s.to_dict() for s in self.scores],
            "recommended_k": self.recommended_k,
            "silhouette_best_k": self.silhouette_best_k,
            "elbow_k": self.elbow_k,
            "davies_bouldin_best_k": self.davies_bouldin_best_k,
            "rationale": self.rationale,
            "silhouette_sample_size": self.silhouette_sample_size,
            "was_sampled": self.was_sampled,
            "scaler": self.scaler_name,
        }


@dataclass(frozen=True)
class ScalerComparison:
    """Measured comparison of scaling strategies."""

    per_scaler: Mapping[str, KSelectionResult]
    chosen: str
    rationale: str

    def to_frame(self) -> pd.DataFrame:
        """One row per scaler and K, for display."""
        frames: list[pd.DataFrame] = []
        for name, result in self.per_scaler.items():
            frame = result.to_frame()
            if frame.empty:
                continue
            frame = frame.copy()
            frame.insert(0, "scaler", name)
            frames.append(frame)
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["scaler", "k", "silhouette"])
        )

    def best_row_per_scaler(self) -> pd.DataFrame:
        """Each scaler's best K by silhouette, side by side."""
        rows: list[dict[str, Any]] = []
        for name, result in self.per_scaler.items():
            best = result.score_for(result.silhouette_best_k)
            if best is None:
                continue
            rows.append(
                {
                    "Scaler": name,
                    "Best K (silhouette)": best.k,
                    "Silhouette": round(best.silhouette, 4),
                    "Davies-Bouldin": round(best.davies_bouldin, 4),
                    "Inertia": round(best.inertia, 1),
                    "Smallest cluster %": round(best.smallest_cluster_share_pct, 2),
                    "Chosen": "Yes" if name == self.chosen else "",
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "chosen": self.chosen,
            "rationale": self.rationale,
            "per_scaler": {name: r.to_dict() for name, r in self.per_scaler.items()},
        }


@dataclass(frozen=True)
class SegmentProfile:
    """Measured profile of one behavioural segment."""

    cluster_id: int
    label: str
    description: str
    traits: tuple[str, ...]
    size: int
    share_pct: float
    metrics: Mapping[str, float]
    z_scores: Mapping[str, float]
    observed_default_rate_pct: float | None = None

    @property
    def display_name(self) -> str:
        """Name shown in the UI."""
        return f"Segment {self.cluster_id + 1} · {self.label}"

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "display_name": self.display_name,
            "description": self.description,
            "traits": list(self.traits),
            "size": self.size,
            "share_pct": round(self.share_pct, 4),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "z_scores": {k: round(float(v), 4) for k, v in self.z_scores.items()},
            "observed_default_rate_pct": (
                round(self.observed_default_rate_pct, 4)
                if self.observed_default_rate_pct is not None
                else None
            ),
        }


@dataclass(frozen=True)
class SegmentationResult:
    """Everything one fitted segmentation produces."""

    labels: pd.Series
    profiles: tuple[SegmentProfile, ...]
    k: int
    k_selection: KSelectionResult
    matrix: FeatureMatrix
    projection: pd.DataFrame
    pca_explained_variance: tuple[float, ...]
    metrics: Mapping[str, float]
    scaler_name: str
    scaler_comparison: ScalerComparison | None = None
    target_excluded: bool = True

    # ------------------------------------------------------------------ #

    @property
    def is_available(self) -> bool:
        """Always True. Lets callers branch without an isinstance check."""
        return True

    @property
    def n_segmented(self) -> int:
        """Number of customers assigned to a segment."""
        return int(self.labels.notna().sum())

    @property
    def pca_total_variance_pct(self) -> float:
        """Share of variance the 2D projection captures, as a percentage."""
        return float(sum(self.pca_explained_variance) * 100.0)

    @property
    def silhouette(self) -> float:
        """Silhouette score of the fitted partition."""
        return float(self.metrics.get("silhouette", float("nan")))

    @property
    def has_degenerate_segment(self) -> bool:
        """True when any segment holds less than the minimum useful share."""
        return any(p.share_pct < DEGENERATE_CLUSTER_SHARE_PCT for p in self.profiles)

    def profile_for(self, cluster_id: int) -> SegmentProfile | None:
        """Look up one segment's profile."""
        return next((p for p in self.profiles if p.cluster_id == cluster_id), None)

    def label_map(self) -> dict[int, str]:
        """Cluster id -> display name."""
        return {p.cluster_id: p.display_name for p in self.profiles}

    def profile_frame(self) -> pd.DataFrame:
        """Profile table with one row per segment, ordered by size."""
        rows: list[dict[str, Any]] = []
        for profile in self.profiles:
            row: dict[str, Any] = {
                "Segment": profile.display_name,
                "Customers": profile.size,
                "% of portfolio": round(profile.share_pct, 2),
            }
            for feature in self.matrix.feature_names:
                row[FEATURE_LABELS.get(feature, feature)] = profile.metrics.get(feature)
            if profile.observed_default_rate_pct is not None:
                row["Observed default rate %"] = round(profile.observed_default_rate_pct, 2)
            rows.append(row)
        frame = pd.DataFrame(rows)
        return frame.sort_values("Customers", ascending=False).reset_index(drop=True)

    def assign_to(self, df: pd.DataFrame, column: str = "segment") -> pd.DataFrame:
        """Attach segment labels to a dataframe by index.

        Used to describe how a filtered selection distributes across the segments
        fitted on the whole portfolio.

        Args:
            df: Frame to annotate. Only its index is used to align.
            column: Name of the column to add.

        Returns:
            A copy of ``df`` with the segment display name added as an ordered
            categorical. Rows absent from the fitted model get null.
        """
        result = df.copy()
        names = self.label_map()
        aligned = self.labels.reindex(result.index)
        ordered = [p.display_name for p in sorted(self.profiles, key=lambda p: p.cluster_id)]
        result[column] = pd.Categorical(
            aligned.map(names), categories=ordered, ordered=True
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialisable payload. Excludes row-level assignments."""
        return {
            "k": self.k,
            "n_segmented": self.n_segmented,
            "scaler": self.scaler_name,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "pca_explained_variance": [round(v, 5) for v in self.pca_explained_variance],
            "pca_total_variance_pct": round(self.pca_total_variance_pct, 3),
            "profiles": [p.to_dict() for p in self.profiles],
            "k_selection": self.k_selection.to_dict(),
            "matrix": self.matrix.to_dict(),
            "scaler_comparison": (
                self.scaler_comparison.to_dict() if self.scaler_comparison else None
            ),
            "target_excluded": self.target_excluded,
            "has_degenerate_segment": self.has_degenerate_segment,
        }


@dataclass(frozen=True)
class SegmentationUnavailable:
    """Returned when segmentation cannot be performed."""

    reason: str
    n_rows: int = 0
    missing_features: tuple[str, ...] = field(default=())

    @property
    def is_available(self) -> bool:
        """Always False."""
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "available": False,
            "reason": self.reason,
            "n_rows": self.n_rows,
            "missing_features": list(self.missing_features),
        }


# --------------------------------------------------------------------------- #
# Feature assessment
# --------------------------------------------------------------------------- #


def assess_features(
    df: pd.DataFrame, features: Sequence[str] = CANDIDATE_FEATURES
) -> pd.DataFrame:
    """Describe candidate features so exclusions can be justified with evidence.

    Reports missingness, spread, skew and the strongest correlation with another
    candidate. A near-duplicate pair shows up as a very high correlation, which is
    the evidence needed to drop one rather than dropping it on instinct.

    Args:
        df: Analysis-ready dataframe.
        features: Candidate feature names.

    Returns:
        One row per available feature.
    """
    available = [f for f in features if f in df.columns]
    if not available:
        return pd.DataFrame(
            columns=["feature", "n_missing", "pct_missing", "mean", "std", "skew"]
        )

    numeric = df[available].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    correlation = numeric.corr(method="spearman").abs()
    # Blank the diagonal so a feature is not reported as correlated with itself.
    # Done via pandas rather than np.fill_diagonal, because `to_numpy()` returns a
    # read-only view under pandas 3.
    for feature in correlation.columns:
        correlation.loc[feature, feature] = np.nan

    rows: list[dict[str, Any]] = []
    for feature in available:
        series = numeric[feature]
        partner = (
            correlation[feature].idxmax()
            if correlation[feature].notna().any()
            else None
        )
        rows.append(
            {
                "feature": feature,
                "n_missing": int(series.isna().sum()),
                "pct_missing": float(series.isna().mean() * 100.0),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if series.notna().sum() > 1 else 0.0,
                "skew": float(series.skew()) if series.notna().sum() > 2 else 0.0,
                "max_abs_corr": (
                    float(correlation[feature].max())
                    if correlation[feature].notna().any()
                    else 0.0
                ),
                "most_correlated_with": partner,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Feature preparation
# --------------------------------------------------------------------------- #


def prepare_features(
    df: pd.DataFrame, config: SegmentationConfig | None = None
) -> FeatureMatrix:
    """Build the scaled clustering matrix.

    Steps, in order: reject forbidden columns, drop unavailable or
    zero-variance features, impute missing values with the median, then scale.

    Customers are **not** dropped for a missing feature. Losing 4.7% of the
    portfolio because one ratio is undefined would bias the segmentation toward
    active accounts, so the median is imputed and recorded instead.

    Args:
        df: Analysis-ready dataframe.
        config: Run configuration.

    Returns:
        A :class:`FeatureMatrix`.

    Raises:
        ValueError: If a forbidden column is requested, or no feature survives.
    """
    from sklearn.preprocessing import RobustScaler, StandardScaler

    cfg = config or SegmentationConfig()

    leaked = sorted(set(cfg.features) & FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(
            f"Refusing to build a feature matrix containing {leaked}. "
            "The target and demographic attributes are blocked by policy."
        )

    excluded: dict[str, str] = {}
    usable: list[str] = []

    for feature in cfg.features:
        if feature not in df.columns:
            excluded[feature] = "Column not present in this dataset."
            continue
        series = pd.to_numeric(df[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if series.notna().sum() == 0:
            excluded[feature] = "Every value is missing or non-numeric."
            continue
        if series.nunique(dropna=True) < 2:
            excluded[feature] = (
                "Constant within this selection, so it cannot separate customers."
            )
            continue
        usable.append(feature)

    if not usable:
        raise ValueError(
            "No usable clustering feature remains. Excluded: "
            + "; ".join(f"{k} ({v})" for k, v in excluded.items())
        )

    raw = df[usable].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    imputations: list[ImputationInfo] = []
    for feature in usable:
        n_missing = int(raw[feature].isna().sum())
        if n_missing == 0:
            continue
        fill_value = float(raw[feature].median())
        pct = n_missing / len(raw) * 100.0 if len(raw) else 0.0
        note = (
            "An undefined repayment ratio means every prior statement was zero or "
            "negative, so there was no balance to repay. That usually indicates an "
            "inactive or in-credit account rather than poor repayment. The median "
            "is imputed so these customers still receive a segment; the imputed "
            "value is not an observed customer value and is excluded from the "
            "stored feature column."
            if feature == "repayment_ratio_capped_mean"
            else (
                "Median imputed so the customer still receives a segment. The "
                "imputed value is not an observed customer value."
            )
        )
        raw[feature] = raw[feature].fillna(fill_value)
        imputations.append(
            ImputationInfo(
                column=feature,
                n_missing=n_missing,
                pct_missing=pct,
                method="median",
                fill_value=fill_value,
                note=note,
            )
        )

    scaler = StandardScaler() if cfg.scaler == "standard" else RobustScaler()
    scaled = scaler.fit_transform(raw.to_numpy(dtype="float64"))

    matrix = FeatureMatrix(
        scaled=np.asarray(scaled, dtype="float64"),
        raw=raw,
        feature_names=tuple(usable),
        scaler_name=cfg.scaler,
        imputations=tuple(imputations),
        excluded=excluded,
    )
    logger.info(
        "Prepared clustering matrix: %d rows x %d features, scaler=%s, %d cell(s) "
        "imputed, %d feature(s) excluded.",
        matrix.n_rows,
        matrix.n_features,
        cfg.scaler,
        matrix.total_imputed,
        len(excluded),
    )
    return matrix


# --------------------------------------------------------------------------- #
# K selection
# --------------------------------------------------------------------------- #


#: Floor on how many rows each cluster contributes to a silhouette sample. Two is
#: the smallest count at which a cluster has any within-cluster distance at all;
#: with one row the a(i) term is undefined and the cluster adds nothing but noise.
MIN_ROWS_PER_CLUSTER_IN_SAMPLE: int = 2


def _stratified_sample_index(
    labels: np.ndarray, sample_size: int, seed: int
) -> np.ndarray:
    """Draw a reproducible row sample that keeps every cluster represented.

    A plain random subset of a 30,000-row book can miss a small cluster entirely.
    The silhouette then sees a single label and is undefined, which is the failure
    this function exists to make impossible rather than to catch after the fact.

    Each cluster is allocated a share of the sample proportional to its real size,
    floored at :data:`MIN_ROWS_PER_CLUSTER_IN_SAMPLE` and capped at the number of
    rows it actually has. Flooring can push the total slightly above
    ``sample_size``; the overshoot is at most ``k * 2`` rows, which is negligible
    against a sample of thousands and is preferred over dropping a cluster to hit
    an exact size.

    Determinism comes from seeding a dedicated generator, so the same data and
    seed always produce the same sample and therefore the same score.

    Args:
        labels: Cluster assignment per row.
        sample_size: Target number of rows in the sample.
        seed: Seed for the sampling generator.

    Returns:
        Sorted array of positional indices into ``labels``.
    """
    rng = np.random.default_rng(seed)
    n_rows = int(labels.shape[0])
    picked: list[np.ndarray] = []

    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        proportional = int(round(sample_size * members.size / n_rows))
        quota = min(max(proportional, MIN_ROWS_PER_CLUSTER_IN_SAMPLE), members.size)
        picked.append(rng.choice(members, size=quota, replace=False))

    # Sorting keeps the sample order independent of label iteration order, so the
    # returned index is a pure function of (labels, sample_size, seed).
    return np.sort(np.concatenate(picked))


def _safe_silhouette(
    x: np.ndarray, labels: np.ndarray, sample_size: int | None, seed: int
) -> float:
    """Compute the silhouette score on a sample that is valid by construction.

    Silhouette is quadratic in row count, so on this dataset it is evaluated on a
    sample rather than all 30,000 customers. The sample is stratified by cluster
    (see :func:`_stratified_sample_index`) which guarantees at least two distinct
    labels and proportional representation of every cluster, including small ones.

    The score is never invented. When the partition genuinely cannot be scored -
    fewer than two clusters exist, or sklearn rejects the input - NaN is returned
    so the caller can skip that K and say so, rather than substituting a value.

    Args:
        x: Scaled feature matrix.
        labels: Cluster assignment per row.
        sample_size: Target sample size, or None to score every row.
        seed: Seed for the stratified sample.

    Returns:
        The silhouette score, or NaN when it genuinely cannot be computed.
    """
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)

    # Not a sampling problem: a single-cluster partition has no silhouette at any
    # sample size, so report it as unavailable immediately.
    if np.unique(labels).size < 2:
        logger.warning(
            "Silhouette is undefined: the partition has %d distinct cluster(s).",
            np.unique(labels).size,
        )
        return float("nan")

    if sample_size is not None and sample_size < labels.shape[0]:
        index = _stratified_sample_index(labels, sample_size, seed)
        x_eval, labels_eval = x[index], labels[index]
        logger.debug(
            "Silhouette on a stratified sample of %d of %d rows, %d clusters "
            "represented.",
            len(index),
            labels.shape[0],
            np.unique(labels_eval).size,
        )
    else:
        x_eval, labels_eval = x, labels

    try:
        return float(silhouette_score(x_eval, labels_eval))
    except ValueError as exc:
        # Reached only for inputs the stratification cannot repair, e.g. a cluster
        # of identical points producing a degenerate distance matrix.
        logger.warning("Silhouette could not be computed: %s", exc)
        return float("nan")


def _elbow_k(ks: Sequence[int], inertias: Sequence[float]) -> int:
    """Locate the inertia elbow by maximum distance from the endpoint chord.

    The standard geometric knee: draw a line from the first to the last point and
    take the K furthest from it. Cheap, deterministic, and needs no extra
    dependency.

    Args:
        ks: K values in ascending order.
        inertias: Corresponding inertia values.

    Returns:
        The elbow K, or the first K when the curve is too short or flat.
    """
    if len(ks) < 3:
        return int(ks[0]) if ks else 2

    x = np.asarray(ks, dtype="float64")
    y = np.asarray(inertias, dtype="float64")

    # Normalise both axes so the distance is not dominated by inertia's scale.
    x_span = x[-1] - x[0]
    y_span = y[0] - y[-1]
    if x_span <= 0 or y_span <= 0:
        return int(ks[0])

    x_norm = (x - x[0]) / x_span
    y_norm = (y - y[-1]) / y_span

    # Distance from each point to the chord between the first and last points.
    x1, y1 = x_norm[0], y_norm[0]
    x2, y2 = x_norm[-1], y_norm[-1]
    numerator = np.abs((y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
    denominator = np.hypot(y2 - y1, x2 - x1)
    if denominator == 0:
        return int(ks[0])

    return int(x[int(np.argmax(numerator / denominator))])


def evaluate_k_range(
    matrix: FeatureMatrix, config: SegmentationConfig | None = None
) -> KSelectionResult:
    """Score every K in the configured range.

    Silhouette is the primary criterion, with the inertia elbow and
    Davies-Bouldin reported alongside. Where silhouette's best K yields a
    degenerate partition - one cluster holding almost nobody - the next best
    non-degenerate K is recommended instead, and the substitution is stated in the
    rationale rather than made quietly.

    Args:
        matrix: Prepared feature matrix.
        config: Run configuration.

    Returns:
        A :class:`KSelectionResult`.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

    cfg = config or SegmentationConfig()
    x = matrix.scaled
    n_rows = matrix.n_rows

    was_sampled = n_rows > cfg.silhouette_sample
    sample_size = cfg.silhouette_sample if was_sampled else n_rows

    scores: list[KScore] = []
    for k in cfg.k_range:
        if k >= n_rows:
            logger.warning("Skipping k=%d: only %d row(s) available.", k, n_rows)
            continue

        model = KMeans(
            n_clusters=k,
            random_state=cfg.seed,
            n_init=cfg.n_init,
            max_iter=cfg.max_iter,
        )
        labels = model.fit_predict(x)

        if len(np.unique(labels)) < 2:
            logger.warning("k=%d collapsed to a single cluster; skipping.", k)
            continue

        counts = np.bincount(labels, minlength=k)
        smallest_share = float(counts.min()) / n_rows * 100.0

        silhouette = _safe_silhouette(
            x, labels, sample_size if was_sampled else None, cfg.seed
        )
        if not np.isfinite(silhouette):
            logger.warning("Skipping k=%d: silhouette could not be computed.", k)
            continue

        scores.append(
            KScore(
                k=k,
                inertia=float(model.inertia_),
                silhouette=silhouette,
                davies_bouldin=float(davies_bouldin_score(x, labels)),
                calinski_harabasz=float(calinski_harabasz_score(x, labels)),
                smallest_cluster_share_pct=smallest_share,
            )
        )

    if not scores:
        return KSelectionResult(
            scores=(),
            recommended_k=cfg.k_min,
            silhouette_best_k=cfg.k_min,
            elbow_k=cfg.k_min,
            davies_bouldin_best_k=cfg.k_min,
            rationale="No value of K could be evaluated on this selection.",
            silhouette_sample_size=sample_size,
            was_sampled=was_sampled,
            scaler_name=matrix.scaler_name,
        )

    silhouette_best = max(scores, key=lambda s: s.silhouette)
    davies_best = min(scores, key=lambda s: s.davies_bouldin)
    elbow = _elbow_k([s.k for s in scores], [s.inertia for s in scores])

    non_degenerate = [s for s in scores if not s.is_degenerate]
    if silhouette_best.is_degenerate and non_degenerate:
        recommended = max(non_degenerate, key=lambda s: s.silhouette)
        rationale = (
            f"K={silhouette_best.k} scored the highest silhouette "
            f"({silhouette_best.silhouette:.4f}) but produced a degenerate "
            f"partition: its smallest cluster held only "
            f"{silhouette_best.smallest_cluster_share_pct:.2f}% of customers, which "
            "describes a handful of outliers rather than a usable segment. "
            f"K={recommended.k} is recommended instead - the best silhouette "
            f"({recommended.silhouette:.4f}) among partitions where every segment "
            "holds a meaningful share. "
        )
    else:
        recommended = silhouette_best
        rationale = (
            f"K={recommended.k} is recommended: it achieved the highest silhouette "
            f"score ({recommended.silhouette:.4f}) across K="
            f"{scores[0].k}-{scores[-1].k}, and its smallest segment holds "
            f"{recommended.smallest_cluster_share_pct:.2f}% of customers. "
        )

    rationale += (
        f"The inertia elbow sits at K={elbow} and Davies-Bouldin is best at "
        f"K={davies_best.k} ({davies_best.davies_bouldin:.4f}); lower is better for "
        "that measure. "
    )
    if elbow != recommended.k or davies_best.k != recommended.k:
        rationale += (
            "The three criteria do not fully agree, which is normal for "
            "continuous behavioural data that has no naturally separated groups. "
            "Silhouette is treated as primary, and the full curve is shown so the "
            "disagreement is visible rather than hidden. "
        )
    if was_sampled:
        rationale += (
            f"Silhouette was computed on a fixed-seed sample of about "
            f"{sample_size:,} of {n_rows:,} customers, because the metric is "
            "quadratic in sample size. The sample is stratified by segment, so "
            "every segment is represented in proportion to its real size and no "
            "small segment can be missed. Inertia and Davies-Bouldin use all rows."
        )

    result = KSelectionResult(
        scores=tuple(scores),
        recommended_k=recommended.k,
        silhouette_best_k=silhouette_best.k,
        elbow_k=elbow,
        davies_bouldin_best_k=davies_best.k,
        rationale=rationale.strip(),
        silhouette_sample_size=sample_size,
        was_sampled=was_sampled,
        scaler_name=matrix.scaler_name,
    )
    logger.info(
        "K sweep (%s): recommended K=%d, silhouette best K=%d, elbow K=%d, "
        "Davies-Bouldin best K=%d.",
        matrix.scaler_name,
        result.recommended_k,
        result.silhouette_best_k,
        result.elbow_k,
        result.davies_bouldin_best_k,
    )
    return result


def compare_scalers(
    df: pd.DataFrame,
    config: SegmentationConfig | None = None,
    scalers: Sequence[str] = VALID_SCALERS,
) -> ScalerComparison:
    """Compare scaling strategies on measured clustering quality.

    K-Means minimises squared Euclidean distance, so the scaler determines which
    features dominate. Standard scaling divides by the standard deviation, which a
    long right tail inflates - the effect is to compress the bulk of a skewed
    feature into a narrow band. Robust scaling uses the median and IQR, so heavy
    skew has far less influence. Which one actually clusters better is an empirical
    question, answered here rather than assumed.

    Args:
        df: Analysis-ready dataframe.
        config: Base configuration; its ``scaler`` field is overridden per run.
        scalers: Scalers to compare.

    Returns:
        A :class:`ScalerComparison`.
    """
    from dataclasses import replace

    cfg = config or SegmentationConfig()
    per_scaler: dict[str, KSelectionResult] = {}

    for name in scalers:
        if name not in VALID_SCALERS:
            logger.warning("Skipping unknown scaler '%s'.", name)
            continue
        scoped = replace(cfg, scaler=name)
        matrix = prepare_features(df, scoped)
        per_scaler[name] = evaluate_k_range(matrix, scoped)

    if not per_scaler:
        raise ValueError("No valid scaler was evaluated.")

    def best_silhouette(result: KSelectionResult) -> float:
        score = result.score_for(result.recommended_k)
        return score.silhouette if score else float("-inf")

    chosen = max(per_scaler, key=lambda name: best_silhouette(per_scaler[name]))

    parts: list[str] = []
    for name, result in per_scaler.items():
        score = result.score_for(result.recommended_k)
        if score is None:
            continue
        parts.append(
            f"{name}: best silhouette {score.silhouette:.4f} at K={score.k} "
            f"(Davies-Bouldin {score.davies_bouldin:.4f}, smallest segment "
            f"{score.smallest_cluster_share_pct:.2f}%)"
        )

    rationale = (
        "Measured comparison: " + "; ".join(parts) + ". "
        f"'{chosen}' scaling was selected because it produced the higher silhouette "
        "score at its recommended K. The choice is empirical, not conventional."
    )

    logger.info("Scaler comparison complete: chose '%s'. %s", chosen, rationale)
    return ScalerComparison(per_scaler=per_scaler, chosen=chosen, rationale=rationale)


# --------------------------------------------------------------------------- #
# Naming and profiling
# --------------------------------------------------------------------------- #


def _derive_label(
    z_scores: Mapping[str, float], threshold: float = TRAIT_THRESHOLD
) -> tuple[str, tuple[str, ...]]:
    """Compose a segment name from its measured centroid position.

    Traits are ranked by how far the segment's mean sits from the portfolio mean,
    in standard deviations, and the two strongest become the name. Every word is
    therefore traceable to a measured value, and nothing is pre-assigned.

    Args:
        z_scores: Feature -> standardised distance from the portfolio mean.
        threshold: Minimum absolute distance to count as a trait.

    Returns:
        ``(label, traits)``.
    """
    ranked = sorted(
        (
            (feature, value)
            for feature, value in z_scores.items()
            if feature in TRAIT_VOCABULARY and np.isfinite(value)
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    traits: list[str] = []
    for feature, value in ranked:
        if abs(value) < threshold:
            continue
        high_word, low_word = TRAIT_VOCABULARY[feature]
        traits.append(high_word if value > 0 else low_word)
        if len(traits) == 2:
            break

    if not traits:
        # Genuinely average on every measured dimension: say so rather than
        # inventing a distinguishing characteristic.
        return "Mid-Range / Typical-Behaviour", ()

    if len(traits) == 1:
        return f"{traits[0]} / Otherwise-Typical", tuple(traits)

    return " / ".join(traits), tuple(traits)


def _describe_segment(
    label: str,
    traits: Sequence[str],
    metrics: Mapping[str, float],
    z_scores: Mapping[str, float],
    feature_names: Sequence[str],
) -> str:
    """Write a factual sentence describing how a segment differs.

    Args:
        label: Derived label.
        traits: Distinguishing trait words.
        metrics: Segment means in original units.
        z_scores: Standardised distances from the portfolio mean.
        feature_names: Features in the model.

    Returns:
        A description grounded in measured values.
    """
    if not traits:
        return (
            "This segment sits close to the portfolio average on every clustering "
            "feature, so it has no strongly distinguishing behaviour."
        )

    ranked = sorted(
        (f for f in feature_names if f in z_scores and np.isfinite(z_scores[f])),
        key=lambda f: abs(z_scores[f]),
        reverse=True,
    )[:3]

    fragments: list[str] = []
    for feature in ranked:
        value = metrics.get(feature)
        if value is None or not np.isfinite(value):
            continue
        direction = "above" if z_scores[feature] > 0 else "below"
        fragments.append(
            f"{FEATURE_LABELS.get(feature, feature).lower()} "
            f"{_format_metric(feature, value)} "
            f"({abs(z_scores[feature]):.2f} SD {direction} the portfolio mean)"
        )

    return "Characterised by " + "; ".join(fragments) + "."


def _format_metric(feature: str, value: float) -> str:
    """Format a metric for prose, according to its declared format."""
    if value is None or not np.isfinite(value):
        return "n/a"

    style = FEATURE_FORMATS.get(feature, "decimal")
    if style == "currency":
        return f"NT${value:,.0f}"
    if style == "currency_signed":
        return f"NT${value:+,.0f}/month"
    if style == "percent":
        return f"{value:.1%}"
    if style == "decimal3":
        return f"{value:.3f}"
    return f"{value:.2f}"


def profile_segments(
    df: pd.DataFrame,
    labels: pd.Series,
    matrix: FeatureMatrix,
    target: str | None = None,
) -> tuple[SegmentProfile, ...]:
    """Measure each segment and derive its name.

    The observed default rate is computed here, **after** clustering, and is
    descriptive only. It played no part in forming the segments.

    Args:
        df: Analysis-ready dataframe.
        labels: Cluster assignment per row.
        matrix: The feature matrix used to cluster.
        target: Optional binary outcome column for profiling.

    Returns:
        One :class:`SegmentProfile` per cluster, ordered by cluster id.
    """
    features = list(matrix.feature_names)
    raw = matrix.raw
    n_total = len(raw)

    overall_mean = raw.mean()
    overall_std = raw.std(ddof=0).replace(0.0, np.nan)

    profiles: list[SegmentProfile] = []
    for cluster_id in sorted(labels.dropna().unique()):
        mask = labels == cluster_id
        size = int(mask.sum())
        if size == 0:
            continue

        segment_raw = raw.loc[mask.reindex(raw.index, fill_value=False)]
        metrics = {f: float(segment_raw[f].mean()) for f in features}
        z_scores = {
            f: float((metrics[f] - overall_mean[f]) / overall_std[f])
            if np.isfinite(overall_std[f]) and overall_std[f] != 0
            else 0.0
            for f in features
        }

        label, traits = _derive_label(z_scores)
        description = _describe_segment(label, traits, metrics, z_scores, features)

        default_rate: float | None = None
        if target and target in df.columns:
            outcome = pd.to_numeric(
                df.loc[mask.reindex(df.index, fill_value=False), target], errors="coerce"
            ).dropna()
            if not outcome.empty:
                default_rate = float(outcome.mean() * 100.0)

        profiles.append(
            SegmentProfile(
                cluster_id=int(cluster_id),
                label=label,
                description=description,
                traits=traits,
                size=size,
                share_pct=size / n_total * 100.0 if n_total else 0.0,
                metrics=metrics,
                z_scores=z_scores,
                observed_default_rate_pct=default_rate,
            )
        )

    logger.info("Profiled %d segment(s).", len(profiles))
    return tuple(profiles)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def fit_segments(
    df: pd.DataFrame,
    k: int,
    config: SegmentationConfig | None = None,
    matrix: FeatureMatrix | None = None,
    k_selection: KSelectionResult | None = None,
    target: str | None = None,
    scaler_comparison: ScalerComparison | None = None,
) -> SegmentationResult:
    """Fit K-Means at a given K and profile the result.

    Args:
        df: Analysis-ready dataframe.
        k: Number of clusters.
        config: Run configuration.
        matrix: Pre-built feature matrix, rebuilt when omitted.
        k_selection: Pre-computed K sweep, recomputed when omitted.
        target: Optional outcome column for profiling only.
        scaler_comparison: Optional scaler evidence to carry into the result.

    Returns:
        A :class:`SegmentationResult`.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

    cfg = config or SegmentationConfig()
    feature_matrix = matrix if matrix is not None else prepare_features(df, cfg)
    selection = (
        k_selection
        if k_selection is not None
        else evaluate_k_range(feature_matrix, cfg)
    )

    x = feature_matrix.scaled
    model = KMeans(
        n_clusters=k, random_state=cfg.seed, n_init=cfg.n_init, max_iter=cfg.max_iter
    )
    raw_labels = model.fit_predict(x)
    labels = pd.Series(raw_labels, index=feature_matrix.index, name="cluster")

    was_sampled = feature_matrix.n_rows > cfg.silhouette_sample
    metrics = {
        "inertia": float(model.inertia_),
        "silhouette": _safe_silhouette(
            x, raw_labels, cfg.silhouette_sample if was_sampled else None, cfg.seed
        ),
        "davies_bouldin": float(davies_bouldin_score(x, raw_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, raw_labels)),
        "n_iter": float(model.n_iter_),
    }

    resolved_target = target
    if resolved_target is None:
        registry = build_registry(df)
        resolved_target = registry.one(Role.TARGET)

    profiles = profile_segments(df, labels, feature_matrix, resolved_target)

    # PCA is for display only. K-Means already ran on the full scaled matrix, so
    # the projection is a lossy view of the model, never a substitute for it.
    n_components = min(cfg.pca_components, feature_matrix.n_features)
    pca = PCA(n_components=n_components, random_state=cfg.seed)
    coordinates = pca.fit_transform(x)

    names = {p.cluster_id: p.display_name for p in profiles}
    ordered = [p.display_name for p in sorted(profiles, key=lambda p: p.cluster_id)]
    projection = pd.DataFrame(
        {
            "pc1": coordinates[:, 0],
            "pc2": coordinates[:, 1] if n_components > 1 else np.zeros(len(coordinates)),
            "cluster": raw_labels,
        },
        index=feature_matrix.index,
    )
    projection["segment"] = pd.Categorical(
        projection["cluster"].map(names), categories=ordered, ordered=True
    )

    result = SegmentationResult(
        labels=labels,
        profiles=profiles,
        k=k,
        k_selection=selection,
        matrix=feature_matrix,
        projection=projection,
        pca_explained_variance=tuple(float(v) for v in pca.explained_variance_ratio_),
        metrics=metrics,
        scaler_name=feature_matrix.scaler_name,
        scaler_comparison=scaler_comparison,
        target_excluded=True,
    )
    logger.info(
        "Fitted K=%d on %d customers: silhouette %.4f, Davies-Bouldin %.4f, "
        "PCA captures %.1f%% of variance.",
        k,
        feature_matrix.n_rows,
        metrics["silhouette"],
        metrics["davies_bouldin"],
        result.pca_total_variance_pct,
    )
    return result


def run_segmentation(
    df: pd.DataFrame,
    registry: ColumnRegistry | None = None,
    k: int | None = None,
    config: SegmentationConfig | None = None,
    compare_scaling: bool = True,
) -> SegmentationResult | SegmentationUnavailable:
    """Run the full segmentation workflow.

    Compares scalers, sweeps K, fits the recommended (or requested) K, and
    profiles the segments.

    Args:
        df: Analysis-ready dataframe.
        registry: Optional pre-built registry, used to resolve the target.
        k: Force a specific K. When omitted the measured recommendation is used.
        config: Run configuration.
        compare_scaling: Compare scalers and adopt the better one. Set False to
            use the configured scaler as-is, which is faster.

    Returns:
        A :class:`SegmentationResult`, or :class:`SegmentationUnavailable`.
    """
    from dataclasses import replace

    cfg = config or SegmentationConfig()
    reg = registry or build_registry(df)
    n_rows = int(len(df))

    if n_rows < MIN_ROWS_FOR_CLUSTERING:
        return SegmentationUnavailable(
            reason=(
                f"Segmentation needs at least {MIN_ROWS_FOR_CLUSTERING} customers to "
                f"produce stable clusters; this selection has {n_rows:,}."
            ),
            n_rows=n_rows,
        )

    missing = tuple(f for f in cfg.features if f not in df.columns)
    if len(missing) == len(cfg.features):
        return SegmentationUnavailable(
            reason=(
                "None of the behavioural clustering features are available in this "
                "dataset. Feature engineering must run first."
            ),
            n_rows=n_rows,
            missing_features=missing,
        )

    try:
        comparison: ScalerComparison | None = None
        active = cfg

        if compare_scaling:
            comparison = compare_scalers(df, cfg)
            active = replace(cfg, scaler=comparison.chosen)
            matrix = prepare_features(df, active)
            selection = comparison.per_scaler[comparison.chosen]
        else:
            matrix = prepare_features(df, active)
            selection = evaluate_k_range(matrix, active)

        if not selection.scores:
            return SegmentationUnavailable(
                reason=(
                    "No value of K could be evaluated on this selection, so no "
                    "segmentation was produced."
                ),
                n_rows=n_rows,
            )

        chosen_k = k if k is not None else selection.recommended_k
        chosen_k = max(2, min(int(chosen_k), max(2, matrix.n_rows - 1)))

        return fit_segments(
            df,
            k=chosen_k,
            config=active,
            matrix=matrix,
            k_selection=selection,
            target=reg.one(Role.TARGET),
            scaler_comparison=comparison,
        )
    except (ValueError, KeyError) as exc:
        logger.exception("Segmentation failed.")
        return SegmentationUnavailable(
            reason=f"Segmentation could not be completed: {exc}", n_rows=n_rows
        )


# --------------------------------------------------------------------------- #
# Narrative
# --------------------------------------------------------------------------- #


def methodology_note(result: SegmentationResult) -> str:
    """Describe how the segments were produced, for display on the page.

    Args:
        result: A fitted segmentation.

    Returns:
        A paragraph grounded in the run's actual settings and measurements.
    """
    features = ", ".join(
        FEATURE_LABELS.get(f, f).replace("Avg ", "") for f in result.matrix.feature_names
    )
    imputed = result.matrix.total_imputed
    imputation_text = (
        f" {imputed:,} missing feature value(s) were median-imputed so no customer "
        "was dropped from the portfolio."
        if imputed
        else " No feature values required imputation."
    )
    return (
        f"Customers were grouped with K-Means on {result.matrix.n_features} measured "
        f"behavioural features ({features}), scaled using {result.scaler_name} "
        "scaling. The outcome variable was excluded from the feature matrix, as were "
        "demographic attributes, so the segments describe behaviour rather than "
        f"restating the answer.{imputation_text} K was chosen by measured silhouette "
        f"score across K={result.k_selection.scores[0].k}-"
        f"{result.k_selection.scores[-1].k}, supported by the inertia elbow and "
        f"Davies-Bouldin index. The observed default rate shown per segment was "
        "calculated afterwards and is descriptive only."
    )


def segment_interpretation(
    profile: SegmentProfile, overall_default_rate: float | None
) -> dict[str, str]:
    """Build the four-part interpretation for one segment.

    Args:
        profile: The segment to interpret.
        overall_default_rate: Portfolio rate for comparison, as a percentage.

    Returns:
        Keys ``fact``, ``behavior``, ``observed_risk`` and ``implication``.
    """
    fact = (
        f"{profile.size:,} customers ({profile.share_pct:.1f}% of the portfolio) fall "
        f"into this behavioural segment."
    )

    behavior = profile.description

    if profile.observed_default_rate_pct is None:
        observed_risk = (
            "No outcome column is available for this selection, so no observed "
            "default rate can be reported for this segment."
        )
    elif overall_default_rate is None:
        observed_risk = (
            f"The observed default rate in this segment is "
            f"{profile.observed_default_rate_pct:.2f}%."
        )
    else:
        difference = profile.observed_default_rate_pct - overall_default_rate
        comparison = (
            f"{abs(difference):.2f} percentage points "
            f"{'above' if difference > 0 else 'below'} the portfolio rate of "
            f"{overall_default_rate:.2f}%"
        )
        observed_risk = (
            f"The observed default rate in this segment is "
            f"{profile.observed_default_rate_pct:.2f}%, {comparison}. This is a "
            "measured historical outcome for customers who happen to share these "
            "behaviours, not a prediction and not an effect of segment membership."
        )

    if profile.observed_default_rate_pct is None or overall_default_rate is None:
        implication = (
            "The segment is defined by measured behaviour and can be monitored as a "
            "group once an outcome measure is available."
        )
    elif profile.observed_default_rate_pct > overall_default_rate:
        implication = (
            "Because this group is defined by behaviour that is measurable every "
            "month, it offers a monitoring handle that does not depend on waiting "
            "for an outcome. The elevated observed rate makes it a candidate for "
            "closer review, and the behaviours in its profile indicate which "
            "measures to watch."
        )
    else:
        implication = (
            "The lower observed rate alongside a distinct behavioural profile makes "
            "this a group worth understanding before assuming it needs the same "
            "treatment as the rest of the portfolio. Whether the gap persists once "
            "credit limit and utilisation are controlled for is the next question."
        )

    return {
        "fact": fact,
        "behavior": behavior,
        "observed_risk": observed_risk,
        "implication": implication,
    }
