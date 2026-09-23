"""
Driver analysis - answering "why is this happening?" without overclaiming.

Four independent lenses are applied and then triangulated, so no single method
carries a conclusion on its own:

1. **Rank correlation** with the outcome (Spearman, robust to the heavy skew).
2. **Group comparison** - outcome rate per category, with a chi-square test of
   independence and Cramer's V as the effect size.
3. **Distribution comparison** - Mann-Whitney U between outcome classes for
   numeric features, with a rank-biserial effect size.
4. **Tree-based importance** - a decision tree's impurity importance, which
   catches non-monotonic relationships the other three miss.

Language discipline
-------------------
This is observational data, so causal claims are not available at any sample
size. Every generated sentence is produced by :func:`describe_driver`, which only
ever emits "associated with", "correlated with" or "identified as a predictive
feature". The words "causes", "drives" and "because of" are not used, and
:data:`CAUSATION_CAVEAT` is attached to every result for display.

On statistical significance
---------------------------
With 30,000 rows nearly any difference is "significant" at p < 0.05, which makes
the p-value nearly worthless on its own. Every test therefore reports an effect
size next to it, and ranking is by effect size, not by p-value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.config import settings
from src.logging_setup import get_logger
from src.schema import ID_COLUMN, ColumnRegistry, Role, build_registry

logger = get_logger(__name__)

CAUSATION_CAVEAT: Final[str] = (
    "These are observed statistical associations in historical data, not causal "
    "relationships. A feature identified here may be predictive of the outcome "
    "without influencing it - both could follow from a third factor not present "
    "in the dataset. No causal claim is made or implied."
)

SIGNIFICANCE_CAVEAT: Final[str] = (
    "With a large sample, almost any difference becomes statistically significant. "
    "Effect sizes are reported alongside p-values and are used for ranking, "
    "because they indicate whether a difference is large enough to matter."
)

#: Columns excluded from driver analysis because using them would be either
#: circular or discriminatory.
DEFAULT_EXCLUDED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        ID_COLUMN,
        "client_id",
        # Protected attributes. Available in the data, but pricing credit risk on
        # them is discriminatory and unlawful in many jurisdictions. They are
        # analysed separately in the fairness audit, never as risk "drivers".
        "sex",
        "sex_code",
        "marriage",
        "marriage_code",
        # Bookkeeping flags, not client behaviour.
        "is_duplicate_excluding_id",
        "has_extreme_value",
    }
)


class EvidenceStrength(str, Enum):
    """How strong the evidence for an association is."""

    NEGLIGIBLE = "negligible"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


# --------------------------------------------------------------------------- #
# Result structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DriverEvidence:
    """Evidence for one feature's association with the outcome, from one method."""

    feature: str
    method: str
    statistic: float
    effect_size: float
    effect_size_name: str
    strength: EvidenceStrength
    direction: str
    p_value: float | None = None
    n_used: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        """p < 0.05. Reported for completeness; never used alone for ranking."""
        return self.p_value is not None and self.p_value < 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "method": self.method,
            "statistic": round(self.statistic, 6),
            "effect_size": round(self.effect_size, 6),
            "effect_size_name": self.effect_size_name,
            "strength": self.strength.value,
            "direction": self.direction,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "n_used": self.n_used,
            "details": self.details,
        }


@dataclass(frozen=True)
class DriverAnalysisResult:
    """Combined output of the driver analysis."""

    target: str
    outcome_rate_pct: float
    correlations: tuple[DriverEvidence, ...]
    numeric_comparisons: tuple[DriverEvidence, ...]
    categorical_tests: tuple[DriverEvidence, ...]
    tree_importances: tuple[DriverEvidence, ...]
    consensus: tuple[dict[str, Any], ...]
    n_rows: int
    caveat: str = CAUSATION_CAVEAT
    significance_note: str = SIGNIFICANCE_CAVEAT

    def top_features(self, limit: int = 10) -> list[dict[str, Any]]:
        """Features ranked by how many methods flagged them and how strongly."""
        return list(self.consensus[:limit])

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "outcome_rate_pct": round(self.outcome_rate_pct, 4),
            "n_rows": self.n_rows,
            "correlations": [e.to_dict() for e in self.correlations],
            "numeric_comparisons": [e.to_dict() for e in self.numeric_comparisons],
            "categorical_tests": [e.to_dict() for e in self.categorical_tests],
            "tree_importances": [e.to_dict() for e in self.tree_importances],
            "consensus": list(self.consensus),
            "caveat": self.caveat,
            "significance_note": self.significance_note,
        }


@dataclass(frozen=True)
class DriverAnalysisUnavailable:
    """Returned when driver analysis cannot run."""

    reason: str
    n_rows: int = 0

    @property
    def is_available(self) -> bool:
        """Always False."""
        return False

    def to_dict(self) -> dict[str, Any]:
        return {"available": False, "reason": self.reason, "n_rows": self.n_rows}


# --------------------------------------------------------------------------- #
# Strength classification
# --------------------------------------------------------------------------- #


def _strength_from_magnitude(magnitude: float) -> EvidenceStrength:
    """Classify an absolute effect size on a shared 0-1 scale.

    Deliberately uniform across methods so a "moderate" correlation and a
    "moderate" Cramer's V mean comparable things when the consensus is built.
    """
    value = abs(magnitude)
    if value >= 0.50:
        return EvidenceStrength.STRONG
    if value >= 0.30:
        return EvidenceStrength.MODERATE
    if value >= 0.10:
        return EvidenceStrength.WEAK
    return EvidenceStrength.NEGLIGIBLE


def describe_driver(evidence: DriverEvidence, target_label: str = "the outcome") -> str:
    """Produce an association-only sentence for one piece of evidence.

    Centralised on purpose: every driver sentence in the application comes from
    here, so the no-causal-claims rule is enforced by construction rather than by
    remembering to phrase things carefully at each call site.

    Args:
        evidence: The evidence to describe.
        target_label: How to refer to the outcome.

    Returns:
        A sentence using association language only.
    """
    feature = evidence.feature

    if evidence.strength is EvidenceStrength.NEGLIGIBLE:
        return (
            f"'{feature}' shows no meaningful association with {target_label} "
            f"({evidence.effect_size_name} = {evidence.effect_size:.3f})."
        )

    if evidence.method == "tree_importance":
        return (
            f"'{feature}' was identified as a {evidence.strength.value} predictive "
            f"feature for {target_label} (relative importance "
            f"{evidence.effect_size:.3f})."
        )

    if evidence.method == "chi_square":
        return (
            f"'{feature}' is {evidence.strength.value}ly associated with "
            f"{target_label}; outcome rates differ across its categories "
            f"(Cramer's V = {evidence.effect_size:.3f})."
        )

    direction_phrase = (
        "higher values are associated with a higher rate of"
        if evidence.direction == "positive"
        else "higher values are associated with a lower rate of"
    )
    return (
        f"'{feature}' shows a {evidence.strength.value} association with "
        f"{target_label}: {direction_phrase} {target_label} "
        f"({evidence.effect_size_name} = {evidence.effect_size:.3f})."
    )


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #


def candidate_features(
    df: pd.DataFrame,
    target: str,
    registry: ColumnRegistry | None = None,
    exclude: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """Choose the numeric and categorical features to analyse.

    Excludes the identifier, the target, protected attributes and any raw
    ``*_code`` column whose labelled twin is present.

    Args:
        df: Dataframe to inspect.
        target: Target column name.
        registry: Optional pre-built registry.
        exclude: Extra columns to exclude.

    Returns:
        ``(numeric_features, categorical_features)``.
    """
    reg = registry or build_registry(df)
    excluded = set(DEFAULT_EXCLUDED_COLUMNS) | set(exclude) | {target}

    # Prefer the readable label column over its raw code twin.
    for code_column, label_column in (
        ("education_code", "education"),
        ("sex_code", "sex"),
        ("marriage_code", "marriage"),
    ):
        if label_column in df.columns:
            excluded.add(code_column)

    numeric: list[str] = []
    categorical: list[str] = []

    for column in df.columns:
        name = str(column)
        if name in excluded:
            continue
        series = df[column]
        if series.nunique(dropna=True) < 2:
            continue

        if pd.api.types.is_bool_dtype(series):
            numeric.append(name)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(name)
        elif isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(series):
            # Guard against a high-cardinality free-text column.
            if series.nunique(dropna=True) <= 30:
                categorical.append(name)

    return numeric, categorical


# --------------------------------------------------------------------------- #
# Method 1: rank correlation with the outcome
# --------------------------------------------------------------------------- #


def correlation_drivers(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    method: str = "spearman",
) -> list[DriverEvidence]:
    """Rank-correlate each numeric feature with the binary outcome.

    Args:
        df: Dataframe to analyse.
        target: Binary 0/1 target column.
        features: Numeric feature columns.
        method: ``spearman`` or ``pearson``.

    Returns:
        Evidence per feature, strongest absolute correlation first.
    """
    if target not in df.columns:
        return []

    target_values = pd.to_numeric(df[target], errors="coerce")
    evidence: list[DriverEvidence] = []

    for feature in features:
        if feature not in df.columns:
            continue
        values = (
            pd.to_numeric(df[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        paired = pd.DataFrame({"x": values, "y": target_values}).dropna()
        if len(paired) < 10 or paired["x"].nunique() < 2 or paired["y"].nunique() < 2:
            continue

        if method == "spearman":
            statistic, p_value = stats.spearmanr(paired["x"], paired["y"])
        else:
            statistic, p_value = stats.pearsonr(paired["x"], paired["y"])

        if not np.isfinite(statistic):
            continue

        evidence.append(
            DriverEvidence(
                feature=feature,
                method=f"{method}_correlation",
                statistic=float(statistic),
                effect_size=float(statistic),
                effect_size_name="r_s" if method == "spearman" else "r",
                strength=_strength_from_magnitude(statistic),
                direction="positive" if statistic > 0 else "negative",
                p_value=float(p_value) if np.isfinite(p_value) else None,
                n_used=int(len(paired)),
            )
        )

    evidence.sort(key=lambda e: abs(e.effect_size), reverse=True)
    logger.info("Correlation drivers: evaluated %d feature(s).", len(evidence))
    return evidence


# --------------------------------------------------------------------------- #
# Method 2: distribution comparison between outcome classes
# --------------------------------------------------------------------------- #


def numeric_group_tests(
    df: pd.DataFrame, target: str, features: Sequence[str]
) -> list[DriverEvidence]:
    """Compare each numeric feature between outcome classes with Mann-Whitney U.

    Mann-Whitney rather than a t-test: it is non-parametric, so the heavy skew in
    the monetary columns does not invalidate it. The effect size reported is the
    rank-biserial correlation, which maps onto the same 0-1 scale as the other
    methods.

    Args:
        df: Dataframe to analyse.
        target: Binary 0/1 target column.
        features: Numeric feature columns.

    Returns:
        Evidence per feature, largest effect size first.
    """
    if target not in df.columns:
        return []

    target_values = pd.to_numeric(df[target], errors="coerce")
    evidence: list[DriverEvidence] = []

    for feature in features:
        if feature not in df.columns:
            continue
        values = (
            pd.to_numeric(df[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        paired = pd.DataFrame({"x": values, "y": target_values}).dropna()
        positive = paired.loc[paired["y"] == 1, "x"]
        negative = paired.loc[paired["y"] == 0, "x"]

        if len(positive) < 10 or len(negative) < 10 or paired["x"].nunique() < 2:
            continue

        try:
            u_statistic, p_value = stats.mannwhitneyu(
                positive, negative, alternative="two-sided"
            )
        except ValueError:
            continue

        # Rank-biserial correlation: 2*U/(n1*n2) - 1, bounded to [-1, 1].
        n1, n2 = len(positive), len(negative)
        rank_biserial = (2.0 * float(u_statistic)) / (n1 * n2) - 1.0

        evidence.append(
            DriverEvidence(
                feature=feature,
                method="mann_whitney_u",
                statistic=float(u_statistic),
                effect_size=float(rank_biserial),
                effect_size_name="rank-biserial r",
                strength=_strength_from_magnitude(rank_biserial),
                direction="positive" if rank_biserial > 0 else "negative",
                p_value=float(p_value) if np.isfinite(p_value) else None,
                n_used=int(len(paired)),
                details={
                    "median_when_outcome_1": float(positive.median()),
                    "median_when_outcome_0": float(negative.median()),
                    "mean_when_outcome_1": float(positive.mean()),
                    "mean_when_outcome_0": float(negative.mean()),
                    "n_outcome_1": n1,
                    "n_outcome_0": n2,
                },
            )
        )

    evidence.sort(key=lambda e: abs(e.effect_size), reverse=True)
    logger.info("Mann-Whitney tests: evaluated %d feature(s).", len(evidence))
    return evidence


# --------------------------------------------------------------------------- #
# Method 3: categorical independence tests
# --------------------------------------------------------------------------- #


def categorical_tests(
    df: pd.DataFrame, target: str, features: Sequence[str], min_expected: int = 5
) -> list[DriverEvidence]:
    """Chi-square test of independence per categorical feature, with Cramer's V.

    Args:
        df: Dataframe to analyse.
        target: Binary 0/1 target column.
        features: Categorical feature columns.
        min_expected: Minimum expected cell count for the test to be trusted.

    Returns:
        Evidence per feature, largest Cramer's V first.
    """
    if target not in df.columns:
        return []

    evidence: list[DriverEvidence] = []

    for feature in features:
        if feature not in df.columns:
            continue
        table = pd.crosstab(df[feature], pd.to_numeric(df[target], errors="coerce"))
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        try:
            chi2, p_value, _dof, expected = stats.chi2_contingency(table)
        except ValueError:
            continue

        n = int(table.to_numpy().sum())
        if n == 0:
            continue

        # Cramer's V normalises chi-square to [0, 1] so it is comparable across
        # features with different numbers of categories.
        min_dimension = min(table.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n * min_dimension))) if min_dimension > 0 else 0.0

        rates = (table[1] / table.sum(axis=1) * 100.0) if 1 in table.columns else None

        evidence.append(
            DriverEvidence(
                feature=feature,
                method="chi_square",
                statistic=float(chi2),
                effect_size=cramers_v,
                effect_size_name="Cramer's V",
                strength=_strength_from_magnitude(cramers_v),
                direction="n/a",
                p_value=float(p_value) if np.isfinite(p_value) else None,
                n_used=n,
                details={
                    "n_categories": int(table.shape[0]),
                    "outcome_rate_by_category_pct": (
                        {str(k): round(float(v), 4) for k, v in rates.items()}
                        if rates is not None
                        else {}
                    ),
                    "min_expected_count": float(np.min(expected)),
                    "test_assumption_met": bool(np.min(expected) >= min_expected),
                    "assumption_note": (
                        ""
                        if np.min(expected) >= min_expected
                        else (
                            f"Smallest expected cell count is {np.min(expected):.1f}, "
                            f"below the recommended {min_expected}. Treat this "
                            "p-value as unreliable."
                        )
                    ),
                },
            )
        )

    evidence.sort(key=lambda e: e.effect_size, reverse=True)
    logger.info("Chi-square tests: evaluated %d feature(s).", len(evidence))
    return evidence


# --------------------------------------------------------------------------- #
# Method 4: tree-based importance
# --------------------------------------------------------------------------- #


def tree_importances(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    max_depth: int = 5,
    random_state: int | None = None,
) -> list[DriverEvidence]:
    """Fit a shallow decision tree and report its impurity importances.

    A **shallow** tree on purpose: depth 5 keeps it interpretable and resists
    overfitting, and this is an exploratory ranking rather than the predictive
    model. Its value is catching non-monotonic relationships that correlation
    and rank tests cannot see.

    Args:
        df: Dataframe to analyse.
        target: Binary 0/1 target column.
        features: Numeric feature columns.
        max_depth: Maximum tree depth.
        random_state: Seed. Defaults to the configured value.

    Returns:
        Evidence per feature with non-zero importance, largest first.
    """
    from sklearn.tree import DecisionTreeClassifier

    seed = random_state if random_state is not None else settings.random_state
    usable = [f for f in features if f in df.columns]
    if not usable or target not in df.columns:
        return []

    frame = df[usable].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    target_values = pd.to_numeric(df[target], errors="coerce")

    combined = pd.concat([frame, target_values.rename("__target__")], axis=1).dropna()
    if len(combined) < 50 or combined["__target__"].nunique() < 2:
        logger.warning(
            "Tree importance skipped: %d complete row(s), %d target class(es).",
            len(combined),
            combined["__target__"].nunique() if not combined.empty else 0,
        )
        return []

    x = combined[usable]
    y = combined["__target__"].astype("int64")

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=seed,
        class_weight="balanced",  # the outcome is imbalanced at roughly 22%
        min_samples_leaf=50,
    )
    tree.fit(x, y)

    evidence: list[DriverEvidence] = []
    for feature, importance in zip(usable, tree.feature_importances_):
        if importance <= 0:
            continue
        evidence.append(
            DriverEvidence(
                feature=feature,
                method="tree_importance",
                statistic=float(importance),
                effect_size=float(importance),
                effect_size_name="relative importance",
                strength=_strength_from_magnitude(importance),
                direction="n/a",
                p_value=None,
                n_used=int(len(combined)),
                details={
                    "model": f"DecisionTreeClassifier(max_depth={max_depth}, class_weight='balanced')",
                    "note": (
                        "Impurity-based importance from a shallow exploratory tree. "
                        "It is biased toward high-cardinality features, so it is used "
                        "as one signal among four rather than on its own."
                    ),
                },
            )
        )

    evidence.sort(key=lambda e: e.effect_size, reverse=True)
    logger.info("Tree importance: %d feature(s) had non-zero importance.", len(evidence))
    return evidence


# --------------------------------------------------------------------------- #
# Consensus
# --------------------------------------------------------------------------- #


def build_consensus(
    *evidence_groups: Sequence[DriverEvidence], limit: int = 25
) -> tuple[dict[str, Any], ...]:
    """Combine evidence from several methods into one ranking.

    Ranking is by the number of methods that found a non-negligible association
    first, then by mean effect size. Agreement across independent methods is
    stronger evidence than a single large statistic, which could be a quirk of
    one method's assumptions.

    Args:
        *evidence_groups: Evidence lists from the different methods.
        limit: Maximum features to return.

    Returns:
        Ranked feature records.
    """
    accumulated: dict[str, dict[str, Any]] = {}

    for group in evidence_groups:
        for item in group:
            record = accumulated.setdefault(
                item.feature,
                {
                    "feature": item.feature,
                    "methods": [],
                    "effect_sizes": {},
                    "strengths": {},
                    "directions": set(),
                    "n_methods_nonneglible": 0,
                },
            )
            record["methods"].append(item.method)
            record["effect_sizes"][item.method] = round(abs(item.effect_size), 6)
            record["strengths"][item.method] = item.strength.value
            if item.direction in {"positive", "negative"}:
                record["directions"].add(item.direction)
            if item.strength is not EvidenceStrength.NEGLIGIBLE:
                record["n_methods_nonneglible"] += 1

    consensus: list[dict[str, Any]] = []
    for record in accumulated.values():
        sizes = list(record["effect_sizes"].values())
        directions = record.pop("directions")
        consensus.append(
            {
                **record,
                "n_methods": len(record["methods"]),
                "mean_effect_size": round(float(np.mean(sizes)), 6) if sizes else 0.0,
                "max_effect_size": round(float(np.max(sizes)), 6) if sizes else 0.0,
                "direction": (
                    next(iter(directions))
                    if len(directions) == 1
                    else ("mixed" if directions else "n/a")
                ),
            }
        )

    consensus.sort(
        key=lambda r: (r["n_methods_nonneglible"], r["mean_effect_size"]), reverse=True
    )
    return tuple(consensus[:limit])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_driver_analysis(
    df: pd.DataFrame,
    target: str | None = None,
    registry: ColumnRegistry | None = None,
    min_rows: int | None = None,
    include_tree: bool = True,
) -> DriverAnalysisResult | DriverAnalysisUnavailable:
    """Run all four driver-analysis methods and triangulate them.

    Args:
        df: Dataframe carrying cleaned data and engineered features.
        target: Target column. Resolved from the registry when omitted.
        registry: Optional pre-built registry.
        min_rows: Minimum rows required.
        include_tree: Include the tree-importance method.

    Returns:
        A :class:`DriverAnalysisResult`, or :class:`DriverAnalysisUnavailable`.
    """
    reg = registry or build_registry(df)
    threshold = min_rows if min_rows is not None else max(
        settings.min_rows_for_analysis, 50
    )

    target_column = target or reg.one(Role.TARGET)
    if target_column is None or target_column not in df.columns:
        return DriverAnalysisUnavailable(
            reason=(
                "No target column is available, so there is no outcome to explain. "
                "Driver analysis requires a binary outcome such as "
                "'default_next_month'."
            ),
            n_rows=len(df),
        )

    if len(df) < threshold:
        return DriverAnalysisUnavailable(
            reason=(
                f"Driver analysis needs at least {threshold} rows but the selection "
                f"has {len(df)}. Statistical tests on very few rows are unreliable."
            ),
            n_rows=len(df),
        )

    target_values = pd.to_numeric(df[target_column], errors="coerce").dropna()
    if target_values.nunique() < 2:
        return DriverAnalysisUnavailable(
            reason=(
                f"The target '{target_column}' has only one class in this selection, "
                "so there is no variation to explain. Try widening the filters."
            ),
            n_rows=len(df),
        )

    numeric_features, categorical_features = candidate_features(df, target_column, reg)
    logger.info(
        "Driver analysis on target '%s': %d numeric and %d categorical candidate(s).",
        target_column,
        len(numeric_features),
        len(categorical_features),
    )

    correlations = correlation_drivers(df, target_column, numeric_features)
    numeric_comparisons = numeric_group_tests(df, target_column, numeric_features)
    categorical_results = categorical_tests(df, target_column, categorical_features)
    tree_results = (
        tree_importances(df, target_column, numeric_features) if include_tree else []
    )

    consensus = build_consensus(
        correlations, numeric_comparisons, categorical_results, tree_results
    )

    result = DriverAnalysisResult(
        target=target_column,
        outcome_rate_pct=float(target_values.mean() * 100.0),
        correlations=tuple(correlations),
        numeric_comparisons=tuple(numeric_comparisons),
        categorical_tests=tuple(categorical_results),
        tree_importances=tuple(tree_results),
        consensus=consensus,
        n_rows=int(len(df)),
    )
    logger.info(
        "Driver analysis complete: %d feature(s) in the consensus ranking.",
        len(consensus),
    )
    return result
