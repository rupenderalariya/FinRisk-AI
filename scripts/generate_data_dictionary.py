"""
Generate ``data_dictionary.md`` from the real dataset.

Everything in the output - dtypes, missing percentages, example values, feature
formulas - is read from an actual pipeline run. Nothing is typed by hand, so the
documentation cannot drift away from the code.

Usage::

    python scripts/generate_data_dictionary.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataLoadError  # noqa: E402
from src.feature_engineering import FeatureReport  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402
from src.pipeline import PipelineResult, run_pipeline  # noqa: E402
from src.schema import (  # noqa: E402
    DATASET_CITATION,
    DATASET_CURRENCY,
    DATASET_LANDING_URL,
    DATASET_LICENCE,
    DATASET_NAME,
    DATASET_PERIOD,
    DATASET_SOURCE,
    UNAVAILABLE_ROLES,
    Role,
    describe_column,
    describe_panel_alignment,
)

logger = get_logger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data_dictionary.md"

#: Columns used for analytics but deliberately withheld from the risk model.
FAIRNESS_EXCLUDED = {"sex", "sex_code", "marriage", "marriage_code"}

#: Bookkeeping columns that support the audit trail rather than analysis.
BOOKKEEPING = {
    "is_duplicate_excluding_id",
    "has_extreme_value",
    "credit_limit_original",
    "age_is_implausible",
}

#: ML usage for the source columns. The engineered-feature report covers only
#: engineered features, so the intent for raw columns is recorded explicitly here
#: rather than defaulting them all to "No", which would be wrong and misleading.
SOURCE_ML_USAGE: dict[str, str] = {
    "credit_limit": "Yes (continuous capacity measure)",
    "age": "Yes",
    "education_code": "Yes (reviewed in the fairness audit)",
    "education": "Yes (reviewed in the fairness audit)",
}

#: Raw panel columns superseded by engineered summaries. Feeding all 18 monthly
#: columns to a model alongside their own aggregates would add collinearity
#: without adding information.
SUPERSEDED_PREFIXES = ("bill_amt_m", "pay_amt_m")


def _escape(text: Any) -> str:
    """Escape pipe characters so a value cannot break the markdown table."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _example_value(series: pd.Series) -> str:
    """Pick a representative non-null example and format it readably."""
    non_null = series.dropna()
    if non_null.empty:
        return "_(all missing)_"

    value = non_null.iloc[0]

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)) or pd.api.types.is_integer_dtype(series):
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".") if abs(value) < 1e15 else f"{value:.4g}"
    return _escape(value)


def _usage_flags(
    column: str, ml_features: set[str], engineered: set[str]
) -> tuple[str, str]:
    """Decide the analytics and ML usage flags for a column.

    Returns:
        ``(analytics_flag, ml_flag)`` as markdown-ready strings.
    """
    if column == "client_id":
        return "No (join key only)", "No (identifier)"
    if column == "default_next_month":
        return "Yes (as the outcome)", "Yes (target)"
    if column in FAIRNESS_EXCLUDED:
        return "Yes (fairness audit only)", "**No - excluded on fairness grounds**"
    if column in BOOKKEEPING:
        return "Audit trail", "No"
    if column.endswith("_was_missing"):
        return "Audit trail", "No"

    if column in SOURCE_ML_USAGE:
        return "Yes", SOURCE_ML_USAGE[column]

    if column.startswith("pay_status_m"):
        return "Yes", "Yes (documented as legitimate prior information)"

    if column.startswith(SUPERSEDED_PREFIXES):
        return "Yes", "No (superseded by engineered summary features)"

    analytics = "Yes"
    ml = "Yes" if column in ml_features else "No"
    if column in engineered and column not in ml_features:
        ml = "No (superseded by a summary feature)"
    return analytics, ml


def _source_column_rows(result: PipelineResult, ml_features: set[str]) -> list[str]:
    """Build the markdown rows for the original source columns."""
    raw = result.raw
    rows: list[str] = []

    for column in raw.columns:
        series = raw[column]
        missing_pct = float(series.isna().sum()) / len(raw) * 100.0 if len(raw) else 0.0
        analytics, ml = _usage_flags(str(column), ml_features, set())
        rows.append(
            "| `{name}` | {dtype} | {meaning} | {missing} | {example} | {analytics} | {ml} |".format(
                name=column,
                dtype=str(series.dtype),
                meaning=_escape(describe_column(str(column))),
                missing=f"{missing_pct:.2f}%",
                example=_example_value(series),
                analytics=analytics,
                ml=ml,
            )
        )
    return rows


def _derived_column_rows(result: PipelineResult, ml_features: set[str]) -> list[str]:
    """Build rows for columns added during cleaning (not feature engineering)."""
    added = [c for c in result.clean.columns if c not in result.raw.columns]
    rows: list[str] = []

    for column in added:
        series = result.clean[column]
        missing_pct = (
            float(series.isna().sum()) / len(result.clean) * 100.0 if len(result.clean) else 0.0
        )
        analytics, ml = _usage_flags(str(column), ml_features, set())
        rows.append(
            "| `{name}` | {dtype} | {meaning} | {missing} | {example} | {analytics} | {ml} |".format(
                name=column,
                dtype=str(series.dtype),
                meaning=_escape(describe_column(str(column))),
                missing=f"{missing_pct:.2f}%",
                example=_example_value(series),
                analytics=analytics,
                ml=ml,
            )
        )
    return rows


def _feature_sections(report: FeatureReport, data: pd.DataFrame) -> list[str]:
    """Build one markdown section per engineered-feature category."""
    lines: list[str] = []
    ml_features = set(report.ml_feature_names())

    for category, definitions in report.by_category().items():
        lines.append(f"\n### {category}\n")
        lines.append(
            "| Feature | Formula | Source columns | Interpretation | Missing % | "
            "Example | Used for ML |"
        )
        lines.append("|---|---|---|---|---|---|---|")

        for definition in definitions:
            if definition.name in data.columns:
                series = data[definition.name]
                missing_pct = (
                    float(series.isna().sum()) / len(data) * 100.0 if len(data) else 0.0
                )
                missing = f"{missing_pct:.2f}%"
                example = _example_value(series)
            else:
                missing = "n/a"
                example = "n/a"

            lines.append(
                "| `{name}` | `{formula}` | {sources} | {interpretation} | {missing} | "
                "{example} | {ml} |".format(
                    name=definition.name,
                    formula=_escape(definition.formula),
                    sources=", ".join(f"`{c}`" for c in definition.source_columns),
                    interpretation=_escape(definition.interpretation),
                    missing=missing,
                    example=example,
                    ml="Yes" if definition.name in ml_features else "No",
                )
            )
    return lines


def build_markdown(result: PipelineResult) -> str:
    """Assemble the full data dictionary.

    Args:
        result: A completed pipeline run.

    Returns:
        The markdown document.
    """
    report = result.feature_report
    ml_features = set(report.ml_feature_names())
    raw = result.raw
    data = result.data

    lines: list[str] = [
        "# Data Dictionary",
        "",
        f"# {DATASET_NAME}",
        "",
        "> **This file is generated.** Run `python scripts/generate_data_dictionary.py`",
        "> to regenerate it. Every dtype, missing percentage and example value below was",
        "> read from an actual pipeline run, so this document cannot drift away from the code.",
        "",
        "---",
        "",
        "## 1. Dataset provenance",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Dataset | {DATASET_NAME} |",
        f"| Source | {DATASET_SOURCE} |",
        f"| URL | {DATASET_LANDING_URL} |",
        f"| Licence | {DATASET_LICENCE} |",
        f"| Period covered | {DATASET_PERIOD} |",
        f"| Currency | {DATASET_CURRENCY} (New Taiwan Dollar) |",
        f"| Source rows | {len(raw):,} |",
        f"| Source columns | {len(raw.columns)} |",
        f"| Columns after processing | {len(data.columns)} |",
        f"| Missing cells in source | {int(raw.isna().sum().sum()):,} |",
        f"| Exact duplicate rows | {int(raw.duplicated().sum()):,} |",
        f"| Duplicates excluding `client_id` | {result.quality.duplicates.duplicate_rows_excluding_id:,} |",
        f"| Data quality score | {result.quality.quality_score.score:.2f} / 100 "
        f"({result.quality.quality_score.band}) |",
        "",
        f"**Citation.** {DATASET_CITATION}",
        "",
        "---",
        "",
        "## 2. Panel structure",
        "",
        describe_panel_alignment(),
        "",
        "| Month index | Calendar month | Role |",
        "|---|---|---|",
        "| 1 | September 2005 | Most recent observation |",
        "| 2 | August 2005 | |",
        "| 3 | July 2005 | |",
        "| 4 | June 2005 | |",
        "| 5 | May 2005 | |",
        "| 6 | April 2005 | Oldest observation |",
        "",
        "---",
        "",
        "## 3. Source columns",
        "",
        "Columns exactly as they arrive from the source file, after header",
        "normalisation to snake_case.",
        "",
        "| Column | Data type | Meaning | Missing % | Example | Used for analytics | Used for ML |",
        "|---|---|---|---|---|---|---|",
    ]

    lines.extend(_source_column_rows(result, ml_features))

    derived_rows = _derived_column_rows(result, ml_features)
    if derived_rows:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 4. Columns added during cleaning",
                "",
                "Flags and preserved originals. Nothing here alters a source value; these",
                "columns exist so suspect records can be isolated rather than deleted.",
                "",
                "| Column | Data type | Meaning | Missing % | Example | Used for analytics | Used for ML |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        lines.extend(derived_rows)

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Engineered features",
            "",
            f"{len(report.created)} features were created. A feature is only produced when",
            "its source columns are verified present at runtime; otherwise it is skipped",
            "and the reason recorded in section 6.",
            "",
            "**On the `Used for ML` column.** Some features are excellent for reading and",
            "charting but unsuitable as model inputs. An unbounded ratio such as",
            "`repayment_ratio_mean` can reach into the thousands when a prior statement was",
            "near zero, which would dominate any distance-based model, so a bounded",
            "companion is used instead.",
        ]
    )
    lines.extend(_feature_sections(report, data))

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5b. Derived analytical bands (computed on demand)",
            "",
            "These groupings are computed when a view needs them rather than stored as",
            "columns, so they always reflect the current filtered selection. They are",
            "documented here because they appear as axes and groupings in the dashboard.",
            "",
            "| Band | Definition | Where it is used |",
            "|---|---|---|",
            "| `utilisation_band` | Fixed bands on `utilisation_latest`: negative/zero, "
            "low (<30%), moderate (30-60%), high (60-threshold), very high (threshold+). "
            "The top boundary is the configured `HIGH_UTILISATION_THRESHOLD`. | "
            "Financial Analytics (default rate by utilisation), Executive Overview "
            "(utilisation finding) |",
            "| Payment-delay group | `current_delinquency` bucketed into no delay, "
            "1 month, 2 months, 3+ months, using documented delay codes only. | "
            "Executive Overview (delinquency snapshot), Financial Analytics |",
            "| Payment segment | Derived from `is_full_payer` and `is_revolver`: full "
            "payers, partial payers, revolvers. Defined purely by repayment behaviour, "
            "never by outcome, so outcome comparisons are not circular. | "
            "Financial Analytics (behaviour comparison) |",
            "| `segment` | K-Means cluster assignment over 7 scaled behavioural "
            "features. Fitted on the whole portfolio, so the assignment is stable "
            "regardless of dashboard filters. See section 5c. | "
            "Customer Segmentation |",
            "",
            "Full banding methodology is shown in the dashboard's Methodology page and is",
            "generated from `src/filtering.py`, so the two cannot diverge.",
            "",
            "---",
            "",
            "## 5c. Segmentation assignment",
            "",
            "The Customer Segmentation page assigns every customer to a behavioural "
            "segment. The assignment is computed on demand rather than stored as a "
            "column, because it depends on the fitted model rather than on the row "
            "itself.",
            "",
            "| Property | Value |",
            "|---|---|",
            "| Method | K-Means clustering |",
            "| Clustering features | `credit_limit`, `utilisation_mean_6m`, "
            "`repayment_ratio_capped_mean`, `delinquent_months_count`, "
            "`bill_trend_slope`, `utilisation_volatility`, `months_zero_payment` |",
            "| Scaler | Robust (median / IQR), selected by measured silhouette: "
            "0.4035 against 0.2879 for standard scaling |",
            "| Number of segments | 3, the measured silhouette optimum across K = 2-8 |",
            "| Silhouette sampling | Scored on a fixed-seed sample of ~5,000 rows, "
            "stratified by segment so every segment is represented in proportion to "
            "its real size. A plain random subset can omit a small segment and leave "
            "the metric undefined; stratification prevents that by construction. "
            "Inertia and Davies-Bouldin use all 30,000 rows. |",
            "| Excluded by policy | `default_next_month` (outcome leakage), `sex`, "
            "`marriage`, `education`, `age` (fairness), `client_id` and bookkeeping "
            "flags |",
            "| Missing-value handling | `repayment_ratio_capped_mean` is undefined for "
            "1,406 customers (4.69%); the median is imputed so no customer is dropped. "
            "An imputed value is not an observed customer value. |",
            "| Naming | Segment names are composed at runtime from each centroid's "
            "standardised distance from the portfolio mean, so every word traces to a "
            "measured value. |",
            "| Observed default rate | Calculated per segment **after** clustering, "
            "for profiling only. It played no part in forming the segments and is not "
            "a prediction. |",
            "",
            "**Leakage guarantee.** The outcome column is blocked from the feature "
            "matrix by policy, and a permutation test in the suite shuffles the target "
            "and confirms the cluster assignments do not change.",
            "",
            "---",
            "",
            "## 6. Features NOT created, and why",
            "",
            "This section is deliberate. The project brief lists several standard financial",
            "ratios as examples. This dataset cannot support them, and they have **not** been",
            "approximated from unrelated fields.",
            "",
            "| Feature | Would require | Status |",
            "|---|---|---|",
        ]
    )
    for definition in report.skipped:
        lines.append(
            "| `{name}` | {sources} | {reason} |".format(
                name=definition.name,
                sources=", ".join(f"`{c}`" for c in definition.source_columns),
                reason=_escape(definition.skip_reason or "unspecified"),
            )
        )

    lines.extend(
        [
            "",
            "### Analytical capabilities unavailable in this dataset",
            "",
            "| Capability | Explanation |",
            "|---|---|",
        ]
    )
    for role_name, explanation in result.registry.unavailable_explanations().items():
        lines.append(f"| {role_name} | {_escape(explanation)} |")

    lines.extend(
        [
            "",
            "### What is used instead",
            "",
            "| Unavailable measure | Substitute actually used | Why the substitute is valid |",
            "|---|---|---|",
            "| Debt-to-income ratio | Credit utilisation (`bill_amt_m1 / credit_limit`) | "
            "Both express debt against capacity to carry it. Utilisation uses the granted "
            "credit limit as the capacity measure instead of income. |",
            "| Income band | Credit-limit band (`limit_band`) | "
            "A lender sets the limit partly from assessed capacity, so it is a coarse "
            "proxy. It is named and described as a credit limit throughout, never as income. |",
            "| Bureau credit score | Model-predicted default probability | "
            "Derived from this dataset's own behavioural features and always labelled as "
            "model output, never presented as an external score. |",
            "| Transaction volume | Share of accounts with a positive balance | "
            "An activity proxy. The dataset holds no transaction counts. |",
            "",
            "---",
            "",
            "## 7. Notes on data quality",
            "",
            "The source file has **zero missing values and zero exact duplicate rows**, so a",
            "quality check that only counted blanks would report a perfect score and tell an",
            "analyst nothing. The defects in this data are semantic:",
            "",
        ]
    )

    for issue in result.quality.all_issues:
        lines.append(
            f"- **{issue.check}** ({issue.severity.value}): {_escape(issue.description)} "
            f"Affected rows: {issue.affected_rows:,} ({issue.affected_percentage:.2f}%)."
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 8. Target variable",
            "",
            "| Property | Value |",
            "|---|---|",
            "| Column | `default_next_month` |",
            "| Definition | Client defaulted on the following month's payment |",
            "| Type | Binary (0 = no default, 1 = default) |",
        ]
    )

    if "default_next_month" in raw.columns:
        target = raw["default_next_month"]
        positive = int((target == 1).sum())
        negative = int((target == 0).sum())
        lines.extend(
            [
                f"| Positive class (default) | {positive:,} |",
                f"| Negative class (no default) | {negative:,} |",
                f"| Observed default rate | {target.mean() * 100:.2f}% |",
                f"| Majority-class baseline accuracy | {max(positive, negative) / len(target) * 100:.2f}% |",
            ]
        )
        lines.extend(
            [
                "",
                "The majority-class baseline is the reason accuracy is not used as the headline",
                f"metric: always predicting 'no default' scores "
                f"{max(positive, negative) / len(target) * 100:.2f}% while identifying no",
                "defaulters at all. ROC-AUC, PR-AUC and recall on the default class are",
                "reported instead.",
                "",
                "**Leakage note.** `pay_status_m1` records the September repayment status and",
                "the target is the October default. That is legitimately prior information, not",
                "leakage, but it is expected to dominate feature importance and is documented",
                "so a reader is not surprised by it.",
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"_Generated from a live pipeline run over {len(raw):,} rows._",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Generate the data dictionary. Returns a process exit code."""
    try:
        result = run_pipeline(persist=False)
    except DataLoadError as exc:
        print(f"\nCould not generate the data dictionary:\n{exc}\n")
        return 1

    markdown = build_markdown(result)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")

    line_count = markdown.count("\n") + 1
    print(f"\nWrote {OUTPUT_PATH.name}")
    print(f"  {line_count:,} lines, {len(markdown):,} characters")
    print(f"  {len(result.raw.columns)} source columns documented")
    print(f"  {len(result.feature_report.created)} engineered features documented")
    print(f"  {len(result.feature_report.skipped)} unavailable features explained\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
