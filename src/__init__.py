"""
FinRisk AI - AI-Powered Financial Analytics & Credit Risk Intelligence Platform.

Layering rule for this package: nothing in `src` imports `streamlit` except
`pipeline.py`, which uses it only for optional caching decorators. Every other
module is a plain Python/pandas module so the analytics stay importable from
tests, notebooks and the consolidated submission build.

Dependency direction is one-way:

    UI  ->  pipeline  ->  analytics/intelligence  ->  data  ->  disk
"""

__version__ = "0.1.0"
__project_name__ = "FinRisk AI"
__subtitle__ = (
    "From Financial Data to Intelligent Insights, Risk Detection and Actionable Decisions"
)
