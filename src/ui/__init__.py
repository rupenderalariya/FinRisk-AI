"""
Presentation layer for FinRisk AI.

This is the only package permitted to import Streamlit. It contains no
analytical logic: every number it renders is computed by a module under ``src/``
and passed in. That separation is what keeps the analytics testable headlessly
and lets the dashboard be replaced without touching the engine.

Modules
-------
``theme``        Dark palette, centralised CSS, shared Plotly template.
``components``   Layout primitives: headers, cards, status pills, empty states.
``kpi_cards``    Compact KPI panels.
``insight_cards``Key intelligence, risk signals, opportunities, action chains.
``charts``       Dashboard chart builders composed over ``src.visualization``.
``navigation``   Custom grouped sidebar.
``filter_bar``   Filter widgets bound to :class:`src.filtering.FilterSpec`.
``pages``        One module per dashboard page.
"""

__all__ = [
    "theme",
    "components",
    "kpi_cards",
    "insight_cards",
    "charts",
    "navigation",
    "filter_bar",
    "pages",
]
