"""
Dashboard pages.

One module per page, each exposing a single ``render(context)`` function. The
router in ``app.py`` picks one and calls it; pages never import each other, so a
page can be replaced or added without touching the rest.
"""

from src.ui.pages import (
    data_explorer,
    data_quality,
    executive,
    financial,
    methodology,
    placeholders,
    segmentation,
)

__all__ = [
    "executive",
    "financial",
    "segmentation",
    "data_explorer",
    "data_quality",
    "methodology",
    "placeholders",
]
