"""The report layer: a labelled Markdown report with an honesty section."""
from __future__ import annotations

from .report import CITATIONS, STANDING_CAVEATS, build_report

__all__ = ["build_report", "CITATIONS", "STANDING_CAVEATS"]
