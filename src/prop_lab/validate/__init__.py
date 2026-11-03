"""Out-of-sample validation layer for prop-lab.

Exposes the purged walk-forward validator: :func:`walk_forward`,
:func:`walk_forward_table` and the :class:`WFWindow` / :class:`WFResult`
containers, plus the edge-significance test in
:mod:`prop_lab.validate.significance`.
"""
from __future__ import annotations

from .significance import (
    DEFAULT_BOOTSTRAP_ATTEMPTS,
    MIN_TRADES_FOR_SIGNIFICANCE,
    EdgeTest,
    test_edge,
)
from .walkforward import (
    DEGRADATION_DEGRADES_MAX,
    DEGRADATION_HOLDS_MAX,
    MIN_OOS_TRADES_FOR_CONCLUSION,
    WFResult,
    WFWindow,
    walk_forward,
    walk_forward_table,
)

__all__ = [
    "walk_forward",
    "walk_forward_table",
    "WFWindow",
    "WFResult",
    "DEGRADATION_HOLDS_MAX",
    "DEGRADATION_DEGRADES_MAX",
    "MIN_OOS_TRADES_FOR_CONCLUSION",
    "EdgeTest",
    "test_edge",
    "MIN_TRADES_FOR_SIGNIFICANCE",
    "DEFAULT_BOOTSTRAP_ATTEMPTS",
]
