"""prop-lab: an agent-driven prop firm challenge lab.

The package answers one question honestly: given a trading strategy
backtested on real data, what is the probability of passing a prop firm
evaluation, and does the edge survive out of sample?

The parts:

``prop_lab.core``
    The backtest engine. One fill model, one source of truth.
``prop_lab.data``
    Free market data in one canonical schema.
``prop_lab.validate``
    Purged walk-forward out-of-sample validation.
``prop_lab.firm``
    Firm rulesets, the bootstrap Monte Carlo, and the funded phase.
``prop_lab.surface``
    Pass-rate surfaces where every axis is labelled measured or modelled.
``prop_lab.report``
    The Markdown report, with a mandatory honesty section.

Architecture lineage: the prop-risk-geometry project, originally
``github.com/hunter547/prop-risk-geometry`` and now
``github.com/dsilva42/prop-risk-geometry``, which built on the
pass-rate surface visualisation work DeltaTrend Trading showed publicly.
The purged walk-forward follows Lopez de Prado's "Advances in Financial
Machine Learning" chapters 7 and 12 and the conventions of
``github.com/eslazarev/purged-cross-validation``.
"""
from .core import (
    Bar,
    Signal,
    StrategyCore,
    Trade,
    iter_bars,
    run_backtest,
    summarize_trades,
)
from .firm import (
    APEX_50K_EOD,
    APEX_50K_INTRADAY,
    FIRM_PRESETS,
    RISK_FN_MAP,
    RISK_GEOMETRIES,
    TOPSTEP_50K_EOD,
    FirmRules,
    get_preset,
    list_presets,
    monte_carlo_challenge,
    monte_carlo_funded,
    simulate_challenge,
)
from .report import build_report
from .surface import (
    AXIS_MEASURED,
    AXIS_MODELLED,
    Surface,
    SurfaceSpec,
    risk_geometry_surface,
    rr_surface,
    surface_table,
    winrate_surface,
)
from .validate import WFResult, WFWindow, walk_forward, walk_forward_table

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Bar",
    "Signal",
    "StrategyCore",
    "Trade",
    "iter_bars",
    "run_backtest",
    "summarize_trades",
    "FirmRules",
    "APEX_50K_EOD",
    "APEX_50K_INTRADAY",
    "TOPSTEP_50K_EOD",
    "FIRM_PRESETS",
    "get_preset",
    "list_presets",
    "RISK_FN_MAP",
    "RISK_GEOMETRIES",
    "simulate_challenge",
    "monte_carlo_challenge",
    "monte_carlo_funded",
    "WFWindow",
    "WFResult",
    "walk_forward",
    "walk_forward_table",
    "AXIS_MEASURED",
    "AXIS_MODELLED",
    "Surface",
    "SurfaceSpec",
    "risk_geometry_surface",
    "rr_surface",
    "winrate_surface",
    "surface_table",
    "build_report",
]
