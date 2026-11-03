"""Prop firm challenge and funded-phase simulation.

Three modules:

``rules``
    The firm rulesets, as dataclasses, with the source and the date each
    was checked against the firm's published terms.
``monte_carlo``
    The bootstrap challenge simulation. The trade outcomes are resampled
    from the real backtest distribution, never invented by a coin flip.
``funded``
    What happens after the evaluation: the payout ladder, the consistency
    rule, and the funded-account lifetime.
"""
from .funded import FundedResult, FundedSummary, monte_carlo_funded, simulate_funded
from .monte_carlo import (
    MCSummary,
    RESAMPLING_BLOCK,
    RESAMPLING_IID,
    RESAMPLING_METHODS,
    RESAMPLING_STATIONARY,
    RISK_GEOMETRIES,
    RISK_FN_MAP,
    PathResult,
    bootstrap_day_blocks,
    bootstrap_stationary,
    bootstrap_trade_array,
    group_by_session,
    monte_carlo_challenge,
    simulate_challenge,
    wilson_interval,
)
from .rules import (
    APEX_50K_EOD,
    APEX_50K_INTRADAY,
    FIRM_PRESETS,
    TOPSTEP_50K_EOD,
    FirmRules,
    get_preset,
    list_presets,
)

__all__ = [
    "FirmRules",
    "APEX_50K_EOD",
    "APEX_50K_INTRADAY",
    "TOPSTEP_50K_EOD",
    "FIRM_PRESETS",
    "get_preset",
    "list_presets",
    "MCSummary",
    "PathResult",
    "RISK_FN_MAP",
    "RISK_GEOMETRIES",
    "RESAMPLING_IID",
    "RESAMPLING_BLOCK",
    "RESAMPLING_STATIONARY",
    "RESAMPLING_METHODS",
    "simulate_challenge",
    "monte_carlo_challenge",
    "wilson_interval",
    "bootstrap_trade_array",
    "bootstrap_day_blocks",
    "bootstrap_stationary",
    "group_by_session",
    "FundedResult",
    "FundedSummary",
    "simulate_funded",
    "monte_carlo_funded",
]
