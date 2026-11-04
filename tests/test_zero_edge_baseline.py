"""The zero-edge baseline is a property of the RULES and the BRACKET.

A pass rate is a comparison, so the comparison point has to be right. Two
things make it right:

1. **The bracket sets the hit rate.** With no information at all, the
   chance a trade reaches its target before its stop is
   ``P = stop / (stop + target) = 1 / (1 + rr)`` -- the gambler's-ruin
   result for two absorbing barriers. A 1:1 bracket wins 50 percent, a
   1:2 wins 33.3, a 1:5 wins 16.7. Comparing a 2R strategy against a 1:1
   baseline compares it with a different game, which is what this module
   used to do: it hard-coded +1R/-1R whatever the strategy.

2. **The baseline must contain no market data.** It is built from the
   bracket and the ruleset, so it describes the challenge rather than any
   particular instrument.

The analytic cross-check is optional stopping on a fair game: with a
STATIC floor at distance ``D`` and a target at distance ``T``,
``P(pass) = D / (D + T) = 2000 / 5000 = 40%``, independent of step size
provided the barriers absorb exactly. That is exact only for additive
(fixed-dollar) sizing: a percentage-of-equity rule makes the steps
multiplicative, which changes the answer. The test below therefore uses
``max_size``, whose flat $500 divides both barriers exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from prop_lab.firm import get_preset
from prop_lab.firm.monte_carlo import _zero_edge_bracket, zero_edge_baseline
from prop_lab.firm.rules import FirmRules

TARGET = 3_000.0
DRAW = 2_000.0


def _static_rules() -> FirmRules:
    """A static floor with no daily limit, so optional stopping applies."""
    return FirmRules(
        name="static floor, no daily limit",
        starting_balance=50_000.0,
        profit_target=TARGET,
        max_drawdown=DRAW,
        max_daily_loss=0.0,
        evaluation_days=0,
        trailing_drawdown=False,
        static_floor=50_000.0 - DRAW,
        flavor="EOD",
    )


@pytest.mark.parametrize("rr", [0.5, 1.0, 2.0, 5.0])
def test_bracket_win_fraction_is_the_yardstick(rr):
    """The zero-edge win rate is 1/(1+rr), not a flat 50 percent."""
    n = 6_000
    arr = _zero_edge_bracket(n, rr)
    assert float((arr > 0).mean()) == pytest.approx(1.0 / (1.0 + rr), abs=1.5 / n)


@pytest.mark.parametrize("rr", [0.5, 1.0, 2.0, 5.0])
def test_bracket_mean_is_zero_up_to_one_trade(rr):
    """Every bracket has zero expectancy; that is what "no edge" means."""
    n = 6_000
    arr = _zero_edge_bracket(n, rr)
    # Rounding can move the mean by at most half a trade's worth.
    assert arr.mean() == pytest.approx(0.0, abs=(1.0 + rr) / n)


@pytest.mark.parametrize("rr", [0.0, -1.0])
def test_bracket_rejects_a_non_positive_bracket(rr):
    with pytest.raises(ValueError, match="rr must be positive"):
        _zero_edge_bracket(100, rr)


def test_static_floor_matches_gamblers_ruin():
    """Optional stopping: P(pass) = floor / (floor + target) = 40%.

    Only meaningful for additive sizing: ``max_size`` risks a flat $500,
    and 500 divides both 2000 and 3000 exactly, so the walk is absorbed at
    the barrier rather than overshooting it. Measured 38.9% at 20,000
    attempts; the band allows for Monte Carlo error and the small
    overshoot that remains.
    """
    value = zero_edge_baseline(
        _static_rules(),
        n_trades=2_000,
        rr=1.0,
        risk_geometry="max_size",
        n_attempts=6_000,
        seed=42,
    )
    analytic = DRAW / (DRAW + TARGET)
    assert analytic == pytest.approx(0.40)
    assert 0.36 <= value <= 0.44, f"static-floor baseline {value:.1%} is not near 40%"


def test_apex_eod_baseline_is_still_about_a_quarter():
    """The documented reference number must not drift.

    This is the 1:1-bracket figure the README and AGENTS.md quote, so it
    is also the regression that would catch a change to the sizing rules
    or to the challenge loop.
    """
    value = zero_edge_baseline(
        get_preset("apex-50k-eod"),
        n_trades=2_000,
        rr=1.0,
        n_attempts=6_000,
        seed=42,
    )
    assert 0.24 <= value <= 0.31, f"APEX 50K EOD baseline {value:.1%} is not about 27%"


def test_bracket_parameter_actually_changes_the_simulation():
    """A 2R bracket is a different challenge, so the call must not ignore rr.

    The two baselines are close (25.6% vs 26.8% measured) because a fair
    game stays a fair game, but the input must reach the simulation.
    """
    rules = get_preset("apex-50k-eod")
    one = zero_edge_baseline(rules, n_trades=2_000, rr=1.0, n_attempts=2_000, seed=7)
    two = zero_edge_baseline(rules, n_trades=2_000, rr=2.0, n_attempts=2_000, seed=7)
    assert 0.0 < one < 1.0
    assert 0.0 < two < 1.0
    assert one != two
