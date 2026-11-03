"""Golden test: validate the Monte Carlo engine against exact values.

The reference case is published in EdgeVeris's journal, "Estimating Your
Prop Challenge Pass Probability with Monte Carlo"
(https://edgeveris.com/blog/monte-carlo-prop-challenge-probability,
checked 2026-11-03).

The setup is deliberately simple so the answer can be computed exactly
with a dynamic program instead of by sampling:

    starting balance   $100,000
    profit target      $110,000
    failure boundary   $90,000
    risk per trade     $1,000 (1 percent of the INITIAL balance, fixed)
    win probability    55 percent
    horizon            100 trades
    daily limit        none
    fees               none

The exact dynamic program keeps the probability mass at every surviving
integer profit state from -9 to +9, moves 55 percent of each state's mass
one step up and 45 percent one step down at each trade, absorbs the mass
reaching +10 into "passed" and -10 into "failed", and after 100
iterations sums the remaining states for "unfinished".

Published exact values:

    Passed      66.784 percent
    Failed       8.978 percent
    Unfinished  24.238 percent

A sufficiently large Monte Carlo simulation of the same assumptions must
approach those values within its sampling error. If it does not, the
engine has a bug in its stopping order, its boundary equality, its
compounding or its handling of failed paths.

This file is the executable form of that check. It also verifies the
bookkeeping that the published numbers imply: the three probabilities
must sum to 1, and the pass rate among RESOLVED attempts is a different
and larger number than the pass rate over all attempts.
"""
from __future__ import annotations

import numpy as np
import pytest

from prop_lab.firm.monte_carlo import monte_carlo_challenge
from prop_lab.firm.rules import FirmRules

# Published exact values.
EXACT_PASS = 0.66784
EXACT_FAIL = 0.08978
EXACT_UNFINISHED = 0.24238

START = 100_000.0
TARGET = 110_000.0
FLOOR = 90_000.0
STEP = 1_000.0  # 1 percent of the initial balance, fixed
HORIZON = 100
WIN_P = 0.55


def _exact_boundary_probabilities() -> tuple[float, float, float]:
    """Compute the pass/fail/unfinished probabilities exactly.

    Uses the dynamic program from the published reference: track the
    probability mass at every reachable integer profit state and absorb
    the mass that reaches a boundary.

    Returns:
        A tuple of (passed, failed, unfinished) probabilities.
    """
    # States are integer multiples of STEP away from the starting balance.
    # -9..+9 survive; +10 is a pass and -10 is a fail.
    mass = {0: 1.0}
    passed = 0.0
    failed = 0.0

    for _ in range(HORIZON):
        nxt: dict[int, float] = {}
        for state, m in mass.items():
            nxt[state + 1] = nxt.get(state + 1, 0.0) + m * WIN_P
            nxt[state - 1] = nxt.get(state - 1, 0.0) + m * (1.0 - WIN_P)
        mass = {}
        for state, m in nxt.items():
            if state >= 10:
                passed += m
            elif state <= -10:
                failed += m
            else:
                mass[state] = mass.get(state, 0.0) + m
        if not mass:
            break

    unfinished = sum(mass.values())
    total = passed + failed + unfinished
    return passed / total, failed / total, unfinished / total


def _reference_rules() -> FirmRules:
    """A ruleset matching the published teaching example exactly."""
    return FirmRules(
        name="reference-100k",
        starting_balance=START,
        profit_target=TARGET - START,
        max_drawdown=START - FLOOR,
        max_daily_loss=0.0,  # deliberately omitted in the reference
        evaluation_days=365,
        min_trading_days=0,
        trailing_drawdown=False,  # a static floor, not a trailing one
        static_floor=FLOOR,
        flavor="EOD",
        dll_is_failure=False,
        rules_checked="2026-11-03",
        source="https://edgeveris.com/blog/monte-carlo-prop-challenge-probability",
    )


def _reference_trade_series() -> np.ndarray:
    """Build a trade series that produces a 55 percent win rate.

    The reference model has each trade add or subtract exactly $1,000,
    with a 55 percent chance of a win. Since the risk is fixed at 1
    percent of the initial balance, $1,000 corresponds to exactly 1R.

    The trade VALUES in the reference are +1R and -1R. To reproduce the
    55 percent win rate with a bootstrap, the sample must contain 55
    percent winners. 20 of 36 trades gives 55.56 percent; 55 of 100
    gives exactly 55 percent but the sample is large, so the bootstrap
    spread would be the dominant error.

    The engine is exercised with an i.i.d. bootstrap over a series of
    +1 and -1 with the right proportion, and the Monte Carlo result is
    compared against the exact values with a tolerance that covers the
    sampling error of the bootstrap.
    """
    # Exactly 55 winners and 45 losers in a shuffled order.
    rng = np.random.default_rng(20261103)
    values = np.array([1.0] * 55 + [-1.0] * 45)
    rng.shuffle(values)
    return values


def test_exact_dynamic_program_matches_published_values():
    """The reference dynamic program reproduces the published numbers."""
    passed, failed, unfinished = _exact_boundary_probabilities()
    assert passed == pytest.approx(EXACT_PASS, abs=0.0005)
    assert failed == pytest.approx(EXACT_FAIL, abs=0.0005)
    assert unfinished == pytest.approx(EXACT_UNFINISHED, abs=0.0005)
    assert passed + failed + unfinished == pytest.approx(1.0, abs=1e-9)


def test_monte_carlo_approaches_exact_values():
    """The Monte Carlo engine lands on the published exact values.

    The engine is run with the i.i.d. bootstrap over a 55 percent win
    rate series, fixed risk of 1 percent of the initial balance, no
    daily limit, a static floor and a 100 trade horizon.

    The tolerance is set by the sampling error of the Monte Carlo itself
    plus the fact that the bootstrap can only reproduce the 55 percent
    win rate to within the resolution of the 100 trade sample.
    """
    rules = _reference_rules()
    pnl = _reference_trade_series()

    summary = monte_carlo_challenge(
        pnl,
        rules,
        risk_geometry="frac_1pct_initial",
        n_attempts=40_000,
        seed=42,
        max_trades=HORIZON,
        trades_per_day=100,  # one synthetic day, so no daily limit bites
        max_days=1,
        resampling="iid",
    )

    # 40,000 attempts at p ~ 0.668 gives a standard error of about
    # sqrt(0.668 * 0.332 / 40000) = 0.0024, so 3 sigma is roughly
    # 0.007. The tolerance below is wider to absorb the bootstrap's own
    # resolution limit on the win rate.
    assert summary.pass_rate == pytest.approx(EXACT_PASS, abs=0.02)
    assert summary.fail_rate == pytest.approx(EXACT_FAIL, abs=0.02)
    assert summary.timeout_rate == pytest.approx(EXACT_UNFINISHED, abs=0.03)


def test_the_three_outcomes_sum_to_one():
    """Pass, fail and unfinished are a complete and exclusive partition.

    The published reference is explicit that discarding unfinished
    attempts silently changes the question. The engine must therefore
    report all three, and they must add up.
    """
    rules = _reference_rules()
    pnl = _reference_trade_series()
    summary = monte_carlo_challenge(
        pnl,
        rules,
        risk_geometry="frac_1pct_initial",
        n_attempts=5_000,
        seed=7,
        max_trades=HORIZON,
        trades_per_day=100,
        max_days=1,
        resampling="iid",
    )
    total = summary.pass_rate + summary.fail_rate + summary.timeout_rate
    assert total == pytest.approx(1.0, abs=1e-9)


def test_unfinished_attempts_are_not_silently_dropped():
    """A meaningful share of attempts must be reported as unfinished.

    The reference case has roughly a quarter of attempts unfinished, so
    an engine that returns zero unfinished attempts is broken.
    """
    rules = _reference_rules()
    pnl = _reference_trade_series()
    summary = monte_carlo_challenge(
        pnl,
        rules,
        risk_geometry="frac_1pct_initial",
        n_attempts=5_000,
        seed=11,
        max_trades=HORIZON,
        trades_per_day=100,
        max_days=1,
        resampling="iid",
    )
    assert summary.timeout_rate > 0.15
    assert summary.timeout_rate < 0.35


def test_pass_rate_among_resolved_exceeds_overall_pass_rate():
    """The resolved-attempt pass rate is larger than the overall rate.

    This is the arithmetic the reference warns about: "among resolved
    attempts the passing share is larger than 66.784 percent, but that
    is not the probability of passing within 100 trades".
    """
    rules = _reference_rules()
    pnl = _reference_trade_series()
    summary = monte_carlo_challenge(
        pnl,
        rules,
        risk_geometry="frac_1pct_initial",
        n_attempts=20_000,
        seed=13,
        max_trades=HORIZON,
        trades_per_day=100,
        max_days=1,
        resampling="iid",
    )
    resolved = summary.pass_rate + summary.fail_rate
    assert resolved > 0
    assert summary.pass_rate / resolved > summary.pass_rate
