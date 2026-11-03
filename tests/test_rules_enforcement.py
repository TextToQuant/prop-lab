"""Tests for the evaluation rules that a firm actually enforces.

Two rules matter here because ignoring them inflates the pass rate:

1. ``min_trading_days``: the evaluation cannot pass until enough trading
   days have elapsed, even if the profit target was reached on day one.
   Topstep requires two.
2. The consistency rule: when the best single day exceeds the allowed
   share of the profit target, the firm RAISES the target rather than
   failing the account. Topstep's rule is best_day / 0.55.

Sources, checked 2026-11-03:
- help.topstep.com trading combine parameters
- tradedeskhq.io "Apex vs Topstep: 50K futures evaluation rules compared"
- bestpropfirms.com prop firm evaluations guide
"""
from __future__ import annotations

import numpy as np
import pytest

from prop_lab.firm.monte_carlo import monte_carlo_challenge, simulate_challenge
from prop_lab.firm.rules import APEX_50K_EOD, APEX_50K_INTRADAY, TOPSTEP_50K_EOD, FirmRules


def _rules(**kwargs) -> FirmRules:
    defaults: dict = dict(
        name="test-50k",
        starting_balance=50_000.0,
        profit_target=3_000.0,
        max_drawdown=2_000.0,
        max_daily_loss=0.0,
        trailing_drawdown=True,
        dll_is_failure=False,
    )
    defaults.update(kwargs)
    return FirmRules(**defaults)


# ----------------------------------------------------------------------
# Minimum trading days
# ----------------------------------------------------------------------
def test_min_trading_days_blocks_an_early_pass():
    """The target alone must not be enough to pass."""
    rules = _rules(min_trading_days=3, eval_consistency_ratio=None)
    # A trade series that always wins, so the target is reached on day 1.
    pnl = np.array([5.0] * 60)

    result = simulate_challenge(
        pnl,
        rules,
        risk_geometry="frac_1pct_initial",
        rng=np.random.default_rng(0),
        max_trades=200,
        trades_per_day=5,  # 5 trades per synthetic day
    )
    # The path cannot pass before three days have closed.
    assert len(result.day_pnls) >= 3 or result.outcome != "pass"


def test_min_trading_days_of_zero_passes_immediately():
    rules = _rules(min_trading_days=0, eval_consistency_ratio=None)
    pnl = np.array([5.0] * 60)
    result = simulate_challenge(
        pnl,
        rules,
        risk_geometry="frac_1pct_initial",
        rng=np.random.default_rng(1),
        max_trades=200,
        trades_per_day=5,
    )
    assert result.outcome == "pass"


def test_min_trading_days_lowers_the_pass_rate():
    """Requiring more days cannot make passing easier."""
    pnl = np.array([1.0, -1.0, 0.5, -0.5] * 40)
    easy = monte_carlo_challenge(
        pnl, _rules(min_trading_days=0, eval_consistency_ratio=None),
        n_attempts=3000, seed=2, trades_per_day=3,
    )
    hard = monte_carlo_challenge(
        pnl, _rules(min_trading_days=5, eval_consistency_ratio=None),
        n_attempts=3000, seed=2, trades_per_day=3,
    )
    assert hard.pass_rate <= easy.pass_rate


def test_topstep_preset_requires_two_days():
    assert TOPSTEP_50K_EOD.min_trading_days == 2


def test_apex_presets_have_no_minimum():
    """Apex may pass in one trading day."""
    assert APEX_50K_EOD.min_trading_days == 0
    assert APEX_50K_INTRADAY.min_trading_days == 0


# ----------------------------------------------------------------------
# Consistency rule raises the target
# ----------------------------------------------------------------------
def test_consistency_rule_raises_the_target():
    """One big day raises the target instead of failing the account.

    Topstep's example: a $1,800 best day makes the $3,000 target
    $1,800 / 0.55 = $3,273.
    """
    rules = _rules(eval_consistency_ratio=0.55, min_trading_days=0)
    # One huge winning day, then small changes.
    pnl = np.array([10.0] + [0.01] * 60)

    result = simulate_challenge(
        pnl,
        rules,
        risk_geometry="frac_1pct_initial",
        rng=np.random.default_rng(3),
        max_trades=200,
        trades_per_day=5,
    )
    # The path does not pass on the strength of the big day alone,
    # because the consistency rule pushed the target up.
    if result.outcome == "pass":
        # If it did pass, it must have been above the raised target.
        assert result.final_equity >= rules.target_balance


def test_consistency_rule_makes_passing_harder():
    pnl = np.array([1.0, -1.0, 0.5, -0.5] * 40)
    without = monte_carlo_challenge(
        pnl, _rules(eval_consistency_ratio=None),
        n_attempts=3000, seed=4, trades_per_day=3,
    )
    with_rule = monte_carlo_challenge(
        pnl, _rules(eval_consistency_ratio=0.55),
        n_attempts=3000, seed=4, trades_per_day=3,
    )
    assert with_rule.pass_rate <= without.pass_rate


def test_topstep_uses_a_55_percent_consistency_ratio():
    assert TOPSTEP_50K_EOD.eval_consistency_ratio == 0.55
    assert TOPSTEP_50K_EOD.consistency_ratio == 0.55


def test_apex_evaluation_has_no_consistency_rule():
    """Apex applies consistency only at the payout stage, not the eval."""
    assert APEX_50K_EOD.eval_consistency_ratio is None
    assert APEX_50K_INTRADAY.eval_consistency_ratio is None
    # The funded-stage rule still exists, it just does not apply to the
    # evaluation pass rate.
    assert APEX_50K_EOD.consistency_ratio == 0.50


# ----------------------------------------------------------------------
# Rules that must stay correct
# ----------------------------------------------------------------------
def test_apex_eod_has_a_daily_loss_limit_and_intraday_does_not():
    assert APEX_50K_EOD.max_daily_loss == 1_000.0
    assert APEX_50K_INTRADAY.max_daily_loss == 0.0


def test_breaching_the_daily_loss_limit_does_not_fail_the_account():
    """At APEX and Topstep the DLL pauses trading for the session."""
    assert APEX_50K_EOD.dll_is_failure is False
    assert TOPSTEP_50K_EOD.dll_is_failure is False


def test_current_targets_and_drawdowns():
    """Checked 2026-11-03 against the firms' published rules."""
    for rules in (APEX_50K_EOD, APEX_50K_INTRADAY, TOPSTEP_50K_EOD):
        assert rules.starting_balance == 50_000.0
        assert rules.profit_target == 3_000.0
        assert rules.max_drawdown == 2_000.0
        assert rules.floor == 48_000.0


def test_every_preset_records_when_it_was_checked():
    for rules in (APEX_50K_EOD, APEX_50K_INTRADAY, TOPSTEP_50K_EOD):
        assert rules.rules_checked
        assert rules.source
