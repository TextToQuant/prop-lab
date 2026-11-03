"""Tests for the resampling methods and the risk geometries.

The resampling methods are the part of the Monte Carlo that decides
whether the pass rate is honest, so each one gets a direct test.

References:
- Politis and Romano (1994), "The Stationary Bootstrap", Journal of
  Economic Perspectives 8(4), 143-158.
- EdgeVeris, "Estimating Your Prop Challenge Pass Probability with Monte
  Carlo", https://edgeveris.com/blog/monte-carlo-prop-challenge-probability
  (checked 2026-11-03).
"""
from __future__ import annotations

import numpy as np
import pytest

from prop_lab.firm.monte_carlo import (
    RESAMPLING_BLOCK,
    RESAMPLING_IID,
    RESAMPLING_STATIONARY,
    bootstrap_day_blocks,
    bootstrap_stationary,
    bootstrap_trade_array,
    group_by_session,
    monte_carlo_challenge,
    risk_frac_1pct_initial,
    risk_frac_2pct,
    risk_frac_2pct_initial,
)
from prop_lab.firm.rules import FirmRules


def _rules(**kwargs) -> FirmRules:
    defaults: dict = dict(
        name="test-50k",
        starting_balance=50_000.0,
        profit_target=5_000.0,
        max_drawdown=2_500.0,
        max_daily_loss=0.0,
        trailing_drawdown=True,
        dll_is_failure=False,
    )
    defaults.update(kwargs)
    return FirmRules(**defaults)


# ----------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------
def test_iid_bootstrap_preserves_the_marginal_distribution():
    values = np.array([1.0, 2.0, -1.0, -0.5, 0.25])
    rng = np.random.default_rng(0)
    draws = bootstrap_trade_array(values, rng, n_trades=50_000)
    assert set(np.unique(draws)).issubset(set(values))
    assert draws.mean() == pytest.approx(values.mean(), abs=0.03)


def test_stationary_bootstrap_returns_the_requested_length():
    values = np.array([1.0, -1.0, 2.0, -2.0, 0.5])
    rng = np.random.default_rng(1)
    draws = bootstrap_stationary(values, rng, n_trades=123)
    assert draws.size == 123


def test_stationary_bootstrap_rejects_a_block_length_below_one():
    values = np.array([1.0, -1.0])
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        bootstrap_stationary(values, rng, n_trades=10, mean_block_length=0.5)


def test_stationary_bootstrap_rejects_an_empty_series():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        bootstrap_stationary(np.array([]), rng)


def test_stationary_bootstrap_preserves_dependence_structure():
    """The whole point: the resample must keep the series' dependence.

    Comparing longest run lengths is not the right test, because an
    i.i.d. bootstrap of a series with a high loss rate produces long
    runs by chance. The sharper property is whether the resample keeps
    the OBSERVED adjacency structure.

    Take a strictly alternating series: wins and losses never sit next
    to each other. An i.i.d. bootstrap of a 50 percent win rate series
    produces adjacent losses about a quarter of the time, because it
    knows nothing about adjacency. A block bootstrap draws contiguous
    chunks of the original, so the alternation largely survives.
    """
    values = np.array([1.0, -1.0] * 30)
    rng = np.random.default_rng(4)

    stationary = bootstrap_stationary(values, rng, n_trades=4000, mean_block_length=12)
    iid = bootstrap_trade_array(values, rng, n_trades=4000)

    def adjacent_loss_fraction(series: np.ndarray) -> float:
        losses = series < 0
        if losses.size < 2:
            return 0.0
        return float(np.mean(losses[:-1] & losses[1:]))

    # The original never has two losses in a row.
    assert adjacent_loss_fraction(values) == 0.0

    stationary_fraction = adjacent_loss_fraction(stationary)
    iid_fraction = adjacent_loss_fraction(iid)

    # The i.i.d. bootstrap invents adjacent losses at roughly the rate
    # independence predicts, a quarter of the time.
    assert iid_fraction == pytest.approx(0.25, abs=0.03)

    # The block bootstrap keeps them far rarer than that.
    assert stationary_fraction < iid_fraction / 2
    assert stationary_fraction < 0.10


def test_stationary_bootstrap_preserves_the_marginal_distribution():
    rng = np.random.default_rng(5)
    values = np.array([2.0, -1.0, -1.0, 0.5, 1.5, -0.5])
    draws = bootstrap_stationary(values, rng, n_trades=40_000, mean_block_length=6)
    assert draws.mean() == pytest.approx(values.mean(), abs=0.05)


def test_block_bootstrap_reuses_whole_days():
    days = [np.array([1.0, -1.0]), np.array([2.0, 2.0]), np.array([-1.0, -1.0])]
    rng = np.random.default_rng(6)
    draws = bootstrap_day_blocks(days, rng, n_days=200)
    assert draws.size == 400
    # Every value comes from one of the original days.
    allowed = set()
    for day in days:
        allowed.update(day.tolist())
    assert set(np.unique(draws)).issubset(allowed)


def test_group_by_session_splits_on_the_labels():
    values = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
    blocks = group_by_session(values, ["a", "a", "b", "b", "c"])
    assert len(blocks) == 3
    assert blocks[0].tolist() == [1.0, -1.0]
    assert blocks[1].tolist() == [2.0, -2.0]
    assert blocks[2].tolist() == [3.0]


def test_group_by_session_without_labels_gives_one_block():
    values = np.array([1.0, -1.0, 2.0])
    blocks = group_by_session(values, None)
    assert len(blocks) == 1
    assert blocks[0].size == 3


# ----------------------------------------------------------------------
# Risk geometries
# ----------------------------------------------------------------------
def test_initial_balance_geometries_do_not_compound():
    rules = _rules()
    assert risk_frac_1pct_initial(50_000.0, 50_000.0, 0.0, rules) == 500.0
    # The risk stays the same as the equity moves.
    assert risk_frac_1pct_initial(60_000.0, 60_000.0, 0.0, rules) == 500.0
    assert risk_frac_2pct_initial(60_000.0, 60_000.0, 0.0, rules) == 1_000.0


def test_current_equity_geometry_compounds():
    rules = _rules()
    assert risk_frac_2pct(50_000.0, 50_000.0, 0.0, rules) == 1_000.0
    assert risk_frac_2pct(60_000.0, 60_000.0, 0.0, rules) == 1_200.0


def test_the_two_bases_give_different_pass_rates():
    """Fixed-cash risk and current-equity risk are not interchangeable.

    The published reference makes this point explicitly: a report must
    never describe both bases with the same number without saying which
    one it used.
    """
    rng = np.random.default_rng(7)
    pnl = rng.normal(0.1, 1.0, size=200)
    rules = _rules()

    compounding = monte_carlo_challenge(
        pnl, rules, risk_geometry="frac_2pct", n_attempts=3000, seed=1
    )
    fixed = monte_carlo_challenge(
        pnl, rules, risk_geometry="frac_2pct_initial", n_attempts=3000, seed=1
    )
    assert compounding.pass_rate != fixed.pass_rate


# ----------------------------------------------------------------------
# Integration
# ----------------------------------------------------------------------
def test_all_three_resampling_methods_run():
    pnl = np.array([1.0, -1.0] * 30)
    sessions = [f"2026-01-{d:02d}" for d in range(1, 21) for _ in range(3)]
    for method in (RESAMPLING_IID, RESAMPLING_BLOCK, RESAMPLING_STATIONARY):
        summary = monte_carlo_challenge(
            pnl,
            _rules(),
            n_attempts=200,
            seed=3,
            resampling=method,
            session_dates=sessions,
        )
        assert 0.0 <= summary.pass_rate <= 1.0
        assert summary.n_attempts == 200


def test_stationary_reports_its_block_length():
    pnl = np.array([1.0, -1.0] * 30)
    summary = monte_carlo_challenge(
        pnl,
        _rules(),
        n_attempts=100,
        seed=4,
        resampling=RESAMPLING_STATIONARY,
        mean_block_length=7.0,
    )
    assert any("mean block length 7.0" in w for w in summary.warnings)


def test_stationary_does_not_require_session_labels():
    """The stationary bootstrap works on the trade series alone."""
    pnl = np.array([1.0, -1.0] * 30)
    summary = monte_carlo_challenge(
        pnl,
        _rules(),
        n_attempts=100,
        seed=5,
        resampling=RESAMPLING_STATIONARY,
    )
    assert summary.n_attempts == 100
