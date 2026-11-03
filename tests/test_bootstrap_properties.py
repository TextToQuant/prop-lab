"""Property tests for the block bootstrap.

Two properties, both of which a block bootstrap must satisfy, and both of
which an i.i.d. bootstrap gets wrong in opposite directions:

1. **Block length 1 reduces exactly to the plain bootstrap.** With
   ``mean_block_length=1`` the stationary bootstrap should draw the same
   distribution as drawing trades independently with replacement. If it
   does not, the geometric block lengths are wrong.

2. **The block bootstrap is WIDER than the plain bootstrap on a
   serially correlated series.** Losses arrive in clusters, so the
   variance of a resampled mean must be larger when the clustering is
   preserved. If the block bootstrap were narrower than the i.i.d.
   bootstrap on a correlated series, it would be flattering the result,
   which is the exact failure the block bootstrap exists to prevent.

These are the two tests that make a block bootstrap trustworthy. They are
borrowed in spirit from the MaverickHQ/prop-challenge-lab harness
(Apache-2.0), which states the same two properties.
"""
from __future__ import annotations

import numpy as np
import pytest

from prop_lab.firm.monte_carlo import (
    bootstrap_stationary,
    bootstrap_trade_array,
)


def _mean_of_bootstrap_mean(pnl: np.ndarray, draws: int, seed: int) -> float:
    """The variance of the resampled mean, estimated by resampling."""
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    n = len(pnl)
    for i in range(draws):
        means[i] = pnl[rng.integers(0, n, n)].mean()
    return float(means.var(ddof=1))


def _mean_of_stationary_bootstrap_mean(
    pnl: np.ndarray, draws: int, seed: int, mean_block_length: float
) -> float:
    """The variance of the mean under the stationary bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(pnl)
    means = np.empty(draws)
    for i in range(draws):
        sample = bootstrap_stationary(
            pnl, rng, n_trades=n, mean_block_length=mean_block_length
        )
        means[i] = sample.mean()
    return float(means.var(ddof=1))


def test_stationary_bootstrap_with_block_length_one_reduces_to_the_plain_bootstrap():
    """Property 1: block length 1 IS the plain bootstrap.

    This is the property that catches a wrong geometric block-length
    parameterisation. If the block length is off by a factor, this fails.
    """
    rng = np.random.default_rng(3)
    pnl = rng.normal(0.0, 1.0, 500)

    plain = _mean_of_bootstrap_mean(pnl, draws=1500, seed=11)
    stationary = _mean_of_stationary_bootstrap_mean(
        pnl, draws=1500, seed=11, mean_block_length=1.0
    )

    # The two estimators must agree closely. Allow a generous tolerance
    # because both are Monte Carlo estimates of the same variance.
    assert stationary == pytest.approx(plain, rel=0.35), (
        f"block length 1 gave a materially different variance from the plain "
        f"bootstrap: {stationary:.6f} vs {plain:.6f}"
    )


def test_block_bootstrap_is_wider_than_iid_on_a_correlated_series():
    """Property 2: preserving clustering must INCREASE the variance.

    A series where losses arrive in streaks. The i.i.d. bootstrap
    destroys the streaks and understates the variance; the block bootstrap
    keeps them and must therefore be wider.

    If this fails with the block bootstrap narrower, the block bootstrap is
    flattering the strategy, which is the failure it exists to prevent.
    """
    # Build a series with strong positive autocorrelation: a slow random
    # walk, so consecutive values are correlated.
    rng = np.random.default_rng(5)
    steps = rng.normal(0.0, 1.0, 600)
    walk = np.cumsum(steps)
    pnl = walk - walk.mean()

    iid_var = _mean_of_bootstrap_mean(pnl, draws=1200, seed=21)
    block_var = _mean_of_stationary_bootstrap_mean(
        pnl, draws=1200, seed=21, mean_block_length=10.0
    )

    assert block_var > iid_var, (
        f"the block bootstrap was narrower ({block_var:.6f}) than the i.i.d. "
        f"bootstrap ({iid_var:.6f}) on a serially correlated series. It "
        "should be wider: preserving the clustering has to increase the "
        "variance of the resampled mean."
    )


def test_stationary_bootstrap_preserves_the_mean():
    """The resampler must not move the mean of the distribution."""
    rng = np.random.default_rng(7)
    pnl = rng.normal(0.2, 1.0, 400)
    for block in (1.0, 5.0, 20.0):
        sample = bootstrap_stationary(
            pnl, np.random.default_rng(1), n_trades=400, mean_block_length=block
        )
        assert sample.mean() == pytest.approx(pnl.mean(), abs=0.15)


def test_stationary_bootstrap_rejects_a_block_length_below_one():
    with pytest.raises(ValueError):
        bootstrap_stationary(
            np.array([1.0, -1.0]), np.random.default_rng(1), mean_block_length=0.5
        )


def test_plain_trade_bootstrap_is_reproducible_under_a_seed():
    """The plain bootstrap must be reproducible, or nothing else is."""
    pnl = np.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
    a = bootstrap_trade_array(pnl, np.random.default_rng(42), n_trades=50)
    b = bootstrap_trade_array(pnl, np.random.default_rng(42), n_trades=50)
    assert np.array_equal(a, b)
