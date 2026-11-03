"""Tests for the edge-significance layer.

The rule this project learned the hard way: a regression test must exercise the
shipped code path, not a helper it happens to call. These tests drive
``walk_forward`` -- the code the CLI actually runs -- and assert on the
verdict and the warning it produces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prop_lab.core import Bar, Signal, StrategyCore
from prop_lab.validate import (
    MIN_TRADES_FOR_SIGNIFICANCE,
    EdgeTest,
    walk_forward,
)
from prop_lab.validate.significance import EdgeTest as DirectEdgeTest
from prop_lab.validate.significance import test_edge as run_edge_test


def _frame(n_bars: int = 4000) -> pd.DataFrame:
    """A deterministic synthetic bar frame with no directional drift.

    The close wanders around its starting value, so a strategy with no
    directional information produces R-multiples indistinguishable from
    zero. That is the case the verdict must not call ``holds``.
    """
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, n_bars))
    return _frame_from(close)


def _trending_frame(n_bars: int = 4000, drift: float = 0.35) -> pd.DataFrame:
    """A synthetic bar frame with a strong upward drift.

    A long-only strategy with a tight stop and a 2R target wins almost
    every trade on this frame, which is what a genuine, detectable edge
    looks like.
    """
    rng = np.random.default_rng(13)
    close = 100.0 + drift * np.arange(n_bars) + np.cumsum(rng.normal(0.0, 0.2, n_bars))
    return _frame_from(close)


def _frame_from(close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": np.arange(len(close), dtype="int64") * 900 + 1_600_000_000,
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(len(close), 100.0),
        }
    )


class _CoinFlip(StrategyCore):
    """Emits a signal every bar with no directional information.

    The R-multiples end up indistinguishable from zero, which is exactly
    the case the verdict must not call ``holds``.
    """

    name = "COIN"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n = 0

    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        self._n += 1
        if self._n % 3 != 0:
            return None
        return Signal(
            direction="long",
            stop_distance=1.0,
            rr_ratio=1.0,
            reason="coin",
            bar_index=bar_index,
            timestamp=bar.timestamp,
        )


class _StrongEdge(StrategyCore):
    """A strategy with a real, large, positive edge."""

    name = "EDGE"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n = 0

    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        self._n += 1
        if self._n % 3 != 0:
            return None
        return Signal(
            direction="long",
            stop_distance=0.5,
            rr_ratio=2.0,
            reason="edge",
            bar_index=bar_index,
            timestamp=bar.timestamp,
        )


# ----------------------------------------------------------------------
# test_edge
# ----------------------------------------------------------------------
def test_edge_detects_a_real_positive_edge():
    """A clearly profitable sample must come back significant."""
    rng = np.random.default_rng(3)
    rs = rng.normal(0.30, 0.5, 500)
    result = run_edge_test(rs, seed=1)
    assert result.significant is True
    assert result.conclusive is True
    assert result.ci_low > 0.0
    assert result.mean_r > 0.0


def test_edge_finds_nothing_in_pure_noise():
    """A zero-mean sample must NOT come back significant.

    This is the defect this module exists for: the sample below has a
    mean R of roughly zero, so an interval that excludes zero would be a
    false positive.
    """
    rng = np.random.default_rng(11)
    rs = rng.normal(0.0, 1.0, 4000)
    result = run_edge_test(rs, seed=1)
    assert result.ci_low <= 0.0 <= result.ci_high
    assert result.significant is False


def test_edge_is_reproducible():
    """Same input, same seed, same interval."""
    rng = np.random.default_rng(5)
    rs = rng.normal(0.05, 0.6, 800)
    a = run_edge_test(rs, seed=99)
    b = run_edge_test(rs, seed=99)
    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high


def test_edge_on_empty_sample_is_inconclusive_not_an_exception():
    """No trades is a fact to report, not a crash."""
    result = run_edge_test(np.array([]), seed=1)
    assert result.n_trades == 0
    assert result.conclusive is False
    assert result.significant is False
    assert result.warning()


def test_edge_on_tiny_sample_is_inconclusive_even_if_interval_excludes_zero():
    """A narrow interval from 10 trades is narrow for the wrong reason."""
    rs = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.2, 0.85, 1.15, 0.92, 1.02])
    result = run_edge_test(rs, seed=1)
    assert result.n_trades < MIN_TRADES_FOR_SIGNIFICANCE
    assert result.conclusive is False
    assert result.significant is False


def test_edge_rejects_nonsense_arguments():
    with pytest.raises(ValueError):
        run_edge_test(np.array([1.0, 2.0]), n_attempts=10)
    with pytest.raises(ValueError):
        run_edge_test(np.array([1.0, 2.0]), confidence=1.5)


def test_edge_test_is_exported_from_the_validate_package():
    """The public import path must work, not just the private one."""
    assert EdgeTest is DirectEdgeTest


# ----------------------------------------------------------------------
# The verdict
# ----------------------------------------------------------------------
def test_walkforward_does_not_say_holds_for_a_strategy_with_no_edge():
    """THE defect.

    Before the fix, a strategy whose out-of-sample mean R was
    indistinguishable from zero was reported with the verdict ``holds``,
    which reads as "the edge survived out of sample" when there was no
    edge to survive.
    """
    df = _frame()
    result = walk_forward(lambda: _CoinFlip(), df, n_splits=4)
    assert result.verdict != "holds"
    assert result.verdict == "no_edge"


def test_walkforward_reports_the_edge_confidence_interval():
    """The result must carry the interval so the report can print it."""
    df = _frame()
    result = walk_forward(lambda: _CoinFlip(), df, n_splits=4)
    assert result.edge_ci_low <= result.edge_ci_high
    assert result.edge_conclusive is True
    assert result.edge_significant is False
    # The interval must actually straddle zero for a no-edge strategy.
    assert result.edge_ci_low <= 0.0 <= result.edge_ci_high


def test_walkforward_warns_that_no_edge_was_detected():
    """The warning is mandatory and must be first in the list."""
    df = _frame()
    result = walk_forward(lambda: _CoinFlip(), df, n_splits=4)
    assert result.warnings, "a no-edge verdict must carry a warning"
    assert "no edge" in result.warnings[0].lower()


def test_walkforward_still_reports_holds_for_a_real_edge():
    """The fix must not break the genuine case."""
    df = _trending_frame()
    result = walk_forward(lambda: _StrongEdge(), df, n_splits=4)
    assert result.edge_significant is True
    assert result.verdict in ("holds", "degrades")


def test_walkforward_still_fails_a_strategy_with_negative_edge():
    """A losing strategy must not be upgraded by the new logic."""
    class _Loser(_CoinFlip):
        name = "LOSER"

        def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
            self._n += 1
            if self._n % 3 != 0:
                return None
            return Signal(
                direction="short",
                stop_distance=1.0,
                rr_ratio=1.0,
                reason="lose",
                bar_index=bar_index,
                timestamp=bar.timestamp,
            )

    df = _frame()
    result = walk_forward(lambda: _Loser(), df, n_splits=4)
    assert result.verdict == "fails"


def test_walkforward_summary_mentions_the_edge_interval():
    """The one-line summary must not hide the interval."""
    df = _frame()
    result = walk_forward(lambda: _CoinFlip(), df, n_splits=4)
    assert "edge CI" in result.summary()
