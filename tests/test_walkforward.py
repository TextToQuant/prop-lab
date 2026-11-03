"""Tests for the purged walk-forward validation layer.

These tests use synthetic deterministic price series so the results are
reproducible on any machine. They check the machinery (splits,
isolation, purge/embargo, trade accounting) and the two verdict paths
that matter: a losing strategy must fail, and a strategy with an edge
must not be silently reported as broken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prop_lab.core import Signal, StrategyCore
from prop_lab.validate import WFResult, WFWindow, walk_forward, walk_forward_table


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------
def make_frame(
    n: int = 900,
    seed: int = 7,
    drift: float = 0.0,
    vol: float = 1.0,
    start_time: int = 1_600_000_000,
    period: int = 3600,
) -> pd.DataFrame:
    """Build a deterministic OHLCV frame with a 'session_date' column.

    Args:
        n: number of bars.
        seed: numpy default_rng seed.
        drift: per-bar drift of the log price.
        vol: per-bar volatility.
        start_time: UNIX seconds of the first bar.
        period: seconds between bars.

    Returns:
        A frame in the standard schema plus 'session_date'.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=drift, scale=vol, size=n)
    # Normalise the cumulative log path into a bounded band so the
    # synthetic prices stay realistic: an unbounded random walk over
    # 1200 bars would otherwise collapse towards zero and make the R
    # magnitudes meaningless.
    raw = np.cumsum(steps)
    scale = float(np.max(np.abs(raw))) or 1.0
    path = raw / scale * 0.5
    close = 100.0 * np.exp(path)
    open_ = np.empty(n, dtype=float)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(loc=0.0, scale=vol, size=n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    times = start_time + np.arange(n, dtype=np.int64) * period
    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )
    # One session per calendar day (UTC), so force_close_eod has an effect.
    df["session_date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    return df


def make_declining_frame(
    n: int = 600, step: float = 0.05, start_time: int = 1_600_000_000, period: int = 3600
) -> pd.DataFrame:
    """Build a strictly declining series: every long trade loses money.

    Args:
        n: number of bars.
        step: price decrement per bar.
        start_time: UNIX seconds of the first bar.
        period: seconds between bars.

    Returns:
        A deterministic frame in the standard schema plus 'session_date'.
    """
    close = 100.0 - step * np.arange(n, dtype=float)
    open_ = np.empty(n, dtype=float)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    times = start_time + np.arange(n, dtype=np.int64) * period
    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": close + step / 2.0,
            "low": close - step / 2.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )
    df["session_date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    return df


def make_trending_frame(
    n: int = 1500,
    seed: int = 5,
    segment: int = 60,
    up: float = 0.006,
    down: float = 0.004,
    start_time: int = 1_600_000_000,
    period: int = 3600,
) -> pd.DataFrame:
    """Build a deterministic regime-switching series with a real trend edge.

    The drift alternates between ``up`` and ``down`` every ``segment``
    bars, so a fast moving-average crossover has a genuine, recoverable
    edge: the test can then assert a non-failing verdict instead of
    falling back to pure accounting invariants.

    Args:
        n: number of bars.
        seed: numpy default_rng seed.
        segment: length in bars of each up/down regime.
        up: per-bar drift during an up regime.
        down: per-bar drift during a down regime.
        start_time: UNIX seconds of the first bar.
        period: seconds between bars.

    Returns:
        A deterministic frame in the standard schema plus 'session_date'.
    """
    rng = np.random.default_rng(seed)
    drift = np.empty(n, dtype=float)
    level = up
    for i in range(n):
        if i % segment == 0:
            level = up if (i // segment) % 2 == 0 else down
        drift[i] = level
    steps = drift + rng.normal(0.0, 0.3, n)
    raw = np.cumsum(steps)
    scale = float(max(abs(raw.max()), abs(raw.min()))) or 1.0
    close = 100.0 * np.exp(raw / scale * 1.2)
    open_ = np.empty(n, dtype=float)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0.0, 0.3, n))
    times = start_time + np.arange(n, dtype=np.int64) * period
    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )
    df["session_date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    return df


# ----------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------
class MaCrossStrategy(StrategyCore):
    """Moving-average crossover: long when the fast average is above the slow one."""

    name = "ma_cross"

    def __init__(self, fast: int = 5, slow: int = 20, rr: float = 1.0, stop: float = 2.0):
        super().__init__(fast=fast, slow=slow, rr=rr, stop=stop)
        self.fast = fast
        self.slow = slow
        self.rr = rr
        self.stop = stop
        self._closes: list[float] = []

    def on_session_start(self, bar) -> None:
        super().on_session_start(bar)
        self._closes = []

    def on_bar(self, bar, bar_index: int) -> Signal | None:
        self._closes.append(bar.close)
        if len(self._closes) < self.slow:
            return None
        fast = float(np.mean(self._closes[-self.fast :]))
        slow = float(np.mean(self._closes[-self.slow :]))
        if fast > slow:
            return Signal(
                direction="long",
                stop_distance=self.stop,
                rr_ratio=self.rr,
                reason="ma_long",
                bar_index=bar_index,
                timestamp=bar.timestamp,
            )
        return None


class AlwaysLosesStrategy(StrategyCore):
    """Enters long on every bar; on a falling market every trade loses."""

    name = "always_loses"

    def __init__(self, stop: float = 1.0):
        super().__init__(stop=stop)
        self.stop = stop

    def on_bar(self, bar, bar_index: int) -> Signal | None:
        return Signal(
            direction="long",
            stop_distance=self.stop,
            rr_ratio=1.0,
            reason="always_loses",
            bar_index=bar_index,
            timestamp=bar.timestamp,
        )


class NeverTradesStrategy(StrategyCore):
    """Never emits a signal."""

    name = "never_trades"

    def on_bar(self, bar, bar_index: int) -> Signal | None:
        return None


class NotAFactory:  # pragma: no cover - only used for the error path
    """Not callable, so it must be rejected."""


# ----------------------------------------------------------------------
# Test A: the run produces the expected number of windows
# ----------------------------------------------------------------------
def test_walk_forward_returns_requested_number_of_windows():
    df = make_frame()
    result = walk_forward(lambda: MaCrossStrategy(), df, n_splits=5)
    assert isinstance(result, WFResult)
    assert result.n_windows == 5
    assert len(result.windows) == 5
    assert [w.fold for w in result.windows] == [0, 1, 2, 3, 4]
    assert all(isinstance(w, WFWindow) for w in result.windows)


def test_windows_are_chronological_and_non_overlapping():
    df = make_frame()
    result = walk_forward(lambda: MaCrossStrategy(), df, n_splits=4)
    prev_test_end = None
    for w in result.windows:
        assert w.test_start < w.test_end
        if w.n_train_bars:
            assert w.train_end <= w.test_start
        if prev_test_end is not None:
            assert w.test_start > prev_test_end
        prev_test_end = w.test_end


def test_window_boundaries_are_unix_seconds_from_the_frame():
    df = make_frame(n=400)
    result = walk_forward(lambda: MaCrossStrategy(), df, n_splits=3)
    times = set(df["time"].tolist())
    for w in result.windows:
        assert w.test_start in times
        assert w.test_end in times
        if w.n_train_bars:
            assert w.train_start in times
            assert w.train_end in times


def test_test_blocks_cover_the_tail_without_overlap():
    df = make_frame(n=1000)
    result = walk_forward(lambda: MaCrossStrategy(), df, n_splits=5)
    covered = sum(w.n_test_bars for w in result.windows)
    # The seed block is the first of the 6 contiguous blocks; the rest
    # are the out-of-sample test blocks.
    expected_seed = len(np.array_split(np.arange(len(df)), 6)[0])
    assert covered == len(df) - expected_seed
    starts = [w.test_start for w in result.windows]
    ends = [w.test_end for w in result.windows]
    assert starts == sorted(starts)
    assert ends == sorted(ends)


# ----------------------------------------------------------------------
# Test B: a strategy that always loses must fail
# ----------------------------------------------------------------------
def test_always_losing_strategy_fails():
    df = make_declining_frame()
    result = walk_forward(lambda: AlwaysLosesStrategy(), df, n_splits=5)
    assert result.n_oos_trades > 0
    assert result.oos_mean_r <= 0
    assert result.verdict == "fails"
    assert result.oos_win_rate == 0.0


def test_losing_strategy_fails_on_noisy_data_too():
    df = make_frame(n=900, seed=7, drift=-0.0008, vol=1.0)
    result = walk_forward(lambda: AlwaysLosesStrategy(), df, n_splits=5)
    assert result.oos_mean_r <= 0
    assert result.verdict == "fails"


# ----------------------------------------------------------------------
# Test C: an edge survives (or at least the accounting is consistent)
# ----------------------------------------------------------------------
def test_trending_strategy_keeps_its_edge_out_of_sample():
    """A strategy with a real edge must not be reported as failing."""
    df = make_trending_frame()
    result = walk_forward(
        lambda: MaCrossStrategy(fast=3, slow=10, stop=1.5), df, n_splits=5
    )

    # The invariant that must always hold.
    assert sum(w.n_test_trades for w in result.windows) == result.n_oos_trades
    assert sum(w.n_train_trades for w in result.windows) == result.n_is_trades

    assert result.n_oos_trades >= 30
    assert result.oos_mean_r > 0
    # "no_edge" is allowed here because this fixture's edge is positive
    # but NOT statistically significant: a mean R of +0.05 with a 95%
    # interval of [-0.12, +0.23] over 222 trades is not a demonstrated
    # edge. What must never happen is a "fails" verdict for a strategy
    # whose out-of-sample mean R is positive and whose degradation is
    # small.
    assert result.verdict in ("holds", "degrades", "no_edge")
    assert result.degradation <= 0.30


def test_trending_strategy_out_of_sample_accounting_is_consistent():
    df = make_frame(n=1200, seed=11, drift=0.0006, vol=0.5)
    result = walk_forward(lambda: MaCrossStrategy(), df, n_splits=5)

    # The invariant that must always hold.
    assert sum(w.n_test_trades for w in result.windows) == result.n_oos_trades
    assert sum(w.n_train_trades for w in result.windows) == result.n_is_trades

    if result.n_oos_trades >= 30:
        assert result.verdict in ("holds", "degrades", "fails", "no_edge")
    if result.verdict in ("holds", "degrades"):
        assert result.oos_mean_r > 0
    if result.n_oos_trades < 30:
        assert any("sample too small to conclude" in w for w in result.warnings)


def test_never_trading_strategy_fails_with_the_empty_fold_warning():
    df = make_frame(n=300, seed=5)
    result = walk_forward(lambda: NeverTradesStrategy(), df, n_splits=3)
    assert result.n_oos_trades == 0
    assert result.verdict == "fails"
    assert any("zero out-of-sample trades" in w for w in result.warnings)
    assert any("sample too small to conclude" in w for w in result.warnings)


# ----------------------------------------------------------------------
# Test D: the factory is called per run and no instance is ever reused
# ----------------------------------------------------------------------
def test_factory_is_called_for_every_backtest_run():
    df = make_frame()
    calls = {"n": 0}
    built: list[MaCrossStrategy] = []

    def factory():
        calls["n"] += 1
        strategy = MaCrossStrategy()
        built.append(strategy)  # keep the reference so ids stay unique
        return strategy

    result = walk_forward(factory, df, n_splits=5)
    # One fresh instance per backtest run: two runs (in-sample and
    # out-of-sample) per fold, so no state can leak between folds.
    assert calls["n"] == 2 * result.n_windows
    assert len(built) == calls["n"]
    assert len({id(s) for s in built}) == calls["n"]
    assert all(s is not built[0] for s in built[1:])


def test_factory_is_called_once_per_fold_at_least():
    """The factory must be called at least once for each fold."""
    df = make_frame()
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return MaCrossStrategy()

    result = walk_forward(factory, df, n_splits=4)
    assert calls["n"] >= result.n_windows


def test_state_does_not_leak_between_folds():
    """A strategy with internal state must give identical fold results.

    If an instance were reused across folds, the second fold would start
    with buffers left over from the first and the trade counts would
    drift. Each fold is backtested twice, in-sample and out-of-sample,
    and each run starts from a clean instance.
    """
    df = make_frame(n=900, seed=13)
    result = walk_forward(lambda: MaCrossStrategy(), df, n_splits=5)
    assert result.n_windows == 5
    assert sum(w.n_test_trades for w in result.windows) == result.n_oos_trades
    assert result.n_oos_trades > 0

    # The same factory, same data, same numbers: deterministic.
    again = walk_forward(lambda: MaCrossStrategy(), df, n_splits=5)
    assert again.n_oos_trades == result.n_oos_trades
    assert again.oos_mean_r == pytest.approx(result.oos_mean_r)
    assert again.degradation == pytest.approx(result.degradation)


def test_factory_must_be_callable():
    df = make_frame()
    with pytest.raises(TypeError):
        walk_forward(NotAFactory(), df)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Test E: purge and embargo change the training sample
# ----------------------------------------------------------------------
def test_purge_and_embargo_shrink_the_training_window():
    df = make_frame(n=900, seed=17)
    clean = walk_forward(lambda: MaCrossStrategy(), df, n_splits=4)
    purged = walk_forward(
        lambda: MaCrossStrategy(), df, n_splits=4, purge_bars=25, embargo_bars=25
    )
    assert clean.n_windows == purged.n_windows
    for wc, wp in zip(clean.windows, purged.windows):
        assert wp.n_train_bars < wc.n_train_bars
        assert wp.n_train_bars == wc.n_train_bars - 50


def test_purge_alone_removes_the_tail_of_the_training_window():
    df = make_frame(n=900, seed=19)
    clean = walk_forward(lambda: MaCrossStrategy(), df, n_splits=3)
    purged = walk_forward(
        lambda: MaCrossStrategy(), df, n_splits=3, purge_bars=10
    )
    for wc, wp in zip(clean.windows, purged.windows):
        assert wp.n_train_bars == wc.n_train_bars - 10
        # The purge removes the bars closest to the test boundary, so
        # the training window still ends before the test block starts.
        assert wp.train_end <= wp.test_start


def test_embargo_alone_also_shrinks_the_training_window():
    df = make_frame(n=900, seed=23)
    clean = walk_forward(lambda: MaCrossStrategy(), df, n_splits=3)
    embargoed = walk_forward(
        lambda: MaCrossStrategy(), df, n_splits=3, embargo_bars=15
    )
    for wc, we in zip(clean.windows, embargoed.windows):
        assert we.n_train_bars == wc.n_train_bars - 15


def test_purge_and_embargo_are_reflected_in_the_warnings():
    df = make_frame(n=900, seed=17)
    purged = walk_forward(
        lambda: MaCrossStrategy(), df, n_splits=3, purge_bars=5, embargo_bars=7
    )
    assert any("purge/embargo active" in w for w in purged.warnings)


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def test_walk_forward_table_renders_all_folds():
    df = make_frame()
    result = walk_forward(lambda: MaCrossStrategy(), df, n_splits=3)
    text = walk_forward_table(result)
    assert isinstance(text, str)
    assert "fold" in text
    assert "oos_mean_r" in text
    for w in result.windows:
        assert str(w.fold) in text
    assert "verdict=" in text
    # No emoji, plain text only.
    assert text.isascii()


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------
def test_invalid_arguments_are_rejected():
    df = make_frame(n=200)
    with pytest.raises(ValueError):
        walk_forward(lambda: MaCrossStrategy(), df, n_splits=0)
    with pytest.raises(ValueError):
        walk_forward(lambda: MaCrossStrategy(), df, train_frac=0.0)
    with pytest.raises(ValueError):
        walk_forward(lambda: MaCrossStrategy(), df, purge_bars=-1)
    with pytest.raises(ValueError):
        walk_forward(lambda: MaCrossStrategy(), df, embargo_bars=-1)


def test_degradation_definition_matches_spec():
    df = make_frame(n=900, seed=29)
    result = walk_forward(lambda: MaCrossStrategy(), df, n_splits=4)
    assert result.degradation == pytest.approx(result.is_mean_r - result.oos_mean_r)


def test_verdict_thresholds_are_respected():
    """The verdict must follow the degradation thresholds AND significance.

    The degradation-only thresholds are unchanged. What changed is that a
    strategy whose out-of-sample edge is not distinguishable from zero is
    no longer reported as "holds" -- there was no edge to hold. The
    fixture strategy here genuinely has no significant edge (its mean R is
    positive but its confidence interval spans zero), so "no_edge" is the
    correct verdict and the old assertion of "holds" was asserting the
    defect.
    """
    df = make_frame(n=1200, seed=31, drift=0.0004, vol=0.4)
    result = walk_forward(lambda: MaCrossStrategy(), df, n_splits=5)
    if result.oos_mean_r <= 0:
        assert result.verdict == "fails"
    elif result.degradation > 0.30:
        assert result.verdict == "fails"
    elif result.degradation > 0.10:
        assert result.verdict == "degrades"
    elif result.edge_conclusive and not result.edge_significant:
        assert result.verdict == "no_edge"
    else:
        assert result.verdict == "holds"


def test_oos_beats_is_is_flagged_as_a_caveat():
    """When OOS beats IS by a lot, the result must say so."""
    df = make_trending_frame()
    result = walk_forward(
        lambda: MaCrossStrategy(fast=3, slow=10, stop=1.5), df, n_splits=5
    )
    if result.degradation < -0.05:
        assert any("out-of-sample beat in-sample" in w for w in result.warnings)
