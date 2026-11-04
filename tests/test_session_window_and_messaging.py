"""Regression tests for four fixes made after an end-to-end agent exercise.

1. **The session window was a bar count, not a duration.** Both example
   strategies converted ``window_minutes`` with a hard-coded
   ``bars_per_minute = 5``, so "30 minutes" silently became 90 minutes on
   15-minute data and no CLI flag could correct it. The window is now
   measured in elapsed time from the first bar of the session.

2. **`fails` and "no edge was detected" were printed as two findings.**
   They are two true statements about one result, but read side by side
   they sound contradictory: `fails` suggests an edge that broke, when
   there was never an edge.

3. **The i.i.d. resampling diagnostic inverted its own adjective.** A
   higher i.i.d. pass rate is the *optimistic* reading; the code called it
   "pessimistic".
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prop_lab.cli import _load_strategy
from prop_lab.core import Bar, Signal, StrategyCore
from prop_lab.firm.monte_carlo import _resampling_diagnostic_direction
from prop_lab.report import build_report
from prop_lab.validate import walkforward as walkforward_module
from prop_lab.validate import walk_forward

# 2024-01-01 00:00:00 UTC, so every bar below sits on one calendar date.
SESSION_START = 1_704_067_200


def _example(filename: str, class_name: str):
    """Load a bundled example by file path, as the CLI does."""
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "strategies" / filename
    return _load_strategy(f"{path}:{class_name}")


def _bar(offset_minutes: int, *, high: float, low: float) -> Bar:
    """One bar, `offset_minutes` after the session start."""
    return Bar(
        timestamp=SESSION_START + offset_minutes * 60,
        open=(high + low) / 2,
        high=high,
        low=low,
        close=(high + low) / 2,
    )


def _feed(strategy, bars):
    """Run on_bar over the bars, returning the strategy for inspection."""
    for index, bar in enumerate(bars):
        strategy.on_bar(bar, index)
    return strategy


# ----------------------------------------------------------------------
# Fix 1: the window is a duration, not a bar count
# ----------------------------------------------------------------------
def test_window_covers_30_minutes_on_15_minute_bars():
    """Two 15-minute bars fill a 30-minute window; the third closes it."""
    strategy = _example("opening_range_breakout.py", "OpeningRangeBreakout")(
        window_minutes=30
    )
    bars = [
        _bar(0, high=101.0, low=99.0),    # inside the window
        _bar(15, high=102.0, low=98.0),   # inside the window
        _bar(30, high=200.0, low=1.0),    # at/after 30 min: must NOT count
    ]
    _feed(strategy, bars)

    assert strategy._range_high == 102.0
    assert strategy._range_low == 98.0
    assert strategy._window_open is False


def test_window_covers_30_minutes_on_5_minute_bars():
    """Six 5-minute bars fill the same window — the count is irrelevant."""
    strategy = _example("opening_range_breakout.py", "OpeningRangeBreakout")(
        window_minutes=30
    )
    bars = [_bar(i * 5, high=100.0 + i, low=90.0 - i) for i in range(6)]
    bars.append(_bar(30, high=999.0, low=1.0))
    _feed(strategy, bars)

    # hi grows to 105 at i=5, lo falls to 85 at i=5.
    assert strategy._range_high == 105.0
    assert strategy._range_low == 85.0
    assert strategy._window_open is False


def test_window_is_not_three_times_too_long_on_15_minute_bars():
    """The defect this guards: 30 minutes used to mean 90 on 15m bars."""
    strategy = _example("opening_range_breakout.py", "OpeningRangeBreakout")(
        window_minutes=30
    )
    # A bar at 45 minutes would have been inside the old 6-bar (90 min)
    # window and must be outside the correct one.
    bars = [_bar(0, high=100.0, low=100.0), _bar(45, high=500.0, low=1.0)]
    _feed(strategy, bars)

    assert strategy._range_high == 100.0, "the 45-minute bar must not be in the range"


def test_trend_continuation_also_uses_elapsed_time():
    strategy = _example("trend_continuation.py", "TrendContinuation")(window_minutes=30)
    bars = [
        _bar(0, high=101.0, low=99.0),
        _bar(15, high=102.0, low=98.0),
        _bar(30, high=300.0, low=1.0),
    ]
    _feed(strategy, bars)

    assert strategy._range_high == 102.0
    assert strategy._range_low == 98.0


@pytest.mark.parametrize(
    "filename,class_name",
    [
        ("opening_range_breakout.py", "OpeningRangeBreakout"),
        ("trend_continuation.py", "TrendContinuation"),
    ],
)
def test_bars_per_minute_is_gone_from_the_contract(filename, class_name):
    """The parameter that could be wrong is removed, not defaulted."""
    cls = _example(filename, class_name)
    parameters = inspect.signature(cls.__init__).parameters
    assert "bars_per_minute" not in parameters
    assert "window_minutes" in parameters


# ----------------------------------------------------------------------
# Fix 3: one sentence, not two contradictory findings
# ----------------------------------------------------------------------
class _AlwaysLong(StrategyCore):
    name = "always_long"

    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        return Signal(
            direction="long",
            stop_distance=1.0,
            rr_ratio=1.0,
            reason="always_long",
            bar_index=bar_index,
            timestamp=bar.timestamp,
        )


def _declining_frame(n: int = 600) -> pd.DataFrame:
    """A steadily falling series, so an always-long strategy loses."""
    time = SESSION_START + np.arange(n) * 900
    close = 200.0 - np.arange(n) * 0.05
    return pd.DataFrame(
        {
            "time": time,
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1.0,
        }
    )


def test_fails_and_no_edge_are_reported_as_one_statement(monkeypatch):
    """A losing strategy with an inconclusive edge must not read as two bugs."""

    class _InconclusiveEdge:
        conclusive = True
        significant = False
        mean_r = -0.012
        ci_low = -0.20
        ci_high = 0.11
        n_trades = 400
        t_statistic = -0.4
        warnings: list = []

        def warning(self) -> str:
            return "no edge was detected: placeholder"

    monkeypatch.setattr(
        walkforward_module, "test_edge", lambda *a, **k: _InconclusiveEdge()
    )

    result = walk_forward(lambda: _AlwaysLong(), _declining_frame(), n_splits=5)

    assert result.verdict == "fails"
    joined = [w for w in result.warnings if "fails, and no edge was detected" in w]
    assert joined, f"expected the joined warning; got {result.warnings}"
    assert "nothing was there" not in joined[0]


def test_report_headline_joins_fails_with_no_edge():
    """The headline must say the two lines are one statement."""

    @dataclass
    class _WF:
        windows: list = field(default_factory=list)
        oos_mean_r: float = -0.012
        oos_win_rate: float = 0.34
        oos_total_r: float = -4.0
        is_mean_r: float = 0.02
        degradation: float = 0.032
        verdict: str = "fails"
        n_windows: int = 5
        warnings: list = field(default_factory=list)
        edge_significant: bool = False
        edge_conclusive: bool = True
        edge_ci_low: float = -0.20
        edge_ci_high: float = 0.11
        edge_t_statistic: float = -0.4

    @dataclass
    class _MC:
        n_attempts: int = 2000
        pass_rate: float = 0.24
        fail_rate: float = 0.76
        timeout_rate: float = 0.0
        mean_final_equity: float = 49_900.0
        p5_final_equity: float = 47_900.0
        p95_final_equity: float = 54_000.0
        mean_trades_to_pass: float = 4.0
        pass_rate_ci_low: float = 0.22
        pass_rate_ci_high: float = 0.26
        n_trades_sample: int = 400
        warnings: list = field(default_factory=list)

    @dataclass
    class _Rules:
        name: str = "APEX 50K EOD"

    markdown = build_report(
        wf=_WF(),
        mc=_MC(),
        surfaces=[],
        strategy_name="AlwaysLong",
        data_source="synthetic",
        data_range="2024-01-01 .. 2024-07-01",
        rules=_Rules(),
        extra_warnings=None,
        baseline_pass_rate=0.27,
    )

    assert "one statement, not two findings" in markdown


# ----------------------------------------------------------------------
# Fix 4: the i.i.d. diagnostic says which way it actually errs
# ----------------------------------------------------------------------
def test_resampling_direction_is_optimistic_when_iid_scores_higher():
    """gap = block - iid, so a negative gap means i.i.d. flatters."""
    assert _resampling_diagnostic_direction(-0.03) == "optimistic"
    assert _resampling_diagnostic_direction(+0.03) == "pessimistic"
