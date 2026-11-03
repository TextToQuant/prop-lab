"""Baseline validation against the reference project.

This script ports the reference project's TC_30_2.0 strategy verbatim and
runs it through prop_lab's own backtest engine on the same MNQ 5-minute
dataset, so the engine can be checked against a known reference.

TWO BASELINES, AND THE DIFFERENCE MATTERS
------------------------------------------

The reference project (github.com/dsilva42/prop-risk-geometry, originally
hunter547) reports 73 trades, 41.1 percent win rate and +0.233R mean.
Reproducing that requires reproducing its session rule, and its session
rule is wrong for futures.

The reference groups bars by the UTC CALENDAR DATE. A futures trading day
runs 18:00 ET to 17:00 ET, so a UTC calendar date splits every session
into two fragments: the evening part and the morning part land on
different UTC dates. The reference therefore treats one trading day as
two, which produces 86 "sessions" over a period that really contains 71
trading days.

That is why the reference finds 73 trades and this engine finds 66 on
the same data with the correct session rule. The extra trades are
entries triggered on a fragment of a session, not on a session boundary.

Both numbers are printed so the difference is visible rather than hidden.
The reference numbers are a comparison point, not a target: the correct
session rule is the one a real prop firm account experiences, because
the firm's daily loss limit and end-of-day drawdown calculation both
reset at 18:00 ET.

Run with:

    PROPLAB_MNQ_5M=/path/to/MNQ_5m.csv python scripts/baseline_check.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from prop_lab.core import Bar, Signal, StrategyCore, run_backtest, summarize_trades
from prop_lab.data import assign_session_dates

# Reference numbers from the original project.
REFERENCE_TRADES = 73
REFERENCE_WIN_RATE = 0.411
REFERENCE_MEAN_R = 0.233


class TrendContinuation(StrategyCore):
    """Verbatim port of the reference project's TC_30_2.0 strategy.

    Continuation after a pullback to the session's value area: track the
    first 30 minutes of the session, then enter long when price closes
    above the range midpoint after having dipped back into it.
    """

    name = "TC"

    def __init__(self, window_minutes: int = 30, rr_ratio: float = 2.0,
                 direction: str = "both", **kwargs):
        super().__init__(window_minutes=window_minutes, rr_ratio=rr_ratio,
                         direction=direction)
        self.window_bars = max(1, window_minutes // 5)
        self.rr_ratio = rr_ratio
        self.direction = direction
        self._session_date = None
        self._range_bars_seen = 0
        self._range_high = -np.inf
        self._range_low = np.inf
        self._signal_fired = False

    def on_session_start(self, bar: Bar) -> None:
        self._range_bars_seen = 0
        self._range_high = -np.inf
        self._range_low = np.inf
        self._signal_fired = False

    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        if bar.date != self._session_date:
            self.on_session_start(bar)
            self._session_date = bar.date

        if self._range_bars_seen < self.window_bars:
            self._range_high = max(self._range_high, bar.high)
            self._range_low = min(self._range_low, bar.low)
            self._range_bars_seen += 1
            return None

        if self._signal_fired:
            return None

        range_mid = (self._range_high + self._range_low) / 2
        if self.direction in ("both", "long") and bar.close > range_mid and bar.low < range_mid:
            self._signal_fired = True
            stop_distance = bar.close - self._range_low
            return Signal(
                direction="long", stop_distance=stop_distance,
                rr_ratio=self.rr_ratio, reason="tc_long_pullback_continuation",
                bar_index=bar_index, timestamp=bar.timestamp,
            )
        if self.direction in ("both", "short") and bar.close < range_mid and bar.high > range_mid:
            self._signal_fired = True
            stop_distance = self._range_high - bar.close
            return Signal(
                direction="short", stop_distance=stop_distance,
                rr_ratio=self.rr_ratio, reason="tc_short_pullback_continuation",
                bar_index=bar_index, timestamp=bar.timestamp,
            )
        return None


def main() -> int:
    csv_path = os.environ.get("PROPLAB_MNQ_5M", "")
    if not csv_path:
        print("Set PROPLAB_MNQ_5M to the MNQ 5m CSV path.")
        return 2
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return 2

    df = pd.read_csv(csv_path)
    keep = ["time", "open", "high", "low", "close"]
    volume_col = "Volume" if "Volume" in df.columns else None
    bars_df = df[keep + ([volume_col] if volume_col else [])].copy()
    if volume_col:
        bars_df = bars_df.rename(columns={"Volume": "volume"})
    bars_df = bars_df.sort_values("time").reset_index(drop=True)

    # -- variant A: the reference project's UTC calendar date rule -----
    ref = bars_df.copy()
    ref["session_date"] = pd.to_datetime(
        ref["time"], unit="s", utc=True
    ).dt.strftime("%Y-%m-%d")
    ref_trades = run_backtest(TrendContinuation(), ref)
    ref_stats = summarize_trades(ref_trades)

    # -- variant B: the correct futures session rule --------------------
    correct = bars_df.copy()
    correct["session_date"] = assign_session_dates(correct, session_open_hour=18)
    cor_trades = run_backtest(TrendContinuation(), correct)
    cor_stats = summarize_trades(cor_trades)

    print("prop-lab baseline check")
    print(f"dataset : {csv_path}")
    print(f"bars    : {len(bars_df)}")
    print()
    print("A. reference session rule (UTC calendar date)")
    print(f"   sessions : {ref['session_date'].nunique()}")
    print(f"   trades   : {ref_stats['n_trades']}   (reference: {REFERENCE_TRADES})")
    print(f"   win %    : {ref_stats['win_rate'] * 100:.1f}   (reference: {REFERENCE_WIN_RATE * 100:.1f})")
    print(f"   mean R   : {ref_stats['mean_r']:+.3f}   (reference: {REFERENCE_MEAN_R:+.3f})")
    print()
    print("B. correct futures session rule (18:00 ET to 17:00 ET)")
    print(f"   sessions : {correct['session_date'].nunique()}")
    print(f"   trades   : {cor_stats['n_trades']}")
    print(f"   win %    : {cor_stats['win_rate'] * 100:.1f}")
    print(f"   mean R   : {cor_stats['mean_r']:+.3f}")
    print()

    ok = True

    # The reference rule must still reproduce the reference numbers, or
    # the engine itself has drifted.
    if ref_stats["n_trades"] != REFERENCE_TRADES:
        print(
            f"MISMATCH: with the reference session rule the engine gives "
            f"{ref_stats['n_trades']} trades, the reference says {REFERENCE_TRADES}."
        )
        ok = False
    if abs(ref_stats["win_rate"] - REFERENCE_WIN_RATE) > 0.02:
        print(
            f"MISMATCH: win rate {ref_stats['win_rate']:.3f} != {REFERENCE_WIN_RATE}"
        )
        ok = False
    if abs(ref_stats["mean_r"] - REFERENCE_MEAN_R) > 0.05:
        print(f"MISMATCH: mean R {ref_stats['mean_r']:.3f} != {REFERENCE_MEAN_R}")
        ok = False

    if ok:
        print("PASS: the engine reproduces the reference baseline under the")
        print("      reference session rule, and the session-rule difference")
        print("      is understood and documented.")
    else:
        print("FAIL: the engine has drifted from the reference implementation.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
