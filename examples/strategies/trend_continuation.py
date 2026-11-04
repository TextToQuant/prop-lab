"""Trend continuation after a pullback to the session value area.

This is the reference strategy from the project that prop_lab was
extracted from. It reproduces the headline baseline: 73 trades, 42.5
percent win rate, +0.262R mean on 14 weeks of MNQ 5-minute data.

The logic
---------
1. Track the first 30 minutes of the trading session. That range defines
   the session's value area.
2. After the range forms, look for a reclaim of its midpoint: price
   closes above the midpoint for a long, having dipped back into the
   range on the way.
3. Stop below the session low (long) or above the session high (short).
   Target is two times the stop distance.

Use it as the worked example in AGENTS.md. Read the strategy, then read
the class docstring on ``StrategyCore`` to understand the contract a
strategy must satisfy.

Verify with::

    PROPLAB_MNQ_5M=/path/to/MNQ_5m.csv python scripts/baseline_check.py
"""
from __future__ import annotations

import numpy as np

from prop_lab.core import Bar, Signal, StrategyCore

__all__ = ["TrendContinuation"]


class TrendContinuation(StrategyCore):
    """Continuation after a pullback to the session's value area.

    Parameters:
        window_minutes: how many minutes at the open of the session form
            the value area. Default 30. This is measured in real elapsed
            time, so it means 30 minutes on any timeframe.
        rr_ratio: target distance as a multiple of the stop distance.
            Default 2.0.
        direction: 'both', 'long' or 'short'. Default 'both'.
    """

    name = "TC"

    def __init__(
        self,
        window_minutes: int = 30,
        rr_ratio: float = 2.0,
        direction: str = "both",
        **kwargs,
    ):
        super().__init__(
            window_minutes=window_minutes,
            rr_ratio=rr_ratio,
            direction=direction,
        )
        self.window_minutes = window_minutes
        self.rr_ratio = rr_ratio
        self.direction = direction
        self._session_date: str | None = None
        self._session_anchor: int | None = None
        self._window_open = True
        self._range_high = -np.inf
        self._range_low = np.inf
        self._signal_fired = False

    def on_session_start(self, bar: Bar) -> None:
        """Reset the value area at the start of each trading session."""
        self._session_anchor = bar.timestamp
        self._window_open = True
        self._range_high = -np.inf
        self._range_low = np.inf
        self._signal_fired = False

    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        """Emit at most one signal per session."""
        if bar.date != self._session_date:
            self.on_session_start(bar)
            self._session_date = bar.date

        # Build the value area for the first `window_minutes` of the
        # session, measured from the first bar this session actually saw.
        # Counting bars instead would silently mean a different window on
        # a different timeframe: 6 bars is 30 minutes at 5m and 90 at 15m.
        if self._window_open:
            elapsed = bar.timestamp - self._session_anchor
            if elapsed < self.window_minutes * 60:
                self._range_high = max(self._range_high, bar.high)
                self._range_low = min(self._range_low, bar.low)
                return None
            self._window_open = False

        if self._signal_fired:
            return None

        range_mid = (self._range_high + self._range_low) / 2.0

        # Long: close back above the midpoint after dipping into the range.
        if (
            self.direction in ("both", "long")
            and bar.close > range_mid
            and bar.low < range_mid
        ):
            self._signal_fired = True
            stop_distance = bar.close - self._range_low
            return Signal(
                direction="long",
                stop_distance=stop_distance,
                rr_ratio=self.rr_ratio,
                reason="tc_long_pullback_continuation",
                bar_index=bar_index,
                timestamp=bar.timestamp,
            )

        # Short: close back below the midpoint after reaching into the range.
        if (
            self.direction in ("both", "short")
            and bar.close < range_mid
            and bar.high > range_mid
        ):
            self._signal_fired = True
            stop_distance = self._range_high - bar.close
            return Signal(
                direction="short",
                stop_distance=stop_distance,
                rr_ratio=self.rr_ratio,
                reason="tc_short_pullback_continuation",
                bar_index=bar_index,
                timestamp=bar.timestamp,
            )

        return None
