"""Opening range breakout.

The second worked example, and deliberately a different shape from the
trend continuation strategy: this one trades the break of the session
range rather than a pullback into it.

The logic
---------
1. Track the first ``window_minutes`` of the session to build the
   opening range.
2. After the range forms, take a long when price closes above the
   range high, and a short when it closes below the range low.
3. Stop on the opposite side of the range, so the stop distance is the
   full range width plus a small buffer.
4. One signal per session.

Why this example matters
------------------------
The point of the prop-lab contract is that an agent can write a
strategy from a plain-language description without touching the engine.
This file is what "written from a description" looks like: everything
that touches the engine is the ``on_bar`` method and the ``Signal``
return. Nothing else in this package knows what a breakout is.

Run it with::

    proplab backtest --strategy examples.strategies.opening_range_breakout \\
        --data MNQ_5m.csv
"""
from __future__ import annotations

import numpy as np

from prop_lab.core import Bar, Signal, StrategyCore

__all__ = ["OpeningRangeBreakout"]


class OpeningRangeBreakout(StrategyCore):
    """Break of the opening range.

    Parameters:
        window_minutes: length of the opening range in minutes. Default 30.
            This is measured in real elapsed time, so it means 30 minutes
            on any timeframe.
        rr_ratio: target distance as a multiple of the stop distance.
            Default 1.5.
        direction: 'both', 'long' or 'short'. Default 'both'.
        buffer_points: extra distance added to the stop beyond the range
            width, to avoid being stopped by a marginal break. Default 0.0.
    """

    name = "ORB"

    def __init__(
        self,
        window_minutes: int = 30,
        rr_ratio: float = 1.5,
        direction: str = "both",
        buffer_points: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            window_minutes=window_minutes,
            rr_ratio=rr_ratio,
            direction=direction,
            buffer_points=buffer_points,
        )
        self.window_minutes = window_minutes
        self.rr_ratio = rr_ratio
        self.direction = direction
        self.buffer_points = buffer_points
        self._session_date: str | None = None
        self._session_anchor: int | None = None
        self._window_open = True
        self._range_high = -np.inf
        self._range_low = np.inf
        self._signal_fired = False

    def on_session_start(self, bar: Bar) -> None:
        """Reset the opening range at the start of each session."""
        self._session_anchor = bar.timestamp
        self._window_open = True
        self._range_high = -np.inf
        self._range_low = np.inf
        self._signal_fired = False

    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        """Emit at most one breakout signal per session."""
        if bar.date != self._session_date:
            self.on_session_start(bar)
            self._session_date = bar.date

        # The opening range is the first `window_minutes` of the session,
        # measured from the first bar this session actually saw. Counting
        # bars instead would silently mean a different window on a
        # different timeframe: 6 bars is 30 minutes at 5m and 90 at 15m.
        if self._window_open:
            elapsed = bar.timestamp - self._session_anchor
            if elapsed < self.window_minutes * 60:
                self._range_high = max(self._range_high, bar.high)
                self._range_low = min(self._range_low, bar.low)
                return None
            self._window_open = False

        if self._signal_fired:
            return None

        range_width = self._range_high - self._range_low
        if range_width <= 0:
            return None

        # Long: close above the opening range high.
        if self.direction in ("both", "long") and bar.close > self._range_high:
            self._signal_fired = True
            stop_distance = bar.close - self._range_low + self.buffer_points
            return Signal(
                direction="long",
                stop_distance=stop_distance,
                rr_ratio=self.rr_ratio,
                reason="orb_long_breakout",
                bar_index=bar_index,
                timestamp=bar.timestamp,
            )

        # Short: close below the opening range low.
        if self.direction in ("both", "short") and bar.close < self._range_low:
            self._signal_fired = True
            stop_distance = self._range_high - bar.close + self.buffer_points
            return Signal(
                direction="short",
                stop_distance=stop_distance,
                rr_ratio=self.rr_ratio,
                reason="orb_short_breakout",
                bar_index=bar_index,
                timestamp=bar.timestamp,
            )

        return None
