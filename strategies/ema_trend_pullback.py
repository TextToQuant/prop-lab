"""EMA trend + pullback continuation on BTCUSDT (worked example).

The plain-language idea this implements
---------------------------------------
"Buy BTC when the trend is up and the price pulls back to the 20 EMA and
then turns back up; do the same mirrored on the downside."

Every ambiguous word in that sentence has been resolved explicitly and
recorded in the ``notes`` attribute, as the prop-lab contract requires.
Nothing is left to a default that was not written down.

Translation table
-----------------
=========================  ==========================================
plain language             interpretation used here
=========================  ==========================================
"the trend is up"          EMA(fast) > EMA(slow) AND the slow EMA's
                           slope over the last ``slope_bars`` bars is
                           positive (a rising, not a flat, trend)
"pulls back to the 20 EMA" the bar's LOW touches or dips below the fast
                           EMA (low <= ema_fast), i.e. price came back
                           to the average rather than closing on it
"turns back up"            the bar's CLOSE is above its own open
                           (close > open), a rejection candle
"do the same mirrored"     a short requires ema_fast < ema_slow, a
                           falling slow EMA, a HIGH that reaches or
                           exceeds the fast EMA (high >= ema_fast), and
                           a down candle (close < open)
"the 20 EMA"               ``ema_fast`` period, default 20
=========================  ==========================================

Signal rules
------------
1. Long  : trend up, low <= ema_fast, close > open.
2. Short : trend down, high >= ema_fast, close < open.
3. One signal per bar at most. The engine fills at the OPEN of the next
   bar, so the close-based conditions above cannot be traded at the
   close they were computed from.

Stop placement
--------------
ATR multiple: ``stop_distance = atr_multiple * ATR(14)``. This is the
default from the contract because a fixed point distance is meaningless
across a price series that moved from 16,000 to 73,000 inside the sample.
The ATR is computed from the bars available so far and is scaled by the
current price so the stop distance is expressed in the same units as
the instrument.

Reward-to-risk
--------------
2.0, the contract default, because the idea did not specify one.

Session handling
----------------
Crypto trades 24/7, so the session is the UTC calendar day
(``session_open_hour=0``). The strategy itself does not depend on the
session boundary, but the engine uses it for end-of-day force-closes and
the Monte Carlo uses it for the daily loss limit, so it has to be right.

What is NOT modelled
--------------------
Slippage, commission, funding costs, and the order of events inside a
bar. The engine assumes the stop fills before the target when a bar
touches both, which is the pessimistic choice.
"""
from __future__ import annotations

import numpy as np

from prop_lab.core import Bar, Signal, StrategyCore

__all__ = ["EmaTrendPullback"]


class EmaTrendPullback(StrategyCore):
    """Trend-filtered pullback continuation.

    Parameters:
        ema_fast: period of the fast EMA, the "pullback" level.
            Default 20.
        ema_slow: period of the slow EMA, the trend reference.
            Default 50.
        slope_bars: how many bars the slow EMA's slope is measured
            over. Default 10.
        atr_period: period of the ATR used for stop sizing.
            Default 14.
        atr_multiple: the stop distance as a multiple of ATR.
            Default 1.5.
        rr_ratio: target distance as a multiple of the stop distance.
            Default 2.0.
        direction: 'both', 'long' or 'short'. Default 'both'.
        warmup_bars: how many bars to wait before the indicators are
            considered valid. Default 100.
    """

    name = "ETP"

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        slope_bars: int = 10,
        atr_period: int = 14,
        atr_multiple: float = 1.5,
        rr_ratio: float = 2.0,
        direction: str = "both",
        warmup_bars: int = 100,
        **kwargs,
    ):
        super().__init__(
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            slope_bars=slope_bars,
            atr_period=atr_period,
            atr_multiple=atr_multiple,
            rr_ratio=rr_ratio,
            direction=direction,
            warmup_bars=warmup_bars,
        )
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.slope_bars = slope_bars
        self.atr_period = atr_period
        self.atr_multiple = atr_multiple
        self.rr_ratio = rr_ratio
        self.direction = direction
        self.warmup_bars = warmup_bars

        # Indicator buffers.
        self._closes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []

        self.notes = (
            f"trend = EMA({ema_fast}) > EMA({ema_slow}) and the slow EMA's slope "
            f"over {slope_bars} bars is positive (mirrored for shorts); "
            f"pullback = the bar's low <= EMA({ema_fast}) for longs "
            f"(high >= EMA({ema_fast}) for shorts); "
            f"confirmation = the rejection candle on the same bar "
            f"(close > open for longs, close < open for shorts); "
            f"stop = {atr_multiple} x ATR({atr_period}); "
            f"rr_ratio = {rr_ratio}; "
            f"direction = {direction}; "
            f"session = UTC calendar day (crypto trades 24/7, "
            f"session_open_hour=0); "
            f"not modelled: slippage, commission, funding, intrabar order of events"
        )

    # -- session hooks -------------------------------------------------
    def on_session_start(self, bar: Bar) -> None:
        """No per-session state to reset: the indicators are continuous.

        The hook is overridden only to document that the indicator
        buffers deliberately span session boundaries.
        """
        return None

    # -- indicators ----------------------------------------------------
    def _ema(self, period: int) -> float | None:
        """Exponential moving average of the closes seen so far."""
        if len(self._closes) < period:
            return None
        alpha = 2.0 / (period + 1.0)
        value = self._closes[0]
        for price in self._closes[1:]:
            value = alpha * price + (1.0 - alpha) * value
        return value

    def _atr(self) -> float | None:
        """Average true range over ``atr_period`` bars."""
        n = self.atr_period
        if len(self._closes) < n + 1:
            return None
        trs: list[float] = []
        for i in range(len(self._closes) - n, len(self._closes)):
            high = self._highs[i]
            low = self._lows[i]
            prev_close = self._closes[i - 1]
            trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        return float(np.mean(trs))

    # -- bar handling --------------------------------------------------
    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        """Emit a signal on the close of a qualifying bar."""
        self._closes.append(bar.close)
        self._highs.append(bar.high)
        self._lows.append(bar.low)

        if bar_index < self.warmup_bars:
            return None

        ema_fast = self._ema(self.ema_fast)
        ema_slow = self._ema(self.ema_slow)
        atr = self._atr()
        if ema_fast is None or ema_slow is None or atr is None:
            return None

        # Slope of the slow EMA: compare it to its own value `slope_bars`
        # ago. Computing the EMA at that point needs a second pass, so
        # the buffer is truncated rather than re-indexed.
        n_slow = len(self._closes)
        ema_slow_past = self._ema_at(n_slow - self.slope_bars, self.ema_slow)
        if ema_slow_past is None:
            return None
        slope_up = ema_slow > ema_slow_past
        slope_down = ema_slow < ema_slow_past

        stop_distance = self.atr_multiple * atr
        if stop_distance <= 0:
            return None

        # Long: uptrend, pullback into the fast EMA, rejection candle.
        if (
            self.direction in ("both", "long")
            and ema_fast > ema_slow
            and slope_up
            and bar.low <= ema_fast
            and bar.close > bar.open
        ):
            return Signal(
                direction="long",
                stop_distance=stop_distance,
                rr_ratio=self.rr_ratio,
                reason="etp_long_trend_pullback",
                bar_index=bar_index,
                timestamp=bar.timestamp,
            )

        # Short: downtrend, rally into the fast EMA, rejection candle.
        if (
            self.direction in ("both", "short")
            and ema_fast < ema_slow
            and slope_down
            and bar.high >= ema_fast
            and bar.close < bar.open
        ):
            return Signal(
                direction="short",
                stop_distance=stop_distance,
                rr_ratio=self.rr_ratio,
                reason="etp_short_trend_pullback",
                bar_index=bar_index,
                timestamp=bar.timestamp,
            )

        return None

    def _ema_at(self, count: int, period: int) -> float | None:
        """EMA of the first ``count`` closes."""
        if count < period:
            return None
        alpha = 2.0 / (period + 1.0)
        value = self._closes[0]
        for price in self._closes[1:count]:
            value = alpha * price + (1.0 - alpha) * value
        return value
