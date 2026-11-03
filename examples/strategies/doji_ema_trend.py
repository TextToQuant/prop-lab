"""Doji entries in an EMA trend, ATR-based stop and target.

The idea, in plain language
---------------------------
When the fast EMA is above the slow EMA the market is in an uptrend; when
the slow EMA is above the fast EMA it is in a downtrend. A doji candle
inside that trend is treated as a pause, and the trade takes the trend's
direction: long above, short below.

The stop sits three ATR away and the target two ATR away. Because the
target is closer than the stop, the reward-to-risk ratio is 2/3, so the
strategy needs a win rate above 60 percent just to break even. That is
stated here rather than buried, because it decides whether the result
means anything.

The logic
---------
1. Track EMA50 and EMA150 on closes, and ATR(14) on true range.
2. A bar is a doji when its body (|close - open|) is at most
   ``body_ratio`` of its full range (high - low). A bar with no range is
   never a doji.
3. When the fast EMA is above the slow EMA and the bar is a doji, go
   long. When the slow EMA is above the fast EMA and the bar is a doji,
   go short.
4. Stop distance is ``stop_atr_mult`` times ATR; the target is
   ``target_atr_mult`` times ATR, so the reward-to-risk ratio is
   ``target_atr_mult / stop_atr_mult``.

Run it with::

    proplab backtest --strategy examples/strategies/doji_ema_trend.py:DojiEmaTrend \\
        --data data/BTCUSDT_15m_sessions.csv
"""
from __future__ import annotations

from prop_lab.core import Bar, Signal, StrategyCore

__all__ = ["DojiEmaTrend"]


class DojiEmaTrend(StrategyCore):
    """Doji entries in an EMA trend with an ATR stop and ATR target.

    Parameters:
        fast_ema: period of the fast EMA on closes. Default 50.
        slow_ema: period of the slow EMA on closes. Default 150.
        atr_period: period of the ATR on true range. Default 14.
        body_ratio: a bar is a doji when |close - open| is at most this
            fraction of (high - low). Default 0.10, which is the
            conventional "the body is a tenth of the range or less".
        stop_atr_mult: stop distance in ATR. Default 3.0.
        target_atr_mult: target distance in ATR. Default 2.0.
        rr_ratio: optional override of the reward-to-risk ratio. When
            None (the default) it is derived from the ATR multiples.
            Set it to use the CLI's ``--rr`` flag.
        direction: 'both', 'long' or 'short'. Default 'both'.
    """

    name = "DOJI_EMA"

    def __init__(
        self,
        fast_ema: int = 50,
        slow_ema: int = 150,
        atr_period: int = 14,
        body_ratio: float = 0.10,
        stop_atr_mult: float = 3.0,
        target_atr_mult: float = 2.0,
        rr_ratio: float | None = None,
        direction: str = "both",
        **kwargs,
    ):
        super().__init__(
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            atr_period=atr_period,
            body_ratio=body_ratio,
            stop_atr_mult=stop_atr_mult,
            target_atr_mult=target_atr_mult,
            direction=direction,
        )
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.body_ratio = body_ratio
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.direction = direction

        # The reward-to-risk ratio is fixed by the ATR multiples unless
        # the caller overrides it.
        self.rr_ratio = (
            rr_ratio
            if rr_ratio is not None
            else target_atr_mult / stop_atr_mult
        )

        # Incremental indicator state.
        self._fast_k = 2.0 / (fast_ema + 1.0)
        self._slow_k = 2.0 / (slow_ema + 1.0)
        self._fast_sum = 0.0
        self._slow_sum = 0.0
        self._fast_ema: float | None = None
        self._slow_ema: float | None = None
        self._fast_n = 0
        self._slow_n = 0

        self._tr_sum = 0.0
        self._atr: float | None = None
        self._tr_n = 0
        self._prev_close: float | None = None

        self.notes = (
            "doji = |close - open| <= {body_ratio} of (high - low); a bar with "
            "zero range is never a doji. trend = EMA{fast_ema} versus "
            "EMA{slow_ema} on closes, evaluated on the doji bar itself. "
            "stop = {stop_atr_mult} x ATR({atr_period}), target = "
            "{target_atr_mult} x ATR({atr_period}), so reward-to-risk = "
            "{rr:.4f} and the strategy needs a win rate above {breakeven:.1f} "
            "percent to break even. entry is on the doji bar close, filled "
            "at the open of the next bar."
        ).format(
            body_ratio=body_ratio,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            stop_atr_mult=stop_atr_mult,
            atr_period=atr_period,
            target_atr_mult=target_atr_mult,
            rr=self.rr_ratio,
            breakeven=100.0 / (1.0 + self.rr_ratio),
        )

    # -- indicators ----------------------------------------------------
    def _update_emas(self, close: float) -> None:
        """Seed each EMA with the SMA of its first period, then roll it.

        The two EMAs are tracked independently so that the slow one
        seeds on its own first ``slow_ema`` closes rather than being
        delayed by the fast one.
        """
        if self._fast_ema is None:
            self._fast_sum += close
            self._fast_n += 1
            if self._fast_n == self.fast_ema:
                self._fast_ema = self._fast_sum / self.fast_ema
        else:
            self._fast_ema = close * self._fast_k + self._fast_ema * (1.0 - self._fast_k)

        if self._slow_ema is None:
            self._slow_sum += close
            self._slow_n += 1
            if self._slow_n == self.slow_ema:
                self._slow_ema = self._slow_sum / self.slow_ema
        else:
            self._slow_ema = close * self._slow_k + self._slow_ema * (1.0 - self._slow_k)

    def _update_atr(self, bar: Bar) -> None:
        """Wilder-smoothed ATR on true range."""
        true_range = bar.high - bar.low
        if self._prev_close is not None:
            true_range = max(
                true_range,
                abs(bar.high - self._prev_close),
                abs(bar.low - self._prev_close),
            )
        self._prev_close = bar.close

        if self._atr is None:
            self._tr_sum += true_range
            self._tr_n += 1
            if self._tr_n == self.atr_period:
                self._atr = self._tr_sum / self.atr_period
            return
        self._atr = (self._atr * (self.atr_period - 1) + true_range) / self.atr_period

    # -- strategy ------------------------------------------------------
    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        """Emit a signal on a doji bar inside an EMA trend."""
        self._update_emas(bar.close)
        self._update_atr(bar)

        if self._fast_ema is None or self._slow_ema is None or self._atr is None:
            return None
        if self._atr <= 0:
            return None

        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return None
        body = abs(bar.close - bar.open)
        if body > self.body_ratio * bar_range:
            return None

        stop_distance = self.stop_atr_mult * self._atr

        if self._fast_ema > self._slow_ema and self.direction in ("both", "long"):
            return Signal(
                direction="long",
                stop_distance=stop_distance,
                rr_ratio=self.rr_ratio,
                reason="doji_ema_long",
                bar_index=bar_index,
                timestamp=bar.timestamp,
            )

        if self._slow_ema > self._fast_ema and self.direction in ("both", "short"):
            return Signal(
                direction="short",
                stop_distance=stop_distance,
                rr_ratio=self.rr_ratio,
                reason="doji_ema_short",
                bar_index=bar_index,
                timestamp=bar.timestamp,
            )

        return None
