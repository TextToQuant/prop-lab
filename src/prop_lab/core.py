"""Core trading primitives: Bar, Signal, StrategyCore, and the backtest engine.

This module is the single source of truth for trade generation. Nothing else
in the package may re-implement it.

The fill model is deliberately conservative:
  - a signal emitted on bar ``i`` is filled at the OPEN of bar ``i+1``;
  - if a bar touches both the stop and the target, the STOP is assumed to
    have filled first (pessimistic assumption);
  - open positions are force-closed at the close of the last bar of their
    session when ``force_close_eod`` is set.

Architecture lineage: the StrategyCore pattern originates from
https://github.com/dsilva42/prop-risk-geometry (originally
hunter547/prop-risk-geometry), which in turn popularised the
"bootstrap the real trade distribution" approach to prop-firm challenge
simulation. See README.md for the full citation list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

__all__ = [
    "Bar",
    "Signal",
    "StrategyCore",
    "Trade",
    "iter_bars",
    "run_backtest",
    "summarize_trades",
]


# ----------------------------------------------------------------------
# Bar
# ----------------------------------------------------------------------
@dataclass
class Bar:
    """One normalised OHLCV bar.

    ``timestamp`` is UNIX seconds UTC and refers to the bar's OPEN time.
    """

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def date(self) -> str:
        """UTC date of the bar, 'YYYY-MM-DD'."""
        import datetime as dt

        return dt.datetime.fromtimestamp(
            self.timestamp, tz=dt.timezone.utc
        ).strftime("%Y-%m-%d")

    @property
    def time_utc(self) -> str:
        """UTC time of the bar, 'HH:MM'."""
        import datetime as dt

        return dt.datetime.fromtimestamp(
            self.timestamp, tz=dt.timezone.utc
        ).strftime("%H:%M")


# ----------------------------------------------------------------------
# Signal
# ----------------------------------------------------------------------
@dataclass
class Signal:
    """An entry signal emitted by a strategy core on a bar close.

    Attributes:
        direction: 'long' or 'short'.
        stop_distance: distance to the stop in price points, always > 0.
        rr_ratio: reward-to-risk; the target sits at
            ``stop_distance * rr_ratio`` points from entry.
        reason: short machine-readable tag, e.g. 'orb_long_breakout'.
        bar_index: index of the bar the signal was emitted on. The fill
            happens at the open of bar ``bar_index + 1``.
        timestamp: optional UNIX seconds of the signal bar.
    """

    direction: str
    stop_distance: float
    rr_ratio: float
    reason: str
    bar_index: int
    timestamp: int | None = None


# ----------------------------------------------------------------------
# Strategy core
# ----------------------------------------------------------------------
class StrategyCore:
    """Base class for every strategy.

    A strategy core receives bars one at a time and emits :class:`Signal`
    objects. It knows nothing about fills, position tracking or risk --
    that all lives in :func:`run_backtest`.

    Subclasses override :meth:`on_bar` and optionally
    :meth:`on_session_start` / :meth:`on_session_end` to reset per-session
    state.
    """

    name: str = "base"

    def __init__(self, **params):
        self.params = params
        self._session_date: str | None = None

    # -- session hooks -------------------------------------------------
    def on_session_start(self, bar: Bar) -> None:
        """Called on the first bar of a new session. Reset state here."""
        self._session_date = bar.date

    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        """Process one bar. Return a Signal or None.

        A returned signal is filled at the open of the NEXT bar.
        """
        raise NotImplementedError

    def on_session_end(self, bar: Bar) -> None:
        """Called on the last bar of a session."""
        return None

    # -- discovery -----------------------------------------------------
    @classmethod
    def params_help(cls) -> str:
        """Human-readable description of the strategy's parameters.

        Introspects ``__init__`` so every subclass gets this for free.
        """
        import inspect

        sig = inspect.signature(cls.__init__)
        lines = [f"{cls.__name__} parameters:"]
        for name, p in sig.parameters.items():
            if name in ("self", "kwargs"):
                continue
            default = (
                "REQUIRED" if p.default is inspect.Parameter.empty else repr(p.default)
            )
            lines.append(f"  {name}: default={default}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Bar stream
# ----------------------------------------------------------------------
def iter_bars(df: pd.DataFrame, ts_col: str = "time") -> Iterator[tuple[int, Bar]]:
    """Yield ``(bar_index, Bar)`` pairs from an OHLCV frame.

    The time column is looked up case-insensitively and falls back to 'ts'
    when 'time' is absent, because different free data sources disagree on
    the name.
    """
    cols = {c.lower(): c for c in df.columns}
    if ts_col in cols:
        ts_c = cols[ts_col]
    elif "time" in cols:
        ts_c = cols["time"]
    elif "ts" in cols:
        ts_c = cols["ts"]
    else:
        raise KeyError(
            f"No 'time' or 'ts' column found. Available columns: {list(df.columns)}"
        )
    o_c = cols.get("open", "open")
    h_c = cols.get("high", "high")
    l_c = cols.get("low", "low")
    c_c = cols.get("close", "close")
    v_c = cols.get("volume", "volume")

    for i, row in df.iterrows():
        yield i, Bar(
            timestamp=int(row[ts_c]),
            open=float(row[o_c]),
            high=float(row[h_c]),
            low=float(row[l_c]),
            close=float(row[c_c]),
            volume=float(row[v_c]) if v_c in row and not pd.isna(row[v_c]) else 0.0,
        )


# ----------------------------------------------------------------------
# Trade
# ----------------------------------------------------------------------
@dataclass
class Trade:
    """One completed round-trip trade."""

    direction: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    stop_distance: float
    rr_ratio: float
    pnl_points: float
    pnl_r: float
    reason: str
    session_date: str = ""
    exit_reason: str = ""  # 'target' | 'stop' | 'session_close' | 'data_end'


# ----------------------------------------------------------------------
# Backtest engine
# ----------------------------------------------------------------------
def run_backtest(
    strategy: StrategyCore,
    df: pd.DataFrame,
    *,
    direction: str = "both",
    use_sessions: bool = True,
    force_close_eod: bool = True,
    risk_points: float | None = None,
) -> list[Trade]:
    """Run a bar-by-bar simulation of ``strategy`` over ``df``.

    Args:
        strategy: a :class:`StrategyCore` instance. Its state is reset per
            session, so one instance can be reused across sessions.
        df: frame in the standard schema (see ``prop_lab.data``). Must be
            sorted by ``time`` ascending.
        direction: 'both', 'long' or 'short'. Filters which signals are
            taken.
        use_sessions: call ``on_session_start`` / ``on_session_end`` when the
            session changes. Sessions come from the ``session_date``
            column when present, otherwise from the bar's UTC date.
        force_close_eod: close any open position at the close of the last
            bar of its session.
        risk_points: optional fixed dollar risk per trade, recorded for
            reference. The engine itself is risk-agnostic -- position
            sizing happens later in the Monte Carlo layer.

    Returns:
        A list of :class:`Trade` objects in chronological order.

    Notes:
        Fill model: a signal on bar ``i`` fills at the open of bar ``i+1``.
        If a bar's range touches both the stop and the target, the stop is
        assumed to have filled first.
    """
    if direction not in ("both", "long", "short"):
        raise ValueError(f"direction must be 'both', 'long' or 'short', got {direction!r}")

    bars = list(iter_bars(df))
    if not bars:
        return []

    # Session key per bar.
    if use_sessions and "session_date" in df.columns:
        session_keys = df["session_date"].astype(str).tolist()
    else:
        session_keys = [b.date for _, b in bars]

    trades: list[Trade] = []
    pending: Signal | None = None      # signal awaiting the next bar's open
    position: dict | None = None       # open position

    n = len(bars)
    for i, (idx, bar) in enumerate(bars):
        session = session_keys[i]

        # -- session boundary -----------------------------------------
        if use_sessions and getattr(strategy, "_session_date", None) != session:
            if position is not None and force_close_eod:
                trade = _close(position, bar.close, bar.timestamp, session, "session_close")
                trades.append(trade)
                position = None
            if getattr(strategy, "_session_date", None) is not None:
                strategy.on_session_end(bars[i - 1][1])
            strategy.on_session_start(bar)
            strategy._session_date = session

        # -- manage the open position --------------------------------
        if position is not None:
            closed = _check_exit(position, bar)
            if closed is not None:
                trades.append(closed)
                position = None

        # -- fill a pending signal at this bar's open ----------------
        if pending is not None and position is None:
            if direction == "both" or pending.direction == direction:
                position = _open(pending, bar, session, idx)
            pending = None

        # -- ask the strategy for a new signal ------------------------
        if position is None:
            sig = strategy.on_bar(bar, idx)
            if sig is not None:
                if sig.direction not in ("long", "short"):
                    raise ValueError(
                        f"Signal.direction must be 'long' or 'short', got {sig.direction!r}"
                    )
                if sig.stop_distance <= 0:
                    raise ValueError(
                        f"Signal.stop_distance must be > 0, got {sig.stop_distance}"
                        " (strategy: %s)" % getattr(strategy, "name", "?")
                    )
                pending = sig

    # -- close anything still open at the end of the data -------------
    if position is not None:
        last_bar = bars[-1][1]
        trades.append(
            _close(position, last_bar.close, last_bar.timestamp, session_keys[-1], "data_end")
        )

    if use_sessions and getattr(strategy, "_session_date", None) is not None:
        strategy.on_session_end(bars[-1][1])

    return trades


def _open(sig: Signal, bar: Bar, session: str, bar_index: int) -> dict:
    """Open a position at the current bar's open price."""
    if sig.direction == "long":
        entry = bar.open
        stop = entry - sig.stop_distance
        target = entry + sig.stop_distance * sig.rr_ratio
    else:
        entry = bar.open
        stop = entry + sig.stop_distance
        target = entry - sig.stop_distance * sig.rr_ratio
    return {
        "direction": sig.direction,
        "entry_time": bar.timestamp,
        "entry_price": entry,
        "stop": stop,
        "target": target,
        "stop_distance": sig.stop_distance,
        "rr_ratio": sig.rr_ratio,
        "reason": sig.reason,
        "session_date": session,
        "entry_index": bar_index,
    }


def _check_exit(position: dict, bar: Bar) -> Trade | None:
    """Return a closed Trade if ``bar`` exits the open position."""
    direction = position["direction"]
    stop = position["stop"]
    target = position["target"]

    if direction == "long":
        hit_stop = bar.low <= stop
        hit_target = bar.high >= target
    else:
        hit_stop = bar.high >= stop
        hit_target = bar.low <= target

    if hit_stop:
        # Pessimistic: when a bar touches both, assume the stop filled.
        return _close(position, stop, bar.timestamp, position["session_date"], "stop")
    if hit_target:
        return _close(position, target, bar.timestamp, position["session_date"], "target")
    return None


def _close(
    position: dict, exit_price: float, exit_time: int, session: str, exit_reason: str
) -> Trade:
    """Build a Trade from an open position at ``exit_price``."""
    entry = position["entry_price"]
    direction = position["direction"]
    pnl_points = (exit_price - entry) if direction == "long" else (entry - exit_price)
    stop_distance = position["stop_distance"]
    pnl_r = pnl_points / stop_distance if stop_distance else 0.0
    return Trade(
        direction=direction,
        entry_time=position["entry_time"],
        exit_time=exit_time,
        entry_price=entry,
        exit_price=exit_price,
        stop_distance=stop_distance,
        rr_ratio=position["rr_ratio"],
        pnl_points=pnl_points,
        pnl_r=pnl_r,
        reason=position["reason"],
        session_date=position["session_date"],
        exit_reason=exit_reason,
    )


# ----------------------------------------------------------------------
# Summary helper
# ----------------------------------------------------------------------
def summarize_trades(trades: list[Trade]) -> dict:
    """Aggregate statistics for a list of trades.

    Returns a dict with the numbers that matter for judging a strategy:
    trade count, win rate, mean/median R, profit factor, expectancy and
    the largest drawdown measured in cumulative R.
    """
    if not trades:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "mean_r": 0.0,
            "median_r": 0.0,
            "total_r": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "max_drawdown_r": 0.0,
        }

    rs = np.array([t.pnl_r for t in trades], dtype=float)
    wins = rs[rs > 0]
    losses = rs[rs < 0]

    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0

    cum = np.cumsum(rs)
    running_max = np.maximum.accumulate(cum)
    max_dd = float((running_max - cum).max()) if len(cum) else 0.0

    return {
        "n_trades": int(len(rs)),
        "win_rate": float((rs > 0).mean()),
        "mean_r": float(rs.mean()),
        "median_r": float(np.median(rs)),
        "total_r": float(rs.sum()),
        "profit_factor": float(profit_factor),
        "expectancy_r": float(rs.mean()),
        "max_drawdown_r": max_dd,
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
    }
