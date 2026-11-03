"""Yahoo Finance market-data adapter, via yfinance.

Source:
    Yahoo Finance, through the ``yfinance`` package
    (``yf.download``). Yahoo publishes daily and intraday history for
    equities, indices, FX, crypto pairs and continuous futures contracts.

Coverage:
    Depends on the symbol and the interval.

    ==========  =========================================================
    interval    history available
    ==========  =========================================================
    1m          last 30 days (8 days with ``period='max'``)
    2m, 5m,     60 days
    15m, 30m
    60m, 1h     730 days
    1d and      full history (no limit)
    coarser
    ==========  =========================================================

    These limits were read from the yfinance source on 2026-10-10 and
    confirmed against the live API on 2026-11-03.

Caveats:
    - **Unadjusted rolls.** Yahoo's continuous futures (``NQ=F``,
      ``MNQ=F``, ``ES=F``, ...) are NOT back-adjusted. The series switches
      from one contract to the next at expiry and leaves a price gap of
      roughly 1 percent that is a fake move. Any strategy that reads
      those gaps as real price action will produce nonsense. This
      adapter warns loudly by default.
    - ``auto_adjust=False`` is used deliberately, so the prices are the
      raw traded prices and splits/dividends are not baked in.
    - Yahoo intraday history is a moving window: a request for 1m data
      from last year returns nothing rather than an error.
    - Yahoo has no official API contract here and can rate-limit.

Verified live: 2026-11-03.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["fetch", "INTERVALS", "YAHOO_INTERVALS", "is_continuous_future"]

#: Timeframes this adapter accepts.
INTERVALS: tuple[str, ...] = ("1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d")

#: Mapping from the prop-lab shorthand to the yfinance interval string.
YAHOO_INTERVALS: dict[str, str] = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
    "1h": "1h",
    "1d": "1d",
}

_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def is_continuous_future(symbol: str) -> bool:
    """Report whether a Yahoo symbol is a continuous futures contract.

    Args:
        symbol: the Yahoo ticker, e.g. ``'NQ=F'``.

    Returns:
        True when the symbol ends with ``'=F'``, which is Yahoo's marker
        for a continuous front-month futures series.
    """
    return str(symbol).strip().upper().endswith("=F")


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten a yfinance MultiIndex column frame into flat columns.

    yfinance returns a MultiIndex column frame even for a single ticker.
    The price names sit either at level 0 (``'Close'``) or at level 1
    (``('Close', 'NQ=F')``) depending on the yfinance version, so both
    shapes are handled here.

    Args:
        frame: the frame returned by ``yf.download``.

    Returns:
        A frame with flat ``Open/High/Low/Close/Volume`` columns.
    """
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    # Prefer the level that carries the field names.
    pick = 0
    for level in range(frame.columns.nlevels):
        values = {str(v).strip().lower() for v in frame.columns.get_level_values(level)}
        if {"open", "high", "low", "close"}.issubset(values):
            pick = level
            break

    flat_index = frame.columns.to_flat_index()
    out = frame.copy()
    out.columns = [
        str(v[pick]) if isinstance(v, tuple) else str(v) for v in flat_index
    ]
    # Drop duplicated names produced by the flattening (e.g. two tickers).
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def _normalise_field(frame: pd.DataFrame, name: str) -> pd.Series:
    """Pull a price field out of a flattened frame.

    Args:
        frame: the flattened frame.
        name: the wanted field name, e.g. ``'Close'``.

    Returns:
        The matching Series, or an all-NaN Series when the field is
        absent (Yahoo omits ``Volume`` for some instruments).
    """
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    key = name.lower()
    if key in lowered:
        return frame[lowered[key]]
    for candidate in frame.columns:
        if isinstance(candidate, tuple):
            for part in candidate:
                if str(part).strip().lower() == key:
                    return frame[candidate]
    return pd.Series(np.nan, index=frame.index)


def _as_yahoo_bound(value: Any) -> str | None:
    """Format a start/end argument the way yfinance expects it.

    Args:
        value: None, a datetime-like value or a ``'YYYY-MM-DD'`` string.

    Returns:
        A ``'YYYY-MM-DD'`` string, or None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def fetch(
    symbol: str = "NQ=F",
    interval: str = "1d",
    start: Any = None,
    end: Any = None,
    *,
    warn_unadjusted_roll: bool = True,
) -> pd.DataFrame:
    """Fetch bars for one Yahoo Finance symbol.

    Source:
        Yahoo Finance via the ``yfinance`` package. ``yf.download`` is
        called with ``auto_adjust=False`` so the raw traded prices are
        returned (splits and dividends are not adjusted into them).

    Coverage:
        Full history for daily and coarser intervals. Intraday history
        is a moving window: 1m covers the last 30 days, 2m/5m/15m/30m
        cover 60 days, 60m/1h cover 730 days.

    Caveats:
        - **Unadjusted continuous futures.** Yahoo's ``=F`` symbols switch
          contract at expiry and leave a ~1 percent gap that is not a real
          price move. A ``warnings.warn`` naming the symbol is emitted
          unless ``warn_unadjusted_roll=False``.
        - Yahoo intraday windows move; old 1m data is simply gone.
        - Yahoo may rate-limit or return an empty frame for an unknown
          symbol.
        - Crypto on Yahoo (``BTC-USD``) trades 24/7, so a UTC-day session
          label is the right grouping there.

    Args:
        symbol: Yahoo ticker, e.g. ``'NQ=F'``, ``'^VIX'``, ``'BTC-USD'``.
        interval: one of :data:`INTERVALS`.
        start: inclusive start, datetime or ``'YYYY-MM-DD'`` string.
        end: exclusive end, datetime or ``'YYYY-MM-DD'`` string (this is
            Yahoo's own convention).
        warn_unadjusted_roll: emit the unadjusted-roll warning for
            continuous futures symbols.

    Returns:
        A frame in the canonical schema (``time`` in UNIX seconds UTC),
        sorted ascending, tagged with ``source='yahoo'``.

    Verified live: 2026-11-03.
    """
    import yfinance as yf

    key = str(interval).strip().lower()
    if key not in YAHOO_INTERVALS:
        raise ValueError(
            f"Unsupported interval {interval!r}. Supported: {sorted(YAHOO_INTERVALS)}"
        )

    if warn_unadjusted_roll and is_continuous_future(symbol):
        warnings.warn(
            f"{symbol} is a Yahoo continuous futures contract and is UNADJUSTED: "
            "the series rolls from one contract to the next at expiry and leaves a "
            "price gap of roughly 1 percent that is a fake move, not real price "
            "action. Treat any signal triggered by such a gap as invalid, or use "
            "an adjusted source instead.",
            UserWarning,
            stacklevel=2,
        )

    raw = yf.download(
        symbol,
        interval=YAHOO_INTERVALS[key],
        start=_as_yahoo_bound(start),
        end=_as_yahoo_bound(end),
        auto_adjust=False,
        progress=False,
    )

    if raw is None or len(raw) == 0:
        return pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "volume", "source"]
        )

    flat = _flatten_columns(raw)

    data = {}
    for wanted, name in zip(_OHLCV, ("open", "high", "low", "close", "volume")):
        data[name] = _normalise_field(flat, wanted)

    frame = pd.DataFrame(data, index=flat.index)

    index = frame.index
    if isinstance(index, pd.DatetimeIndex):
        if index.tz is None:
            index = index.tz_localize("UTC")
        else:
            index = index.tz_convert("UTC")
        frame["time"] = (index.astype("int64") // 10**9).astype("int64")
    else:
        frame["time"] = (
            pd.to_datetime(index, utc=True).astype("int64") // 10**9
        ).astype("int64")

    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
    frame["volume"] = frame["volume"].fillna(0.0)

    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame.sort_values("time", kind="mergesort")
    frame = frame.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    frame["source"] = "yahoo"
    return frame[["time", "open", "high", "low", "close", "volume", "source"]]
