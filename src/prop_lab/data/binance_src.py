"""Binance public REST market-data adapter.

Source:
    ``GET https://api.binance.com/api/v3/klines`` -- the public spot
    klines endpoint. No API key is required and no account is touched.

Coverage:
    Crypto spot pairs (BTCUSDT, ETHUSDT, ...) with full history from the
    listing date of the pair to now. The market trades 24/7, so there are
    no session gaps and no holidays. Intraday intervals are available
    from 1 minute upwards.

Caveats:
    - Binance returns the *spot* market. Futures (perp) prices differ.
    - The kline volume field is base-asset volume, not quote volume.
    - Binance caps a single request at 1000 bars, so long windows are
      paged. A short sleep is inserted between pages to stay polite.
    - The API has no fixed historical end; ``end`` defaults to "now".

Verified live: 2026-11-03.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

__all__ = ["fetch", "INTERVALS", "BINANCE_INTERVALS"]

#: Timeframes this adapter accepts, in the shorthand used across prop-lab.
INTERVALS: tuple[str, ...] = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
)

#: Mapping from the prop-lab shorthand to the Binance interval string.
BINANCE_INTERVALS: dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
}

#: Binance returns at most 1000 klines per request.
_PAGE_LIMIT = 1000

#: Seconds to sleep between paged requests.
_PAGE_SLEEP = 0.25

#: Columns of a Binance kline row, in order.
_KLINE_FIELDS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
)


def _to_epoch_ms(value: Any) -> int | None:
    """Convert a start/end argument to epoch milliseconds.

    Args:
        value: None, a datetime-like value or a ``'YYYY-MM-DD'`` string.

    Returns:
        Epoch milliseconds, or None when ``value`` is None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    stamp = pd.to_datetime(value)
    stamp = pd.Timestamp(stamp)
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return int(stamp.value // 10**6)


def fetch(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start: Any = None,
    end: Any = None,
    *,
    base_url: str = "https://api.binance.com",
) -> pd.DataFrame:
    """Fetch spot klines from the Binance public REST API.

    Source:
        ``GET {base_url}/api/v3/klines`` with the query parameters
        ``symbol``, ``interval``, ``limit``, ``startTime`` and ``endTime``.
        The endpoint is public: no API key, no signature, no account.

    Coverage:
        Full history for any listed spot pair, 24/7, from listing to
        now. Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h,
        1d.

    Caveats:
        - Spot market only. Binance perpetual futures are a different
          endpoint and different prices.
        - One request returns at most 1000 bars; longer windows are
          paged forward with a 0.25 s sleep between pages.
        - ``end`` is inclusive on the Binance side; the adapter trims the
          final page so the returned frame never contains bars beyond
          ``end``.
        - Rate limits (1200 weight/min) apply; a heavy paging loop can
          trip them.

    Args:
        symbol: spot pair, e.g. ``'BTCUSDT'``.
        interval: one of :data:`INTERVALS`.
        start: inclusive start, datetime or ``'YYYY-MM-DD'`` string.
        end: inclusive end, datetime or ``'YYYY-MM-DD'`` string.
        base_url: API root, overridable for tests or mirrors.

    Returns:
        A frame in the canonical schema (``time`` in UNIX seconds UTC),
        sorted ascending, tagged with ``source='binance'``.

    Raises:
        ValueError: if ``interval`` is not supported.
        RuntimeError: if the API returns a non-200 status.

    Verified live: 2026-11-03.
    """
    import requests

    key = str(interval).strip().lower()
    if key not in BINANCE_INTERVALS:
        raise ValueError(
            f"Unsupported interval {interval!r}. Supported: {sorted(BINANCE_INTERVALS)}"
        )
    binance_interval = BINANCE_INTERVALS[key]

    start_ms = _to_epoch_ms(start)
    end_ms = _to_epoch_ms(end)

    url = base_url.rstrip("/") + "/api/v3/klines"
    rows: list[list[Any]] = []

    # Paging cursor, in milliseconds.
    cursor = start_ms
    while True:
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": binance_interval,
            "limit": _PAGE_LIMIT,
        }
        if cursor is not None:
            params["startTime"] = cursor
        if end_ms is not None:
            params["endTime"] = end_ms

        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                "Binance request failed: HTTP "
                f"{response.status_code} for {url} params={params}. "
                f"Body: {response.text[:300]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Binance returned a non-JSON body for {url}: {response.text[:300]}"
            ) from exc

        if not isinstance(payload, list):
            raise RuntimeError(
                f"Binance returned an unexpected payload for {url}: {str(payload)[:300]}"
            )
        if not payload:
            break

        rows.extend(payload)

        if len(payload) < _PAGE_LIMIT:
            break

        last_open = int(payload[-1][0])
        next_cursor = last_open + 1
        if cursor is not None and next_cursor <= cursor:
            break
        cursor = next_cursor
        if end_ms is not None and cursor > end_ms:
            break
        time.sleep(_PAGE_SLEEP)

    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "source"])

    frame = pd.DataFrame(rows, columns=list(_KLINE_FIELDS))
    frame["time"] = (frame["open_time"].astype("int64") // 1000).astype("int64")
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")

    if end_ms is not None:
        frame = frame[frame["time"] * 1000 <= end_ms]

    frame = frame[["time", "open", "high", "low", "close", "volume"]]
    frame = frame.sort_values("time", kind="mergesort")
    frame = frame.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    frame["source"] = "binance"
    return frame
