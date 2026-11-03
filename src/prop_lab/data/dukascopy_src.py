"""Dukascopy tick-data adapter (.bi5 files).

Source:
    ``https://datafeed.dukascopy.com/datafeed/{INSTR}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5``
    -- the free public tick archive. The month in the path is
    **zero-based**, so ``06`` is July.

Coverage:
    Very long history for FX (EURUSD and friends go back many years) and
    for the CFD index instruments, one file per clock hour, each file
    holding every tick that printed in that hour. FX trades roughly 24
    hours a day, five days a week; the CFD indices trade around the clock
    with a daily break.

Caveats:
    - **Indices on Dukascopy are CFDs, not futures.** ``USA30IDXUSD`` is
      a cash-index CFD; it is not the CME E-mini Dow, and its price
      formation, session hours and spreads differ from the futures
      contract. Do not mix the two in one backtest.
    - A quiet hour (weekend, daily break, holiday) returns **HTTP 200
      with an empty body**. That is "no data", not an error, and it is
      never retried.
    - The server silently slows down when more than about 4 files are
      fetched in parallel. Measured on 2026-11-03: 4 parallel hour
      files took 33 s, a single file took about 19 s.
    - The **monthly bulk .zip endpoint times out** at 120 s and is
      therefore not used here.
    - Volume here is tick volume (size attached to the bid/ask quotes),
      not exchange-cleared volume.
    - The price divisor differs per instrument; the default map below
      covers the common cases, and ``price_divisor`` overrides it.

Verified live: 2026-11-03.
"""
from __future__ import annotations

import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .schema import _to_epoch_seconds

__all__ = [
    "fetch",
    "coverage_report",
    "DEFAULT_PRICE_DIVISORS",
    "TIMEFRAME_RULES",
    "BI5_RECORD",
]

#: ``struct`` format of one decompressed .bi5 record (20 bytes).
BI5_RECORD = ">IIIff"

#: Price divisor per instrument. FX uses 100000; USDJPY, XAUUSD and the
#: index CFDs use 1000.
DEFAULT_PRICE_DIVISORS: dict[str, int] = {
    "USA30IDXUSD": 1000,
    "USA500IDXUSD": 1000,
    "USATECHIDXUSD": 1000,
    "XAUUSD": 1000,
    "USDJPY": 1000,
}

#: Default divisor for any instrument not listed above.
_FALLBACK_DIVISOR = 100000

#: Prop-lab timeframe shorthand to pandas offset alias.
TIMEFRAME_RULES: dict[str, str] = {
    "1m": "1min",
    "2m": "2min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1D",
}

_BASE_URL = "https://datafeed.dukascopy.com/datafeed"


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------
def _hour_url(instrument: str, hour: datetime) -> str:
    """Build the .bi5 URL for one clock hour.

    Args:
        instrument: Dukascopy instrument name, e.g. ``'EURUSD'``.
        hour: the UTC hour to fetch.

    Returns:
        The full URL. The month is zero-based.
    """
    return (
        f"{_BASE_URL}/{instrument}/{hour.year:04d}/{hour.month - 1:02d}/"
        f"{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"
    )


def _decode_hour(payload: bytes, hour: datetime, divisor: int) -> pd.DataFrame:
    """Decode one decompressed .bi5 blob into a tick frame.

    Args:
        payload: the decompressed bytes of one hour file.
        hour: the UTC hour the file covers.
        divisor: the price divisor for the instrument.

    Returns:
        A frame with ``time`` (UNIX seconds UTC), ``price`` (mid) and
        ``volume``. Empty when the payload holds no records.
    """
    if not payload:
        return pd.DataFrame(columns=["time", "price", "volume"])

    record_size = struct.calcsize(BI5_RECORD)
    usable = len(payload) - (len(payload) % record_size)
    if usable <= 0:
        return pd.DataFrame(columns=["time", "price", "volume"])

    # The byte-order prefix ">" is only legal at the very start of a
    # format string, so the format must NOT be repeated by string
    # multiplication (">IIIff>IIIff" raises "bad char in struct
    # format").
    #
    # ``Struct.unpack_from`` also only unpacks a SINGLE record, so it
    # cannot be used to read the whole file. ``iter_unpack`` walks every
    # fixed-size record from an offset, which is exactly what the
    # format needs.
    record_struct = struct.Struct(BI5_RECORD)
    records = record_struct.iter_unpack(payload[:usable])
    columns = list(zip(*records))

    ms = np.array(columns[0], dtype=np.int64)
    ask = np.array(columns[1], dtype=np.float64) / float(divisor)
    bid = np.array(columns[2], dtype=np.float64) / float(divisor)
    ask_vol = np.array(columns[3], dtype=np.float64)
    bid_vol = np.array(columns[4], dtype=np.float64)

    base = int(hour.replace(tzinfo=timezone.utc).timestamp())
    seconds = base + (ms // 1000).astype("int64")

    frame = pd.DataFrame(
        {
            "time": seconds,
            "price": (ask + bid) / 2.0,
            "volume": ask_vol + bid_vol,
        }
    )
    return frame


#: Transient HTTP statuses worth retrying (the server throttles hard).
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

#: Number of attempts for a transient failure.
_MAX_ATTEMPTS = 4

#: Seconds to sleep between retries, multiplied by the attempt number.
_RETRY_BACKOFF = 5.0

_EMPTY_TICKS = pd.DataFrame(columns=["time", "price", "volume"])


def _fetch_hour(
    instrument: str, hour: datetime, divisor: int, timeout: float
) -> tuple[datetime, pd.DataFrame, str | None]:
    """Download and decode one hour file.

    The server is throttled and answers with 503/504 or resets the
    connection under load. Those transient failures are retried a few
    times with a backoff. A 200 with an EMPTY body is the documented
    quiet-hour case: it is returned immediately and never retried, and
    it is not an error.

    Args:
        instrument: Dukascopy instrument name.
        hour: the UTC hour to fetch.
        divisor: price divisor.
        timeout: HTTP timeout in seconds.

    Returns:
        A tuple of (hour, ticks, error). ``error`` is None on success; an
        empty tick frame with a None error means the hour was quiet.
    """
    import requests

    url = _hour_url(instrument, hour)
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) prop-lab/0.1"}

    last_error: str | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(_RETRY_BACKOFF * attempt)
            continue

        status = response.status_code
        if status == 200:
            body = response.content or b""
            if len(body) == 0:
                # HTTP 200 with an empty body is the documented quiet-hour
                # case. Not an error, and never retried.
                return hour, _EMPTY_TICKS.copy(), None
            try:
                decompressed = lzma.decompress(body)
            except lzma.LZMAError as exc:
                return hour, _EMPTY_TICKS.copy(), f"LZMA decode failed: {exc}"
            return hour, _decode_hour(decompressed, hour, divisor), None

        if status == 404:
            # No file for this hour: quiet period or pre-history. Not an error.
            return hour, _EMPTY_TICKS.copy(), None

        if status in _RETRY_STATUSES and attempt < _MAX_ATTEMPTS:
            time.sleep(_RETRY_BACKOFF * attempt)
            continue

        return hour, _EMPTY_TICKS.copy(), f"HTTP {status}"

    return hour, _EMPTY_TICKS.copy(), last_error or "request failed"


def _resample_ticks(ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate ticks into OHLCV bars.

    The bar price is the mid price ``(bid + ask) / 2``; open is the first
    tick's mid, high and low are the extremes of the mid, close is the
    last mid, and volume is the sum of the tick volumes.

    Args:
        ticks: a frame with ``time``, ``price`` and ``volume``.
        timeframe: prop-lab timeframe shorthand.

    Returns:
        A frame in the canonical schema.
    """
    if ticks.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    rule = TIMEFRAME_RULES[timeframe]
    indexed = ticks.copy()
    indexed["_ts"] = pd.to_datetime(indexed["time"], unit="s", utc=True)
    indexed = indexed.set_index("_ts").sort_index()

    grouped = indexed["price"].resample(rule)
    out = pd.DataFrame(
        {
            "open": grouped.first(),
            "high": grouped.max(),
            "low": grouped.min(),
            "close": grouped.last(),
            "volume": indexed["volume"].resample(rule).sum(),
        }
    ).dropna(subset=["open"])
    out = out.reset_index()
    # The resampled index keeps the source resolution (microseconds in
    # pandas 3.0), so convert it to seconds explicitly rather than
    # assuming a nanosecond divisor.
    out["time"] = _to_epoch_seconds(pd.Series(out["_ts"]))
    out = out.drop(columns=["_ts"])
    out = out.sort_values("time", kind="mergesort").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = out[column].astype("float64")
    out["source"] = "dukascopy"
    return out[["time", "open", "high", "low", "close", "volume", "source"]]


def _parse_moment(value: Any, default: datetime) -> datetime:
    """Parse a start/end argument into an aware UTC datetime.

    Args:
        value: None, a datetime-like value or a ``'YYYY-MM-DD'`` string.
        default: value used when ``value`` is None.

    Returns:
        A timezone-aware UTC datetime.
    """
    if value is None:
        return default
    if isinstance(value, datetime):
        stamp = value
    else:
        stamp = pd.Timestamp(value).to_pydatetime()
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def coverage_report(df: pd.DataFrame) -> dict:
    """Return the coverage report attached to a Dukascopy frame.

    Args:
        df: a frame produced by :func:`fetch`.

    Returns:
        A dict with the keys ``hours_requested``, ``hours_with_data``,
        ``missing_hours``, ``instrument`` and ``timeframe``. An empty
        dict is returned when the frame carries no report.
    """
    report = df.attrs.get("coverage") if isinstance(df, pd.DataFrame) else None
    if not report:
        return {}
    return dict(report)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def fetch(
    instrument: str = "USA30IDXUSD",
    start: Any = None,
    end: Any = None,
    *,
    timeframe: str = "5m",
    max_workers: int = 4,
    price_divisor: int | None = None,
) -> pd.DataFrame:
    """Fetch tick data from Dukascopy and resample it to OHLCV bars.

    Source:
        ``https://datafeed.dukascopy.com/datafeed/{INSTR}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5``
        -- one LZMA-compressed file per clock hour, month zero-based.
        Each decompressed file is a sequence of 20-byte big-endian
        records with the ``'>IIIff'`` layout: milliseconds since the
        start of the hour, ask price, bid price, ask volume, bid volume.

    Coverage:
        Long history for FX (EURUSD and other majors) and for the CFD
        indices (USA30IDXUSD, USA500IDXUSD, USATECHIDXUSD). FX trades
        about 24 hours a day, five days a week; index CFDs have a daily
        break. Any hour with no trading prints is simply absent.

    Caveats:
        - **The index instruments are CFDs, not futures.** ``USA30IDXUSD``
          is a cash-index CFD, not the CME E-mini Dow. Prices, session
          hours and spreads differ from the futures contract.
        - A quiet hour returns **HTTP 200 with an empty body**. That is
          treated as "no data" and is never retried. Missing hours are
          reported in the coverage report, not silently dropped.
        - The server silently slows down beyond about 4 parallel files.
          ``max_workers`` defaults to 4.
        - The monthly bulk .zip endpoint times out at 120 s and is not
          used.
        - Volume is tick volume, not exchange-cleared volume.
        - Prices are mid prices, ``(bid + ask) / 2``.

    Args:
        instrument: Dukascopy instrument name, e.g. ``'EURUSD'`` or
            ``'USA30IDXUSD'``.
        start: inclusive start, datetime-like or ``'YYYY-MM-DD'``.
        end: inclusive end, datetime-like or ``'YYYY-MM-DD'``.
        timeframe: bar timeframe, e.g. ``'5m'``, ``'1h'``, ``'1d'``.
        max_workers: maximum parallel HTTP downloads. Keep it at 4 or
            below; the server slows down silently above that.
        price_divisor: override the price divisor. When None the
            default map is used: 1000 for USA30IDXUSD, USA500IDXUSD,
            USATECHIDXUSD, XAUUSD and USDJPY, and 100000 otherwise.

    Returns:
        A frame in the canonical schema. The coverage report is attached
        as ``df.attrs['coverage']`` and is also available through
        :func:`coverage_report`.

    Raises:
        ValueError: if ``timeframe`` is not supported.

    Verified live: 2026-11-03.
    """
    key = str(timeframe).strip().lower()
    if key not in TIMEFRAME_RULES:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}. Supported: {sorted(TIMEFRAME_RULES)}"
        )

    divisor = (
        int(price_divisor)
        if price_divisor is not None
        else DEFAULT_PRICE_DIVISORS.get(instrument.upper(), _FALLBACK_DIVISOR)
    )

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = _parse_moment(start, now - timedelta(hours=1))
    end_dt = _parse_moment(end, now)
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    hours: list[datetime] = []
    cursor = start_dt
    while cursor <= end_dt:
        hours.append(cursor)
        cursor = cursor + timedelta(hours=1)

    workers = max(1, min(int(max_workers), 4))
    results: list[tuple[datetime, pd.DataFrame, str | None]] = []
    if hours:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_fetch_hour, instrument, hour, divisor, 60.0)
                for hour in hours
            ]
            for future in futures:
                results.append(future.result())

    results.sort(key=lambda item: item[0])

    frames = [ticks for _, ticks, _ in results if not ticks.empty]
    ticks = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["time", "price", "volume"])
    )

    bars = _resample_ticks(ticks, key)

    hours_with_data = sum(1 for _, ticks_i, _ in results if not ticks_i.empty)
    missing_hours = [
        hour.strftime("%Y-%m-%dT%H:00:00Z") for hour, ticks_i, _ in results if ticks_i.empty
    ]
    errors = [
        {"hour": hour.strftime("%Y-%m-%dT%H:00:00Z"), "error": err}
        for hour, _, err in results
        if err
    ]

    report = {
        "hours_requested": len(hours),
        "hours_with_data": int(hours_with_data),
        "missing_hours": missing_hours,
        "instrument": instrument,
        "timeframe": key,
        "errors": errors,
    }

    bars.attrs["coverage"] = report
    return bars
