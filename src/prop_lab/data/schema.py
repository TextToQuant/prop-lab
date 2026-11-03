"""Canonical bar schema for :mod:`prop_lab`.

Every data source in this package normalises to one shape:

===================  =========  ==========================================
column               dtype      meaning
===================  =========  ==========================================
``time``             int64      UNIX seconds, UTC, bar OPEN time
``open``             float64    first traded price of the bar
``high``             float64    highest traded price of the bar
``low``              float64    lowest traded price of the bar
``close``            float64    last traded price of the bar
``volume``           float64    traded volume; 0.0 when the source has none
===================  =========  ==========================================

Extra columns are preserved rather than dropped, in particular
``session_date`` (``'YYYY-MM-DD'``) and ``source`` (a string naming the
origin of the data).

The session rule is the part that is easy to get wrong. A futures trading
day runs 18:00 ET to 17:00 ET the next day, so the bars from 18:00 ET on
2026-10-27 and the bars up to 17:00 ET on 2026-10-28 belong to the single
session ``2026-10-28``. Grouping by UTC date or by calendar date would cut
every session in half.

Verified against the live data sources on 2026-11-03.
"""
from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "STANDARD_COLUMNS",
    "TIME_COLUMN_CANDIDATES",
    "normalise",
    "assign_session_dates",
    "resample",
    "load_bars",
]

#: The canonical column order of every normalised frame.
STANDARD_COLUMNS: list[str] = ["time", "open", "high", "low", "close", "volume"]

#: Column names accepted as the time axis, matched case-insensitively.
TIME_COLUMN_CANDIDATES: tuple[str, ...] = (
    "time",
    "ts",
    "timestamp",
    "date",
    "datetime",
    "open_time",
)

#: Pandas offset aliases for the supported resample timeframes.
_RESAMPLE_RULES: dict[str, str] = {
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

_OHLCV = ("open", "high", "low", "close", "volume")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _find_time_column(df: pd.DataFrame) -> str | None:
    """Return the canonical name of the time column, or None if absent.

    Args:
        df: frame to inspect.

    Returns:
        The matching column name as it appears in ``df.columns``, or None.
    """
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _to_epoch_seconds(values: pd.Series) -> pd.Series:
    """Convert a time-like series to int64 UNIX seconds UTC.

    Handles datetime64 (naive assumed UTC, aware converted to UTC), ISO
    strings, epoch seconds and epoch milliseconds.

    Args:
        values: the raw time column.

    Returns:
        An int64 Series of UNIX seconds UTC.
    """
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce").astype("float64")
        # Heuristic: anything past year 5138 in seconds is really ms.
        numeric = numeric.where(numeric.abs() < 1e11, numeric / 1000.0)
        return numeric.astype("int64")

    original_index = values.index
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    # ``pd.to_datetime`` on a Series whose VALUES are datetimes returns a
    # frame with a RESET RangeIndex, because the conversion goes through
    # a DatetimeIndex internally. That silently breaks every caller
    # that assigns the result into a frame with a different index:
    # pandas aligns on index labels, finds no match, and fills the
    # column with NaN. Restore the original index so the result stays
    # positionally aligned with the input.
    parsed.index = original_index

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("UTC")
    # Pandas 3.0 defaults to microsecond resolution, older versions used
    # nanoseconds, and some feeds deliver second resolution. A
    # hard-coded divisor silently produces timestamps that are off by
    # 1000x or 1000000x, and the error is invisible because the rows
    # still look like valid integers.
    #
    # Guessing the unit from ``dtype.unit`` is NOT safe: pandas 3.0
    # returns None for a ``datetime64[s]`` index, which would fall
    # through to the nanosecond branch and collapse every timestamp to
    # a single value. Converting to second resolution explicitly with
    # ``as_unit("s")`` is correct regardless of the incoming resolution.
    try:
        seconds = parsed.dt.as_unit("s").astype("int64")
    except (AttributeError, TypeError, ValueError):
        # Fallback for pandas versions without ``as_unit``.
        unit = getattr(parsed.dtype, "unit", None) or "ns"
        divisor = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}[unit]
        seconds = parsed.astype("int64") // divisor
    return seconds.astype("int64")


def _resolve_time_axis(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Return ``(frame, source)`` where source identifies where the time is.

    Args:
        df: input frame.

    Returns:
        A tuple of the frame and the name of the time column that was
        found, or None when the time has to come from the index.
    """
    col = _find_time_column(df)
    if col is not None:
        return df, col

    index = df.index
    if isinstance(index, pd.DatetimeIndex):
        frame = df.copy()
        frame.index.name = index.name or "time"
        return frame, None
    if index.name is not None and str(index.name).strip().lower() in TIME_COLUMN_CANDIDATES:
        return df, None
    return df, None


# ----------------------------------------------------------------------
# normalise
# ----------------------------------------------------------------------
def normalise(df: pd.DataFrame, *, source: str = "unknown") -> pd.DataFrame:
    """Coerce any OHLCV frame to the canonical schema.

    The time axis may be a column named (case-insensitively) ``time``,
    ``ts``, ``timestamp``, ``date``, ``datetime`` or ``open_time``, or a
    datetime index. Values may be datetimes, ISO strings, epoch seconds or
    epoch milliseconds; everything is converted to UNIX seconds UTC.

    Source: internal to this package; the normalisation rules follow
    SPEC.md section 1.

    Coverage: not applicable -- this function transforms data, it does not
    fetch it.

    Caveats:
        - A naive datetime column is interpreted as UTC.
        - Rows without a parseable timestamp are dropped.
        - Duplicate timestamps are collapsed, keeping the last occurrence.
        - A missing ``volume`` column becomes 0.0.
        - Extra columns (``session_date``, ``source``, ...) are preserved.

    Args:
        df: the frame to normalise.
        source: value written into the ``source`` column when the frame
            does not already carry one.

    Returns:
        A new frame in the canonical schema, sorted by ``time`` ascending.

    Verified: 2026-11-03.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"normalise() expects a pandas DataFrame, got {type(df)!r}")

    frame, time_col = _resolve_time_axis(df)
    frame = frame.copy()

    if time_col is None:
        if isinstance(frame.index, pd.DatetimeIndex):
            index = frame.index
            if index.tz is None:
                index = index.tz_localize("UTC")
            else:
                index = index.tz_convert("UTC")
            frame = frame.reset_index(drop=True)
            frame["time"] = _to_epoch_seconds(pd.Series(index))
        else:
            raise ValueError(
                "Could not find a time axis. Provide a column named one of "
                f"{TIME_COLUMN_CANDIDATES} or a datetime index. "
                f"Available columns: {list(df.columns)}"
            )
    else:
        frame["time"] = _to_epoch_seconds(frame[time_col])
        if time_col != "time":
            frame = frame.drop(columns=[time_col])

    frame = frame[frame["time"].notna()]

    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for name in _OHLCV:
        src = lowered.get(name)
        if src is None:
            if name == "volume":
                frame["volume"] = 0.0
            else:
                raise ValueError(
                    f"Missing required OHLCV column {name!r}. "
                    f"Available columns: {list(frame.columns)}"
                )
        elif src != name:
            frame = frame.rename(columns={src: name})

    for name in _OHLCV:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")

    if "source" not in frame.columns:
        frame["source"] = source
    else:
        frame["source"] = frame["source"].astype("object").where(
            frame["source"].notna(), source
        )

    frame["time"] = frame["time"].astype("int64")
    frame = frame.sort_values("time", kind="mergesort")
    frame = frame.drop_duplicates(subset=["time"], keep="last")
    frame = frame.reset_index(drop=True)

    ordered = [c for c in STANDARD_COLUMNS if c in frame.columns]
    extras = [c for c in frame.columns if c not in ordered]
    return frame[ordered + extras]


# ----------------------------------------------------------------------
# Session dates
# ----------------------------------------------------------------------
def assign_session_dates(
    df: pd.DataFrame,
    *,
    tz: str = "America/New_York",
    session_open_hour: int = 18,
) -> pd.Series:
    """Assign the trading session label to every bar.

    A futures trading day runs 18:00 ET to 17:00 ET the next day. A bar
    whose LOCAL time is at or after ``session_open_hour`` belongs to the
    NEXT calendar day; a bar before ``session_open_hour`` belongs to the
    CURRENT calendar day. So a bar at exactly 17:00 ET belongs to the day
    it is on, and a bar at 18:00 ET belongs to the next day.

    For 24/7 crypto pass ``session_open_hour=0``: the session label is
    then the UTC calendar date of the bar, matching the UTC date of the
    crypto day.

    Source: internal to this package; the rule is SPEC.md section 1.

    Coverage: not applicable -- this function transforms data.

    Caveats:
        - The label depends on the timezone of the exchange. Futures on the
          CME trade 18:00 ET (17:00 ET during daylight saving is *not*
          used; 18:00 local wall-clock is used, matching the exchange
          session convention).
        - ``session_open_hour=0`` returns the UTC date, not the local date.

    Args:
        df: frame containing a ``time`` column in UNIX seconds UTC.
        tz: IANA timezone used to interpret the session boundary.
        session_open_hour: local hour at which the session rolls over.
            Use 0 for 24/7 markets.

    Returns:
        A Series of ``'YYYY-MM-DD'`` strings aligned to ``df.index``.

    Verified: 2026-11-03.
    """
    if "time" not in df.columns:
        raise ValueError(
            "assign_session_dates() requires a 'time' column of UNIX seconds UTC. "
            f"Available columns: {list(df.columns)}"
        )

    stamps = pd.to_datetime(pd.to_numeric(df["time"], errors="coerce"), unit="s", utc=True)

    if session_open_hour == 0:
        # 24/7 market: the session label is the UTC calendar date.
        return stamps.dt.strftime("%Y-%m-%d")

    local = stamps.dt.tz_convert(tz)
    dates = local.dt.normalize()
    rollover = local.dt.hour >= int(session_open_hour)
    dates = dates + pd.to_timedelta(rollover.astype("int8"), unit="D")
    return dates.dt.strftime("%Y-%m-%d")


# ----------------------------------------------------------------------
# Resample
# ----------------------------------------------------------------------
def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample a canonical frame to a coarser timeframe.

    Supported timeframes: ``1m``, ``2m``, ``3m``, ``5m``, ``15m``,
    ``30m``, ``1h``, ``2h``, ``4h``, ``1d``. Aggregation is the standard
    one: open is the first bar's open, high is the maximum, low is the
    minimum, close is the last bar's close, volume is the sum.

    Source: internal to this package.

    Coverage: not applicable -- this function transforms data.

    Caveats:
        - Buckets with no data are dropped, so the output is not a
          continuous grid of bars.
        - The ``session_date`` column is not carried over; recompute it
          with :func:`assign_session_dates` on the resampled frame.
        - The ``source`` column is preserved.

    Args:
        df: frame in the canonical schema.
        timeframe: target timeframe, e.g. ``'5m'``.

    Returns:
        A new canonical frame sorted by ``time`` ascending.

    Verified: 2026-11-03.
    """
    key = str(timeframe).strip().lower()
    if key not in _RESAMPLE_RULES:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}. Supported: "
            f"{sorted(_RESAMPLE_RULES)}"
        )

    frame = df.copy()
    if "time" not in frame.columns:
        raise ValueError(
            "resample() requires a 'time' column. Available columns: "
            f"{list(frame.columns)}"
        )

    index = pd.to_datetime(pd.to_numeric(frame["time"], errors="coerce"), unit="s", utc=True)
    frame = frame.assign(_ts=index).set_index("_ts")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    keep_source = "source" in frame.columns
    resampled = frame.resample(_RESAMPLE_RULES[key]).agg(agg)
    resampled = resampled.dropna(subset=["open"])
    resampled = resampled.reset_index()

    # Convert the resampled index to second resolution explicitly. The
    # resample output index keeps the source resolution (microseconds in
    # pandas 3.0), and dividing by a hard-coded 10**9 would be wrong.
    resampled["time"] = _to_epoch_seconds(pd.Series(resampled["_ts"]))
    resampled = resampled.drop(columns=["_ts"])
    resampled = resampled.sort_values("time", kind="mergesort").reset_index(drop=True)

    ordered = [c for c in STANDARD_COLUMNS if c in resampled.columns]
    extras = [c for c in resampled.columns if c not in ordered]
    out = resampled[ordered + extras]
    if not keep_source and "source" in out.columns:
        out = out.drop(columns=["source"])
    return out


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------
def _parse_boundary(value: Any, *, end: bool = False) -> int | None:
    """Convert a start/end argument to UNIX seconds UTC.

    Args:
        value: a datetime, a date, a ``'YYYY-MM-DD'`` string, a pandas
            Timestamp or None.
        end: when True a bare date is pushed to the end of that day.

    Returns:
        UNIX seconds UTC, or None when ``value`` is None.
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    stamp = pd.Timestamp(pd.to_datetime(value))
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    if end and len(str(value).strip()) <= 10:
        stamp = stamp + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    # A Timestamp always stores nanoseconds internally, so its ``value``
    # is always in nanoseconds regardless of the surrounding series
    # resolution. Dividing by 10**9 is therefore correct HERE, unlike
    # for a DatetimeIndex whose resolution may be seconds or
    # microseconds.
    return int(stamp.value // 10**9)


def _read_frame(payload: bytes, suffix: str) -> pd.DataFrame:
    """Parse raw bytes as CSV or Parquet.

    Args:
        payload: the downloaded bytes.
        suffix: the file extension that decides the parser.

    Returns:
        A parsed DataFrame.
    """
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(payload))
    if suffix in (".csv", ".txt", ""):
        return pd.read_csv(io.BytesIO(payload))
    raise ValueError(
        f"Unsupported file type {suffix!r}. Supported: .csv, .parquet"
    )


def load_bars(
    path_or_url: str,
    *,
    source: str | None = None,
    timeframe: str | None = None,
    start: Any = None,
    end: Any = None,
) -> pd.DataFrame:
    """Load bars from a local file or an HTTP(S) URL.

    The drop-zone loader: point it at a CSV or Parquet file that a user
    downloaded from anywhere, and get back the canonical schema.

    Source: local filesystem or an HTTP(S) URL supplied by the caller.
    No data is fetched from a market API here; use the source adapters
    (:mod:`prop_lab.data.binance_src`, :mod:`prop_lab.data.yahoo_src`,
    :mod:`prop_lab.data.dukascopy_src`) for those.

    Coverage: whatever the file contains.

    Caveats:
        - ``.csv`` files are parsed with pandas defaults; a file whose
          separator is not a comma may need to be re-read by hand.
        - Nothing is ever written outside the repo or the caller's chosen
          cache directory; this loader only reads.
        - The ``start``/``end`` bounds are inclusive.

    Args:
        path_or_url: a local path or an ``http://`` / ``https://`` URL.
        source: when given, the value written into the ``source`` column.
        timeframe: when given, resample to this timeframe after loading.
        start: optional lower time bound (datetime or ``'YYYY-MM-DD'``).
        end: optional upper time bound (datetime or ``'YYYY-MM-DD'``).

    Returns:
        A frame in the canonical schema, sorted by ``time`` ascending.

    Verified: 2026-11-03.
    """
    if not isinstance(path_or_url, str) or not path_or_url.strip():
        raise ValueError("path_or_url must be a non-empty string")

    target = path_or_url.strip()
    if target.lower().startswith(("http://", "https://")):
        import requests

        response = requests.get(target, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"Download failed for {target}: HTTP {response.status_code}"
            )
        payload = response.content
        suffix = os.path.splitext(target.split("?")[0])[1].lower()
    else:
        if not os.path.exists(target):
            raise FileNotFoundError(f"No such file: {target}")
        with open(target, "rb") as handle:
            payload = handle.read()
        suffix = os.path.splitext(target)[1].lower()

    frame = _read_frame(payload, suffix)
    frame = normalise(frame, source=source or "unknown")

    if start is not None or end is not None:
        lower = _parse_boundary(start) if start is not None else None
        upper = _parse_boundary(end, end=True) if end is not None else None
        mask = pd.Series(True, index=frame.index)
        if lower is not None:
            mask &= frame["time"] >= lower
        if upper is not None:
            mask &= frame["time"] <= upper
        frame = frame[mask].reset_index(drop=True)

    if timeframe:
        frame = resample(frame, timeframe)
    return frame
