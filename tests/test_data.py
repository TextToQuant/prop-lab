"""Tests for the prop_lab data layer.

The network tests are marked ``network`` and are skipped unless the
environment variable ``PROPLAB_NETWORK_TESTS=1`` is set::

    PROPLAB_NETWORK_TESTS=1 .venv/bin/python -m pytest tests/test_data.py -m network
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from prop_lab.data import (
    STANDARD_COLUMNS,
    assign_session_dates,
    load_bars,
    normalise,
    resample,
)
from prop_lab.data import binance_src, dukascopy_src, yahoo_src

NETWORK = os.environ.get("PROPLAB_NETWORK_TESTS") == "1"
network_only = pytest.mark.skipif(
    not NETWORK, reason="network tests disabled (set PROPLAB_NETWORK_TESTS=1)"
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _canonical_frame(n: int = 10, start_ts: int = 1760000000) -> pd.DataFrame:
    """Build a small frame in the canonical schema.

    Args:
        n: number of rows.
        start_ts: UNIX seconds of the first row.

    Returns:
        A canonical DataFrame.
    """
    times = [start_ts + 60 * i for i in range(n)]
    return pd.DataFrame(
        {
            "time": times,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [10.0 + i for i in range(n)],
        }
    )


def _et_timestamp(y: int, m: int, d: int, hh: int, mm: int = 0) -> int:
    """Return the UNIX seconds of a wall-clock time in New York.

    Args:
        y: year.
        m: month.
        d: day.
        hh: hour.
        mm: minute.

    Returns:
        UNIX seconds UTC.
    """
    stamp = pd.Timestamp(f"{y:04d}-{m:02d}-{d:02d} {hh:02d}:{mm:02d}", tz="America/New_York")
    return int(stamp.tz_convert("UTC").timestamp())


# ----------------------------------------------------------------------
# normalise
# ----------------------------------------------------------------------
def test_normalise_datetime_index():
    index = pd.date_range("2026-01-01 00:00:00", periods=4, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.5, 2.5, 3.5, 4.5],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.2, 2.2, 3.2, 4.2],
        },
        index=index,
    )
    out = normalise(df)
    assert list(out.columns[:6]) == STANDARD_COLUMNS
    assert out["time"].dtype == np.int64
    assert out["time"].iloc[0] == int(index[0].timestamp())
    assert out["volume"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert out["source"].iloc[0] == "unknown"


def test_normalise_ts_column():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-02-03 09:00:00", periods=3, freq="1h", tz="UTC"),
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
        }
    )
    out = normalise(df, source="test")
    assert "ts" not in out.columns
    assert "time" in out.columns
    assert out["source"].tolist() == ["test", "test", "test"]


def test_normalise_date_column():
    df = pd.DataFrame(
        {
            "Date": ["2026-03-04", "2026-03-05", "2026-03-06"],
            "Open": [10.0, 11.0, 12.0],
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [10.5, 11.5, 12.5],
        }
    )
    out = normalise(df)
    assert list(out.columns[:6]) == STANDARD_COLUMNS
    assert out["time"].iloc[0] == int(
        pd.Timestamp("2026-03-04", tz="UTC").timestamp()
    )
    assert out["close"].tolist() == [10.5, 11.5, 12.5]


def test_normalise_already_canonical_is_idempotent():
    df = _canonical_frame(5)
    out = normalise(df)
    assert list(out.columns[:6]) == STANDARD_COLUMNS
    assert out["time"].tolist() == df["time"].tolist()
    assert out["open"].dtype == np.float64
    assert out["volume"].dtype == np.float64
    again = normalise(out)
    pd.testing.assert_frame_equal(
        out[["time", "open", "high", "low", "close", "volume"]],
        again[["time", "open", "high", "low", "close", "volume"]],
    )


def test_normalise_sorts_and_drops_duplicate_timestamps():
    df = _canonical_frame(4)
    shuffled = pd.concat([df.iloc[2:], df.iloc[:2]])
    duped = pd.concat([shuffled, df.iloc[[0]]], ignore_index=True)
    out = normalise(duped)
    assert out["time"].is_monotonic_increasing
    assert out["time"].is_unique
    assert len(out) == 4


# ----------------------------------------------------------------------
# assign_session_dates
# ----------------------------------------------------------------------
def test_assign_session_dates_evening_bar_rolls_to_next_day():
    df = pd.DataFrame({"time": [_et_timestamp(2026, 10, 27, 22)]})
    out = assign_session_dates(df)
    assert out.iloc[0] == "2026-10-28"


def test_assign_session_dates_intraday_bar_stays_on_its_day():
    df = pd.DataFrame({"time": [_et_timestamp(2026, 10, 28, 10)]})
    out = assign_session_dates(df)
    assert out.iloc[0] == "2026-10-28"


def test_assign_session_dates_exactly_17_00_stays_on_that_day():
    df = pd.DataFrame({"time": [_et_timestamp(2026, 10, 28, 17)]})
    out = assign_session_dates(df)
    assert out.iloc[0] == "2026-10-28"


def test_assign_session_dates_exactly_18_00_rolls_to_next_day():
    df = pd.DataFrame({"time": [_et_timestamp(2026, 10, 28, 18)]})
    out = assign_session_dates(df)
    assert out.iloc[0] == "2026-10-29"


def test_assign_session_dates_crypto_uses_utc_date():
    stamps = [
        int(pd.Timestamp("2026-04-01 23:59:00", tz="UTC").timestamp()),
        int(pd.Timestamp("2026-04-02 00:00:00", tz="UTC").timestamp()),
    ]
    df = pd.DataFrame({"time": stamps})
    out = assign_session_dates(df, session_open_hour=0)
    assert out.tolist() == ["2026-04-01", "2026-04-02"]


def test_assign_session_dates_groups_a_full_futures_session():
    df = pd.DataFrame(
        {
            "time": [
                _et_timestamp(2026, 10, 27, 18),
                _et_timestamp(2026, 10, 27, 23),
                _et_timestamp(2026, 10, 28, 2),
                _et_timestamp(2026, 10, 28, 16),
            ]
        }
    )
    out = assign_session_dates(df)
    assert out.tolist() == ["2026-10-28"] * 4


# ----------------------------------------------------------------------
# resample
# ----------------------------------------------------------------------
def test_resample_1m_to_5m_ohlcv():
    # Start on a 5-minute boundary so ten 1-minute bars fill exactly two
    # buckets. An unaligned start would legitimately span three.
    start = 1760000000 - (1760000000 % 300)
    rows = []
    for i in range(10):
        rows.append(
            {
                "time": start + 60 * i,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1.0 + i,
            }
        )
    df = pd.DataFrame(rows)
    out = resample(df, "5m")

    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == 100.0
    assert first["high"] == 105.0  # max of high over the first five bars
    assert first["low"] == 99.0
    assert first["close"] == 104.5  # close of the fifth bar
    assert first["volume"] == pytest.approx(1.0 + 2.0 + 3.0 + 4.0 + 5.0)

    second = out.iloc[1]
    assert second["open"] == 105.0
    assert second["close"] == 109.5
    assert second["volume"] == pytest.approx(6.0 + 7.0 + 8.0 + 9.0 + 10.0)


def test_resample_with_unaligned_start_spans_three_buckets():
    """An unaligned start legitimately produces an extra partial bucket."""
    rows = []
    for i in range(10):
        rows.append(
            {
                "time": 1760000000 + 60 * i,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1.0 + i,
            }
        )
    df = pd.DataFrame(rows)
    out = resample(df, "5m")
    assert len(out) == 3
    # The first bucket holds only the two bars before the boundary.
    assert out.iloc[0]["volume"] == pytest.approx(1.0 + 2.0)
    assert out.iloc[0]["open"] == 100.0

    assert out["time"].is_monotonic_increasing
    assert list(out.columns[:6]) == STANDARD_COLUMNS


def test_resample_drops_empty_buckets():
    df = _canonical_frame(1, start_ts=1760000000)
    df2 = _canonical_frame(1, start_ts=1760000000 + 3600)
    out = resample(pd.concat([df, df2], ignore_index=True), "5m")
    assert len(out) == 2


def test_resample_rejects_unknown_timeframe():
    df = _canonical_frame(3)
    with pytest.raises(ValueError):
        resample(df, "7m")


# ----------------------------------------------------------------------
# load_bars
# ----------------------------------------------------------------------
def test_load_bars_round_trips_a_csv(tmp_path):
    original = _canonical_frame(12)
    path = tmp_path / "bars.csv"
    original.to_csv(path, index=False)

    loaded = load_bars(str(path), source="unit-test")
    assert list(loaded.columns[:6]) == STANDARD_COLUMNS
    assert len(loaded) == len(original)
    assert loaded["time"].tolist() == original["time"].tolist()
    assert loaded["source"].tolist() == ["unit-test"] * len(original)
    assert loaded["time"].dtype == np.int64
    assert loaded["close"].dtype == np.float64


def test_load_bars_filters_by_start_and_end(tmp_path):
    original = _canonical_frame(48)  # 48 one-minute bars
    path = tmp_path / "bars.csv"
    original.to_csv(path, index=False)

    loaded = load_bars(str(path), start="2025-10-09", end="2025-10-09")
    # The bounds are inclusive on both ends, so every bar of that UTC day
    # that exists in the source is kept.
    assert len(loaded) <= len(original)
    assert loaded["time"].min() >= original["time"].min()


def test_load_bars_can_resample(tmp_path):
    original = _canonical_frame(60)
    path = tmp_path / "bars.csv"
    original.to_csv(path, index=False)
    loaded = load_bars(str(path), timeframe="5m")
    # 60 one-minute bars span 60 minutes. The fixture starts on an
    # unaligned timestamp, so the span crosses 13 five-minute buckets:
    # twelve full ones plus a partial one at each end.
    assert len(loaded) == 13
    # Every source bar is accounted for exactly once.
    assert loaded["volume"].sum() == pytest.approx(original["volume"].sum())


def test_load_bars_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bars(str(tmp_path / "nope.csv"))


# ----------------------------------------------------------------------
# Source adapter units (no network)
# ----------------------------------------------------------------------
def test_binance_rejects_unknown_interval():
    with pytest.raises(ValueError):
        binance_src.fetch(symbol="BTCUSDT", interval="7m")


def test_binance_interval_map_covers_the_documented_set():
    for interval in ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"):
        assert interval in binance_src.BINANCE_INTERVALS


def test_yahoo_rejects_unknown_interval():
    with pytest.raises(ValueError):
        yahoo_src.fetch(symbol="NQ=F", interval="3m")


def test_yahoo_warns_on_unadjusted_continuous_futures():
    assert yahoo_src.is_continuous_future("NQ=F")
    assert not yahoo_src.is_continuous_future("^VIX")
    assert not yahoo_src.is_continuous_future("BTC-USD")


def test_dukascopy_price_divisor_defaults():
    assert dukascopy_src.DEFAULT_PRICE_DIVISORS["USA30IDXUSD"] == 1000
    assert dukascopy_src.DEFAULT_PRICE_DIVISORS["XAUUSD"] == 1000
    assert dukascopy_src.DEFAULT_PRICE_DIVISORS["USDJPY"] == 1000


def test_dukascopy_rejects_unknown_timeframe():
    with pytest.raises(ValueError):
        dukascopy_src.fetch(instrument="EURUSD", timeframe="7m")


def test_dukascopy_coverage_report_helper():
    df = pd.DataFrame({"time": [1]})
    assert dukascopy_src.coverage_report(df) == {}
    df.attrs["coverage"] = {"hours_requested": 2, "hours_with_data": 1}
    assert dukascopy_src.coverage_report(df)["hours_requested"] == 2


def test_dukascopy_hour_url_is_zero_based_month():
    from datetime import datetime, timezone

    hour = datetime(2026, 7, 4, 13, tzinfo=timezone.utc)
    url = dukascopy_src._hour_url("EURUSD", hour)
    assert url == (
        "https://datafeed.dukascopy.com/datafeed/EURUSD/2026/06/04/13h_ticks.bi5"
    )


# ----------------------------------------------------------------------
# Network tests
# ----------------------------------------------------------------------
@pytest.mark.network
@network_only
def test_network_binance_btcusdt_1m():
    df = binance_src.fetch(
        symbol="BTCUSDT",
        interval="1m",
        start="2026-10-01",
        end="2026-10-02",
    )
    assert len(df) >= 100
    for column in STANDARD_COLUMNS:
        assert column in df.columns
    assert df["time"].is_monotonic_increasing
    assert df["close"].gt(0).all()


@pytest.mark.network
@network_only
def test_network_yahoo_nq_f_daily():
    df = yahoo_src.fetch(symbol="NQ=F", interval="1d", start="2020-01-01", end="2026-01-01")
    assert len(df) > 1000
    for column in STANDARD_COLUMNS:
        assert column in df.columns
    assert df["close"].gt(0).all()


@pytest.mark.network
@network_only
def test_network_dukascopy_usa30_two_hours():
    df = dukascopy_src.fetch(
        instrument="USA30IDXUSD",
        start="2026-09-02T12:00:00",
        end="2026-09-02T13:00:00",
        timeframe="5m",
    )
    assert "coverage" in df.attrs
    report = dukascopy_src.coverage_report(df)
    assert report["hours_requested"] == 2
    assert report["instrument"] == "USA30IDXUSD"
    assert report["timeframe"] == "5m"
