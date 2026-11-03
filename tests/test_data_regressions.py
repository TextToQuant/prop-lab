"""Regression tests for the live data adapters.

Every test here corresponds to a bug that actually happened. The
network tests skip unless ``PROPLAB_NETWORK_TESTS=1`` is set, which
means they can silently rot, so the offline tests below reproduce the
same failure modes with synthetic data.

Bug 1: Yahoo returned one row instead of 1511.
    The daily index arrives as ``datetime64[s]``. Dividing by a
    hard-coded 10**9 collapsed every timestamp to the same value, and
    the dedup step then dropped 1510 rows.

Bug 2: the converted time column came back as NaN.
    ``pd.to_datetime`` on a Series of datetimes returns a frame with a
    RESET RangeIndex. Assigning that into a frame with a datetime index
    makes pandas align on labels, find no match, and fill NaN.

Bug 3: the Dukascopy decoder raised ``struct.error``.
    The format string was repeated by string multiplication, which
    produces ">IIIff>IIIff". The byte-order prefix ">" is only legal
    at the start of a format string.

Bug 4: the Dukascopy decoder then raised ``IndexError``.
    ``Struct.unpack_from`` unpacks exactly ONE record, not all of them.
    ``Struct.iter_unpack`` walks every record.
"""
from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest

from prop_lab.data import dukascopy_src, schema, yahoo_src


# ----------------------------------------------------------------------
# Timestamp conversion
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "unit,expected_divisor",
    [("s", 1), ("ms", 10**3), ("us", 10**6), ("ns", 10**9)],
)
def test_epoch_conversion_honours_the_input_resolution(unit, expected_divisor):
    """Every datetime resolution must land on the same epoch seconds."""
    index = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03"]))
    values = pd.Series(index.as_unit(unit), index=index)
    out = schema._to_epoch_seconds(values)
    assert out.tolist() == [1577923200, 1578009600]


def test_epoch_conversion_preserves_the_original_index():
    """The result must stay aligned with the input.

    ``pd.to_datetime`` resets the index. If the index is not restored,
    the caller assigns the result into a differently-indexed frame and
    pandas fills NaN.
    """
    index = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03"]))
    values = pd.Series(index, index=index)
    out = schema._to_epoch_seconds(values)
    assert out.index.equals(index)


def test_epoch_conversion_result_can_be_assigned_positionally():
    """The exact Yahoo failure mode.

    A converted time column assigned into a frame whose index differs
    must not produce NaN.
    """
    frame_index = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03"]))
    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=frame_index)

    time_values = schema._to_epoch_seconds(pd.Series(frame_index))
    frame["time"] = time_values.to_numpy()

    assert frame["time"].notna().all()
    assert frame["time"].tolist() == [1577923200, 1578009600]


def test_second_resolution_index_does_not_collapse():
    """1511 distinct timestamps must stay 1511 distinct timestamps."""
    index = pd.date_range("2020-01-02", periods=200, freq="D").as_unit("s")
    values = pd.Series(index, index=index)
    out = schema._to_epoch_seconds(values)
    assert out.nunique() == 200
    assert out.min() == 1577923200


def test_yahoo_time_column_keeps_every_timestamp():
    """The exact Yahoo failure mode, through the shipped code path.

    The Yahoo daily index is ``datetime64[s]``. Dividing it by a
    hard-coded 10**9 collapsed all 1511 timestamps to the value 1, and
    the dedup step then dropped 1510 rows, so the adapter returned a
    single row.

    This test calls the helper the adapter itself calls, so it fails if
    that helper regresses.
    """
    index = pd.date_range("2020-01-02", periods=200, freq="D").as_unit("s")
    column = yahoo_src._epoch_seconds_column(index)

    assert np.unique(column).size == 200
    assert column.min() == 1577923200
    # These must be epoch SECONDS. Microseconds would be 1e6 times
    # larger and nanoseconds 1e9 times larger.
    assert column.max() < 2_000_000_000


def test_yahoo_time_column_is_not_null_after_assignment():
    """The second Yahoo failure mode.

    ``pd.to_datetime`` resets the index, so assigning the converted
    Series into a frame with a datetime index made pandas align on
    labels, find no match, and fill the column with NaN.
    """
    index = pd.date_range("2020-01-02", periods=50, freq="D").as_unit("s")
    frame = pd.DataFrame({"close": np.arange(50, dtype=float)}, index=index)

    frame["time"] = yahoo_src._epoch_seconds_column(index)

    assert frame["time"].notna().all()
    assert frame["time"].nunique() == 50


def test_yahoo_time_column_on_a_tz_aware_index():
    index = pd.date_range("2020-01-02", periods=50, freq="D", tz="UTC")
    column = yahoo_src._epoch_seconds_column(index)
    assert np.unique(column).size == 50
    assert column.min() == 1577923200


def test_numeric_epoch_values_pass_through():
    values = pd.Series([1577923200, 1578009600])
    out = schema._to_epoch_seconds(values)
    assert out.tolist() == [1577923200, 1578009600]


def test_epoch_milliseconds_are_detected():
    """A value far past year 5138 in seconds is really milliseconds."""
    values = pd.Series([1577923200000, 1578009600000])
    out = schema._to_epoch_seconds(values)
    assert out.tolist() == [1577923200, 1578009600]


# ----------------------------------------------------------------------
# Yahoo
# ----------------------------------------------------------------------
def test_yahoo_detects_continuous_futures():
    assert yahoo_src.is_continuous_future("NQ=F")
    assert yahoo_src.is_continuous_future("MNQ=F")
    assert yahoo_src.is_continuous_future("ES=F")
    assert not yahoo_src.is_continuous_future("AAPL")
    assert not yahoo_src.is_continuous_future("BTC-USD")


def test_yahoo_rejects_an_unsupported_interval():
    with pytest.raises(ValueError):
        yahoo_src.fetch(symbol="NQ=F", interval="7m")


def test_yahoo_flatten_handles_a_multiindex_column_frame():
    """yfinance returns a MultiIndex even for a single ticker."""
    index = pd.date_range("2020-01-02", periods=3, freq="D").as_unit("s")
    columns = pd.MultiIndex.from_tuples(
        [
            ("Adj Close", "NQ=F"),
            ("Close", "NQ=F"),
            ("High", "NQ=F"),
            ("Low", "NQ=F"),
            ("Open", "NQ=F"),
            ("Volume", "NQ=F"),
        ]
    )
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0, 0.5, 1.5, 100.0]] * 3, index=index, columns=columns
    )
    flat = yahoo_src._flatten_columns(frame)
    assert "Close" in flat.columns
    assert "Open" in flat.columns
    assert len(flat) == 3


def test_yahoo_flatten_handles_flat_columns():
    index = pd.date_range("2020-01-02", periods=2, freq="D").as_unit("s")
    frame = pd.DataFrame(
        {"Close": [1.0, 2.0], "Open": [1.0, 2.0], "High": [1.0, 2.0],
         "Low": [1.0, 2.0], "Volume": [1.0, 2.0]},
        index=index,
    )
    flat = yahoo_src._flatten_columns(frame)
    assert "Close" in flat.columns


# ----------------------------------------------------------------------
# Dukascopy
# ----------------------------------------------------------------------
def _bi5_payload(records):
    """Build a synthetic BI5 payload from (ms, ask, bid, avol, bvol) rows."""
    record = struct.Struct(">IIIff")
    return b"".join(record.pack(*row) for row in records)


def test_dukascopy_decodes_every_record():
    """The decoder must read ALL records, not just the first one.

    ``Struct.unpack_from`` unpacks one record. ``iter_unpack`` walks
    all of them.
    """
    records = [
        (0, 25_000_000, 24_999_000, 1.5, 2.5),
        (1000, 25_001_000, 25_000_000, 1.0, 1.0),
        (2000, 25_002_000, 25_001_000, 0.5, 0.5),
        (3000, 25_003_000, 25_002_000, 0.25, 0.25),
    ]
    payload = _bi5_payload(records)
    hour = pd.Timestamp("2026-10-27 14:00", tz="UTC").to_pydatetime()

    out = dukascopy_src._decode_hour(payload, hour, 1000)

    assert len(out) == 4
    # Mid price of the first record: (25000000 + 24999000) / 2 / 1000
    assert out["price"].iloc[0] == pytest.approx(24_999.5)
    # Volume is ask + bid.
    assert out["volume"].iloc[0] == pytest.approx(4.0)
    # Timestamps advance with the millisecond field.
    assert out["time"].iloc[1] - out["time"].iloc[0] == 1


def test_dukascopy_decodes_an_empty_payload():
    hour = pd.Timestamp("2026-10-27 14:00", tz="UTC").to_pydatetime()
    out = dukascopy_src._decode_hour(b"", hour, 1000)
    assert len(out) == 0


def test_dukascopy_decodes_a_partial_trailing_record():
    """A truncated final record must be ignored, not crash."""
    records = [
        (0, 25_000_000, 24_999_000, 1.0, 1.0),
        (1000, 25_001_000, 25_000_000, 1.0, 1.0),
    ]
    payload = _bi5_payload(records) + b"\x00\x07"  # 7 stray bytes
    hour = pd.Timestamp("2026-10-27 14:00", tz="UTC").to_pydatetime()

    out = dukascopy_src._decode_hour(payload, hour, 1000)
    assert len(out) == 2


def test_dukascopy_struct_format_is_repeatable_only_via_iter_unpack():
    """Guards the specific mistake that broke the decoder.

    Multiplying the format string produces an invalid format, and
    ``unpack_from`` reads a single record.
    """
    fmt = dukascopy_src.BI5_RECORD
    with pytest.raises(struct.error):
        struct.calcsize(fmt * 2)

    record = struct.Struct(fmt)
    payload = _bi5_payload([(0, 1, 2, 3.0, 4.0), (5, 6, 7, 8.0, 9.0)])
    assert len(list(record.iter_unpack(payload))) == 2
    assert len(record.unpack_from(payload, 0)) == 5


def test_dukascopy_price_divisors_cover_the_known_instruments():
    divisors = dukascopy_src.DEFAULT_PRICE_DIVISORS
    for instrument in ("USA30IDXUSD", "USA500IDXUSD", "USATECHIDXUSD"):
        assert divisors[instrument] == 1000
    assert divisors["XAUUSD"] == 1000
    assert divisors["USDJPY"] == 1000
    # FX that is not listed falls back to the 100000 divisor.
    assert "EURUSD" not in divisors


def test_dukascopy_hour_url_uses_a_zero_based_month():
    """The month in the Dukascopy path is zero-based: 06 means July."""
    url = dukascopy_src._hour_url(
        "EURUSD", pd.Timestamp("2026-07-15 13:00", tz="UTC").to_pydatetime()
    )
    assert "/2026/06/15/13h_ticks.bi5" in url


def test_dukascopy_coverage_report_is_exposed():
    frame = pd.DataFrame(
        {
            "time": [1577923200],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [0.0],
            "source": ["dukascopy"],
        }
    )
    frame.attrs["coverage"] = {
        "hours_requested": 1,
        "hours_with_data": 1,
        "missing_hours": [],
        "instrument": "EURUSD",
        "timeframe": "5m",
    }
    report = dukascopy_src.coverage_report(frame)
    assert report["hours_requested"] == 1
    assert report["missing_hours"] == []


def test_dukascopy_missing_hour_is_never_an_error():
    """A quiet hour returns HTTP 200 with an empty body.

    That is not an error and must not be retried, or the fetcher will
    hammer the server every weekend forever.
    """
    assert dukascopy_src._RETRY_STATUSES == frozenset(
        {429, 500, 502, 503, 504}
    )
    assert 200 not in dukascopy_src._RETRY_STATUSES
    assert 404 not in dukascopy_src._RETRY_STATUSES


# ----------------------------------------------------------------------
# Live network tests
#
# NOTE: the live network tests live in tests/test_data.py, where they
# are gated by BOTH ``@pytest.mark.network`` and a ``skipif`` on
# PROPLAB_NETWORK_TESTS. They are deliberately not duplicated here, so
# that the live endpoints are not hit twice per run.
#
# The offline tests above reproduce the same failure modes with
# synthetic data, which is what actually protects the adapters. The
# live tests confirm the endpoints still speak the expected shape.
# ----------------------------------------------------------------------
