"""Data layer for :mod:`prop_lab`.

One canonical bar schema, several free sources, and an honest account of
what each source can and cannot deliver.

The canonical schema (SPEC.md section 1)::

    time    int64    UNIX seconds, UTC, bar OPEN time
    open    float64
    high    float64
    low     float64
    close   float64
    volume  float64  0.0 when the source has no volume

Extra columns are preserved, in particular ``session_date``
(``'YYYY-MM-DD'``) and ``source``.

Modules:
    :mod:`prop_lab.data.schema` -- the schema, the session rule, the
        resampler and the CSV drop-zone loader.
    :mod:`prop_lab.data.binance_src` -- Binance public REST (crypto spot).
    :mod:`prop_lab.data.yahoo_src` -- Yahoo Finance via yfinance.
    :mod:`prop_lab.data.dukascopy_src` -- Dukascopy .bi5 tick archive.

Verified against the live data sources on 2026-11-03.
"""
from __future__ import annotations

from .schema import (
    STANDARD_COLUMNS,
    TIME_COLUMN_CANDIDATES,
    assign_session_dates,
    load_bars,
    normalise,
    resample,
)

__all__ = [
    "STANDARD_COLUMNS",
    "TIME_COLUMN_CANDIDATES",
    "normalise",
    "load_bars",
    "resample",
    "assign_session_dates",
    "binance_src",
    "yahoo_src",
    "dukascopy_src",
]


def __getattr__(name: str):
    """Import the source adapters lazily.

    The source adapters pull in optional third-party dependencies
    (``requests``, ``yfinance``), so they are not imported when the
    package is imported. The core package must import with stdlib plus
    numpy, pandas and matplotlib only.

    Args:
        name: the attribute being looked up.

    Returns:
        The imported submodule.

    Raises:
        AttributeError: when the name is not a known submodule.
    """
    if name in ("binance_src", "yahoo_src", "dukascopy_src"):
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
