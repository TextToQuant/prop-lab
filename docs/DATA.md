# DATA.md

Every free source this package can pull from, what it actually gives you,
and what it costs you in accuracy.

All facts below were verified by running the fetchers on 2026-11-03.

---

## The canonical schema

Everything normalises to these columns:

| column   | dtype   | meaning                            |
|----------|---------|------------------------------------|
| `time`   | int64   | UNIX seconds UTC, bar OPEN time     |
| `open`   | float64 |                                     |
| `high`   | float64 |                                     |
| `low`    | float64 |                                     |
| `close`  | float64 |                                     |
| `volume` | float64 | 0.0 when the source has no volume   |

Extra columns are preserved: `session_date` (YYYY-MM-DD) and `source`.

---

## Source comparison

| source | key | free coverage | intraday | catch |
|--------|-----|---------------|----------|-------|
| Binance | none | full history, crypto spot, 24/7 | 1m to 1d | USDT pairs only, spot only |
| Yahoo Finance | none | daily back to 2000 for NQ=F | 1m 30d, 5m-30m 60d, 1h 730d | unadjusted rolls, personal use |
| Dukascopy | none | tick-level, per instrument per hour | any, you resample | slow, rate limited, CFD indices |
| CSV drop zone | n/a | whatever you bring | whatever you bring | your file, your problem |

---

## Binance

```python
from prop_lab.data import binance_src
bars = binance_src.fetch(symbol="BTCUSDT", interval="1h", start="2024-01-01")
```

Endpoint: `GET https://api.binance.com/api/v3/klines`. No API key, no
account, no registration.

Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d.

Verified 2026-11-03: the endpoint returns real klines JSON with no
authentication.

Caveats:
- Spot only. Futures markets use a different endpoint and are not
  implemented.
- USDT-quoted pairs only.
- Geo-restrictions apply in some countries, so the API can refuse.
- Volume is real trade volume for the spot pair.

This is the best source in the package for long intraday history,
because nothing truncates it.

---

## Yahoo Finance

```python
from prop_lab.data import yahoo_src
bars = yahoo_src.fetch(symbol="NQ=F", interval="1d", start="2020-01-01")
```

Uses `yfinance`. No API key. There is no official Yahoo Finance API;
`yfinance` reads Yahoo's public endpoints and its maintainers say it is
for personal and research use.

Interval reach, checked in the `yfinance` 1.7.0 source on 2026-10-10:

| interval | how far back |
|----------|---------------|
| 1m | last 30 days (8 days with `period="max"`) |
| 2m, 5m, 15m, 30m | last 60 days |
| 60m, 1h | last 730 days |
| 1d and longer | full history, decades |

Verified 2026-11-03: `yf.download("NQ=F", interval="1d",
start="2020-01-01", end="2026-11-01", auto_adjust=False)` returns 1720
rows.

**The unadjusted roll problem.** Yahoo's continuous futures symbols
(`NQ=F`, `MNQ=F`, `ES=F`) are NOT back-adjusted. They switch contract at
expiry and leave a price gap of roughly 1 percent, which is a fake move.
In September 2026 the December contract traded about 1 percent above the
September contract every day until expiry. A strategy that trades that
gap is trading a data artefact, and a backtest over it will show a
drawdown that never happened in the market.

The loader emits a warning when it sees a `=F` symbol. Do not suppress
it. If your strategy lives or dies on the roll, use a properly adjusted
series or trade a single expiry.

Also note: the daily bar timestamps sit at midnight in the exchange's
local timezone, and a futures "day" runs 18:00 to 17:00 ET. A calendar
day is not a trading day.

---

## Dukascopy

```python
from prop_lab.data import dukascopy_src
bars = dukascopy_src.fetch(
    instrument="USA30IDXUSD", start="2026-06-01", end="2026-06-02",
    timeframe="5m",
)
```

Endpoint: `https://datafeed.dukascopy.com/datafeed/{INSTR}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5`

The month in the path is **zero-based**: 06 means July.

Each file is LZMA-compressed. The decompressed bytes are fixed 20-byte
big-endian records:

| bytes | type     | meaning                     |
|-------|----------|-----------------------------|
| 0-3   | uint32   | milliseconds into the hour  |
| 4-7   | uint32   | ask price, scaled           |
| 8-11  | uint32   | bid price, scaled           |
| 12-15 | float32  | ask volume                  |
| 16-19 | float32  | bid volume                  |

The price scale is 100000 for most FX, 1000 for USDJPY, XAUUSD and the
index instruments.

Verified 2026-11-03:
- EURUSD and USA30IDXUSD decode to real bid and ask prices.
- A single hour file takes about 19 seconds.
- Four files in parallel take about 33 seconds.
- The **monthly bulk `.zip` endpoint timed out at 120 seconds**, so this
  package does not use it.

Caveats:
- The server silently slows down past about 4 concurrent requests. It
  does not error, it just gets slow and stays slow.
- It rate limits with HTTP 429 if you push too hard.
- A quiet hour (weekend, the daily break, a holiday) returns **HTTP 200
  with an empty body**, not a 404. Treat empty as "no data" and never
  retry it, or you will hammer the server every weekend forever.
- The index instruments (`USA30IDXUSD`, `USA500IDXUSD`,
  `USATECHIDXUSD`) are **CFDs, not futures**. They track the index but
  they are not the instrument a prop firm trades. Do not mix them with
  futures data without saying so.
- Redistributing the data is subject to Dukascopy's terms. This package
  downloads for research and does not redistribute.

The fetcher returns a coverage report in `df.attrs["coverage"]` listing
how many hours were requested, how many had data, and which were missing.
Check it before trusting a backtest.

---

## The CSV drop zone

```python
from prop_lab.data import load_bars
bars = load_bars("data/MNQ_5m.csv", source="my-broker-export")
```

The loader accepts many spellings for the time column: `time`, `ts`,
`timestamp`, `date`, `datetime`, `Date`, `Datetime`, `open_time`. Real
datetime values are converted to UNIX seconds UTC; the resolution is read
from the data rather than assumed, because pandas 3.0 uses microseconds
while older versions used nanoseconds.

CSV and Parquet are both supported, as are http(s) URLs.

---

## Session dates: the rule that decides your results

A futures trading day runs **18:00 ET to 17:00 ET the next day**.

```python
from prop_lab.data import assign_session_dates
bars["session_date"] = assign_session_dates(bars, session_open_hour=18)
```

For 24/7 crypto use `session_open_hour=0`.

Why this matters more than it looks: grouping futures bars by the UTC
calendar date splits every trading session into two fragments. On 14
weeks of MNQ 5-minute data that turns 71 real trading days into 86
"session" fragments, and a session-anchored strategy will fire on a
fragment boundary that never existed in the market. In this package that
difference moved the reference strategy from 73 trades to 66.

Prop firms reset the daily loss limit and calculate the end-of-day
drawdown at 18:00 ET. If your backtest uses a different session
boundary, it does not match the account it is supposed to predict.

---

## Finding a source yourself

When no built-in route has the instrument:

1. Search for the instrument plus "historical data csv free" or "free
   tick data api".
2. Prefer a documented API over a page that needs a browser session.
3. Check what the free tier actually covers before downloading. Plenty
   of sources give daily bars free and charge for intraday.
4. Verify what you downloaded: row count, date range, and sane first
   and last rows. Print the coverage.
5. Add it as a loader in `prop_lab/data/` so the next person does not
   have to repeat the work.

Do not scrape a site that forbids it. Do not use a source that requires
a paid tier for the data the user asked for.
