# AGENTS.md

Instructions for an AI agent using this repository.

You are helping a person answer one question honestly:

> "I have a trading idea. If I backtest it on real data and size it against
> a prop firm's rules, what is my actual probability of passing the
> evaluation, and does the edge survive out of sample?"

This repository exists to answer that question without lying about it. A
number that looks good but came from a fabricated assumption is worse
than no number, because the person will act on it.

---

## 1. What this repository is

A Python package (`prop_lab`) that:

1. Fetches or loads free market data in one canonical format.
2. Backtests a strategy you write against that data.
3. Tests the strategy out of sample with purged walk-forward, and tests
   whether the edge is statistically distinguishable from zero.
4. Simulates thousands of prop firm challenge attempts by bootstrapping
   the real trade distribution, and reports the result against the
   zero-edge baseline.
5. Produces labelled surfaces showing pass rate against risk geometry
   and rulesets, each axis marked as **measured** or **modelled**.
6. Writes a Markdown report with a mandatory honesty section.

The command line also exposes `inspect` (a strategy's parameters and the
assumptions it carries), `trades` (the trade list, for auditing by hand)
and `sweep` (parameter sweeps).

Read `SPEC.md` for the full API contract. This file is the workflow.

---

## 2. Setup

```bash
git clone <repo-url> prop-lab
cd prop-lab
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Core dependencies are numpy, pandas, matplotlib, `yfinance` and `requests`.
The data adapters are **core**, not an optional extra: the Yahoo adapter is
imported by the regression tests, and the Binance and Dukascopy adapters
need `requests`. A clean `pip install -e .` must produce an environment
where `import prop_lab.data` works. The only extra is `dev` (pytest).

The suite is offline by default. Tests that need the network are marked
`network` and are skipped unless `PROPLAB_NETWORK_TESTS=1` is set.

---

## 3. Turning a plain-language idea into a strategy

This is the core of your job. A user says something like "buy when the
market pulls back to the 20 EMA in the first hour and then breaks out".
You translate that into a class that satisfies the strategy contract.

### The contract

```python
from prop_lab.core import Bar, Signal, StrategyCore

class MyStrategy(StrategyCore):
    name = "my_strategy"

    def __init__(self, window: int = 20, rr_ratio: float = 2.0, **kwargs):
        super().__init__(window=window, rr_ratio=rr_ratio)
        self.window = window
        self.rr_ratio = rr_ratio
        self._buffer: list[float] = []

    def on_session_start(self, bar: Bar) -> None:
        """Reset per-session state here."""
        self._buffer = []

    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        """Return a Signal on this bar's close, or None.

        The fill happens at the OPEN of the next bar. Never assume the
        fill is at this bar's close.
        """
        ...
        return Signal(
            direction="long",            # 'long' or 'short'
            stop_distance=12.5,          # price points, MUST be > 0
            rr_ratio=2.0,                # target = stop_distance * rr_ratio
            reason="my_long_setup",      # short machine-readable tag
            bar_index=bar_index,
            timestamp=bar.timestamp,
        )
```

Rules that are not negotiable:

- `stop_distance` must be greater than zero. The engine raises if it is
  not.
- `direction` must be exactly `'long'` or `'short'`.
- One signal per bar at most.
- All state that should reset each session goes in `on_session_start`.
- Every parameter that affects behaviour is an `__init__` argument with a
  default. The engine introspects `__init__` for `params_help()`, and the
  report prints the parameters.

### Translation rules: how to resolve ambiguity

The user will use vague words. Resolve them explicitly, choose one
interpretation, and write the choice into a `notes` attribute on the
strategy. Never silently pick.

| The user says | Interpret as | Say so in notes |
|---|---|---|
| "momentum" | EMA slope over N bars, or ROC | "momentum = EMA(20) rising over 5 bars" |
| "pullback" | price touches a moving average | "pullback = low <= EMA(20)" |
| "breakout" | close beyond the N-bar high/low | "breakout = close > max(high[-20:])" |
| "volatility" | ATR(N) | "volatility = ATR(14)" |
| "trend" | EMA(50) vs EMA(20) alignment | "trend = EMA20 > EMA50" |
| "session" | the futures session, 18:00 ET to 17:00 ET | use the `session_date` column |
| "confirmation" | the condition must hold on the entry bar | "confirmation = both conditions on the same bar" |

Rules for the translation:

1. Every condition must be computable from the bar stream alone. If it
   needs data you do not have, say so and stop.
2. Prefer the simplest interpretation that a trader would recognise.
3. Write the assumption down. The strategy class should carry a
   `notes` string listing every assumption you made.
4. If the user's description is genuinely ambiguous in a way that
   changes the result, ask one specific question. Do not ask about
   things you can decide.

### Stop placement

Pick one and document it:

- **ATR multiple**: `stop_distance = atr_multiple * ATR(14)`. Most
  robust across instruments.
- **Structure**: stop beyond the swing high/low or the session extreme.
  Requires the strategy to track those levels.
- **Fixed points**: `stop_distance = 12.5`. Only sensible for one
  instrument at one price level.

### Defaults when the user does not say

- R:R: 2.0
- Direction: both, unless the wording is directional
- Stop: ATR(14) multiple of 1.0, unless structure is implied

---

## 4. Getting data

Four routes, all free. See `docs/DATA.md` for the full table.

### Route A: the built-in fetchers

```python
from prop_lab.data import binance_src, yahoo_src, dukascopy_src

bars = binance_src.fetch(symbol="BTCUSDT", interval="1h", start="2024-01-01")
bars = yahoo_src.fetch(symbol="NQ=F", interval="1d", start="2020-01-01")
bars = dukascopy_src.fetch(instrument="USA30IDXUSD", start=..., end=..., timeframe="5m")
```

What each gives you, verified 2026-11-03:

- **Binance** (`api.binance.com/api/v3/klines`): no API key, full
  history, crypto spot, 24/7. Intervals 1m through 1d. The most reliable
  source for long intraday history.
- **Yahoo** (via `yfinance`): daily history goes back decades for
  `NQ=F`, `MNQ=F`, `ES=F`. Intraday is limited by Yahoo, not by you:
  1m reaches 30 days back, 5m to 15m reach 60 days, 1h reaches 730 days.
  Continuous futures are **unadjusted**: they switch contract at expiry
  and leave a roughly 1 percent gap that is a fake price move. The loader
  warns about this. Do not hide the warning.
- **Dukascopy** (`datafeed.dukascopy.com`): tick-level history per
  instrument per hour, LZMA-compressed `.bi5` files. Long history for FX
  and indices. Slow: roughly 19 seconds per hour file, so use 4 parallel
  workers and be patient. The server silently slows down beyond that, and
  it rate-limits with HTTP 429 if you push too hard. The monthly bulk zip
  times out, so do not use it.

### Route B: the CSV drop zone

If the user already has data, drop a CSV in the repo and load it:

```python
from prop_lab.data import load_bars
bars = load_bars("data/MNQ_5m.csv", source="my-export")
```

The loader accepts many column spellings (`time`, `ts`, `timestamp`,
`Date`, `Datetime`) and normalises to `time, open, high, low, close,
volume`. If the CSV has a `Date` column with real datetimes it converts
them to UNIX seconds UTC.

### Route C: find the data yourself

When no built-in route has the instrument, search for it. The pattern
that works:

1. Search for the instrument plus "historical data csv free" or
   "free tick data api".
2. Prefer a source with a documented API over a page that needs a
   browser session.
3. Check what the free tier actually covers before downloading. Many
   sources give daily bars free and charge for intraday.
4. Verify the download: check the row count, the date range, and that
   the first and last rows are sane. Print the coverage in the report.
5. Write the download into a loader in `prop_lab/data/` so the next
   person does not repeat the work.

Do not scrape a site that forbids it, and do not use a source that
requires a paid tier for the data the user wants.

### The session date rule

A futures trading day runs **18:00 ET to 17:00 ET**. Group bars by
session, not by calendar date, or every session gets split in two.

```python
from prop_lab.data import assign_session_dates
bars["session_date"] = assign_session_dates(bars, session_open_hour=18)
```

For 24/7 crypto use `session_open_hour=0`. The CLI infers this from the
source column: `binance` gets `0`, everything else gets `18`.

---

## 5. The workflow

Run these in order. Do not skip to the Monte Carlo.

### The stop rule

**If the walk-forward verdict is `fails` or `no_edge`, that is the answer.
Report it and stop.**

The temptation at that point is to start "fixing" the strategy: nudging a
parameter, sweeping for a cell that passes, trying another firm's ruleset,
dropping the folds that look bad, or trimming the sample until the number
improves. All of that is overfitting the test set. It produces a number that
describes the search, not the strategy, and the person reading it will trade
on it.

- **Forbidden after a negative verdict:** tuning parameters to escape it,
  sweeping to find a passing cell, switching ruleset until one passes,
  dropping folds, trimming the sample, or re-running with a different seed
  until the interval clears zero.
- **Allowed:** producing the Monte Carlo and the surfaces for completeness,
  labelled as descriptive context. Never present them as the result of the
  strategy.
- **If you want to keep going**, that is a new hypothesis, not a
  continuation. Declare it, and test it on data you did not use to choose it.

Saying "this idea has no edge" in one paragraph is a finished, useful
deliverable. It is not a failure to produce.

### Step 1: Load and check the data

```python
from prop_lab.data import load_bars, assign_session_dates
bars = load_bars("data/MNQ_5m.csv", source="...")
print(len(bars), bars["time"].min(), bars["time"].max())
print(bars.isna().sum().sum(), "missing values")

# Sessions are assigned separately. 18 is the futures convention;
# 0 is correct for a 24/7 market such as crypto.
bars["session_date"] = assign_session_dates(bars, session_open_hour=18)
```

`load_bars` does **not** take a `session_open_hour`. It reads the bars
and, if the source column says `binance`, the CLI infers `0`; everything
else defaults to `18`. If you load bars directly, assign the session
yourself as above, and get it right: it changes where the daily loss
limit resets and where positions are force-closed.

Report the coverage and the gap count. Gaps in the middle of a session
matter, gaps at the weekend do not.

### Step 2: Backtest

```python
from prop_lab.core import run_backtest, summarize_trades

trades = run_backtest(MyStrategy(), bars)
stats = summarize_trades(trades)
print(stats)
```

Then sanity-check the result before trusting it:

- Trade count plausible for the timeframe? 73 trades over 14 weeks of
  5-minute data is plausible. 4,000 is a bug.
- Win rate and mean R consistent? A 60 percent win rate with a 0.1R
  mean means the winners are tiny.
- `max_drawdown_r` plausible? If it is 0.3R over 70 trades, something
  is wrong with the fill model.
- Does the strategy actually trade when you expect it to? Print the
  first few trades and read them.

### Step 3: Out of sample

```python
from prop_lab.validate import walk_forward

wf = walk_forward(
    lambda: MyStrategy(), bars,
    n_splits=5, purge_bars=200, embargo_bars=200,
)
print(wf.verdict, wf.is_mean_r, wf.oos_mean_r, wf.degradation)
print(wf.edge_ci_low, wf.edge_ci_high, wf.edge_t_statistic)
```

Verdicts:

| verdict | meaning |
|---|---|
| `holds` | The edge survived out of sample **and** is statistically distinguishable from zero. |
| `degrades` | There is an edge, but it shrank out of sample. |
| `no_edge` | **No edge was detected.** The confidence interval on the mean R includes zero. |
| `fails` | The out-of-sample mean R is zero or negative, or the edge degraded badly. |

`holds` requires `wf.edge_significant` to be true. Degradation alone is
never enough: a strategy with an edge of exactly zero does not "hold",
there was nothing to hold. Use the `no_edge` verdict for that.

**Always use a purge and an embargo.** `purge_bars` removes the end of
each training window so trades straddling the boundary cannot leak
test-period prices into the in-sample statistics; `embargo_bars` adds a
gap between the training window and the test block. The default of 0 for
both means the walk-forward is **not** purged, even though it is called a
purged walk-forward. Pick a value that covers the longest holding
period in bars.

If the verdict is `fails` or `no_edge`, that is the answer. Do not go on
to the Monte Carlo and present a pass rate for a strategy that has
nothing to hold out of sample. Say it fails, or say no edge was found.

### Step 3b: Test the edge separately

```python
from prop_lab.validate import test_edge

edge = test_edge(np.array([t.pnl_r for t in trades]))
print(edge.significant, edge.ci_low, edge.ci_high, edge.t_statistic)
print(edge.describe())
print(edge.warning())
```

This is a bootstrap confidence interval on the mean R per trade. It is the
primary edge test, because the R-multiple distribution is skewed and
bounded and a t-test assumes it is not. The t-statistic is a cross-check,
not the primary.

`test_edge` never raises on an empty sample; it returns an inconclusive
result with zeroed numbers. `edge.conclusive` is false when there are
fewer than 30 trades, and a non-significant result on a small sample
means **inconclusive**, not **no edge**. Say which one it is.

### Step 4: Monte Carlo

```python
from prop_lab.firm import get_preset, monte_carlo_challenge
import numpy as np

rules = get_preset("apex-50k-eod")
pnl = np.array([t.pnl_r for t in trades])
sessions = [t.session_date for t in trades]

mc = monte_carlo_challenge(
    pnl, rules, risk_geometry="frac_2pct",
    n_attempts=10_000, seed=42,
    session_dates=sessions, compare_iid=True,
)
print(mc.describe())
```

Use 10,000 attempts. 500 gives a confidence interval about 4 percentage
points wide, which cannot distinguish the risk geometries.

Pass `session_dates` so the block bootstrap can preserve the within-day
loss clustering. Without it the bootstrap degenerates to i.i.d., which
systematically **understates** the risk because losses arrive in
clusters and a trailing drawdown is a path-dependent barrier. Set
`compare_iid=True` to report the gap.

### Always report the zero-edge baseline beside the pass rate

```python
from prop_lab.firm.monte_carlo import zero_edge_baseline

baseline = zero_edge_baseline(rules, n_trades=len(pnl), n_attempts=10_000, seed=42)
print(f"strategy {mc.pass_rate:.1%} vs zero-edge baseline {baseline:.1%}")
```

A pass rate on its own is a score, not a comparison, and it reads like
a result. **A challenge is not a coin toss**: the profit target sits
above the starting balance and the trailing drawdown sits below it, so a
random walk that happens to run up early reaches the target before it
breaches the floor. For the APEX 50K EOD rules a strategy with **no edge
at all** scores around **27%**.

So a strategy scoring 27% has demonstrated nothing, and a strategy
scoring 40% has demonstrated something. Print both numbers, always.
The `report` command does this for you when you pass
`baseline_pass_rate=baseline`.

A narrow confidence interval on the pass rate is a precise measurement
of a possibly-worthless strategy. It is not evidence of an edge. The
edge test in step 3b is what speaks to that.

### Step 5: Surfaces

```python
from prop_lab.surface import risk_geometry_surface, rr_surface, winrate_surface

s1 = risk_geometry_surface(pnl, rules, session_dates=sessions)   # both axes measured
s2 = rr_surface(pnl, rules, measured=True,
                strategy_factory=lambda: MyStrategy(), df=bars,
                session_dates=sessions)                          # R:R axis measured
s3 = winrate_surface(pnl, rules, session_dates=sessions,
                     operating_point=(edge.win_rate, edge.rr_ratio),
                     operating_point_label="this strategy")       # both axes modelled
```

**Pass `session_dates` to every builder.** Without it the surfaces use
i.i.d. resampling while the headline Monte Carlo uses the block
bootstrap, so the report can print two different pass rates for one
strategy. This was a real defect; do not reintroduce it.

`rr_surface(measured=True)` re-runs the backtest at every R:R, which
changes which trades exist. Only pass `strategy_factory` and `df` when
you have them, because without them the R:R axis falls back to the
modelled approximation, which can be tens of percentage points away
from the measured value. Check the `R:R axis: measured` line in the
report's honesty section; if it says `modelled`, the surface is an
approximation and you must say so.

Pass `operating_point` to `winrate_surface` so the plot marks where the
strategy actually sits. The win-rate axis is modelled in every case, so
the marker is the only thing on that surface that is a fact.

### Step 6: Report

```python
from prop_lab.report import build_report

md = build_report(
    wf=wf, mc=mc, surfaces=[s1, s2, s3],
    strategy_name="MyStrategy", data_source="MNQ 5m CSV",
    data_range="2026-05-10 .. 2026-08-17", rules=rules,
    extra_warnings=surface_notes,          # any caveat the surfaces add
    baseline_pass_rate=baseline,           # the zero-edge comparison
)
print(md)
```

`build_report` does not know about the surfaces' own caveats, so pass
them through `extra_warnings`. `baseline_pass_rate` is what puts the
zero-edge comparison in the headline. Omit either and the report is
missing something the reader needs.

### Or use the CLI

```bash
proplab fetch --source binance --symbol BTCUSDT --interval 15m --start 2023-01-01 --out data/
proplab inspect --strategy examples/strategies/trend_continuation.py:TrendContinuation
proplab backtest --strategy examples/strategies/trend_continuation.py:TrendContinuation --data data/MNQ_5m.csv
proplab walkforward --strategy examples/strategies/trend_continuation.py:TrendContinuation \
    --data data/MNQ_5m.csv --purge-bars 200 --embargo-bars 200
proplab montecarlo --strategy examples/strategies/trend_continuation.py:TrendContinuation \
    --data data/MNQ_5m.csv --firm apex-50k-eod --attempts 10000
proplab surface --strategy examples/strategies/trend_continuation.py:TrendContinuation \
    --data data/MNQ_5m.csv --firm apex-50k-eod --out reports/
proplab report --strategy examples/strategies/trend_continuation.py:TrendContinuation \
    --data data/MNQ_5m.csv --firm apex-50k-eod --out reports/
proplab trades --strategy examples/strategies/trend_continuation.py:TrendContinuation \
    --data data/MNQ_5m.csv --out trades.csv
proplab sweep --strategy examples/strategies/trend_continuation.py:TrendContinuation \
    --data data/MNQ_5m.csv --param window=10,20,40 --out sweep.csv
```

Notes on the CLI:

- The strategy spec is `path/to/file.py:ClassName`. The path may be
  absolute or relative. On Windows the spec is split on the **last**
  colon, because the drive letter is itself a colon.
- `--session-open-hour` defaults to inferring from the data source: 0
  for Binance, 18 otherwise. Pass it explicitly for anything that is
  neither.
- `--purge-bars` and `--embargo-bars` default to 0, which means the
  walk-forward is **not** purged. Set them.
- `report` writes `report.md` **and the surface figures** into the
  output directory. `surface` writes only the figures.
- **A 3D surface is only written when the grid varies along both axes.** A
  grid with one value on an axis is a list of numbers, and its 3D view is a
  vertical curtain, not a surface; it gets a heatmap and nothing else.
  `surface` and `report` default to **all three firm presets**, so the
  ruleset axis is a real axis. Pass one firm (`--firm apex-50k-eod`) to
  model that account only — the risk-geometry figure then becomes a
  heatmap, because its ruleset axis is constant. The other surfaces and
  the headline Monte Carlo always use the first firm named.
- Only two surfaces get a 3D view: the ruleset x risk-geometry cross, and
  win rate x R:R. The R:R x risk-geometry surface is a heatmap, because
  its cells are read one at a time.
- `inspect` shows the strategy's parameters and the assumptions it
  carries. Use it before quoting a strategy's configuration.
- `trades` exports the trade list so the backtest can be audited by
  hand.
- `sweep` sweeps strategy parameters and writes a CSV. Sweeping is how
  you find out whether an idea is real or whether you found one lucky
  cell. A parameter sweep on the same data you backtested on is a
  multiple-comparisons problem; treat the best cell as a hypothesis, not
  a result.

---

## 6. The honesty rules

These are the part that matters. A number that looks good but came from
an assumption is worse than no number.

### Surface axes carry provenance

Every axis of every surface is either:

- **measured**: computed from a real backtest or a real simulation.
- **modelled**: assumed. Not from a backtest.

Which is which:

| Axis | Provenance | Why |
|---|---|---|
| Risk geometry | measured | The trade distribution is fixed; only the sizing changes. Each cell is a genuine simulation. |
| R:R | measured, but only if you re-run the backtest | Changing R:R changes which trades exist. Scaling winners in an existing array is modelled, not measured. The gap between the two was measured at up to **56 percentage points**, so "modelled" here is not a rounding difference. |
| Win rate | **always modelled** | You cannot re-backtest a win rate. Forcing a different win rate means fabricating trades. |

The R:R axis is measured only when the surface builder was given a
`strategy_factory` and the bars, so it can re-run the backtest at every
R:R. Check the axis label in the report's honesty section before you
describe a surface.

The `winrate_surface` is the surface whose shape DeltaTrend Trading
popularised on TikTok and YouTube using Topstep rules: pass rate as a
function of win rate and R:R. It is useful for understanding the
geometry of a challenge, and its win-rate axis is parametric by
construction. Say that whenever you show it.

### Never claim more than the data supports

- Fewer than 30 trades: the pass rate is illustrative, not an estimate.
  Fewer than 10: it is a toy.
- The confidence interval on the pass rate must be printed with it.
- **The zero-edge baseline must be printed next to it.** A pass rate
  without the baseline cannot be read. Around 27% on the APEX 50K EOD
  rules is what a strategy with no edge scores.
- **A pass rate is not an edge.** Say what the confidence interval on the
  mean R says. If it includes zero, no edge was detected, and the
  verdict is `no_edge`.
- "Holds out of sample" is a statement about the tested folds. It is not
  a statement about the future.
- If the data covers 14 weeks, the report says 14 weeks. Do not imply
  more.

### Always include the warnings

The engine produces warnings for small samples, degenerate bootstraps,
synthetic-day approximations and resampling gaps. Put them in the
report. Do not filter them out to make the result look cleaner.

### What is not modelled

State it when it matters:

- slippage and commission
- news gaps and intrabar order of events
- contract scaling plans and payout delays
- the firms changing their rules
- the correlation between the strategy and the person running it
- **intrabar equity excursions.** The drawdown is checked on closed
  trade equity, not intrabar. An account that dips below the trailing
  floor mid-trade and recovers into the close is scored as surviving,
  and real firms check the floor continuously against equity including
  unrealised profit. **The pass rate is an upper bound, not a central
  estimate.** This is the largest known limitation and it must be stated
  wherever a pass rate is quoted.

---

## 7. Known failure modes

These have all cost time in this project. Check them before you trust a
result.

1. **Look-ahead.** A signal computed on bar `i` must be filled at the
   open of bar `i+1`. Filling at the close of bar `i` inflates results.
   The engine handles this; a hand-rolled backtest usually does not.

2. **The last open trade is dropped.** If you write your own backtest,
   close any position still open at the end of the data. Otherwise you
   silently lose 10 to 15 percent of the trades.

3. **NaN in a filter deletes trades.** If a filter source has missing
   data, NaN bars must pass through. Treating NaN as "no signal" quietly
   removes most of the trades and the strategy looks like it does nothing.

4. **Comparing against a drifted baseline.** Before changing anything,
   run the unchanged code and confirm you still get the numbers you
   expect. `scripts/baseline_check.py` does exactly this for the
   reference strategy: it must print 73 trades, 41.1 percent win rate,
   +0.233R. If it does not, stop and find out why before you proceed.

5. **Unadjusted continuous futures.** Yahoo's `NQ=F` switches contract at
   expiry and leaves a fake 1 percent gap. A strategy that trades the
   gap is trading a data artefact.

6. **Session splitting.** Grouping futures bars by calendar date splits
   every session in half. Use `session_date`.

7. **Coin-flip Monte Carlo.** If you see `rng.random() < win_rate` in a
   challenge simulation, the number is meaningless. Every strategy gets
   roughly the same pass rate when the win/loss size distribution is
   thrown away.

8. **Sampling on a daily loss limit.** A bootstrap has no calendar. The
   engine approximates session boundaries with synthetic days of a
   configurable number of trades. This is an approximation and it is
   reported as one.

9. **A bootstrap that does not resample.** `bootstrap_stationary` once
   walked the source array cyclically from a single random start, so with
   `n_trades == len(source)` its output was a *permutation of the source*
   and the resampled mean never varied at all. A bootstrap whose mean
   has zero sampling variability cannot estimate the uncertainty of that
   mean. It is fixed, and `tests/test_bootstrap_properties.py` now
   checks the statistics of the resample rather than only its length.
   **When you change a resampler, test what the output's distribution
   looks like, not just its shape.** The old tests could not have caught
   this.

10. **Two different resamplings in one report.** The surfaces and the
    headline Monte Carlo once ran without `session_dates`, so the report
    could print two pass rates for one strategy. Always thread
    `session_dates` through every surface builder.

11. **The walk-forward that is not purged.** `purge_bars` and
    `embargo_bars` both default to 0, which means the default
    walk-forward leaks across the train/test boundary. If you call it a
    purged walk-forward, set them.

12. **A window rule applied to a 24/7 market.** The futures session runs
    18:00 ET to 17:00 ET. Applied to crypto it splits every UTC day in
    two. Use `--session-open-hour 0` for a 24/7 market.

13. **An absolute path on Windows.** A strategy spec is
    `path:Class`. Split it on the **last** colon. Splitting on the first
    one turns the drive letter `C:` into a module name and the load
    fails with `No module named 'C'`.

14. **`on_bar` is only called while flat.** The engine calls
    `strategy.on_bar` only when there is no open position
    (`core.py:301`), so bars that pass during a trade never reach the
    strategy. A stateless rule is fine. **An indicator folded bar by bar
    inside `on_bar` is not**: `self._closes.append(...)` accumulates over
    a gapped stream, and every derived value — RSI, ATR, EMA — is then
    wrong. The backtest still runs and still prints trades, which is what
    makes this dangerous. Two bundled strategies carry the latent problem
    (`examples/strategies/doji_ema_trend.py`, `strategies/ema_trend_pullback.py`);
    in an end-to-end test a stateful implementation saw 36,818 of 49,628
    bars. Precompute the indicator over the whole series and look it up by
    `bar_index`, or keep the rule stateless. If you are auditing a
    strategy's stops, check that each one matches the indicator of a real
    bar — if none do, this is why.

---

## 8. Report structure

The report is generated by `build_report` and always has these sections
in this order. Do not remove or reorder them.

1. **Headline** — strategy, data, trade count, in-sample versus
   out-of-sample mean R, pass rate with its confidence interval, the
   zero-edge baseline it must be compared against, and the verdict. The
   headline also states the confidence interval on the mean R, and says
   plainly when no edge was detected.
2. **Honesty** — every surface axis with its provenance. Mandatory.
3. **Risk geometry** — pass rate per geometry.
4. **Out-of-sample** — per fold.
5. **Caveats** — every warning plus the standing ones, including the
   intrabar-excursion caveat.
6. **Provenance** — citations with URLs and the date each was checked.

The two the reader most needs are easy to lose, so check them every
time: **the zero-edge baseline** next to the pass rate, and **the edge
interval** next to the verdict. A pass rate without the baseline and a
verdict without the interval are both unreadable.

Plain text. No emoji. Numbers, not adjectives.

---

## 9. Reference material

Read these before guessing:

- `SPEC.md` — the full API contract.
- `docs/DATA.md` — every data source, what it gives, what it costs.
- `docs/HONESTY.md` — the measured versus modelled argument in full.
- `src/prop_lab/firm/rules.py` — the firm rulesets, with the source URL
  and the date each was checked. **Re-verify before trusting them**,
  firms change their terms.

---

## 10. Attribution

**The pass-rate surface that this project exists to extend is DeltaTrend
Trading's public work** — Thomas Skinner, `@deltatrendtrading` on YouTube,
TikTok and Instagram. He popularised the surface on TikTok and YouTube
using Topstep rules, and he has publicly pushed back on creators who
copied the surface without understanding what the axes mean. That pushback
is why every axis here is labelled measured or modelled, and why the
win-rate axis is stated to be parametric by construction. Credit him by
name and URL. It is the inspiration, and it belongs at the top of the
README, not in a footnote.

The architecture comes from the prop-risk-geometry project, originally
`github.com/hunter547/prop-risk-geometry` and now
`github.com/dsilva42/prop-risk-geometry`. **It carries no license file**,
so it is cited as the origin of the idea and no code was taken from it;
this is an independent implementation. The purged walk-forward follows
Lopez de Prado's "Advances in Financial Machine Learning" (chapters 7 and
12) and the implementation conventions of
`github.com/eslazarev/purged-cross-validation`.

Cite these when you reuse the methods. Do not present them as your own.
