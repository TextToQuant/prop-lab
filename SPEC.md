# prop-lab — Build Spec (source of truth)

A repo a stranger's AI agent can clone and use to answer one question honestly:

> "I have a trading idea. If I backtest it on real data and then size it
> against a prop firm's rules, what is my actual probability of passing the
> evaluation — and can the edge survive out of sample?"

Everything below is a contract. Implementations MUST match the signatures.

---

## 0. Package layout

```
prop-lab/
├── AGENTS.md                 ← the agent contract (deliverable #1)
├── README.md                 ← human quickstart
├── SPEC.md                   ← this file
├── pyproject.toml
├── src/prop_lab/
│   ├── __init__.py
│   ├── core.py               ← Bar, Signal, StrategyCore, run_backtest
│   ├── indicators.py         ← pure indicator functions
│   ├── data/
│   │   ├── __init__.py       ← load_bars(), STANDARD COLUMNS, normalise()
│   │   ├── csv_drop.py       ← CSV drop-zone loader
│   │   ├── yahoo_src.py      ← yfinance
│   │   ├── binance_src.py    ← public REST
│   │   └── dukascopy_src.py  ← .bi5 LZMA tick decode
│   ├── validate/
│   │   ├── __init__.py
│   │   └── walkforward.py    ← purged walk-forward OOS
│   ├── firm/
│   │   ├── __init__.py
│   │   ├── rules.py          ← FirmRules dataclass + presets
│   │   ├── monte_carlo.py    ← bootstrap challenge sim
│   │   └── funded.py         ← funded-phase sim
│   ├── surface/
│   │   ├── __init__.py
│   │   └── surfaces.py       ← measured vs modelled surfaces
│   ├── report/
│   │   ├── __init__.py
│   │   └── report.py         ← labeled markdown report
│   └── cli.py                ← `proplab` CLI
├── examples/
│   ├── strategies/
│   │   ├── trend_continuation.py
│   │   └── opening_range_breakout.py
│   └── README.md
├── tests/
└── docs/
    ├── DATA.md
    └── HONESTY.md
```

---

## 1. Data schema — THE canonical bar format

Every source normalises to exactly this. Non-negotiable.

| column  | dtype      | meaning                                   |
|---------|------------|-------------------------------------------|
| `time`  | int64      | UNIX seconds, UTC, bar OPEN time          |
| `open`  | float64    |                                          |
| `high`  | float64    |                                          |
| `low`   | float64    |                                          |
| `close` | float64    |                                          |
| `volume`| float64    | 0.0 when the source has no volume         |

Extra columns allowed and preserved: `session_date` (str YYYY-MM-DD),
`source` (str).

### `session_date` rule (futures)

A futures trading day runs **18:00 ET → 17:00 ET next day**. The bar at
16:59 ET on 2026-10-28 belongs to session `2026-10-28`, together with the
bars from 18:00 ET on 2026-10-27. A UTC or calendar day would split each
session in two — never group by calendar date for futures.

Implement `assign_session_dates(df, tz="America/New_York", session_open_hour=18)`.
Bars with local time >= 18:00 belong to the NEXT calendar day; bars before
18:00 belong to the current calendar day.

For crypto (24/7) use `session_open_hour=0` → session_date == UTC date.

---

## 2. `prop_lab.core`

```python
@dataclass
class Bar:
    timestamp: int          # UNIX seconds UTC
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def date(self) -> str: ...        # 'YYYY-MM-DD' UTC
    @property
    def time_utc(self) -> str: ...    # 'HH:MM' UTC

@dataclass
class Signal:
    direction: str          # 'long' | 'short'
    stop_distance: float    # price points, always > 0
    rr_ratio: float         # target = stop_distance * rr_ratio
    reason: str
    bar_index: int
    timestamp: int | None = None

class StrategyCore:
    name: str = "base"
    def __init__(self, **params): ...
    def on_session_start(self, bar: Bar) -> None: ...
    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        raise NotImplementedError
    def on_session_end(self, bar: Bar) -> None: ...
    @classmethod
    def params_help(cls) -> str: ...   # introspects __init__ signature

def iter_bars(df, ts_col="time") -> Iterator[tuple[int, Bar]]: ...
```

`iter_bars` must accept a frame whose time column is `time` OR `ts`
(column-flexible — case-insensitive header lookup).

### `run_backtest`

```python
@dataclass
class Trade:
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
    session_date: str

def run_backtest(strategy: StrategyCore, df, *, direction="both",
                 use_sessions=True, force_close_eod=True,
                 risk_points: float | None = None) -> list[Trade]:
```

Fill model:
- A signal on bar `i` is filled at the **open of bar i+1**. Never at the
  close of bar `i` (that is look-ahead).
- Stop = `entry_price ∓ stop_distance`. Target = `entry_price ±
  stop_distance * rr_ratio`.
- Within the holding bar, if BOTH stop and target are touched, assume the
  **stop** filled (pessimistic). Document this.
- `force_close_eod=True` closes any open position at the close of the last
  bar of its session.
- `pnl_points` = signed (exit_price − entry_price) for long, negated for
  short.
- `pnl_r` = `pnl_points / stop_distance`.

This engine is the single source of truth for trade generation. Nothing
else may re-implement it.

---

## 3. `prop_lab.data`

```python
STANDARD_COLUMNS = ["time", "open", "high", "low", "close", "volume"]

def normalise(df, *, source="unknown") -> pd.DataFrame:
    """Coerce any OHLCV frame to STANDARD_COLUMNS + session_date."""

def load_bars(path_or_url, *, source=None, timeframe=None,
               start=None, end=None) -> pd.DataFrame:
    """CSV drop-zone loader. Accepts .csv/.parquet path or http(s) URL."""

def resample(df, timeframe: str) -> pd.DataFrame:
    """'5m','15m','1h','4h','1d'. Drops empty buckets, sorts by time."""
```

### Source adapters

Each exposes `fetch(...) -> pd.DataFrame` returning the standard schema.

**`binance_src.fetch(symbol="BTCUSDT", interval="1m", start=None, end=None)`**
- Public REST, no key: `GET https://api.binance.com/api/v3/klines`.
- `interval` ∈ {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d}.
- Page backwards from `end` until `start` (each call = 1000 bars).
- VERIFIED LIVE 2026-11-03: returns real klines JSON.

**`yahoo_src.fetch(symbol="NQ=F", interval="1d", start=None, end=None)`**
- Via `yfinance`. Symbols: `NQ=F`, `MNQ=F`, `ES=F`, `^VIX`, `BTC-USD`.
- VERIFIED LIVE 2026-11-03: `yf.download('NQ=F', interval='1d',
  start='2020-01-01')` → 1720 rows.
- yfinance intraday limits (checked in source 2026-10-10): 1m → last 30d
  (8d with period="max"), 2m–30m → 60d, 60m/1h → 730d, 1d+ → full history.
- Use `auto_adjust=False`.
- Yahoo continuous futures are **unadjusted** — they switch contract at
  expiry and leave a ~1% price gap. The loader MUST expose
  `warn_unadjusted_roll=True` by default and print a warning naming the
  affected symbol. Do not silently hide this.

**`dukascopy_src.fetch(instrument="USA30IDXUSD", start=..., end=..., timeframe="5m")`**
- Per-hour `.bi5` tick files:
  `https://datafeed.dukascopy.com/datafeed/{INSTR}/{YYYY}/{MM_zero_based}/{DD}/{HH}h_ticks.bi5`
- Month in the path is **zero-based** (06 = July).
- LZMA-compressed, then fixed 20-byte big-endian records:
  `>IIIff` = ms_since_hour_start, ask_int, bid_int, ask_vol, bid_vol.
- Price divisor: 100000 for FX, 1000 for USDJPY/XAUUSD/indices.
- A quiet hour returns **HTTP 200 with an empty body**, not 404. Treat empty
  as "no data", never as failure. Never retry weekends forever.
- VERIFIED LIVE 2026-11-03: EURUSD and USA30IDXUSD decode to real bid/ask;
  ~19 s per hour file; 4 parallel files in 33 s. The **monthly bulk .zip
  timed out at 120 s** — use per-hour, with a thread pool (max 4 workers;
  the server silently slows beyond that).
- Resample ticks to OHLCV: bar `open` = first tick price, `high`/`low` =
  extremes, `close` = last tick price, `volume` = sum of tick volumes.

### Data honesty requirements

Every loader docstring MUST state: the source, the coverage it can deliver,
the known caveats (unadjusted rolls for Yahoo, index-CFD-not-futures for
Dukascopy indices, no volume for some), and the verification date.

---

## 4. `prop_lab.validate.walkforward`

Purged walk-forward, following López de Prado's AFML ch. 7 + the
`eslazarev/purged-cross-validation` conventions.

```python
@dataclass
class WFWindow:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    n_train_trades: int
    n_test_trades: int
    test_mean_r: float
    test_win_rate: float

@dataclass
class WFResult:
    windows: list[WFWindow]
    oos_mean_r: float
    oos_win_rate: float
    oos_total_r: float
    is_mean_r: float
    degradation: float      # is_mean_r - oos_mean_r
    verdict: str             # 'holds' | 'degrades' | 'fails'
    n_windows: int
    warnings: list[str]

def walk_forward(strategy_factory, df, *, n_splits=5, train_frac=0.6,
                 purge_bars=0, embargo_bars=0,
                 direction="both", risk_points=None) -> WFResult:
```

- `strategy_factory` is a **callable** returning a fresh strategy instance
  (state must not leak between folds).
- Split the session-ordered bar index into `n_splits` contiguous test
  blocks; each test block is preceded by a training block of
  `train_frac` of the remaining history (expanding or rolling — rolling
  by default).
- `purge_bars`: drop training bars whose labels could overlap the test
  block. `embargo_bars`: drop training bars right after the test block.
- Verdict thresholds (documented constants):
  - `degradation <= 0.10` → `"holds"`
  - `0.10 < degradation <= 0.30` → `"degrades"`
  - `degradation > 0.30` → `"fails"`
  - if `oos_mean_r <= 0` → `"fails"` regardless.
- Warn when total OOS trades < 30 ("sample too small to conclude").

---

## 5. `prop_lab.firm`

### `rules.py`

```python
@dataclass(frozen=True)
class FirmRules:
    name: str
    starting_balance: float
    profit_target: float          # dollars
    max_drawdown: float           # dollars, from the trailing peak
    max_daily_loss: float         # dollars
    evaluation_days: int = 30
    min_trading_days: int = 0
    trailing_drawdown: bool = True
    static_floor: float | None = None   # None = balance - max_drawdown
    flavor: str = "EOD"           # 'EOD' | 'Intraday'
    fee: float = 0.0
    payout_ladder: tuple[float, ...] = ()
    consistency_ratio: float | None = None
    min_daily_profit: float | None = None
    max_payouts: int = 6

APEX_50K_EOD = FirmRules(...)      # 50k start, 3k target, 2k trailing DD, 1k DLL
APEX_50K_INTRADAY = FirmRules(...) # 50k start, 3k target, 2k trailing DD, 1.5k DLL
TOPSTEP_50K_EOD = FirmRules(...)
```

Rules constants must be documented with the source and the date checked.

### `monte_carlo.py`

```python
@dataclass
class ChallengeResult:
    outcome: str            # 'pass' | 'fail' | 'timeout'
    final_equity: float
    n_trades: int
    breach_kind: str | None # 'max_dd' | 'daily_loss' | 'timeout'
    equity_path: list[float]

def simulate_challenge(pnl_r_array, rules: FirmRules, *,
                       risk_geometry="frac_2pct", atr=0.0, rng=None,
                       max_trades=2000) -> ChallengeResult: ...

def monte_carlo_challenge(pnl_r_array, rules, *, risk_geometry="frac_2pct",
                          n_attempts=10000, seed=42,
                          **kw) -> MCSummary: ...

@dataclass
class MCSummary:
    n_attempts: int
    pass_rate: float
    fail_rate: float
    timeout_rate: float
    mean_final_equity: float
    p5_final_equity: float
    p95_final_equity: float
    mean_trades_to_pass: float
    pass_rate_ci_low: float      # Wilson 95%
    pass_rate_ci_high: float
    n_trades_sample: int
    warnings: list[str]
```

**THE CORE RULE — bootstrap, never coin flip.** The simulated trade outcome
must be drawn from the *empirical* per-trade `pnl_r` array:

```python
pnl_r = float(pnl_r_array[rng.integers(0, len(pnl_r_array))])
```

Never `rng.random() < win_rate`. Never `rng.choice([-1, rr])`. The real
win/loss size distribution is the whole point.

**Risk geometries** (all `risk_fn(equity, peak, atr) -> dollars`):

| name             | logic                                        |
|------------------|----------------------------------------------|
| `max_size`       | fixed $500                                   |
| `frac_2pct`      | 2% of current equity                         |
| `dd_frac_40pct`  | 40% of remaining room to the floor           |
| `anti_martingale`| base + 15% of profit above start             |
| `floor_aware`    | shrinks as equity approaches the floor       |

Each risk function MUST also respect the firm's remaining room: cap the
risk at `max_daily_loss` and at the remaining drawdown room, floor at $25.

**Breach logic:**
- Equity is marked per trade. `peak` tracks the high-water mark.
- Trailing floor = `max(static_floor, peak - max_drawdown)`.
- Daily loss: accumulate realised P&L per session; breach when the session
  loss exceeds `max_daily_loss`. In bootstrap mode there is no true session
  boundary, so trades are grouped into synthetic days of
  `trades_per_day` (default 3) — **document this approximation in the
  summary warnings**.
- Pass when `equity >= starting_balance + profit_target`.

**Sample-size honesty:** if `len(pnl_r_array) < 30`, add a warning that the
pass rate is low-confidence. If `< 10`, warn loudly.

### `funded.py`

```python
@dataclass
class FundedResult:
    total_payout: float
    n_payouts: int
    days_survived: int
    blew_up: bool
    daily_profits: list[float]

def simulate_funded(pnl_r_array, rules, *, risk_geometry="frac_2pct",
                     n_attempts=10000, seed=42) -> FundedSummary: ...
```

- Apply the payout ladder in order.
- Enforce the **consistency rule**: no single day's profit may exceed
  `consistency_ratio` (e.g. 0.50) of total profit since the last payout.
  If violated, that day is not eligible to bank a payout.
- Enforce `min_daily_profit` and `min_balance_to_payout` if set.
- Stop after `max_payouts`, on blow-up, or after the funded lifetime.

---

## 6. `prop_lab.surface` — THE HONESTY LAYER

This is the part that must not lie. Three surface kinds, each explicitly
labelled.

```python
AXIS_MEASURED = "measured"   # computed from a real backtest / real sim
AXIS_MODELLED = "modelled"   # assumed / parametric — NOT from a backtest

@dataclass
class SurfaceSpec:
    x_name: str
    x_values: list[float]
    x_provenance: str          # AXIS_MEASURED | AXIS_MODELLED
    y_name: str
    y_values: list[float]
    y_provenance: str
    z_name: str                # e.g. 'pass_rate'
    z_provenance: str
    note: str = ""

@dataclass
class Surface:
    spec: SurfaceSpec
    z: np.ndarray              # shape (len(x_values), len(y_values))
    axis_labels: dict          # human labels for each axis
    def to_dataframe(self) -> pd.DataFrame: ...
```

### Three surface builders

**`risk_geometry_surface(pnl_r_array, rules, n_attempts=...)`**
- x axis = risk geometry (5 values). y axis = firm ruleset (EOD vs
  Intraday, or a list of rulesets). z = pass rate.
- **BOTH axes MEASURED.** Each cell is a real simulated challenge over the
  real trade distribution. This is the surface that carries real
  information, and it is the one to lead with.

**`rr_surface(pnl_r_array, rules, rr_values=..., n_attempts=..., measured=True)`**
- x axis = R:R. y axis = risk geometry.
- `measured=True` → re-run the strategy's backtest at each R:R, rebuild the
  `pnl_r` array, then simulate. **This is the honest version.** Cost: one
  backtest per R:R value.
- `measured=False` → scale winners in the existing `pnl_r` array to hit the
  target R:R. Fast, but the trade *sequence* was not re-generated — label
  the x axis `AXIS_MODELLED` and set `note` accordingly.
- The report MUST state which mode produced the surface. If a measured
  surface was requested and is affordable, do measured.

**`winrate_surface(pnl_r_array, rules, win_rates=..., rr_values=...)`**
- x axis = assumed win rate. y axis = R:R. z = pass rate.
- **BOTH AXES MODELLED — ALWAYS.** You cannot "re-backtest" a win rate; you
  would be fabricating trades. This surface assumes each trade wins with
  probability `p` at `rr` and loses `-1R` otherwise.
- The SurfaceSpec MUST carry `x_provenance = y_provenance = AXIS_MODELLED`
  and the note MUST say this is a parametric surface, not a backtest
  result.
- This is the surface whose *shape* DeltaTrend Trading popularised. Say so
  in the note, with the caveat that its win-rate axis is parametric by
  construction.

### Why this design (state it in the report)

The risk-geometry axis can be measured because the trade distribution is
fixed and only the *sizing* changes — each cell is a genuine simulation
over real trades. The R:R axis can be measured, but only by re-running the
backtest, because R:R changes which trades exist. The win-rate axis
**cannot** be measured from a backtest at all: forcing a different win rate
means inventing trades. Splitting a real trade distribution into "wins"
and "losses" and re-drawing them at a different rate is a model, and must
be labelled as one.

---

## 7. `prop_lab.report`

```python
def build_report(*, wf: WFResult, mc: MCSummary, surfaces: list[Surface],
                 strategy_name: str, data_source: str,
                 data_range: str, rules: FirmRules) -> str:
```

Returns Markdown. It MUST contain, in order:

1. **Headline** — strategy, data source + range, n trades, IS vs OOS mean R,
   pass rate with 95% CI, degradation verdict.
2. **Honesty section** — a table of every surface axis with its provenance
   (`measured` / `modelled`) and a one-line note per surface. This section
   is mandatory and must not be omitted for any reason.
3. **Risk geometry table** — pass rate per geometry with CI.
4. **Out-of-sample table** — per-fold train/test trade counts and mean R.
5. **Caveats** — sample size, bootstrap day approximation, unadjusted rolls,
   data provenance, what was NOT modelled (slippage, news gaps, scaling
   plans, payout delays, firm rule changes).
6. **Provenance** — citations with URLs, and the date each source was
   checked.

No emojis. Plain prose. Numbers must be present, not described.

---

## 8. CLI

`proplab <command>`:

| command            | what it does                                                |
|--------------------|-------------------------------------------------------------|
| `fetch`            | download data to a local cache, print coverage               |
| `backtest`         | run a strategy, print trade stats                           |
| `walkforward`      | run purged walk-forward OOS, print per-fold table            |
| `montecarlo`       | run the bootstrap MC, print pass rate + CI                   |
| `surface`          | build surfaces, save PNGs                                   |
| `report`           | run everything, write a labeled Markdown report + PNGs      |
| `strategies`       | list available strategies                                   |

`report` is the one-shot path a stranger's agent runs.

---

## 9. Agent contract (AGENTS.md)

The single most important deliverable. It must tell an AI agent:

1. What this repo is and the one question it answers.
2. The exact steps to take a user's plain-language idea and produce a
   backtestable strategy (the "translation rules"):
   - Map the idea to explicit, testable conditions.
   - Every condition must be computable from the bar stream.
   - Ambiguity resolution rules (e.g. "momentum" → ROC / EMA slope /
     Donchian breakout — pick one and SAY which).
   - Stop placement rules (ATR multiple, structure-based, or fixed points).
   - R:R and direction defaults.
   - Every assumption must be written into a `notes` field on the strategy.
3. How to get data (free sources, the table of what each gives).
4. The exact command sequence to run.
5. The honesty rules — what the agent may and may not claim, and the
   mandatory "modelled vs measured" labelling.
6. The known failure modes (the ones that wasted time before):
   - look-ahead in signal → fill timing
   - NaN in a filter source silently deleting trades
   - the last open trade being dropped
   - comparing against a drifted baseline
   - unadjusted continuous-future rolls
   - small samples reported as conclusions
7. A worked example (a real strategy file + the commands).

---

## 10. Non-negotiable rules for the implementation

1. **No look-ahead.** Signal on bar `i` fills at open of bar `i+1`.
2. **Bootstrap, never coin flip**, in every MC.
3. **Every surface axis carries provenance.** No exceptions.
4. **Every number in a report traces to a computation** in the same run.
5. **Warnings are not optional.** Sample size, day approximation, roll
   adjustment — surfaced in the report, not buried in logs.
6. **Deterministic.** Every stochastic step takes an explicit `seed`.
7. **Python 3.10+**, `numpy`, `pandas`, `matplotlib` only as hard deps.
   `yfinance` optional (Yahoo source), `requests` optional (Binance/Dukascopy).
   The core package must import with stdlib+numpy+pandas+matplotlib only.
