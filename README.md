# prop-lab

An agent-driven prop firm challenge lab.

Give an AI agent this repository and a trading idea in plain language.
It turns the idea into a backtestable strategy, tests it on real free
market data, checks whether the edge survives out of sample, and tells
you the probability of passing a prop firm evaluation — with every
assumption labelled.

```bash
git clone https://github.com/<owner>/prop-lab.git
cd prop-lab
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e ".[data,dev]"
.venv/bin/python -m pytest -q
```

Requires Python 3.10+. Core dependencies: numpy, pandas, matplotlib.

---

## What it does

1. **Data.** Free market data in one canonical schema, from Binance,
   Yahoo Finance, Dukascopy, or any CSV you drop in.
2. **Backtest.** You write a strategy; the engine handles fills, stops,
   targets and session handling. One fill model, one source of truth.
3. **Out of sample.** Purged walk-forward validation with an explicit
   verdict: `holds`, `degrades` or `fails`.
4. **Monte Carlo.** Thousands of challenge attempts, bootstrapped from
   the real per-trade distribution, never from a coin flip.
5. **Surfaces.** Pass-rate surfaces where every axis is marked
   `measured` or `modelled`.
6. **Report.** A Markdown report with a mandatory honesty section.

---

## Quick start

```bash
# backtest the reference strategy
proplab backtest \
  --strategy examples/strategies/trend_continuation.py:TrendContinuation \
  --data data/MNQ_5m.csv

# check whether the edge survives out of sample
proplab walkforward \
  --strategy examples/strategies/trend_continuation.py:TrendContinuation \
  --data data/MNQ_5m.csv

# estimate the pass rate
proplab montecarlo \
  --strategy examples/strategies/trend_continuation.py:TrendContinuation \
  --data data/MNQ_5m.csv --firm apex-50k-eod

# everything, plus the labelled surfaces
proplab report \
  --strategy examples/strategies/trend_continuation.py:TrendContinuation \
  --data data/MNQ_5m.csv --firm apex-50k-eod --out reports/
```

`proplab strategies` lists the built-in examples, firm presets and risk
geometries.

---

## Writing a strategy

Subclass `StrategyCore`, implement `on_bar`, return a `Signal`:

```python
from prop_lab.core import Bar, Signal, StrategyCore

class MyStrategy(StrategyCore):
    name = "my_strategy"

    def __init__(self, window: int = 20, rr_ratio: float = 2.0, **kwargs):
        super().__init__(window=window, rr_ratio=rr_ratio)
        self.window = window
        self.rr_ratio = rr_ratio

    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        ...
        return Signal(
            direction="long",
            stop_distance=12.5,
            rr_ratio=self.rr_ratio,
            reason="my_long_setup",
            bar_index=bar_index,
            timestamp=bar.timestamp,
        )
```

A signal on bar `i` is filled at the **open of bar `i+1`**.

`AGENTS.md` is the full contract for an agent: how to translate
vague language into testable conditions, how to get free data, the
command sequence, the honesty rules, and the failure modes that have
cost time in this project.

---

## The honesty rules

This is the part that matters, and it is enforced by the code rather
than by good intentions.

**Bootstrap, never coin flip.** The outcome of every simulated trade is
drawn from the real per-trade R-multiple array. If you see
`rng.random() < win_rate` in a challenge simulation, the number is
meaningless.

**Block bootstrap by default.** Losses arrive in clusters, and a
trailing drawdown is a path-dependent barrier, so the order of trades
matters. An i.i.d. bootstrap breaks the clusters apart and flatters the
result. Pass `session_dates` and use `compare_iid=True` to see the gap.

**Every surface axis is labelled.** The risk-geometry axis is measured:
the trades are fixed and only the sizing changes, so every cell is a
real simulation. The R:R axis is measured only if the backtest is
re-run. The win-rate axis is **always modelled**, because forcing a
different win rate means fabricating trades. The win-rate surface is
the one whose shape DeltaTrend Trading popularised on TikTok and
YouTube using Topstep rules, and this package says out loud that its
win-rate axis is parametric by construction.

**The sample size is printed.** Under 30 trades the pass rate is
illustrative, not an estimate. Every pass rate comes with its 95
percent confidence interval, computed with the Wilson interval.

**The warnings are in the report.** Small samples, degenerate
bootstraps, synthetic-day approximations, resampling gaps. They are not
filtered out to make the result look cleaner.

**The firm rules are dated.** Every preset records when it was checked
against the firm's published terms. Re-verify them.

See `docs/HONESTY.md` for the full argument.

---

## Data

Four routes, all free: Binance (full crypto history, no key), Yahoo
Finance (decades of daily futures bars), Dukascopy (tick-level history),
or any CSV you drop in. See `docs/DATA.md` for what each one actually
gives you, verified live on 2026-11-03, including the caveats that
matter: Yahoo's continuous futures are unadjusted and leave a fake 1
percent gap at each roll.

One rule decides more results than any other: a futures trading day runs
18:00 ET to 17:00 ET. Grouping bars by UTC calendar date splits every
session in two.

---

## Attribution

The bootstrap-resampled challenge simulation and the risk-geometry
framing come from the prop-risk-geometry project, originally
`github.com/hunter547/prop-risk-geometry` and now
`github.com/dsilva42/prop-risk-geometry`.

The pass-rate surface visualisation that started the public conversation
about this geometry is DeltaTrend Trading's work on TikTok and YouTube,
using Topstep rules. He has publicly pushed back on creators who copied
his surface without understanding what the axes mean.

The purged walk-forward follows Lopez de Prado, "Advances in Financial
Machine Learning", chapters 7 and 12, and the conventions of
`github.com/eslazarev/purged-cross-validation`.

Cite these when you reuse the methods.

---

## Not financial advice

This is research tooling. It produces a number under a stated set of
assumptions. It does not predict the future, and the assumptions are
listed in every report so you can judge them yourself.

MIT licensed.
