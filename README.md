# prop-lab

**An agent-driven prop firm challenge lab.** Give it a trading idea in plain
language. It turns the idea into a backtestable strategy, tests it on real
free market data, checks whether the edge survives out of sample, and tells
you the probability of passing a prop firm evaluation — with every assumption
labelled.

```bash
git clone https://github.com/<owner>/prop-lab.git
cd prop-lab
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Requires Python 3.10+. Core dependencies: numpy, pandas, matplotlib.

---

## Inspiration

**This project exists because of [DeltaTrend Trading](https://www.youtube.com/@deltatrendtrading)
(Thomas Skinner, also on [TikTok](https://www.tiktok.com/@deltatrendtrading)
and [Instagram](https://www.instagram.com/deltatrendtrading)).**

His pass-rate surface — pass rate as a function of win rate and
reward-to-risk ratio — is the public conversation this lab was built to
join. He popularised it on TikTok and YouTube using Topstep rules, and he
has publicly pushed back on creators who copied the surface without
understanding what the axes mean.

`prop-lab` takes that idea and does what a surface cannot: it measures it.
Where DeltaTrend's surface is a picture of a hypothesis, this package
re-runs the strategy at every R:R, bootstraps the real per-trade
distribution, tests the edge for statistical significance, and prints the
zero-edge baseline next to your number so a pass rate can be read as a
comparison rather than a score.

The credit goes to him for the idea. The implementation is independent, and
the difference in approach is stated everywhere it matters: this package
labels every axis `measured` or `modelled`, because the win-rate axis
**cannot** be measured from a backtest — forcing a different win rate means
fabricating trades that never happened.

---

## Start here: idea to number in one command

```bash
proplab fetch --source binance --symbol BTCUSDT --interval 15m --start 2023-01-01 --out data/
proplab report --strategy examples/strategies/trend_continuation.py:TrendContinuation \
    --data data/BTCUSDT_15m.csv --firm apex-50k-eod --out reports/
```

That downloads free data, backtests, validates out of sample, simulates the
challenge 10,000 times, renders the surfaces and writes a Markdown report.

## The one thing to understand before you trust a number

**A pass rate is a comparison, not a score.**

A challenge is not a coin toss. The profit target sits *above* your starting
balance and the trailing drawdown sits *below* it, so a random walk that
happens to run up early can reach the target before it ever retraces enough
to breach the floor. For the APEX 50K EOD rules, a strategy with **no edge
at all** scores around **27%**.

Every report therefore states the zero-edge baseline next to your number.
A strategy scoring 27% has demonstrated nothing. A strategy scoring 40% has
demonstrated something.

And a pass rate is not an edge either. The report also gives a confidence
interval on the mean R per trade, and the verdict says `no_edge` when that
interval includes zero. A narrow pass-rate interval is a precise
measurement of a possibly-worthless strategy.

## How to read the verdict

| verdict | meaning |
|---|---|
| `holds` | The edge survived out of sample, and it is statistically distinguishable from zero. |
| `degrades` | There is an edge, but it shrank out of sample. Part of the in-sample result does not generalise. |
| `no_edge` | No edge was detected. The mean R's confidence interval includes zero. Nothing was there to survive. |
| `fails` | The out-of-sample mean R is zero or negative, or the edge degraded badly. |

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
        # ... your condition ...
        return Signal(
            direction="long",        # 'long' or 'short'
            stop_distance=12.5,      # price points, MUST be > 0
            rr_ratio=self.rr_ratio,  # target = stop_distance * rr_ratio
            reason="my_long_setup",  # short machine-readable tag
            bar_index=bar_index,
            timestamp=bar.timestamp,
        )
```

A signal on bar `i` is filled at the **open of bar `i+1`**. If a bar
touches both the stop and the target, the stop is assumed to have filled
first — the pessimistic assumption.

Write every assumption you made into a `notes` string on the class. The
report prints it, so the reader knows what was actually tested.

`AGENTS.md` is the full contract: how to resolve vague language, where to
get data, the command sequence, the honesty rules, and the failure modes
that have cost time in this project.

## Commands

| command | what it does |
|---|---|
| `proplab fetch` | download free market data |
| `proplab backtest` | run a backtest and print the trade statistics |
| `proplab walkforward` | purged out-of-sample validation with a verdict |
| `proplab montecarlo` | bootstrap the challenge 10,000 times |
| `proplab surface` | render the pass-rate surfaces as PNGs |
| `proplab report` | all of the above, plus a labelled Markdown report |
| `proplab inspect` | show a strategy's parameters and its assumptions |
| `proplab trades` | export the trade list so you can audit the result |
| `proplab sweep` | sweep strategy parameters; every cell is a real backtest |
| `proplab strategies` | list the built-in examples, firm presets and geometries |

## Seeing how parameters fit the data

```bash
proplab sweep --strategy my_strategy.py:MyStrategy --data data/BTCUSDT_15m.csv \
    --param body_ratio=0.05,0.1,0.2 --param stop_atr_mult=2.0,3.0
```

Every combination is backtested for real and simulated for real. The output
gives the trade count, win rate, mean R, the edge's confidence interval and
the pass rate for each cell, and says how many cells have an edge that is
distinguishable from zero.

That last number is the one to read. Six cells and "0 of 6 have an edge"
is a complete answer, and a more useful one than any pass rate.

## Data

Free, no API key, no account:

| source | what you get |
|---|---|
| **Binance** | crypto spot, full history from listing, 1m to 1d, 24/7 |
| **Yahoo Finance** | decades of daily futures bars (`NQ=F`, `MNQ=F`, `ES=F`) |
| **Dukascopy** | tick-level FX and index history |
| **Any CSV** | drop it in and load it |

One rule decides more results than any other: **a futures trading day runs
18:00 ET to 17:00 ET.** Grouping bars by UTC calendar date splits every
session in two. For 24/7 crypto the session is the UTC calendar day
(`--session-open-hour 0`, which the CLI infers automatically for Binance).

Yahoo's continuous futures are **unadjusted**: they switch contract at
expiry and leave a roughly 1 percent gap that is a fake price move. The
loader warns about it. Do not hide the warning.

## The honesty rules

These are enforced by the code, not by good intentions.

**Bootstrap, never coin flip.** Every simulated trade outcome is drawn from
the real per-trade R-multiple array. If you see `rng.random() < win_rate`
in a challenge simulation, the number is meaningless.

**Block bootstrap by default.** Losses arrive in clusters and a trailing
drawdown is a path-dependent barrier, so the order of trades matters. An
i.i.d. bootstrap breaks the clusters apart and flatters the result. Pass
`session_dates` and use `compare_iid=True` to see the gap.

**Every surface axis is labelled `measured` or `modelled`.** The
risk-geometry axis is measured: the trades are fixed and only the sizing
changes, so every cell is a real simulation. The R:R axis is measured only
if the backtest is re-run. The win-rate axis is **always modelled**,
because forcing a different win rate means fabricating trades that never
happened. That surface is the one DeltaTrend Trading popularised on TikTok
and YouTube using Topstep rules, and this package says out loud that its
win-rate axis is parametric by construction.

**The sample size is printed.** Under 30 trades the pass rate is
illustrative, not an estimate. Every pass rate comes with its 95 percent
confidence interval, computed with the Wilson interval.

**The edge is tested for significance.** The report gives a bootstrap
confidence interval on the mean R per trade, so a strategy that is
statistically indistinguishable from a random sequence is told so instead
of being given a reassuring verdict.

**The warnings are in the report.** Small samples, degenerate bootstraps,
synthetic-day approximations, resampling gaps. They are not filtered out to
make the result look cleaner.

**The firm rules are dated.** Every preset records when it was checked
against the firm's published terms. Re-verify them.

See `docs/HONESTY.md` for the full argument.

## What is not modelled

State it when it matters. Slippage and commission. News gaps and intrabar
order of events. Contract scaling plans and payout delays. The firms
changing their rules. The correlation between the strategy and the person
running it.

And the one that matters most here: **the drawdown is checked on closed
trade equity, not intrabar.** The simulation sees one equity number per
trade, so an account that dips below the trailing floor mid-trade and
recovers into the close is scored as surviving. Real firms check the floor
continuously against equity including unrealised profit. **The pass rate
here is an upper bound, not a central estimate.**

## Attribution

The bootstrap-resampling approach and the risk-geometry framing are
credited to the [prop-risk-geometry](https://github.com/dsilva42/prop-risk-geometry)
project, originally published as `hunter547/prop-risk-geometry` under a
different owner (that older URL no longer resolves). It carries no license
file, so it is cited as the origin of the idea and **no code was taken from
it** — this package is an independent implementation.

**The inspiration for this entire project is
[DeltaTrend Trading](https://www.youtube.com/@deltatrendtrading)
(Thomas Skinner)**, whose pass-rate surface — pass rate against win rate
and R:R — started the public conversation this lab joins. He popularised it
on TikTok and YouTube using Topstep rules, and has publicly pushed back on
creators who copied the surface without understanding what the axes mean.
This package is built on that idea and credits it explicitly; the surfaces
here are the same subject, measured rather than asserted. See
[the Inspiration section](#inspiration) at the top of this file.

The purged walk-forward follows Lopez de Prado, "Advances in Financial
Machine Learning", chapters 7 and 12, and the conventions of
`github.com/eslazarev/purged-cross-validation`.

The Monte Carlo engine is checked against an exact dynamic-program
reference published at
[edgeveris.com](https://edgeveris.com/blog/monte-carlo-prop-challenge-probability).

The full prior-art review, with every license verified, is in
[`docs/PRIOR-ART.md`](docs/PRIOR-ART.md).

Cite these when you reuse the methods.

## Not financial advice

This is research tooling. It produces a number under a stated set of
assumptions. It does not predict the future, and the assumptions are
listed in every report so you can judge them yourself.

MIT licensed. See [`LICENSE`](LICENSE).
