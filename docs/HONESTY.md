# HONESTY.md

Why this package refuses to produce a number it cannot defend.

---

## The question

> "I have a trading idea. What is my probability of passing a prop firm
> evaluation?"

The tempting answer is a single percentage. The honest answer is a
percentage plus a list of everything that had to be assumed to produce
it.

This package exists because the difference matters. A prop firm
evaluation costs real money, and the gap between "this strategy passes
60 percent of the time" and "this strategy passes 60 percent of the time
*in a model that assumes X, Y and Z*" is the difference between a
decision and a coin toss.

---

## Rule 1: bootstrap the real trades, never coin-flip them

The textbook approach to Monte Carlo a prop firm challenge assumes
winners pay +2R and losers pay -1R. That assumption throws away the
actual win and loss size distribution the strategy produced, and the
result is that every strategy scores roughly the same pass rate
regardless of whether it has an edge.

This package draws the outcome of each simulated trade from the
empirical per-trade R-multiple array the backtest produced:

```python
pnl_r = float(pnl_r_array[rng.integers(0, len(pnl_r_array))])
```

If you see `rng.random() < win_rate` anywhere in a challenge
simulation, the number is meaningless.

---

## Rule 2: block bootstrap by default, because i.i.d. understates risk

There are two resampling methods in this package.

**i.i.d.** draws individual trades with replacement. It preserves the
exact marginal distribution of R multiples and destroys serial
dependence.

**Block** resamples whole trading days. It preserves the correlation
between trades inside a session.

Why it matters: losses arrive in clusters. A bad regime produces several
losing trades in a row. A trailing drawdown is a path-dependent barrier,
so whether it gets hit depends on the ORDER of trades, not just the
multiset. Breaking the clusters apart makes drawdowns look shallower
than they will be. The i.i.d. bootstrap therefore systematically
*understates* the risk, and the understatement grows with the
clustering in the original sample.

So the default is the block bootstrap, and `monte_carlo_challenge` takes
`session_dates` so it can group trades into real sessions.

Set `compare_iid=True` to run both and report the gap. When the gap is
large, the sample has meaningful day-level clustering and the i.i.d.
number is flattering.

### The stationary bootstrap

A third method is available: `resampling="stationary"`, which is the
stationary bootstrap of Politis and Romano (1994). It draws contiguous
blocks whose lengths follow a geometric distribution, so the resampled
series stays stationary, which a fixed-length moving-block bootstrap
does not.

The mean block length is a manual parameter (`mean_block_length`,
default 5 trades) and it is a real modelling choice. Too short and the
resampling drifts back towards i.i.d. Too long and there are too few
effectively independent blocks, so the simulation looks precise while
resting on very little evidence. The value used is always reported
alongside the result.

An automatic selector exists: Politis and White (2004),
"Automatic Block-Length Selection for the Dependent Bootstrap",
https://doi.org/10.1081/etc-120028836. It is NOT implemented here yet,
so `mean_block_length` is currently a manual choice.

---

## The engine is checked against published exact values

A Monte Carlo engine that nobody has validated against a known answer is
a black box that produces confident numbers. This one is checked.

The reference case is published in EdgeVeris's journal, "Estimating Your
Prop Challenge Pass Probability with Monte Carlo"
(https://edgeveris.com/blog/monte-carlo-prop-challenge-probability,
checked 2026-11-03). It is a deliberately simple model: a $100,000
account, a $10,000 profit target, a $10,000 failure boundary, $1,000
fixed risk per trade, a 55 percent win rate and a 100 trade horizon.

The answer can be computed exactly with a dynamic program instead of by
sampling. Track the probability mass at every surviving integer profit
state, move 55 percent of each state's mass one step up and 45 percent
one step down at each trade, absorb the mass reaching +10 into "passed"
and -10 into "failed", and after 100 iterations sum what is left.

The published exact values:

```
Passed      66.784 percent
Failed       8.978 percent
Unfinished  24.238 percent
```

`tests/test_golden_reference.py` reproduces the dynamic program, asserts
the published numbers, and asserts that this engine converges on them.

Writing that test found two real defects, both now fixed:

1. The drawdown breach was checked with a strict comparison
   (`equity < threshold`), so a path that landed exactly on the
   threshold survived and could step past it on the next trade. It is
   now inclusive.
2. There was no risk geometry for a fixed percentage of the INITIAL
   balance, which is the basis the reference case uses and how most
   prop firm traders actually size.

The reference also makes three points that this package follows:

- **Do not discard unfinished attempts.** Pass, fail and unfinished are
  a complete partition and sum to 1. Removing the unfinished attempts
  silently changes the question.
- **The pass rate among resolved attempts is a different number.** It is
  larger than the pass rate over all attempts, and the two must never
  be confused.
- **State the risk basis.** Fixed-cash risk and current-equity risk
  reach different account values and produce different pass rates, so
  they are separate geometries with separate names.

The block length is a modelling choice and it is a real limitation. Too
short and you drift back toward i.i.d. Too long and there are too few
independent blocks to draw from, which makes the simulation look
precise while still resting on very little evidence.

---

## Rule 3: every surface axis carries provenance

A surface plot is the most persuasive and the most dangerous object in
this whole field. A 3D pass-rate surface looks like knowledge. Whether
it IS knowledge depends entirely on what the axes mean.

This package tags every axis as one of:

- **measured** — computed from a real backtest or a real simulation.
- **modelled** — assumed. Not the output of a backtest.

| Axis | Provenance | Reason |
|------|-----------|--------|
| Risk geometry | measured | The trade distribution is fixed by the backtest and only the sizing changes. Every cell is a genuine simulation. |
| R:R | measured, only if the backtest is re-run | Changing R:R changes which trades exist. A trade that reached its target at 2R would not have reached it at 1R. Rescaling the winners of an existing trade array is a model. |
| Win rate | **always modelled** | You cannot re-backtest a win rate. Forcing a different win rate means fabricating trades. |

### The win-rate axis

This is the one people get wrong, and it is the axis that made
DeltaTrend Trading's videos famous. His surface showed pass rate as a
function of win rate and R:R under Topstep rules, and the point he was
making was that strategies with zero expected value have very different
pass rates depending on the risk geometry.

The shape is real and the insight is correct. But the win-rate axis is
parametric by construction. You cannot take a backtest and re-run it at
55 percent win rate, because that requires inventing trades that never
happened. Splitting the real distribution into wins and losses and
re-drawing them at a different rate is a model of the strategy, not a
measurement of it.

So the win-rate surface in this package is labelled `modelled` on both
axes, always, and the report says so in the Honesty section without
being asked.

### Why the risk-geometry axis is the one to lead with

Because it is genuinely measured. The trades, their order and their
sizes in R are fixed by the backtest. Only the position-sizing rule
changes from cell to cell. Every cell is a real simulation over real
trades. When you want to know "does my sizing matter, and how much",
that is the surface to trust.

---

## Rule 4: the fill model is pessimistic and the sample must be big

A signal on bar `i` is filled at the open of bar `i+1`. Filling at the
close of bar `i` is look-ahead and it inflates results.

When a bar touches both the stop and the target, this package assumes
the **stop** filled. Real intrabar order is unknowable from OHLC bars,
and the pessimistic assumption costs a little performance and buys
honesty.

Concretely: this package's engine reproduces the reference project's
73 trades exactly under the reference session rule, but at 42.5 percent
win rate against the reference's 41.1 percent, and +0.262R against
+0.233R. The difference is the pessimistic fill model. It is a small
gap, and it points in the direction that does not flatter the strategy.

Sample size rules that the report enforces:

- Under 30 trades: the pass rate is illustrative, not an estimate.
- Under 10 trades: it is a toy.
- The 95 percent confidence interval is printed next to every pass rate.
  Use the Wilson interval, not the normal approximation, because prop
  firm pass rates live near 0 and 1 where the normal approximation
  produces intervals that run outside the valid range.
- 10,000 Monte Carlo attempts is the default. At 500 attempts the
  confidence interval is about 4 percentage points wide, which cannot
  distinguish the risk geometries.

---

## Rule 5: the session rule is not a detail

A futures trading day runs 18:00 ET to 17:00 ET. Grouping bars by the
UTC calendar date splits every session in two.

On 14 weeks of MNQ 5-minute data, the UTC calendar date produces 86
"session" fragments where 71 real trading days exist. A session-
anchored strategy then fires on fragment boundaries that never existed
in the market, and the reference implementation of this exact strategy
gained 7 phantom trades that way (73 instead of 66).

Prop firms reset the daily loss limit and calculate end-of-day
drawdown at 18:00 ET. A backtest with a different boundary does not
predict the account.

This package uses the correct rule by default and reports the
difference where it matters.

---

## Rule 6: the warnings are part of the result

The engine raises warnings for small samples, degenerate bootstraps,
synthetic-day approximations, resampling gaps and low attempt counts.
They go into the report. They are not filtered to make the result look
cleaner.

Standing caveats that are always printed:

- slippage and commission are not modelled
- news gaps and intrabar order of events are not modelled
- contract scaling plans and payout delays are not modelled
- firms change their rules, so the ruleset date matters
- the bootstrap approximates session boundaries with synthetic days
- a small trade sample cannot support a confident conclusion

---

## Rule 7: the firms' rules are dated, and the date is part of the number

Every preset in `prop_lab/firm/rules.py` carries a `rules_checked` date
and a `source` URL.

Two corrections that came out of checking the official sources on
2026-11-03, both of which change results:

1. **The current APEX 50K Intraday plan has no daily loss limit.** The
   EOD plan has $1,000; the intraday plan has none. Getting this wrong
   changes every intraday pass rate.
2. **Breaching the daily loss limit does not fail the account at APEX or
   Topstep.** It ends the trading session. The account only fails if the
   trailing drawdown is touched. So `dll_is_failure` defaults to False.

Three more rules that are now enforced in the simulator, each one
checked against the firms' published terms on 2026-11-03:

3. **`min_trading_days`.** Topstep's Trading Combine needs 2 trading
   days to pass, even if the profit target was reached on day one.
   APEX has no minimum and may pass in a single day. The rule was
   missing entirely, and ignoring it inflates the pass rate for anyone
   whose strategy earns its target in a burst.

4. **The evaluation-stage consistency rule.** Topstep's best single day
   must be at or below 55 percent of the profit target. Exceeding it
   does NOT fail the account: it RAISES the target to
   `best_day / 0.55`. A $1,800 best day turns the $3,000 target into
   $3,273. The simulator applies that same adjustment, so a strategy
   that earns everything in one big day has to earn more overall.

5. **Risk basis has to be named.** Fixed-cash risk and current-equity
   risk reach different account values and therefore produce different
   pass probabilities. They are separate geometries with separate names:
   `frac_2pct` compounds with equity, `frac_2pct_initial` does not.

Note the two consistency rules are different things and live in
different fields:

- `consistency_ratio` is the FUNDED-stage rule. It gates whether a
  payout is available. Breaching it does not fail the account.
- `eval_consistency_ratio` is the EVALUATION-stage rule. It raises the
  profit target. Topstep sets it to 0.55; APEX sets it to None,
  because APEX applies consistency only at the payout stage.

Both are recorded and tested in `tests/test_rules_enforcement.py`.

Re-verify the rules before trusting them with money behind them.

---

## What "measured" does not mean either

A measured axis means the number came from a real backtest or a real
simulation over real trades. It does not mean the strategy will make
money. The following are outside what any of this can tell you:

- whether the person running it will follow the rules
- slippage on a specific fill
- what happens when the market gaps through your stop
- whether the firm changes its rules mid-evaluation
- whether the edge survives a market regime that was not in the sample

The honest framing of a pass rate is: "under the assumptions listed
above, and on the historical sample above, the simulation produced this
number".

---

## Attribution

The bootstrap-resampled challenge simulation and the risk-geometry
framing come from the prop-risk-geometry project, originally
`github.com/hunter547/prop-risk-geometry` and now
`github.com/dsilva42/prop-risk-geometry`.

The pass-rate surface visualisation that started the public conversation
about this geometry is DeltaTrend Trading's work, on TikTok and YouTube,
using Topstep rules. He has publicly pushed back on creators who copied
his surface without understanding it. The reasoning in this document is
an attempt to make the copying unnecessary.

The purged walk-forward follows Lopez de Prado, "Advances in Financial
Machine Learning", chapters 7 and 12, and the implementation conventions
of `github.com/eslazarev/purged-cross-validation`.
