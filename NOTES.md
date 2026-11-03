# prop-lab — working notes

Status as of 2026-11-03. Repo: `<repo-root>/prop-lab`
Upstream GitHub repo: `<owner>/prop-lab`

## Where things stand

**DONE AND PUSHED.** Commit `1deb68b` on `master`, 33 files, verified via
the GitHub API. 143 tests pass, 4 skipped.

- Repo: https://github.com/<owner>/prop-lab (private)
- Baseline check passes (reference rule reproduces 73 trades exactly)
- Surfaces re-rendered and visually audited, annotation bug fixed
- Research scratch and temp test files deleted
- Older reference checkout left untouched (reference/attic)

## What is built

```
SPEC.md                    package contract
README.md                  human-facing
AGENTS.md                  agent contract (was AGENT_GUIDE.md, renamed + deleted)
docs/DATA.md               every free source, verified live
docs/HONESTY.md            the measured vs modelled argument
src/prop_lab/core.py       backtest engine
src/prop_lab/data/         schema, binance, yahoo, dukascopy, CSV
src/prop_lab/validate/     purged walk-forward
src/prop_lab/firm/         rules.py, monte_carlo.py, funded.py
src/prop_lab/surface/      surfaces.py (measured/modelled axes)
src/prop_lab/report/       report.py
src/prop_lab/cli.py        proplab CLI
tests/                     143 tests
scripts/baseline_check.py  reference validation
examples/strategies/       trend_continuation.py, opening_range_breakout.py
```

## Bugs found by actually looking at output (not by reading code)

These are the ones that would have shipped broken:

1. **pandas 3.0 timestamp bug.** `// 10**9` gave 1767225 instead of
   1767225600 because pandas 3.0 defaults to microsecond resolution.
   Fixed by reading `dtype.unit` and dividing by the right factor.
   `src/prop_lab/data/schema.py` `_to_epoch_seconds`.

2. **Drawdown breach used a strict comparison.** `equity < threshold`
   let a path survive sitting exactly on the threshold, then step past
   it next trade. Now `equity <= threshold`.
   `src/prop_lab/firm/monte_carlo.py`.

3. **Heatmap annotations were transposed.** `z` is
   `(len(x_values), len(y_values))`, the image is drawn from `z.T`, so
   `z[i,j]` belongs at `(x=i, y=j)`. It was annotated at `(j, i)`, so
   with a single ruleset row the numbers landed OUTSIDE the image and
   floated in the blank margin above the chart. Caught by
   `vision_analyze` on the rendered PNG, not by the test suite.
   Fixed in `src/prop_lab/surface/surfaces.py` `to_heatmap`.

4. **APEX consistency rule was applied to the evaluation.** APEX has
   no consistency rule on the evaluation, only at the payout stage.
   One field `consistency_ratio` was doing two different jobs, so a
   new field `eval_consistency_ratio` was added.
   `src/prop_lab/firm/rules.py`.

5. **`min_trading_days` was never enforced.** Topstep requires 2 days.
   Now enforced in `simulate_challenge`, and the Topstep preset was
   corrected from 0 to 2.

6. **CLI printed `TC {}`** because it built the strategy label with an
   empty kwargs dict. Fixed in `src/prop_lab/cli.py`.

## The session-date finding

The reference project grouped bars by the **UTC calendar date**. For
futures that splits every 18:00-17:00 ET session in two. On 14 weeks
of MNQ 5m data that is 86 fragments where 71 real trading days exist,
and it produced 7 phantom trades.

`scripts/baseline_check.py` prints BOTH:
- reference session rule: 73 trades (matches reference exactly)
- correct ET session rule: 66 trades

The engine uses the correct rule by default. The reference numbers are
a comparison point, not a target.

## Research sources (all checked 2026-11-03)

**Golden test with exact numbers:**
EdgeVeris, "Estimating Your Prop Challenge Pass Probability with Monte
Carlo" — https://edgeveris.com/blog/monte-carlo-prop-challenge-probability
$100K account, $10K target, $10K boundary, $1,000 fixed risk, 55% win
rate, 100 trades. Exact DP answer: pass 66.784%, fail 8.978%,
unfinished 24.238%. `tests/test_golden_reference.py` reproduces the DP,
asserts the published numbers, and asserts the engine converges on them.
**Writing this test found bugs 2 and the missing initial-balance risk
geometry.**

**Bootstrap method:**
Politis and Romano (1994) stationary bootstrap — implemented as
`bootstrap_stationary`, geometric block lengths.
Block length selection: Politis and White (2004),
https://doi.org/10.1081/etc-120028836 — Patton's Matlab implementation
at https://public.econ.duke.edu/~ap172/code.html was read but the
selector is NOT yet implemented. `mean_block_length` is a manual
parameter, default 5.

**Prop firm rules:**
- https://tradedeskhq.io/guides/apex-vs-topstep-futures-50k-rules/
- https://www.bestpropfirms.com/education/evaluations/
- https://tradingfunder.com/compare/topstep-vs-apex/
- https://makeitsweep.com/prop-firm-minimum-days

Confirmed for 50K: APEX and Topstep both $3,000 target, $2,000 trailing
drawdown. APEX EOD has $1,000 DLL, Intraday has none. APEX no minimum
days, may pass in 1 day. Topstep minimum 2 days. Topstep consistency
55% RAISES the target. Breaching the DLL pauses trading, does not fail
the account. APEX 50% consistency only at payout stage.

**GitHub repos read:**
dsilva42/prop-risk-geometry (moved from hunter547/prop-risk-geometry),
gabrielee5/prop-firm-simulator, lhenriet-bit/prop-odds, Makeph/prop-ev,
eslazarev/purged-cross-validation, LuxAlgo/prop-firm-sim.

**Data sources verified live:** Binance klines API, Yahoo `NQ=F` daily
(1720 rows from 2020), Dukascopy BI5 (~19s per hour file, 429 on push).

## The honesty contract (do not weaken this)

- Every surface axis is labelled `measured` or `modelled`.
- Risk geometry axis: measured. Trades fixed, only sizing changes.
- R:R axis: measured ONLY if the backtest is re-run. Rescaling winners
  is modelled.
- Win rate axis: ALWAYS modelled. Forcing a win rate fabricates trades.
- Monte Carlo bootstraps the real per-trade `pnl_r` array. Never
  `rng.random() < win_rate`.
- Block bootstrap is primary; i.i.d. is a fallback and a diagnostic.
- Pass / fail / unfinished are a complete partition and sum to 1.
- Risk basis must be named: `frac_2pct` (compounding) vs
  `frac_2pct_initial` (fixed cash) are different geometries.
- Every ruleset carries `rules_checked` and `source`.
- DeltaTrend Trading gets credit for the pass-rate surface concept.

## Remaining work

1. Implement the Politis-White automatic block-length selector so
   `mean_block_length` is not a manual choice.
2. `docs/DATA.md` does not yet mention the stationary bootstrap. The
   `docs/HONESTY.md` update already covers it, the golden test,
   `min_trading_days` and `eval_consistency_ratio`.
3. Consider more firm presets (MyFundedFutures, Tradeify, Alpha Futures
   rules were read during research and are not encoded yet).
4. The `rr_surface` measured mode needs a strategy with an overridable
   `rr_ratio`; the two shipped examples both have one, but it is worth
   documenting the requirement more prominently.
5. `docs/images/` is referenced in `.gitignore` but does not exist yet.
   If example surface images are wanted, generate them into that folder
   so they are not gitignored.

## Local commands

```bash
cd <repo-root>/prop-lab
.venv/bin/python -m pytest -q
PROPLAB_MNQ_5M=<path-to>/nq-apex-strategy/data/MNQ_5m.csv \
  .venv/bin/python scripts/baseline_check.py
.venv/bin/proplab report \
  --strategy examples/strategies/trend_continuation.py:TrendContinuation \
  --data <path-to>/nq-apex-strategy/data/MNQ_5m.csv \
  --firm apex-50k-eod --out reports/
```
