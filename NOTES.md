# prop-lab — working notes

Status as of 2026-11-03. Repo: `<repo-root>/prop-lab`
Upstream GitHub repo: `<owner>/prop-lab`

## Where things stand

**Repo pushed and working.** Verified from a CLEAN CLONE: install works,
tests pass, real MNQ backtest matches the reference.

**Test counts:** 169 pass / 4 skipped offline, 172 pass with
`PROPLAB_NETWORK_TESTS=1`. The 4 offline skips are: 3 network tests
gated by the env var, and 1 baseline regression test that needs
`PROPLAB_MNQ_5M` pointing at the MNQ CSV.

**The offline suite must stay around 10 seconds.** If it takes 90+
seconds, a live network test has leaked into the offline run. That
happened once and is documented below.

- Repo: https://github.com/<owner>/prop-lab (private)
- Branch: `master`
- Baseline check passes (reference rule reproduces 73 trades exactly)
- Surfaces re-rendered and visually audited, annotation bug fixed
- Research scratch and temp test files deleted
- Older reference checkout left untouched (reference/attic)

## The lesson from 10 bugs

EVERY bug was found by RUNNING something, not by reading code:
a live endpoint, a rendered PNG, a clean clone. Three of them would
have shipped broken to every user of the repo.

The network tests were skipping the whole time. Writing them, seeing
them skip, and moving on was the mistake. Running them found two more.
**Never treat a skipped test as a passing test.**

## The one bug that would have broken every user

`.gitignore` had a bare `data/` pattern. Git matches that against any
directory named `data` at any depth, so `src/prop_lab/data/` was never
committed. It worked locally because the editable install puts `src` on
the path, and the test suite passed, and the CLI ran. A fresh clone was
broken.

The only reason it was caught is that the clone test was actually run
instead of assumed. Keep doing that.

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

7. **`.gitignore` excluded the `prop_lab.data` source package.** A bare
   `data/` pattern matches any directory named `data` at any depth, so
   `src/prop_lab/data/` was never committed. The package existed on disk
   and the editable install worked locally, but a FRESH CLONE could not
   import `prop_lab.data` and the test suite failed at collection.
   Fixed with the negation `!src/prop_lab/data/`.
   **Lesson: never trust an editable install. Always verify from a clean
   clone.** This would have shipped a broken repo to every single user.

8. **Yahoo returned 1 row instead of 1511.** The daily index arrives
   as `datetime64[s]`. The code divided by a hard-coded `10**9`, which
   collapsed all 1511 timestamps to the value `1`, and the dedup step
   then dropped 1510 rows.
   Fixed by converting to second resolution explicitly with
   `.as_unit("s")`. Note `dtype.unit` is NOT a usable attribute here:
   pandas 3.0 returns `None` for it on `datetime64[s]` and
   `datetime64[us]`, so guessing the unit always fell through to the
   nanosecond branch. `Timestamp.unit` does exist and returns 'us'.

9. **The converted time column came back as NaN.** `pd.to_datetime`
   on a Series whose VALUES are datetimes returns a frame with a RESET
   RangeIndex, because the conversion goes through a DatetimeIndex
   internally. Assigning that into a frame with a datetime index makes
   pandas align on index labels, find no match, and fill NaN.
   Fixed by restoring `parsed.index = original_index` in
   `_to_epoch_seconds`, and by assigning `.to_numpy()` at the call
   sites so the result is positional.

10. **Dukascopy was completely broken, twice over.**
    - `struct.unpack(">IIIff" * n, ...)` raises
      `struct.error: bad char in struct format`. The `>` byte-order
      prefix is only legal at the START of a format string, so the
      format must not be repeated by string multiplication.
    - After fixing that, `Struct.unpack_from` unpacked only a SINGLE
      record (5 values), so `columns[1]` raised `IndexError`. The
      correct API for fixed-size records is `Struct.iter_unpack`.

    Fixed with `struct.Struct(BI5_RECORD).iter_unpack(payload[:usable])`.
    Every real Dukascopy download would have crashed before this.

## A bug in the BUGS: my own regression tests did not guard the code

Found 2026-11-03, after the first round of fixes.

The first version of `tests/test_data_regressions.py` tested
`_to_epoch_seconds` DIRECTLY. That meant it would keep passing even
if `yahoo_src.py` went back to dividing by a hard-coded 10**9, because
the test never touched the adapter.

Proved by reintroducing the bug and re-running:

```
# reintroduce the bug in yahoo_src, then run the regression tests
3 failed, 21 passed     <- 3 Yahoo tests caught it
```

But before that, the Dukascopy tests were proved the same way:

```
# reintroduce unpack_from instead of iter_unpack
2 failed, 19 passed     <- 2 Dukascopy tests caught it
```

And the original tests did NOT catch it:

```
# reintroduce the Yahoo bug, run the OLD regression tests
2 failed, 19 passed     <- the Yahoo tests PASSED against the broken code
```

**A regression test must exercise the SHIPPED code path, not a helper
it happens to call.** Fixed by extracting the conversion into
`yahoo_src._epoch_seconds_column(index)` and testing that directly,
which is what the adapter itself calls.

**Rule for the next session: never write a regression test that calls
something other than what production calls. Prove each new test by
reintroducing the bug and confirming it fails.**

## Two more real defects found in my own test code

1. **Duplicate network tests.** The three live network tests existed in
   both `tests/test_data.py` and `tests/test_data_regressions.py`, so
   every network run hit the live endpoints twice.

2. **My new network tests were not actually skipped offline.** The old
   file gates its network tests with BOTH ``@pytest.mark.network`` and
   a ``skipif`` on ``PROPLAB_NETWORK_TESTS``. My new file only had the
   marker, so the live tests RAN during the offline suite. That is why
   the offline run took 98 seconds instead of 9.

   Fixed by removing the duplicated network tests from the new file
   and keeping them only in `tests/test_data.py`, where the gating is
   correct.

The offline suite is now back to about 10 seconds, and the network
suite is the only thing that touches the network.

## The visual audit of the six surfaces (2026-11-03)

Rendered from the real MNQ 5m data with `proplab surface`, 1500
attempts, apex-50k-eod. Files were written to the scratch dir because
`/tmp/pl_test` was pruned before the audit could run.

Each image was checked with a failure-mode-specific prompt, not a
generic "does it look right".

| image | result |
|---|---|
| `surface_risk_geometry_heatmap.png` | clean. All 5 annotations inside their cells, no margin overflow, no overlap, no clipped labels. |
| `surface_win_rate_heatmap.png` | clean placement. All 42 annotations present in a 7x6 grid, none transposed, none missing. After the contrast fix all 42 are legible. |
| `surface_rr_ratio_heatmap.png` | placement clean, all 30 annotations correct. After the contrast fix all 30 are legible (15 white on dark, 15 black on bright). |
| `surface_risk_geometry_3d.png` | clean. Z-axis title `pass_rate` fully visible, not hidden by or clipped by the colorbar. |
| `surface_rr_ratio_3d.png` | clean. Z-axis title fully visible. |
| `surface_win_rate_3d.png` | clean. Z-axis title fully visible. |

These were re-rendered after the contrast fix and re-audited; the
results above for the two heatmaps are from the second render.

The annotation transposition bug is verified fixed both by eye and
programmatically: a synthetic 3x2 surface with distinct values per
cell places `z[i, j]` at `(x=i, y=j)`, and the imshow grid has shape
`(len(y), len(x))`.

## Remaining work, NOT fixed as of 2026-11-03

Recorded here deliberately, not fixed. Do not treat this list as a
to-do to be silently skipped later; each item needs a decision.

### 1. The stationary bootstrap block length is still a manual parameter

`mean_block_length` defaults to 5 trades and the user has to pick it.

The principled fix is the Politis and White (2004) automatic selector.
The algorithm was researched and a port was drafted
(https://github.com/InvestmentSystems/recombinator,
recombinator/optimal_block_length.py, which is Patton's Matlab code
adapted by Michael C. Nowotny; original at
https://public.econ.duke.edu/~ap172/opt_block_length_REV_dec07.txt).
It was NOT committed because it was unwired and untested at the time
this note was written, and shipping untested numerics is exactly the
kind of thing this project is supposed to avoid.

To do: port it into `src/prop_lab/firm/block_length.py`, wire
`resampling="stationary"` to use it when `mean_block_length="auto"`,
add tests against Patton's published behaviour, and document that the
Politis-White length targets the long-run variance of the series, not
a path-dependent barrier, so it is a principled default rather than a
proven optimum.

### 2. The Politis-White selector targets LINEAR dependence

Real trade series show little linear autocorrelation in the R values
while showing strong dependence in volatility and loss clustering. A
selector that sees little linear structure can pick a short block even
when clustering is pronounced. Compare several defensible lengths
rather than trusting one.

### 3. More firm presets are not encoded

Rules were read during research but never encoded: MyFundedFutures
(Builder 1 day, Rapid 2 days, Rapid EOD 4 days, 50 percent
consistency), Tradeify Select (3 days, 40 percent consistency),
Alpha Futures (Zero 1 day, Standard 2 days, Advanced 3 days), Lucid
Trading, and the Apex Legacy 7-day plan. Sources:
https://makeitsweep.com/prop-firm-minimum-days
https://www.bestpropfirms.com/education/evaluations/
Every preset must carry its source and the date it was checked.

### 4. `docs/DATA.md` does not mention the stationary bootstrap

`docs/HONESTY.md` covers it, the golden test, `min_trading_days` and
`eval_consistency_ratio`. `docs/DATA.md` was not updated.

### 5. `docs/images/` is referenced in `.gitignore` but does not exist

If example surface images are wanted, generate them into that folder
so they are not gitignored.

### 6. The `rr_surface` measured mode needs a documented requirement

It requires a strategy with an overridable `rr_ratio`. Both shipped
examples have one, but the requirement is not prominent enough.

### 7. Heatmap annotations were unreadable on dark cells — FIXED 2026-11-03

Found 2026-11-03 during the visual audit of the six rendered surfaces,
and fixed the same day.

The heatmap annotations were hard-coded to `color="black"`, but the
heatmap uses the `plasma` colormap, whose dark end is a deep
blue-purple. Measured contrast of black text against the plasma ramp:

```
colormap t=0.0  rgb=(0.05,0.03,0.53)  luminance=0.070  contrast=1.1:1  INVISIBLE
colormap t=0.2  rgb=(0.42,0.00,0.66)  luminance=0.137  contrast=1.3:1  INVISIBLE
colormap t=0.3  rgb=(0.56,0.05,0.64)  luminance=0.203  contrast=1.7:1  INVISIBLE
colormap t=0.5  rgb=(0.80,0.28,0.47)  luminance=0.404  contrast=3.7:1  poor
colormap t=0.8  rgb=(0.99,0.65,0.21)  luminance=0.692  contrast=9.7:1  ok
```

44.6 percent of the plasma colormap has luminance below 0.35, which is
where black text stops being readable. WCAG AA for body text wants
4.5:1.

**The fix:** `Surface._annotation_colour(normalised)` picks white or
black from the WCAG relative luminance of the actual cell the text
sits on, choosing whichever gives the higher contrast. The caller
normalises the cell against the grid min and max, which is the same
normalisation matplotlib applies to the colormap, so the text colour
and the cell colour always agree about which end of the ramp a cell
is at.

**Verification:**
- worst contrast over the ENTIRE plasma ramp is now 4.59:1, so every
  cell in every heatmap passes WCAG AA (was 1.1:1)
- `test_heatmap_annotation_colour_is_readable_against_its_cell`
  recomputes the contrast for every annotation and asserts 4.5:1
- proved by reintroducing the bug: the new test fails with
  `contrast 1.40:1` at cell (0, 0)
- re-rendered and audited by eye: all 30 annotations on
  `surface_rr_ratio_heatmap.png` legible (15 white on dark, 15 black on
  bright), all 42 on `surface_win_rate_heatmap.png` legible, none
  transposed or overlapping

### 8. Known limitations of the Monte Carlo, not bugs

- The daily loss limit is applied to SYNTHETIC days of a configurable
  number of trades, because a bootstrap has no calendar.
- Slippage, commission, news gaps, intrabar order of events, contract
  scaling plans and payout delays are not modelled.
- The bootstrap draws more i.i.d. trades when the resampled sequence
  runs out, which slightly dilutes the dependence preservation at the
  tail of a long path.

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

Paths below are relative to the repository root. The MNQ 5m dataset is not
part of this repository; point `PROPLAB_MNQ_5M` at your own copy.

```bash
.venv/bin/python -m pytest -q
PROPLAB_MNQ_5M=<path-to>/MNQ_5m.csv \
  .venv/bin/python scripts/baseline_check.py
.venv/bin/proplab report \
  --strategy examples/strategies/trend_continuation.py:TrendContinuation \
  --data <path-to>/MNQ_5m.csv \
  --firm apex-50k-eod --out reports/
```
