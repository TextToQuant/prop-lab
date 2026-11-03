# Goal prompt — prop-lab public release

> Copy everything under "THE PROMPT" into an agent session to drive the work.
> The rest of this file is the reasoning behind it and the defect ledger.

---

## THE PROMPT

You are working on `prop-lab`, a Python package that answers one question for a
retail trader:

> "I have a trading idea. If I backtest it on real free data and size it
> against a prop firm's rules, what is my actual probability of passing the
> evaluation, and does the edge survive out of sample?"

**Your goal: make it as adaptable and useful in a chat-like harness or AI tool
as it is for a human at a terminal — with no errors, and without unnecessary
or annoying changes in the process.**

### What "useful in a chat harness" concretely means

The package will be driven by an AI agent that has never seen it before. It
reads a README, writes one strategy file, runs some commands, and reports the
result to a human. Every one of those steps must work on the first try.

1. **A newcomer can go from idea to result in one command.** No manual CSV
   wrangling, no session-rule guesswork, no "you must call the Python API
   instead of the CLI" traps.
2. **The numbers on screen are the numbers in the report.** If the CLI prints
   a pass rate and the report prints a different one, that is a bug. Today it
   is, and an agent will not notice.
3. **Nothing in the output is silently wrong.** A tool that confidently prints
   a misleading number is worse than a tool that refuses to print one.
4. **The tool tells the agent when a result is not trustworthy**, in the same
   place the agent is already looking.
5. **No step needs the agent to guess a parameter that changes the answer.**
   Every parameter that materially changes the result must be reachable from
   the command line, with a sane default and a documented meaning.
6. **Failure is loud and specific.** An error must name the file, the line, the
   parameter and what to do about it. Not a stack trace, not a silent
   fallback.

### Non-negotiable: no unnecessary changes

This is a working, tested package with a passing suite. Change only what is
wrong. Specifically:

- **Do not rename public API.** `prop_lab.core`, `prop_lab.firm`,
  `prop_lab.surface`, `prop_lab.validate`, `prop_lab.report`, the `proplab`
  CLI, the `StrategyCore` / `Signal` / `Bar` contract, and the
  `measured` / `modelled` axis vocabulary are the product. They stay.
- **Do not change the fill model, the session rule semantics, the bootstrap
  method, or the verdict vocabulary** (`holds` / `degrades` / `fails`). They
  are the method. Only their *application* may be fixed.
- **Do not add dependencies** beyond what is already there, unless the
  alternative is genuinely worse.
- **Do not remove warnings.** Add them if they are missing; the honesty
  contract says warnings are never filtered out.
- **Every fix ships with a test that fails before it and passes after.**
  Follow the rule already written in NOTES.md: *a regression test must
  exercise the shipped code path, not a helper it happens to call. Prove each
  new test by reintroducing the bug and confirming it fails.*
- **Prove it end to end.** After the changes, the offline suite must still
  pass in about ten seconds, and a clean clone must install and run.

### Work in this order

1. **Fix the wrong numbers.** Defects that make the tool display a number
   that is not the truth about the user's strategy. These come first, because
   they are the ones that mislead.
2. **Fix the traps.** Places where an agent following the documentation
   silently gets a different result than intended.
3. **Close the gaps that force manual work.** Anything that makes the agent
   write Python instead of run a command.
4. **Make the release public-grade.** README for a stranger, license,
   contributing guide, CI, and a repo that does not leak its author's identity.
5. **Write down what changed and why**, so the next person does not have to
   rediscover it.

### Definition of done

- A new user, given only the README and one idea, produces a defensible
  number without asking a question and without hand-editing a file.
- Every number the CLI prints matches the report.
- No output claims more than the data supports.
- The test suite passes offline in about ten seconds and from a clean clone.
- The repository can be published under MIT with no personal information in
  the working tree, the history, or the build metadata.

---

## Why this framing

The package already does the hard part. The backtest fill model is
conservative, the bootstrap is honest, the surfaces are labelled, the report
has a mandatory honesty section, and the test suite is fast and thorough.

What it does not yet do is survive being driven by an agent. The defects
below were all found by *using* the package the way an agent would, not by
reading its source. That is the evidence that the goal is right.

---

## Defect ledger

Status: `open` / `fixed` / `wontfix`. Every entry names the file and line
where it was found, and the observable wrong behaviour.

### Status

| id | status | note |
|---|---|---|
| A1 | fixed | the modelled R:R surface is no longer used when the strategy allows a measured one |
| A2 | fixed | the CLI passes `strategy_factory` and `df`; the surface is re-backtested at every R:R |
| A3 | fixed | `session_dates` threaded through all three surface builders |
| A4 | fixed | the verdict vocabulary gains `no_edge`; `holds` requires a significant edge |
| A5 | fixed | `prop_lab.validate.test_edge`; the report headline states the edge interval |
| A6 | fixed | `--session-open-hour`, inferred from the data source |
| A7 | fixed | the daily-loss warning reports the run's actual `trades_per_day`, not the default |
| B1 | fixed | `--purge-bars` and `--embargo-bars` |
| B2 | fixed | surfaces carry an operating point and mark it, including off-grid |
| B3 | fixed | the modelled win-rate surface is anchored to the strategy's real win rate and R:R, and the report says the value there is not the measured result |
| C1 | fixed | `proplab sweep` |
| C2 | fixed | `proplab trades` |
| C3 | fixed | `proplab inspect` |
| D1 | fixed | personal information removed from the tree, the history, the object database, the reflog and the remote |
| D2 | fixed | CI on three operating systems and four Python versions |
| D3 | fixed | `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, issue and PR templates |
| D4 | fixed | the README is rewritten for a beginner, with the zero-edge baseline explained |
| D5 | fixed | PEP 621 metadata, `license = { file = ... }`, classifiers, keywords, project URLs |
| D6 | fixed | the CI has a step that fails the build if the offline suite stops being fast |

### Found during the work, not in the original list

**E1. `bootstrap_stationary` did not resample at all.**
`src/prop_lab/firm/monte_carlo.py`. The function walked the source array
cyclically from a single random start, so with `n_trades == len(source)`
the output was a **permutation of the source**: the resampled mean was
pinned to the source mean exactly, with **zero** sampling variability
(measured 0.000000 against 0.063277 for the i.i.d. bootstrap). A bootstrap
whose mean never varies cannot estimate the uncertainty of that mean. It
was not the Politis and Romano stationary bootstrap, which redraws the
start index for every block. **Fixed**, and two property tests now guard
it: block length 1 reduces to the plain bootstrap, and the block bootstrap
is wider than the i.i.d. bootstrap on a correlated series. The default
`resampling='block'` path was never affected.

**E2. A pass rate with no comparison point is a score, and it should not
be read as one.** A strategy with *exactly* zero edge scores 25-27% under
the APEX 50K EOD rules, because the profit target sits above the starting
balance and the trailing drawdown sits below it, so a random walk that
runs up early can reach the target before it breaches the floor. This was
verified against an independent random-walk implementation written from
the rules rather than from the package: 27.5% and 25.2% for exactly
zero-edge, agreeing with the package's Monte Carlo within 0.5pp. The
**zero-edge baseline** is now computed and printed next to every pass
rate.

**E3. The drawdown was checked on closed trade equity, not intrabar.**
`src/prop_lab/firm/monte_carlo.py`. The simulation sees one equity number
per trade, so an account that dips below the trailing floor mid-trade and
recovers into the close is scored as surviving. Real firms check the
floor continuously against equity including unrealised profit. This is now
a standing caveat, and the report says the pass rate is an upper bound.

**E4. The reference project carries no license file.**
`dsilva42/prop-risk-geometry` has no `LICENSE`, so it is all rights
reserved. It is cited as the origin of the approach and no code was taken
from it: a verbatim-overlap check against its 113 Python files found a
maximum of 0.87% shingle overlap, which is Python boilerplate. The
`hunter547` URL no longer resolves; the live one is cited.

**E5. The real-world benchmark is ~14%, not ~25%.**
The PropSim-Enterprise empirical dataset reports a phase-1 clear rate of
13-14% on live accounts. That is a *different quantity* from the
model's geometric baseline: it includes intrabar excursions, news and
over-leverage, none of which this simulator models. The right response is
to publish the baseline so the two are not confused, not to treat the
model as broken.

**E6. `docs/PRIOR-ART.md`** now records the full prior-art review with
every license verified from disk, and the project ships it publicly with
attribution.

### A. Wrong numbers — highest priority

**A1. The modelled R:R surface reports pass rates that are wildly wrong for
this strategy.**
`src/prop_lab/surface/surfaces.py`, `_scale_winners_to_rr` and `rr_surface`.
`rr_surface(measured=False)` rescales the winners of the existing trade
array so the mean winner equals the target R:R, keeping the trade count and
the win rate fixed. But changing the R:R changes *which trades exist*. Real
backtest, same signal set, BTCUSDT 15m:

| R:R | modelled (plotted) | measured (re-backtested) | error |
|---|---|---|---|
| 0.5 | 18.0% | 25.0% | 7pp |
| 1.0 | 54.4% | 26.2% | 28pp |
| 2.0 | 76.1% | 26.4% | 50pp |
| 3.0 | 86.0% | 30.1% | 56pp |

The modelled surface says "raise your R:R and your pass rate climbs to 86%".
The truth is 30%. This is the single most misleading output in the package:
it is a picture of a strategy that does not exist.

**A2. The CLI always renders the modelled R:R surface, even when the
strategy supports a measured one.**
`src/prop_lab/cli.py`, `cmd_surface` and `cmd_report` both call
`rr_surface(..., measured=False)` unconditionally. `rr_surface` already
implements `measured=True` correctly, and `_build_strategy_with_rr` already
knows how to override `rr_ratio` on a strategy that exposes it. The CLI never
tries, so a user with a perfectly measurable strategy still gets the fake
surface. The report's honesty section then labels the axis `modelled`, which
is *true* but hides that a measured version was available.

**A3. The surface layer and the headline Monte Carlo use different
resampling, so the report contradicts itself.**
`src/prop_lab/surface/surfaces.py:478` calls `monte_carlo_challenge` without
`session_dates`. `src/prop_lab/cli.py` passes them for the headline. Without
session dates the block bootstrap degenerates toward i.i.d. Measured: surface
`frac_2pct` cell 26.40%, headline 26.54%. Small here, but it is two
different simulations presented as the same number.

**A4. The walk-forward verdict says `holds` for a strategy with no edge.**
`src/prop_lab/validate/walkforward.py:250`, `_verdict`. The rule is
`oos_mean_r <= 0 -> fails`, then `degradation <= 0.10 -> holds`. A strategy
with OOS mean R of +0.009R — a bootstrap 95% CI of [-0.007, +0.036], i.e.
indistinguishable from zero — is reported as `holds`. `holds` reads as
"the edge survived". There was no edge to survive. The verdict is a
statement about *stability*, not about *significance*, and the report
presents it as the latter.

**A5. Nothing anywhere tests whether the edge is statistically
distinguishable from zero.**
The report prints a Wilson CI on the *pass rate* — a CI on the Monte Carlo
estimate, which is precise (25.7% to 27.4%) and says nothing about whether
the strategy has an edge at all. The walk-forward prints mean R with no
interval. A user reads "26.5% (95% CI 25.7% - 27.4%)" and concludes the
number is solid. The number is solid. The *strategy* is not.

**A6. The crypto session rule is wrong on the CLI path, and the CLI cannot
express the right one.**
`src/prop_lab/cli.py`, `_load_bars` calls `assign_session_dates(bars)` with
the default `session_open_hour=18`, which is the *futures* rule. For a 24/7
market like BTC the correct rule is `session_open_hour=0`, which the
docstring in `src/prop_lab/data/schema.py:284` explicitly states. The CLI
has no flag for it. Using the CLI on crypto data silently applies a futures
session boundary.

**A7. The daily-loss-limit warning says "synthetic days of 3 trades" while
the real data has a different trades-per-day.**
`src/prop_lab/firm/monte_carlo.py`, `monte_carlo_challenge` has
`trades_per_day: int = 3` as a default, and the warning reports that
default back to the user as if it were a property of the run. On the
BTCUSDT 15m dataset the real figure is about 3.0 trades per UTC day, so it
happens to be right — by coincidence, not by construction.

### B. Traps

**B1. The CLI never applies purge or embargo, so the "purged walk-forward"
is not purged.**
`src/prop_lab/cli.py:180` calls `walk_forward(factory, bars, n_splits=..., direction=...)`.
`walk_forward` accepts `purge_bars` and `embargo_bars` and the engine
correctly implements both, but the CLI never passes them and never exposes
them as flags. The engine emits a warning every run; the warning is the only
signal. The package is named for purged walk-forward validation and the
documented command sequence never purges.

**B2. The strategy's own operating point is usually not on the surfaces.**
The R:R grid is `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]`. A strategy with R:R 0.667
has no cell. The win-rate grid tops out at 0.60 and starts at 0.30. A user
reading a heatmap cannot find their own strategy on it.

**B3. `winrate_surface` assumes a two-point trade distribution, which is
wrong for any strategy with partial exits.**
`src/prop_lab/surface/surfaces.py`. Every cell assumes (+rr with probability
win_rate, -1R otherwise). A strategy with `session_close` exits has partial
winners and partial losers: measured avg win +0.586R, avg loss -0.734R, and
25.8% of trades exiting at -0.098R average. The parametric surface at the
nearest cell to the real operating point says 6.1% where the truth is
26.5% — 20.5pp out.

### C. Manual-work gaps

**C1. There is no way to sweep strategy parameters from the command line.**
The user goal is "see how different parameters fit into the data". Today the
only way is to write Python. The package already has everything needed for
a parameter sweep: a strategy factory, a backtest, and a simulation.

**C2. There is no way to get the trade list, so a user cannot audit the
result.** The backtest returns `Trade` objects with entry, exit, stop,
target, R and exit reason. The CLI prints only aggregate statistics. For a
tool whose entire pitch is honesty, the user cannot check the trades.

**C3. The strategy's parameters are not introspectable from the CLI.**
`StrategyCore.params_help()` exists and is used in `cmd_backtest`, but there
is no command that lists the parameters of a strategy file, so an agent
cannot discover what it can tune.

### D. Release hygiene

**D1. Personal information.** Already removed from the working tree, the git
history, the object database, the reflog and the remote. See the scrubbing
record. Must be re-verified before publishing.

**D2. No CI.** A public repository needs a workflow that runs the suite on
every push, on every supported Python version.

**D3. No `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md` or issue
templates.** For a repository whose whole purpose is to be used and
improved by strangers.

**D4. The README is written for someone who already knows the domain.** It
does not walk a beginner from idea to number. The "Writing a strategy"
section assumes the reader has written a backtest before.

**D5. `pyproject.toml` has no `readme` content-type, no `keywords` for
discovery, no classifiers for the supported Python versions, and the
version is hardcoded rather than driven by a single source of truth.**

**D6. The offline test suite has a 10-second budget that is easy to break
silently.** NOTES.md records that a network test once leaked into the
offline run and pushed it to 98 seconds. There is no mechanism to stop it
happening again.

---

## Evidence log

Every defect above was found by running the package, not by reading it.

- Baseline backtest, BTCUSDT 15m, 134,675 bars, 2023-01-01 to 2026-11-03:
  4,128 trades, 56.7% win, +0.0145R mean, PF 1.046.
- Bootstrap 95% CI on mean R: **[-0.0073, +0.0360]** — includes zero.
  t-statistic +1.31. The edge is not distinguishable from noise.
- Walk-forward, 5 folds: OOS mean R +0.0094, degradation -0.0015,
  verdict `holds`. Verdict unchanged with purge/embargo at 50/50, 200/200
  and 500/500.
- Modelled vs measured R:R surface: divergence to 56pp (A1).
- Surface `frac_2pct` cell 26.40% vs headline 26.54% (A3).
- Session rule h=0 vs h=18 on BTC: 4,128 vs 4,157 trades, +0.0145 vs
  +0.0162 mean R (A6).
- `session_close` exits: 1,067 of 4,128 (25.8%), mean R -0.098, all firing
  at 00:00 UTC (B3).
- Parity check on the fill model: all 1,125 stop exits at exactly -1.0R and
  all 1,935 target exits at exactly +0.6667R; win-rate/mean-R consistency
  reconciles to 1e-6.
- Visual audit of all six surfaces: heatmap annotations correct and legible;
  the z-axis label and the colorbar label are two distinct texts (verified
  by pixel A/B diff against the repo's own `to_3d`), so the earlier
  "missing z-axis title" reading was wrong and NOTES.md stands.
