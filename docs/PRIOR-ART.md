# Prior art: public open-source prop firm challenge pass-rate simulators

Research date: **2026-11-03**. Every repository below was fetched and read (README +
license file + core source files), unless explicitly marked *unverified*. Repositories that
were unreachable are listed as such rather than described from memory.

**Verdict up front:** of the five repos named in this project's docs, **four are MIT** and
safe to reuse; the fifth (**dsilva42/prop-risk-geometry**) has **no license file at all** and
is legally all-rights-reserved. **No GPL or AGPL code appears anywhere in the prop-firm
simulator prior art** — every simulator found is MIT, Apache-2.0, or unlicensed. (GPL-3.0
and AGPL-3.0 repos do appear in *generic* trading/backtesting searches — e.g.
`SmartsvXD/QuantEvo`, `bartonguestier1725-collab/backtest-engine` — but none of them
simulate prop firm challenges, and LuxAlgo's separate `LuxAlgo/PineTS` runtime is AGPL-3.0
though unrelated to this problem.) The
purged-walk-forward + bootstrap-Monte-Carlo combination you already have is **not** present
anywhere else as a single pipeline — the closest prior art splits it in two.

---

## Part 1 — The five repos named in this project's docs

### 1. dsilva42/prop-risk-geometry

- **URL:** https://github.com/dsilva42/prop-risk-geometry
- **Former URL:** https://github.com/hunter547/prop-risk-geometry — **404 (gone)**.
  The `hunter547` GitHub account still exists (37 public repos, none of them this one) and
  the git history in `dsilva42/prop-risk-geometry` is authored by *Hunter Evanoff*, including
  a merge commit message referencing `github.com:hunter547/prop-risk-geometry`. So the repo
  was renamed/transferred, not rewritten. **Cite the dsilva42 URL.**
- **LICENSE: NONE. No `LICENSE`, `LICENSE.md`, `COPYING`, or `NOTICE` file exists.**
  GitHub's API reports `"license": null`. The README badge reads
  `License-Educational` and the README's License section is a *disclaimer*, not a grant:
  > "This project is for educational and research purposes. It simulates hypothetical
  > outcomes and does not constitute financial advice."
  **Legal status: all rights reserved by default.** You may read it and learn from it; you
  may not copy its code into an MIT-licensed release without explicit permission from the
  author. This is the single most important licensing fact in this report, because this
  repo is the lineage your own `prop_lab/firm/monte_carlo.py` docstring cites as the
  populariser of the bootstrap approach.
- **Stack:** Python 3.7+, numpy/pandas/matplotlib/seaborn/scipy, `backtesting.py`,
  `yfinance`, optional scikit-learn/xgboost, `project-x-py` (TopstepX). 141 files,
  52 commits, first commit 2026-04-14, last push 2026-06-11. Zero stars.
- **What it actually does:**
  - Backtests three intraday strategies (Opening Range Breakout, Mean Reversion, Trend
    Continuation) on real NQ/MNQ 5-minute bars, across (entry window × direction × R:R).
  - **Challenge phase Monte Carlo** (`simulation/engine.py`): 500 attempts per
    position-sizing method. Two modes: a legacy weighted-coin-flip and a **bootstrap** mode
    that draws each trade's `pnl_r` from the real historical distribution
    (`pnl_r_distribution[rng.integers(n_outcomes)]`).
  - **Funded phase Monte Carlo** (`simulation/funded_phase.py`): 60-day funded simulation
    with a trailing EOD floor that ratchets up and **caps at `$50,000`**, periodic payouts
    (every 5 days, 50% of profit above start, capped at `$5,000`), and a "micro fallback"
    that shrinks risk when equity comes within a threshold of the floor.
  - **Risk geometry** is the core concept: five position-sizing functions in
    `RISK_FN_MAP` — `max_size` (fixed $500), `frac_2pct` (2% of equity),
    `dd_frac_40pct` (40% of remaining drawdown room), `anti_martingale` (scale with
    cushion), `floor_aware` (shrink near the floor).
  - **Surfaces: yes.** `plots/surface_3d.py` renders pass rate over (win rate × R:R) as a
    3-D surface with the EV=0 breakeven curve `WR = 1/(1+RR)` projected onto it, plus a
    50%-pass-rate reference plane. `plots/heatmaps.py` renders (R:R × risk geometry) and
    (strategy × direction) heatmaps. `plots/ev_breakeven.py` maps backtest results onto the
    EV=0 breakeven map.
  - Independent validation: top strategies are re-run through `backtesting.py` and must
    agree within ~2pp on win rate and ~5% on trade count.
- **How it estimates the pass rate:** Monte Carlo, **per-trade i.i.d. bootstrap** from the
  real per-trade R distribution (no block structure in `engine.py`; the newer
  `simulation/strategy_engine/mc.py` adds a **per-day block bootstrap** that resamples whole
  trading days as units to preserve within-day correlation).
- **Notable implementation details worth copying:**
  - The **risk-capping** pattern in `simulate_path`: risk is clamped to
    `min(risk_fn(...), MAX_DAILY_DRAWDOWN, dist_to_floor - 1)`. This is the mechanism behind
    the "risk cliff" that `lhenriet-bit/prop-odds` describes in prose.
  - The **funded-phase floor cap** (`FLOOR_AFTER_PAYOUT`) — the floor ratchets up but never
    above the starting balance, and withdrawals reduce equity without moving the floor.
  - **Bayesian shrinkage** toward 50% win rate for small samples, with `MIN_BACKTEST_TRADES
    = 30` hard filter.
  - A dedicated **contract-quantisation** layer in `strategy_engine/mc.py`
    (`contract_min_policy="force_one"`, point-value lookup per instrument, a
    `fixed_contracts_ladder` for tiered sizing) with diagnostics
    `pct_trades_min_contract_forced`, `pct_trades_skipped_undersized`,
    `pct_trades_capped_by_risk_fn`. This is more thorough than anything else in the field.
- **Stated limitations and known weaknesses:**
  - The README is explicit that surfaces are **averages across buckets** (`aggfunc="mean"`),
    and the 3-D surface is **smoothed with a 3×3 averaging kernel applied 4 times**
    (`_smooth_2d(Z, passes=4)`) purely "for visual quality". **The plotted surface is not
    the computed surface.** Worth noting; do not copy the smoothing.
  - The rules are **hard-coded in `config.py`**: `$50,000` balance, `$3,000` target,
    `$1,000` max daily drawdown, `$2,000` trailing floor, `MAX_TRADES_PER_DAY = 20`.
    There is **no ruleset dimension** — the "risk geometry × ruleset" surface you have
    does not exist here. `FLOOR` is *static* in the challenge phase (`equity <= cfg.FLOOR`),
    i.e. the challenge engine has **no trailing drawdown at all** in the challenge phase;
    trailing only appears in the funded phase.
  - **No purged walk-forward.** `simulation/ml/cv.py` has `walk_forward_splits` and a
    `PurgedKFold` class that **raises `NotImplementedError`** — "NOT YET IMPLEMENTED — use
    walk_forward_splits for now." The MC modules contain no OOS/walk-forward references at
    all. The backtest and the Monte Carlo are **not connected**: the MC runs on the full
    in-sample backtest output.
  - The daily loss limit is enforced as a **per-trade cap** (`min(risk, MAX_DAILY_DRAWDOWN)`),
    not as an intraday account-level daily drawdown rule.
  - No slippage/commission model in the MC, no minimum trading days, no consistency rules,
    no payout gating, no minimum-trade-count enforcement on the simulator side.
  - 0 stars, 0 issues, no CHANGELOG, single author, last push 2026-06-11 (nearly 5 months
    stale as of today).

---

### 2. LuxAlgo/prop-firm-sim

- **URL:** https://github.com/LuxAlgo/prop-firm-sim
- **LICENSE: MIT** (`LICENSE`, "MIT License, Copyright (c) 2026 LuxAlgo Global, LLC").
  Per-package `LICENSE` files in `packages/core`, `packages/cli`, `packages/mcp`. Also
  `TRADEMARKS.md` reserving "Prop Firm Sim" and the LuxAlgo name/logo — **do not use the
  project name or LuxAlgo branding in your release.** `DISCLAIMER.md` also exists.
- **Stack:** TypeScript, npm monorepo (`@luxalgo/prop-firm-sim-core` / `-cli` / `-mcp`),
  zod schemas, zero-runtime-dependency core. 104 stars, 35 forks, 15 commits on `dev`
  (default branch), first commit 2026-08-26, last push 2026-10-28. Tagged releases
  `core@1.1.0 … 1.3.0`, `cli@1.1.0 … 1.2.0`, `mcp@1.1.0 … 1.3.0`.
- **What it actually does:** This is the most complete simulator in the space. A full
  challenge **journey** — repeated attempts (attempt cap, default 25) with fees, resets,
  then a funded stage over a 90-trading-day horizon with payouts and blow-up risk.
- **How it estimates the pass rate:** Monte Carlo over **paths**, with a deterministic,
  byte-reproducible RNG (`xoshiro128**` seeded via `splitmix32`, integer-only arithmetic so
  streams are bit-identical across JS engines). Two trader profiles:
  - `ParametricSource` — win rate + lognormal avg-win/avg-loss R distributions;
    trades/day fixed or Poisson.
  - `BootstrapSource` — **stationary block bootstrap** (Politis & Romano 1994) over a
    user-supplied R series, preserving autocorrelation.
  - Per-path seeds derived from `(masterSeed, pathIndex)` so **common random numbers** work
    across sweep grid points (the risk sweep is smooth because of this).
  - Wilson 95% CIs on pass probabilities.
- **Surfaces:** not a 2-D (win rate × R:R) surface. What it has instead are **1-D sweeps**:
  `optimalRisk` (risk per trade → pass probability and EV, reporting the argmax of each
  separately and a `diverges` flag when they differ) and `sensitivity` (win rate ±1..3 points,
  reporting `passProbabilityGradientPerPoint`). **No measured-vs-modelled axis tagging.**
- **Notable implementation details worth copying — this is the richest vein in the report:**
  - **The four max-loss modes** (`packages/core/src/engine/rules.ts`, `spec/challenge.ts`):
    `static-initial`, `trailing-realized-eod`, `trailing-intraday-unrealized`,
    `trailing-locks-at-initial`. Plus `locksAtInitial` and `lockOffsetAmount` for firms
    that lock at start + $100.
  - **The daily-loss rule is decomposed into four independent switches**:
    `limitBasis` (`initial-balance` vs `anchor`), `basis` (`prior-day-balance` vs
    `prior-day-equity`), `includesOpenPnl`, `evaluation` (`intraday` vs `end-of-day`).
  - **Breach attribution:** when one trade crosses both floors, the failure is attributed to
    the **higher** floor — "the one that was crossed first as equity fell".
  - **Consistency rules are simulated as a moving target**, with a rational-trader model:
    the trader stops a day when further profit cannot improve compliance, and keeps trading
    across days until the best day's share complies. The docs are blunt that
    **"the rule forces extra days at risk"** and can convert a pass into a blowup.
  - **`maxDays: null` → unlimited**, with a configurable `unlimitedStepDayCap` and the
    capped attempts counted as failures and flagged.
  - **Three-tier rule provenance policy:** structured columns verbatim; free text inferred
    only when one reading exists (and disclosed via `provenance: "directory+inferred"` +
    `inferredFields`); **ambiguity refused** — a challenge whose loss semantics can't be
    established is reported *not simulatable* rather than guessed.
  - **`flagsNotSimulated`** — every unsimulated rule is declared and echoed into every
    result, plus an `assumptions.flags` list.
  - **Golden snapshot tests** pin published numbers; "Silent drift in published odds is the
    one bug this repo treats as unforgivable."
  - `docs/rule-semantics.md` contains hand-computed worked examples for every rule
    semantic, each also a unit test.
- **Stated limitations and known weaknesses (from the README, "Assumptions, owned loudly"):**
  - "Trades are same-day round trips; intra-trade excursions are not modeled. This slightly
    **understates trailing-drawdown risk**, so real odds are somewhat worse."
  - "Attempts are i.i.d. draws from your stated stats: no learning, no tilt."
  - Consistency rules, payout gating and locking trails **are** simulated; scaling plans
    and firm-specific niceties are not.
  - "The funded stage is a horizon simulation: withdrawals take the maximum the rules allow
    and never cross the loss floor; monthly billing approximates to 21 trading days."
  - "The news calendar is a recurring template, not a historical feed."
  - **Open issue #5 (Apex)**: the live directory carries **Apex's Legacy plans**, not the
    current Intraday Evaluation; **18 of 21 rows are not simulatable** because
    `maxLossType: "Trailing"` with `maxLossMode: null` is refused by the adapter; Apex
    prohibits automation but the directory sets `autoTrading: true` on every row.
  - **Open issue #6**: conduct rules (automation, news, copying) don't reach the results —
    the adapter doesn't set `constraints` and leaves `flagsNotSimulated` empty.
  - **No walk-forward, no purge, no embargo** — `grep -riE "walk.?forward|purge|embargo"`
    over the whole repo returns **nothing**. The simulator takes your stats as given.
  - Rules come from a **proprietary runtime API** (`app.luxalgo.com/api/propfirms/list`),
    not a shipped dataset. You can pass a spec inline, but the dataset is not in the repo.
- **Reusability:** MIT, so the rule-semantics enum, the `RuleState` class, the stationary
  block bootstrap, the consistency-rule rational-trader model, and the common-random-numbers
  path seeding are all copyable with attribution. The TypeScript must be ported to Python.

---

### 3. lhenriet-bit/prop-odds

- **URL:** https://github.com/lhenriet-bit/prop-odds
- **LICENSE: MIT** (`LICENSE`, "MIT License, Copyright (c) 2026 Lucien Henriet").
- **Stack:** Python 3.10+, **numpy only** (matplotlib optional extra), `pytest`. 19 files,
  **1 commit** ("Initial release", 2026-07-29). 0 stars, 0 issues, 0 forks.
- **What it actually does:** A focused pass-probability calculator for evaluation challenges:
  two-step / one-step / trailing-drawdown rule sets, phase-by-phase pass rates, expected
  attempts, expected fees per funded account, and a risk sweep.
- **How it estimates the pass rate:** **Fully vectorised numpy Monte Carlo** — all `n_paths`
  paths are advanced simultaneously with boolean masks (`active`, `passed`,
  `failed_daily`, `failed_total`). Trades are **i.i.d. Bernoulli**: win `rr × risk` with
  probability `win_rate`, else lose `risk`. No resampling of any kind — no bootstrap, no
  block bootstrap. `rng = np.random.default_rng(seed)`, reused across phases.
- **Surfaces:** no 2-D surfaces. A 1-D **risk sweep** (`optimize.py::sweep_risk`) producing
  `P(funded)`, `E[days]`, `cost per funded` per risk level. Notably it **derives a new seed
  per grid point** (`seed + i`).
- **Notable implementation details worth copying:**
  - **The risk cliff.** The README makes a point of it: with a 5% daily limit and 3
    trades/day, risking 1.5% means three straight losses cost 4.5% — the daily limit is
    *mathematically out of reach*; at 1.75% the account dies. The sweep shows
    `1.00% → 93.9%` then `1.75% → 42.5%`. **This is a discontinuity, not a smooth slope**,
    and it is the most practically useful single insight in any of these repos.
  - **Three-way failure decomposition**: `fail_daily_rate` / `fail_total_rate` /
    `timeout_rate`, plus `days_to_pass` as an array over passing paths only (so "days to
    pass" is explicitly conditional on passing).
  - **The "coast at zero risk" convention**: hitting the target before `min_trading_days`
    sets `days_to_pass = max(day, min_trading_days)` and stops trading. Documented.
  - **Trailing total drawdown** is a boolean `trailing_total` on `PhaseRules`: floor is
    `hwm - max_total_loss_pct` when trailing, `100 - max_total_loss_pct` otherwise.
  - `expected_cost_per_funded = fee / funded_rate` (the non-refundable fee model).
  - Presets are JSON with a `source` field and a `refundable_fee` flag.
- **Stated limitations and known weaknesses (README, "Model assumptions"):**
  - "Trades are i.i.d.: you win `rr × risk` with probability `win_rate`, otherwise you lose
    `risk`. **No fat tails, no autocorrelation, no tilt after a losing streak. Real trading
    is usually worse, so read these odds as optimistic.**"
  - "Not modeled yet: **consistency rules**, news-trading restrictions, slippage, payout
    mechanics."
  - Fixed risk per trade as a % of the **initial phase balance** (not compounding).
  - "Limits are checked after every trade."
  - The README explicitly frames the whole tool as **"a calculator"** that "does not predict
    your results."
- **Reusability:** MIT, small codebase (5 source modules), readable. The `PhaseRules` /
  `ChallengeRules` dataclasses and the vectorised mask-loop are directly portable.

---

### 4. Makeph/prop-ev

- **URL:** https://github.com/Makeph/prop-ev
- **LICENSE: MIT** (`LICENSE`, "MIT License, Copyright (c) 2026 Makeph").
- **Stack:** Python 3.9+, **standard library only** (`random`, `dataclasses`). 11 files,
  2 commits, tag `v0.1.0`. 2 stars, 0 issues, 0 forks. Live browser calculator at
  `makeph.github.io/prop-ev` (plain `docs/index.html` + GSAP/ScrollTrigger).
- **What it actually does:** The most *honest-scoped* of the five. It does **not** claim to
  predict pass rates; it frames a challenge as a **variance play** and asks whether
  attempting it is +EV after the fee.
- **How it estimates the pass rate:** Plain Monte Carlo, **no resampling**. Each trade's P&L
  is `account * rng.gauss(edge_bps, std_bps) / 1e4` — a **normal** distribution with
  configurable mean and standard deviation in basis points. `random.Random(seed)`, default
  `n_sims=20000`. Phases are run with `all(_run_phase(...) for _ in range(firm.phases))`,
  so a run passes only if every phase passes.
- **Surfaces:** no 2-D surfaces, but a **1-D edge grid**: `python -m propev table` sweeps
  per-trade edge from −2.0 to +2.0 bps across four archetypes and prints the P(pass) matrix.
  That table is the tool's centrepiece: "at zero edge every archetype passes ~0% of the
  time."
- **Notable implementation details worth copying:**
  - **The consistency rule is implemented as a deferral, not a filter**: if the best day's
    profit exceeds `consistency × total profit`, the day `continue`s — the trader keeps
    trading until the best day's share complies.
  - **Breach is checked before the pass**, and the total-DD reference is `peak` (trailing)
    or `start` (static) — re-derived per trade from the running peak.
  - `expected_value()` is deliberately conservative: funded value ≈
    `target × account × profit_split` — "one cycle of withdrawable profit… most funded
    accounts are lost before compounding."
  - **The EV sweep framing.** Instead of "what's my pass rate", it asks "where does this
    challenge flip +EV", and deliberately shows `e ≤ 0` cells as `−fee` rather than
    hiding them. This framing is worth adopting for the honesty section.
  - Property-based tests: "more edge ⇒ higher pass rate; tighter drawdown / more phases ⇒
    lower pass rate; a zero-edge fee-charging challenge is always −EV; same seed ⇒ same
    result."
- **Stated limitations and known weaknesses:**
  - README: "Deliberately simple and transparent — a go/no-go sanity tool, **not a
    predictor**."
  - "What it does **not** model: news gaps, slippage tails, scaling plans, payout delays,
    or the firm changing the rules on you. All of those make reality *worse* than this
    sim, not better."
  - "The built-in presets are **approximate archetypes**, not any specific firm's live
    terms."
  - **Normal** per-trade distribution is unrealistic for bounded-risk strategies (no hard
    stop at −1R). Also `max_days: int = 30` default time limit.
- **Reusability:** MIT, ~200 lines of core logic, trivially portable. Good model for the
  EV-framing and the consistency-deferral rule.

---

### 5. gabrielee5/prop-firm-simulator

- **URL:** https://github.com/gabrielee5/prop-firm-simulator
  (note: the README's clone instructions say `github.com/gabrielefabietti/prop-firm-simulator`
  — that is a **different, wrong owner string**; the working URL is the `gabrielee5` one.
  The author's real name is Gabriele Fabietti.)
- **LICENSE: MIT** (`LICENSE`, "MIT License, Copyright (c) 2026 Gabriele Fabietti").
- **Stack:** Python, numpy/matplotlib/pandas (for plotting only), PyYAML config. 18 files,
  17 commits, first 2026-03-11, last 2026-03-14. 2 stars, 1 fork, 0 issues.
- **What it actually does:** Evaluates whether a **zero-edge (EV=0) strategy** can pass
  challenges purely through variance, and estimates the value of a funded account via
  repeated withdrawals.
- **How it estimates the pass rate:** Plain Monte Carlo, **no resampling**. For each R:R
  ratio the win rate is *derived* to force EV=0: `calculate_ev_zero_win_rate(rr) =
  1/(1+rr)`. Each trade is a Bernoulli coin flip. `random.Random(seed + int(rr*1000))` so
  different R:R values get different but reproducible streams.
- **Surfaces:** yes, but **EV surfaces rather than pass-rate surfaces**.
  `unified_simulator.py` builds a **3-D parameter space** — (risk % × R:R × withdrawal
  target %) — with an `ev_matrix` numpy array for 3-D surface plots and heatmaps. The
  README calls it "**optimizes across a 3D parameter space**". **This is the closest any
  prior art comes to a 3-D risk-geometry surface**, though the third axis is withdrawal
  target rather than ruleset, and the objective is EV not pass rate.
- **Notable implementation details worth copying:**
  - **Risk capping against remaining headroom** in `simulator.py`:
    ```python
    remaining_overall_room = firm.overall_max_loss + running_pnl
    risk_amount = min(base_risk_amount, remaining_overall_room)
    remaining_daily_room = firm.daily_max_loss + daily_pnl
    risk_amount = min(risk_amount, remaining_daily_room)
    ```
    Risk shrinks as the limits approach — a realistic sizing policy that most simulators
    omit.
  - **Round-trip fee deduction upfront**, estimated from assumed leverage:
    `balance × leverage × fee_pct/100 × 2`, charged at the **start** of each trade.
    README quantifies the impact: 1:2 R:R pass rate drops from **32.10% → 23.70%** with a
    $240/trade fee. A concrete reminder that fees are not a rounding error.
  - **A caching optimisation**: challenge results are computed once over the 2-D grid and
    reused for all withdrawal targets in the 3-D sweep.
  - **The Gambler's Ruin derivation** in `docs/PRIMER.md`: for a symmetric random walk,
    `P(success) = MaxLoss / (MaxLoss + Target)`, and for an asymmetric walk
    `P = (1 − r^a)/(1 − r^(a+b))` with `r = q/p`, `a = target/win_amount`,
    `b = max_loss/loss_amount`. `E[withdrawals] = P/(1−P)`. **This is the only closed-form
    treatment of the problem in the entire prior-art set** and is a genuinely useful
    analytical cross-check for your MC engine.
  - Funded-phase loop: trade until the withdrawal target → withdraw → reset P&L to 0 →
    continue until blown. `EV = P(pass) × E[total_withdrawals | passed] × payout% −
    challenge_cost`.
- **Stated limitations and known weaknesses:**
  - **No trailing drawdown in the challenge phase.** The overall loss limit is
    `running_pnl <= -firm.overall_max_loss` — a **static** limit from the starting balance.
    `peak_pnl` is tracked but only to compute `max_drawdown` for reporting. So this repo
    cannot model Apex/Topstep trailing rules at all.
  - **No minimum trading days, no consistency rule, no time-limit realism** beyond
    `max_trading_days`.
  - **The EV=0 constraint is the whole thesis**, which means it cannot answer "what if my
    edge is real?" — the win rate is a function of R:R, not an input.
  - No slippage, no commission realism beyond the flat fee estimate, no weekend/overnight
    handling.
  - The git history contains a commit literally titled "**fix: it was not respecting max
    overall loss**" (2026-03-11) — a reminder that these engines get the boundary
    conditions wrong.
- **Reusability:** MIT. The Gambler's Ruin primer, the headroom-based risk capping, and the
  3-D EV sweep structure are the useful parts.

---

## Part 2 — Other open-source simulators you did not list

All verified by fetching the repository and its license file.

### 6. MaverickHQ/prop-challenge-lab — **Apache-2.0**

- **URL:** https://github.com/MaverickHQ/prop-challenge-lab
- **LICENSE: Apache-2.0** (`LICENSE`, plus a `NOTICE` file). Python, 223 files, 0 stars,
  last push 2026-08-31.
- **What it does:** Not a pass-rate calculator — a **backtesting harness engineered to
  refuse false results**, plus the closed research programme that used it (24 pre-registered
  hypotheses, three sealed verdicts, no edge found, no fee ever paid).
- **How it estimates pass probability:** `occams/harness.py::monte_carlo` — one attempt
  from **every viable start day** over a full horizon. **Not a resampling bootstrap**: it
  replays the *same* historical path from different start dates. It caches a per-day
  ledger so days are simulated once and windows replay the cheap rules engine over ledger
  slices.
- **Notable algorithmic ideas — several are directly transplantable:**
  - **`occams/rules.py::ChallengeState`** — a clean daily-mark rules engine. Trailing floor
    `min(eod_high − trailing_dd, account)` (trails but locks at the start balance),
    **breach checked before pass**, sticky terminal states, consistency rule as
    `max_day_profit <= frac × net_profit` evaluated **at pass time** (it gates the pass but
    never breaches), and a `snapshot()`/`restore()` pair so state is persistable.
  - **`occams/profiles.py`** — rule sets as **versioned, dated external JSON profiles**
    with `effective_date` and `source`, and `assert_fresh()` which **raises** on a profile
    older than a tolerance rather than warning. *"A rule snapshot with no date on it is a
    silent expiry waiting to happen."* Missing rules are refused, never defaulted: "a rule
    that silently becomes zero is a rule that never binds."
  - **`occams/stats.py::block_bootstrap_ci`** + `optimal_block_length`, with two
    load-bearing properties enforced by tests: `block=1` must reduce **exactly** to the
    plain bootstrap, and on a serially correlated series the block bootstrap must be
    **wider**. numpy-only at runtime, scipy as a test-only oracle.
  - **`occams/power.py`** — design effect `n_eff = n / (1 + (m−1)·rho)` for correlated
    data; underpowered runs are **refused**, not caveated.
  - **`occams/calibration.py`** — every estimator is run against a **dead world** (must
    return ~0) and a **planted-edge world** (must recover the effect). Both halves
    required.
  - `seed` is a **required keyword argument** everywhere — no default seeds.
- **Stated limitations:** The research programme is **closed**. No edge was found; the
  harness "stands on its own". Note the name: despite "prop-challenge-lab" it is **not** a
  pass-rate simulator — it is a validation harness. Its challenge engine is
  **EOD-mark-based** (one equity mark per day), so it does not do intraday trailing.

### 7. jalv92/PropSim — **MIT**

- **URL:** https://github.com/jalv92/PropSim
- **LICENSE: MIT** ("MIT License, Copyright (c) 2026 jalv92"). Python 3.9+, numpy only,
  88 files, 0 stars, last push 2026-07-25.
- **What it does:** Scores real NinjaTrader 8 trades against the published rules of five
  prop firms (148 account variants), reading `NinjaTrader.sqlite` directly.
- **How it estimates the pass rate:** **Bootstrap that resamples whole DAYS** (the pool is
  day-level, not trade-level), vectorised numpy over `sims` paths.
- **The most valuable part of this repo is its documented bug history** — `sim.py`'s
  module docstring records three bugs that "made every previous pass-rate figure
  optimistic":
  1. **Breach was tested only at the close.** Four of five firms state the floor is
     breached in real time on equity *including unrealised P&L*. A strategy that dipped
     below the floor intraday and recovered into the close was surviving in the sim and
     dead in reality.
  2. **The funded high-water mark ignored unrealised gains.** "The trailing drawdown is
     calculated intraday based on your peak balance, which includes both realized and
     unrealized gains."
  3. **The ratchet used the closed P&L, not the unrealised peak.** Worse, "two extremes do
     not describe a path at all": a trade that ran +1500, fell to −600, then ran +1800 has
     its low *before* its peak, yet no model built on MAE/MFE — with or without their
     order — sees that. They introduced **`intra_mdd`**, the fall from the *running* peak
     inside the trade, making the test exact for any path shape:
     `breach ⇔ max(hwm − balance − mae, intra_mdd) >= max_dd`.
  - It also has an explicit **honesty gate**: `MIN_DAYS = 10` and `MEANINGFUL_DAYS = 60`.
     Below `MIN_DAYS` the headline pass rate is **withheld**, not rendered with a warning:
     "a number on a screen outranks a caption every time." Measured on a real two-trade
     session, the bootstrap reported P(pass) = 100%.
  - Sizing policies include a **Dubins–Savage bold play**: "enough contracts to reach the
    target in one winning trade, but never more than the buffer survives."
- **Surfaces:** none (0 hits for "surface" in `sim.py`). PropSim is a per-firm scorer.

### 8. greybeard-code/gbBacktester — **MIT**

- **URL:** https://github.com/greybeard-code/gbBacktester
- **LICENSE: MIT** ("MIT License, Copyright (c) 2026 GreyBeard"). Python 3.12+, numpy +
  pyarrow + plotly, **no pandas by design**; pytest + hypothesis. 141 files, 0 stars.
- **What it does:** Tick-level futures backtester over NinjaTrader Market Replay Parquet
  data, with prop-firm rule validation built in.
- **How it estimates the pass rate:** `backtester/montecarlo.py` — **Davey BWATS /
  Masters SSML** trade resampling. **This is the closest prior art to your pipeline**:
  - **Automatic method selection**: it computes the max |autocorrelation| of the trade P&L
    series over lags 1..10; if `> 0.2`, plain i.i.d. resampling is **invalid** and a
    **circular block bootstrap** is used instead. `method="auto"` picks automatically.
    Default block size `max(2, round(sqrt(n)))`.
  - **Prop-firm breach detection uses each trade's recorded MFE/MAE**, so intraday
    excursions count. It states the limitation honestly: "The true intra-trade ordering of
    MFE vs MAE is unknowable after resampling, so the model assumes the adverse excursion
    comes first." A trade's MAE is tested against the floor trailed from *prior* trades;
    its close is additionally tested against the floor raised by the trade's own MFE.
  - **Peak accounting excludes the trade's own MFE**: `peak_prior` is the running peak of
    *prior* trades, because "a winner's own MFE must not count as a peak preceding its own
    dip".
  - **Ties count as a breach**: `prob_pass = (first_hit < first_breach).mean()`; a target
    and a breach in the same trade resolves as a breach.
  - It reports `prob_breach`, `prob_pass`, `prob_fail` and **unresolved** separately.
  - **Walk-forward exists separately** (`backtester/walkforward.py`) — rolling IS/OOS
    windows, optimise on IS, run best params on OOS, stitch, report **Walk-Forward
    Efficiency** = mean(OOS metric)/mean(IS metric), "> ~0.5 commonly acceptable", plus a
    `manifest` with `engine_sha` and raw-data fingerprints. **But the two are not
    connected**: `walkforward.py` contains **zero** references to the Monte Carlo, and the
    MC is not fed by OOS output. This is the same gap your project fills.
  - It has **no purge and no embargo** — the windows are contiguous, no label-overlap
    handling.
- **Surfaces:** none.

### 9. ranjeet867/Metatrader — **MIT**

- **URL:** https://github.com/ranjeet867/Metatrader
- **LICENSE: MIT** ("MIT License, Copyright (c) 2026 mt5_quant_trader_v2 contributors").
  Python 3.11+, **9786 files**, 539 tests claimed passing.
- **What it does:** Full MT5 quant platform. Contains `core/ftmo_simulator.py` — a
  Monte-Carlo bootstrap pass-rate simulator for an FTMO two-phase test over a **portfolio**
  of strategies.
- **How it estimates the pass rate:** Bootstrap from each strategy's **OOS R-multiple
  distribution** (the docstring is explicit: "each strategy's **OOS** R-distribution"),
  with `rng.poisson(trades_per_day)` trades per day per strategy, and `rng.choice(pool,
  replace=True)` per trade. Risk is expressed as a % of the **day-start** balance.
  Reports `p_pass`, `p_daily_breach`, `p_total_breach`, `p_no_resolution`, plus **per-strategy
  attribution** of both expected return and max drawdown. Seeded (invariant #7, default 42).
- **Notable:** it combines a portfolio of strategies with per-strategy contribution
  attribution, and it separates `p_no_resolution` from the breaches. Same three-way
  decomposition as `prop-odds`.
- **Limitations:** the total-loss cap is `total_loss >= total_loss_cap_pct` measured from
  `starting_balance` — again **static, not trailing**. Daily and total caps are both
  checked at **end of day** only, so intraday breaches are missed.

### 10. PropSim-Enterprise/prop-firm-rules — **MIT** (rules engine, not a simulator)

- **URL:** https://github.com/PropSim-Enterprise/prop-firm-rules
- **LICENSE: MIT** ("MIT License, Copyright (c) 2026 PropSim"). TypeScript, zero runtime
  dependencies, Node 20+ / browser.
- **What it does:** Evaluates prop-firm challenge rules against a **time-ordered series of
  balance/equity snapshots** and returns status + which rule was breached + when + distance
  to every limit. Covers profit target, max daily loss, max overall loss, **trailing
  drawdown**, min/max trading days, the **consistency rule**, and a **news rule** (no trade
  opened or closed around a central-bank decision). Handles **day reset with timezone
  offsets** (`dayResetUtcOffsetMinutes`).
- **Relevance:** it is a *checker*, not a *sampler* — no Monte Carlo. But it is the most
  complete public implementation of the **rule semantics** (including the parts no
  simulator bothers with: day-reset timezone, news rule, min/max trading days, and
  per-rule "distance to floor" telemetry). Useful as a reference when you model APEX/Topstep.

### 11. GithinjiHans/PropFirmEval — **NO LICENSE**

- **URL:** https://github.com/GithinjiHans/PropFirmEval
- **LICENSE: none found** — no `LICENSE` file. Two files total: `README.md` and
  `PropFirm.ipynb`. **All rights reserved.**
- **What it does:** Monte Carlo for a challenge phase + a funded phase with profit
  harvesting and an insurance-fund split. Fixed-dollar risk, fixed win rate and R:R.
- **Value:** it is one of the few READMEs with an explicit **"Assumptions (idealized firm)
  vs What real firms often add"** table — static drawdown vs trailing, no daily limit, no
  time limit, i.i.d. outcomes. Honest about the gap.
- **Not reusable** without permission.

### 12. michaelsboost/PropForge — **MIT** (a training game, not a simulator)

- **URL:** https://github.com/michaelsboost/PropForge
- **LICENSE: MIT** (`LICENSE.md`, "The MIT License (MIT) Copyright (c) 2025 Michael
  Schwartz"). Vanilla JS + Tailwind + Pico CSS, no build step.
- **What it does:** An **interactive prop-firm training simulator** — phase-based
  evaluations, $25K→$1M tier progression, live candlestick chart with simulated ticks, SL/TP,
  scaling, flipping positions, unrealised P&L, margin management, lot limits.
- **Relevance:** a *manual practice* environment, not a pass-rate estimator. No Monte
  Carlo, no surfaces. Useful only as a UX reference for a challenge-simulator interface.

### 13. GithinjiHans-style micro-simulators and other no-license repos (verified, not reusable)

- **Lihui-03/PropFirm** — https://github.com/Lihui-03/PropFirm — **no LICENSE**; 7 files,
  mostly an IntelliJ project; README is one paragraph: "A free calculator that picks the
  prop firm your trading is most likely to pass." Nothing to reuse.
- **GT-mandem/backtester** — https://github.com/GT-mandem/backtester — **no LICENSE**;
  "backtesting framework for equity/index trading strategies with prop-firm rule validation."
- **Cooderhome/Prop-Firm-Challenge-Simulator-PFCS-** —
  https://github.com/Cooderhome/Prop-Firm-Challenge-Simulator-PFCS- — **no LICENSE**; HTML/JS
  app that "mimics 2-step prop firm evaluations."
- **plutustradebase/awesome-funded-trading** —
  https://github.com/plutustradebase/awesome-funded-trading — **no LICENSE** (only 5.6 KB
  README, mostly a link list; no simulator listed).
- **PropSim-Enterprise/awesome-prop-firms** —
  https://github.com/PropSim-Enterprise/awesome-prop-firms — **CC0-1.0**. Curated list of
  prop-firm software. Its "tools" section links LuxAlgo/prop-firm-sim, PropForge,
  prop-firm-rules and the pass-rates dataset — it is a good index of the field.
- **tylerscomic-lab** MCP suite — https://github.com/tylerscomic-lab/walkforward-validator-mcp
  and https://github.com/tylerscomic-lab/montecarlo-validator-mcp — both **MIT**. An
  "AlgoForge" suite of small MCP servers. **`montecarlo-validator-mcp`** exposes
  `monte_carlo_validate` (bootstrap 90% CI on per-trade EV, drawdown-path percentiles via
  reshuffling) and **`prop_firm_pass_probability`**, plus a **`risk_geometry_comparator`**
  that "ranks multiple win-rate/risk-reward geometries by simulated pass rate". **`walkforward-validator-mcp`** exposes `walk_forward_efficiency`,
  `parameter_stability_score`, `lock_after_wfo_check` (audits whether parameters were
  genuinely locked after WFO or quietly re-tuned against the "OOS" data) and a WFO window
  generator. **These are separate servers**: the walk-forward tools and the Monte-Carlo
  pass-probability tool live in different repos and are not composed into one pipeline.
  I did not read their source, so I cannot state their internals beyond the README.

### 14. Rules *datasets* (no simulator, but reusable as the ruleset axis)

| Repo | URL | License |
|---|---|---|
| kburrus64-max/prop-firm-rules | https://github.com/kburrus64-max/prop-firm-rules | MIT (FloorGuard) |
| plutustradebase/prop-firm-rules | https://github.com/plutustradebase/prop-firm-rules | CC-BY-4.0 |
| powerFC/propfirm-rules-dataset | https://github.com/powerFC/propfirm-rules-dataset | CC-BY-4.0 |
| propfirmvs/prop-firm-rules-dataset | https://github.com/propfirmvs/prop-firm-rules-dataset | CC-BY-4.0 |
| PropSim-Enterprise/prop-firm-pass-rates | https://github.com/PropSim-Enterprise/prop-firm-pass-rates | CC-BY-4.0 |

`kburrus64-max/prop-firm-rules` is the most useful: sourced rules for FTMO, FundedNext,
The5ers, E8, FundingPips, Alpha Capital, Blue Guardian, Topstep and Apex, every rule
linking to the firm's own page with a `lastVerified` date, unconfirmed fields flagged
`verified: false`. Ships a zero-dependency JS **and** Python drawdown calculator, a REST
API and a no-auth MCP server.

`PropSim-Enterprise/prop-firm-pass-rates` is the only **empirical** pass-rate dataset
found. It publishes weekly, from real live accounts on the PropSim platform:
**phase-1 clear rate ~13.1–14.1%, phase-2 ~33–34.9%**, days-to-clear p25/median/p75
(4.5 / 7.3–7.4 / 13.8), median account size $100,000, median profit split 80%, median
max duration 45 days, win rate ~56% overall vs ~71.5% for clearers vs ~46% for failures,
payoff ratio ~1.4–1.8 for clearers vs ~0.8 for failures, median lifespan ~10.5 days,
retry rate ~84%. It also reports **failure causes: 54% of failed accounts breach the
daily loss limit**, and a `ruleEfficacy` field comparing relaxed vs tight settings.

> **This is the single most valuable external validation target for your project.** Every
> simulator above produces pass rates in the 20–95% range for plausible traders. Real
> PropSim accounts clear phase 1 at **~14%**. That gap is the thing your report should
> address explicitly — most likely because the simulators assume i.i.d. trades and a
> coherent strategy, while real accounts contain the leveraged blow-ups that breach the
> daily limit 54% of the time.

---

## Part 3 — The specific question: has anyone published purged walk-forward + bootstrap Monte Carlo for trading challenges?

**No — not as a single pipeline. Verified by reading source, not inferring from READMEs.**

| Repo | Purged walk-forward? | Bootstrap MC? | Composed? |
|---|---|---|---|
| dsilva42/prop-risk-geometry | **No** — `PurgedKFold` raises `NotImplementedError` | Yes (per-trade + per-day block) | **No** — MC runs on full in-sample output |
| LuxAlgo/prop-firm-sim | **No** — grep for walkforward/purge/embargo returns nothing | Yes (stationary block) | **No** |
| lhenriet-bit/prop-odds | No | No (i.i.d. Bernoulli) | No |
| Makeph/prop-ev | No | No (Gaussian) | No |
| gabrielee5/prop-firm-simulator | No | No (Bernoulli) | No |
| MaverickHQ/prop-challenge-lab | Rolling WFO? No — has **block bootstrap CI** and start-date MC | Yes (`block_bootstrap_ci`, `optimal_block_length`) | **No** — and it is EOD-mark-based, so not intraday-trailing |
| greybeard-code/gbBacktester | **Rolling WFO yes**, but **no purge/embargo** | Yes (circular block, auto-selected by autocorrelation) | **No** — `walkforward.py` has zero MC references; the MC is not fed by OOS output |
| ranjeet867/Metatrader | WFO yes, no purge | Yes (bootstrap from OOS R-distribution) | Not verified |
| tylerscomic-lab MCP suite | WFO in one server, MC in another | Bootstrap + reshuffle | **No** — separate repos |

**Every project splits it.** The two halves exist in adjacent repos (gbBacktester,
MaverickHQ) and in adjacent repos by the same author (tylerscomic-lab), but in **no
project** do purged OOS results feed a bootstrap Monte Carlo pass-rate estimate, and in
**no project** is the block-bootstrap block length chosen by an autocorrelation test
*and* the MC fed by purged OOS trades. dsilva42 is the only project with a `PurgedKFold`
class at all, and it is an explicit stub.

**Conclusion: your combination is genuinely novel.** It is also the correct combination
on the merits, for a reason the prior art supports: gbBacktester and MaverickHQ both
demonstrate *why* block bootstrap is required (streaks breach trailing floors), and
edgeveris.com demonstrates *why* the i.i.d. assumption inflates pass rates. Your
pipeline — purged OOS to select what is honest, then block bootstrap over the surviving
trades to estimate the pass rate — is the only construction in the set that does both.

---

## Part 4 — Synthesis: what to reuse, what to avoid, what is genuinely new

### Reuse (MIT/Apache-2.0 — copy freely with attribution)

1. **LuxAlgo's rule-semantics enums and `RuleState` class (MIT).** The four max-loss
   modes, the four daily-loss switches, and the breach-attribution rule ("the higher
   floor was crossed first") are the most valuable thing in the entire prior-art set.
   Port `packages/core/src/engine/rules.ts` and `spec/challenge.ts` to Python. This is
   what turns "trailing drawdown" from a boolean into a real model.
2. **LuxAlgo's consistency-rule rational-trader model (MIT).** It answers the question
   nobody else asks: consistency rules *add days at risk*, and those extra days are
   sometimes the days the trailing floor gets hit. Makeph's deferral (`continue`) is a
   simpler version of the same idea; LuxAlgo's version is the complete one.
3. **LuxAlgo's per-path seed derivation (`pathSeed(masterSeed, pathIndex)`) for common
   random numbers (MIT).** Makes risk sweeps smooth and the argmax stable instead of
   Monte Carlo noise. Cheap to implement, high value.
4. **MaverickHQ's `block_bootstrap_ci` + `optimal_block_length` + the two property tests
   (Apache-2.0).** `block=1` must reduce *exactly* to the plain bootstrap; the block
   bootstrap must be *wider* on a serially correlated series. Those are the two tests that
   make a block bootstrap trustworthy, and they are ready-made.
5. **MaverickHQ's dated rule profiles + `assert_fresh` (Apache-2.0).** Rulesets are
   *inputs*, versioned, dated, with a source, and stale ones raise rather than warn.
   This is exactly the "measured vs modelled" discipline you already apply to surface
   axes, applied to the ruleset axis. **"A rule that silently becomes zero is a rule that
   never binds"** is the sentence to steal.
6. **gbBacktester's autocorrelation-triggered method switch (MIT).** Compute max
   |autocorr| over lags 1..10; if > 0.2, i.i.d. resampling is invalid — switch to
   circular block bootstrap. `block = max(2, round(sqrt(n)))` as the default.
7. **PropSim's `intra_mdd` (MIT) and the three-bug post-mortem (MIT).** The intra-trade
   excursion problem is the hardest part of trailing-drawdown modelling and the prior art
   is honest that no MAE/MFE pair can solve it. `intra_mdd` (fall from the *running* peak
   inside the trade) is the closest available approach. Read the post-mortem before
   trusting any simulator, including yours.
8. **gabrielee5's Gambler's Ruin closed form (MIT) and the headroom-based risk capping
   (MIT).** `P = MaxLoss/(MaxLoss+Target)` for symmetric walks and
   `P = (1−r^a)/(1−r^(a+b))` for asymmetric ones are the only analytic cross-checks in
   the set. Use them as sanity bounds on the MC.
9. **prop-odds' failure decomposition and the risk cliff (MIT).** `fail_daily` /
   `fail_total` / `timeout` separately; days-to-pass computed over passing paths only.
   The risk-cliff discontinuity (1.00% → 93.9%, 1.75% → 42.5%) is the most practically
   useful single finding in any repo here.
10. **kburrus64-max/prop-firm-rules dataset (MIT) or the CC-BY-4.0 datasets.** A sourced,
    dated ruleset table to drive the ruleset axis of your surfaces. Prefer MIT if you want
    zero attribution obligations.
11. **MaverickHQ's calibration and power gates (Apache-2.0).** "A harness that cannot
    find a planted edge cannot be trusted to report its absence either." Dead-world and
    planted-edge controls; `n_eff = n/(1+(m−1)·rho)`; refuse underpowered runs.

### Avoid (licensing or correctness)

1. **dsilva42/prop-risk-geometry: do not copy any code.** No license file = all rights
   reserved. You may read it and cite it. Your `prop_lab/firm/monte_carlo.py` docstring
   already credits it as the populariser of the bootstrap approach — **keep that credit
   and keep the code yours**. If you want its per-day block bootstrap, write it yourself
   (MaverickHQ's and gbBacktester's are Apache-2.0/MIT and do the same job properly).
2. **GithinjiHans/PropFirmEval, Lihui-03/PropFirm, GT-mandem/backtester,
   Cooderhome/PFCS, plutustradebase/awesome-funded-trading** — no license, nothing to
   reuse.
3. **Do not copy the smoothed 3-D surface** from dsilva42's `plots/surface_3d.py`. It
   applies a 3×3 kernel 4× for "visual quality", so the plotted surface is not the
   computed surface. Your surfaces should show the computed values.
4. **Do not copy LuxAlgo's project name or branding.** `TRADEMARKS.md` reserves both. MIT
   covers the code, not the name.
5. **Do not use LuxAlgo's rules directory API as a dependency.** It is a proprietary
   runtime API. Ship your own dated ruleset files (MaverickHQ profile pattern) or use an
   MIT/CC dataset.
6. **Avoid the "i.i.d. coin-flip + linear-looking risk sweep" pattern** shared by
   `prop-odds`, `prop-ev` and `gabrielee5`. It is fast but it misses the streaks that
   breach trailing floors, and it will systematically overstate pass rates.
7. **Do not copy the EV=0 constraint as your primary frame** (gabrielee5). It answers
   "what does a zero-edge strategy do?", which is a different question from "what does my
   edge do?".

### Genuinely new — no prior art found

1. **Purged walk-forward (purge + embargo, rolling training window) whose OOS trades
   feed a bootstrap Monte Carlo pass-rate estimator.** No project in the set does this.
   dsilva42 has the stub; gbBacktester has the two halves in separate modules with no
   link; MaverickHQ has the block bootstrap but no OOS feed; LuxAlgo has neither.
   **This is your flagship differentiator and it is defensible as such.**
2. **A measured-vs-modelled provenance tag on every surface axis.** dsilva42's surfaces
   average across buckets and label nothing. Nobody else labels either. LuxAlgo has
   rule-level provenance (`directory` vs `directory+inferred` vs not-simulatable) but
   nothing for surface axes. Your `AXIS_MEASURED` / `AXIS_MODELLED` distinction has no
   precedent.
3. **An exact-dynamic-program golden reference for a Monte Carlo engine.** edgeveris.com
   publishes the exact DP for a $100k / ±$10k / 55% / 100-trade case
   (Passed 66.784%, Failed 8.978%, Unfinished 24.238%) and explicitly proposes it as a
   unit test for a Monte Carlo engine. Your `tests/test_golden_reference.py` already
   reproduces it. **Publish that test.** It is the strongest possible evidence that the
   engine is right, and no public simulator has one.
   (Source: https://edgeveris.com/blog/monte-carlo-prop-challenge-probability,
   fetched and verified 2026-11-03. It also independently recommends the same three
   things you already do: keep *unfinished* visible, use session/block resampling rather
   than i.i.d. trades, and report simulation error separately from source-data
   uncertainty.)
4. **A ruleset × risk-geometry surface with per-cell provenance.** dsilva42 has
   R:R × risk-geometry but with **hard-coded single rules** and no ruleset dimension.
   LuxAlgo has multi-firm rulesets but no 2-D surfaces. Nobody crosses ruleset against
   risk geometry.
5. **Reconciling simulator output against empirical platform data.** The
   PropSim-Enterprise/prop-firm-pass-rates dataset (CC-BY-4.0) publishes real phase-1
   clear rates of ~13–14% and "54% of failed accounts breach the daily loss limit".
   **No simulator in the prior-art set is validated against any empirical pass-rate
   data.** Doing that — and reporting the gap honestly — would be a genuinely new
   contribution and would be the single most valuable thing your release could add.

### Recommended citation posture for the release

Cite by name and URL, with license and the specific mechanism borrowed. The credits that
matter most, in order: **LuxAlgo (MIT)** for rule semantics and the consistency model;
**MaverickHQ (Apache-2.0)** for the block bootstrap, dated profiles and calibration gates;
**greybeard-code/gbBacktester (MIT)** for autocorrelation-selected block length and the
MAE/MFE breach accounting; **jalv92/PropSim (MIT)** for the intraday-excursion
post-mortem; **lhenriet-bit/prop-odds (MIT)** for the risk cliff and failure
decomposition; **Makeph/prop-ev (MIT)** for the EV framing; **gabrielee5 (MIT)** for the
Gambler's Ruin closed form. **Credit dsilva42/prop-risk-geometry as the origin of the
bootstrap-resampling idea — cited, not copied.** Do not copy its code, and do not omit
the attribution.

---

## Appendix — every URL visited

**Verified reachable, content read in this session:**

- https://github.com/dsilva42/prop-risk-geometry (repo, README, LICENSE → 404,
  `simulation/engine.py`, `simulation/funded_phase.py`, `simulation/strategy_engine/mc.py`,
  `simulation/ml/cv.py`, `config.py`, `plots/heatmaps.py`, `plots/surface_3d.py`,
  `plots/ev_breakeven.py`)
- https://github.com/hunter547/prop-risk-geometry — **404 (gone)**
- https://github.com/hunter547 — account exists, 37 repos, none is prop-risk-geometry
- https://github.com/LuxAlgo/prop-firm-sim (repo, README, LICENSE, `docs/rule-semantics.md`,
  `packages/core/src/engine/{rules,simulate,attempt,trades,rng}.ts`,
  `packages/core/src/bootstrap/block.ts`, `packages/core/src/analyze/{sensitivity,optimalRisk}.ts`,
  `packages/core/src/spec/challenge.ts`, issues #5 and #6)
- https://github.com/lhenriet-bit/prop-odds (repo, README, LICENSE,
  `src/prop_odds/{simulate,model,optimize}.py`, `src/prop_odds/presets/two-step-classic.json`)
- https://github.com/Makeph/prop-ev (repo, README, LICENSE, `propev/{core,firms}.py`)
- https://github.com/gabrielee5/prop-firm-simulator (repo, README, LICENSE, `docs/PRIMER.md`,
  `src/{simulator,models,unified_simulator,funded_simulator,utils}.py`, `config.yaml`)
- https://github.com/MaverickHQ/prop-challenge-lab (repo, README, LICENSE, NOTICE,
  `occams/{sim,rules,profiles,stats,harness}.py`, `tests/test_block_bootstrap.py`)
- https://github.com/jalv92/PropSim (repo, README, LICENSE, `sim.py`, `prop_rules.py` referenced)
- https://github.com/greybeard-code/gbBacktester (repo, README, LICENSE,
  `backtester/montecarlo.py`, `backtester/walkforward.py`)
- https://github.com/ranjeet867/Metatrader (repo, README, LICENSE, `core/ftmo_simulator.py`)
- https://github.com/PropSim-Enterprise/prop-firm-rules (repo, README, LICENSE)
- https://github.com/PropSim-Enterprise/prop-firm-pass-rates (repo, README, LICENSE,
  `data/weekly-summary.csv`)
- https://github.com/PropSim-Enterprise/awesome-prop-firms (README, LICENSE)
- https://github.com/michaelsboost/PropForge (README, LICENSE.md)
- https://github.com/GithinjiHans/PropFirmEval (README; LICENSE → 404)
- https://github.com/Lihui-03/PropFirm (README; LICENSE → 404)
- https://github.com/GT-mandem/backtester (LICENSE → 404)
- https://github.com/Cooderhome/Prop-Firm-Challenge-Simulator-PFCS- (LICENSE → 404)
- https://github.com/plutustradebase/awesome-funded-trading (README; LICENSE → 404)
- https://github.com/plutustradebase/prop-firm-rules (LICENSE → CC-BY-4.0)
- https://github.com/powerFC/propfirm-rules-dataset (LICENSE → CC-BY-4.0)
- https://github.com/propfirmvs/prop-firm-rules-dataset (LICENSE.md → CC-BY-4.0)
- https://github.com/kburrus64-max/prop-firm-rules (README, LICENSE → MIT)
- https://github.com/tylerscomic-lab/walkforward-validator-mcp (README, LICENSE → MIT)
- https://github.com/tylerscomic-lab/montecarlo-validator-mcp (README, LICENSE → MIT)
- https://github.com/tylerscomic-lab/prop-rules-mcp (README, LICENSE → MIT)
- **this project** — public, `license: null` at the time of the review,
  0 stars, created 2026-11-03, default branch `master`
- https://edgeveris.com/blog/monte-carlo-prop-challenge-probability (full text read)
- https://github.com/topics/ftmo, https://github.com/topics/funded-account,
  https://github.com/topics/walk-forward-validation, https://github.com/topics/risk-calculator
- https://www.luxalgo.com/prop-firms (search snippets only)
- https://github.com/LabinatorSolutions/awesome-institutional-trading (search snippet only)

**Not verified / could not confirm:**

- The `montecarlo-validator-mcp` and `walkforward-validator-mcp` **source code** was not
  read — only their READMEs. Their internals are described from their own documentation,
  not from code inspection.
- `jalv92/PropSim`'s `prop_rules.py` and its `docs/research/prop-rules/` corpus were not
  read; the firm-rule table structure is described from `sim.py`'s usage of it.
- `ranjeet867/Metatrader` has 9786 files; only `core/ftmo_simulator.py` was read. Whether
  its walk-forward output feeds its Monte Carlo is **not verified**.
- `GithinjiHans/PropFirmEval` — only the README was read; the notebook was not executed
  or parsed.
- `Cooderhome/Prop-Firm-Challenge-Simulator-PFCS-` and `Lihui-03/PropFirm` — README only,
  and both are near-empty; no source was read.
- `github.com/gabrielefabietti/prop-firm-simulator` (the URL inside gabrielee5's README)
  was not checked — it is a different owner string and the repo may not exist under it.
