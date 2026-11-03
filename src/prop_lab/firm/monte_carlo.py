"""Bootstrap Monte Carlo for prop firm challenge simulation.

THE CORE RULE
--------------
The simulated outcome of every trade is drawn from the *empirical*
per-trade ``pnl_r`` array produced by the real backtest:

    pnl_r = float(pnl_r_array[rng.integers(0, len(pnl_r_array))])

Never ``rng.random() < win_rate``. Never ``rng.choice([-1.0, rr])``.

Textbook Monte Carlo for prop firm challenges assumes winners pay +2R and
losers pay -1R. That assumption destroys the actual win/loss size
distribution the strategy produced, and the result is that every strategy
scores roughly the same pass rate regardless of whether it has an edge.
Resampling the real distribution is what makes the number mean something.

This is the approach popularised by the prop-risk-geometry project
(originally github.com/hunter547/prop-risk-geometry, now
github.com/dsilva42/prop-risk-geometry), which in turn built on the
surface-plot line of work DeltaTrend Trading showed publicly.

What is honestly modelled here, and what is not:

Modelled
    - equity path from real per-trade R multiples
    - trailing drawdown following the equity peak
    - a daily loss limit, approximated with synthetic days
    - position sizing via the risk geometries

Not modelled
    - slippage and commission
    - news gaps and intrabar order of events
    - the firm's contract scaling plans
    - payout delays and the calendar reality of the funded account

VALIDATED AGAINST PUBLISHED EXACT VALUES
----------------------------------------
The engine is checked against a published reference with exact,
non-sampled answers. The reference case
(https://edgeveris.com/blog/monte-carlo-prop-challenge-probability,
checked 2026-11-03) is a $100,000 account, $10,000 target, $10,000
failure boundary, $1,000 fixed risk per trade, 55 percent win rate and a
100 trade horizon. Its exact dynamic program gives:

    Passed      66.784 percent
    Failed       8.978 percent
    Unfinished  24.238 percent

``tests/test_golden_reference.py`` reproduces the dynamic program, asserts
the published numbers, and asserts that this engine converges on them.

Writing that test found two real defects in this engine, both now fixed:

1. The drawdown breach was checked with a strict comparison
   (``equity < threshold``), so a path that landed exactly on the
   threshold survived and could step past it on the next trade. The
   boundary is now inclusive (``equity <= threshold``).
2. There was no risk geometry for a fixed percentage of the INITIAL
   balance, which is how most prop firm traders actually size and the
   basis the reference case uses. ``frac_1pct_initial`` and
   ``frac_2pct_initial`` were added.

The reference also makes three points that this module follows:

- Pass, fail and UNFINISHED are a complete partition. Dropping the
  unfinished attempts silently changes the question, so all three are
  reported and they sum to one.
- The pass rate among RESOLVED attempts is larger than the pass rate
  over all attempts, and the two must not be confused.
- Risk basis must be stated. Fixed-cash risk and current-equity risk
  reach different account values and produce different pass rates, so
  they are separate geometries with separate names.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .rules import FirmRules

__all__ = [
    "BootstrapTrade",
    "PathResult",
    "MCSummary",
    "RISK_FN_MAP",
    "RISK_GEOMETRIES",
    "risk_max_size",
    "risk_frac_2pct",
    "risk_frac_1pct_initial",
    "risk_frac_2pct_initial",
    "risk_dd_frac_40pct",
    "risk_anti_martingale",
    "risk_floor_aware",
    "simulate_challenge",
    "monte_carlo_challenge",
    "wilson_interval",
    "bootstrap_trade_array",
    "bootstrap_day_blocks",
    "bootstrap_stationary",
    "group_by_session",
    "RESAMPLING_METHODS",
    "RESAMPLING_IID",
    "RESAMPLING_BLOCK",
    "RESAMPLING_STATIONARY",
]

# The two resampling methods and what each one assumes.
#
# 'iid' draws individual trades with replacement. It preserves the exact
# marginal distribution of per-trade R multiples, but it destroys
# serial dependence, and serial dependence is exactly what makes a
# drawdown deep. Losses arrive in clusters: a bad regime produces several
# losing trades in a row, and a trailing drawdown is a path-dependent
# barrier, so whether it is hit depends on the ORDER of trades, not just
# the multiset. An i.i.d. bootstrap therefore systematically UNDERSTATES
# the risk, and the understatement grows with the clustering in the
# original sample.
#
# 'block' resamples whole trading DAYS as units, which preserves the
# within-day correlation that a regime-driven strategy produces. This is
# the method the prop-risk-geometry project (originally
# github.com/hunter547/prop-risk-geometry, now
# github.com/dsilva42/prop-risk-geometry) uses for its Lab Model, and it
# is the default here.
#
# Block bootstrap is not free of caveats: too short a block and you drift
# back to i.i.d.; too long a block and there are too few independent
# blocks to draw from, which makes the simulation look precise while
# still resting on little evidence. The block length is a modelling
# choice and must be reported alongside the result.
RESAMPLING_IID = "iid"
RESAMPLING_BLOCK = "block"
RESAMPLING_STATIONARY = "stationary"
RESAMPLING_METHODS = (RESAMPLING_IID, RESAMPLING_BLOCK, RESAMPLING_STATIONARY)


# ----------------------------------------------------------------------
# Risk geometries -- position sizing functions
# ----------------------------------------------------------------------
def risk_max_size(equity: float, peak: float, atr: float, rules: FirmRules | None = None) -> float:
    """Fixed risk: $500 per trade regardless of the account state."""
    return 500.0


def risk_frac_2pct(equity: float, peak: float, atr: float, rules: FirmRules | None = None) -> float:
    """Risk 2 percent of current equity."""
    return equity * 0.02


def risk_frac_1pct_initial(
    equity: float, peak: float, atr: float, rules: FirmRules | None = None
) -> float:
    """Risk 1 percent of the INITIAL balance, never compounding.

    This is how most prop firm traders actually size: the loss limit is a
    fixed dollar amount, so the risk per trade is a fixed dollar amount
    too. A report must never describe fixed-cash risk and current-equity
    risk with the same number without saying which basis is used,
    because the two produce different reachable account values and
    therefore different pass probabilities.
    """
    start = rules.starting_balance if rules is not None else 50_000.0
    return start * 0.01


def risk_frac_2pct_initial(
    equity: float, peak: float, atr: float, rules: FirmRules | None = None
) -> float:
    """Risk 2 percent of the INITIAL balance, never compounding.

    This is how most prop firm traders actually size: the loss limit is a
    fixed dollar amount, so the risk per trade is a fixed dollar amount
    too. A report must never describe fixed-cash risk and current-equity
    risk with the same number without saying which basis is used,
    because the two produce different reachable account values and
    therefore different pass probabilities.
    """
    start = rules.starting_balance if rules is not None else 50_000.0
    return start * 0.02


def risk_dd_frac_40pct(equity: float, peak: float, atr: float, rules: FirmRules | None = None) -> float:
    """Risk 40 percent of the room left to the drawdown floor."""
    if rules is None:
        floor = equity - 2_000.0
        room = equity - floor
        return max(room * 0.40, 25.0) if room > 0 else 25.0
    room = equity - rules.floor
    return max(room * 0.40, 25.0) if room > 0 else 25.0


def risk_anti_martingale(equity: float, peak: float, atr: float, rules: FirmRules | None = None) -> float:
    """Scale up with profit above the starting balance."""
    base = 400.0
    start = rules.starting_balance if rules is not None else 50_000.0
    profit = equity - start
    return base + profit * 0.15 if profit > 0 else base


def risk_floor_aware(equity: float, peak: float, atr: float, rules: FirmRules | None = None) -> float:
    """Shrink the risk as equity approaches the drawdown floor."""
    if rules is None:
        floor = equity - 2_000.0
        total_room = 2_000.0
    else:
        floor = rules.floor
        total_room = rules.starting_balance - floor
    if total_room <= 0:
        return 25.0
    frac = max((equity - floor) / total_room, 0.05)
    return frac * 800.0


RISK_FN_MAP: dict[str, Callable[..., float]] = {
    "max_size": risk_max_size,
    "frac_2pct": risk_frac_2pct,
    "frac_1pct_initial": risk_frac_1pct_initial,
    "frac_2pct_initial": risk_frac_2pct_initial,
    "dd_frac_40pct": risk_dd_frac_40pct,
    "anti_martingale": risk_anti_martingale,
    "floor_aware": risk_floor_aware,
}

RISK_GEOMETRIES: list[str] = list(RISK_FN_MAP)


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
@dataclass
class BootstrapTrade:
    """One real trade expressed in R units, ready for resampling.

    ``pnl_r`` is the realised P&L divided by the dollar risk taken.
    Positive means a winner, and its magnitude is the reward-to-risk
    actually achieved (2.0 for a clean 2R winner). Negative means a
    loser, normally -1.0 when the stop was hit.
    """

    pnl_r: float
    entry_ts: int = 0
    side: str = "LONG"


@dataclass
class PathResult:
    """One simulated challenge attempt."""

    outcome: str = "ongoing"  # 'pass' | 'fail' | 'ongoing'
    final_equity: float = 0.0
    n_trades: int = 0
    breach_kind: str | None = None  # 'max_dd' | 'daily_loss' | 'timeout'
    equity_path: list[float] = field(default_factory=list)
    day_pnls: list[float] = field(default_factory=list)


@dataclass
class MCSummary:
    """Aggregate result of N Monte Carlo challenge attempts."""

    n_attempts: int
    pass_rate: float
    fail_rate: float
    timeout_rate: float
    mean_final_equity: float
    p5_final_equity: float
    p95_final_equity: float
    mean_trades_to_pass: float
    pass_rate_ci_low: float
    pass_rate_ci_high: float
    n_trades_sample: int
    risk_geometry: str = ""
    resampling: str = "block"
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Plain-text summary of the Monte Carlo result."""
        return (
            f"risk geometry : {self.risk_geometry}\n"
            f"resampling    : {self.resampling}\n"
            f"attempts      : {self.n_attempts}\n"
            f"pass rate     : {self.pass_rate:.1%} "
            f"(95% CI {self.pass_rate_ci_low:.1%} - {self.pass_rate_ci_high:.1%})\n"
            f"fail rate     : {self.fail_rate:.1%}\n"
            f"timeout rate  : {self.timeout_rate:.1%}\n"
            f"mean equity   : ${self.mean_final_equity:,.0f}\n"
            f"p5 / p95      : ${self.p5_final_equity:,.0f} / ${self.p95_final_equity:,.0f}\n"
            f"trades to pass: {self.mean_trades_to_pass:.1f}\n"
            f"sample trades : {self.n_trades_sample}"
        )


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because a pass rate near 0 or
    1 would otherwise produce an interval that runs outside [0, 1], which
    is exactly the regime where prop firm pass rates live.

    Args:
        successes: number of successful attempts.
        n: total attempts.
        z: z-score for the confidence level (1.96 for 95 percent).

    Returns:
        A ``(low, high)`` tuple of proportions.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    low = max(0.0, centre - margin)
    high = min(1.0, centre + margin)
    # Guard against floating point drift at the boundaries, where the
    # interval should touch 0 or 1 exactly.
    if successes == 0:
        low = 0.0
    if successes == n:
        high = 1.0
    return (low, high)


# ----------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------
def group_by_session(
    pnl_r_array: np.ndarray, session_dates: list[str] | None = None
) -> list[np.ndarray]:
    """Split the trade array into per-session blocks.

    Args:
        pnl_r_array: 1-D array of per-trade ``pnl_r`` values.
        session_dates: optional parallel list of session labels, one per
            trade. When ``None`` every trade lands in a single block.

    Returns:
        A list of numpy arrays, one per session, in the order the sessions
        first appear.
    """
    pnl_r_array = np.asarray(pnl_r_array, dtype=float)
    if session_dates is None:
        return [pnl_r_array] if pnl_r_array.size else []

    buckets: dict[str, list[float]] = {}
    for value, session in zip(pnl_r_array, session_dates):
        buckets.setdefault(session, []).append(float(value))
    return [np.array(vals, dtype=float) for vals in buckets.values()]


def bootstrap_trade_array(
    pnl_r_array: np.ndarray,
    rng: np.random.Generator,
    n_trades: int | None = None,
) -> np.ndarray:
    """Draw individual trades with replacement (i.i.d. bootstrap).

    Args:
        pnl_r_array: the real per-trade ``pnl_r`` values.
        rng: numpy Generator.
        n_trades: how many trades to draw. Defaults to the size of the
            input array, so the simulated path has the same length as the
            original.

    Returns:
        An array of ``n_trades`` resampled values.
    """
    pnl_r_array = np.asarray(pnl_r_array, dtype=float)
    if pnl_r_array.size == 0:
        raise ValueError("pnl_r_array is empty")
    n = n_trades if n_trades is not None else pnl_r_array.size
    idx = rng.integers(0, pnl_r_array.size, size=n)
    return pnl_r_array[idx]


def bootstrap_stationary(
    pnl_r_array: np.ndarray,
    rng: np.random.Generator,
    n_trades: int | None = None,
    mean_block_length: float = 5.0,
) -> np.ndarray:
    """Stationary bootstrap of Politis and Romano (1994).

    Draws contiguous blocks whose lengths follow a geometric
    distribution with mean ``mean_block_length``. Because every position
    in the resampled series is statistically equivalent, the result stays
    stationary, which a fixed-length moving-block bootstrap does not.

    The mean block length is the one free parameter and it is a real
    modelling choice. Too short and the resampling degenerates towards
    i.i.d., which destroys the loss streaks that make drawdown risk
    dangerous. Too long and there are too few effectively independent
    blocks, so the simulation looks precise while resting on very
    little evidence. The value used is always reported alongside the
    result.

    Args:
        pnl_r_array: the real per-trade ``pnl_r`` values.
        rng: numpy Generator.
        n_trades: how many trades to draw. Defaults to the size of the
            input array.
        mean_block_length: expected block length in trades.

    Returns:
        An array of ``n_trades`` resampled values.
    """
    pnl_r_array = np.asarray(pnl_r_array, dtype=float)
    if pnl_r_array.size == 0:
        raise ValueError("pnl_r_array is empty")
    if mean_block_length < 1:
        raise ValueError("mean_block_length must be at least 1")

    n = n_trades if n_trades is not None else pnl_r_array.size
    size = pnl_r_array.size
    p = 1.0 / float(mean_block_length)

    out: list[float] = []
    # Start each block at a uniformly random index so the resampled
    # series is stationary rather than cyclically anchored.
    idx = int(rng.integers(0, size))
    while len(out) < n:
        length = int(rng.geometric(p))
        for _ in range(length):
            out.append(float(pnl_r_array[idx]))
            idx = (idx + 1) % size
            if len(out) >= n:
                break
    return np.array(out[:n], dtype=float)


def bootstrap_day_blocks(
    day_blocks: list[np.ndarray],
    rng: np.random.Generator,
    n_days: int | None = None,
) -> np.ndarray:
    """Resample whole trading days and concatenate them (block bootstrap).

    Resampling at the day level preserves the correlation between trades
    inside a session. When a strategy's trades share a regime framing,
    several of them tend to fail together, and that joint failure is what
    pushes a trailing drawdown over its threshold. An i.i.d. bootstrap
    breaks those clusters apart and flatters the result.

    Args:
        day_blocks: list of per-session trade arrays.
        rng: numpy Generator.
        n_days: how many days to draw. Defaults to the number of blocks.

    Returns:
        A flat array of the resampled trades, day by day.
    """
    if not day_blocks:
        raise ValueError("day_blocks is empty")
    n = n_days if n_days is not None else len(day_blocks)
    picks = rng.integers(0, len(day_blocks), size=n)
    return np.concatenate([day_blocks[p] for p in picks])


# ----------------------------------------------------------------------
# Single path simulation
# ----------------------------------------------------------------------
def simulate_challenge(
    pnl_r_array: np.ndarray,
    rules: FirmRules,
    *,
    risk_geometry: str = "frac_2pct",
    atr: float = 0.0,
    rng: np.random.Generator | None = None,
    max_trades: int = 2000,
    trades_per_day: int = 3,
    max_days: int | None = None,
    resampling: str = RESAMPLING_BLOCK,
    day_blocks: list[np.ndarray] | None = None,
    mean_block_length: float = 5.0,
) -> PathResult:
    """Simulate ONE challenge attempt by bootstrapping the real trades.

    Args:
        pnl_r_array: 1-D array of per-trade ``pnl_r`` values from a real
            backtest. Positive is a winner, negative a loser.
        rules: the firm ruleset to simulate against.
        risk_geometry: key into :data:`RISK_FN_MAP`.
        atr: per-trade ATR, passed through to the risk function. Unused by
            the built-in geometries but available for custom ones.
        rng: numpy Generator. Seeded for reproducibility.
        max_trades: hard cap on trades in one attempt.
        trades_per_day: how many bootstrapped trades make up a synthetic
            trading day. This is an approximation: the bootstrap has no
            real session boundaries, so the daily loss limit is applied to
            groups of this many trades.
        max_days: optional cap on synthetic days (the evaluation window).
        resampling: ``'block'`` (default) resamples whole trading days and
            preserves within-day correlation; ``'iid'`` draws individual
            trades and understates risk when losses cluster.
        day_blocks: per-session trade arrays for the block bootstrap. When
            ``None`` and ``resampling='block'``, the whole trade array is
            treated as a single block, which is equivalent to i.i.d. and is
            reported as a warning.

    Returns:
        A :class:`PathResult`.

    Raises:
        ValueError: on an unknown risk geometry or an empty trade array.
    """
    if risk_geometry not in RISK_FN_MAP:
        raise ValueError(
            f"Unknown risk_geometry {risk_geometry!r}. "
            f"Available: {', '.join(RISK_FN_MAP)}."
        )
    if resampling not in RESAMPLING_METHODS:
        raise ValueError(
            f"Unknown resampling method {resampling!r}. "
            f"Available: {', '.join(RESAMPLING_METHODS)}."
        )
    pnl_r_array = np.asarray(pnl_r_array, dtype=float)
    if pnl_r_array.size == 0:
        raise ValueError(
            "pnl_r_array is empty. Run a backtest first: "
            "trades = run_backtest(strategy, df); "
            "pnl = np.array([t.pnl_r for t in trades])"
        )
    if rng is None:
        rng = np.random.default_rng()

    # -- build the resampled trade sequence ---------------------------
    if resampling == RESAMPLING_BLOCK:
        if day_blocks and len(day_blocks) > 1:
            sequence = bootstrap_day_blocks(day_blocks, rng)
        else:
            # A single block carries no extra dependence information.
            sequence = bootstrap_trade_array(pnl_r_array, rng)
    elif resampling == RESAMPLING_STATIONARY:
        sequence = bootstrap_stationary(
            pnl_r_array, rng, mean_block_length=mean_block_length
        )
    else:
        sequence = bootstrap_trade_array(pnl_r_array, rng)

    risk_fn = RISK_FN_MAP[risk_geometry]
    start = rules.starting_balance
    target = rules.target_balance
    floor = rules.floor
    max_dd = rules.max_drawdown

    equity = float(start)
    peak = float(start)
    threshold = floor
    path = [equity]
    day_pnls: list[float] = []
    day_running = 0.0
    trades_in_day = 0
    n_seq = sequence.size

    # The effective profit target can RISE when the consistency rule
    # demands it. Topstep's rule is that the best single day must be at
    # or below 55 percent of the profit target, so a big day raises the
    # target to best_day / ratio rather than failing the account.
    effective_target = target

    for t in range(1, max_trades + 1):
        if max_days is not None and len(day_pnls) >= max_days:
            return PathResult(
                equity_path=path,
                outcome="ongoing",
                final_equity=equity,
                n_trades=t - 1,
                breach_kind="timeout",
                day_pnls=day_pnls,
            )
        if t > n_seq:
            # The resampled sequence ran out. Draw more i.i.d. trades so
            # the path can continue rather than truncating the attempt.
            extra = bootstrap_trade_array(pnl_r_array, rng, n_trades=n_seq)
            sequence = np.concatenate([sequence, extra])
            n_seq = sequence.size

        # -- size the position ---------------------------------------
        risk_dollars = risk_fn(equity, peak, atr, rules)
        # Never risk more than the daily loss limit allows.
        if rules.max_daily_loss > 0:
            risk_dollars = min(risk_dollars, rules.max_daily_loss)
        # Never risk more than the remaining room to the floor.
        room = equity - threshold
        if room > 0:
            risk_dollars = min(risk_dollars, room)
        risk_dollars = max(risk_dollars, 25.0)

        # -- bootstrap the trade outcome ------------------------------
        pnl_r = float(sequence[t - 1])
        pnl = risk_dollars * pnl_r

        equity += pnl
        peak = max(peak, equity)
        day_running += pnl
        trades_in_day += 1
        path.append(equity)

        # -- rolling drawdown threshold -------------------------------
        if rules.trailing_drawdown:
            threshold = max(floor, peak - max_dd)
        else:
            threshold = floor

        # -- close the synthetic day ----------------------------------
        if trades_in_day >= trades_per_day:
            day_pnls.append(day_running)
            day_running = 0.0
            trades_in_day = 0

            # -- consistency rule: the target can RISE ---------------
            # The best day's profit must stay at or below the ratio of
            # the profit target. If it does not, the firm raises the
            # target to best_day / ratio instead of failing the
            # account. Applying this here means a strategy that earns
            # everything in one big day has to earn more overall.
            #
            # This is the EVALUATION-stage rule. APEX has none on the
            # evaluation, it applies consistency only at the payout
            # stage, so its evaluation pass rate is unaffected.
            if rules.eval_consistency_ratio:
                best_day = max(day_pnls)
                required_target = best_day / rules.eval_consistency_ratio
                if required_target > effective_target - start:
                    effective_target = start + required_target

        # -- pass check ------------------------------------------------
        # The minimum trading days must be met before the evaluation can
        # pass, even when the profit target has already been reached.
        days_elapsed = len(day_pnls)
        if equity >= effective_target and days_elapsed >= rules.min_trading_days:
            return PathResult(
                equity_path=path,
                outcome="pass",
                final_equity=equity,
                n_trades=t,
                breach_kind=None,
                day_pnls=day_pnls,
            )

        # -- breach checks ---------------------------------------------
        # Touching the boundary counts as a breach. The drawdown limit is
        # "the account may not exceed this drawdown", so landing exactly
        # on the threshold is already at the limit, not safely inside it.
        # Using a strict comparison here would let a path survive on the
        # boundary and then step past it on the next trade, which
        # understates the failure rate.
        if equity <= threshold:
            return PathResult(
                equity_path=path,
                outcome="fail",
                final_equity=equity,
                n_trades=t,
                breach_kind="max_dd",
                day_pnls=day_pnls,
            )

        # -- daily loss limit ------------------------------------------
        # Note the semantics: at APEX and Topstep, breaching the daily
        # loss limit ends the trading session, it does NOT fail the
        # account. The account only fails if the trailing drawdown is
        # touched. So this is only a failure when the ruleset says so.
        if (
            rules.dll_is_failure
            and rules.max_daily_loss > 0
            and trades_in_day == 0
            and day_pnls
            and day_pnls[-1] <= -rules.max_daily_loss
        ):
            return PathResult(
                equity_path=path,
                outcome="fail",
                final_equity=equity,
                n_trades=t,
                breach_kind="daily_loss",
                day_pnls=day_pnls,
            )

    return PathResult(
        equity_path=path,
        outcome="ongoing",
        final_equity=equity,
        n_trades=max_trades,
        breach_kind="timeout",
        day_pnls=day_pnls,
    )


# ----------------------------------------------------------------------
# Monte Carlo over many attempts
# ----------------------------------------------------------------------
def monte_carlo_challenge(
    pnl_r_array: np.ndarray,
    rules: FirmRules,
    *,
    risk_geometry: str = "frac_2pct",
    n_attempts: int = 10_000,
    seed: int = 42,
    atr: float = 0.0,
    max_trades: int = 2000,
    trades_per_day: int = 3,
    max_days: int | None = None,
    resampling: str = RESAMPLING_BLOCK,
    session_dates: list[str] | None = None,
    compare_iid: bool = False,
    mean_block_length: float = 5.0,
) -> MCSummary:
    """Run N simulated challenge attempts and summarise the outcome.

    Args:
        pnl_r_array: per-trade ``pnl_r`` values from the real backtest.
        rules: the firm ruleset.
        risk_geometry: key into :data:`RISK_FN_MAP`.
        n_attempts: number of simulated attempts. 10,000 is the default
            because a 500-attempt run gives a pass-rate confidence
            interval roughly +/-4 percentage points wide, which is too
            coarse to distinguish the geometries.
        seed: RNG seed. The same seed and the same inputs always produce
            the same numbers.
        atr: passed to the risk function.
        max_trades: cap on trades per attempt.
        trades_per_day: trades per synthetic day.
        max_days: optional cap on synthetic days.
        resampling: ``'block'`` (default) or ``'iid'``. See the module
            docstring for why block is the honest default.
        session_dates: optional per-trade session labels. Required for the
            block bootstrap to preserve anything; without it the block
            bootstrap degenerates to i.i.d. and a warning says so.
        compare_iid: when True, also run the i.i.d. bootstrap and report
            the pass-rate gap in the warnings. This is a diagnostic: a
            large gap means the i.i.d. number is flattering the strategy
            and the loss clustering in the sample matters.

    Returns:
        An :class:`MCSummary`.
    """
    pnl_r_array = np.asarray(pnl_r_array, dtype=float)
    if pnl_r_array.size == 0:
        raise ValueError("pnl_r_array is empty; run a backtest first")
    if resampling not in RESAMPLING_METHODS:
        raise ValueError(
            f"Unknown resampling method {resampling!r}. "
            f"Available: {', '.join(RESAMPLING_METHODS)}."
        )

    msg: list[str] = []
    n_sample = int(pnl_r_array.size)
    if n_sample < 30:
        msg.append(
            f"sample of only {n_sample} trades: the pass rate is low-confidence. "
            "Use at least 30 trades, and 100+ for a firm conclusion."
        )
    if n_sample < 10:
        msg.append(
            "fewer than 10 trades: treat this pass rate as illustrative only, "
            "not as an estimate of anything."
        )
    msg.append(
        "session boundaries are approximated: the bootstrap has no real calendar, "
        "so the daily loss limit is applied to synthetic days of "
        f"{trades_per_day} trades."
    )
    if n_attempts < 1000:
        msg.append(
            f"only {n_attempts} Monte Carlo attempts: the confidence interval is wide. "
            "Use 10,000 for a firm number."
        )

    day_blocks = group_by_session(pnl_r_array, session_dates)
    if resampling == RESAMPLING_BLOCK:
        if len(day_blocks) <= 1:
            msg.append(
                "no session labels were supplied, so the block bootstrap had a "
                "single block and is equivalent to the i.i.d. bootstrap. Pass "
                "session_dates to preserve the within-day loss clustering."
            )
        else:
            msg.append(
                f"block bootstrap over {len(day_blocks)} trading days: the "
                "within-day correlation between trades is preserved, so the "
                "pass rate reflects clustered losses rather than independent ones."
            )
    elif resampling == RESAMPLING_STATIONARY:
        msg.append(
            f"stationary bootstrap with mean block length {mean_block_length} "
            "trades (Politis and Romano 1994). The block length is a modelling "
            "choice: too short destroys loss streaks, too long leaves too few "
            "effectively independent blocks. Compare several defensible "
            "lengths rather than tuning it until the result looks attractive."
        )

    rng = np.random.default_rng(seed)
    outcomes: list[str] = []
    finals: list[float] = []
    trades_to_pass: list[int] = []

    for _ in range(n_attempts):
        res = simulate_challenge(
            pnl_r_array,
            rules,
            risk_geometry=risk_geometry,
            atr=atr,
            rng=rng,
            max_trades=max_trades,
            trades_per_day=trades_per_day,
            max_days=max_days,
            resampling=resampling,
            day_blocks=day_blocks,
            mean_block_length=mean_block_length,
        )
        outcomes.append(res.outcome)
        finals.append(res.final_equity)
        if res.outcome == "pass":
            trades_to_pass.append(res.n_trades)

    n = len(outcomes)
    n_pass = sum(1 for o in outcomes if o == "pass")
    n_fail = sum(1 for o in outcomes if o == "fail")
    n_timeout = n - n_pass - n_fail

    finals_arr = np.array(finals, dtype=float)
    ci_low, ci_high = wilson_interval(n_pass, n)
    pass_rate = n_pass / n if n else 0.0

    # -- diagnostic: how much is the i.i.d. bootstrap flattering this? --
    if compare_iid and resampling == RESAMPLING_BLOCK:
        iid_summary = monte_carlo_challenge(
            pnl_r_array,
            rules,
            risk_geometry=risk_geometry,
            n_attempts=max(1000, n_attempts // 4),
            seed=seed + 1,
            atr=atr,
            max_trades=max_trades,
            trades_per_day=trades_per_day,
            max_days=max_days,
            resampling=RESAMPLING_IID,
        )
        gap = pass_rate - iid_summary.pass_rate
        if abs(gap) >= 0.02:
            direction = "optimistic" if gap > 0 else "pessimistic"
            msg.append(
                f"resampling diagnostic: the i.i.d. bootstrap gives a pass rate "
                f"{abs(gap) * 100:.1f} percentage points "
                f"{direction} than the block bootstrap "
                f"({iid_summary.pass_rate:.1%} vs {pass_rate:.1%}). The gap is the "
                "cost of destroying the within-day loss clustering."
            )
        else:
            msg.append(
                f"resampling diagnostic: the i.i.d. and block bootstraps agree "
                f"within {abs(gap) * 100:.1f} percentage points "
                f"({iid_summary.pass_rate:.1%} vs {pass_rate:.1%}), so this sample "
                "shows little day-level clustering."
            )

    return MCSummary(
        n_attempts=n,
        pass_rate=pass_rate,
        fail_rate=n_fail / n if n else 0.0,
        timeout_rate=n_timeout / n if n else 0.0,
        mean_final_equity=float(finals_arr.mean()) if n else 0.0,
        p5_final_equity=float(np.percentile(finals_arr, 5)) if n else 0.0,
        p95_final_equity=float(np.percentile(finals_arr, 95)) if n else 0.0,
        mean_trades_to_pass=float(np.mean(trades_to_pass)) if trades_to_pass else 0.0,
        pass_rate_ci_low=ci_low,
        pass_rate_ci_high=ci_high,
        n_trades_sample=n_sample,
        risk_geometry=risk_geometry,
        resampling=resampling,
        warnings=msg,
    )
