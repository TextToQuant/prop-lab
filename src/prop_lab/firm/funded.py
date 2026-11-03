"""Funded-phase simulation.

Once a trader passes the evaluation, the question changes from "will I
pass?" to "how much will I actually collect before the account dies?".

This module simulates the funded lifetime: the payout ladder, the
consistency rule, the minimum daily profit requirement, and the point at
which the trailing drawdown finally takes the account down.

Two rules deserve attention because they are easy to get wrong:

Consistency rule
    A single day may not contribute more than ``consistency_ratio`` of the
    profit made since the last payout. At APEX the limit is 50 percent on
    current accounts and 30 percent on legacy ones. Breaching it does not
    fail the account. It only makes the payout button unavailable, so the
    simulation stops counting that day's profit toward the payout and
    keeps trading. Note that losing days count against you, because they
    shrink the denominator.

Minimum daily profit
    A day only qualifies for a payout if it cleared
    ``min_daily_profit``. Days below it still contribute to the total
    profit, they just do not count toward the qualifying-day count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .monte_carlo import RISK_FN_MAP, simulate_challenge
from .rules import FirmRules

__all__ = ["FundedResult", "FundedSummary", "simulate_funded", "monte_carlo_funded"]


@dataclass
class FundedResult:
    """One simulated funded-account lifetime."""

    total_payout: float = 0.0
    n_payouts: int = 0
    days_survived: int = 0
    blew_up: bool = False
    daily_profits: list[float] = field(default_factory=list)
    consistency_blocked: int = 0
    final_equity: float = 0.0


@dataclass
class FundedSummary:
    """Aggregate result of N simulated funded lifetimes."""

    n_attempts: int
    mean_total_payout: float
    median_total_payout: float
    p5_total_payout: float
    p95_total_payout: float
    mean_payouts: float
    mean_days_survived: float
    blow_up_rate: float
    expected_value: float
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Plain-text summary of the funded-phase result."""
        return (
            f"attempts            : {self.n_attempts}\n"
            f"mean total payout   : ${self.mean_total_payout:,.0f}\n"
            f"median total payout : ${self.median_total_payout:,.0f}\n"
            f"p5 / p95 payout     : ${self.p5_total_payout:,.0f} / ${self.p95_total_payout:,.0f}\n"
            f"mean payouts        : {self.mean_payouts:.2f}\n"
            f"mean days survived  : {self.mean_days_survived:.1f}\n"
            f"blow-up rate        : {self.blow_up_rate:.1%}\n"
            f"expected value      : ${self.expected_value:,.0f}"
        )


def simulate_funded(
    pnl_r_array: np.ndarray,
    rules: FirmRules,
    *,
    risk_geometry: str = "frac_2pct",
    atr: float = 0.0,
    rng: np.random.Generator | None = None,
    max_trades: int = 2000,
    trades_per_day: int = 3,
    max_days: int = 60,
) -> FundedResult:
    """Simulate one funded-account lifetime.

    The funded account starts at the starting balance and never gets to
    pass again: it only survives, collects payouts, and eventually dies
    or runs out of road.

    Args:
        pnl_r_array: per-trade ``pnl_r`` values from the real backtest.
        rules: the firm ruleset. The payout ladder, consistency ratio and
            minimum daily profit come from here.
        risk_geometry: key into the risk geometry map.
        atr: passed to the risk function.
        rng: numpy Generator for reproducibility.
        max_trades: cap on trades in the lifetime.
        trades_per_day: trades per synthetic day.
        max_days: cap on synthetic days, the funded lifetime length.

    Returns:
        A :class:`FundedResult`.
    """
    pnl_r_array = np.asarray(pnl_r_array, dtype=float)
    if pnl_r_array.size == 0:
        raise ValueError("pnl_r_array is empty; run a backtest first")
    if rng is None:
        rng = np.random.default_rng()

    risk_fn = RISK_FN_MAP[risk_geometry]
    start = rules.starting_balance
    target = rules.target_balance
    floor = rules.floor
    max_dd = rules.max_drawdown
    ladder = list(rules.payout_ladder) or [1_500.0]

    equity = float(start)
    peak = float(start)
    threshold = floor
    total_payout = 0.0
    n_payouts = 0
    daily_profits: list[float] = []
    day_running = 0.0
    trades_in_day = 0
    consistency_blocked = 0

    # Profit accumulated since the last payout, for the consistency check.
    period_profit = 0.0
    period_best_day = 0.0
    qualifying_days = 0

    for t in range(1, max_trades + 1):
        if len(daily_profits) >= max_days:
            break

        risk_dollars = risk_fn(equity, peak, atr, rules)
        if rules.max_daily_loss > 0:
            risk_dollars = min(risk_dollars, rules.max_daily_loss)
        room = equity - threshold
        if room > 0:
            risk_dollars = min(risk_dollars, room)
        risk_dollars = max(risk_dollars, 25.0)

        pnl_r = float(pnl_r_array[rng.integers(0, pnl_r_array.size)])
        pnl = risk_dollars * pnl_r

        equity += pnl
        peak = max(peak, equity)
        day_running += pnl
        trades_in_day += 1

        if rules.trailing_drawdown:
            threshold = max(floor, peak - max_dd)

        # -- end of synthetic day -------------------------------------
        if trades_in_day >= trades_per_day:
            daily_profits.append(day_running)
            period_profit += day_running
            if day_running > period_best_day:
                period_best_day = day_running

            qualifies = True
            if rules.min_daily_profit is not None and day_running < rules.min_daily_profit:
                qualifies = False

            # Consistency: one day may not be more than the ratio of the
            # period profit. Breaching it blocks the payout, it does not
            # fail the account.
            consistency_ok = True
            if rules.consistency_ratio is not None and period_profit > 0:
                if period_best_day >= rules.consistency_ratio * period_profit:
                    consistency_ok = False
                    consistency_blocked += 1

            if qualifies and consistency_ok:
                qualifying_days += 1

            # -- payout check ------------------------------------------
            can_payout = equity >= target and qualifying_days >= 5
            if rules.min_daily_profit is not None:
                can_payout = can_payout and qualifying_days >= 5
            if can_payout and n_payouts < rules.max_payouts:
                cap = ladder[min(n_payouts, len(ladder) - 1)]
                payout = min(cap, equity - target)
                if payout > 0:
                    total_payout += payout
                    n_payouts += 1
                    # The reset after a payout: the consistency window and
                    # the qualifying-day count both start over.
                    period_profit = 0.0
                    period_best_day = 0.0
                    qualifying_days = 0
                    # A payout takes the balance down to the target level.
                    equity = max(target, equity - payout)

            day_running = 0.0
            trades_in_day = 0

        # -- did the account die? -------------------------------------
        if equity < threshold:
            return FundedResult(
                total_payout=total_payout,
                n_payouts=n_payouts,
                days_survived=len(daily_profits),
                blew_up=True,
                daily_profits=daily_profits,
                consistency_blocked=consistency_blocked,
                final_equity=equity,
            )

    return FundedResult(
        total_payout=total_payout,
        n_payouts=n_payouts,
        days_survived=len(daily_profits),
        blew_up=False,
        daily_profits=daily_profits,
        consistency_blocked=consistency_blocked,
        final_equity=equity,
    )


def monte_carlo_funded(
    pnl_r_array: np.ndarray,
    rules: FirmRules,
    *,
    risk_geometry: str = "frac_2pct",
    n_attempts: int = 5_000,
    seed: int = 42,
    atr: float = 0.0,
    max_trades: int = 2_000,
    trades_per_day: int = 3,
    max_days: int = 60,
    challenge_pass_rate: float | None = None,
) -> FundedSummary:
    """Run N simulated funded lifetimes and summarise them.

    Args:
        pnl_r_array: per-trade ``pnl_r`` values.
        rules: the firm ruleset.
        risk_geometry: key into the risk geometry map.
        n_attempts: number of simulated funded accounts.
        seed: RNG seed.
        atr: passed to the risk function.
        max_trades: cap on trades per lifetime.
        trades_per_day: trades per synthetic day.
        max_days: the funded lifetime length in synthetic days.
        challenge_pass_rate: if given, the expected value is computed as
            ``challenge_pass_rate * mean_payout - fee``. If not given, the
            expected value is the mean payout alone.

    Returns:
        A :class:`FundedSummary`.
    """
    pnl_r_array = np.asarray(pnl_r_array, dtype=float)
    if pnl_r_array.size == 0:
        raise ValueError("pnl_r_array is empty; run a backtest first")

    msg: list[str] = []
    if pnl_r_array.size < 30:
        msg.append(
            f"sample of only {pnl_r_array.size} trades: funded-phase numbers "
            "inherit that uncertainty."
        )
    msg.append(
        "session boundaries are approximated with synthetic days of "
        f"{trades_per_day} trades, and the payout timing is approximate."
    )
    msg.append(
        "payout delays, contract scaling plans and the firm's approval "
        "process are not modelled."
    )

    rng = np.random.default_rng(seed)
    payouts: list[float] = []
    n_payouts: list[int] = []
    days: list[int] = []
    blowups: list[bool] = []

    for _ in range(n_attempts):
        res = simulate_funded(
            pnl_r_array,
            rules,
            risk_geometry=risk_geometry,
            atr=atr,
            rng=rng,
            max_trades=max_trades,
            trades_per_day=trades_per_day,
            max_days=max_days,
        )
        payouts.append(res.total_payout)
        n_payouts.append(res.n_payouts)
        days.append(res.days_survived)
        blowups.append(res.blew_up)

    arr = np.array(payouts, dtype=float)
    mean_payout = float(arr.mean()) if len(arr) else 0.0
    if challenge_pass_rate is None:
        ev = mean_payout
    else:
        ev = challenge_pass_rate * mean_payout - rules.fee

    return FundedSummary(
        n_attempts=len(payouts),
        mean_total_payout=mean_payout,
        median_total_payout=float(np.median(arr)) if len(arr) else 0.0,
        p5_total_payout=float(np.percentile(arr, 5)) if len(arr) else 0.0,
        p95_total_payout=float(np.percentile(arr, 95)) if len(arr) else 0.0,
        mean_payouts=float(np.mean(n_payouts)) if n_payouts else 0.0,
        mean_days_survived=float(np.mean(days)) if days else 0.0,
        blow_up_rate=float(np.mean(blowups)) if blowups else 0.0,
        expected_value=ev,
        warnings=msg,
    )
