"""Prop firm rulesets.

Every number in this file was checked against the firm's own published
rules on the date recorded next to it. Prop firms change their terms
often, so the ``rules_checked`` field on each preset is not decoration:
re-verify before trusting a number with real money behind it.

The presets here cover APEX Trader Funding 50K and Topstep 50K because
those are the two most commonly modelled. Any other firm can be expressed
by constructing :class:`FirmRules` directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "FirmRules",
    "APEX_50K_EOD",
    "APEX_50K_INTRADAY",
    "TOPSTEP_50K_EOD",
    "FIRM_PRESETS",
    "get_preset",
    "list_presets",
]


@dataclass(frozen=True)
class FirmRules:
    """A prop firm evaluation ruleset.

    All dollar amounts are absolute, not percentages, because that is how
    the firms state them and how the drawdown is enforced.

    Attributes:
        name: display name, used as the axis label on surfaces.
        starting_balance: account balance at the start of the evaluation.
        profit_target: dollars of profit required to pass.
        max_drawdown: dollars of drawdown allowed from the trailing peak.
        max_daily_loss: dollars of loss allowed in one session. Set to 0.0
            for firms with no daily loss limit.
        evaluation_days: calendar days available to pass.
        min_trading_days: minimum number of trading days required.
        trailing_drawdown: whether the drawdown follows the equity peak.
        static_floor: the hard floor the drawdown can never go below.
            ``None`` means ``starting_balance - max_drawdown``.
        flavor: 'EOD' or 'Intraday'. EOD firms recalculate the threshold
            once per session from the closing balance; intraday firms track
            the peak continuously including unrealised profit.
        fee: the evaluation fee, if any.
        payout_ladder: the ordered payout caps.
        consistency_ratio: the largest fraction of period profit that a
            single day may contribute. ``None`` means no consistency rule.
            This is the FUNDED-stage rule: it gates whether a payout is
            available. At APEX the limit is 50 percent on current
            accounts. Breaching it does not fail the account, it only
            makes the payout unavailable.
        eval_consistency_ratio: the consistency rule that applies DURING
            the evaluation, where exceeding it RAISES the profit target
            instead of failing the account. Topstep's Trading Combine
            uses 0.55: a $1,800 best day turns the $3,000 target into
            $1,800 / 0.55 = $3,273. APEX has no such rule on the
            evaluation, it applies consistency only at the payout stage.
            ``None`` means no evaluation-stage consistency rule.
        min_daily_profit: minimum profit a day must have to qualify as a
            payout day. ``None`` means no such requirement.
        max_payouts: number of payouts available.
        dll_is_failure: when True, breaching the daily loss limit ends the
            evaluation. When False, it only pauses trading for the session
            (this is how APEX and Topstep actually treat it).
        rules_checked: the date the numbers were verified against the
            firm's published rules.
        source: where the numbers came from.
    """

    name: str
    starting_balance: float
    profit_target: float
    max_drawdown: float
    max_daily_loss: float = 0.0
    evaluation_days: int = 30
    min_trading_days: int = 0
    trailing_drawdown: bool = True
    static_floor: float | None = None
    flavor: str = "EOD"
    fee: float = 0.0
    payout_ladder: tuple[float, ...] = ()
    consistency_ratio: float | None = None
    eval_consistency_ratio: float | None = None
    min_daily_profit: float | None = None
    max_payouts: int = 6
    dll_is_failure: bool = False
    rules_checked: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        if self.profit_target < 0:
            raise ValueError("profit_target must not be negative")
        if self.max_drawdown <= 0:
            raise ValueError("max_drawdown must be positive")
        if self.max_daily_loss < 0:
            raise ValueError("max_daily_loss must not be negative")
        if self.flavor not in ("EOD", "Intraday"):
            raise ValueError(f"flavor must be 'EOD' or 'Intraday', got {self.flavor!r}")
        if self.consistency_ratio is not None and not 0 < self.consistency_ratio < 1:
            raise ValueError("consistency_ratio must be strictly between 0 and 1")
        if (
            self.eval_consistency_ratio is not None
            and not 0 < self.eval_consistency_ratio < 1
        ):
            raise ValueError(
                "eval_consistency_ratio must be strictly between 0 and 1"
            )

    # -- derived -------------------------------------------------------
    @property
    def target_balance(self) -> float:
        """The balance at which the evaluation is passed."""
        return self.starting_balance + self.profit_target

    @property
    def floor(self) -> float:
        """The hard floor the drawdown can never go below."""
        if self.static_floor is not None:
            return self.static_floor
        return self.starting_balance - self.max_drawdown

    def describe(self) -> str:
        """One-paragraph human summary of the ruleset."""
        lines = [
            f"{self.name} ({self.flavor})",
            f"  starting balance : ${self.starting_balance:,.0f}",
            f"  profit target    : ${self.profit_target:,.0f} "
            f"(balance ${self.target_balance:,.0f})",
            f"  max drawdown     : ${self.max_drawdown:,.0f} "
            f"({'trailing' if self.trailing_drawdown else 'static'}, floor ${self.floor:,.0f})",
        ]
        if self.max_daily_loss > 0:
            lines.append(
                f"  daily loss limit : ${self.max_daily_loss:,.0f} "
                f"({'ends the evaluation' if self.dll_is_failure else 'pauses trading only'})"
            )
        else:
            lines.append("  daily loss limit : none")
        lines.append(f"  evaluation window: {self.evaluation_days} days")
        if self.min_trading_days:
            lines.append(f"  min trading days : {self.min_trading_days}")
        if self.consistency_ratio is not None:
            lines.append(
                f"  consistency      : best day <= {self.consistency_ratio:.0%} of period profit "
                "(funded-stage payout gate)"
            )
        if self.eval_consistency_ratio is not None:
            lines.append(
                f"  eval consistency : best day <= {self.eval_consistency_ratio:.0%} of profit "
                "target; exceeding it raises the target"
            )
        if self.min_daily_profit is not None:
            lines.append(f"  min daily profit : ${self.min_daily_profit:,.0f}")
        if self.payout_ladder:
            ladder = ", ".join(f"${p:,.0f}" for p in self.payout_ladder)
            lines.append(f"  payout ladder    : {ladder}")
        if self.fee:
            lines.append(f"  fee              : ${self.fee:,.0f}")
        if self.rules_checked:
            lines.append(f"  rules checked    : {self.rules_checked}")
        if self.source:
            lines.append(f"  source           : {self.source}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------
# APEX 50K EOD Trail.
# Checked 2026-11-03 against support.apextraderfunding.com and the
# tradervana / mytradeedge rule summaries of the current plans.
#
# Current APEX accounts (sold since 2026-03-01) differ from legacy ones:
#   - profit target $3,000 on 50K
#   - trailing drawdown $2,000 on 50K
#   - daily loss limit $1,000 on the EOD plan, and NONE on the Intraday plan
#   - no minimum trading days, no consistency rule in the evaluation
#   - trailing stops at starting balance + $100 on a performance account
#
# IMPORTANT: hitting the daily loss limit does NOT fail an APEX account. It
# pauses trading for the rest of the session. That is why
# dll_is_failure defaults to False.
APEX_50K_EOD = FirmRules(
    name="APEX 50K EOD",
    starting_balance=50_000.0,
    profit_target=3_000.0,
    max_drawdown=2_000.0,
    max_daily_loss=1_000.0,
    evaluation_days=30,
    min_trading_days=0,
    trailing_drawdown=True,
    flavor="EOD",
    fee=147.0,
    payout_ladder=(1_500.0, 1_500.0, 2_000.0, 2_500.0, 2_500.0, 3_000.0),
    consistency_ratio=0.50,
    eval_consistency_ratio=None,
    min_daily_profit=250.0,
    max_payouts=6,
    dll_is_failure=False,
    rules_checked="2026-11-03",
    source="support.apextraderfunding.com evaluation rules; tradervana apex rules 2026",
)

# APEX 50K Intraday Trail: same target and drawdown, but no daily loss limit.
APEX_50K_INTRADAY = FirmRules(
    name="APEX 50K Intraday",
    starting_balance=50_000.0,
    profit_target=3_000.0,
    max_drawdown=2_000.0,
    max_daily_loss=0.0,
    evaluation_days=30,
    min_trading_days=0,
    trailing_drawdown=True,
    flavor="Intraday",
    fee=147.0,
    payout_ladder=(1_500.0, 2_000.0, 2_500.0, 2_500.0, 3_000.0, 3_000.0),
    consistency_ratio=0.50,
    eval_consistency_ratio=None,
    min_daily_profit=200.0,
    max_payouts=6,
    dll_is_failure=False,
    rules_checked="2026-11-03",
    source="support.apextraderfunding.com evaluation rules; tradervana apex rules 2026",
)

# Topstep 50K Trading Combine.
# Checked 2026-11-03 against help.topstep.com and the proea.app /
# tradeharbor.co summaries.
#
#   - profit target $3,000
#   - Maximum Loss Limit $2,000, trailing the highest END-OF-DAY balance,
#     and it locks at the starting balance once a close reaches $52,000
#   - daily loss limit $1,000, optional at purchase; hitting it ends the
#     session but does NOT fail the account
#   - consistency target: best single day at or below 55% of the profit
#     target; exceeding it RAISES the target rather than failing you
#   - flat by 3:10 PM CT
#   - 5 minis or 50 micros max
#   - minimum 2 trading days to pass (checked 2026-11-03)
#
# Two of these rules are enforced in the simulator:
#   - min_trading_days: the evaluation cannot pass until enough trading
#     days have elapsed, even if the profit target is already reached.
#   - consistency_ratio: when the best day exceeds the ratio, Topstep
#     RAISES the profit target to best_day / ratio instead of failing
#     the account. The simulator applies that same adjustment.
TOPSTEP_50K_EOD = FirmRules(
    name="Topstep 50K",
    starting_balance=50_000.0,
    profit_target=3_000.0,
    max_drawdown=2_000.0,
    max_daily_loss=1_000.0,
    evaluation_days=0,  # no time limit while the subscription runs
    min_trading_days=2,
    trailing_drawdown=True,
    flavor="EOD",
    fee=149.0,
    payout_ladder=(1_500.0, 2_000.0, 2_500.0, 2_500.0, 3_000.0, 3_000.0),
    consistency_ratio=0.55,
    eval_consistency_ratio=0.55,
    min_daily_profit=150.0,
    max_payouts=6,
    dll_is_failure=False,
    rules_checked="2026-11-03",
    source="help.topstep.com trading combine parameters; proea.app topstep combine rules",
)


FIRM_PRESETS: dict[str, FirmRules] = {
    "apex-50k-eod": APEX_50K_EOD,
    "apex-50k-intraday": APEX_50K_INTRADAY,
    "topstep-50k": TOPSTEP_50K_EOD,
}


def get_preset(name: str) -> FirmRules:
    """Look up a firm preset by key.

    Args:
        name: one of the keys in :data:`FIRM_PRESETS`, case-insensitive.

    Returns:
        The matching :class:`FirmRules`.

    Raises:
        KeyError: if no preset matches, with the available keys listed.
    """
    key = name.strip().lower()
    if key not in FIRM_PRESETS:
        raise KeyError(
            f"unknown firm preset {name!r}; available: {', '.join(sorted(FIRM_PRESETS))}"
        )
    return FIRM_PRESETS[key]


def list_presets() -> list[str]:
    """Return the sorted list of available firm preset keys."""
    return sorted(FIRM_PRESETS)
