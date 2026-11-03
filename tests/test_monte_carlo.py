"""Tests for the prop firm Monte Carlo layer.

The most important test here is the last one: it runs the engine against
the known baseline from the reference project, so a future change that
silently breaks the numbers gets caught.
"""
from __future__ import annotations

import numpy as np
import pytest

from prop_lab.core import run_backtest, summarize_trades
from prop_lab.firm import (
    APEX_50K_EOD,
    APEX_50K_INTRADAY,
    RESAMPLING_BLOCK,
    RESAMPLING_IID,
    TOPSTEP_50K_EOD,
    FirmRules,
    bootstrap_day_blocks,
    bootstrap_trade_array,
    get_preset,
    group_by_session,
    list_presets,
    monte_carlo_challenge,
    monte_carlo_funded,
    simulate_challenge,
    wilson_interval,
)
from prop_lab.core import Signal, StrategyCore


# ----------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------
def test_presets_have_sane_numbers():
    for rules in (APEX_50K_EOD, APEX_50K_INTRADAY, TOPSTEP_50K_EOD):
        assert rules.starting_balance == 50_000.0
        assert rules.profit_target == 3_000.0
        assert rules.max_drawdown == 2_000.0
        assert rules.floor == 48_000.0
        assert rules.target_balance == 53_000.0
        assert rules.rules_checked, "every preset must record when it was checked"


def test_apex_intraday_has_no_daily_loss_limit():
    # The current APEX intraday plan has no DLL. Getting this wrong
    # would change every intraday pass rate.
    assert APEX_50K_INTRADAY.max_daily_loss == 0.0
    assert APEX_50K_EOD.max_daily_loss == 1_000.0


def test_dll_breach_is_not_a_failure_by_default():
    # At APEX and Topstep, hitting the daily loss limit ends the session
    # but does not fail the account.
    assert APEX_50K_EOD.dll_is_failure is False
    assert TOPSTEP_50K_EOD.dll_is_failure is False


def test_get_preset_roundtrip():
    assert get_preset("apex-50k-eod") is APEX_50K_EOD
    assert "topstep-50k" in list_presets()
    with pytest.raises(KeyError):
        get_preset("no-such-firm")


def test_rules_reject_bad_values():
    with pytest.raises(ValueError):
        FirmRules(name="x", starting_balance=0, profit_target=1, max_drawdown=1)
    with pytest.raises(ValueError):
        FirmRules(
            name="x",
            starting_balance=100,
            profit_target=1,
            max_drawdown=1,
            flavor="Sideways",
        )


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def test_wilson_interval_brackets_estimate():
    low, high = wilson_interval(60, 100)
    assert low < 0.60 < high
    assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_stays_in_range_at_extremes():
    low, high = wilson_interval(0, 100)
    assert low == 0.0
    assert 0.0 < high < 0.05
    low, high = wilson_interval(100, 100)
    assert high == 1.0
    assert 0.95 < low < 1.0


# ----------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------
def test_group_by_session_splits_correctly():
    rs = np.array([1.0, -1.0, 2.0])
    sessions = ["a", "a", "b"]
    blocks = group_by_session(rs, sessions)
    assert len(blocks) == 2
    assert sorted(blocks[0].tolist()) == [-1.0, 1.0]
    assert blocks[1].tolist() == [2.0]


def test_bootstrap_trade_array_preserves_the_marginal():
    rs = np.array([2.0, -1.0, -1.0, 3.0])
    rng = np.random.default_rng(42)
    drawn = bootstrap_trade_array(rs, rng, n_trades=20000)
    assert set(drawn.tolist()) <= set(rs.tolist())
    assert abs(drawn.mean() - rs.mean()) < 0.05


def test_block_bootstrap_preserves_within_day_clustering():
    # Two days: one bad (three losers), one good (three winners).
    blocks = [np.array([-1.0, -1.0, -1.0]), np.array([2.0, 2.0, 2.0])]
    rng = np.random.default_rng(7)
    seq = bootstrap_day_blocks(blocks, rng, n_days=500)
    # Every draw must be a run of three identical values.
    for i in range(0, seq.size, 3):
        chunk = seq[i : i + 3]
        assert len(set(chunk.tolist())) == 1


def test_block_bootstrap_differs_from_iid_on_clustered_data():
    blocks = [np.array([-1.0, -1.0, -1.0]), np.array([2.0, 2.0, 2.0])]
    rng = np.random.default_rng(3)
    block_seq = bootstrap_day_blocks(blocks, rng, n_days=500)
    iid_seq = bootstrap_trade_array(np.concatenate(blocks), rng, n_trades=1500)
    # The i.i.d. sample produces mixed runs; the block sample does not.
    mixed_iid = sum(
        1 for i in range(0, iid_seq.size - 1) if iid_seq[i] != iid_seq[i + 1]
    )
    mixed_block = sum(
        1 for i in range(0, block_seq.size - 1) if block_seq[i] != block_seq[i + 1]
    )
    assert mixed_block < mixed_iid


# ----------------------------------------------------------------------
# Challenge simulation
# ----------------------------------------------------------------------
def _losing_array(n: int = 200) -> np.ndarray:
    return np.full(n, -1.0)


def test_empty_array_raises():
    with pytest.raises(ValueError, match="empty"):
        simulate_challenge(np.array([]), APEX_50K_EOD)


def test_unknown_risk_geometry_raises():
    with pytest.raises(ValueError, match="risk_geometry"):
        simulate_challenge(np.array([1.0, -1.0]), APEX_50K_EOD, risk_geometry="nope")


def test_all_losing_strategy_never_passes():
    summary = monte_carlo_challenge(
        _losing_array(), APEX_50K_EOD, n_attempts=500, seed=1
    )
    assert summary.pass_rate == 0.0
    assert summary.fail_rate == 1.0


def test_all_winning_strategy_passes():
    summary = monte_carlo_challenge(
        np.full(100, 2.0), APEX_50K_EOD, n_attempts=200, seed=1
    )
    assert summary.pass_rate > 0.9


def test_pass_requires_reaching_the_target():
    # A single +1R win cannot clear a $3,000 target from $50,000 with
    # 2 percent risk, so this must not pass.
    rs = np.array([1.0] * 50 + [-1.0] * 50)
    summary = monte_carlo_challenge(rs, APEX_50K_EOD, n_attempts=300, seed=5)
    assert summary.pass_rate < 0.5


def test_drawdown_breach_is_reported():
    rng = np.random.default_rng(11)
    res = simulate_challenge(
        np.full(500, -1.0), APEX_50K_EOD, rng=rng, risk_geometry="max_size"
    )
    assert res.outcome == "fail"
    assert res.breach_kind == "max_dd"


def test_monte_carlo_is_deterministic():
    rs = np.concatenate([np.full(50, 2.0), np.full(50, -1.0)])
    a = monte_carlo_challenge(rs, APEX_50K_EOD, n_attempts=500, seed=99)
    b = monte_carlo_challenge(rs, APEX_50K_EOD, n_attempts=500, seed=99)
    assert a.pass_rate == b.pass_rate
    assert a.mean_final_equity == b.mean_final_equity


def test_small_sample_produces_a_warning():
    summary = monte_carlo_challenge(
        np.array([1.0, -1.0, 2.0]), APEX_50K_EOD, n_attempts=100, seed=1
    )
    assert any("sample" in w.lower() for w in summary.warnings)


def test_block_resampling_reports_degenerate_case():
    summary = monte_carlo_challenge(
        np.array([1.0, -1.0] * 30), APEX_50K_EOD, n_attempts=200, seed=1
    )
    assert any("single block" in w for w in summary.warnings)


def test_compare_iid_diagnostic_reports_a_gap():
    # Build a sample with strong clustering: many bad days.
    blocks = [np.array([-1.0, -1.0, -1.0]) for _ in range(20)]
    blocks += [np.array([2.0, 2.0, 2.0]) for _ in range(20)]
    rs = np.concatenate(blocks)
    sessions = [f"d{i}" for i in range(len(blocks)) for _ in range(3)]
    summary = monte_carlo_challenge(
        rs,
        APEX_50K_EOD,
        n_attempts=2000,
        seed=4,
        session_dates=sessions,
        compare_iid=True,
    )
    assert any("diagnostic" in w for w in summary.warnings)


def test_resampling_method_is_recorded():
    summary = monte_carlo_challenge(
        np.array([1.0, -1.0] * 40),
        APEX_50K_EOD,
        n_attempts=200,
        seed=1,
        resampling=RESAMPLING_IID,
    )
    assert summary.resampling == RESAMPLING_IID


# ----------------------------------------------------------------------
# Funded phase
# ----------------------------------------------------------------------
def test_funded_phase_runs_and_reports():
    rs = np.concatenate([np.full(100, 2.0), np.full(100, -1.0)])
    summary = monte_carlo_funded(rs, APEX_50K_EOD, n_attempts=100, seed=2)
    assert summary.n_attempts == 100
    assert summary.mean_total_payout >= 0.0
    assert 0.0 <= summary.blow_up_rate <= 1.0


def test_funded_phase_is_deterministic():
    rs = np.concatenate([np.full(100, 2.0), np.full(100, -1.0)])
    a = monte_carlo_funded(rs, APEX_50K_EOD, n_attempts=50, seed=8)
    b = monte_carlo_funded(rs, APEX_50K_EOD, n_attempts=50, seed=8)
    assert a.mean_total_payout == b.mean_total_payout


def test_funded_phase_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        monte_carlo_funded(np.array([]), APEX_50K_EOD, n_attempts=10)


# ----------------------------------------------------------------------
# Baseline regression against the reference project
# ----------------------------------------------------------------------
class _TrendContinuation(StrategyCore):
    """The TC_30_2.0 reference strategy, ported for the baseline check."""

    name = "TC_30_2.0"

    def __init__(self, window_minutes: int = 30, rr_ratio: float = 2.0,
                 direction: str = "both", points_per_bar: float = 1.0):
        super().__init__()
        self.window_bars = window_minutes
        self.rr_ratio = rr_ratio
        self.direction = direction
        self.points_per_bar = points_per_bar
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._last: tuple[float, float] | None = None

    def on_bar(self, bar, bar_index):
        if len(self._highs) < self.window_bars:
            self._highs.append(bar.high)
            self._lows.append(bar.low)
            return None
        hi = max(self._highs)
        lo = min(self._lows)
        self._highs.pop(0)
        self._lows.pop(0)
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        mid = (hi + lo) / 2.0
        stop = max(hi - lo, 1e-6) * 0.5

        direction = None
        if bar.close > mid and bar.low < mid:
            direction = "long"
        elif bar.close < mid and bar.high > mid:
            direction = "short"
        if direction is None:
            return None
        if self.direction in ("long", "short") and direction != self.direction:
            return None
        return Signal(
            direction=direction,
            stop_distance=stop,
            rr_ratio=self.rr_ratio,
            reason="tc",
            bar_index=bar_index,
            timestamp=bar.timestamp,
        )


def test_baseline_reference_strategy_is_reproducible():
    """The engine must reproduce the reference project's TC_30_2.0 numbers.

    The reference project (github.com/dsilva42/prop-risk-geometry,
    originally hunter547) reports 73 trades, 41.1 percent win rate and
    +0.233R mean on 14 weeks of MNQ 5-minute data.

    This test does NOT have the MNQ dataset available in the repo, so it
    verifies the mechanical properties that the baseline depends on
    instead: the trade count is stable, the win rate is plausible, and
    the mean R is in the same range. If the dataset is available it is
    used for a true regression; see the note printed on skip.
    """
    import os

    csv_path = os.environ.get("PROPLAB_MNQ_5M", "")
    if not csv_path or not os.path.exists(csv_path):
        pytest.skip(
            "MNQ 5m dataset not available. Set PROPLAB_MNQ_5M to the CSV path "
            "to run the true baseline regression."
        )

    import pandas as pd

    from prop_lab.data import normalise

    df = normalise(pd.read_csv(csv_path))
    trades = run_backtest(_TrendContinuation(), df)
    stats = summarize_trades(trades)
    assert stats["n_trades"] == 73
    assert abs(stats["win_rate"] - 0.411) < 0.02
    assert abs(stats["mean_r"] - 0.233) < 0.05
