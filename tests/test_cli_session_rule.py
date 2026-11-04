"""The session rule must be decided once and applied by every command.

The loader infers the rule from the data: `binance` means a 24/7 market
and the session is the UTC day (0), anything else means the futures
convention (18). That inference is useless if half the commands do not
route the override through it, which was the defect: `surface` and
`report` honoured `--session-open-hour` while `backtest`, `walkforward`
and `montecarlo` silently used the inferred value. On a non-Binance CSV
those two halves of the same run described different accounts.

These tests pin the plumbing, and pin the fact that the loader says which
rule it applied, so a mismatch is visible rather than silent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prop_lab.cli as cli


def _strategy_path() -> str:
    root = Path(__file__).resolve().parents[1]
    return str(root / "examples" / "strategies" / "trend_continuation.py")


def _frame(n: int = 300) -> pd.DataFrame:
    """A minimal canonical bar frame; the commands only need it to run."""
    time = 1_704_067_200 + np.arange(n) * 900
    close = 100.0 + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "time": time,
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1.0,
        }
    )


@pytest.mark.parametrize(
    "command,extra",
    [
        ("backtest", []),
        ("walkforward", []),
        ("montecarlo", ["--attempts", "100"]),
    ],
)
def test_session_open_hour_reaches_the_loader(monkeypatch, command, extra):
    """Every data command must forward the override to `_load_bars`."""
    seen: list = []

    def fake_load_bars(path, *, timeframe=None, session_open_hour=None):
        seen.append(session_open_hour)
        return _frame()

    monkeypatch.setattr(cli, "_load_bars", fake_load_bars)

    cli.main(
        [
            command,
            "--strategy",
            f"{_strategy_path()}:TrendContinuation",
            "--data",
            "unused.csv",
            "--session-open-hour",
            "0",
            *extra,
        ]
    )

    assert seen == [0], f"{command} did not forward --session-open-hour (got {seen})"


@pytest.mark.parametrize("command", ["backtest", "walkforward", "montecarlo"])
def test_commands_still_infer_when_no_override_is_given(monkeypatch, command):
    """No flag must mean "decide from the data", not "use 18"."""
    seen: list = []

    def fake_load_bars(path, *, timeframe=None, session_open_hour=None):
        seen.append(session_open_hour)
        return _frame()

    monkeypatch.setattr(cli, "_load_bars", fake_load_bars)

    cli.main(
        [
            command,
            "--strategy",
            f"{_strategy_path()}:TrendContinuation",
            "--data",
            "unused.csv",
        ]
    )

    assert seen == [None], f"{command} should pass None so the loader infers"


def test_loader_reports_the_inferred_rule(tmp_path, capsys):
    """A CSV with no source column is a futures-style day, and says so."""
    csv = tmp_path / "bars.csv"
    _frame().to_csv(csv, index=False)

    cli.main(
        [
            "backtest",
            "--strategy",
            f"{_strategy_path()}:TrendContinuation",
            "--data",
            str(csv),
        ]
    )

    captured = capsys.readouterr().out
    assert "session rule: 18 (inferred from source=unknown)" in captured


def test_loader_reports_an_explicit_rule(tmp_path, capsys):
    """An override is applied and named, so it cannot be silently ignored."""
    csv = tmp_path / "bars.csv"
    _frame().to_csv(csv, index=False)

    cli.main(
        [
            "backtest",
            "--strategy",
            f"{_strategy_path()}:TrendContinuation",
            "--data",
            str(csv),
            "--session-open-hour",
            "0",
        ]
    )

    captured = capsys.readouterr().out
    assert "session rule: 0 (--session-open-hour)" in captured
