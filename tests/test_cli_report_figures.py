"""The report command must write the figures it documents.

The README says `proplab report` produces "everything, plus the labelled
surfaces". If it writes only the Markdown, the reader is told about plots
and receives none.

A 3D figure is written only for a grid that varies along both axes. A
grid with a constant axis is a list of numbers, and the 3D view of it is
a vertical curtain: it is not a surface and it is not shipped. That grid
still gets its heatmap, which shows the same values readably.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prop_lab.cli import main


def _strategy_path() -> str:
    root = Path(__file__).resolve().parents[1]
    return str(root / "examples" / "strategies" / "trend_continuation.py")


def _write_bars(csv: Path) -> None:
    """A synthetic series, so the test needs no network and no fixture."""
    idx = pd.date_range("2024-01-01", periods=400, freq="15min", tz="UTC")
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.2, len(idx)))
    pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 1000.0,
        }
    ).to_csv(csv, index=False)


def test_absolute_strategy_path_loads():
    """An absolute strategy path must work, including on Windows.

    The spec is "module:Class". Splitting it on the first colon breaks
    every Windows path, because the drive letter is itself a colon.
    """
    from prop_lab.cli import _load_strategy

    spec = f"{_strategy_path()}:TrendContinuation"
    assert _load_strategy(spec).__name__ == "TrendContinuation"


def _run_report(tmp_path, firm: str):
    """Run `proplab report` on synthetic bars and return the output dir."""
    csv = tmp_path / "bars.csv"
    _write_bars(csv)
    out = tmp_path / "reports"
    code = main(
        [
            "report",
            "--strategy",
            f"{_strategy_path()}:TrendContinuation",
            "--data",
            str(csv),
            "--firm",
            firm,
            "--attempts",
            "60",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    return out


def test_report_writes_surface_figures(tmp_path):
    """One firm: heatmaps only, plus the win-rate 3D.

    With a single ruleset the risk-geometry grid is 5x1, so one axis is
    constant and no 3D figure is written for it. The R:R surface is never
    drawn in 3D, because its cells are read one at a time. The win-rate
    surface varies on both axes and is the one 3D view this run has.
    """
    out = _run_report(tmp_path, "apex-50k-eod")

    assert (out / "report.md").exists()

    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs, "the report command wrote no figures"
    assert pngs == [
        "surface_risk_geometry_heatmap.png",
        "surface_rr_ratio_heatmap.png",
        "surface_win_rate_3d.png",
        "surface_win_rate_heatmap.png",
    ], pngs

    # The degenerate grid still gets its heatmap, and must not get a
    # curtain.
    assert not (out / "surface_risk_geometry_3d.png").exists()
    # The R:R surface is heatmap-only by decision, not by degeneracy.
    assert not (out / "surface_rr_ratio_3d.png").exists()


def test_report_writes_a_3d_figure_once_rulesets_are_crossed(tmp_path):
    """Crossing rulesets makes the risk-geometry grid a real surface.

    The ruleset x risk-geometry cross is the point of the package, and it
    only exists when more than one ruleset is supplied.
    """
    out = _run_report(tmp_path, "apex-50k-eod,topstep-50k")

    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == [
        "surface_risk_geometry_3d.png",
        "surface_risk_geometry_heatmap.png",
        "surface_rr_ratio_heatmap.png",
        "surface_win_rate_3d.png",
        "surface_win_rate_heatmap.png",
    ], pngs
