"""The report command must write the figures it documents.

The README says `proplab report` produces "everything, plus the labelled
surfaces". If it writes only the Markdown, the reader is told about six
plots and receives none.
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


def test_report_writes_surface_figures(tmp_path):
    """`proplab report` writes a PNG for every surface it describes."""
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
            "apex-50k-eod",
            "--attempts",
            "60",
            "--out",
            str(out),
        ]
    )
    assert code == 0

    assert (out / "report.md").exists()

    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs, "the report command wrote no figures"
    # One 3d and one heatmap for each of the three surfaces.
    assert len(pngs) == 6, pngs
    assert any(n.endswith("_3d.png") for n in pngs)
    assert any(n.endswith("_heatmap.png") for n in pngs)
