"""Tests for the report layer.

The report's contract is that every section is present, that the
honesty section lists the provenance of every surface axis, and that
nothing decorative sneaks into the output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from prop_lab.report.report import build_report
from prop_lab.surface.surfaces import (
    AXIS_MEASURED,
    AXIS_MODELLED,
    Surface,
    SurfaceSpec,
)

_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x1F1E6, 0x1F1FF),
    (0xFE0F, 0xFE0F),
    (0x2190, 0x21FF),
    (0x2B00, 0x2BFF),
)


def _has_emoji(text: str) -> bool:
    """Return True when ``text`` contains a character in an emoji range."""
    for ch in text:
        cp = ord(ch)
        for low, high in _EMOJI_RANGES:
            if low <= cp <= high:
                return True
    return False


# ----------------------------------------------------------------------
# Fake inputs
# ----------------------------------------------------------------------
@dataclass
class FakeWindow:
    fold: int = 1
    train_start: int = 1_700_000_000
    train_end: int = 1_700_086_400
    test_start: int = 1_700_172_800
    test_end: int = 1_700_259_200
    n_train_trades: int = 120
    n_test_trades: int = 30
    test_mean_r: float = 0.21
    test_win_rate: float = 0.47


@dataclass
class FakeWFResult:
    windows: list = field(default_factory=lambda: [FakeWindow(), FakeWindow()])
    oos_mean_r: float = 0.18
    oos_win_rate: float = 0.46
    oos_total_r: float = 5.4
    is_mean_r: float = 0.27
    degradation: float = 0.09
    verdict: str = "holds"
    n_windows: int = 2
    warnings: list = field(default_factory=lambda: ["only 60 out-of-sample trades: sample too small to conclude"])


@dataclass
class FakeMCSummary:
    n_attempts: int = 2000
    pass_rate: float = 0.3125
    fail_rate: float = 0.6
    timeout_rate: float = 0.0875
    mean_final_equity: float = 48_200.0
    p5_final_equity: float = 46_100.0
    p95_final_equity: float = 53_800.0
    mean_trades_to_pass: float = 41.2
    pass_rate_ci_low: float = 0.2893
    pass_rate_ci_high: float = 0.3369
    n_trades_sample: int = 214
    warnings: list = field(default_factory=lambda: ["session boundaries are approximated with synthetic days of 3 trades"])


@dataclass
class FakeRules:
    name: str = "APEX 50K EOD"


def _geometry_surface() -> Surface:
    spec = SurfaceSpec(
        x_name="risk_geometry",
        x_values=["frac_2pct", "max_size"],
        x_provenance=AXIS_MEASURED,
        y_name="ruleset",
        y_values=["APEX 50K EOD"],
        y_provenance=AXIS_MEASURED,
        z_name="pass_rate",
        z_provenance=AXIS_MEASURED,
        note="Each cell is a genuine simulation over the real trades.",
    )
    # z has shape (len(x_values), len(y_values)) = (2, 1) here.
    z = np.array([[0.31], [0.22]])
    return Surface(
        spec=spec,
        z=z,
        axis_labels={"name": "risk_geometry x ruleset"},
        ci_low=np.array([[0.29], [0.20]]),
        ci_high=np.array([[0.34], [0.24]]),
    )


def _winrate_surface() -> Surface:
    spec = SurfaceSpec(
        x_name="win_rate",
        x_values=[0.40, 0.50],
        x_provenance=AXIS_MODELLED,
        y_name="rr_ratio",
        y_values=[1.0, 2.0],
        y_provenance=AXIS_MODELLED,
        z_name="pass_rate",
        z_provenance=AXIS_MEASURED,
        note="This is a parametric surface, not a backtest result.",
    )
    z = np.array([[0.10, 0.25], [0.35, 0.62]])
    return Surface(spec=spec, z=z, axis_labels={"name": "win_rate x rr_ratio"})


@pytest.fixture
def report() -> str:
    """Build a report with two surfaces."""
    return build_report(
        wf=FakeWFResult(),
        mc=FakeMCSummary(),
        surfaces=[_geometry_surface(), _winrate_surface()],
        strategy_name="Trend Continuation v3",
        data_source="Yahoo Finance (NQ=F daily)",
        data_range="2020-01-01 to 2026-10-31",
        rules=FakeRules(),
        extra_warnings=["the last bar of the data is force-closed"],
    )


# ----------------------------------------------------------------------
# Section presence
# ----------------------------------------------------------------------
def test_all_sections_present(report):
    """Every required section heading is in the report, in order."""
    for heading in (
        "Headline",
        "Honesty",
        "Risk geometry",
        "Out-of-sample",
        "Caveats",
        "Provenance",
    ):
        assert heading in report, f"missing section heading: {heading}"

    positions = [report.index(h) for h in (
        "## 1. Headline",
        "## 2. Honesty",
        "## 3. Risk geometry",
        "## 4. Out-of-sample",
        "## 5. Caveats",
        "## 6. Provenance",
    )]
    assert positions == sorted(positions), "sections are out of order"


def test_report_has_no_emoji(report):
    """The report contains no emoji."""
    assert not _has_emoji(report)


def test_honesty_section_lists_every_axis(report):
    """The honesty table names every axis with its provenance tag."""
    for axis in ("risk_geometry", "ruleset", "pass_rate", "win_rate", "rr_ratio"):
        assert axis in report
    # The modelled axes are labelled as such.
    assert "modelled" in report
    assert "measured" in report
    # The honesty section is present and is not a stub.
    honesty = report.split("## 2. Honesty")[1].split("## 3.")[0]
    assert "measured" in honesty
    assert "modelled" in honesty


def test_honesty_section_present_even_with_no_surfaces():
    """The honesty section is mandatory, even with no surfaces."""
    text = build_report(
        wf=FakeWFResult(),
        mc=FakeMCSummary(),
        surfaces=[],
        strategy_name="No Surface Strategy",
        data_source="synthetic",
        data_range="2026-01-01 to 2026-02-01",
        rules=FakeRules(),
    )
    assert "Honesty" in text
    assert "## 3. Risk geometry" in text
    assert not _has_emoji(text)


def test_headline_contains_the_numbers(report):
    """The headline prints the pass rate and the IS/OOS mean R."""
    assert "31.2%" in report          # pass rate
    assert "28.9%" in report          # CI low
    assert "33.7%" in report          # CI high
    assert "0.270" in report          # IS mean R
    assert "0.180" in report          # OOS mean R
    assert "holds" in report          # verdict
    assert "214" in report            # trade count


def test_risk_geometry_section_has_rows(report):
    """The risk geometry table is derived from the surface."""
    section = report.split("## 3. Risk geometry")[1].split("## 4.")[0]
    assert "frac_2pct" in section
    assert "max_size" in section
    # The fixture's z values are 0.31 and 0.22.
    assert "31.0%" in section
    assert "22.0%" in section


def test_risk_geometry_section_without_breakdown():
    """Without a geometry breakdown the section says so."""
    text = build_report(
        wf=FakeWFResult(),
        mc=FakeMCSummary(),
        surfaces=[_winrate_surface()],
        strategy_name="No Geometry",
        data_source="synthetic",
        data_range="2026-01-01 to 2026-02-01",
        rules=FakeRules(),
    )
    section = text.split("## 3. Risk geometry")[1].split("## 4.")[0]
    assert "No per-risk-geometry breakdown was supplied" in section


def test_oos_table_has_one_row_per_fold(report):
    """The walk-forward table lists both folds."""
    section = report.split("## 4. Out-of-sample")[1].split("## 5.")[0]
    assert section.count("2023-11-14") >= 1
    assert "120" in section   # train trades
    assert "30" in section    # test trades
    assert "47.0%" in section  # test win rate


def test_caveats_include_warnings_and_standing_caveats(report):
    """Warnings from the run, the extra warnings, and the standing caveats."""
    section = report.split("## 5. Caveats")[1].split("## 6.")[0]
    assert "sample too small to conclude" in section
    assert "synthetic days" in section
    assert "the last bar of the data is force-closed" in section
    assert "Slippage and commission are not modelled" in section
    assert "News gaps are not modelled" in section
    assert "Contract scaling plans and payout delays are not modelled" in section
    assert "Firms change their rules" in section
    assert "small trade sample cannot support a confident conclusion" in section


def test_provenance_has_urls_and_dates(report):
    """The provenance section lists citations with URLs and check dates."""
    section = report.split("## 6. Provenance")[1]
    assert "https://github.com/dsilva42/prop-risk-geometry" in section
    assert "hunter547" in section
    assert "eslazarev" in section
    assert "Lopez de Prado" in section
    assert "DeltaTrend Trading" in section
    assert "2026-11-03" in section


def test_report_is_deterministic():
    """Two identical calls produce identical text."""
    kwargs = dict(
        wf=FakeWFResult(),
        mc=FakeMCSummary(),
        surfaces=[_geometry_surface()],
        strategy_name="Deterministic",
        data_source="synthetic",
        data_range="2026-01-01 to 2026-02-01",
        rules=FakeRules(),
    )
    assert build_report(**kwargs) == build_report(**kwargs)
