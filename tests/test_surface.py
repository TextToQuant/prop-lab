"""Tests for the surface layer.

The point of these tests is the honesty contract: the provenance tag on
each axis must be correct, and the rendering must never contain emoji.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from prop_lab.surface import surfaces as surf
from prop_lab.surface.surfaces import (
    AXIS_MEASURED,
    AXIS_MODELLED,
    Surface,
    SurfaceSpec,
    risk_geometry_surface,
    rr_surface,
    surface_table,
    winrate_surface,
)

# Characters outside the ASCII range that would indicate an emoji slipped
# into the output. A quick scan is enough: the report layer is plain text.
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


class FakeRules:
    """A stand-in for a firm ruleset."""

    name = "Fake Firm 50K EOD"


@pytest.fixture(autouse=True)
def stub_mc(monkeypatch):
    """Replace the Monte Carlo indirection with a fixed pass rate.

    This keeps the surface tests independent of the firm package, which
    is exactly why the indirection exists.
    """
    calls = []

    def fake(pnl_r_array, rules, risk_geometry, n_attempts, seed):
        calls.append((risk_geometry, n_attempts, seed, len(pnl_r_array)))
        return 0.42

    monkeypatch.setattr(surf, "_simulate_pass_rate", fake)
    return calls


# ----------------------------------------------------------------------
# Hand-built surface
# ----------------------------------------------------------------------
def test_to_dataframe_row_count():
    """A hand-built surface has len(x) * len(y) rows in long format."""
    spec = SurfaceSpec(
        x_name="risk_geometry",
        x_values=["a", "b", "c"],
        x_provenance=AXIS_MEASURED,
        y_name="ruleset",
        y_values=["r1", "r2"],
        y_provenance=AXIS_MEASURED,
        z_name="pass_rate",
        z_provenance=AXIS_MEASURED,
        note="a note",
    )
    z = np.arange(6, dtype=float).reshape(3, 2)
    surface = Surface(spec=spec, z=z, axis_labels={"name": "demo"})

    df = surface.to_dataframe()
    assert len(df) == 3 * 2
    assert list(df.columns) == ["x", "y", "z", "risk_geometry", "ruleset", "pass_rate"]
    assert df["z"].sum() == pytest.approx(z.sum())
    assert set(df["risk_geometry"]) == {"a", "b", "c"}
    assert set(df["ruleset"]) == {"r1", "r2"}


def test_surface_shape_is_validated():
    """A z grid whose shape disagrees with the axes is rejected."""
    spec = SurfaceSpec(
        x_name="x",
        x_values=[1, 2],
        x_provenance=AXIS_MEASURED,
        y_name="y",
        y_values=[1, 2, 3],
        y_provenance=AXIS_MEASURED,
        z_name="z",
        z_provenance=AXIS_MEASURED,
    )
    with pytest.raises(ValueError):
        Surface(spec=spec, z=np.zeros((2, 2)))


# ----------------------------------------------------------------------
# surface_table
# ----------------------------------------------------------------------
def test_surface_table_states_provenance_and_has_no_emoji():
    """Every surface in the table carries measured/modelled and no emoji."""
    geometry = risk_geometry_surface(
        np.array([1.0, -1.0, 2.0]), FakeRules(), n_attempts=100
    )
    winrate = winrate_surface(
        np.array([1.0, -1.0, 2.0]), FakeRules(), win_rates=[0.4, 0.5], rr_values=[1.0]
    )
    text = surface_table([geometry, winrate])

    assert "measured" in text
    assert "modelled" in text
    assert not _has_emoji(text)
    # Each surface's axes appear with their provenance tags.
    assert "risk_geometry [measured]" in text
    assert "win_rate [modelled]" in text
    assert "rr_ratio [modelled]" in text


def test_surface_table_handles_empty_list():
    """An empty surface list still renders something usable."""
    text = surface_table([])
    assert isinstance(text, str)
    assert "No surfaces" in text


# ----------------------------------------------------------------------
# winrate_surface
# ----------------------------------------------------------------------
def test_winrate_surface_axes_are_modelled():
    """The win-rate surface is modelled on BOTH axes, always."""
    surface = winrate_surface(
        np.array([1.0, -1.0, 2.0]),
        FakeRules(),
        win_rates=[0.3, 0.5, 0.6],
        rr_values=[1.0, 2.0],
        n_attempts=100,
    )
    assert surface.spec.x_provenance == AXIS_MODELLED
    assert surface.spec.y_provenance == AXIS_MODELLED
    assert surface.spec.z_provenance == AXIS_MEASURED
    assert "parametric" in surface.spec.note.lower()
    assert "DeltaTrend" in surface.spec.note
    assert surface.z.shape == (3, 2)


def test_winrate_surface_provenance_rows():
    """provenance_rows() lists all three axes with their tags."""
    surface = winrate_surface(
        np.array([1.0, -1.0]), FakeRules(), win_rates=[0.5], rr_values=[1.0], n_attempts=50
    )
    rows = surface.spec.provenance_rows()
    assert len(rows) == 3
    names = [r[0] for r in rows]
    assert names == ["win_rate", "rr_ratio", "pass_rate"]
    provs = [r[1] for r in rows]
    assert provs == [AXIS_MODELLED, AXIS_MODELLED, AXIS_MEASURED]
    assert all(r[2] for r in rows)


def test_winrate_surface_synthetic_distribution_is_two_point():
    """The synthetic cell is +rr / -1 with the right number of winners."""
    arr = surf._synthetic_two_point_pnl(0.45, 2.0, 42)
    assert len(arr) == 1000
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    assert len(wins) == 450
    assert len(losses) == 550
    assert set(np.unique(wins)) == {2.0}
    assert set(np.unique(losses)) == {-1.0}


def test_winrate_surface_is_reproducible():
    """The same seed gives the same synthetic array."""
    a = surf._synthetic_two_point_pnl(0.45, 2.0, 7)
    b = surf._synthetic_two_point_pnl(0.45, 2.0, 7)
    assert np.array_equal(a, b)


# ----------------------------------------------------------------------
# risk_geometry_surface
# ----------------------------------------------------------------------
def test_risk_geometry_surface_axes_are_measured():
    """Both axes of the risk-geometry surface are measured."""
    surface = risk_geometry_surface(
        np.array([1.0, -1.0, 2.0]), FakeRules(), n_attempts=100
    )
    assert surface.spec.x_provenance == AXIS_MEASURED
    assert surface.spec.y_provenance == AXIS_MEASURED
    assert surface.spec.z_provenance == AXIS_MEASURED
    assert surface.spec.x_values == surf.DEFAULT_RISK_GEOMETRIES
    assert surface.z.shape == (len(surf.DEFAULT_RISK_GEOMETRIES), 1)


def test_risk_geometry_surface_accepts_a_list_of_rulesets():
    """A list of rulesets becomes a multi-element y axis."""

    class OtherRules(FakeRules):
        name = "Other Firm 100K EOD"

    surface = risk_geometry_surface(
        np.array([1.0, -1.0]),
        [FakeRules(), OtherRules()],
        risk_geometries=["frac_2pct"],
        n_attempts=50,
    )
    assert surface.spec.y_values == ["Fake Firm 50K EOD", "Other Firm 100K EOD"]
    assert surface.z.shape == (1, 2)


def test_risk_geometry_surface_calls_mc_with_the_geometry(stub_mc):
    """Each cell asks the Monte Carlo layer for its own geometry."""
    risk_geometry_surface(
        np.array([1.0, -1.0]), FakeRules(), risk_geometries=["frac_2pct"], n_attempts=10
    )
    assert stub_mc[0][0] == "frac_2pct"
    assert stub_mc[0][1] == 10


# ----------------------------------------------------------------------
# rr_surface
# ----------------------------------------------------------------------
def test_rr_surface_unmeasured_is_modelled():
    """The fast R:R surface labels its x axis as modelled."""
    surface = rr_surface(
        np.array([1.0, 2.0, -1.0]),
        FakeRules(),
        rr_values=[1.0, 2.0],
        risk_geometries=["frac_2pct"],
        measured=False,
        n_attempts=50,
    )
    assert surface.spec.x_provenance == AXIS_MODELLED
    assert surface.spec.y_provenance == AXIS_MEASURED
    assert surface.spec.z_provenance == AXIS_MEASURED
    assert "not regenerated" in surface.spec.note
    assert "scaled" in surface.spec.note


def test_rr_surface_unmeasured_scales_only_the_winners():
    """Losers are untouched; winners are scaled to the target mean."""
    base = np.array([2.0, -1.0, 4.0, -2.0])
    out = surf._scale_winners_to_rr(base, 1.0)
    assert out[1] == -1.0
    assert out[3] == -2.0
    assert out[0] + out[2] == pytest.approx(2.0)


def test_rr_surface_measured_requires_factory_and_df():
    """measured=True without a strategy_factory and df is an error."""
    with pytest.raises(ValueError):
        rr_surface(np.array([1.0, -1.0]), FakeRules(), measured=True)


def test_rr_surface_measured_requires_an_rr_overridable_strategy():
    """A strategy that cannot take an R:R raises a clear ValueError."""

    class NoRRStrategy:
        name = "no_rr"

        def __init__(self):
            self.calls = 0

    with pytest.raises(ValueError, match="rr_ratio"):
        rr_surface(
            np.array([1.0, -1.0]),
            FakeRules(),
            rr_values=[1.0],
            risk_geometries=["frac_2pct"],
            measured=True,
            strategy_factory=NoRRStrategy,
            df=object(),
        )


def test_rr_surface_measured_rebuilds_and_reruns_the_backtest(monkeypatch):
    """measured=True re-runs the backtest per R:R and labels the axis measured."""
    import pandas as pd

    from prop_lab.core import Trade

    seen_rrs = []

    class RRStrategy:
        name = "rr_strategy"

        def __init__(self, rr_ratio=1.0):
            self.rr_ratio = rr_ratio

    def fake_run_backtest(strategy, df, **kwargs):
        seen_rrs.append(strategy.rr_ratio)
        return [
            Trade(
                direction="long",
                entry_time=0,
                exit_time=1,
                entry_price=1.0,
                exit_price=1.0,
                stop_distance=1.0,
                rr_ratio=strategy.rr_ratio,
                pnl_points=1.0,
                pnl_r=strategy.rr_ratio,
                reason="x",
                session_date="2026-01-01",
            )
        ]

    monkeypatch.setattr("prop_lab.core.run_backtest", fake_run_backtest)

    surface = rr_surface(
        np.array([1.0, -1.0]),
        FakeRules(),
        rr_values=[1.0, 2.0, 3.0],
        risk_geometries=["frac_2pct"],
        measured=True,
        strategy_factory=RRStrategy,
        df=pd.DataFrame(),
        n_attempts=50,
    )
    assert seen_rrs == [1.0, 2.0, 3.0]
    assert surface.spec.x_provenance == AXIS_MEASURED
    assert surface.spec.z_provenance == AXIS_MEASURED


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def test_to_3d_returns_a_figure():
    """to_3d returns a matplotlib figure."""
    from matplotlib.figure import Figure

    surface = risk_geometry_surface(
        np.array([1.0, -1.0, 2.0]), FakeRules(), n_attempts=50
    )
    fig = surface.to_3d()
    assert isinstance(fig, Figure)
    title = fig.axes[0].get_title()
    assert "MEASURED" in title
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_to_heatmap_returns_a_figure():
    """to_heatmap returns a matplotlib figure with a provenance title."""
    from matplotlib.figure import Figure

    surface = winrate_surface(
        np.array([1.0, -1.0]),
        FakeRules(),
        win_rates=[0.4, 0.5],
        rr_values=[1.0],
        n_attempts=50,
    )
    fig = surface.to_heatmap()
    assert isinstance(fig, Figure)
    assert "MODELLED" in fig.axes[0].get_title()
    import matplotlib.pyplot as plt

    plt.close(fig)
