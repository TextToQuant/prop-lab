"""Surfaces: measured versus modelled, and never confusing the two.

This module is the honesty layer of ``prop_lab``. Every surface it builds
carries, for each of its three axes, an explicit provenance tag:

``AXIS_MEASURED`` ("measured")
    The axis value was computed from a real backtest or a real simulation
    over real trades. Nothing on this axis was assumed.

``AXIS_MODELLED`` ("modelled")
    The axis value was assumed or drawn from a parametric model. It is NOT
    the output of a backtest, and it must never be presented as one.

Why the distinction exists
---------------------------

Three surfaces are built here, and they differ in exactly one way: what
their axes actually are.

1. The **risk geometry** surface crosses five position-sizing rules
   against one or more firm rulesets. Both axes are MEASURED. The reason
   is structural: the set of trades is fixed by the backtest, and the only
   thing that changes from cell to cell is how many dollars are risked per
   trade. The Monte Carlo layer then bootstraps those same real trades
   under each sizing rule. Every cell is a genuine simulation over the
   real per-trade R-multiple distribution. This is the surface that carries
   real information, and it is the one to lead with.

2. The **R:R** surface crosses a reward-to-risk ratio against risk
   geometry. The R:R axis CAN be measured, but only expensively: changing
   the R:R changes which trades exist, because a trade that hits its
   target at 2R would not have hit it at 1R. The honest version therefore
   re-runs the backtest at every R:R value, rebuilds the per-trade
   ``pnl_r`` array, and simulates that. The cheap version ("modelled")
   rescales the winners of the existing trade array so that the mean
   winner equals the target R:R; the trade sequence is then NOT
   regenerated, and the axis is labelled ``AXIS_MODELLED`` to say so.

3. The **win rate** surface crosses an assumed win rate against an R:R
   ratio. BOTH AXES ARE MODELLED, ALWAYS, and this is not optional. You
   cannot "re-backtest" a win rate: to force a strategy to have a win
   rate of 55 percent when the real backtest produced 41 percent, you
   would have to invent trades that never happened. Splitting a real
   trade distribution into "wins" and "losses" and re-drawing them at a
   different rate is a model, and it is labelled as one.

The shape of the win-rate/R:R surface -- pass rate as a function of win
rate and reward-to-risk -- was popularised by DeltaTrend Trading on TikTok
and YouTube, using Topstep rules and a custom Monte Carlo simulation. The
point of this module is not to dispute that visualisation; it is to be
explicit that its win-rate axis is parametric by construction, and to say
which parts of the idea can and cannot be measured from a real backtest.

The rule this module never breaks: a surface may never present a modelled
axis as a measured one, and the pass-rate axis (z) is measured in every
case because it always comes from an actual simulation over the trade
distribution that was handed in -- whatever that distribution is.

The bootstrap Monte Carlo method used downstream originates from the
prop-risk-geometry project, originally ``hunter547/prop-risk-geometry``
and now ``dsilva42/prop-risk-geometry``.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "AXIS_MEASURED",
    "AXIS_MODELLED",
    "SurfaceSpec",
    "Surface",
    "risk_geometry_surface",
    "rr_surface",
    "winrate_surface",
    "surface_table",
    "DEFAULT_RISK_GEOMETRIES",
    "DEFAULT_RR_VALUES",
    "DEFAULT_WIN_RATES",
]

#: Provenance tag for an axis that was computed from real data.
AXIS_MEASURED = "measured"
#: Provenance tag for an axis that was assumed or modelled.
AXIS_MODELLED = "modelled"

#: The five position-sizing rules the prop-risk-geometry project defines.
DEFAULT_RISK_GEOMETRIES = [
    "max_size",
    "frac_2pct",
    "dd_frac_40pct",
    "anti_martingale",
    "floor_aware",
]

#: Default R:R grid for :func:`rr_surface`.
DEFAULT_RR_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

#: Default assumed win rates for :func:`winrate_surface`.
DEFAULT_WIN_RATES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# Number of synthetic trades used per cell of the win-rate surface.
_WINRATE_SYNTH_N = 1000

_PROVENANCE_EXPLANATION = {
    AXIS_MEASURED: (
        "measured: computed from the real backtest / a real simulation over "
        "the real per-trade distribution; nothing here was assumed."
    ),
    AXIS_MODELLED: (
        "modelled: assumed or parametric. Not the output of a backtest, and "
        "must not be read as one."
    ),
}


# ----------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------
@dataclass
class SurfaceSpec:
    """The description of a surface: its axes, their provenance, its note.

    Attributes:
        x_name: name of the x axis, e.g. ``'risk_geometry'``.
        x_values: the x axis values, in axis order.
        x_provenance: ``AXIS_MEASURED`` or ``AXIS_MODELLED``.
        y_name: name of the y axis.
        y_values: the y axis values, in axis order.
        y_provenance: ``AXIS_MEASURED`` or ``AXIS_MODELLED``.
        z_name: name of the z axis, e.g. ``'pass_rate'``.
        z_provenance: ``AXIS_MEASURED`` or ``AXIS_MODELLED``.
        note: a plain-English note on what this surface is and is not.
        operating_point: optional ``(x_value, y_value)`` of the strategy
            the surface was built for. When set, the renderer marks where
            that strategy actually sits on the grid, so a user can see
            whether their own parameters are even on the surface they are
            looking at.
        operating_point_label: how to refer to the operating point in
            labels.
    """

    x_name: str
    x_values: list
    x_provenance: str
    y_name: str
    y_values: list
    y_provenance: str
    z_name: str
    z_provenance: str
    note: str = ""
    operating_point: tuple[float, float] | None = None
    operating_point_label: str = "this strategy"

    def provenance_rows(self) -> list[tuple[str, str, str]]:
        """Return one ``(axis_name, provenance, explanation)`` row per axis.

        The rows are ordered x, y, z. The explanation is a one-line,
        plain-English statement of what the provenance tag means for that
        axis. The report renders these rows verbatim, so they must be
        self-explanatory without any surrounding prose.

        Returns:
            A list of three ``(str, str, str)`` tuples.
        """
        rows: list[tuple[str, str, str]] = []
        for axis_name, provenance in (
            (self.x_name, self.x_provenance),
            (self.y_name, self.y_provenance),
            (self.z_name, self.z_provenance),
        ):
            explanation = _PROVENANCE_EXPLANATION.get(
                provenance,
                "unknown provenance tag; treat this axis with suspicion.",
            )
            rows.append((axis_name, provenance, explanation))
        return rows

    @property
    def title(self) -> str:
        """A short ``'x axis by y axis'`` title for the surface."""
        return f"{self.x_name} x {self.y_name}"


# ----------------------------------------------------------------------
# Surface
# ----------------------------------------------------------------------
@dataclass
class Surface:
    """A surface: a spec, a z grid, and human labels for the axes.

    Attributes:
        spec: the :class:`SurfaceSpec` describing the axes and their
            provenance.
        z: array of shape ``(len(x_values), len(y_values))``.
        axis_labels: human-readable labels, keyed ``'x'``, ``'y'``,
            ``'z'`` and ``'name'``.
        ci_low: optional lower 95 percent confidence bound per cell, same
            shape as ``z``. ``None`` when no bound was computed.
        ci_high: optional upper 95 percent confidence bound per cell.
    """

    spec: SurfaceSpec
    z: np.ndarray
    axis_labels: dict = field(default_factory=dict)
    ci_low: np.ndarray | None = None
    ci_high: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate the z grid against the axis lengths."""
        self.z = np.asarray(self.z, dtype=float)
        expected = (len(self.spec.x_values), len(self.spec.y_values))
        if self.z.shape != expected:
            raise ValueError(
                f"z has shape {self.z.shape}, expected {expected} "
                f"(len(x_values) x len(y_values))"
            )
        if not self.axis_labels:
            self.axis_labels = {
                "x": self.spec.x_name,
                "y": self.spec.y_name,
                "z": self.spec.z_name,
                "name": self.spec.title,
            }

    # -- data ----------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Return the surface in long format.

        The frame has one row per cell and the columns ``x``, ``y`` and
        ``z``, plus one column named after each axis (``x_name``,
        ``y_name``, ``z_name``) holding the same values under their
        readable names.

        Returns:
            A :class:`pandas.DataFrame` with
            ``len(x_values) * len(y_values)`` rows.
        """
        xs = list(self.spec.x_values)
        ys = list(self.spec.y_values)
        rows: list[tuple[Any, Any, float]] = []
        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                rows.append((xv, yv, float(self.z[i, j])))
        df = pd.DataFrame(rows, columns=["x", "y", "z"])
        df[self.spec.x_name] = df["x"]
        df[self.spec.y_name] = df["y"]
        df[self.spec.z_name] = df["z"]
        return df

    # -- plots ---------------------------------------------------------
    def to_3d(self):
        """Render the surface as a 3D matplotlib figure.

        Uses the ``plasma`` colormap. The title states the provenance of
        the surface so that the figure cannot be screenshotted out of
        context and mislead.

        Returns:
            A ``matplotlib.figure.Figure``.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)

        fig = plt.figure(figsize=(9.0, 7.0))
        ax = fig.add_subplot(111, projection="3d")

        xs = [str(v) for v in self.spec.x_values]
        ys = [str(v) for v in self.spec.y_values]
        xi = np.arange(len(xs))
        yi = np.arange(len(ys))
        xx, yy = np.meshgrid(xi, yi, indexing="ij")

        surf = ax.plot_surface(
            xx,
            yy,
            self.z,
            cmap="plasma",
            edgecolor="none",
            antialiased=True,
        )
        fig.colorbar(surf, ax=ax, shrink=0.6, label=self.spec.z_name)

        ax.set_xticks(xi)
        ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(yi)
        ax.set_yticklabels(ys, fontsize=8)
        ax.set_xlabel(self.spec.x_name)
        ax.set_ylabel(self.spec.y_name)
        ax.set_zlabel(self.spec.z_name)
        if self.spec.z_name == "pass_rate":
            # A pass rate is a probability, so the axis reads in percent.
            ax.zaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
            )
        ax.set_title(self._title_text())
        fig.tight_layout()
        return fig

    def to_heatmap(self):
        """Render the surface as an annotated 2D heatmap figure.

        Each cell is annotated with its z value. The title carries the
        provenance note.

        Returns:
            A ``matplotlib.figure.Figure``.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        # z has shape (len(x_values), len(y_values)); transpose it so that
        # rows are the y axis and columns are the x axis, which is how
        # imshow draws a table.
        grid = np.asarray(self.z, dtype=float).T
        im = ax.imshow(grid, cmap="plasma", aspect="auto", origin="lower")
        fig.colorbar(im, ax=ax, label=self.spec.z_name)

        # Matplotlib normalises the grid against its own min and max.
        # The annotation colour has to be chosen against the SAME
        # normalisation, otherwise the text colour and the cell colour
        # disagree about which end of the ramp a cell sits at.
        vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
        span = vmax - vmin

        ax.set_xticks(np.arange(len(self.spec.x_values)))
        ax.set_xticklabels(
            [str(v) for v in self.spec.x_values], rotation=30, ha="right", fontsize=8
        )
        ax.set_yticks(np.arange(len(self.spec.y_values)))
        ax.set_yticklabels([str(v) for v in self.spec.y_values], fontsize=8)
        ax.set_xlabel(self.spec.x_name)
        ax.set_ylabel(self.spec.y_name)

        # Annotate every cell with its value.
        #
        # Orientation matters here. ``self.z`` has shape
        # (len(x_values), len(y_values)). The image is drawn from
        # ``grid = z.T``, which has shape (len(y_values), len(x_values)),
        # so imshow places grid[j][i] at x position i and y position j.
        # The value z[i, j] therefore belongs at (x=i, y=j). Annotating
        # at (j, i) puts the text in the wrong cell, and when there is
        # only one ruleset row the text lands outside the image
        # entirely and floats in the empty margin.
        n_x, n_y = self.z.shape
        for i in range(n_x):
            for j in range(n_y):
                value = float(self.z[i, j])
                text = self._format_cell(value)
                ax.text(
                    i,
                    j,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=self._annotation_colour(
                        (grid[j, i] - vmin) / span if span > 0 else 0.5
                    ),
                )

        self._mark_operating_point(ax, xi=len(self.spec.x_values), yi=len(self.spec.y_values))
        ax.set_title(self._title_text(), fontsize=9)
        fig.tight_layout()
        return fig

    def _mark_operating_point(self, ax, *, xi: int, yi: int) -> None:
        """Mark where the strategy actually sits on the grid.

        A user looking at a grid of assumed win rates needs to know
        whether their own strategy is even on it. When the operating
        point falls outside the grid the marker is drawn at the edge and
        labelled as being off the grid, because a marker silently clamped
        to the nearest cell would read as "this cell is my strategy".
        """
        op = self.spec.operating_point
        if op is None:
            return
        import matplotlib

        x_val, y_val = float(op[0]), float(op[1])
        x_values = [float(v) for v in self.spec.x_values]
        y_values = [float(v) for v in self.spec.y_values]

        if not x_values or not y_values:
            return

        def _nearest(values: list[float], target: float) -> tuple[float, bool]:
            best = min(values, key=lambda v: abs(v - target))
            inside = min(values) <= target <= max(values)
            return best, inside

        x_near, x_inside = _nearest(x_values, x_val)
        y_near, y_inside = _nearest(y_values, y_val)

        # Pixel position of the nearest grid value.
        x_index = x_values.index(x_near)
        y_index = y_values.index(y_near)
        on_grid = x_inside and y_inside

        ax.scatter(
            [x_index],
            [y_index],
            marker="*",
            s=420,
            facecolors="none",
            edgecolors=matplotlib.colormaps["plasma"](0.98) if on_grid else "white",
            linewidths=2.0,
            zorder=5,
        )
        label = (
            f"{self.spec.operating_point_label}: "
            f"{x_val:.0%} win, {y_val:.2f} R:R"
            if on_grid
            else (
                f"{self.spec.operating_point_label} is OFF this grid "
                f"(win {x_val:.0%}, R:R {y_val:.2f})"
            )
        )
        # Place the label so it does not sit on top of a cell
        # annotation. Offsetting up and to the right covers the cell to
        # the upper-right, so the label is pushed toward whichever edge
        # of the plot has room: upward when the marker is in the lower
        # half of the grid, downward when it is in the upper half, and
        # toward whichever side keeps it inside the axes.
        offset_y = 14 if y_index < len(y_values) / 2 else -14
        offset_x = -8 if x_index > len(x_values) / 2 else 8
        ax.annotate(
            label,
            xy=(x_index, y_index),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=7,
            color="white",
            ha="left" if offset_x > 0 else "right",
            va="bottom" if offset_y > 0 else "top",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "black",
                "alpha": 0.85,
                "edgecolor": "white",
                "linewidth": 0.5,
            },
            zorder=6,
        )

    @staticmethod
    def _annotation_colour(normalised: float) -> str:
        """Pick a readable annotation colour for a cell.

        The heatmap uses the ``plasma`` colormap, whose dark end is a
        deep blue-purple with a relative luminance of about 0.07. Black
        text on that background has a contrast ratio of 1.1:1, which is
        invisible in practice; WCAG AA wants 4.5:1.

        So the colour is chosen from the luminance of the cell the text
        actually sits on: white on the dark end of the ramp, black on
        the bright end. The switch happens at the point where white
        text and black text have equal contrast.

        Args:
            normalised: the cell value mapped into the 0-1 colormap
                range, as matplotlib would normalise it.

        Returns:
            ``"white"`` or ``"black"``.
        """
        import matplotlib

        position = float(np.clip(normalised, 0.0, 1.0))
        rgba = matplotlib.colormaps["plasma"](position)
        red, green, blue = (float(c) for c in rgba[:3])

        # Relative luminance per WCAG 2.x.
        def channel(c: float) -> float:
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        luminance = (
            0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)
        )
        # Contrast of black text and of white text against this
        # background. Pick whichever is higher.
        black_contrast = (luminance + 0.05) / 0.05
        white_contrast = 1.05 / (luminance + 0.05)
        return "white" if white_contrast > black_contrast else "black"

    @staticmethod
    def _format_cell(value: float) -> str:
        """Format a cell value for display.

        A pass rate is a probability, so it is shown as a percentage.
        Values that are clearly not probabilities (a drawdown in R, for
        example) keep a plain decimal format.

        Args:
            value: the cell value.

        Returns:
            A short display string.
        """
        if 0.0 <= value <= 1.0:
            return f"{value * 100:.1f}%"
        return f"{value:.2f}"

    def _title_text(self) -> str:
        """Build the figure title, including the provenance statement."""
        prov = f"{self.spec.x_provenance} / {self.spec.y_provenance}"
        return (
            f"{self.spec.x_name} x {self.spec.y_name} - {self.spec.z_name} "
            f"({prov.upper()})"
        )


# ----------------------------------------------------------------------
# Monte Carlo indirection (module level so tests can patch it)
# ----------------------------------------------------------------------
def _simulate_pass_rate(
    pnl_r_array: np.ndarray,
    rules: Any,
    risk_geometry: str,
    n_attempts: int,
    seed: int,
    session_dates: list[str] | None = None,
) -> float:
    """Run the Monte Carlo challenge layer and return the pass rate.

    This is the single point where the surface layer touches the firm
    layer. It is a module-level function on purpose: tests patch
    ``prop_lab.surface.surfaces._simulate_pass_rate`` so the surface
    builders can be exercised without the firm package installed.

    The import is lazy on purpose as well: the core package must import
    with nothing but stdlib, numpy, pandas and matplotlib.

    Args:
        pnl_r_array: per-trade ``pnl_r`` values from a real backtest.
        rules: the firm ruleset object.
        risk_geometry: key into the risk-geometry map, e.g.
            ``'frac_2pct'``.
        n_attempts: number of simulated challenge attempts.
        seed: RNG seed.

    Returns:
        The pass rate as a float in ``[0, 1]``.

    Raises:
        ImportError: if the Monte Carlo layer is missing, has a
            different call signature, or does not return an object with a
            ``pass_rate`` attribute. The message states the expected
            interface. No data is ever fabricated.
    """
    try:
        from prop_lab.firm.monte_carlo import monte_carlo_challenge
    except Exception as exc:  # pragma: no cover - depends on install state
        raise ImportError(
            "The Monte Carlo layer is not available, so no surface can be "
            "computed honestly. Expected "
            "'from prop_lab.firm.monte_carlo import monte_carlo_challenge'. "
            f"Import failed with: {exc!r}"
        ) from exc

    try:
        summary = monte_carlo_challenge(
            pnl_r_array,
            rules,
            risk_geometry=risk_geometry,
            n_attempts=n_attempts,
            seed=seed,
            session_dates=session_dates,
        )
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword" in message or "positional argument" in message:
            raise ImportError(
                "prop_lab.firm.monte_carlo.monte_carlo_challenge has an "
                "unexpected signature. Expected it to accept "
                "monte_carlo_challenge(pnl_r_array, rules, *, "
                "risk_geometry=..., n_attempts=..., seed=...). "
                f"Got: {message}"
            ) from exc
        raise

    if not hasattr(summary, "pass_rate"):
        raise ImportError(
            "prop_lab.firm.monte_carlo.monte_carlo_challenge did not return "
            "an object with a 'pass_rate' attribute, so the pass rate "
            "cannot be read. No data is fabricated in its place."
        )
    return float(summary.pass_rate)


def _wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion.

    Args:
        successes: number of successful attempts (may be fractional when
            derived from a reported rate).
        n: total number of attempts.
        z: z-score for the confidence level (1.96 for 95 percent).

    Returns:
        A ``(low, high)`` tuple clipped to ``[0, 1]``.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _as_list(value: Any) -> list:
    """Return ``value`` as a list, treating a bare object as one element."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _rules_label(rules: Any) -> str:
    """Return a short display label for a firm ruleset."""
    name = getattr(rules, "name", None)
    if name:
        return str(name)
    return type(rules).__name__


def _build_strategy_with_rr(strategy_factory: Any, rr: float) -> Any:
    """Build a strategy instance whose R:R equals ``rr``.

    Three strategies are attempted, in this order:

    1. the instance exposes a parameter named ``rr_ratio`` (in its
       ``params`` mapping or in its constructor signature);
    2. the instance has an ``rr_ratio`` attribute, which is set;
    3. the factory accepts an ``rr_ratio`` keyword argument.

    Args:
        strategy_factory: callable returning a fresh strategy instance.
        rr: the target reward-to-risk ratio.

    Returns:
        A strategy instance configured for ``rr``.

    Raises:
        ValueError: if none of the three strategies works. The message
            explains that ``measured=True`` needs an rr-overridable
            strategy.
    """
    strategy = strategy_factory()

    params = getattr(strategy, "params", None)
    if isinstance(params, dict) and "rr_ratio" in params:
        params["rr_ratio"] = rr
        return strategy

    if hasattr(strategy, "rr_ratio"):
        try:
            setattr(strategy, "rr_ratio", rr)
            return strategy
        except Exception:  # pragma: no cover - read-only attribute
            pass

    try:
        sig = inspect.signature(type(strategy).__init__)
    except (TypeError, ValueError):  # pragma: no cover - exotic objects
        sig = None
    if sig is not None and "rr_ratio" in sig.parameters:
        new_params = dict(params) if isinstance(params, dict) else {}
        new_params["rr_ratio"] = rr
        try:
            return type(strategy)(**new_params)
        except Exception:  # pragma: no cover - fall through to the factory
            pass

    try:
        return strategy_factory(rr_ratio=rr)
    except TypeError as exc:
        raise ValueError(
            "measured=True needs a strategy whose reward-to-risk can be "
            "overridden, but the supplied strategy_factory did not accept "
            "an rr_ratio parameter, an rr_ratio attribute, or an rr_ratio "
            "keyword argument. Either expose 'rr_ratio' on the strategy or "
            "call rr_surface(..., measured=False) to use the modelled "
            f"approximation. Original error: {exc}"
        ) from exc


def _scale_winners_to_rr(pnl_r_array: np.ndarray, rr: float) -> np.ndarray:
    """Rescale the winners of ``pnl_r_array`` so the mean winner equals ``rr``.

    Losers are left untouched. The trade sequence is NOT regenerated: the
    same trades, in the same order, with the same signs, are kept. Only
    the size of the winning trades changes.

    Args:
        pnl_r_array: per-trade ``pnl_r`` values.
        rr: the target mean winner, in R.

    Returns:
        A new float array with the same length as the input.
    """
    arr = np.asarray(pnl_r_array, dtype=float).copy()
    wins = arr > 0
    n_wins = int(wins.sum())
    if n_wins == 0:
        return arr
    mean_win = float(arr[wins].mean())
    if mean_win <= 0:  # pragma: no cover - impossible for positive wins
        return arr
    arr[wins] = arr[wins] * (rr / mean_win)
    return arr


# ----------------------------------------------------------------------
# Surface builders
# ----------------------------------------------------------------------
def risk_geometry_surface(
    pnl_r_array: np.ndarray,
    rules: Any,
    *,
    risk_geometries: Sequence[str] | None = None,
    n_attempts: int = 2000,
    seed: int = 42,
    session_dates: list[str] | None = None,
) -> Surface:
    """Build the risk-geometry surface: sizing rules against firm rulesets.

    Both axes are MEASURED. The trade distribution is fixed by the
    backtest; only the position-sizing rule changes from cell to cell, so
    every cell is a genuine simulation over the real trades.

    Args:
        pnl_r_array: per-trade ``pnl_r`` values from the real backtest.
        rules: a firm ruleset, or a list of firm rulesets. A single
            ruleset produces a one-element y axis.
        risk_geometries: the risk-geometry names to use. Defaults to
            :data:`DEFAULT_RISK_GEOMETRIES`.
        n_attempts: Monte Carlo attempts per cell.
        seed: RNG seed, so the surface is reproducible.

    Returns:
        A :class:`Surface` whose x, y and z axes are all
        ``AXIS_MEASURED``.

    Raises:
        ImportError: if the Monte Carlo layer is unavailable or has a
            different signature.
    """
    geometries = list(risk_geometries) if risk_geometries else list(DEFAULT_RISK_GEOMETRIES)
    rulesets = _as_list(rules)
    if not rulesets:
        raise ValueError("rules must be a firm ruleset or a non-empty list of them")

    arr = np.asarray(pnl_r_array, dtype=float)
    if arr.size == 0:
        raise ValueError("pnl_r_array is empty; run a backtest first")

    z = np.zeros((len(geometries), len(rulesets)), dtype=float)
    ci_low = np.zeros_like(z)
    ci_high = np.zeros_like(z)

    for i, geometry in enumerate(geometries):
        for j, ruleset in enumerate(rulesets):
            rate = _simulate_pass_rate(
                arr, ruleset, geometry, n_attempts, seed, session_dates
            )
            z[i, j] = rate
            low, high = _wilson(rate * n_attempts, n_attempts)
            ci_low[i, j] = low
            ci_high[i, j] = high

    labels = [_rules_label(r) for r in rulesets]
    spec = SurfaceSpec(
        x_name="risk_geometry",
        x_values=list(geometries),
        x_provenance=AXIS_MEASURED,
        y_name="ruleset",
        y_values=labels,
        y_provenance=AXIS_MEASURED,
        z_name="pass_rate",
        z_provenance=AXIS_MEASURED,
        note=(
            "Each cell is a genuine Monte Carlo simulation over the real "
            "per-trade R-multiple distribution produced by the backtest. "
            "Only the position-sizing rule changes between cells: the "
            "trades, their order and their sizes in R are fixed. Both axes "
            "are therefore measured."
        ),
    )
    axis_labels = {
        "x": "risk geometry",
        "y": "firm ruleset",
        "z": "pass rate",
        "name": "risk_geometry x ruleset",
    }
    return Surface(spec=spec, z=z, axis_labels=axis_labels, ci_low=ci_low, ci_high=ci_high)


def rr_surface(
    pnl_r_array: np.ndarray,
    rules: Any,
    *,
    rr_values: Sequence[float] | None = None,
    risk_geometries: Sequence[str] | None = None,
    n_attempts: int = 2000,
    seed: int = 42,
    measured: bool = True,
    strategy_factory: Any = None,
    df: Any = None,
    session_dates: list[str] | None = None,
) -> Surface:
    """Build the R:R surface: reward-to-risk against risk geometry.

    With ``measured=True`` the R:R axis is MEASURED, but only honestly:
    for every R:R value the strategy is rebuilt with that R:R, the
    backtest is re-run, the new per-trade ``pnl_r`` array is extracted,
    and that array is what gets simulated. This is required because
    changing the R:R changes which trades exist -- a trade that reached
    its target at 2R would not have reached it at 1R.

    With ``measured=False`` the R:R axis is MODELLED: the winners of the
    existing trade array are rescaled so that the mean winner equals the
    target R:R, and the losers are left alone. The trade sequence is not
    regenerated, only the winner sizes are scaled, and the axis is
    labelled ``AXIS_MODELLED`` to say so.

    The z axis (pass rate) is MEASURED in both modes, because the pass
    rate itself always comes from a real simulation over the trade
    distribution that was used.

    Args:
        pnl_r_array: per-trade ``pnl_r`` values from the backtest. Used
            directly in ``measured=False`` mode, and as a fallback
            reference in ``measured=True`` mode (the per-R:R backtests
            replace it).
        rules: the firm ruleset.
        rr_values: the R:R values to use. Defaults to
            :data:`DEFAULT_RR_VALUES`.
        risk_geometries: the risk-geometry names. Defaults to
            :data:`DEFAULT_RISK_GEOMETRIES`.
        n_attempts: Monte Carlo attempts per cell.
        seed: RNG seed.
        measured: ``True`` for the honest (re-backtested) version,
            ``False`` for the fast modelled approximation.
        strategy_factory: callable returning a fresh strategy instance.
            Required when ``measured=True``.
        df: the bar frame to backtest. Required when ``measured=True``.

    Returns:
        A :class:`Surface`.

    Raises:
        ValueError: if ``measured=True`` and ``strategy_factory`` or
            ``df`` is missing, or if the strategy cannot be made to use
            the target R:R.
        ImportError: if the Monte Carlo layer is unavailable.
    """
    rrs = [float(v) for v in (rr_values if rr_values else DEFAULT_RR_VALUES)]
    geometries = list(risk_geometries) if risk_geometries else list(DEFAULT_RISK_GEOMETRIES)
    base_arr = np.asarray(pnl_r_array, dtype=float)
    if base_arr.size == 0:
        raise ValueError("pnl_r_array is empty; run a backtest first")

    if measured:
        if strategy_factory is None or df is None:
            raise ValueError(
                "measured=True requires both strategy_factory and df, because "
                "the R:R axis can only be measured by re-running the backtest "
                "at each R:R value. Either supply them, or call "
                "rr_surface(..., measured=False) to get the modelled "
                "approximation, which is labelled as such."
            )
        from prop_lab.core import run_backtest

        per_rr_arrays: list[np.ndarray] = []
        for rr in rrs:
            strategy = _build_strategy_with_rr(strategy_factory, rr)
            trades = run_backtest(strategy, df)
            if not trades:
                raise ValueError(
                    f"the backtest produced no trades at R:R {rr}; the R:R "
                    "axis cannot be measured on an empty trade list"
                )
            per_rr_arrays.append(np.array([t.pnl_r for t in trades], dtype=float))
    else:
        per_rr_arrays = [_scale_winners_to_rr(base_arr, rr) for rr in rrs]

    z = np.zeros((len(rrs), len(geometries)), dtype=float)
    ci_low = np.zeros_like(z)
    ci_high = np.zeros_like(z)
    for i, arr in enumerate(per_rr_arrays):
        for j, geometry in enumerate(geometries):
            rate = _simulate_pass_rate(
                arr, rules, geometry, n_attempts, seed, session_dates
            )
            z[i, j] = rate
            low, high = _wilson(rate * n_attempts, n_attempts)
            ci_low[i, j] = low
            ci_high[i, j] = high

    if measured:
        x_provenance = AXIS_MEASURED
        note = (
            "The R:R axis is measured: for every R:R value the strategy was "
            "rebuilt with that reward-to-risk and the backtest was re-run, "
            "so each cell simulates the trades that actually existed at that "
            "R:R. The cost is one backtest per R:R value."
        )
    else:
        x_provenance = AXIS_MODELLED
        note = (
            "The R:R axis is modelled. The trade sequence was not "
            "regenerated, only the winner sizes were scaled so that the mean "
            "winner equals the target R:R; losers are unchanged. This is a "
            "fast approximation, not a backtest result, and it misses the "
            "fact that a different R:R produces a different set of trades."
        )

    spec = SurfaceSpec(
        x_name="rr_ratio",
        x_values=rrs,
        x_provenance=x_provenance,
        y_name="risk_geometry",
        y_values=list(geometries),
        y_provenance=AXIS_MEASURED,
        z_name="pass_rate",
        z_provenance=AXIS_MEASURED,
        note=note,
    )
    axis_labels = {
        "x": "R:R ratio",
        "y": "risk geometry",
        "z": "pass rate",
        "name": "rr_ratio x risk_geometry",
    }
    return Surface(spec=spec, z=z, axis_labels=axis_labels, ci_low=ci_low, ci_high=ci_high)


def winrate_surface(
    pnl_r_array: np.ndarray,
    rules: Any,
    *,
    win_rates: Sequence[float] | None = None,
    rr_values: Sequence[float] | None = None,
    n_attempts: int = 2000,
    seed: int = 42,
    session_dates: list[str] | None = None,
    operating_point: tuple[float, float] | None = None,
    operating_point_label: str = "this strategy",
) -> Surface:
    """Build the win-rate surface: assumed win rate against R:R.

    BOTH AXES ARE MODELLED, ALWAYS. This is not optional and not a
    default that can be turned off. A win rate cannot be re-backtested:
    forcing a strategy to win 55 percent of the time when its real
    backtest produced 41 percent means fabricating trades that never
    happened. Splitting a real trade distribution into wins and losses
    and re-drawing them at a different rate is a model, and the surface
    is labelled accordingly.

    Each cell is built from a two-point synthetic distribution: with
    probability ``win_rate`` a trade is ``+rr``, otherwise it is ``-1``.
    The synthetic array has 1000 entries: ``-1.0`` everywhere, with the
    first ``round(win_rate * 1000)`` entries set to ``+rr``, then shuffled
    with a generator seeded from ``seed`` so the result is reproducible.
    The pass rate on that array is a real simulation and is therefore
    labelled MEASURED on the z axis.

    The shape of this surface -- pass rate as a function of win rate and
    R:R -- is the visualisation popularised by DeltaTrend Trading on
    TikTok and YouTube using Topstep rules. The idea is sound; the
    caveat is that its win-rate axis is parametric by construction.

    Args:
        pnl_r_array: the real per-trade ``pnl_r`` values. They are not
            used to build the cells (that would mix a real distribution
            with an assumed one); they are accepted so the call signature
            matches the other surface builders.
        rules: the firm ruleset.
        win_rates: the assumed win rates. Defaults to
            :data:`DEFAULT_WIN_RATES`.
        rr_values: the R:R values. Defaults to :data:`DEFAULT_RR_VALUES`.
        n_attempts: Monte Carlo attempts per cell.
        seed: RNG seed.
        session_dates: optional per-trade session labels, forwarded to the
            Monte Carlo layer so this surface uses the same resampling as
            the headline number.
        operating_point: optional ``(win_rate, rr_ratio)`` of the strategy
            that produced ``pnl_r_array``. When given, the surface records
            it so the renderer can mark where the user actually sits on a
            grid they did not choose.
        operating_point_label: how to refer to the operating point in
            labels and warnings.

    Returns:
        A :class:`Surface` whose x and y axes are ``AXIS_MODELLED``.

    Raises:
        ImportError: if the Monte Carlo layer is unavailable.
    """
    rates = [float(v) for v in (win_rates if win_rates else DEFAULT_WIN_RATES)]
    rrs = [float(v) for v in (rr_values if rr_values else DEFAULT_RR_VALUES)]

    z = np.zeros((len(rates), len(rrs)), dtype=float)
    ci_low = np.zeros_like(z)
    ci_high = np.zeros_like(z)

    for i, rate in enumerate(rates):
        for j, rr in enumerate(rrs):
            synthetic = _synthetic_two_point_pnl(rate, rr, seed)
            pass_rate = _simulate_pass_rate(
                synthetic, rules, "frac_2pct", n_attempts, seed, session_dates
            )
            z[i, j] = pass_rate
            low, high = _wilson(pass_rate * n_attempts, n_attempts)
            ci_low[i, j] = low
            ci_high[i, j] = high

    spec = SurfaceSpec(
        x_name="win_rate",
        x_values=rates,
        x_provenance=AXIS_MODELLED,
        y_name="rr_ratio",
        y_values=rrs,
        y_provenance=AXIS_MODELLED,
        z_name="pass_rate",
        z_provenance=AXIS_MEASURED,
        note=(
            "This is a parametric surface, not a backtest result. Both the "
            "win-rate axis and the R:R axis are modelled: each cell assumes "
            "a two-point trade distribution (+rr with probability win_rate, "
            "-1R otherwise) rather than the real trade distribution. The "
            "win-rate axis cannot be measured from a backtest at all, "
            "because forcing a different win rate means fabricating trades "
            "that never happened. The pass rate on the z axis is a real "
            "simulation over the assumed distribution. The shape of this "
            "surface was popularised by DeltaTrend Trading on TikTok and "
            "YouTube using Topstep rules; the caveat there, and here, is "
            "that the win-rate axis is parametric by construction."
        ),
    )
    axis_labels = {
        "x": "assumed win rate",
        "y": "R:R ratio",
        "z": "pass rate",
        "name": "win_rate x rr_ratio",
    }
    if operating_point is not None:
        op_win, op_rr = float(operating_point[0]), float(operating_point[1])
        spec.operating_point = (op_win, op_rr)
        spec.operating_point_label = operating_point_label
        spec.note += (
            f" The strategy that produced the real trade distribution sits "
            f"at win rate {op_win:.1%} and R:R {op_rr:.2f}. The grid may not "
            f"contain that point; where it does not, the marked position is "
            f"an interpolation between the nearest cells and the surface "
            f"value there is NOT the measured pass rate for the strategy."
        )
    return Surface(spec=spec, z=z, axis_labels=axis_labels, ci_low=ci_low, ci_high=ci_high)


def _synthetic_two_point_pnl(win_rate: float, rr: float, seed: int) -> np.ndarray:
    """Build a synthetic two-point ``pnl_r`` array for one win-rate cell.

    The array holds ``_WINRATE_SYNTH_N`` entries: ``-1.0`` everywhere,
    except the first ``round(win_rate * n)`` entries which are set to
    ``+rr``, after which the array is shuffled with a generator seeded
    from ``seed``.

    Args:
        win_rate: assumed probability that a trade wins, in ``[0, 1]``.
        rr: the reward-to-risk of a winning trade.
        seed: RNG seed.

    Returns:
        A numpy array of length :data:`_WINRATE_SYNTH_N`.
    """
    n = _WINRATE_SYNTH_N
    arr = np.full(n, -1.0, dtype=float)
    n_wins = int(round(win_rate * n))
    if n_wins > 0:
        arr[:n_wins] = float(rr)
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    return arr


# ----------------------------------------------------------------------
# Plain-text rendering
# ----------------------------------------------------------------------
def surface_table(surfaces: Sequence[Surface]) -> str:
    """Render every surface as a plain-text table.

    The output lists, for each surface, its name, the x and y axes with
    their provenance tags, the z axis with its provenance tag, and the
    surface note. There is no emoji and no colour, so the table can be
    pasted into a terminal, a log or a report as-is.

    Args:
        surfaces: the surfaces to render. May be empty.

    Returns:
        A plain-text string.
    """
    if not surfaces:
        return "No surfaces were built for this run."

    lines: list[str] = []
    for surface in surfaces:
        spec = surface.spec
        name = surface.axis_labels.get("name", spec.title)
        lines.append(f"Surface: {name}")
        lines.append(f"  x axis: {spec.x_name} [{spec.x_provenance}]")
        lines.append(f"  y axis: {spec.y_name} [{spec.y_provenance}]")
        lines.append(f"  z axis: {spec.z_name} [{spec.z_provenance}]")
        lines.append(f"  x values: {', '.join(str(v) for v in spec.x_values)}")
        lines.append(f"  y values: {', '.join(str(v) for v in spec.y_values)}")
        if spec.operating_point is not None:
            x_val, y_val = spec.operating_point
            x_values = [float(v) for v in spec.x_values]
            y_values = [float(v) for v in spec.y_values]
            on_x = bool(x_values) and min(x_values) <= x_val <= max(x_values)
            on_y = bool(y_values) and min(y_values) <= y_val <= max(y_values)
            if on_x and on_y:
                lines.append(
                    f"  operating point: {spec.operating_point_label} is at "
                    f"{x_val:.0%} on {spec.x_name}, {y_val:.2f} on {spec.y_name} "
                    f"(on this grid)"
                )
            else:
                missing = []
                if not on_x:
                    missing.append(spec.x_name)
                if not on_y:
                    missing.append(spec.y_name)
                lines.append(
                    f"  operating point: {spec.operating_point_label} is at "
                    f"{x_val:.0%} on {spec.x_name}, {y_val:.2f} on {spec.y_name} "
                    f"-- OFF this grid on {', '.join(missing)}. The nearest cell "
                    f"is an interpolation, not this strategy's measured "
                    f"result."
                )
        if spec.note:
            lines.append(f"  note: {spec.note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
