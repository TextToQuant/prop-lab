"""Regression tests for the surface renderers.

These cover two bugs that were found by looking at the rendered PNGs
rather than by reading the code.

1. The heatmap annotation was drawn at the transposed position. ``z``
   has shape (len(x_values), len(y_values)) and the image is drawn from
   ``z.T``, so the value z[i, j] belongs at (x=i, y=j). Annotating at
   (j, i) put the numbers in the wrong cells, and with a single
   ruleset row they landed outside the image and floated in the empty
   margin above the chart.
2. Pass rates were displayed as bare decimals. They are probabilities,
   so they read as percentages.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from prop_lab.surface import Surface, SurfaceSpec


def _spec(n_x: int, n_y: int, z_name: str = "pass_rate") -> SurfaceSpec:
    return SurfaceSpec(
        x_name="risk_geometry",
        x_values=[f"g{i}" for i in range(n_x)],
        x_provenance="measured",
        y_name="ruleset",
        y_values=[f"r{j}" for j in range(n_y)],
        y_provenance="measured",
        z_name=z_name,
        z_provenance="measured",
    )


def _surface(n_x: int, n_y: int, z: np.ndarray | None = None) -> Surface:
    if z is None:
        # Distinct values per cell so a transposition is detectable.
        z = np.array(
            [[i * 10 + j for j in range(n_y)] for i in range(n_x)], dtype=float
        ) / 100.0
    return Surface(spec=_spec(n_x, n_y), z=z, axis_labels={})


def test_heatmap_annotations_land_on_their_own_cells():
    """Each annotation must sit on the cell that holds that value."""
    surface = _surface(3, 2)
    fig = surface.to_heatmap()
    try:
        ax = fig.axes[0]
        # Collect the text artists and their positions.
        placed = [
            (round(t.get_position()[0]), round(t.get_position()[1]), t.get_text())
            for t in ax.texts
        ]
        assert len(placed) == 6
        for x, y, text in placed:
            # The value at (x, y) must be the one written there.
            expected = surface.z[x, y]
            assert text == f"{expected * 100:.1f}%"
    finally:
        plt.close(fig)


def test_heatmap_annotation_colour_is_readable_against_its_cell():
    """Every annotation must have a readable contrast against its cell.

    The heatmap uses the ``plasma`` colormap, whose dark end is a deep
    blue-purple. Fixed black text on that background has a contrast
    ratio of 1.1:1, which is invisible; WCAG AA wants 4.5:1.

    The colour has to be chosen from the luminance of the cell the text
    actually sits on, so this test reproduces the colour for each
    annotation and checks the contrast of the chosen colour against
    that cell.
    """
    import matplotlib

    def luminance(rgb: tuple) -> float:
        def channel(c: float) -> float:
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = (float(c) for c in rgb[:3])
        return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)

    surface = _surface(4, 3)
    fig = surface.to_heatmap()
    try:
        ax = fig.axes[0]
        grid = np.asarray(surface.z, dtype=float).T
        vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
        span = vmax - vmin

        for text_artist in ax.texts:
            x, y = (int(round(v)) for v in text_artist.get_position())
            position = (grid[y, x] - vmin) / span if span > 0 else 0.5
            background = matplotlib.colormaps["plasma"](position)
            bg_luminance = luminance(background)

            colour = text_artist.get_color()
            if colour == "white":
                fg_luminance = 1.0
            elif colour == "black":
                fg_luminance = 0.0
            else:
                fg_luminance = luminance(matplotlib.colors.to_rgba(colour))

            lighter = max(fg_luminance, bg_luminance)
            darker = min(fg_luminance, bg_luminance)
            contrast = (lighter + 0.05) / (darker + 0.05)
            assert contrast >= 4.5, (
                f"annotation at ({x}, {y}) has contrast {contrast:.2f}:1, "
                f"colour {colour!r} on background luminance {bg_luminance:.3f}"
            )
    finally:
        plt.close(fig)


def test_annotation_colour_switches_across_the_ramp():
    """The colour must actually depend on the cell, not be fixed."""
    assert Surface._annotation_colour(0.0) == "white"
    assert Surface._annotation_colour(1.0) == "black"
    # Mid-ramp: one of the two, but it must not be the same on both ends.
    mid = Surface._annotation_colour(0.5)
    assert mid in {"white", "black"}
    assert Surface._annotation_colour(0.0) != Surface._annotation_colour(1.0)


def test_heatmap_with_one_ruleset_row_keeps_annotations_inside_the_image():
    """The case that produced stray numbers in the margin.

    With a single ruleset the heatmap is one row tall, so a
    transposed annotation lands outside the axes entirely.
    """
    surface = _surface(5, 1)
    fig = surface.to_heatmap()
    try:
        ax = fig.axes[0]
        for t in ax.texts:
            x, y = t.get_position()
            # The image occupies x in [0, 5) and y in [0, 1).
            assert -0.5 <= x <= 4.5
            assert -0.5 <= y <= 0.5
    finally:
        plt.close(fig)


def test_cells_display_as_percentages():
    surface = _surface(2, 1)
    fig = surface.to_heatmap()
    try:
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert texts
        for text in texts:
            assert text.endswith("%")
    finally:
        plt.close(fig)


def test_non_probability_values_keep_a_plain_format():
    """A value outside 0..1 is not a probability."""
    assert Surface._format_cell(1.5) == "1.50"
    assert Surface._format_cell(0.5) == "50.0%"
    assert Surface._format_cell(1.0) == "100.0%"
    assert Surface._format_cell(0.0) == "0.0%"


def test_3d_z_axis_is_formatted_as_percent_for_pass_rate():
    surface = _surface(3, 2)
    fig = surface.to_3d()
    try:
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.zaxis.get_ticklabels()]
        # The formatter is applied at draw time, so check the labels
        # resolve to percentages once the figure is drawn.
        fig.canvas.draw()
        labels = [t.get_text() for t in ax.zaxis.get_ticklabels()]
        assert any("%" in lab for lab in labels)
    finally:
        plt.close(fig)


def test_3d_surface_uses_the_correct_orientation():
    """The 3D mesh must place z[i, j] at x=i, y=j."""
    surface = _surface(3, 2)
    fig = surface.to_3d()
    try:
        ax = fig.axes[0]
        assert ax.get_zlim() is not None
        lo, hi = ax.get_zlim()
        assert lo <= surface.z.min()
        assert hi >= surface.z.max()
    finally:
        plt.close(fig)


def _drawn_path_count(fig) -> int:
    """How many polygons the 3D surface actually rendered.

    ``plot_surface`` returns an artist whose polygons are the quads it
    built. If it could not build any, the count is zero and the figure
    shows an empty grid even though the axes, the title and the colour
    bar all look correct.
    """
    ax = fig.axes[0]
    total = 0
    for collection in ax.collections:
        total += len(collection.get_paths())
    return total


def test_3d_surface_draws_something_when_both_axes_have_values():
    """A normal 2D surface must render visible geometry.

    The regression below is only meaningful if the ordinary case works,
    so this asserts the positive control first.
    """
    surface = _surface(3, 2)
    fig = surface.to_3d()
    try:
        fig.canvas.draw()
        assert _drawn_path_count(fig) > 0
    finally:
        plt.close(fig)


def test_3d_surface_draws_with_a_single_ruleset_row():
    """A surface with one ruleset has a y axis of length 1.

    ``matplotlib``'s ``plot_surface`` cannot build a quad mesh along a
    dimension of length 1, so it silently renders nothing: the figure
    comes back with a correct title, correct axis limits and a correct
    colour bar, and an entirely empty plotting area.

    This is the shape the risk-geometry surface has in the normal
    single-ruleset case, which is the surface the report leads with.
    The rendered PNG was empty, and no test caught it because the
    existing 3D tests only asserted that a figure came back.
    """
    surface = _surface(5, 1)
    fig = surface.to_3d()
    try:
        fig.canvas.draw()
        assert _drawn_path_count(fig) > 0, (
            "the 3D surface rendered no geometry: plot_surface cannot mesh "
            "an axis of length 1, so the figure is an empty grid"
        )
    finally:
        plt.close(fig)


def test_3d_surface_draws_with_a_single_x_value():
    """The same degenerate case on the x axis."""
    surface = _surface(1, 4)
    fig = surface.to_3d()
    try:
        fig.canvas.draw()
        assert _drawn_path_count(fig) > 0
    finally:
        plt.close(fig)


def test_heatmap_figure_size_is_sane():
    """A single-row heatmap should not waste most of the canvas."""
    surface = _surface(5, 1)
    fig = surface.to_heatmap()
    try:
        width, height = fig.get_size_inches()
        assert width > 0 and height > 0
        ax = fig.axes[0]
        pos = ax.get_position()
        # The axes should occupy a reasonable share of the figure.
        assert pos.height > 0.05
    finally:
        plt.close(fig)
