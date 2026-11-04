"""Tests for the surface mesh geometry guard.

These cover the rule that a surface needs at least two values on BOTH
axes to be drawable, and the two guards that enforce it: the check at
build time and the check after rendering.

The background is a real defect. ``matplotlib``'s ``plot_surface``
builds its quads from consecutive index pairs, so an axis of length 1
produces no quads, and it does not raise when it produces none. The
result is a figure with a correct title, a correct colour bar and
correct axis limits, and an entirely empty plotting area.

The risk-geometry surface has one firm ruleset on its y axis by
default, so the surface the report leads with rendered empty.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from prop_lab.surface import (
    MIN_MESH_POINTS_PER_AXIS,
    Surface,
    SurfaceSpec,
    assert_mesh_is_drawable,
    assert_rendered_surface_has_geometry,
    build_mesh_grid,
    mesh_point_count,
)


# ----------------------------------------------------------------------
# mesh_point_count: the rule, stated as a number
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "rows,cols,expected",
    [
        (1, 1, 0),
        (1, 2, 0),
        (1, 5, 0),
        (2, 1, 0),
        (5, 1, 0),
        (2, 2, 1),
        (2, 5, 1),
        (5, 2, 1),
        (5, 5, 1),
        (7, 6, 1),
    ],
)
def test_mesh_point_count_matches_what_matplotlib_draws(rows, cols, expected):
    """A grid with fewer than two values on an axis draws nothing.

    The expected counts mirror matplotlib's default ``rstride``/``cstride``
    of 10, which means a grid smaller than 11 points on an axis is
    sampled down to a single quad.
    """
    assert mesh_point_count(rows, cols) == expected


def test_mesh_point_count_grows_when_the_grid_exceeds_the_default_stride():
    """Past 11 points on an axis the default stride starts subdividing."""
    assert mesh_point_count(12, 12) > 1


def test_mesh_point_count_is_zero_for_every_degenerate_axis():
    """No combination with a single-value axis can be meshed."""
    for rows in range(1, 6):
        for cols in range(1, 6):
            if rows < 2 or cols < 2:
                assert mesh_point_count(rows, cols) == 0


# ----------------------------------------------------------------------
# build_mesh_grid
# ----------------------------------------------------------------------
def test_build_mesh_grid_leaves_a_meshable_grid_untouched():
    z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mesh = build_mesh_grid(z)
    assert mesh.z.shape == z.shape
    assert np.array_equal(mesh.z, z)
    assert mesh.widened_x is False
    assert mesh.widened_y is False
    assert mesh.original_shape == (2, 3)


def test_build_mesh_grid_widens_a_single_y_value():
    """The single-ruleset case: the y axis has one value."""
    z = np.array([[0.223], [0.231], [0.237], [0.216], [0.236]])
    mesh = build_mesh_grid(z)
    assert mesh.z.shape == (5, 2)
    assert mesh.widened_y is True
    assert mesh.widened_x is False
    # The widened column must repeat the original numbers exactly.
    assert np.array_equal(mesh.z[:, 0], z[:, 0])
    assert np.array_equal(mesh.z[:, 1], z[:, 0])
    # The tick positions stay at the ORIGINAL axis values.
    assert np.array_equal(mesh.y_positions, np.arange(1, dtype=float))
    assert np.array_equal(mesh.x_positions, np.arange(5, dtype=float))


def test_build_mesh_grid_widens_a_single_x_value():
    z = np.array([[1.0, 2.0, 3.0, 4.0]])
    mesh = build_mesh_grid(z)
    assert mesh.z.shape == (2, 4)
    assert mesh.widened_x is True
    assert np.array_equal(mesh.z[0, :], z[0, :])
    assert np.array_equal(mesh.z[1, :], z[0, :])


def test_build_mesh_grid_widens_both_axes():
    z = np.array([[0.5]])
    mesh = build_mesh_grid(z)
    assert mesh.z.shape == (2, 2)
    assert mesh.widened_x and mesh.widened_y
    assert mesh.z[0, 0] == mesh.z[1, 1] == 0.5


def test_build_mesh_grid_preserves_the_values_exactly():
    """Widening must not alter a single number.

    The surface that gets drawn has to be the surface that was measured.
    A widening that changed a value would turn a measured axis into a
    modelled one without saying so.
    """
    rng = np.random.default_rng(7)
    z = rng.random((5, 1))
    mesh = build_mesh_grid(z)
    assert set(np.unique(mesh.z)) == set(np.unique(z))


def test_build_mesh_grid_rejects_an_empty_axis():
    with pytest.raises(ValueError, match="empty axis"):
        build_mesh_grid(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="empty axis"):
        build_mesh_grid(np.zeros((3, 0)))


def test_build_mesh_grid_rejects_a_non_2d_grid():
    with pytest.raises(ValueError, match="two-dimensional"):
        build_mesh_grid(np.zeros(5))


# ----------------------------------------------------------------------
# assert_mesh_is_drawable: the build-time guard
# ----------------------------------------------------------------------
def test_assert_mesh_is_drawable_accepts_a_normal_grid():
    assert_mesh_is_drawable(np.zeros((5, 5)), context="test")


def test_assert_mesh_is_drawable_accepts_a_degenerate_grid():
    """A single-value axis is legitimate and is widened, not rejected.

    The risk-geometry surface has one ruleset by default. Rejecting that
    would break the ordinary case rather than the broken one.
    """
    assert_mesh_is_drawable(np.zeros((5, 1)), context="test")
    assert_mesh_is_drawable(np.zeros((1, 5)), context="test")


def test_assert_mesh_is_drawable_rejects_an_empty_axis():
    with pytest.raises(ValueError, match="empty axis"):
        assert_mesh_is_drawable(np.zeros((0, 3)), context="test")
    with pytest.raises(ValueError, match="empty axis"):
        assert_mesh_is_drawable(np.zeros((3, 0)), context="test")


def test_assert_mesh_is_drawable_rejects_a_non_2d_grid():
    with pytest.raises(ValueError, match="two-dimensional"):
        assert_mesh_is_drawable(np.zeros(4), context="test")


def test_assert_mesh_is_drawable_message_names_the_context():
    with pytest.raises(ValueError, match="my_surface"):
        assert_mesh_is_drawable(np.zeros((0, 2)), context="my_surface")


# ----------------------------------------------------------------------
# assert_rendered_surface_has_geometry: the after-the-fact guard
# ----------------------------------------------------------------------
def _surface(n_x: int, n_y: int) -> Surface:
    spec = SurfaceSpec(
        x_name="risk_geometry",
        x_values=[f"g{i}" for i in range(n_x)],
        x_provenance="measured",
        y_name="ruleset",
        y_values=[f"r{j}" for j in range(n_y)],
        y_provenance="measured",
        z_name="pass_rate",
        z_provenance="measured",
    )
    z = np.array(
        [[i * 10 + j for j in range(n_y)] for i in range(n_x)], dtype=float
    ) / 100.0
    return Surface(spec=spec, z=z, axis_labels={})


def test_assert_rendered_surface_has_geometry_accepts_a_drawn_surface():
    fig = _surface(5, 1).to_3d()
    try:
        assert_rendered_surface_has_geometry(fig, context="test")
    finally:
        plt.close(fig)


def test_assert_rendered_surface_has_geometry_rejects_an_empty_figure():
    """The general safety net.

    This does not care why the figure is empty. It checks the thing
    that matters: whether any polygons were drawn. A figure that fails
    this check must never reach a report, because a correct title and a
    correct colour bar make an empty plot look legitimate.
    """
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    # A surface with no geometry at all, which is what the bug produced.
    ax.plot([], [], [])
    try:
        with pytest.raises(RuntimeError, match="no geometry"):
            assert_rendered_surface_has_geometry(fig, context="test")
    finally:
        plt.close(fig)


def test_assert_rendered_surface_has_geometry_rejects_a_figure_with_no_axes():
    fig = plt.figure()
    try:
        with pytest.raises(RuntimeError, match="no axes"):
            assert_rendered_surface_has_geometry(fig, context="test")
    finally:
        plt.close(fig)


def test_assert_rendered_surface_has_geometry_rejects_a_2d_empty_figure():
    """A 2D figure with nothing in it must also be caught."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([], [])
    try:
        with pytest.raises(RuntimeError, match="no geometry"):
            assert_rendered_surface_has_geometry(fig, context="test")
    finally:
        plt.close(fig)


# ----------------------------------------------------------------------
# The rule applies to every surface, whatever shape it has
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n_x,n_y", [(1, 1), (1, 4), (4, 1), (2, 2), (5, 1), (1, 5)])
def test_every_surface_shape_renders_something(n_x, n_y):
    """No surface shape may produce an empty figure.

    This is the property the whole guard exists to provide: whatever
    grid a builder hands over, the figure has geometry in it.
    """
    fig = _surface(n_x, n_y).to_3d()
    try:
        fig.canvas.draw()
        count = assert_rendered_surface_has_geometry(fig, context="test")
        assert count > 0
    finally:
        plt.close(fig)


def test_min_mesh_points_is_two():
    """The constant is the reason the rule exists; pin it."""
    assert MIN_MESH_POINTS_PER_AXIS == 2
