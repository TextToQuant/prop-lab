"""Surface mesh geometry: the one place that decides how a grid becomes a mesh.

Why this module exists
----------------------
``matplotlib``'s ``Axes3D.plot_surface`` cannot build a quad mesh along a
dimension of length 1, and it does not say so. Reading the matplotlib
source (``mpl_toolkits/mplot3d/axes3d.py``)::

    row_inds = list(range(0, rows - 1, rstride)) + [rows - 1]
    col_inds = list(range(0, cols - 1, cstride)) + [cols - 1]

With ``rows == 1`` the ``range(0, 0, ...)`` is empty, so ``row_inds``
holds a single index. The quad loop below it pairs consecutive indices,
so with one index there are no pairs, no quads are produced, and the
function returns an empty collection **without raising**.

The result is a figure with a correct title, correct axis limits and a
correct colour bar, and an entirely empty plotting area. It looks
plausible in a thumbnail and says nothing.

This is not a hypothetical shape. The risk-geometry surface has one
firm ruleset on its y axis by default, so the surface the report leads
with was rendered empty.

The rule, stated once
---------------------
**A surface mesh needs at least two values on BOTH axes.** A grid with a
single value on an axis is a legitimate surface -- it is a ribbon, or a
line -- and it is drawn as one.

This module owns that rule so that every surface builder in the package
gets it, and so that a new builder cannot reintroduce the bug by
forgetting a special case.

What this module does NOT do
----------------------------
It does not decide what a surface should contain, and it does not change
any measured or modelled value. It only decides the coordinates used to
draw the grid that has already been computed. Widening a degenerate axis
repeats the same numbers, so the drawn surface is still the surface the
backtest produced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MIN_MESH_POINTS_PER_AXIS",
    "MeshGrid",
    "build_mesh_grid",
    "mesh_point_count",
    "assert_mesh_is_drawable",
]

#: The smallest number of grid points a meshable axis may have.
#:
#: Two, because one produces zero quads in ``plot_surface``. This is not
#: a stylistic choice: it is the minimum for which the renderer emits
#: any geometry at all.
MIN_MESH_POINTS_PER_AXIS = 2


@dataclass(frozen=True)
class MeshGrid:
    """The coordinates and heights used to draw a surface.

    Attributes:
        xx: array of shape ``(n_x, n_y)`` holding the x coordinate of
            every vertex.
        yy: array of the same shape holding the y coordinate.
        z: array of the same shape holding the height of every vertex.
        x_positions: the x coordinates to tick and label on the axis.
            These are the ORIGINAL axis values, not the widened ones.
        y_positions: the y coordinates to tick and label on the axis.
        widened_x: True when the x axis had to be widened to be meshable.
        widened_y: True when the y axis had to be widened to be meshable.
        original_shape: the shape of the grid before widening.
    """

    xx: np.ndarray
    yy: np.ndarray
    z: np.ndarray
    x_positions: np.ndarray
    y_positions: np.ndarray
    widened_x: bool
    widened_y: bool
    original_shape: tuple[int, int]


def mesh_point_count(n_rows: int, n_cols: int) -> int:
    """How many quads ``plot_surface`` will produce for a grid of this size.

    This mirrors matplotlib's own index arithmetic, so the answer is the
    number of polygons the renderer will actually draw.

    Args:
        n_rows: number of rows in the z grid.
        n_cols: number of columns in the z grid.

    Returns:
        The number of quads. Zero when either axis has a single value.
    """
    rows = max(int(n_rows), 0)
    cols = max(int(n_cols), 0)
    if rows < 2 or cols < 2:
        return 0
    row_inds = list(range(0, rows - 1, 10)) + [rows - 1]
    col_inds = list(range(0, cols - 1, 10)) + [cols - 1]
    return (len(row_inds) - 1) * (len(col_inds) - 1)


def build_mesh_grid(z: np.ndarray) -> MeshGrid:
    """Turn a z grid into the coordinates needed to draw it.

    A grid whose x or y axis holds a single value is widened to two by
    repeating that value. The repeated numbers are identical to the
    original ones, so the surface that gets drawn is the surface that
    was computed. A grid that is already meshable is passed through
    untouched.

    Args:
        z: array of shape ``(n_x, n_y)``.

    Returns:
        A :class:`MeshGrid` whose arrays are ready for ``plot_surface``.

    Raises:
        ValueError: if ``z`` is not two-dimensional, or if either axis
            is empty. An empty axis cannot be widened and cannot be
            drawn, so it is an error rather than something to paper over.
    """
    grid = np.asarray(z, dtype=float)
    if grid.ndim != 2:
        raise ValueError(f"z must be two-dimensional, got shape {grid.shape}")
    n_x, n_y = grid.shape
    if n_x == 0 or n_y == 0:
        raise ValueError(
            f"z has shape {grid.shape}, which has an empty axis; "
            "there is nothing to draw"
        )

    original_shape = (n_x, n_y)
    widened_x = n_x < MIN_MESH_POINTS_PER_AXIS
    widened_y = n_y < MIN_MESH_POINTS_PER_AXIS

    out = grid
    if widened_x:
        out = np.repeat(out, MIN_MESH_POINTS_PER_AXIS, axis=0)
    if widened_y:
        out = np.repeat(out, MIN_MESH_POINTS_PER_AXIS, axis=1)

    # The tick positions stay at the ORIGINAL axis values, so a
    # single-value axis is labelled with one label and not with a
    # phantom second one.
    x_positions = np.arange(n_x, dtype=float)
    y_positions = np.arange(n_y, dtype=float)

    xi = np.arange(out.shape[0], dtype=float)
    yi = np.arange(out.shape[1], dtype=float)
    xx, yy = np.meshgrid(xi, yi, indexing="ij")

    return MeshGrid(
        xx=xx,
        yy=yy,
        z=out,
        x_positions=x_positions,
        y_positions=y_positions,
        widened_x=widened_x,
        widened_y=widened_y,
        original_shape=original_shape,
    )


def assert_mesh_is_drawable(z: np.ndarray, *, context: str = "surface") -> None:
    """Raise if a grid cannot be drawn as a mesh.

    This is the standardised guard. It is called by the surface
    builders so that a grid shape which cannot produce geometry is
    rejected at the point where the grid is built, rather than
    producing a PNG that looks fine and says nothing.

    A grid with a single value on one axis IS drawable, because
    :func:`build_mesh_grid` widens it into a ribbon. The only grid that
    cannot be drawn is one with an empty axis, and that is an error
    rather than something to paper over.

    Args:
        z: the grid to check.
        context: a short description of what is being checked, used in
            the error message.

    Raises:
        ValueError: if the grid has an empty axis, or is not
            two-dimensional.
    """
    grid = np.asarray(z, dtype=float)
    if grid.ndim != 2:
        raise ValueError(f"{context}: z must be two-dimensional, got {grid.shape}")
    n_x, n_y = grid.shape
    if n_x == 0 or n_y == 0:
        raise ValueError(
            f"{context}: z has shape {grid.shape}, which has an empty axis; "
            "there is nothing to draw. A grid with a single value on an axis "
            "is fine (it is drawn as a ribbon), but a grid with no values on "
            "one cannot be drawn at all."
        )
