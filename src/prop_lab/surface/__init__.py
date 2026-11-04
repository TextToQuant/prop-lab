"""The honesty layer: surfaces whose axes are explicitly labelled.

See :mod:`prop_lab.surface.surfaces` for the full explanation of the
measured-versus-modelled distinction, and
:mod:`prop_lab.surface.geometry` for the rule that makes sure a surface
can actually be drawn before it is rendered.
"""
from __future__ import annotations

from .geometry import (
    MIN_MESH_POINTS_PER_AXIS,
    MeshGrid,
    assert_mesh_is_drawable,
    build_mesh_grid,
    mesh_point_count,
)
from .surfaces import (
    AXIS_MEASURED,
    AXIS_MODELLED,
    DEFAULT_RR_VALUES,
    DEFAULT_RISK_GEOMETRIES,
    DEFAULT_WIN_RATES,
    Surface,
    SurfaceSpec,
    assert_rendered_surface_has_geometry,
    risk_geometry_surface,
    rr_surface,
    surface_table,
    winrate_surface,
)

__all__ = [
    "AXIS_MEASURED",
    "AXIS_MODELLED",
    "DEFAULT_RR_VALUES",
    "DEFAULT_RISK_GEOMETRIES",
    "DEFAULT_WIN_RATES",
    "MIN_MESH_POINTS_PER_AXIS",
    "MeshGrid",
    "Surface",
    "SurfaceSpec",
    "assert_mesh_is_drawable",
    "assert_rendered_surface_has_geometry",
    "build_mesh_grid",
    "mesh_point_count",
    "risk_geometry_surface",
    "rr_surface",
    "surface_table",
    "winrate_surface",
]
