"""The honesty layer: surfaces whose axes are explicitly labelled.

See :mod:`prop_lab.surface.surfaces` for the full explanation of the
measured-versus-modelled distinction.
"""
from __future__ import annotations

from .surfaces import (
    AXIS_MEASURED,
    AXIS_MODELLED,
    DEFAULT_RR_VALUES,
    DEFAULT_RISK_GEOMETRIES,
    DEFAULT_WIN_RATES,
    Surface,
    SurfaceSpec,
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
    "Surface",
    "SurfaceSpec",
    "risk_geometry_surface",
    "rr_surface",
    "surface_table",
    "winrate_surface",
]
