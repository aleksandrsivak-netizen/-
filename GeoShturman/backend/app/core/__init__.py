"""Navigation core package.

The package contains pure Python algorithm modules that can be imported by
API, CLI, or tests without starting a web framework.
"""

from .dem import (
    DEMData,
    create_synthetic_dem,
    dem_xy_to_geodetic,
    geodetic_to_dem_xy,
    is_inside_dem,
    is_inside_dem_geodetic,
    load_dem,
    sample_dem,
    sample_dem_geodetic,
    sample_profile,
)
from .geodesy import GeoPoint, GeoReference
from .navigation import NavigationSolution, solve_navigation

__all__ = [
    "DEMData",
    "GeoPoint",
    "GeoReference",
    "NavigationSolution",
    "create_synthetic_dem",
    "dem_xy_to_geodetic",
    "geodetic_to_dem_xy",
    "is_inside_dem",
    "is_inside_dem_geodetic",
    "load_dem",
    "sample_dem",
    "sample_dem_geodetic",
    "sample_profile",
    "solve_navigation",
]
