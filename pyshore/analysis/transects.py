"""
pyshore.analysis.transects
===========================
Generate perpendicular transects from a baseline shoreline,
or load existing transects from a shapefile.
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point


def generate_transects(
    baseline: gpd.GeoDataFrame,
    spacing: float = 100.0,
    length: float = 2000.0,
    smoothing_window: int = 5,
) -> gpd.GeoDataFrame:
    """
    Generate shore-perpendicular transects from a baseline GeoDataFrame.

    Parameters
    ----------
    baseline : GeoDataFrame
        Baseline shoreline (LineString or MultiLineString features).
    spacing : float
        Along-shore spacing between transect origins (metres).
    length : float
        Total transect length; the transect extends length/2 on each side
        of the baseline so it crosses both landward and seaward.
    smoothing_window : int
        Number of baseline segments to average when computing perpendicular
        direction (reduces jagged transects on noisy baselines).

    Returns
    -------
    GeoDataFrame
        Transects with columns: transect_id, geometry.
    """
    transects = []
    tid = 0

    for geom in baseline.geometry:
        # Explode MultiLineString into individual parts
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        elif geom.geom_type == "LineString":
            lines = [geom]
        else:
            continue

        for line in lines:
            distances = np.arange(0, line.length, spacing)

            for dist in distances:
                origin = line.interpolate(dist)

                # Smooth perpendicular direction over a small window
                half_w = smoothing_window / 2.0
                d0 = max(0, dist - half_w)
                d1 = min(line.length - 1e-6, dist + half_w)

                p0 = line.interpolate(d0).coords[0]
                p1 = line.interpolate(d1).coords[0]

                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                norm = np.hypot(dx, dy)
                if norm < 1e-9:
                    continue

                # Perpendicular unit vector (rotate 90°)
                ux, uy = -dy / norm, dx / norm

                half = length / 2.0
                ox, oy = origin.x, origin.y
                start = (ox - ux * half, oy - uy * half)
                end   = (ox + ux * half, oy + uy * half)

                transects.append({
                    "transect_id": tid,
                    "geometry": LineString([start, end]),
                    "origin_x": ox,
                    "origin_y": oy,
                })
                tid += 1

    if not transects:
        raise ValueError("No transects generated — check baseline geometry and spacing.")

    gdf = gpd.GeoDataFrame(transects, crs=baseline.crs)
    print(f"[Transects] Generated {len(gdf)} transects (spacing={spacing} m, length={length} m)")
    return gdf


def load_transects(shapefile_path: str, target_crs: str | int | None = None) -> gpd.GeoDataFrame:
    """
    Load pre-existing transects from a shapefile.

    Parameters
    ----------
    shapefile_path : str
        Path to the transects shapefile.
    target_crs : optional
        Reproject to this CRS if provided.

    Returns
    -------
    GeoDataFrame
    """
    gdf = gpd.read_file(shapefile_path)
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    if "transect_id" not in gdf.columns:
        gdf["transect_id"] = gdf.index
    print(f"[Transects] Loaded {len(gdf)} transects from {shapefile_path}")
    return gdf
