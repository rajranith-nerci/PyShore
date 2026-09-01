"""
pyshore.analysis.intersection
===============================
Fast transect–shoreline intersection using spatial indexing.

For each transect the function finds all shoreline features that intersect it,
resolves the intersection geometry, and keeps the single best point (closest to
the seaward end of the transect, or to the origin if orientation is unknown).
"""

from __future__ import annotations

import os
import json
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, GeometryCollection


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def _extract_points(geom) -> list:
    """Extract all Point objects from any intersection geometry."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [geom]
    if geom.geom_type == "MultiPoint":
        return list(geom.geoms)
    if geom.geom_type in ("GeometryCollection", "MultiLineString", "MultiPolygon"):
        pts = []
        for g in geom.geoms:
            pts.extend(_extract_points(g))
        return pts
    # LineString / Polygon intersection — sample midpoint as fallback
    if geom.geom_type == "LineString":
        return [geom.interpolate(0.5, normalized=True)]
    return []


# ---------------------------------------------------------------------------
# Main intersection function
# ---------------------------------------------------------------------------

def intersect_transects(
    transects: gpd.GeoDataFrame,
    shorelines: List[gpd.GeoDataFrame],
    keep: str = "seaward",
) -> gpd.GeoDataFrame:
    """
    Intersect transects with a list of annual shoreline GeoDataFrames.

    Parameters
    ----------
    transects : GeoDataFrame
        Transect lines with column 'transect_id'.
    shorelines : list of GeoDataFrame
        One GeoDataFrame per year, each must have a 'year' column.
        May also have 'sensor' and 'uncertainty_m' columns.
    keep : {'seaward', 'origin', 'all_mean'}
        How to resolve multiple intersection points per transect-year:
        - 'seaward' : keep the point farthest from the transect start
        - 'origin'  : keep the point closest to the transect midpoint
        - 'all_mean': average all intersection distances

    Returns
    -------
    GeoDataFrame
        Columns: transect_id, year, geometry (Point), distance_m,
                 sensor (if available), uncertainty_m (if available)
    """
    records = []

    # Diagnostic: print CRS and bounds for first shoreline vs transects
    if shorelines:
        sl0 = shorelines[0]
        print(f"  [CRS check] Transects: {transects.crs}  |  Shoreline: {sl0.crs}")
        tb = transects.total_bounds
        sb = sl0.total_bounds
        print(f"  [Bounds] Transects : xmin={tb[0]:.1f} ymin={tb[1]:.1f} "
              f"xmax={tb[2]:.1f} ymax={tb[3]:.1f}")
        print(f"  [Bounds] Shoreline : xmin={sb[0]:.1f} ymin={sb[1]:.1f} "
              f"xmax={sb[2]:.1f} ymax={sb[3]:.1f}")
        # CRS mismatch guard
        if transects.crs != sl0.crs:
            print("  [WARNING] CRS mismatch — reprojecting shorelines to transect CRS")
            shorelines = [s.to_crs(transects.crs) for s in shorelines]

    for shoreline in shorelines:
        year = int(shoreline["year"].iloc[0])
        # Handle truncated column names from Shapefile (uncertainty_m → uncertaint)
        has_sensor = "sensor" in shoreline.columns
        has_unc = ("uncertainty_m" in shoreline.columns
                   or "uncertaint" in shoreline.columns)
        unc_col = ("uncertainty_m" if "uncertainty_m" in shoreline.columns
                   else "uncertaint" if "uncertaint" in shoreline.columns else None)

        # Explode multi-part geometries so spatial index works correctly
        shoreline = shoreline.explode(index_parts=False).reset_index(drop=True)

        # Spatial index over the shoreline features
        sindex = shoreline.sindex

        for _, row in transects.iterrows():
            tid = int(row["transect_id"])
            tran_geom = row.geometry

            # Candidate shoreline features
            candidate_idx = list(sindex.query(tran_geom, predicate="intersects"))
            if not candidate_idx:
                continue

            all_pts: list = []
            for sid in candidate_idx:
                shore_geom = shoreline.iloc[sid].geometry
                inter = shore_geom.intersection(tran_geom)
                all_pts.extend(_extract_points(inter))

            if not all_pts:
                continue

            # Project all points onto the transect line (distance from start)
            distances = [tran_geom.project(pt) for pt in all_pts]

            if keep == "seaward":
                idx = int(np.argmax(distances))
                best_pt = all_pts[idx]
                dist = distances[idx]
            elif keep == "all_mean":
                dist = float(np.mean(distances))
                best_pt = tran_geom.interpolate(dist)
            else:  # 'origin' — closest to midpoint
                mid = tran_geom.length / 2.0
                idx = int(np.argmin([abs(d - mid) for d in distances]))
                best_pt = all_pts[idx]
                dist = distances[idx]

            record = {
                "transect_id": tid,
                "year": year,
                "geometry": best_pt,
                "distance_m": dist,
            }
            if has_sensor:
                record["sensor"] = shoreline.iloc[candidate_idx[0]]["sensor"]
            if has_unc and unc_col:
                record["uncertainty_m"] = float(shoreline.iloc[candidate_idx[0]][unc_col])

            records.append(record)

    if not records:
        raise ValueError(
            "No transect–shoreline intersections found.\n"
            "Common causes:\n"
            "  1. CRS mismatch — check the [CRS check] output above.\n"
            "  2. Spatial gap — transects and shorelines don't overlap "
            "(check [Bounds] output above).\n"
            "  3. Empty shoreline geometry — GEE extraction produced no valid "
            "lines (open the .shp in QGIS to inspect).\n"
            "  4. Transect length too short — increase TRANSECT_LENGTH in "
            "example_run.py so transects reach the water.\n"
            "  5. Baseline orientation — ensure baseline.shp is a line along "
            "the coast (not a polygon)."
        )

    gdf = gpd.GeoDataFrame(records, crs=transects.crs)
    print(f"[Intersection] {len(gdf)} intersection points across "
          f"{gdf['year'].nunique()} years × {gdf['transect_id'].nunique()} transects")
    return gdf


# ---------------------------------------------------------------------------
# Load shorelines from directory
# ---------------------------------------------------------------------------

def load_shorelines_from_dir(
    shoreline_dir: str,
    epsg: int,
    sensor_map: Optional[Dict[int, str]] = None,
    uncertainty_map: Optional[Dict[str, float]] = None,
) -> List[gpd.GeoDataFrame]:
    """
    Load all annual shoreline shapefiles from a directory.

    Parameters
    ----------
    shoreline_dir : str
        Directory containing shapefiles named '{year}.shp'.
    epsg : int
        Target CRS EPSG code.
    sensor_map : dict, optional
        {year: sensor_name} — used to attach uncertainty info if not already in the file.
    uncertainty_map : dict, optional
        {sensor_name: uncertainty_metres} — defaults to SENSOR_UNCERTAINTY.

    Returns
    -------
    list of GeoDataFrame
    """
    from pyshore.config import SENSOR_UNCERTAINTY

    if uncertainty_map is None:
        uncertainty_map = SENSOR_UNCERTAINTY

    # Try to load sensor_map from JSON if not provided
    if sensor_map is None:
        sm_path = os.path.join(shoreline_dir, "sensor_map.json")
        if os.path.exists(sm_path):
            with open(sm_path) as f:
                sensor_map = {int(k): v for k, v in json.load(f).items()}

    shorelines = []
    for fname in sorted(os.listdir(shoreline_dir)):
        if not fname.endswith(".shp"):
            continue
        year_str = fname.replace(".shp", "")
        try:
            year = int(year_str)
        except ValueError:
            continue

        path = os.path.join(shoreline_dir, fname)
        gdf = gpd.read_file(path).to_crs(epsg=epsg)
        gdf["year"] = year

        # Attach sensor / uncertainty if missing
        if "sensor" not in gdf.columns and sensor_map and year in sensor_map:
            gdf["sensor"] = sensor_map[year]
        if "uncertainty_m" not in gdf.columns and "sensor" in gdf.columns:
            gdf["uncertainty_m"] = gdf["sensor"].map(uncertainty_map)

        shorelines.append(gdf)

    if not shorelines:
        raise FileNotFoundError(f"No shoreline shapefiles found in {shoreline_dir}")

    print(f"[Shorelines] Loaded {len(shorelines)} annual shorelines from {shoreline_dir}")
    return shorelines
