"""
pyshore.export.exporter
========================
Export pyshore results to:
  • CSV              — flat table, compatible with Excel / ArcGIS / QGIS
  • Shapefile (.shp) — transect polylines with metric attribute table
  • GeoPackage (.gpkg) — single-file GIS format, preferred for QGIS

Column naming follows DSAS conventions where possible so results can be
loaded alongside DSAS outputs for comparison.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import geopandas as gpd


# ---------------------------------------------------------------------------
# Helper: shorten column names for Shapefile DBF (10-char limit)
# ---------------------------------------------------------------------------

_SHP_COL_MAP = {
    "transect_id":    "TR_ID",
    "n_obs":          "N_OBS",
    "year_min":       "YR_MIN",
    "year_max":       "YR_MAX",
    "NSM":            "NSM",
    "EPR":            "EPR",
    "SCE":            "SCE",
    "LRR":            "LRR",
    "LRR_intercept":  "LRR_INT",
    "LRR_SE":         "LRR_SE",
    "LRR_CI_lo":      "LRR_CI_L",
    "LRR_CI_hi":      "LRR_CI_H",
    "LRR_R2":         "LRR_R2",
    "LRR_p":          "LRR_P",
    "Trend_LRR":      "TRD_LRR",
    "WLR":            "WLR",
    "WLR_intercept":  "WLR_INT",
    "WLR_SE":         "WLR_SE",
    "WLR_CI_lo":      "WLR_CI_L",
    "WLR_CI_hi":      "WLR_CI_H",
    "WLR_R2":         "WLR_R2",
    "WLR_p":          "WLR_P",
    "Trend_WLR":      "TRD_WLR",
    "LMS":            "LMS",
    "LMS_intercept":  "LMS_INT",
    "LMS_SE":         "LMS_SE",
    "LMS_CI_lo":      "LMS_CI_L",
    "LMS_CI_hi":      "LMS_CI_H",
    "LMS_R2":         "LMS_R2",
    "LMS_p":          "LMS_P",
    "Trend_LMS":      "TRD_LMS",
    "LRR_bs_mean":    "LRR_BS_M",
    "LRR_bs_std":     "LRR_BS_S",
    "LRR_bs_ci_lo":   "LRR_BS_L",
    "LRR_bs_ci_hi":   "LRR_BS_H",
    "WLR_bs_mean":    "WLR_BS_M",
    "WLR_bs_std":     "WLR_BS_S",
    "WLR_bs_ci_lo":   "WLR_BS_L",
    "WLR_bs_ci_hi":   "WLR_BS_H",
    "NSM_unc":        "NSM_UNC",
    "EPR_unc":        "EPR_UNC",
    "years_span":     "YRS_SPAN",
    "origin_x":       "ORIG_X",
    "origin_y":       "ORIG_Y",
}


def _rename_for_shp(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to ≤10 characters for Shapefile DBF compatibility."""
    rename_map = {k: v for k, v in _SHP_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Truncate any remaining columns that are still > 10 chars, avoiding duplicates
    final_map = {}
    seen = set(df.columns)
    for col in list(df.columns):
        if len(col) > 10:
            base = col[:10]
            candidate = base
            suffix = 0
            while candidate in seen or candidate in final_map.values():
                suffix += 1
                candidate = base[:9] + str(suffix)
            final_map[col] = candidate
            seen.add(candidate)
    if final_map:
        import warnings
        warnings.warn(
            f"[Shapefile] Truncated column names: {final_map}", stacklevel=2
        )
        df = df.rename(columns=final_map)
    return df


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_csv(result_gdf: gpd.GeoDataFrame, path: str) -> None:
    """Export all metrics as a flat CSV (geometry excluded)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(result_gdf.drop(columns="geometry", errors="ignore"))
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"[Export] CSV saved: {path}  ({len(df)} rows × {len(df.columns)} cols)")


def export_shapefile(result_gdf: gpd.GeoDataFrame, path: str) -> None:
    """Export transects with metrics as a Shapefile."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    shp_gdf = result_gdf.copy()
    shp_gdf = _rename_for_shp(shp_gdf)
    shp_gdf.to_file(path, driver="ESRI Shapefile")
    print(f"[Export] Shapefile saved: {path}")


def export_geopackage(result_gdf: gpd.GeoDataFrame, path: str,
                      layer: str = "shoreline_change") -> None:
    """Export transects with metrics as a GeoPackage (recommended for QGIS)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    result_gdf.to_file(path, driver="GPKG", layer=layer)
    print(f"[Export] GeoPackage saved: {path}  (layer='{layer}')")


def export_intersection_points(
    intersection_gdf: gpd.GeoDataFrame,
    output_dir: str,
    prefix: str = "pyshore",
) -> None:
    """Export raw intersection points as CSV and Shapefile for QC / visualisation."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{prefix}_intersections.csv")
    shp_path = os.path.join(output_dir, f"{prefix}_intersections.shp")

    df = pd.DataFrame(intersection_gdf).copy()
    df["x"] = intersection_gdf.geometry.x
    df["y"] = intersection_gdf.geometry.y
    df.drop(columns=["geometry"], errors="ignore").to_csv(csv_path, index=False, float_format="%.4f")
    intersection_gdf.to_file(shp_path)
    print(f"[Export] Intersection points → {csv_path}  |  {shp_path}")


# ---------------------------------------------------------------------------
# Master export function
# ---------------------------------------------------------------------------

def export_results(
    result_gdf: gpd.GeoDataFrame,
    intersection_gdf: Optional[gpd.GeoDataFrame],
    output_dir: str,
    prefix: str = "pyshore",
    csv: bool = True,
    shapefile: bool = True,
    geopackage: bool = True,
    export_intersections: bool = True,
) -> None:
    """
    Export all outputs in one call.

    Parameters
    ----------
    result_gdf : GeoDataFrame
        Transects GeoDataFrame with all computed metrics as attributes.
    intersection_gdf : GeoDataFrame or None
        Raw intersection points (optional, for QC export).
    output_dir : str
        Root output directory.
    prefix : str
        Filename prefix.
    csv, shapefile, geopackage : bool
        Which formats to write.
    export_intersections : bool
        Whether to also export the raw intersection points.
    """
    os.makedirs(output_dir, exist_ok=True)

    if csv:
        export_csv(result_gdf, os.path.join(output_dir, f"{prefix}_metrics.csv"))

    if shapefile:
        export_shapefile(result_gdf, os.path.join(output_dir, f"{prefix}_transects.shp"))

    if geopackage:
        export_geopackage(result_gdf, os.path.join(output_dir, f"{prefix}_transects.gpkg"))

    if export_intersections and intersection_gdf is not None:
        export_intersection_points(intersection_gdf, output_dir, prefix)

    print(f"\n[Export] All outputs written to: {output_dir}")
