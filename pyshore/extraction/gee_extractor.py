"""
pyshore.extraction.gee_extractor
==================================
Google Earth Engine based shoreline extraction.

Supports:
  • Sentinel-2 L2A  (2017+, 10 m) — preferred when available
  • Landsat 8 / 9   (2013+, 30 m)
  • Landsat 5 / 7   (1984–2012, 30 m)

For each year the module:
  1. Picks the best available sensor
  2. Builds a cloud-free seasonal composite
  3. Computes MNDWI or NDWI
  4. Estimates threshold (Otsu / percentile / fixed)
  5. Vectorises the water mask and extracts the shoreline boundary
  6. Saves a dated shapefile + records the sensor name used
"""

from __future__ import annotations

import json
import os
from typing import Optional, List, Dict

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

try:
    import ee
    import geemap
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False


def _ee_to_gdf(feature_collection: "ee.FeatureCollection") -> "gpd.GeoDataFrame":
    """
    Convert an EE FeatureCollection to a GeoDataFrame.
    Handles geemap API changes across versions:
      - geemap >= 0.20  : geemap.ee_to_gdf()
      - geemap < 0.20   : geemap.ee_to_geopandas()
      - fallback        : getInfo() → GeoJSON → geopandas
    """
    import geopandas as gpd

    # Try newer API first
    if hasattr(geemap, "ee_to_gdf"):
        return geemap.ee_to_gdf(feature_collection)

    # Try older API
    if hasattr(geemap, "ee_to_geopandas"):
        return geemap.ee_to_geopandas(feature_collection)

    # Pure GEE fallback — download as GeoJSON via getInfo()
    geojson = feature_collection.getInfo()
    features = geojson.get("features", [])
    if not features:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame.from_features(features)

from pyshore.config import (
    ExtractionConfig,
    SENSOR_UNCERTAINTY,
    SENSOR_YEAR_RANGES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_gee() -> None:
    if not GEE_AVAILABLE:
        raise ImportError(
            "earthengine-api and geemap are required for GEE extraction.\n"
            "Install them with:  pip install earthengine-api geemap"
        )


def initialize_gee(project: Optional[str] = None) -> None:
    """
    Authenticate and initialise Google Earth Engine.

    On Streamlit Cloud: reads service account credentials from st.secrets[gee].
    Locally: falls back to cached OAuth credentials (ee.Authenticate()).
    """
    _assert_gee()

    # ── Streamlit Cloud: service account from secrets ────────────────────────
    try:
        import streamlit as st
        gee_secrets = st.secrets.get("gee", {})
        sa_email = gee_secrets.get("service_account_email", "")
        sa_key   = gee_secrets.get("service_account_key", "")
        if sa_email and sa_key:
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tf:
                tf.write(sa_key if isinstance(sa_key, str) else json.dumps(sa_key))
                key_path = tf.name
            credentials = ee.ServiceAccountCredentials(sa_email, key_path)
            _proj = project or gee_secrets.get("project")
            ee.Initialize(credentials=credentials, project=_proj)
            print("[GEE] Initialised via service account (Streamlit secrets).")
            return
    except Exception:
        pass  # Not running under Streamlit, or secrets not set — fall through

    # ── Local: OAuth / cached credentials ───────────────────────────────────
    _NOT_REGISTERED_MSG = (
        "\n\n[GEE] *** Earth Engine project not registered ***\n"
        "Follow these steps:\n"
        "  1. Sign up at https://signup.earthengine.google.com/\n"
        "  2. Create a Google Cloud project at https://console.cloud.google.com/\n"
        "  3. Enable the Earth Engine API for that project:\n"
        "       https://console.cloud.google.com/apis/library/earthengine.googleapis.com\n"
        "  4. Set GEE_PROJECT = 'your-cloud-project-id' in example_run.py\n"
        "  5. Re-run the script.\n"
    )
    try:
        ee.Initialize(project=project)
        print("[GEE] Initialised successfully.")
    except Exception as e:
        err = str(e)
        if "Not signed up" in err or "not registered" in err or "project" in err.lower():
            raise RuntimeError(_NOT_REGISTERED_MSG) from e
        print("[GEE] Authenticating …")
        try:
            ee.Authenticate()
            ee.Initialize(project=project)
            print("[GEE] Initialised after authentication.")
        except Exception as e2:
            if "Not signed up" in str(e2) or "not registered" in str(e2):
                raise RuntimeError(_NOT_REGISTERED_MSG) from e2
            raise


def _select_sensor(year: int, prefer_sentinel2: bool = True) -> str:
    """Return the preferred sensor name for a given year."""
    if prefer_sentinel2 and year >= SENSOR_YEAR_RANGES["sentinel2"][0]:
        return "sentinel2"
    if year >= SENSOR_YEAR_RANGES["landsat89"][0]:
        return "landsat89"
    return "landsat57"


# ---------------------------------------------------------------------------
# Cloud masking
# ---------------------------------------------------------------------------

def _mask_s2_clouds(image: "ee.Image") -> "ee.Image":
    """Bit-field cloud mask for Sentinel-2 SCL layer."""
    scl = image.select("SCL")
    # SCL classes to keep: 4 (vegetation), 5 (bare), 6 (water), 11 (snow)
    mask = (
        scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(11))
    )
    return image.updateMask(mask)


def _mask_landsat_clouds(image: "ee.Image") -> "ee.Image":
    """QA_PIXEL bit-field cloud mask for Landsat Collection 2."""
    qa = image.select("QA_PIXEL")
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(shadow_bit).eq(0))
    return image.updateMask(mask)


# ---------------------------------------------------------------------------
# Water index
# ---------------------------------------------------------------------------

def _water_index(image: "ee.Image", sensor: str, method: str) -> "ee.Image":
    """Return single-band water index image."""
    method = method.upper()
    if sensor == "sentinel2":
        green, nir, swir1 = "B3", "B8", "B11"
    else:
        # Landsat 8/9 OLI band names (Collection 2 SR)
        green, nir, swir1 = "SR_B3", "SR_B5", "SR_B6"
        # Landsat 5/7 TM/ETM+ band names differ
        if sensor == "landsat57":
            green, nir, swir1 = "SR_B2", "SR_B4", "SR_B5"

    if method == "NDWI":
        return image.normalizedDifference([green, nir]).rename("WI")
    else:  # MNDWI
        return image.normalizedDifference([green, swir1]).rename("WI")


# ---------------------------------------------------------------------------
# Threshold estimation
# ---------------------------------------------------------------------------

def _otsu_threshold_client(wi_image: "ee.Image", region: "ee.Geometry",
                            scale: int = 30) -> float:
    """
    Compute Otsu threshold client-side from a GEE histogram.
    Handles both GEE histogram formats:
      - new: {'histogram': [...], 'bucketMeans': [...]}
      - old: {'histogram': [...], 'min': float, 'max': float, 'bucketWidth': float}
    Falls back to percentile-based threshold if histogram is unavailable.
    """
    histogram = wi_image.reduceRegion(
        reducer=ee.Reducer.histogram(maxBuckets=256),
        geometry=region,
        scale=scale,
        maxPixels=1e10,
        bestEffort=True,
    ).getInfo()

    hist_data = histogram.get("WI")
    if hist_data is None or "histogram" not in hist_data:
        # Fall back to median threshold
        print("  [Otsu] No histogram data — falling back to percentile threshold.")
        return _percentile_threshold_client(wi_image, region, scale)

    counts = np.array(hist_data["histogram"], dtype=float)

    # Resolve bin centres — handle both GEE histogram formats
    if "bucketMeans" in hist_data:
        bin_centers = np.array(hist_data["bucketMeans"], dtype=float)
    elif "min" in hist_data and "bucketWidth" in hist_data:
        bw = float(hist_data["bucketWidth"])
        start = float(hist_data["min"]) + bw / 2.0
        bin_centers = np.array([start + i * bw for i in range(len(counts))], dtype=float)
    elif "min" in hist_data and "max" in hist_data:
        bin_edges = np.linspace(float(hist_data["min"]), float(hist_data["max"]), len(counts) + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    else:
        print(f"  [Otsu] Unrecognised histogram format (keys: {list(hist_data.keys())}) "
              "— falling back to percentile.")
        return _percentile_threshold_client(wi_image, region, scale)

    # Otsu's method
    total = counts.sum()
    if total == 0:
        return 0.0

    cum_sum = np.cumsum(counts)
    cum_mean = np.cumsum(counts * bin_centers)
    global_mean = cum_mean[-1] / total

    between_class_variance = np.zeros(len(counts))
    for t in range(1, len(counts)):
        w0 = cum_sum[t] / total
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mean0 = cum_mean[t] / cum_sum[t]
        mean1 = (global_mean * total - cum_mean[t]) / (total - cum_sum[t])
        between_class_variance[t] = w0 * w1 * (mean0 - mean1) ** 2

    return float(bin_centers[np.argmax(between_class_variance)])


def _percentile_threshold_client(wi_image: "ee.Image", region: "ee.Geometry",
                                  scale: int = 30) -> float:
    """50th-percentile threshold computed client-side (fallback for Otsu)."""
    stats = wi_image.reduceRegion(
        reducer=ee.Reducer.percentile([50]),
        geometry=region,
        scale=scale,
        maxPixels=1e10,
        bestEffort=True,
    ).getInfo()
    val = stats.get("WI")
    return float(val) if val is not None else 0.0


def _percentile_threshold(wi_image: "ee.Image", region: "ee.Geometry",
                           scale: int = 30) -> "ee.Number":
    """50th-percentile adaptive threshold (fast, runs server-side)."""
    stats = wi_image.reduceRegion(
        reducer=ee.Reducer.percentile([50]),
        geometry=region,
        scale=scale,
        maxPixels=1e10,
        bestEffort=True,
    )
    return ee.Number(stats.get("WI"))


# ---------------------------------------------------------------------------
# Collection builders
# ---------------------------------------------------------------------------

def _build_sentinel2_collection(
    aoi: "ee.Geometry", year: int, cloud_thresh: int,
    months: Optional[List[int]]
) -> "ee.ImageCollection":
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        .map(_mask_s2_clouds)
    )
    if months:
        col = col.filter(ee.Filter.calendarRange(min(months), max(months), "month"))
    return col


def _build_landsat89_collection(
    aoi: "ee.Geometry", year: int, cloud_thresh: int,
    months: Optional[List[int]]
) -> "ee.ImageCollection":
    col = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_thresh))
        .map(_mask_landsat_clouds)
        .map(lambda img: img.multiply(ee.Image([0.0000275, 0.0000275, 0.0000275,
                                                 0.0000275, 0.0000275, 0.0000275])
                                       .add(-0.2))
             )
    )
    if months:
        col = col.filter(ee.Filter.calendarRange(min(months), max(months), "month"))
    return col


def _build_landsat57_collection(
    aoi: "ee.Geometry", year: int, cloud_thresh: int,
    months: Optional[List[int]]
) -> "ee.ImageCollection":
    l5 = (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_thresh))
        .map(_mask_landsat_clouds)
    )
    l7 = (
        ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_thresh))
        .map(_mask_landsat_clouds)
    )
    col = l5.merge(l7)
    if months:
        col = col.filter(ee.Filter.calendarRange(min(months), max(months), "month"))
    return col


def _build_collection(
    sensor: str, aoi: "ee.Geometry", year: int,
    cloud_thresh: int, months: Optional[List[int]]
) -> "ee.ImageCollection":
    dispatch = {
        "sentinel2": _build_sentinel2_collection,
        "landsat89": _build_landsat89_collection,
        "landsat57": _build_landsat57_collection,
    }
    return dispatch[sensor](aoi, year, cloud_thresh, months)


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class GEEExtractor:
    """
    Extract annual shorelines from Google Earth Engine imagery.

    Parameters
    ----------
    config : ExtractionConfig
        Extraction settings (sensor, AOI, year range, etc.)

    Usage
    -----
    >>> from pyshore.extraction import GEEExtractor
    >>> extractor = GEEExtractor(config)
    >>> sensor_map = extractor.run()   # returns {year: sensor_name}
    """

    def __init__(self, config: ExtractionConfig, gee_project: Optional[str] = None):
        _assert_gee()
        self.cfg = config
        self.gee_project = gee_project
        self._sensor_map: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[int, str]:
        """
        Extract shorelines for all years.

        Returns
        -------
        dict
            {year: sensor_name} mapping — pass to AnalysisConfig.sensor_per_year
        """
        initialize_gee(self.gee_project)
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        # Load AOI — if aoi_shapefile is not set or missing, derive from baseline
        aoi_source = self.cfg.aoi_shapefile
        if (not aoi_source or not os.path.exists(aoi_source)) and self.cfg.baseline_shapefile:
            print(f"  [AOI] aoi_shapefile not found — deriving AOI from baseline "
                  f"(buffer={self.cfg.aoi_buffer_m} m)")
            aoi_source = None

        if aoi_source and os.path.exists(aoi_source):
            aoi_gdf = gpd.read_file(aoi_source)
            # Validate spatial overlap with baseline if available
            if self.cfg.baseline_shapefile and os.path.exists(self.cfg.baseline_shapefile):
                bl = gpd.read_file(self.cfg.baseline_shapefile)
                bl_bounds = bl.to_crs(aoi_gdf.crs).total_bounds
                aoi_bounds = aoi_gdf.total_bounds
                x_overlap = (aoi_bounds[0] < bl_bounds[2]) and (aoi_bounds[2] > bl_bounds[0])
                y_overlap = (aoi_bounds[1] < bl_bounds[3]) and (aoi_bounds[3] > bl_bounds[1])
                if not (x_overlap and y_overlap):
                    print(f"  [WARNING] AOI ({os.path.basename(aoi_source)}) and baseline "
                          f"do NOT overlap spatially!")
                    print(f"    AOI bounds   : {aoi_bounds.round(0)}")
                    print(f"    Baseline bounds: {bl_bounds.round(0)}")
                    print(f"  [AOI] Switching to baseline-derived AOI automatically.")
                    aoi_source = None

        if aoi_source is None or not os.path.exists(str(aoi_source)):
            # Derive AOI from baseline + buffer
            bl = gpd.read_file(self.cfg.baseline_shapefile)
            bl_proj = bl.to_crs(epsg=self.cfg.target_epsg)
            aoi_gdf = gpd.GeoDataFrame(
                geometry=[bl_proj.geometry.buffer(self.cfg.aoi_buffer_m).unary_union],
                crs=f"EPSG:{self.cfg.target_epsg}"
            )
            print(f"  [AOI] Derived from baseline with {self.cfg.aoi_buffer_m} m buffer. "
                  f"Bounds: {aoi_gdf.total_bounds.round(0)}")

        aoi_gdf = aoi_gdf.to_crs(epsg=4326)

        # Convert to EE geometry — handle geemap API changes
        _gdf_to_ee = (getattr(geemap, "gdf_to_ee", None)
                      or getattr(geemap, "geopandas_to_ee", None))
        if _gdf_to_ee is None:
            import json as _json
            geojson = _json.loads(aoi_gdf.to_json())
            aoi_ee = ee.FeatureCollection(geojson["features"]).geometry()
        else:
            aoi_ee = _gdf_to_ee(aoi_gdf)

        for year in range(self.cfg.start_year, self.cfg.end_year + 1):
            sensor = _select_sensor(year, self.cfg.prefer_sentinel2)
            print(f"[{year}] Sensor: {sensor}")

            try:
                ok = self._extract_year(aoi_ee, year, sensor)
                if ok:
                    self._sensor_map[year] = sensor
            except Exception as exc:
                print(f"  [ERROR] {year}: {exc}")

        # Persist sensor map alongside shapefiles
        sensor_map_path = os.path.join(self.cfg.output_dir, "sensor_map.json")
        with open(sensor_map_path, "w") as f:
            json.dump(self._sensor_map, f, indent=2)

        print(f"\n[GEE] Extraction complete. Sensor map saved to {sensor_map_path}")
        return {int(k): v for k, v in self._sensor_map.items()}

    @property
    def sensor_map(self) -> Dict[int, str]:
        return self._sensor_map

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_year(self, aoi_ee: "ee.Geometry", year: int, sensor: str) -> bool:
        """Extract and export shoreline for a single year. Returns True on success."""
        cloud_thresh = (
            self.cfg.cloud_threshold if sensor == "sentinel2"
            else self.cfg.landsat_cloud_threshold
        )
        collection = _build_collection(sensor, aoi_ee, year, cloud_thresh, self.cfg.months)

        n_images = collection.size().getInfo()
        if n_images == 0:
            print(f"  No valid images for {year} ({sensor}), skipping.")
            return False
        print(f"  {n_images} valid images found.")

        # Annual composite (median reduces transient wave / tide noise)
        composite = collection.median()

        # Water index
        scale = self.cfg.scale if sensor == "sentinel2" else 30
        wi = _water_index(composite, sensor, self.cfg.water_index)

        # Threshold
        if self.cfg.threshold_method == "otsu":
            threshold = _otsu_threshold_client(wi, aoi_ee, scale=scale)
            print(f"  Otsu threshold: {threshold:.4f}")
            water_mask = wi.gt(threshold)
        elif self.cfg.threshold_method == "percentile":
            threshold = _percentile_threshold(wi, aoi_ee, scale=scale)
            water_mask = wi.gt(threshold)
        else:
            water_mask = wi.gt(self.cfg.fixed_threshold)

        # Vectorise water pixels
        water_vec = (
            water_mask.selfMask()
            .reduceToVectors(
                geometry=aoi_ee,
                scale=scale,
                geometryType="polygon",
                eightConnected=True,
                maxPixels=1e10,
                bestEffort=True,
            )
        )

        gdf = _ee_to_gdf(water_vec)
        if gdf is None or gdf.empty:
            print(f"  Water mask empty for {year}, skipping.")
            return False

        # Reproject and extract shoreline boundary
        gdf = gdf.set_crs(epsg=4326, allow_override=True).to_crs(epsg=self.cfg.target_epsg)
        water_union = unary_union(gdf.geometry)
        shoreline = water_union.boundary

        # Filter out tiny artefact polygons (keep only lines / multilines)
        if hasattr(shoreline, "geoms"):
            from shapely.geometry import MultiLineString
            parts = [g for g in shoreline.geoms if g.geom_type in ("LineString", "MultiLineString")]
            if not parts:
                print(f"  No valid shoreline geometry for {year}.")
                return False
            from shapely.ops import linemerge
            shoreline = linemerge(parts)

        shoreline_gdf = gpd.GeoDataFrame(
            {"year": [year], "sensor": [sensor],
             "uncertainty_m": [SENSOR_UNCERTAINTY.get(sensor, 15.0)]},
            geometry=[shoreline],
            crs=f"EPSG:{self.cfg.target_epsg}",
        )

        out_path = os.path.join(self.cfg.output_dir, f"{year}.shp")
        shoreline_gdf.to_file(out_path)
        print(f"  Saved: {out_path}")
        return True
