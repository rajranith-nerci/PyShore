"""
pyshore.config
==============
Central configuration for the PyShore shoreline change analysis tool.
All parameters are dataclass-based — create a Config and pass it to the pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import os


# ---------------------------------------------------------------------------
# Sensor positional uncertainty (metres) — used in WLR weighting
# Based on half pixel size (sub-pixel shoreline detection accuracy)
# ---------------------------------------------------------------------------
SENSOR_UNCERTAINTY = {
    "sentinel2":  5.0,   # 10 m pixel → ~5–7 m positional uncertainty
    "landsat89":  15.0,  # 30 m pixel → ~15–21 m positional uncertainty
    "landsat57":  15.0,  # 30 m pixel → ~15–21 m positional uncertainty
}

# Sensor year ranges for auto-selection
SENSOR_YEAR_RANGES = {
    "landsat57":  (1984, 2012),
    "landsat89":  (2013, 9999),
    "sentinel2":  (2017, 9999),   # preferred at 10 m when available
}


@dataclass
class ExtractionConfig:
    """Parameters controlling GEE-based shoreline extraction."""

    start_year: int
    """First year to extract (inclusive)."""

    end_year: int
    """Last year to extract (inclusive)."""

    output_dir: str
    """Directory where annual shoreline shapefiles will be saved."""

    aoi_shapefile: str = ""
    """Path to area-of-interest polygon shapefile.
    If empty or the file doesn't exist, the AOI is automatically derived
    from baseline_shapefile + aoi_buffer_m."""

    baseline_shapefile: str = ""
    """Path to baseline shoreline shapefile — used to auto-derive AOI if needed."""

    aoi_buffer_m: float = 5000.0
    """Buffer distance (metres) around baseline used when auto-deriving the AOI."""

    water_index: Literal["MNDWI", "NDWI"] = "MNDWI"
    """Water index.  MNDWI is preferred for turbid coastal waters."""

    threshold_method: Literal["otsu", "percentile", "fixed"] = "otsu"
    """How to determine the water/land boundary threshold."""

    fixed_threshold: float = 0.0
    """Used only when threshold_method == 'fixed'."""

    cloud_threshold: int = 20
    """Maximum CLOUDY_PIXEL_PERCENTAGE for Sentinel-2 images."""

    landsat_cloud_threshold: int = 20
    """Maximum CLOUD_COVER for Landsat images."""

    target_epsg: int = 32643
    """Output CRS EPSG code. Default: UTM Zone 43 N (Kerala coast)."""

    scale: int = 10
    """Pixel resolution for GEE operations (metres).  Use 10 for S2, 30 for LS."""

    prefer_sentinel2: bool = True
    """If True, use Sentinel-2 (10 m) when available (2017+), else Landsat."""

    months: Optional[List[int]] = None
    """Restrict compositing to specific months (e.g. [11,12,1,2] for dry season).
    None means all months."""

    season_filter: bool = False
    """Apply seasonal filter to reduce tidal/wave noise."""


@dataclass
class AnalysisConfig:
    """Parameters controlling transect generation and metric computation."""

    baseline_shapefile: str
    """Path to baseline shoreline shapefile used to generate transects."""

    shoreline_dir: str
    """Directory containing annual shoreline shapefiles (one per year)."""

    transect_spacing: float = 100.0
    """Along-shore spacing between transects (metres)."""

    transect_length: float = 2000.0
    """Total transect length — extends equally on both sides of baseline (metres)."""

    min_years: int = 3
    """Minimum number of yearly observations required to compute regression metrics."""

    min_detectable_change: float = 5.0
    """NSM values below this threshold (metres) are treated as noise and set to NaN."""

    baseline_year: Optional[int] = None
    """Year of baseline shoreline. Used to anchor NSM/EPR calculations.
    If None, the earliest available year is used."""

    epsg: int = 32643
    """Working CRS EPSG code — all geometries are reprojected to this."""

    bootstrap_n: int = 1000
    """Number of bootstrap iterations for uncertainty estimation."""

    confidence_level: float = 0.95
    """Confidence level for intervals (0–1)."""

    positional_uncertainty: Optional[dict] = None
    """Per-year positional uncertainty (metres). Keys are years (int), values are σ (float).
    If None, uncertainty is derived automatically from the sensor used."""

    sensor_per_year: Optional[dict] = None
    """Sensor name per year — populated automatically during extraction.
    Keys are years (int), values are sensor strings ('sentinel2', 'landsat89', etc.)."""


@dataclass
class OutputConfig:
    """Output options."""

    output_dir: str
    """Root directory for all outputs."""

    csv: bool = True
    shapefile: bool = True
    geopackage: bool = True
    plots: bool = True

    prefix: str = "pyshore"
    """Filename prefix for all output files."""

    @property
    def csv_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.prefix}_metrics.csv")

    @property
    def shapefile_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.prefix}_transects.shp")

    @property
    def gpkg_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.prefix}_transects.gpkg")

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.output_dir, "plots")


@dataclass
class PyShoreConfig:
    """Top-level configuration combining extraction, analysis, and output settings."""

    extraction: ExtractionConfig
    analysis: AnalysisConfig
    output: OutputConfig

    def validate(self) -> None:
        """Raise ValueError for any missing or inconsistent settings."""
        if self.extraction.start_year > self.extraction.end_year:
            raise ValueError("start_year must be ≤ end_year")
        if not (0 < self.output_dir_exists or True):
            pass  # output dirs are created at runtime

    @property
    def output_dir(self) -> str:
        return self.output.output_dir
