"""
example_run.py
==============
Example script showing two usage modes:

  Mode A — Full pipeline (GEE extraction + analysis)
  Mode B — Analysis only (if shoreline shapefiles already exist)

Edit the paths and settings in the CONFIG section below, then run:
  python example_run.py
"""

# Clear stale .pyc bytecode cache so edits to pyshore/ take effect immediately
import pathlib, shutil
for _p in pathlib.Path(__file__).parent.rglob("__pycache__"):
    shutil.rmtree(_p, ignore_errors=True)

# =============================================================================
# CONFIGURATION — edit these values
# =============================================================================

# Path to your Area of Interest polygon shapefile
AOI_SHAPEFILE    = "D:/Paper works Working 2025/Shorelina_AOI/aoi.shp"

# Path to your baseline shoreline shapefile
BASELINE_SHP     = "D:/Paper works Working 2025/2015/shoreline_2015.shp"

# Directory where annual shoreline shapefiles are (or will be) saved
SHORELINE_DIR    = "D:/Paper works Working 2025/shoreline_output"

# Output directory for metrics, shapefiles, and plots
OUTPUT_DIR       = "output"

# Year range to process
START_YEAR       = 2017
END_YEAR         = 2023

# Transect parameters
TRANSECT_SPACING = 100    # metres along coast
TRANSECT_LENGTH  = 2000   # metres total perpendicular length

# Water index: "MNDWI" (recommended for turbid coasts) or "NDWI"
WATER_INDEX      = "MNDWI"

# Threshold method: "otsu" (recommended), "percentile", or "fixed"
THRESHOLD_METHOD = "otsu"

# GEE project ID — REQUIRED when run_extraction=True.
# This must match the Google Cloud project that has Earth Engine API enabled.
# Steps if you haven't done this yet:
#   1. Sign up: https://signup.earthengine.google.com/
#   2. Create a project: https://console.cloud.google.com/
#   3. Enable EE API: https://console.cloud.google.com/apis/library/earthengine.googleapis.com
#   4. Paste your project ID below (looks like "my-project-123456")
GEE_PROJECT      = "rr-geefiles"

# Set to True to run GEE extraction, False if shapefiles already exist
RUN_EXTRACTION   = True

# Bootstrap iterations (1000 recommended, reduce to 200 for faster testing)
BOOTSTRAP_N      = 1000

# =============================================================================
# PIPELINE
# =============================================================================

from pyshore import (
    PyShoreConfig,
    ExtractionConfig,
    AnalysisConfig,
    OutputConfig,
)
from pyshore.pipeline import PyShore


def main():
    cfg = PyShoreConfig(
        extraction=ExtractionConfig(
            aoi_shapefile=AOI_SHAPEFILE,       # leave as "" to auto-derive from baseline
            baseline_shapefile=BASELINE_SHP,   # used to auto-derive AOI if needed
            aoi_buffer_m=5000,                 # 5 km buffer around baseline for AOI
            start_year=START_YEAR,
            end_year=END_YEAR,
            output_dir=SHORELINE_DIR,
            water_index=WATER_INDEX,
            threshold_method=THRESHOLD_METHOD,
            prefer_sentinel2=True,
            cloud_threshold=20,
            target_epsg=32643,                 # UTM 43N — change for your region
        ),
        analysis=AnalysisConfig(
            baseline_shapefile=BASELINE_SHP,
            shoreline_dir=SHORELINE_DIR,
            transect_spacing=TRANSECT_SPACING,
            transect_length=TRANSECT_LENGTH,
            min_years=3,
            min_detectable_change=5.0,
            epsg=32643,
            bootstrap_n=BOOTSTRAP_N,
            confidence_level=0.95,
        ),
        output=OutputConfig(
            output_dir=OUTPUT_DIR,
            csv=True,
            shapefile=True,
            geopackage=True,
            plots=True,
            prefix="pyshore",
        ),
    )

    ps = PyShore(cfg)
    result = ps.run(
        run_extraction=RUN_EXTRACTION,
        gee_project=GEE_PROJECT,
        skip_plots=False,
    )

    # Quick summary
    print("\n=== Summary ===")
    n_total = len(result)
    print(f"  Transects: {n_total}")

    # NSM / EPR — work with ≥ 2 years
    for col, unit in [("NSM", "m"), ("EPR", "m/yr"), ("SCE", "m")]:
        if col in result.columns:
            valid = result[col].dropna()
            if not valid.empty:
                print(f"  {col}: mean={valid.mean():.2f} {unit}  "
                      f"min={valid.min():.2f}  max={valid.max():.2f}  "
                      f"(n={len(valid)} transects)")

    # Regression rates — need ≥ min_years observations
    print("\nMean regression rates (m/yr):")
    for col in ["LRR", "WLR", "LMS"]:
        if col in result.columns:
            valid = result[col].dropna()
            tag = f"n={len(valid)}" if not valid.empty else "no data — need more years"
            val = f"{valid.mean():.3f}" if not valid.empty else "nan"
            print(f"  {col}: {val}  ({tag})")

    # Trend classification
    for trend_col in ["Trend_LRR", "Trend_WLR", "Trend_LMS"]:
        if trend_col in result.columns:
            counts = result[trend_col].value_counts()
            if "No Data" not in counts or len(counts) > 1:
                print(f"\n{trend_col}:")
                print(counts.to_string())

    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print(f"  CSV        : {OUTPUT_DIR}/pyshore_metrics.csv")
    print(f"  Shapefile  : {OUTPUT_DIR}/pyshore_transects.shp")
    print(f"  GeoPackage : {OUTPUT_DIR}/pyshore_transects.gpkg")
    print(f"  Plots      : {OUTPUT_DIR}/plots/")


if __name__ == "__main__":
    main()
