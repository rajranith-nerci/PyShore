"""
PyShore — Automated Shoreline Change Analysis
==============================================

A Python package for:
  • Extracting shorelines from Sentinel-2 / Landsat via Google Earth Engine
  • Computing DSAS-equivalent change metrics (NSM, EPR, SCE, LRR, WLR, LMS)
  • Full uncertainty analysis (OLS CI, bootstrap, positional error propagation)
  • Exporting results as CSV, Shapefile, and GeoPackage

Quick start
-----------
>>> from pyshore import PyShoreConfig, ExtractionConfig, AnalysisConfig, OutputConfig
>>> from pyshore.pipeline import PyShore
>>>
>>> cfg = PyShoreConfig(
...     extraction=ExtractionConfig(aoi_shapefile="aoi.shp", start_year=2017, end_year=2023,
...                                  output_dir="data/shorelines"),
...     analysis=AnalysisConfig(baseline_shapefile="baseline.shp",
...                              shoreline_dir="data/shorelines"),
...     output=OutputConfig(output_dir="output/")
... )
>>> ps = PyShore(cfg)
>>> ps.run()
"""

__version__ = "1.0.0"
__author__ = "PyShore"

from pyshore.config import (
    PyShoreConfig,
    ExtractionConfig,
    AnalysisConfig,
    OutputConfig,
    SENSOR_UNCERTAINTY,
    SENSOR_YEAR_RANGES,
)

__all__ = [
    "PyShoreConfig",
    "ExtractionConfig",
    "AnalysisConfig",
    "OutputConfig",
    "SENSOR_UNCERTAINTY",
    "SENSOR_YEAR_RANGES",
]
