"""
pyshore.pipeline
=================
Main orchestrator for the PyShore shoreline change analysis workflow.

Workflow stages
---------------
1.  [Optional] GEE extraction  → annual shoreline shapefiles
2.  Load shorelines from directory
3.  Generate transects from baseline
4.  Intersect transects × shorelines
5.  Compute metrics (NSM, EPR, SCE, LRR, WLR, LMS)
6.  Compute uncertainty (bootstrap, positional propagation)
7.  Merge into result GeoDataFrame
8.  Export (CSV, Shapefile, GeoPackage)
9.  Generate plots
"""

from __future__ import annotations

import os
import json
from typing import Optional, List, Dict

import geopandas as gpd
import pandas as pd

from pyshore.config import PyShoreConfig, SENSOR_UNCERTAINTY
from pyshore.analysis.transects import generate_transects, load_transects
from pyshore.analysis.intersection import intersect_transects, load_shorelines_from_dir
from pyshore.analysis.metrics import compute_all_metrics
from pyshore.analysis.uncertainty import compute_uncertainty
from pyshore.export.exporter import export_results
from pyshore.visualization.plots import generate_all_plots


class PyShore:
    """
    End-to-end shoreline change analysis pipeline.

    Parameters
    ----------
    config : PyShoreConfig
        Full configuration object.

    Example
    -------
    >>> from pyshore import PyShoreConfig, ExtractionConfig, AnalysisConfig, OutputConfig
    >>> from pyshore.pipeline import PyShore
    >>>
    >>> cfg = PyShoreConfig(
    ...     extraction=ExtractionConfig(
    ...         aoi_shapefile="data/aoi.shp",
    ...         start_year=2017,
    ...         end_year=2023,
    ...         output_dir="data/shorelines",
    ...     ),
    ...     analysis=AnalysisConfig(
    ...         baseline_shapefile="data/baseline.shp",
    ...         shoreline_dir="data/shorelines",
    ...         transect_spacing=100,
    ...         transect_length=2000,
    ...     ),
    ...     output=OutputConfig(output_dir="output/"),
    ... )
    >>> ps = PyShore(cfg)
    >>> ps.run()
    """

    def __init__(self, config: PyShoreConfig):
        self.cfg = config
        self.transects: Optional[gpd.GeoDataFrame] = None
        self.shorelines: Optional[List[gpd.GeoDataFrame]] = None
        self.intersection_gdf: Optional[gpd.GeoDataFrame] = None
        self.metrics_df: Optional[pd.DataFrame] = None
        self.result_gdf: Optional[gpd.GeoDataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        run_extraction: bool = False,
        gee_project: Optional[str] = None,
        skip_plots: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Execute the full pipeline.

        Parameters
        ----------
        run_extraction : bool
            If True, run GEE extraction before analysis.
            Set to False if shoreline shapefiles already exist.
        gee_project : str, optional
            GEE project ID (required if run_extraction=True).
        skip_plots : bool
            Skip plot generation (useful for headless environments).

        Returns
        -------
        GeoDataFrame
            Transects with all metrics and uncertainty columns.
        """
        print("=" * 60)
        print("  PyShore — Shoreline Change Analysis")
        print("=" * 60)

        # Stage 1: GEE Extraction (optional)
        if run_extraction:
            self._run_extraction(gee_project)
        else:
            # Pre-flight: warn early if shoreline dir is missing or empty
            sl_dir = self.cfg.analysis.shoreline_dir
            shp_files = []
            if os.path.isdir(sl_dir):
                shp_files = [f for f in os.listdir(sl_dir) if f.endswith(".shp")]
            if not shp_files:
                raise FileNotFoundError(
                    f"\nNo shoreline shapefiles found in:\n  {sl_dir}\n\n"
                    "You have two options:\n"
                    "  1. Set RUN_EXTRACTION = True in example_run.py to extract "
                    "shorelines from Google Earth Engine first.\n"
                    "  2. Point SHORELINE_DIR to an existing folder containing "
                    "annual shoreline shapefiles named <year>.shp (e.g. 2017.shp)."
                )

        # Stage 2–4: Load, transect, intersect
        self._load_shorelines()
        self._build_transects()
        self._compute_intersections()

        # Stage 5–6: Metrics and uncertainty
        self._compute_metrics()
        self._compute_uncertainty()

        # Stage 7: Assemble result GeoDataFrame
        self._assemble_result()

        # Stage 8: Export
        os.makedirs(self.cfg.output.output_dir, exist_ok=True)
        export_results(
            self.result_gdf,
            self.intersection_gdf,
            output_dir=self.cfg.output.output_dir,
            prefix=self.cfg.output.prefix,
            csv=self.cfg.output.csv,
            shapefile=self.cfg.output.shapefile,
            geopackage=self.cfg.output.geopackage,
        )

        # Stage 9: Plots
        if not skip_plots and self.cfg.output.plots:
            generate_all_plots(
                self.result_gdf,
                self.intersection_gdf,
                output_dir=self.cfg.output.plots_dir,
            )

        print("\n" + "=" * 60)
        print("  Analysis complete.")
        print(f"  Results: {self.cfg.output.output_dir}")
        print("=" * 60)

        return self.result_gdf

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _run_extraction(self, gee_project: Optional[str]) -> None:
        print("\n[Stage 1] GEE Shoreline Extraction")
        from pyshore.extraction.gee_extractor import GEEExtractor
        extractor = GEEExtractor(self.cfg.extraction, gee_project=gee_project)
        sensor_map = extractor.run()
        # Feed sensor map back into analysis config
        self.cfg.analysis.sensor_per_year = sensor_map
        print(f"  Extracted {len(sensor_map)} annual shorelines.")

    def _load_shorelines(self) -> None:
        print("\n[Stage 2] Loading Shorelines")
        analysis = self.cfg.analysis

        # Build sensor_map from config or JSON on disk
        sensor_map = analysis.sensor_per_year
        if sensor_map is None:
            sm_path = os.path.join(analysis.shoreline_dir, "sensor_map.json")
            if os.path.exists(sm_path):
                with open(sm_path) as f:
                    sensor_map = {int(k): v for k, v in json.load(f).items()}

        self.shorelines = load_shorelines_from_dir(
            shoreline_dir=analysis.shoreline_dir,
            epsg=analysis.epsg,
            sensor_map=sensor_map,
        )

    def _build_transects(self) -> None:
        print("\n[Stage 3] Building Transects")
        analysis = self.cfg.analysis
        baseline = gpd.read_file(analysis.baseline_shapefile).to_crs(epsg=analysis.epsg)
        self.transects = generate_transects(
            baseline,
            spacing=analysis.transect_spacing,
            length=analysis.transect_length,
        )

    def _compute_intersections(self) -> None:
        print("\n[Stage 4] Computing Intersections")
        self.intersection_gdf = intersect_transects(self.transects, self.shorelines)

    def _compute_metrics(self) -> None:
        print("\n[Stage 5] Computing Metrics")
        analysis = self.cfg.analysis
        self.metrics_df = compute_all_metrics(
            self.intersection_gdf,
            min_years=analysis.min_years,
            min_detectable_change=analysis.min_detectable_change,
            baseline_year=analysis.baseline_year,
        )

    def _compute_uncertainty(self) -> None:
        print("\n[Stage 6] Uncertainty Analysis")
        analysis = self.cfg.analysis
        self.metrics_df = compute_uncertainty(
            self.intersection_gdf,
            self.metrics_df,
            bootstrap_n=analysis.bootstrap_n,
            confidence=analysis.confidence_level,
            baseline_year=analysis.baseline_year,
            min_years=analysis.min_years,
        )

    def _assemble_result(self) -> None:
        print("\n[Stage 7] Assembling Results")
        result = self.transects.copy()
        result = result.merge(self.metrics_df, on="transect_id", how="left")
        self.result_gdf = result
        print(f"  Result GeoDataFrame: {len(result)} transects × {len(result.columns)} columns")

    # ------------------------------------------------------------------
    # Convenience: run analysis only (skipping GEE extraction)
    # ------------------------------------------------------------------

    def run_analysis_only(self, skip_plots: bool = False) -> gpd.GeoDataFrame:
        """Run stages 2–9 (analysis + export), skipping GEE extraction."""
        return self.run(run_extraction=False, skip_plots=skip_plots)
