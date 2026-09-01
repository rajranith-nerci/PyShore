"""
pyshore.visualization.plots
============================
Publication-quality plots for shoreline change analysis results.

Plots generated
---------------
1.  LRR / WLR / LMS distribution histograms (with KDE)
2.  Trend classification map (transects coloured by trend)
3.  Scatter: LRR vs R²
4.  Error-bar plot: LRR ± bootstrap CI along coast
5.  Time series for selected transects
6.  Uncertainty comparison: OLS CI vs Bootstrap CI
7.  NSM / EPR spatial profiles
8.  Sensor coverage timeline
"""

from __future__ import annotations

import os
from typing import Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

TREND_COLORS = {
    "Eroding":    "#d73027",   # red
    "Stable":     "#4575b4",   # blue
    "Accreting":  "#1a9850",   # green
    "Uncertain":  "#fee08b",   # yellow
    "No Data":    "#d9d9d9",   # grey
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_rate_histograms(result_gdf: pd.DataFrame, output_dir: str) -> None:
    """Overlay histogram of LRR, WLR, and LMS rates."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    pairs = [("LRR", "#4393c3"), ("WLR", "#d6604d"), ("LMS", "#74c476")]

    for ax, (col, color) in zip(axes, pairs):
        data = result_gdf[col].dropna()
        if data.empty:
            ax.set_title(f"{col} — no data")
            continue
        sns.histplot(data, bins=30, kde=True, color=color, ax=ax, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"{col} (m/yr)")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {col}")

    fig.suptitle("Shoreline Change Rate Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "rate_histograms.png"))


def plot_trend_map(result_gdf: gpd.GeoDataFrame, output_dir: str,
                   metric: str = "Trend_LRR") -> None:
    """Map of transects coloured by trend classification."""
    if metric not in result_gdf.columns:
        print(f"  [Plots] Column '{metric}' not found, skipping trend map.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    for trend, color in TREND_COLORS.items():
        subset = result_gdf[result_gdf[metric] == trend]
        if subset.empty:
            continue
        subset.plot(ax=ax, color=color, linewidth=1.2, label=trend)

    patches = [mpatches.Patch(color=c, label=t) for t, c in TREND_COLORS.items()
               if t in result_gdf[metric].unique()]
    ax.legend(handles=patches, loc="lower left", fontsize=9)
    ax.set_title(f"Shoreline Change Trend — {metric}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.axis("equal")
    _save(fig, os.path.join(output_dir, f"trend_map_{metric}.png"))


def plot_lrr_vs_r2(result_gdf: pd.DataFrame, output_dir: str) -> None:
    """Scatter: LRR vs R² — shows reliability of trend estimate."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = result_gdf["Trend_LRR"].map(TREND_COLORS).fillna("#d9d9d9")
    ax.scatter(result_gdf["LRR"], result_gdf["LRR_R2"], c=colors, alpha=0.6, edgecolors="none", s=20)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("LRR (m/yr)")
    ax.set_ylabel("R²")
    ax.set_title("LRR vs R² (goodness of fit)")
    patches = [mpatches.Patch(color=c, label=t) for t, c in TREND_COLORS.items()]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    _save(fig, os.path.join(output_dir, "LRR_vs_R2.png"))


def plot_uncertainty_errorbar(result_gdf: pd.DataFrame, output_dir: str) -> None:
    """Error-bar plot: LRR ± bootstrap CI along the coast."""
    df = result_gdf.dropna(subset=["LRR", "LRR_bs_ci_lo", "LRR_bs_ci_hi"]).copy()
    if df.empty:
        print("  [Plots] No bootstrap CI data, skipping error-bar plot.")
        return

    df = df.sort_values("transect_id")
    fig, ax = plt.subplots(figsize=(14, 5))
    x = df["transect_id"].values
    y = df["LRR"].values
    lo = y - df["LRR_bs_ci_lo"].values
    hi = df["LRR_bs_ci_hi"].values - y

    ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor="steelblue", alpha=0.4, linewidth=0.6)
    ax.scatter(x, y, c=df["Trend_LRR"].map(TREND_COLORS).fillna("#d9d9d9"),
               s=8, zorder=3, edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Transect ID")
    ax.set_ylabel("LRR (m/yr)")
    ax.set_title("LRR with Bootstrap 95% Confidence Interval")
    _save(fig, os.path.join(output_dir, "LRR_bootstrap_errorbar.png"))


def plot_time_series(
    intersection_gdf: pd.DataFrame,
    transect_ids: List[int],
    output_dir: str,
    result_gdf: Optional[pd.DataFrame] = None,
) -> None:
    """Time series of shoreline position for selected transects."""
    n = len(transect_ids)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)

    for ax, tid in zip(axes[0], transect_ids):
        group = intersection_gdf[intersection_gdf["transect_id"] == tid].sort_values("year")
        if group.empty:
            ax.set_title(f"Transect {tid} — no data")
            continue

        x = group["year"].values
        y = group["distance_m"].values
        ax.scatter(x, y, color="steelblue", zorder=5, s=40)
        ax.plot(x, y, color="steelblue", alpha=0.4)

        # Overlay regression lines
        if result_gdf is not None and not result_gdf.empty:
            row = result_gdf[result_gdf["transect_id"] == tid]
            if not row.empty:
                for col, color, label in [("LRR", "#4393c3", "LRR"),
                                           ("WLR", "#d6604d", "WLR"),
                                           ("LMS", "#74c476", "LMS")]:
                    rate = row[col].values[0]
                    intercept_col = f"{col}_intercept"
                    if np.isnan(rate) or intercept_col not in row.columns:
                        continue
                    intercept = row[intercept_col].values[0]
                    x_line = np.array([x.min(), x.max()])
                    ax.plot(x_line, intercept + rate * x_line, color=color,
                            linewidth=1.5, label=f"{label}: {rate:.2f} m/yr")
                ax.legend(fontsize=8)

        ax.set_xlabel("Year")
        ax.set_ylabel("Distance from baseline (m)")
        ax.set_title(f"Transect {tid}")

    fig.suptitle("Shoreline Position Time Series", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "time_series_selected.png"))


def plot_nsm_epr_profile(result_gdf: pd.DataFrame, output_dir: str) -> None:
    """Along-coast profiles of NSM and EPR."""
    df = result_gdf.sort_values("transect_id")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    for ax, col, label, color in [
        (axes[0], "NSM", "NSM (m)",    "#4393c3"),
        (axes[1], "EPR", "EPR (m/yr)", "#d6604d"),
    ]:
        data = df[col].values
        x    = df["transect_id"].values
        ax.fill_between(x, data, 0,
                        where=(data > 0), alpha=0.4, color="#1a9850", label="Accretion")
        ax.fill_between(x, data, 0,
                        where=(data < 0), alpha=0.4, color="#d73027", label="Erosion")
        ax.plot(x, data, color=color, linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc="upper right")

    axes[1].set_xlabel("Transect ID")
    fig.suptitle("Along-Coast Shoreline Change Profile", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "nsm_epr_profile.png"))


def plot_ols_vs_bootstrap(result_gdf: pd.DataFrame, output_dir: str) -> None:
    """Comparison: OLS 95% CI width vs Bootstrap 95% CI width."""
    df = result_gdf.dropna(subset=["LRR_CI_lo", "LRR_CI_hi",
                                    "LRR_bs_ci_lo", "LRR_bs_ci_hi"]).copy()
    if df.empty:
        return

    ols_width = df["LRR_CI_hi"] - df["LRR_CI_lo"]
    bs_width  = df["LRR_bs_ci_hi"] - df["LRR_bs_ci_lo"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ols_width, bs_width, alpha=0.5, s=15, color="steelblue", edgecolors="none")
    max_val = max(ols_width.max(), bs_width.max())
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, label="1:1 line")
    ax.set_xlabel("OLS 95% CI width (m/yr)")
    ax.set_ylabel("Bootstrap 95% CI width (m/yr)")
    ax.set_title("OLS vs Bootstrap Confidence Interval Width (LRR)")
    ax.legend()
    _save(fig, os.path.join(output_dir, "OLS_vs_bootstrap_CI.png"))


# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------

def generate_all_plots(
    result_gdf: gpd.GeoDataFrame,
    intersection_gdf: pd.DataFrame,
    output_dir: str,
    sample_transects: Optional[List[int]] = None,
) -> None:
    """
    Generate all standard plots.

    Parameters
    ----------
    result_gdf : GeoDataFrame
        Transects with all metrics.
    intersection_gdf : DataFrame
        Raw intersection points.
    output_dir : str
        Directory where plots are saved.
    sample_transects : list of int, optional
        Transect IDs for time-series plots. If None, 5 are auto-selected.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Plots] Generating plots in {output_dir} …")

    plot_rate_histograms(result_gdf, output_dir)
    plot_trend_map(result_gdf, output_dir, metric="Trend_LRR")
    plot_trend_map(result_gdf, output_dir, metric="Trend_WLR")
    plot_lrr_vs_r2(result_gdf, output_dir)
    plot_uncertainty_errorbar(result_gdf, output_dir)
    plot_nsm_epr_profile(result_gdf, output_dir)
    plot_ols_vs_bootstrap(result_gdf, output_dir)

    # Auto-select transects for time series if not provided
    if sample_transects is None:
        all_ids = result_gdf["transect_id"].dropna().astype(int).tolist()
        step = max(1, len(all_ids) // 5)
        sample_transects = all_ids[::step][:5]

    plot_time_series(intersection_gdf, sample_transects, output_dir, result_gdf)

    print(f"[Plots] Done. {len(os.listdir(output_dir))} files in {output_dir}")
