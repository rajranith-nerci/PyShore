"""
pyshore.analysis.metrics
=========================
DSAS-equivalent shoreline change statistics with extended metrics.

Metrics computed per transect
------------------------------
NSM   – Net Shoreline Movement (m)
EPR   – End Point Rate (m/yr)
SCE   – Shoreline Change Envelope (m)
LRR   – Linear Regression Rate via OLS (m/yr)
WLR   – Weighted Linear Regression Rate (m/yr) — weights by 1/σ²
LMS   – Least Median of Squares / Theil-Sen robust regression (m/yr)

All regression metrics also produce:
  • Standard error
  • 95% confidence interval bounds
  • R² (where applicable)
  • p-value (where applicable)
  • Trend classification (Eroding / Stable / Accreting / Uncertain / No Data)
"""

from __future__ import annotations

from typing import Optional, Dict
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import TheilSenRegressor


# ---------------------------------------------------------------------------
# Trend classification
# ---------------------------------------------------------------------------

def classify_trend(
    rate: Optional[float],
    p_value: Optional[float],
    threshold: float = 0.5,
    alpha: float = 0.05,
) -> str:
    """
    Classify shoreline trend.

    Parameters
    ----------
    rate : float or None
        Change rate (m/yr). Positive = accretion, negative = erosion.
    p_value : float or None
        Statistical significance.
    threshold : float
        Minimum rate magnitude (m/yr) to classify as Eroding/Accreting.
    alpha : float
        Significance level.
    """
    if rate is None or np.isnan(rate):
        return "No Data"
    if p_value is None or np.isnan(p_value) or p_value >= alpha:
        return "Uncertain"
    if rate < -threshold:
        return "Eroding"
    if rate > threshold:
        return "Accreting"
    return "Stable"


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def _nsm_epr_sce(group: pd.DataFrame, min_change: float, baseline_year: Optional[int]):
    """Compute NSM, EPR, SCE for one transect."""
    group = group.sort_values("year")

    if len(group) < 2:
        return np.nan, np.nan, np.nan

    if baseline_year is not None and baseline_year in group["year"].values:
        first_row = group[group["year"] == baseline_year].iloc[0]
    else:
        first_row = group.iloc[0]

    last_row = group.iloc[-1]

    nsm = last_row["distance_m"] - first_row["distance_m"]
    years_diff = last_row["year"] - first_row["year"]
    epr = nsm / years_diff if years_diff > 0 else np.nan
    sce = group["distance_m"].max() - group["distance_m"].min()

    if abs(nsm) < min_change:
        nsm = np.nan
        epr = np.nan

    return nsm, epr, sce


def _ols_regression(x: np.ndarray, y: np.ndarray):
    """OLS linear regression. Returns (slope, intercept, se, ci_lo, ci_hi, r2, p)."""
    if len(x) < 3:
        nan = np.nan
        return nan, nan, nan, nan, nan, nan, nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

    slope = model.params[1]
    intercept = model.params[0]
    se = model.bse[1]
    ci = model.conf_int(alpha=0.05)
    ci_arr = ci.values if hasattr(ci, "values") else ci
    ci_lo, ci_hi = float(ci_arr[1, 0]), float(ci_arr[1, 1])
    r2 = model.rsquared
    p = model.pvalues[1]
    return slope, intercept, se, ci_lo, ci_hi, r2, p


def _wlr_regression(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    """
    Weighted Linear Regression (WLR).
    Weights = 1/σ² where σ is positional uncertainty (metres).
    Returns (slope, intercept, se, ci_lo, ci_hi, r2, p).
    """
    if len(x) < 3:
        nan = np.nan
        return nan, nan, nan, nan, nan, nan, nan

    # Replace zero/nan weights with smallest positive weight
    weights = np.where(
        (weights <= 0) | np.isnan(weights),
        np.nanmin(weights[weights > 0]) if np.any(weights > 0) else 1.0,
        weights,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = sm.add_constant(x)
        model = sm.WLS(y, X, weights=weights).fit()

    slope = model.params[1]
    intercept = model.params[0]
    se = model.bse[1]
    ci = model.conf_int(alpha=0.05)
    ci_arr = ci.values if hasattr(ci, "values") else ci
    ci_lo, ci_hi = float(ci_arr[1, 0]), float(ci_arr[1, 1])
    r2 = model.rsquared
    p = model.pvalues[1]
    return slope, intercept, se, ci_lo, ci_hi, r2, p


def _theil_sen_regression(x: np.ndarray, y: np.ndarray):
    """
    Theil-Sen robust regression (equivalent to DSAS LMS intent).
    Resistant to up to ~29% outliers.
    Returns (slope, intercept, se_approx, ci_lo, ci_hi, r2_approx, p_approx).
    """
    if len(x) < 3:
        nan = np.nan
        return nan, nan, nan, nan, nan, nan, nan

    # scipy Theil-Sen gives slope + CI directly
    slope, intercept, lo_slope, hi_slope = stats.theilslopes(y, x, alpha=0.05)

    # Approximate standard error from CI
    se_approx = (hi_slope - lo_slope) / (2 * 1.96)

    # R² approximation using predicted values from Theil-Sen line
    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_approx = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Significance test using Mann-Kendall tau
    tau, p_approx = stats.kendalltau(x, y)

    return slope, intercept, se_approx, lo_slope, hi_slope, r2_approx, p_approx


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_all_metrics(
    intersection_gdf: pd.DataFrame,
    min_years: int = 3,
    min_detectable_change: float = 5.0,
    baseline_year: Optional[int] = None,
    lrr_threshold: float = 0.5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute all DSAS-equivalent metrics for every transect.

    Parameters
    ----------
    intersection_gdf : GeoDataFrame / DataFrame
        Output from intersect_transects(). Must have columns:
        transect_id, year, distance_m.
        Optional: uncertainty_m (used for WLR weights).
    min_years : int
        Minimum observations required for regression metrics.
    min_detectable_change : float
        NSM values below this are set to NaN.
    baseline_year : int or None
        Anchor year for NSM/EPR.
    lrr_threshold : float
        Rate threshold (m/yr) for trend classification.
    alpha : float
        Significance level for trend classification.

    Returns
    -------
    DataFrame
        One row per transect with all computed metrics.
    """
    has_uncertainty = "uncertainty_m" in intersection_gdf.columns
    results = []

    for tid, group in intersection_gdf.groupby("transect_id"):
        group = group.sort_values("year").reset_index(drop=True)
        n = len(group)
        x = group["year"].values.astype(float)
        y = group["distance_m"].values.astype(float)

        # Positional uncertainty → WLR weights (1/σ²)
        if has_uncertainty:
            sigma = group["uncertainty_m"].values.astype(float)
            sigma = np.where(sigma <= 0, 1.0, sigma)
            weights = 1.0 / (sigma ** 2)
        else:
            weights = np.ones(n)

        # --- NSM / EPR / SCE ---
        nsm, epr, sce = _nsm_epr_sce(group, min_detectable_change, baseline_year)

        # --- LRR (OLS) ---
        if n >= min_years:
            lrr, lrr_int, lrr_se, lrr_ci_lo, lrr_ci_hi, lrr_r2, lrr_p = _ols_regression(x, y)
        else:
            lrr = lrr_int = lrr_se = lrr_ci_lo = lrr_ci_hi = lrr_r2 = lrr_p = np.nan

        # --- WLR (Weighted OLS) ---
        if n >= min_years:
            wlr, wlr_int, wlr_se, wlr_ci_lo, wlr_ci_hi, wlr_r2, wlr_p = _wlr_regression(x, y, weights)
        else:
            wlr = wlr_int = wlr_se = wlr_ci_lo = wlr_ci_hi = wlr_r2 = wlr_p = np.nan

        # --- LMS / Theil-Sen ---
        if n >= min_years:
            lms, lms_int, lms_se, lms_ci_lo, lms_ci_hi, lms_r2, lms_p = _theil_sen_regression(x, y)
        else:
            lms = lms_int = lms_se = lms_ci_lo = lms_ci_hi = lms_r2 = lms_p = np.nan

        # --- Trend classification (based on LRR as primary) ---
        trend_lrr = classify_trend(lrr, lrr_p, lrr_threshold, alpha)
        trend_wlr = classify_trend(wlr, wlr_p, lrr_threshold, alpha)
        trend_lms = classify_trend(lms, lms_p, lrr_threshold, alpha)

        results.append({
            "transect_id": tid,
            "n_obs": n,
            "year_min": int(group["year"].min()),
            "year_max": int(group["year"].max()),
            # Simple metrics
            "NSM": nsm,
            "EPR": epr,
            "SCE": sce,
            # LRR (OLS)
            "LRR": lrr,
            "LRR_intercept": lrr_int,
            "LRR_SE": lrr_se,
            "LRR_CI_lo": lrr_ci_lo,
            "LRR_CI_hi": lrr_ci_hi,
            "LRR_R2": lrr_r2,
            "LRR_p": lrr_p,
            "Trend_LRR": trend_lrr,
            # WLR
            "WLR": wlr,
            "WLR_intercept": wlr_int,
            "WLR_SE": wlr_se,
            "WLR_CI_lo": wlr_ci_lo,
            "WLR_CI_hi": wlr_ci_hi,
            "WLR_R2": wlr_r2,
            "WLR_p": wlr_p,
            "Trend_WLR": trend_wlr,
            # LMS / Theil-Sen
            "LMS": lms,
            "LMS_intercept": lms_int,
            "LMS_SE": lms_se,
            "LMS_CI_lo": lms_ci_lo,
            "LMS_CI_hi": lms_ci_hi,
            "LMS_R2": lms_r2,
            "LMS_p": lms_p,
            "Trend_LMS": trend_lms,
        })

    df = pd.DataFrame(results)
    print(f"[Metrics] Computed {len(df)} transect records.")
    return df
