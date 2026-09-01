"""
pyshore.analysis.uncertainty
==============================
Full uncertainty analysis for shoreline change metrics.

Three complementary approaches
--------------------------------
1. OLS / WLR confidence intervals   — already in metrics.py via statsmodels
2. Bootstrap resampling             — non-parametric CI for all regression rates
3. Positional uncertainty propagation — propagates sensor-level σ through NSM / EPR

For each transect the bootstrap produces:
  • Bootstrap mean rate (should match OLS closely)
  • Bootstrap 2.5th / 97.5th percentile CI
  • Bootstrap standard deviation of rate

Positional uncertainty propagation:
  NSM_unc  = √(σ_last² + σ_first²)   [combined uncertainty, metres]
  EPR_unc  = NSM_unc / years_span     [m/yr]
"""

from __future__ import annotations

from typing import Optional, Dict
import warnings

import numpy as np
import pandas as pd
from sklearn.utils import resample
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_regression(
    x: np.ndarray,
    y: np.ndarray,
    n_iter: int = 1000,
    confidence: float = 0.95,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Bootstrap resampling for regression slope (LRR or WLR).

    Parameters
    ----------
    x, y : arrays
        Year and distance arrays.
    n_iter : int
        Number of bootstrap iterations.
    confidence : float
        Confidence level (e.g. 0.95 for 95% CI).
    weights : array, optional
        Observation weights. If provided, WLS is used; otherwise OLS.

    Returns
    -------
    dict with keys: bs_mean, bs_std, bs_ci_lo, bs_ci_hi
    """
    alpha = 1 - confidence
    slopes = []

    for _ in range(n_iter):
        indices = np.random.choice(len(x), size=len(x), replace=True)
        x_s, y_s = x[indices], y[indices]

        if len(np.unique(x_s)) < 2:
            continue

        X_s = sm.add_constant(x_s)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if weights is not None:
                    w_s = weights[indices]
                    model = sm.WLS(y_s, X_s, weights=w_s).fit()
                else:
                    model = sm.OLS(y_s, X_s).fit()
            slopes.append(model.params[1])
        except Exception:
            continue

    if len(slopes) < 10:
        nan = np.nan
        return {"bs_mean": nan, "bs_std": nan, "bs_ci_lo": nan, "bs_ci_hi": nan}

    slopes = np.array(slopes)
    return {
        "bs_mean": float(np.mean(slopes)),
        "bs_std": float(np.std(slopes)),
        "bs_ci_lo": float(np.percentile(slopes, 100 * alpha / 2)),
        "bs_ci_hi": float(np.percentile(slopes, 100 * (1 - alpha / 2))),
    }


# ---------------------------------------------------------------------------
# Positional uncertainty propagation
# ---------------------------------------------------------------------------

def propagate_positional_uncertainty(
    group: pd.DataFrame,
    baseline_year: Optional[int] = None,
) -> Dict[str, float]:
    """
    Propagate per-epoch positional uncertainty into NSM and EPR.

    Requires column 'uncertainty_m' in the group DataFrame.

    Returns
    -------
    dict with keys: NSM_unc, EPR_unc, years_span
    """
    if "uncertainty_m" not in group.columns:
        return {"NSM_unc": np.nan, "EPR_unc": np.nan, "years_span": np.nan}

    group = group.sort_values("year")

    if baseline_year is not None and baseline_year in group["year"].values:
        first_row = group[group["year"] == baseline_year].iloc[0]
    else:
        first_row = group.iloc[0]

    last_row = group.iloc[-1]

    sigma_first = float(first_row["uncertainty_m"])
    sigma_last  = float(last_row["uncertainty_m"])
    years_span  = float(last_row["year"] - first_row["year"])

    nsm_unc = np.sqrt(sigma_first ** 2 + sigma_last ** 2)
    epr_unc = nsm_unc / years_span if years_span > 0 else np.nan

    return {
        "NSM_unc": nsm_unc,
        "EPR_unc": epr_unc,
        "years_span": years_span,
    }


# ---------------------------------------------------------------------------
# Master uncertainty function
# ---------------------------------------------------------------------------

def compute_uncertainty(
    intersection_gdf: pd.DataFrame,
    metrics_df: pd.DataFrame,
    bootstrap_n: int = 1000,
    confidence: float = 0.95,
    baseline_year: Optional[int] = None,
    min_years: int = 3,
) -> pd.DataFrame:
    """
    Compute full uncertainty for each transect and merge into metrics_df.

    Adds columns
    ------------
    Bootstrap (LRR):   LRR_bs_mean, LRR_bs_std, LRR_bs_ci_lo, LRR_bs_ci_hi
    Bootstrap (WLR):   WLR_bs_mean, WLR_bs_std, WLR_bs_ci_lo, WLR_bs_ci_hi
    Positional:        NSM_unc, EPR_unc, years_span

    Parameters
    ----------
    intersection_gdf : GeoDataFrame / DataFrame
        Raw intersection data (transect_id, year, distance_m, uncertainty_m).
    metrics_df : DataFrame
        Output of compute_all_metrics().
    bootstrap_n : int
        Number of bootstrap iterations.
    confidence : float
        Confidence level.
    baseline_year : int or None
        Anchor year for positional uncertainty propagation.
    min_years : int
        Minimum observations required to run bootstrap.

    Returns
    -------
    DataFrame
        metrics_df with additional uncertainty columns.
    """
    has_uncertainty = "uncertainty_m" in intersection_gdf.columns
    uncertainty_records = []

    print(f"[Uncertainty] Running {bootstrap_n}-iteration bootstrap ...")

    for tid, group in intersection_gdf.groupby("transect_id"):
        group = group.sort_values("year")
        n = len(group)
        x = group["year"].values.astype(float)
        y = group["distance_m"].values.astype(float)

        record: Dict[str, float] = {"transect_id": tid}

        # --- Bootstrap LRR ---
        if n >= min_years:
            bs_lrr = bootstrap_regression(x, y, n_iter=bootstrap_n, confidence=confidence)
        else:
            bs_lrr = {"bs_mean": np.nan, "bs_std": np.nan,
                      "bs_ci_lo": np.nan, "bs_ci_hi": np.nan}
        record.update({f"LRR_{k}": v for k, v in bs_lrr.items()})

        # --- Bootstrap WLR ---
        if n >= min_years and has_uncertainty:
            sigma = group["uncertainty_m"].values.astype(float)
            sigma = np.where(sigma <= 0, 1.0, sigma)
            weights = 1.0 / sigma ** 2
            bs_wlr = bootstrap_regression(x, y, n_iter=bootstrap_n,
                                          confidence=confidence, weights=weights)
        else:
            bs_wlr = {"bs_mean": np.nan, "bs_std": np.nan,
                      "bs_ci_lo": np.nan, "bs_ci_hi": np.nan}
        record.update({f"WLR_{k}": v for k, v in bs_wlr.items()})

        # --- Positional uncertainty propagation ---
        pos_unc = propagate_positional_uncertainty(group, baseline_year)
        record.update(pos_unc)

        uncertainty_records.append(record)

    unc_df = pd.DataFrame(uncertainty_records)

    # Merge into metrics_df (avoid duplicate columns)
    existing_cols = set(metrics_df.columns) - {"transect_id"}
    new_cols = [c for c in unc_df.columns if c not in existing_cols or c == "transect_id"]
    merged = metrics_df.merge(unc_df[new_cols], on="transect_id", how="left")

    print(f"[Uncertainty] Done. Added {len(new_cols) - 1} uncertainty columns.")
    return merged
