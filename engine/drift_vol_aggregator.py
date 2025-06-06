# AI_Scenario_Sim/engine/drift_vol_aggregator.py
# ----------------------------------------------
"""
DriftVolAggregator
==================

Combines baseline μ/σ and event-driven deltas into 
year-by-year drift and covariance matrices, using
an array-based stochastic noise model.

Public API
----------
agg = DriftVolAggregator("data/asset_baseline.json", tickers)
combo = agg.combine(event_drift_for_year, vol_mult_for_year)
#   - event_drift_for_year: np.ndarray of shape (n_assets),
#       containing additive annual drift deltas (in decimals) from fired events for a single year.
#   - vol_mult_for_year:     float,
#       representing the total volatility multiplier for that single year.
#
# Returns:
#   {
#     "mu":  np.ndarray shape (n_assets),
#     "cov": np.ndarray shape (n_assets, n_assets)
#   }
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict


class DriftVolAggregator:
    def __init__(self, baseline_path: str | Path, tickers: List[str]):
        with open(baseline_path, "r") as f:
            base = json.load(f)

        # Store tickers as an instance attribute
        self.tickers = tickers # <-- ADDED THIS LINE

        # Load baseline drifts (in %) and volatilities (in %), convert to decimals
        self.base_mu  = np.array([base["mu"][t]    for t in tickers]) / 100.0
        self.base_sig = np.array([base["sigma"][t] for t in tickers]) / 100.0
        rho = base.get("rho", 0.75)

        # Build baseline covariance matrix assuming equal correlation = rho
        n = len(tickers)
        corr_mat = rho * np.ones((n, n))
        np.fill_diagonal(corr_mat, 1.0)
        self.base_cov = corr_mat * np.outer(self.base_sig, self.base_sig)

        # Master RNG placeholder (BatchRunner will assign .rng)
        self.rng = None  # type: ignore

    def combine(self,
                event_drift_for_year: np.ndarray,   # shape: (n_assets,) for a single year
                vol_mult_for_year:    float         # scalar for a single year
               ) -> Dict[str, np.ndarray]:
        """
        Combine baseline and event deltas for a single year into a drift & covariance.

        Steps:
          1) Initialize mu_this_year = baseline μ for this year.
          2) For each cell where event_drift_for_year != 0, draw noise ~ Normal(0, 0.4*|event_drift_for_year|).
             Add (drift + noise) to base μ for that asset.
          3) Build covariance for this year by scaling base_cov by (vol_mult_for_year)^2.

        Returns dict with keys "mu" (shape: n_assets) and "cov" (shape: n_assets, n_assets).
        """
        # 1) Initialize mu_this_year with baseline mu
        mu_this_year = self.base_mu.copy()  # shape (n_assets)

        # 2) Introduce stochastic noise proportional to |event_drift_for_year|
        noise = np.zeros_like(event_drift_for_year)
        # Mask where event_drift_for_year is nonzero
        nonzero_mask = np.abs(event_drift_for_year) > 0.0
        # Standard deviation per cell = 0.4 * |event_drift_for_year|
        sigma_vector = 0.4 * np.abs(event_drift_for_year)
        # Draw noise only where event_drift_for_year != 0
        noise[nonzero_mask] = self.rng.normal(
            loc=0.0,
            scale=sigma_vector[nonzero_mask]
        )
        # Final drift = baseline + event_drift_for_year + noise
        mu_this_year += event_drift_for_year + noise

        # 3) Build covariance for this year by scaling base_cov
        scale = vol_mult_for_year  # volumetric multiplier for the current year
        cov_this_year = self.base_cov * (scale ** 2)

        return {"mu": mu_this_year, "cov": cov_this_year}