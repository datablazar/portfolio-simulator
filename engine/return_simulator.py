"""
AI_Scenario_Sim/engine/return_simulator.py
-----------------------------------------
Monthly Monte‑Carlo portfolio engine with quarterly re‑balancing.

Public API
----------
rs = ReturnSimulator(target_weights, monthly_contrib=100, rebalance="quarter")
res = rs.simulate_path(mu_year, cov_year)  # mu_year : (years × n_assets)
                                             cov_year: (years × n × n)
res keys:  'wealth_series' (np.ndarray months+1)
           'terminal_wealth' (float)
           'cagr' (float)
           'max_drawdown' (float)

Notes
-----
* Each annual (μ, Σ) is scaled to 12 monthly steps:
    μ_m  = μ / 12
    Σ_m  = Σ / 12          (≈ 1/12 variance → √(1/12) σ)
* Quarterly re‑balance: every 3rd month, holdings reset to `self.w_target`.
* Master RNG (`self.rng`) is supplied externally (BatchRunner overrides).
"""
from __future__ import annotations
import numpy as np
from typing import Dict


class ReturnSimulator:
    def __init__(self,
                 target_weights: np.ndarray,
                 monthly_contrib: float = 100.0,
                 rebalance: str = "quarter"):
        """`target_weights` must sum to 1 in the order of tickers used upstream."""
        self.w_target = target_weights.astype(float)
        if abs(self.w_target.sum() - 1) > 1e-6:
            raise ValueError("target_weights must sum to 1.0")

        self.contrib   = float(monthly_contrib)
        self.rebal_flag = rebalance  # "none" | "quarter"

        # RNG placeholder – BatchRunner will inject its master_rng
        self.rng = None  # type: ignore

    # ---------------------------------------------------------------------
    def simulate_path(self, mu_y: np.ndarray, cov_y: np.ndarray) -> Dict:
        """Simulate wealth path given annual mu & covariance.
        Parameters
        ----------
        mu_y  : (years × n_assets) annual arithmetic mean returns (decimal)
        cov_y : (years × n × n)    annual covariance matrices (decimal^2)
        """
        years, n = mu_y.shape
        months_total = years * 12

        wealth_series = np.zeros(months_total + 1)
        wealth = 0.01  # placeholder tiny initial capital to avoid 0/0 in CAGR
        weights = self.w_target.copy()

        # Pre‑compute Cholesky factors for speed
        chol_year = [np.linalg.cholesky(cov_y[y] / 12.0) for y in range(years)]

        m_idx = 0  # month index in wealth_series
        monthly_returns = []
        for y in range(years):
            mu_m = mu_y[y] / 12.0  # monthly mean vector
            chol = chol_year[y]

            # Draw 12 correlated monthly excess returns
            z = self.rng.standard_normal((12, n))  # (12 × n)
            monthly_ret = (mu_m + (z @ chol.T))  # (12 × n)

            for m in range(12):
                # Portfolio return this month
                port_r = monthly_ret[m] @ weights
                monthly_returns.append(port_r)
                wealth = wealth * (1.0 + port_r) + self.contrib
                m_idx += 1
                wealth_series[m_idx] = wealth

                # Re‑balance at quarter ends if requested
                if self.rebal_flag == "quarter" and (m % 3 == 2):
                    weights = self.w_target.copy()
                # otherwise weights evolve passively (buy‑and‑hold)

        # CAGR using time-weighted monthly returns
        if monthly_returns:
            prod_r = np.prod(1.0 + np.array(monthly_returns))
            cagr = prod_r ** (12.0 / months_total) - 1.0
        else:
            cagr = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(wealth_series[1:])
        drawdown = (wealth_series[1:] - peak) / peak
        max_dd = drawdown.min() if len(drawdown) else 0.0

        return {
            "wealth_series": wealth_series,
            "terminal_wealth": wealth,
            "cagr": cagr,
            "max_drawdown": max_dd
        }
