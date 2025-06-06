# AI_Scenario_Sim/engine/batch_runner.py
# --------------------------------------
"""
BatchRunner
===========

Monte-Carlo driver:
  • draws a capability timeline
  • triggers the event tree yearly
  • aggregates drift/vol (with stochastic noise)
  • simulates wealth path yearly
  • stores summary metrics

Saves a JSON file like:
  results/scenario_run_20250604_1432.json
"""

from __future__ import annotations
from pathlib import Path
import json, time, numpy as np
from typing import Dict, List

# Import the new get_current_macro_state function
from .timeline_sampler     import TimelineSampler
from .event_tree_engine    import EventTreeEngine
from .drift_vol_aggregator import DriftVolAggregator
from .return_simulator     import ReturnSimulator
# Import get_current_macro_state and MACRO_MAP (which now holds asset-specific shifts)
from .macro                import get_current_macro_state, MACRO_MAP

class BatchRunner:
    def __init__(self,
                 n_paths: int,
                 tickers: List[str],
                 monthly_contrib: float = 100.0,
                 seed: int | None = None,
                 horizon_years: int = 40): # Added horizon_years to init
        
        self.n_paths = n_paths
        self.tickers = tickers
        self.horizon_years = horizon_years # Store horizon_years

        # --- Use InvestEngine allocations ---
        TARGET_ALLOC = {
            "PACW": 0.25, "SEMI": 0.06, "VAGS": 0.05, "EMIM": 0.05, "XAIX": 0.05,
            "ITPS": 0.05, "IWFQ": 0.04, "RBTX": 0.04, "WLDS": 0.04, "MINV": 0.04,
            "IWFV": 0.04, "MEUD": 0.03, "EQQQ": 0.03, "PRIJ": 0.03, "XCX5": 0.03,
            "WDEP": 0.03, "LCUK": 0.03, "WCOM": 0.03, "ISPY": 0.03, "NUCG": 0.03,
            "SGLN": 0.02
        }
        self.weights = np.array([TARGET_ALLOC[t] for t in tickers])

        # Create a single master RNG (seed=None → random)
        self.master_rng = np.random.default_rng(seed)

        # Initialize modules
        self.ts = TimelineSampler("data/timeline_buckets.json")
        self.et = EventTreeEngine("data/events_catalogue.json",
                                  tickers=tickers,
                                  horizon_years=self.horizon_years) # Pass horizon_years
        self.agg = DriftVolAggregator("data/asset_baseline.json", tickers)
        self.rs  = ReturnSimulator(self.weights, monthly_contrib) 

        # Override each module's .rng to use the shared master RNG
        self.ts.rng  = self.master_rng
        self.et.rng  = self.master_rng
        self.agg.rng = self.master_rng
        self.rs.rng  = self.master_rng


    # -----------------------------------------------------------------
    def run(self, store_paths: bool = False) -> Dict:
        # --- ENSURE THESE LISTS ARE INITIALIZED AT THE START OF THE METHOD ---
        wealths: List[float] = [] 
        cagrs: List[float] = []
        maxdds: List[float] = []
        
        if store_paths:
            all_wealth_series_for_paths: List[np.ndarray] = []
        
        bucket_counts = {b["name"]:0 for b in
                         json.load(open("data/timeline_buckets.json"))["buckets"]}
        fired_counts  = {}

        for i in range(self.n_paths):
            # Per-path initializations
            timeline = self.ts.sample_timeline()
            bucket_counts[timeline["bucket"]] += 1
            
            # These arrays will store the combined mu and cov for each year of this path
            path_mu_arr  = np.zeros((self.horizon_years, len(self.tickers)))
            path_cov_arr = np.zeros((self.horizon_years, len(self.tickers), len(self.tickers)))
            
            # Track E-P3 event and AGI breakthrough year for dynamic macro
            current_triggered_ep3_event_id: str | None = None
            agi_breakthrough_year = timeline.get("agi_year")

            # Initialize wealth for this path
            current_wealth_series_path = np.zeros(self.horizon_years * 12 + 1)
            current_wealth = 0.01 # placeholder tiny initial capital
            current_wealth_series_path[0] = current_wealth

            # --- Simulate year by year for this path ---
            for year in range(self.horizon_years):
                # 2) Simulate event tree for the current year
                event_results = self.et.simulate_events(
                    timeline=timeline,
                    current_year=year
                )
                
                # Update fired_counts based on events that fired *this year*
                for ev in event_results["fired"]:
                    fired_counts[ev.id] = fired_counts.get(ev.id, 0) + 1

                # Update the persistent E-P3 event ID for this path
                # If a new E-P3 event fired, it overrides previous ones
                if event_results["triggered_ep3_event_id"] is not None:
                    current_triggered_ep3_event_id = event_results["triggered_ep3_event_id"]

                # 3) Determine macro-state for this year using the dynamic logic
                macro_state_this_year = get_current_macro_state(
                    current_year=year,
                    agi_breakthrough_year=agi_breakthrough_year,
                    triggered_ep3_event_id=current_triggered_ep3_event_id
                )
                
                # Retrieve asset-specific mu_shift and sigma_mult from MACRO_MAP
                mu_shifts_dict, sigma_mults_dict = MACRO_MAP[macro_state_this_year]

                # Convert dictionary shifts/multipliers to numpy arrays aligned with tickers
                # Initialize with DEFAULT values
                macro_mu_shift_arr = np.full(len(self.tickers), mu_shifts_dict.get("DEFAULT", 0.00))
                macro_sigma_mult_arr = np.full(len(self.tickers), sigma_mults_dict.get("DEFAULT", 1.00))

                # Apply asset-specific overrides
                for idx, ticker in enumerate(self.tickers):
                    if ticker in mu_shifts_dict:
                        macro_mu_shift_arr[idx] = mu_shifts_dict[ticker]
                    if ticker in sigma_mults_dict:
                        macro_sigma_mult_arr[idx] = sigma_mults_dict[ticker]
                
                # 4) Combine baseline + event-induced drifts/vols for this year
                combo = self.agg.combine(
                    event_results["drift"], # event_results["drift"] is for current year
                    event_results["volmul"] # event_results["volmul"] is for current year
                )
                mu_year_combined  = combo["mu"]   # shape: (n_assets)
                cov_year_combined = combo["cov"]  # shape: (n_assets, n_assets)

                # 5a) Apply macro-state mu_shift PER ASSET for this year
                mu_year_combined = mu_year_combined + macro_mu_shift_arr

                # 5b) Scale covariance by sigma_mult^2 PER ASSET for this year
                sigma_mult_matrix = np.diag(macro_sigma_mult_arr)
                cov_year_combined = sigma_mult_matrix @ cov_year_combined @ sigma_mult_matrix.T
                
                # Store adjusted mu and cov for this specific year
                path_mu_arr[year] = mu_year_combined
                path_cov_arr[year] = cov_year_combined
            
            # --- After annual loops, simulate the full path with aggregated annual data ---
            res = self.rs.simulate_path(path_mu_arr, path_cov_arr)
            
            if store_paths:
                all_wealth_series_for_paths.append(res["wealth_series"])

            wealths.append(res["terminal_wealth"])
            cagrs.append(res["cagr"])
            maxdds.append(res["max_drawdown"])

        if store_paths:
            wealth_matrix = np.vstack(all_wealth_series_for_paths)

        wealths = np.array(wealths)
        cagrs   = np.array(cagrs)
        maxdds  = np.array(maxdds)

        summary = {
            "n_paths": self.n_paths,
            "terminal_wealth_percentiles": np.percentile(
                wealths, [5,25,50,75,95]).round(0).tolist(),
            "cagr_percentiles": (np.percentile(cagrs, [5,25,50,75,95])*100
                                 ).round(2).tolist(),
            "max_dd_percentiles": (np.percentile(maxdds, [5,50,95])*100
                                   ).round(1).tolist(),
            "bucket_frequency": {k: round(v/self.n_paths*100,2)
                                 for k,v in bucket_counts.items()},
            "event_frequency": {k: round(v/self.n_paths*100,2)
                                for k,v in fired_counts.items()}
        }

        if store_paths:
            summary["wealth_matrix"] = wealth_matrix.tolist()
            summary["cagrs_raw"]     = cagrs.tolist()

        return summary


    # -----------------------------------------------------------------
    def save(self, summary: Dict, out_dir: str | Path = "results") -> Path:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M")
        path = Path(out_dir)/f"scenario_run_{ts}.json"
        json.dump(summary, open(path,"w"), indent=2)
        return path