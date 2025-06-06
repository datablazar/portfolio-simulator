#!/usr/bin/env python3
# AI_Scenario_Sim/main.py
# -----------------------
"""
Entry-point for the AI scenario Monte-Carlo engine.
Usage:
    python main.py --paths 20000 --contrib 100 [--seed SEED]
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

from engine.batch_runner     import BatchRunner
from engine.reporting        import save_summary_csv, fan_chart, cagr_histogram
from engine.macro            import MACRO_STATES, MACRO_PROBS, MACRO_MAP


# --------------------------------------------------------------------- CLI ---
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths",   type=int,   default=20_000,
                    help="number of Monte-Carlo worlds")
    ap.add_argument("--contrib", type=float, default=100.0,
                    help="monthly contribution (£)")
    ap.add_argument("--seed",    type=int,   default=None,
                    help="optional PRNG seed for reproducibility")
    args = ap.parse_args()

    # ----------------------------------------------------------------- tickers ---
    base = json.load(open("data/asset_baseline.json"))
    tickers = list(base["mu"].keys())

    # ----------------------------------------------------- run Monte-Carlo batch ---
    runner = BatchRunner(n_paths=args.paths,
                         tickers=tickers,
                         monthly_contrib=args.contrib,
                         seed=args.seed)

    # Inside BatchRunner.run(), each path does:
    #   1) Sample AI timeline
    #   2) Simulate event tree → ev_out["drift"], ev_out["volmul"]
    #   3) Draw one macro-state and get (mu_shift, sigma_mult)
    #   4) Call agg.combine(ev_out["drift"], ev_out["volmul"]) to get mu_arr, cov_arr
    #   5) Apply mu_shift and sigma_mult to mu_arr, cov_arr
    #   6) Pass adjusted arrays to rs.simulate_path()

    summary = runner.run(store_paths=True)
    json_path = runner.save(summary)

    # --------------------------------------------------------------- save summary -
    csv_path = save_summary_csv(summary, "results")

    # ----------------------------------------------------------------- make plots -
    stamp = time.strftime("%Y%m%d_%H%M")
    fan_chart(np.array(summary["wealth_matrix"]),
              f"results/fan_chart_{stamp}.png")
    cagr_histogram(np.array(summary["cagrs_raw"]),
                   f"results/cagr_hist_{stamp}.png")

    # ------------------------------------------------------------------- console --
    print("\nMonte-Carlo complete ✅")
    print(f"Worlds simulated  : {args.paths:,}")
    print(f"Monthly contrib £ : {args.contrib}")
    seed_label = args.seed if args.seed is not None else "random"
    print("Seed              :", seed_label)
    print("Summary JSON      :", json_path)
    print("Master CSV        :", csv_path)
    print("Fan chart PNG     :", f"results/fan_chart_{stamp}.png")
    print("CAGR hist PNG     :", f"results/cagr_hist_{stamp}.png")
    print("\nTerminal wealth percentiles (5,25,50,75,95):",
          summary["terminal_wealth_percentiles"])
    print("CAGR percentiles (5,25,50,75,95)             :",
          summary["cagr_percentiles"])
    print("Max draw-down percentiles (5,50,95)          :",
          summary["max_dd_percentiles"])


if __name__ == "__main__":
    main()
