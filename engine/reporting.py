# AI_Scenario_Sim/engine/reporting.py
# -----------------------------------
"""
Reporting helpers
• save_summary_csv   – one-row CSV with percentiles and metadata
• fan_chart          – wealth percentile ribbons
• cagr_histogram     – histogram of CAGRs
"""

from __future__ import annotations
from pathlib import Path
import csv, time, numpy as np
import matplotlib.pyplot as plt

# ---------- CSV --------------------------------------------------------------

def save_summary_csv(summary: dict, out_dir: str | Path) -> Path:
    """
    Appends/creates a CSV with one row of headline stats.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "summary_master.csv"
    headers = [
        "timestamp","n_paths",
        "p5_w","p25_w","p50_w","p75_w","p95_w",
        "p5_cagr","p25_cagr","p50_cagr","p75_cagr","p95_cagr"
    ]
    row = [time.strftime("%Y-%m-%d %H:%M"),
           summary["n_paths"],
           *summary["terminal_wealth_percentiles"],
           *summary["cagr_percentiles"]]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(headers)
        w.writerow(row)
    return path

# ---------- Fan chart --------------------------------------------------------

def fan_chart(wealth_matrix: np.ndarray, out_file: str | Path):
    """
    wealth_matrix :  (paths, years+1)  include t0
    Saves a PNG fan-chart of wealth percentiles over time.
    """
    years = wealth_matrix.shape[1] - 1
    qs = [5,25,50,75,95]
    pct = np.percentile(wealth_matrix, qs, axis=0)   # shape (5, years+1)

    x = np.arange(years+1)
    plt.figure(figsize=(8,5))
    plt.plot(x, pct[2], label="median")              # 50th
    plt.fill_between(x, pct[1], pct[3], alpha=0.3, label="25–75 %")
    plt.fill_between(x, pct[0], pct[4], alpha=0.15, label="5–95 %")
    plt.title("Wealth fan chart")
    plt.xlabel("Years")
    plt.ylabel("£")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=120)
    plt.close()

# ---------- CAGR histogram ---------------------------------------------------

def cagr_histogram(cagrs: np.ndarray, out_file: str | Path):
    plt.figure(figsize=(6,4))
    plt.hist(cagrs*100, bins=40, density=True)
    plt.title("CAGR distribution")
    plt.xlabel("CAGR %")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_file, dpi=120)
    plt.close()
