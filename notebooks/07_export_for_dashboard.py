# PriceIQ - Week 6 (part 1): export pipeline results for the dashboard
#
# Goal: a dashboard should READ precomputed results, not recompute a
# LightGBM backtest or an elasticity regression on every page load. This
# script is the seam between "pipeline" (notebooks 01-06, already run) and
# "presentation" (the Streamlit app) - the same separation dbt enforces
# between transformation and consumption.
#
# REQUIRES: run in the same Colab session as scripts 01-06 (needs
# fold_results_df, all_preds/monthly from 03; results from 04; opt_df,
# frontier_df, tier_bridge from 05; exp_summary + the raw/cuped stats
# from 06). If any are missing, this script tells you exactly which
# earlier script to rerun.
#
# AFTER RUNNING: download the whole DASH_DIR folder from Colab (zip it, or
# mount Drive) and place it next to app.py locally as `dashboard_data/`.

import os
import json
import pandas as pd

_default_out_dir = "/content/priceiq_data" if os.path.isdir("/content") else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "priceiq_data"
)
OUT_DIR = os.environ.get("PRICEIQ_DATA_DIR", _default_out_dir)
DASH_DIR = os.path.join(OUT_DIR, "dashboard")
os.makedirs(DASH_DIR, exist_ok=True)

REQUIRED = {
    "fold_results_df": "03_baseline_and_lightgbm.py",
    "monthly": "03_baseline_and_lightgbm.py",
    "results": "04_elasticity_estimation.py",
    "mae": "04_elasticity_estimation.py",
    "tier_mae": "04_elasticity_estimation.py",
    "opt_df": "05_price_optimization.py",
    "frontier_df": "05_price_optimization.py",
    "tier_bridge": "05_price_optimization.py",
    "exp_summary": "06_experiment_cuped.py",
}
_defined_names = dir()  # compute once, outside the comprehension - dir() inside a comprehension
                        # only sees the comprehension's own tiny scope, not the module's variables
missing = {name: script for name, script in REQUIRED.items() if name not in _defined_names}
if missing:
    raise RuntimeError(
        "Missing variables needed for export - rerun these scripts first, in this same "
        f"Colab session: {sorted(set(missing.values()))}"
    )

# ---- forecast results (Week 2) ----------------------------------------------
fold_results_df.to_csv(f"{DASH_DIR}/forecast_fold_results.csv", index=False)
monthly.reset_index().to_csv(f"{DASH_DIR}/forecast_monthly.csv", index=False)

# ---- elasticity results (Week 3) --------------------------------------------
results.to_csv(f"{DASH_DIR}/elasticity_results.csv", index=False)
mae.to_csv(f"{DASH_DIR}/elasticity_mae.csv", index=False)
tier_mae.reset_index().to_csv(f"{DASH_DIR}/elasticity_tier_mae.csv", index=False)

# ---- optimization results (Week 4) ------------------------------------------
opt_df.to_csv(f"{DASH_DIR}/optimization_results.csv", index=False)
frontier_df.to_csv(f"{DASH_DIR}/optimization_frontier.csv", index=False)
tier_bridge.reset_index().to_csv(f"{DASH_DIR}/optimization_tier_bridge.csv", index=False)

# ---- experiment results (Week 5) --------------------------------------------
exp_summary.to_csv(f"{DASH_DIR}/experiment_summary.csv", index=False)

experiment_stats = {
    "raw_diff": float(raw_diff), "raw_ci_lo": float(raw_ci[0]), "raw_ci_hi": float(raw_ci[1]),
    "raw_p": float(raw_p),
    "cuped_diff": float(cuped_diff), "cuped_ci_lo": float(cuped_ci[0]), "cuped_ci_hi": float(cuped_ci[1]),
    "cuped_p": float(cuped_p),
    "variance_reduction": float(var_reduction),
    "units_pct_change": float(units_pct_change),
    "guardrail_threshold": float(GUARDRAIL_THRESHOLD),
    "guardrail_breached": bool(units_pct_change < GUARDRAIL_THRESHOLD),
}
with open(f"{DASH_DIR}/experiment_stats.json", "w") as f:
    json.dump(experiment_stats, f, indent=2)

# ---- star schema for the "Ask the Data" NL-to-SQL feature -------------------
# Parquet instead of CSV for the fact table - same data, a fraction of the
# size, so the dashboard repo stays lightweight. We deliberately do NOT
# export ground_truth_demand or true_elasticity here - the dashboard is
# what a real analyst would see, and that answer key still doesn't belong
# in front of them.
#
# Reload fresh from the original CSV rather than trusting the in-memory
# `fact_sales` - earlier scripts in this session (04) reassign the global
# `fact_sales` to a trimmed column subset, so by this point in the session
# it may be missing columns (e.g. stockout_flag) that were never dropped
# from disk, only from that in-memory copy.

BASE_COLS = ["date", "store_id", "sku_id", "units_sold", "price", "cost", "promo_flag",
             "price_reason", "stockout_flag"]
fact_sales_export = pd.read_csv(f"{OUT_DIR}/fact_sales.csv", usecols=BASE_COLS, parse_dates=["date"])
fact_sales_export[BASE_COLS].to_parquet(f"{DASH_DIR}/fact_sales.parquet", index=False)
dim_calendar.to_csv(f"{DASH_DIR}/dim_calendar.csv", index=False)
dim_stores.to_csv(f"{DASH_DIR}/dim_stores.csv", index=False)
dim_products[["sku_id", "sku_name", "category", "brand_tier", "base_cost", "base_price"]].to_csv(
    f"{DASH_DIR}/dim_products.csv", index=False
)  # true_elasticity intentionally excluded

print(f"[checkpoint] Exported dashboard data to {DASH_DIR}:")
for f in sorted(os.listdir(DASH_DIR)):
    size_mb = os.path.getsize(os.path.join(DASH_DIR, f)) / 1e6
    print(f"  - {f} ({size_mb:.1f} MB)")

print(f"\nNext: copy or symlink this folder next to dashboard/app.py locally as `dashboard_data/` "
      f"(or just set PRICEIQ_DATA_DIR={DASH_DIR} when running the dashboard).")
if os.path.isdir("/content"):
    print("Quick zip-and-download in Colab:")
    print(f"  import shutil; shutil.make_archive('/content/priceiq_dashboard_data', 'zip', '{DASH_DIR}')")
    print("  from google.colab import files; files.download('/content/priceiq_dashboard_data.zip')")
