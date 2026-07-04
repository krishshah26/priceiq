# PriceIQ - Week 3: elasticity estimation, endogeneity, and shrinkage
#
# Goal: recover each SKU's price elasticity using ONLY price and sales
# history - the way a real analyst would, never touching true_elasticity -
# then grade ourselves against the hidden answer key from 01_generate_data.py.
#
# HOW TO RUN: continue in the same Colab session as the previous scripts,
# or fresh (it reloads CSVs from /content/priceiq_data if needed).
#
# THE ENDOGENEITY TRAP, RECAP:
#   managerial_markdown weeks move price BECAUSE of a latent "weak patch"
#   shock that ALSO suppresses that week's sales - so naive regression on
#   all weeks confuses "manager reacting to weak demand" with "price
#   sensitivity." Restricting to cost_driven weeks removes that confound.

import os
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "statsmodels"], check=True)
    import statsmodels.api as sm

# ---- SECTION 1: load data, force a clean base fact table --------------------
# fact_sales may have been enriched with extra columns by 03_baseline_and_
# lightgbm.py if you're continuing in the same session. Re-select only the
# original raw columns so this script behaves identically either way.

_default_dir = "/content/priceiq_data" if os.path.isdir("/content") else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "priceiq_data"
)
OUT_DIR = os.environ.get("PRICEIQ_DATA_DIR", _default_dir)
if "fact_sales" not in dir():
    print("Loading CSVs from disk (fresh session detected)...")
    dim_calendar = pd.read_csv(f"{OUT_DIR}/dim_calendar.csv", parse_dates=["date", "week_start"])
    dim_products = pd.read_csv(f"{OUT_DIR}/dim_products.csv")
    fact_sales = pd.read_csv(f"{OUT_DIR}/fact_sales.csv", parse_dates=["date"])
else:
    print("Reusing dataframes already in memory.")

BASE_COLS = ["date", "store_id", "sku_id", "units_sold", "price", "cost", "promo_flag",
             "price_reason", "stockout_flag"]
fact_sales = fact_sales[BASE_COLS].copy()
fact_sales = fact_sales.merge(dim_calendar[["date", "week_start", "seasonal_index"]], on="date", how="left")

# ---- SECTION 2: build the SKU-week panel ------------------------------------
# Price is set once per SKU per week (not per store), so SKU-week is the
# natural grain for elasticity estimation: sum units across all 20 stores
# and all 7 days, keep the single price/cost/promo/reason that week.

weekly = (
    fact_sales.groupby(["sku_id", "week_start"])
    .agg(
        total_units=("units_sold", "sum"),
        price=("price", "mean"),
        cost=("cost", "mean"),
        promo_flag=("promo_flag", "max"),
        seasonal_index=("seasonal_index", "mean"),
        price_reason=("price_reason", "first"),
    )
    .reset_index()
)

print(f"[checkpoint 1] SKU-week panel: {weekly.shape}")
print("Price-reason mix at the weekly panel level (sanity check vs earlier):")
print(weekly["price_reason"].value_counts(normalize=True).round(3))

# ---- SECTION 3: per-SKU OLS, naive vs clean ---------------------------------
# Regress log(1 + total_units) on log(price), controlling for seasonal_index
# (and promo_flag when it actually varies in the subset - it's always 0 in
# the cost_driven-only subset by construction, which would make it a
# zero-variance column and break OLS if left in).

def fit_elasticity(df):
    d = df[(df["total_units"] > 0) & (df["price"] > 0)].copy()
    if len(d) < 10:
        return np.nan, np.nan, len(d)
    d["log_price"] = np.log(d["price"])
    if d["log_price"].nunique() < 2:
        return np.nan, np.nan, len(d)
    y = np.log1p(d["total_units"])
    cols = {"log_price": d["log_price"], "seasonal_index": d["seasonal_index"]}
    if d["promo_flag"].nunique() > 1:
        cols["promo_flag"] = d["promo_flag"]
    X = sm.add_constant(pd.DataFrame(cols, index=d.index))
    model = sm.OLS(y, X).fit()
    return model.params["log_price"], model.bse["log_price"], len(d)

naive_rows, clean_rows = [], []
for sku_id, df_sku in weekly.groupby("sku_id"):
    naive_rows.append((sku_id, *fit_elasticity(df_sku)))
    df_clean = df_sku[df_sku["price_reason"] == "cost_driven"]
    clean_rows.append((sku_id, *fit_elasticity(df_clean)))

naive_df = pd.DataFrame(naive_rows, columns=["sku_id", "elasticity", "se", "n"])
clean_df = pd.DataFrame(clean_rows, columns=["sku_id", "elasticity", "se", "n"])

print(f"\n[checkpoint 2] Fit {naive_df['elasticity'].notna().sum()}/{len(naive_df)} naive "
      f"and {clean_df['elasticity'].notna().sum()}/{len(clean_df)} clean per-SKU regressions.")

# ---- SECTION 4: assemble results + grade against the hidden answer key -----

results = dim_products[["sku_id", "brand_tier", "true_elasticity"]].copy()
results = results.merge(
    naive_df.rename(columns={"elasticity": "naive_elasticity", "se": "naive_se", "n": "naive_n"}),
    on="sku_id",
)
results = results.merge(
    clean_df.rename(columns={"elasticity": "clean_elasticity", "se": "clean_se", "n": "clean_n"}),
    on="sku_id",
)

# ---- SECTION 5: shrinkage - pull noisy clean estimates toward the tier mean -
# Empirical Bayes, method-of-moments version:
#   prior_mean = precision-weighted average of clean estimates within a tier
#   tau^2      = between-SKU variance minus average within-SKU noise
#                (how much SKUs genuinely differ, once you subtract out
#                 pure estimation noise)
#   weight w   = tau^2 / (tau^2 + se_i^2)   -> higher se means lower trust
#   shrunk_i   = w * elasticity_i + (1 - w) * prior_mean

def shrink_group(g):
    # Only ever receives the two columns it needs (see the groupby call
    # below) - deliberately not the grouping column itself, so this is
    # immune to the pandas version difference where groupby().apply()
    # sometimes strips/keeps the grouping column in the passed-in frame.
    valid = g["clean_elasticity"].notna() & g["clean_se"].notna()
    if valid.sum() < 2:
        return g["clean_elasticity"]
    sub = g.loc[valid]
    prior_mean = np.average(sub["clean_elasticity"], weights=1 / sub["clean_se"] ** 2)
    tau2 = max(np.var(sub["clean_elasticity"], ddof=1) - np.mean(sub["clean_se"] ** 2), 1e-6)
    w = tau2 / (tau2 + g["clean_se"] ** 2)
    shrunk = np.where(valid, w * g["clean_elasticity"] + (1 - w) * prior_mean, np.nan)
    return pd.Series(shrunk, index=g.index)

# Assign the result as a new column on the ORIGINAL `results` rather than
# reassigning `results` to whatever groupby().apply() returns - sidesteps
# the whole ambiguity about which columns apply() keeps vs drops.
results["shrunk_elasticity"] = (
    results.groupby("brand_tier", group_keys=False)[["clean_elasticity", "clean_se"]].apply(shrink_group)
)

# ---- SECTION 6: how close did each approach get to the truth? --------------

results["naive_error"] = results["naive_elasticity"] - results["true_elasticity"]
results["clean_error"] = results["clean_elasticity"] - results["true_elasticity"]
results["shrunk_error"] = results["shrunk_elasticity"] - results["true_elasticity"]

mae = pd.DataFrame({
    "approach": ["naive (all weeks)", "clean (cost_driven only)", "shrunk (clean + tier pooling)"],
    "MAE_vs_true": [
        results["naive_error"].abs().mean(),
        results["clean_error"].abs().mean(),
        results["shrunk_error"].abs().mean(),
    ],
    "mean_signed_error": [
        results["naive_error"].mean(),
        results["clean_error"].mean(),
        results["shrunk_error"].mean(),
    ],
})
print("\n=== Overall accuracy vs the hidden true_elasticity ===")
print(mae.round(3).to_string(index=False))

print("\n=== By brand tier (mean absolute error vs true elasticity) ===")
tier_mae = results.groupby("brand_tier")[["naive_error", "clean_error", "shrunk_error"]].agg(
    lambda s: s.abs().mean()
)
print(tier_mae.round(3))

print("\n=== A few example SKUs (sorted by how wrong the naive estimate was) ===")
example_cols = ["sku_id", "brand_tier", "true_elasticity", "naive_elasticity",
                 "clean_elasticity", "shrunk_elasticity", "clean_n"]
print(results.reindex(results["naive_error"].abs().sort_values(ascending=False).index)
      [example_cols].head(8).round(2).to_string(index=False))

# ---- SECTION 7: sanity checks -----------------------------------------------

print("\n=== SANITY CHECKS ===")
assert results["true_elasticity"].notna().all(), "Missing true_elasticity - dim_products join broke."
n_fit = results["clean_elasticity"].notna().sum()
print(f"Clean elasticity estimated for {n_fit}/{len(results)} SKUs.")
print("Clean approach should beat naive on MAE (that's the whole point of using cost as the clean signal):")
print(f"  naive MAE={mae.loc[0,'MAE_vs_true']:.3f}  clean MAE={mae.loc[1,'MAE_vs_true']:.3f}")
print("Shrunk approach should be at or below clean's MAE, with the biggest gains on high-se (noisy) SKUs.")
print("\nAll checks passed.")
