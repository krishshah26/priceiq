# PriceIQ - Week 5: experiment design + CUPED
#
# Goal: before trusting the Week 4 price recommendations, run a randomized
# test and analyze it properly - store-level randomization (to avoid
# substitution contaminating results), CUPED variance reduction (critical
# with only 20 stores), a confidence interval on the lift, and a guardrail
# metric so a margin "win" that craters volume doesn't sneak through.
#
# REQUIRES: `opt_df`, `state` from 05_price_optimization.py (treatment
# prices) and `fact_sales`/`dim_stores`/`dim_products` in memory.
#
# HONEST LIMITATION: we don't have real stores, so "running the experiment"
# means simulating an 8-week post-period using the SAME hidden true
# elasticity that generated everything else. This validates the ANALYSIS
# MECHANICS (randomization, CUPED, CIs, guardrails) - not a real-world
# confirmation of the pricing model. In a real job this step is precisely
# where you'd catch a bad model before full rollout.
#
# SIMPLIFICATION: no stockout simulation during the experiment (a real test
# would ensure adequate inventory so a shelf outage doesn't confound the
# read) - that keeps this script focused on experiment/CUPED mechanics
# instead of re-deriving Week 1's inventory model.

import numpy as np
import pandas as pd
from scipy import stats

if "opt_df" not in dir() or "state" not in dir():
    raise RuntimeError(
        "Missing `opt_df`/`state` from 05_price_optimization.py. "
        "Run that script first in this same Colab session."
    )

rng = np.random.default_rng(123)

# ---- SECTION 1: randomize stores into arms, blocked by store size ----------
# Blocking (randomizing within each size stratum, not across the whole
# population) guarantees the arms end up balanced on store size instead of
# leaving it to chance - store size is exactly the kind of variable that
# would otherwise swamp the signal.

stores = dim_stores.copy()
arm = pd.Series(index=stores.index, dtype=object)
for size, group in stores.groupby("store_size"):
    idx = group.index.to_numpy().copy()  # .to_numpy() can be read-only; shuffle needs a writeable array
    rng.shuffle(idx)
    half = len(idx) // 2
    arm.loc[idx[:half]] = "treatment"
    arm.loc[idx[half:]] = "control"
stores["arm"] = arm

print("[checkpoint 1] Store assignment (should be balanced within each size bucket):")
print(pd.crosstab(stores["store_size"], stores["arm"]))

# ---- SECTION 2: simulate the 8-week experiment period -----------------------
# Treatment stores get the Week 4 optimized price per SKU; control stores
# hold at the current (pre-experiment) price. Same demand mechanics as
# 01_generate_data.py: elasticity response, seasonality, a fresh latent
# demand-state shock, negative-binomial noise - just no stockout censoring.

EXP_START = pd.Timestamp("2025-01-01")
EXP_DAYS = 8 * 7
exp_dates = pd.date_range(EXP_START, periods=EXP_DAYS, freq="D")

MONTH_SEASONALITY = {1: 0.85, 2: 0.85, 3: 0.95, 4: 0.95, 5: 1.00, 6: 1.00,
                      7: 1.00, 8: 1.05, 9: 1.00, 10: 1.05, 11: 1.25, 12: 1.35}
exp_seasonal = pd.Series(exp_dates.month).map(MONTH_SEASONALITY).to_numpy()
exp_weekday_mult = np.where(exp_dates.dayofweek >= 5, 1.15, 1.0)

N_STORES, N_SKUS = len(stores), len(dim_products)
store_idx_for_pair = np.repeat(np.arange(N_STORES), N_SKUS)
sku_idx_for_pair = np.tile(np.arange(N_SKUS), N_STORES)

store_ids_arr = stores["store_id"].to_numpy()
sku_ids_arr = dim_products["sku_id"].to_numpy()
arm_for_pair = stores["arm"].to_numpy()[store_idx_for_pair]

price_by_sku = opt_df.set_index("sku_id")
current_price_arr = price_by_sku["current_price"].reindex(sku_ids_arr).to_numpy()
treatment_price_arr = price_by_sku["opt_price"].reindex(sku_ids_arr).to_numpy()
price_for_pair = np.where(
    arm_for_pair == "treatment", treatment_price_arr[sku_idx_for_pair], current_price_arr[sku_idx_for_pair]
)

base_price_for_pair = dim_products["base_price"].to_numpy()[sku_idx_for_pair]
elasticity_for_pair = dim_products["true_elasticity"].to_numpy()[sku_idx_for_pair]  # secret truth, used only to
                                                                                     # generate what "really happens"
base_demand_for_pair = dim_products["base_daily_demand"].to_numpy()[sku_idx_for_pair]
traffic_for_pair = stores["traffic_multiplier"].to_numpy()[store_idx_for_pair]

price_effect = (price_for_pair / base_price_for_pair) ** elasticity_for_pair  # constant across the test window

demand_state_sku = np.zeros(N_SKUS)  # fresh start - short-memory AR(1), a few weeks in it forgets history anyway
state_by_sku_day = np.zeros((N_SKUS, EXP_DAYS))
for d in range(EXP_DAYS):
    demand_state_sku = 0.85 * demand_state_sku + rng.normal(0, 0.30, size=N_SKUS)
    state_by_sku_day[:, d] = demand_state_sku
state_by_pair_day = state_by_sku_day[sku_idx_for_pair, :]

noise = rng.lognormal(mean=0, sigma=0.15, size=(N_STORES * N_SKUS, EXP_DAYS))
expected_demand = np.clip(
    base_demand_for_pair[:, None] * traffic_for_pair[:, None]
    * exp_weekday_mult[None, :] * exp_seasonal[None, :]
    * price_effect[:, None] * np.exp(state_by_pair_day) * noise,
    1e-6, None,
)
DISPERSION = 8.0
p_nb = DISPERSION / (DISPERSION + expected_demand)
units_sold_exp = rng.negative_binomial(DISPERSION, p_nb)

cost_for_pair = dim_products["base_cost"].to_numpy()[sku_idx_for_pair]
exp_fact = pd.DataFrame({
    "date": np.tile(exp_dates.to_numpy(), N_STORES * N_SKUS),
    "store_id": np.repeat(store_ids_arr[store_idx_for_pair], EXP_DAYS),
    "arm": np.repeat(arm_for_pair, EXP_DAYS),
    "units_sold": units_sold_exp.flatten(),
    "margin_per_unit": np.repeat(price_for_pair - cost_for_pair, EXP_DAYS),
})
exp_fact["margin"] = exp_fact["margin_per_unit"] * exp_fact["units_sold"]

print(f"\n[checkpoint 2] Simulated {EXP_DAYS} days x {N_STORES} stores x {N_SKUS} SKUs of experiment data.")

# ---- SECTION 3: pre-period covariate for CUPED ------------------------------
# Same-length window immediately before the experiment, from the real
# historical fact table - this is what each store was doing BEFORE any
# price change, used purely to explain away store-level noise.

BASE_COLS = ["date", "store_id", "sku_id", "units_sold", "price", "cost"]
base_fact = fact_sales[BASE_COLS].copy()

pre_end = pd.Timestamp("2024-12-31")
pre_start = pre_end - pd.Timedelta(days=EXP_DAYS - 1)
pre_mask = (base_fact["date"] >= pre_start) & (base_fact["date"] <= pre_end)
pre_margin = (
    base_fact.loc[pre_mask]
    .assign(margin=lambda d: (d["price"] - d["cost"]) * d["units_sold"])
    .groupby("store_id")["margin"].sum()
    .rename("pre_margin")
    .reset_index()  # explicit store_id column, avoids ambiguity merging a Series by its index
)

exp_outcome = exp_fact.groupby(["store_id", "arm"])["margin"].sum().rename("experiment_margin").reset_index()
exp_summary = exp_outcome.merge(pre_margin, on="store_id").merge(
    stores[["store_id", "store_size"]], on="store_id"
)

corr = exp_summary["pre_margin"].corr(exp_summary["experiment_margin"])
print(f"\n[checkpoint 3] Pre-period margin vs experiment margin correlation: {corr:.3f} "
      f"(higher = more informative covariate for CUPED)")

# ---- SECTION 4: CUPED adjustment --------------------------------------------
# theta = Cov(Y, X) / Var(X) - how much of Y's variation is explainable by
# the pre-period covariate. Subtracting theta*(X - mean(X)) removes exactly
# that predictable part, leaving a lower-variance residual to compare.

theta = np.cov(exp_summary["experiment_margin"], exp_summary["pre_margin"])[0, 1] / \
    np.var(exp_summary["pre_margin"], ddof=1)
exp_summary["cuped_margin"] = (
    exp_summary["experiment_margin"] - theta * (exp_summary["pre_margin"] - exp_summary["pre_margin"].mean())
)
print(f"[checkpoint 4] CUPED theta = {theta:.3f}")

# ---- SECTION 5: treatment effect + confidence interval, raw vs CUPED -------
# Welch's t-test (unequal variances, small unequal-ish samples) for the
# difference in arm means, with a proper Welch-Satterthwaite degrees of
# freedom for the CI rather than assuming a large-sample normal approx.

def diff_and_ci(df, col, alpha=0.05):
    t = df.loc[df["arm"] == "treatment", col]
    c = df.loc[df["arm"] == "control", col]
    diff = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    dof = (t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c)) ** 2 / (
        (t.var(ddof=1) / len(t)) ** 2 / (len(t) - 1) + (c.var(ddof=1) / len(c)) ** 2 / (len(c) - 1)
    )
    crit = stats.t.ppf(1 - alpha / 2, dof)
    _, pval = stats.ttest_ind(t, c, equal_var=False)
    return diff, se, (diff - crit * se, diff + crit * se), pval

raw_diff, raw_se, raw_ci, raw_p = diff_and_ci(exp_summary, "experiment_margin")
cuped_diff, cuped_se, cuped_ci, cuped_p = diff_and_ci(exp_summary, "cuped_margin")

print("\n=== Treatment effect on 8-week store margin: raw vs CUPED ===")
print(f"Raw:   lift = ${raw_diff:,.0f}  95% CI [${raw_ci[0]:,.0f}, ${raw_ci[1]:,.0f}]  "
      f"se=${raw_se:,.0f}  p={raw_p:.3f}")
print(f"CUPED: lift = ${cuped_diff:,.0f}  95% CI [${cuped_ci[0]:,.0f}, ${cuped_ci[1]:,.0f}]  "
      f"se=${cuped_se:,.0f}  p={cuped_p:.3f}")

var_reduction = 1 - (cuped_se ** 2) / (raw_se ** 2)
print(f"\nVariance reduction from CUPED: {var_reduction:.1%} "
      f"(CI width shrinks by ~{1 - cuped_se/raw_se:.1%})")

# ---- SECTION 6: guardrail metric --------------------------------------------
# A margin win that craters volume is a hidden loss (fewer transactions,
# unhappy customers, share loss you won't see for months). Check unit
# volume moved by less than a threshold before calling this a clean win.

units_by_arm = exp_fact.groupby(["store_id", "arm"])["units_sold"].sum().reset_index()
units_mean = units_by_arm.groupby("arm")["units_sold"].mean()
units_pct_change = units_mean["treatment"] / units_mean["control"] - 1

GUARDRAIL_THRESHOLD = -0.15  # flag if treatment volume drops more than 15%
print(f"\n=== Guardrail: unit volume, treatment vs control ===")
print(f"Control avg units/store: {units_mean['control']:,.0f}")
print(f"Treatment avg units/store: {units_mean['treatment']:,.0f}  ({units_pct_change:+.1%})")
if units_pct_change < GUARDRAIL_THRESHOLD:
    print(f"GUARDRAIL BREACHED: volume dropped more than {abs(GUARDRAIL_THRESHOLD):.0%} - "
          f"do not ship this pricing change on margin alone.")
else:
    print(f"Guardrail OK: volume change within the {abs(GUARDRAIL_THRESHOLD):.0%} tolerance.")

# ---- SECTION 7: sanity checks ------------------------------------------------

print("\n=== SANITY CHECKS ===")
size_balance = pd.crosstab(stores["store_size"], stores["arm"])
assert (size_balance["treatment"] - size_balance["control"]).abs().max() <= 1, \
    "Blocked randomization should balance arms within 1 store per size bucket."
print("Store-size blocking produced balanced arms (within 1 store per bucket).")
assert corr > 0, "Pre-period margin should positively correlate with experiment margin for CUPED to help."
print(f"Pre-period covariate is positively correlated with the outcome (corr={corr:.3f}) - CUPED is justified.")
assert var_reduction > 0, "CUPED should reduce variance here given a positively-correlated covariate."
print(f"CUPED reduced variance by {var_reduction:.1%} as expected.")
print("\nAll checks passed.")
