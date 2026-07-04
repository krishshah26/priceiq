# PriceIQ - Synthetic Data Generator (Meridian Goods)
#
# Week 1 goal: build the star schema everything downstream (forecasting,
# elasticity, optimization, experiment) will run on.
#
# HOW TO RUN THIS IN COLAB:
#   Simplest: paste this entire file into one Colab code cell and run it.
#   It executes top to bottom and prints a "teaching checkpoint" after each
#   major section so you can see what just got built and why.
#
#   If you'd rather build it piece by piece (recommended for learning),
#   split at the "# ---- SECTION" comment lines into separate cells and
#   run them in order - each section depends only on the ones above it.
#
# WHAT THIS PRODUCES:
#   dim_calendar, dim_stores, dim_products  -> the dimension tables
#   fact_sales                              -> one row per (store, sku, day):
#                                              what a real analyst would see
#   ground_truth_demand                     -> the HIDDEN answer key (true,
#                                              uncensored demand) - never feed
#                                              this to a model, only use it
#                                              later to grade ourselves
#
# WHY IT'S DELIBERATELY "BROKEN" IN REALISTIC WAYS:
#   1. Stockouts: units_sold sometimes undercounts true demand because the
#      shelf ran empty. (This is Post #1's story.)
#   2. Three distinct reasons price moves: cost-driven (clean), managerial
#      markdown (confounded with a demand slump), promo (a separate lever).
#      Naive elasticity estimation gets fooled by #2 - that's Week 3's story.

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixed seed -> reproducible data

# ---- SECTION 1: config ----------------------------------------------------

N_STORES = 20
N_SKUS = 100
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"  # 2 full years -> yearly seasonality shows up twice

dates = pd.date_range(START_DATE, END_DATE, freq="D")
N_DAYS = len(dates)
print(f"[checkpoint 1] {N_STORES} stores x {N_SKUS} SKUs x {N_DAYS} days "
      f"= {N_STORES * N_SKUS * N_DAYS:,} fact rows")

# ---- SECTION 2: dim_calendar -----------------------------------------------
# A fact table holds one row per event (store, sku, day). A dimension table
# holds descriptive attributes that don't change every day, referenced by id,
# so we don't repeat "Store 004, Northeast, Large" 730 times per SKU.

dim_calendar = pd.DataFrame({"date": dates})
dim_calendar["day_of_week"] = dim_calendar["date"].dt.day_name()
dim_calendar["is_weekend"] = dim_calendar["date"].dt.dayofweek >= 5
dim_calendar["month"] = dim_calendar["date"].dt.month
dim_calendar["year"] = dim_calendar["date"].dt.year
dim_calendar["week_start"] = dim_calendar["date"] - pd.to_timedelta(
    dim_calendar["date"].dt.dayofweek, unit="D"
)

# Retail seasonality by month: Nov/Dec holiday peak, Jan/Feb post-holiday slump.
MONTH_SEASONALITY = {1: 0.85, 2: 0.85, 3: 0.95, 4: 0.95, 5: 1.00, 6: 1.00,
                      7: 1.00, 8: 1.05, 9: 1.00, 10: 1.05, 11: 1.25, 12: 1.35}
dim_calendar["seasonal_index"] = dim_calendar["month"].map(MONTH_SEASONALITY)
dim_calendar["weekday_mult"] = np.where(dim_calendar["is_weekend"], 1.15, 1.0)

print("[checkpoint 2] dim_calendar built:")
print(dim_calendar.head(3))

# ---- SECTION 3: dim_stores -------------------------------------------------
# Not every store sells the same volume - a Large store just has more foot
# traffic than a Small one. Each store also gets its own idiosyncratic
# "how much better/worse than typical for its size" multiplier.

REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
SIZE_TRAFFIC_MULT = {"Small": 0.7, "Medium": 1.0, "Large": 1.5}

store_size = rng.choice(list(SIZE_TRAFFIC_MULT), size=N_STORES, p=[0.30, 0.45, 0.25])
store_noise = rng.lognormal(mean=0, sigma=0.15, size=N_STORES)

dim_stores = pd.DataFrame({
    "store_id": np.arange(1, N_STORES + 1),
    "store_name": [f"Store {i:03d}" for i in range(1, N_STORES + 1)],
    "region": rng.choice(REGIONS, size=N_STORES),
    "store_size": store_size,
})
dim_stores["traffic_multiplier"] = dim_stores["store_size"].map(SIZE_TRAFFIC_MULT) * store_noise

print("[checkpoint 3] dim_stores built:")
print(dim_stores.head(3))

# ---- SECTION 4: dim_products -----------------------------------------------
# Every SKU gets a brand tier (Good/Better/Best), which sets its markup and,
# crucially, its TRUE price elasticity - how sharply demand falls when price
# rises. Cheap "Good"-tier goods are more price-sensitive (elasticity ~ -2.4)
# than premium "Best"-tier goods (~ -0.8) - shoppers price-shop commodities
# but are stickier on things they consider premium.
#
# true_elasticity is our SECRET ANSWER KEY. In Week 3 we'll try to recover
# this number using only price and sales history, the way a real analyst
# would - never seeing this column. We keep it here only so we can grade
# our own estimate later.

CATEGORIES = ["Beverages", "Snacks", "Household", "Personal Care", "Dairy",
              "Frozen", "Bakery", "Produce"]
TIER_PARAMS = {
    "Good":   {"markup": (1.25, 1.45), "elasticity_mean": -2.4, "elasticity_sd": 0.4},
    "Better": {"markup": (1.45, 1.70), "elasticity_mean": -1.5, "elasticity_sd": 0.3},
    "Best":   {"markup": (1.70, 2.10), "elasticity_mean": -0.8, "elasticity_sd": 0.2},
}

sku_tier = rng.choice(list(TIER_PARAMS), size=N_SKUS, p=[0.40, 0.40, 0.20])
base_cost = np.clip(rng.lognormal(mean=1.1, sigma=0.6, size=N_SKUS), 0.5, 25.0).round(2)
markup = np.array([rng.uniform(*TIER_PARAMS[t]["markup"]) for t in sku_tier])
base_price = (base_cost * markup).round(2)
true_elasticity = np.clip(
    np.array([rng.normal(TIER_PARAMS[t]["elasticity_mean"], TIER_PARAMS[t]["elasticity_sd"])
              for t in sku_tier]),
    -4.0, -0.2,
)
# Long right tail on purpose: some SKUs end up naturally slow movers
# (< 1 unit/store/day). See roadmap 4.1 "intermittent / slow-mover handling"
# for how we'd treat these separately in v2 - for now they just live in the
# blend, same as a first-pass real project would.
base_daily_demand = rng.lognormal(mean=0.3, sigma=1.1, size=N_SKUS)

dim_products = pd.DataFrame({
    "sku_id": np.arange(1, N_SKUS + 1),
    "sku_name": [f"SKU {i:04d}" for i in range(1, N_SKUS + 1)],
    "category": rng.choice(CATEGORIES, size=N_SKUS),
    "brand_tier": sku_tier,
    "base_cost": base_cost,
    "base_price": base_price,
    "true_elasticity": true_elasticity,
    "base_daily_demand": base_daily_demand.round(3),
})

print("[checkpoint 4] dim_products built. Elasticity by tier (sanity check - "
      "Good should be most negative, Best least):")
print(dim_products.groupby("brand_tier")["true_elasticity"].mean().round(2))

# ---- SECTION 5: weekly price/cost simulation per SKU -----------------------
# THE key design choice in this whole dataset. Every week, each SKU's price
# moves for exactly one of three reasons:
#
#   1. cost_driven        - the supplier's cost shifted, price follows it.
#                            Has nothing to do with how well the item is
#                            currently selling -> "clean" variation.
#   2. managerial_markdown - a manager marks down an item that's in a weak
#                            patch. The catch: the "weak patch" is the SAME
#                            latent shock that is ALSO suppressing that
#                            week's sales. Price and demand move together
#                            for a reason that has nothing to do with
#                            elasticity -> this is confounding/endogeneity.
#   3. promo               - a scheduled, calendar-driven discount,
#                            independent of how the item happens to be
#                            selling, plus an extra demand "buzz" beyond
#                            the pure price effect (feature ad, end-cap).
#
# If you naively regress log(units_sold) on log(price) using ALL the data,
# mechanism #2 biases the elasticity estimate toward zero (or the wrong
# sign) - markdowns coincide with weak-demand periods for a reason that has
# nothing to do with price sensitivity. Week 3's fix: isolate price moves
# attributable to mechanism #1 and estimate elasticity from those alone.
# That's what "using cost as an instrument" means in plain English.

weeks = dim_calendar["week_start"].drop_duplicates().sort_values().reset_index(drop=True)
N_WEEKS = len(weeks)

price_rows = []
for s_idx, sku_id in enumerate(dim_products["sku_id"]):
    base_cost_s = dim_products.loc[s_idx, "base_cost"]
    base_price_s = dim_products.loc[s_idx, "base_price"]
    markup_s = base_price_s / base_cost_s

    cost = base_cost_s
    demand_state = 0.0  # latent "how is this SKU trending" shock, AR(1)

    n_promos = rng.integers(3, 6)
    promo_weeks = set(rng.choice(N_WEEKS, size=n_promos, replace=False))

    for w in range(N_WEEKS):
        cost *= 1 + rng.normal(0, 0.004)  # small weekly drift
        if rng.random() < 0.04:  # occasional supplier cost shock
            cost *= 1 + rng.choice([-1, 1]) * rng.uniform(0.05, 0.12)
        cost = min(max(cost, base_cost_s * 0.6), base_cost_s * 1.8)  # keep sane

        demand_state = 0.85 * demand_state + rng.normal(0, 0.30)  # mean-reverting

        if w in promo_weeks:
            price, reason, promo_flag = round(base_price_s * 0.75, 2), "promo", 1
        elif demand_state < -0.55 and rng.random() < 0.6:
            price, reason, promo_flag = round(cost * markup_s * 0.85, 2), "managerial_markdown", 0
        else:
            price, reason, promo_flag = round(cost * markup_s, 2), "cost_driven", 0

        price_rows.append((sku_id, weeks[w], round(cost, 2), price, reason, promo_flag, demand_state))

sku_week_prices = pd.DataFrame(
    price_rows,
    columns=["sku_id", "week_start", "cost", "price", "price_reason", "promo_flag", "demand_state"],
)

print("[checkpoint 5] Price-change mix across all SKU-weeks (cost_driven should "
      "dominate, the other two should be meaningful minorities):")
print(sku_week_prices["price_reason"].value_counts(normalize=True).round(3))

# ---- SECTION 6: expand to daily, compute true demand -----------------------
# true_demand = base_daily_demand
#             x store_traffic_multiplier
#             x weekday_multiplier          (weekends sell more)
#             x month_seasonal_index        (Nov/Dec > Jan/Feb)
#             x (price / base_price) ^ true_elasticity   <- actual price response
#             x promo_bonus_lift            (only on promo weeks)
#             x demand_state_multiplier     (same latent shock as markdowns)
#             x random noise
#
# We then draw the OBSERVED unit count from a Negative Binomial around that
# mean - real sales are "over-dispersed" (spikier than a plain Poisson would
# produce; a promo day or a lucky Saturday can blow way past average).

N_PAIRS = N_STORES * N_SKUS
store_idx_for_pair = np.repeat(np.arange(N_STORES), N_SKUS)
sku_idx_for_pair = np.tile(np.arange(N_SKUS), N_STORES)

week_index = {w: i for i, w in enumerate(weeks)}
sku_index = {sid: i for i, sid in enumerate(dim_products["sku_id"])}

price_mat = np.zeros((N_SKUS, N_WEEKS))
cost_mat = np.zeros((N_SKUS, N_WEEKS))
reason_mat = np.empty((N_SKUS, N_WEEKS), dtype=object)
promo_mat = np.zeros((N_SKUS, N_WEEKS), dtype=int)
state_mat = np.zeros((N_SKUS, N_WEEKS))

for row in sku_week_prices.itertuples(index=False):
    si, wi = sku_index[row.sku_id], week_index[row.week_start]
    price_mat[si, wi] = row.price
    cost_mat[si, wi] = row.cost
    reason_mat[si, wi] = row.price_reason
    promo_mat[si, wi] = row.promo_flag
    state_mat[si, wi] = row.demand_state

day_week_idx = dim_calendar["week_start"].map(week_index).to_numpy().astype(int)

price_by_sku_day = price_mat[:, day_week_idx]
cost_by_sku_day = cost_mat[:, day_week_idx]
promo_by_sku_day = promo_mat[:, day_week_idx]
state_by_sku_day = state_mat[:, day_week_idx]
reason_by_sku_day = reason_mat[:, day_week_idx]

price_by_pair_day = price_by_sku_day[sku_idx_for_pair, :]
cost_by_pair_day = cost_by_sku_day[sku_idx_for_pair, :]
promo_by_pair_day = promo_by_sku_day[sku_idx_for_pair, :]
state_by_pair_day = state_by_sku_day[sku_idx_for_pair, :]
reason_by_pair_day = reason_by_sku_day[sku_idx_for_pair, :]

base_price_for_pair = dim_products["base_price"].to_numpy()[sku_idx_for_pair]
elasticity_for_pair = dim_products["true_elasticity"].to_numpy()[sku_idx_for_pair]
base_demand_for_pair = dim_products["base_daily_demand"].to_numpy()[sku_idx_for_pair]
traffic_for_pair = dim_stores["traffic_multiplier"].to_numpy()[store_idx_for_pair]

month_seasonal_by_day = dim_calendar["seasonal_index"].to_numpy()
weekday_mult_by_day = dim_calendar["weekday_mult"].to_numpy()

price_effect = (price_by_pair_day / base_price_for_pair[:, None]) ** elasticity_for_pair[:, None]
promo_bonus = np.where(promo_by_pair_day == 1, 1.25, 1.0)
demand_state_mult = np.exp(state_by_pair_day)
noise = rng.lognormal(mean=0, sigma=0.15, size=(N_PAIRS, N_DAYS))

expected_demand = np.clip(
    base_demand_for_pair[:, None]
    * traffic_for_pair[:, None]
    * weekday_mult_by_day[None, :]
    * month_seasonal_by_day[None, :]
    * price_effect
    * promo_bonus
    * demand_state_mult
    * noise,
    1e-6, None,
)

DISPERSION = 8.0  # lower = spikier/more over-dispersed than Poisson
p_nb = DISPERSION / (DISPERSION + expected_demand)
true_demand_units = rng.negative_binomial(DISPERSION, p_nb)

print(f"[checkpoint 6] true_demand_units matrix shape: {true_demand_units.shape}, "
      f"mean daily demand per store-sku: {true_demand_units.mean():.2f}")

# ---- SECTION 7: stockout / inventory simulation ----------------------------
# Stores don't have infinite shelf stock. Every (store, sku) gets an
# inventory level, restocked weekly to a "par level" sized for TYPICAL
# demand. When a spike (a promo, a holiday) exceeds what's on the shelf,
# units_sold gets capped at whatever was available - that's stockout_flag=1.
# True demand that day was higher than what we observe. This censoring is
# exactly why "just forecast off units_sold" quietly teaches a model that
# demand caps out lower than it really does, especially in Nov/Dec when
# seasonal demand outruns a par level sized for an average week.

par_level = np.ceil(12 * base_demand_for_pair * traffic_for_pair).astype(int)
RESTOCK_CYCLE = 7
restock_offset = rng.integers(0, RESTOCK_CYCLE, size=N_PAIRS)

inventory = par_level.copy()
units_sold = np.zeros((N_PAIRS, N_DAYS), dtype=int)
stockout_flag = np.zeros((N_PAIRS, N_DAYS), dtype=bool)

for d in range(N_DAYS):
    restock_today = (d % RESTOCK_CYCLE) == restock_offset
    inventory = np.where(restock_today, par_level, inventory)

    demand_today = true_demand_units[:, d]
    sold_today = np.minimum(demand_today, inventory)
    units_sold[:, d] = sold_today
    stockout_flag[:, d] = demand_today > inventory
    inventory = inventory - sold_today

print(f"[checkpoint 7] Overall stockout rate: {stockout_flag.mean():.1%}")

# ---- SECTION 8: flatten to long-format fact tables -------------------------

store_id_for_pair = dim_stores["store_id"].to_numpy()[store_idx_for_pair]
sku_id_for_pair = dim_products["sku_id"].to_numpy()[sku_idx_for_pair]
date_arr = dim_calendar["date"].to_numpy()

fact_sales = pd.DataFrame({
    "date": np.tile(date_arr, N_PAIRS),
    "store_id": np.repeat(store_id_for_pair, N_DAYS),
    "sku_id": np.repeat(sku_id_for_pair, N_DAYS),
    "units_sold": units_sold.flatten(),
    "price": price_by_pair_day.flatten(),
    "cost": cost_by_pair_day.flatten(),
    "promo_flag": promo_by_pair_day.flatten(),
    "price_reason": reason_by_pair_day.flatten(),
    "stockout_flag": stockout_flag.flatten(),
})

ground_truth_demand = pd.DataFrame({
    "date": np.tile(date_arr, N_PAIRS),
    "store_id": np.repeat(store_id_for_pair, N_DAYS),
    "sku_id": np.repeat(sku_id_for_pair, N_DAYS),
    "true_demand": true_demand_units.flatten(),
})

print(f"[checkpoint 8] fact_sales: {fact_sales.shape}, "
      f"ground_truth_demand: {ground_truth_demand.shape}")

# ---- SECTION 9: sanity checks -----------------------------------------------
# Always verify a generator did what you think it did before trusting it.

print("\n=== SANITY CHECKS ===")
print("fact_sales shape:", fact_sales.shape)
print("\nStockout rate by month (should spike in Nov/Dec):")
print(fact_sales.assign(month=fact_sales["date"].dt.month)
      .groupby("month")["stockout_flag"].mean().round(3))
print("\nPrice reason mix:")
print(fact_sales["price_reason"].value_counts(normalize=True).round(3))
print("\nSample rows:")
print(fact_sales.head(10))

# ---- SECTION 10: save -------------------------------------------------------

# Works both in Colab (defaults to /content/priceiq_data if that path exists)
# and locally (defaults to <project_root>/data/priceiq_data). Override with
# the PRICEIQ_DATA_DIR env var if you want it somewhere else.
_default_dir = "/content/priceiq_data" if os.path.isdir("/content") else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "priceiq_data"
)
OUT_DIR = os.environ.get("PRICEIQ_DATA_DIR", _default_dir)
os.makedirs(OUT_DIR, exist_ok=True)

dim_calendar.to_csv(f"{OUT_DIR}/dim_calendar.csv", index=False)
dim_stores.to_csv(f"{OUT_DIR}/dim_stores.csv", index=False)
dim_products.to_csv(f"{OUT_DIR}/dim_products.csv", index=False)
fact_sales.to_csv(f"{OUT_DIR}/fact_sales.csv", index=False)
ground_truth_demand.to_csv(f"{OUT_DIR}/ground_truth_demand.csv", index=False)

print(f"\n[checkpoint 10] Saved to {OUT_DIR}:")
for f in sorted(os.listdir(OUT_DIR)):
    size_mb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1e6
    print(f"  - {f} ({size_mb:.1f} MB)")

# To get files out of Colab's ephemeral disk, either:
#   from google.colab import files; files.download(f"{OUT_DIR}/fact_sales.csv")
# or mount Drive and copy OUT_DIR there:
#   from google.colab import drive; drive.mount('/content/drive')
