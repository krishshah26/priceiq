# PriceIQ - Week 2: baseline forecast + LightGBM + honest backtesting
#
# Goal: build the "naive vs model" comparison the whole forecasting story
# depends on, using rolling-origin backtesting so the accuracy numbers are
# actually trustworthy (see 01_generate_data.py / 02_sql_views.py for the
# data this builds on).
#
# HOW TO RUN: continue in the same Colab session as the previous two
# scripts (reuses fact_sales, dim_stores, dim_products, dim_calendar,
# ground_truth_demand already in memory). If starting fresh, it reloads
# the CSVs from /content/priceiq_data.
#
# WHAT NEVER GOES INTO THE MODEL (this is the leakage boundary):
#   - stockout_flag  -> contemporaneous with the target, computed FROM it
#   - true_demand, true_elasticity -> the hidden answer key
#   - units_sold itself, except as lagged/rolled history (the past is fair
#     game, the present/future is not)
#
# WHAT'S ALLOWED EVEN THOUGH IT'S "IN THE FUTURE" RELATIVE TO A CUTOFF:
#   - price, promo_flag, price_reason for the day being predicted. This is
#     NOT leakage: a retailer sets its own price and promo calendar in
#     advance, so "what will the price be next Tuesday" is a legitimate
#     known input, unlike "how many units did we actually sell next
#     Tuesday" which is the thing we're trying to predict.

import os
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "lightgbm"], check=True)
    import lightgbm as lgb

# ---- SECTION 1: load data if not already in memory -------------------------

_default_dir = "/content/priceiq_data" if os.path.isdir("/content") else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "priceiq_data"
)
OUT_DIR = os.environ.get("PRICEIQ_DATA_DIR", _default_dir)
if "fact_sales" not in dir():
    print("Loading CSVs from disk (fresh session detected)...")
    dim_calendar = pd.read_csv(f"{OUT_DIR}/dim_calendar.csv", parse_dates=["date", "week_start"])
    dim_stores = pd.read_csv(f"{OUT_DIR}/dim_stores.csv")
    dim_products = pd.read_csv(f"{OUT_DIR}/dim_products.csv")
    fact_sales = pd.read_csv(f"{OUT_DIR}/fact_sales.csv", parse_dates=["date"])
    ground_truth_demand = pd.read_csv(f"{OUT_DIR}/ground_truth_demand.csv", parse_dates=["date"])
else:
    print("Reusing dataframes already in memory.")
    fact_sales = fact_sales.copy()  # don't mutate the original in place

# ---- SECTION 2: feature engineering ----------------------------------------
# Lag/rolling features only ever look BACKWARD from a row's own date, so
# they're safe to compute once, globally, before any train/test split -
# a row's "sales 7 days ago" doesn't change depending on which fold you're
# scoring. What changes per fold is which rows count as "train" vs "test".

fact_sales = fact_sales.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)

g = fact_sales.groupby(["store_id", "sku_id"])["units_sold"]
fact_sales["lag_7"] = g.shift(7)    # "same day last week" - also our baseline
fact_sales["lag_14"] = g.shift(14)
fact_sales["lag_28"] = g.shift(28)
# shift(1) first so the rolling window never includes the day being predicted
fact_sales["roll_mean_28"] = (
    fact_sales.groupby(["store_id", "sku_id"])["units_sold"]
    .transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())
)

fact_sales = fact_sales.merge(
    dim_calendar[["date", "day_of_week", "is_weekend", "month", "seasonal_index"]],
    on="date", how="left",
)
fact_sales = fact_sales.merge(dim_products[["sku_id", "category", "brand_tier"]], on="sku_id", how="left")
fact_sales = fact_sales.merge(dim_stores[["store_id", "store_size", "region"]], on="store_id", how="left")

CATEGORICAL_FEATURES = ["price_reason", "day_of_week", "store_size", "region",
                         "category", "brand_tier", "store_id", "sku_id"]
for c in CATEGORICAL_FEATURES:
    fact_sales[c] = fact_sales[c].astype("category")

NUMERIC_FEATURES = ["price", "cost", "promo_flag", "is_weekend", "seasonal_index",
                     "lag_7", "lag_14", "lag_28", "roll_mean_28"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "units_sold"

FORBIDDEN = {"stockout_flag", "true_demand", "true_elasticity", "units_sold"}
assert not (set(FEATURES) & FORBIDDEN), "Leakage: a forbidden column is in FEATURES!"
print(f"[checkpoint 1] {len(FEATURES)} features ready, none of them from the "
      f"forbidden set {FORBIDDEN}.")

# ---- SECTION 3: metrics -----------------------------------------------------
# WMAPE: on average, how far off are we (either direction), weighted so
# big sellers count more. Bias: are we systematically too high or too low -
# this is the one that hides in a good-looking WMAPE and costs real money
# in inventory decisions.

def wmape(actual, pred):
    return np.abs(actual - pred).sum() / actual.sum()

def bias(actual, pred):
    return (pred - actual).sum() / actual.sum()

# ---- SECTION 4: rolling-origin backtest folds -------------------------------
# Never shuffle time-series data into train/test randomly - that lets the
# model see the future while "training" on the past, giving a fake accuracy
# number. Instead: pick a cutoff, train on everything before it, predict
# forward, slide the cutoff, repeat.

FIRST_CUTOFF_IDX = 150   # enough history for lag_28 / roll_mean_28 to be populated
CUTOFF_STRIDE_DAYS = 90
HORIZON_DAYS = 28

all_dates = sorted(fact_sales["date"].unique())
cutoff_indices = list(range(FIRST_CUTOFF_IDX, len(all_dates) - HORIZON_DAYS, CUTOFF_STRIDE_DAYS))
cutoff_dates = [pd.Timestamp(all_dates[i]) for i in cutoff_indices]

print(f"\n[checkpoint 2] {len(cutoff_dates)} backtest folds:")
for cd in cutoff_dates:
    print(f"  train up to {cd.date()}  ->  test {(cd + pd.Timedelta(days=1)).date()} "
          f"to {(cd + pd.Timedelta(days=HORIZON_DAYS)).date()}")

# ---- SECTION 5: train + score each fold -------------------------------------

fold_results = []
row_predictions = []

for fold_i, cutoff in enumerate(cutoff_dates):
    train_mask = fact_sales["date"] <= cutoff
    test_mask = (fact_sales["date"] > cutoff) & (fact_sales["date"] <= cutoff + pd.Timedelta(days=HORIZON_DAYS))

    train_df = fact_sales.loc[train_mask]
    test_df = fact_sales.loc[test_mask]

    # leakage guard: assert no test-period row leaked into training
    assert train_df["date"].max() <= cutoff < test_df["date"].min(), \
        f"Fold {fold_i}: train/test date overlap detected!"

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    model = lgb.LGBMRegressor(
        objective="tweedie",          # count data: lots of small values, occasional big spikes
        tweedie_variance_power=1.1,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)

    pred_lgb = np.clip(model.predict(X_test), 0, None)
    pred_baseline = test_df["lag_7"].to_numpy()  # seasonal-naive: same weekday last week

    wmape_base, bias_base = wmape(y_test.values, pred_baseline), bias(y_test.values, pred_baseline)
    wmape_lgb, bias_lgb = wmape(y_test.values, pred_lgb), bias(y_test.values, pred_lgb)

    fold_results.append({
        "fold": fold_i, "cutoff": cutoff.date(), "n_test_rows": len(test_df),
        "wmape_baseline": wmape_base, "bias_baseline": bias_base,
        "wmape_lgb": wmape_lgb, "bias_lgb": bias_lgb,
    })

    fold_preds = test_df[["date", "store_id", "sku_id", "units_sold"]].copy()
    fold_preds["pred_baseline"] = pred_baseline
    fold_preds["pred_lgb"] = pred_lgb
    fold_preds["fold"] = fold_i
    row_predictions.append(fold_preds)

    print(f"[fold {fold_i}] cutoff={cutoff.date()} n_test={len(test_df):,} "
          f"| WMAPE base={wmape_base:.3f} lgb={wmape_lgb:.3f} "
          f"| bias base={bias_base:+.3f} lgb={bias_lgb:+.3f}")

# ---- SECTION 6: aggregate results -------------------------------------------

fold_results_df = pd.DataFrame(fold_results)
print("\n=== Fold-by-fold results ===")
print(fold_results_df.round(3).to_string(index=False))

all_preds = pd.concat(row_predictions, ignore_index=True)
all_preds["month"] = all_preds["date"].dt.month

print("\n=== Overall (pooled across all fold test windows) ===")
print(f"Baseline  - WMAPE: {wmape(all_preds['units_sold'], all_preds['pred_baseline']):.3f}  "
      f"Bias: {bias(all_preds['units_sold'], all_preds['pred_baseline']):+.3f}")
print(f"LightGBM  - WMAPE: {wmape(all_preds['units_sold'], all_preds['pred_lgb']):.3f}  "
      f"Bias: {bias(all_preds['units_sold'], all_preds['pred_lgb']):+.3f}")

monthly = all_preds.groupby("month")[["units_sold", "pred_baseline", "pred_lgb"]].apply(lambda d: pd.Series({
    "n": len(d),
    "wmape_baseline": wmape(d["units_sold"], d["pred_baseline"]),
    "bias_baseline": bias(d["units_sold"], d["pred_baseline"]),
    "wmape_lgb": wmape(d["units_sold"], d["pred_lgb"]),
    "bias_lgb": bias(d["units_sold"], d["pred_lgb"]),
}))
print("\n=== Monthly breakdown (watch bias in Nov/Dec - stockout censoring at work) ===")
print(monthly.round(3).to_string())

# ---- SECTION 7: bonus - bias against TRUE demand ----------------------------
# units_sold is censored by stockouts; true_demand is the number a real
# analyst would never have. Comparing against it shows exactly how much
# stockout censoring corrupts what "good accuracy" even means in December.

all_preds_gt = all_preds.merge(ground_truth_demand, on=["date", "store_id", "sku_id"], how="left")
dec = all_preds_gt[all_preds_gt["month"] == 12]

print("\n=== Bonus: LightGBM bias against the hidden true_demand answer key ===")
print(f"Bias vs units_sold (observed), all months: "
      f"{bias(all_preds_gt['units_sold'], all_preds_gt['pred_lgb']):+.3f}")
print(f"Bias vs true_demand (hidden),  all months: "
      f"{bias(all_preds_gt['true_demand'], all_preds_gt['pred_lgb']):+.3f}")
print(f"Bias vs true_demand (hidden),  December only: "
      f"{bias(dec['true_demand'], dec['pred_lgb']):+.3f}")

print("\nAll checks passed.")
