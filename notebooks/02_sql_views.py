# PriceIQ - Week 1 SQL views (Meridian Goods)
#
# Goal: put a real SQL layer between the raw fact/dim tables and anyone
# who wants to analyze them, instead of everyone writing their own pandas
# joins. This is the beginner version of what dbt formalizes later
# (roadmap 4.4) - for now, just clean SQL views.
#
# HOW TO RUN: if you're continuing in the SAME Colab session as
# 01_generate_data.py, the dataframes (fact_sales, dim_stores, etc.) are
# already in memory and this will just work. If you're in a fresh
# session, it loads the CSVs from OUT_DIR instead - run that cell first.
#
# We use DuckDB because it can run real SQL directly against pandas
# dataframes with zero setup (no server, no schema migration) - perfect
# for "I want SQL, not a database project."

import os
try:
    import duckdb
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "duckdb"], check=True)
    import duckdb

import pandas as pd

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
    print("Reusing dataframes already in memory from 01_generate_data.py.")

con = duckdb.connect()
# Register explicitly rather than relying on DuckDB's auto-detection of
# same-named Python variables - guarantees this works the same way
# regardless of DuckDB version.
con.register("dim_calendar", dim_calendar)
con.register("dim_stores", dim_stores)
con.register("dim_products", dim_products)
con.register("fact_sales", fact_sales)
con.register("ground_truth_demand", ground_truth_demand)
print("[checkpoint 1] DuckDB connection open. Tables registered: "
      "dim_calendar, dim_stores, dim_products, fact_sales, ground_truth_demand")

# ---- SECTION 2: v_daily_sales -----------------------------------------------
# The single most useful view in the project: one row per (store, sku, day)
# with everything joined and the derived metrics (revenue, margin) computed
# ONCE, here, instead of every downstream notebook recomputing them slightly
# differently. This is the whole point of a "mart" - raw facts stay raw,
# derived business metrics live in one place.

con.execute("""
    CREATE OR REPLACE VIEW v_daily_sales AS
    SELECT
        f.date,
        f.store_id,
        s.store_name,
        s.region,
        s.store_size,
        f.sku_id,
        p.sku_name,
        p.category,
        p.brand_tier,
        f.units_sold,
        f.price,
        f.cost,
        f.units_sold * f.price               AS revenue,
        f.units_sold * (f.price - f.cost)    AS margin,
        f.promo_flag,
        f.price_reason,
        f.stockout_flag,
        c.day_of_week,
        c.is_weekend,
        c.month,
        c.year,
        c.seasonal_index
    FROM fact_sales f
    JOIN dim_stores   s ON f.store_id = s.store_id
    JOIN dim_products p ON f.sku_id   = p.sku_id
    JOIN dim_calendar c ON f.date     = c.date
""")

print("[checkpoint 2] v_daily_sales created. Preview:")
print(con.execute("SELECT * FROM v_daily_sales LIMIT 5").df())

# ---- SECTION 3: v_monthly_category_performance ------------------------------
# A simple rollup: the kind of query a merchandising analyst actually runs -
# "how is each category trending by month, and how much of our potential
# revenue are we losing to stockouts?"

con.execute("""
    CREATE OR REPLACE VIEW v_monthly_category_performance AS
    SELECT
        year,
        month,
        category,
        brand_tier,
        SUM(units_sold)                              AS total_units,
        SUM(revenue)                                 AS total_revenue,
        SUM(margin)                                  AS total_margin,
        AVG(stockout_flag::INT)                      AS stockout_rate,
        SUM(promo_flag)                              AS promo_days
    FROM v_daily_sales
    GROUP BY year, month, category, brand_tier
    ORDER BY year, month, category, brand_tier
""")

print("\n[checkpoint 3] v_monthly_category_performance created. Preview:")
print(con.execute("SELECT * FROM v_monthly_category_performance LIMIT 5").df())

# ---- SECTION 4: v_stockout_lost_revenue -------------------------------------
# The money chart for Post 1. We join in ground_truth_demand (the number a
# real analyst would NEVER have - here it's our answer key) purely to
# quantify: on days we stocked out, how much revenue did we actually lose
# versus what we would have made if the shelf had been full?
#
# This view only exists for OUR validation/storytelling - it should never
# feed a model, since in a real company you would not have true_demand.

con.execute("""
    CREATE OR REPLACE VIEW v_stockout_lost_revenue AS
    SELECT
        d.date,
        d.store_id,
        d.sku_id,
        g.true_demand,
        d.units_sold,
        g.true_demand - d.units_sold                        AS units_lost,
        (g.true_demand - d.units_sold) * d.price             AS revenue_lost,
        d.stockout_flag
    FROM v_daily_sales d
    JOIN ground_truth_demand g
        ON d.date = g.date AND d.store_id = g.store_id AND d.sku_id = g.sku_id
    WHERE d.stockout_flag = true
""")

lost_summary = con.execute("""
    SELECT
        COUNT(*)                    AS stockout_days,
        SUM(units_lost)             AS total_units_lost,
        SUM(revenue_lost)           AS total_revenue_lost
    FROM v_stockout_lost_revenue
""").df()

print("\n[checkpoint 4] v_stockout_lost_revenue created. Summary:")
print(lost_summary)

total_revenue = con.execute("SELECT SUM(revenue) AS r FROM v_daily_sales").df()["r"][0]
lost = lost_summary["total_revenue_lost"][0]
print(f"\nRevenue actually recorded: ${total_revenue:,.0f}")
print(f"Revenue left on the table by stockouts: ${lost:,.0f} "
      f"({lost / (total_revenue + lost):.1%} of true potential revenue)")

# ---- SECTION 5: sanity checks ------------------------------------------------

print("\n=== SANITY CHECKS ===")
row_check = con.execute("SELECT COUNT(*) AS n FROM v_daily_sales").df()["n"][0]
print(f"v_daily_sales row count: {row_check:,} (should equal fact_sales row count: {len(fact_sales):,})")
assert row_check == len(fact_sales), "Row count mismatch - a join is dropping or duplicating rows!"

print("\nMonthly stockout rate from the view (should match what 01_generate_data.py printed):")
print(con.execute("""
    SELECT month, ROUND(AVG(stockout_flag::INT), 3) AS stockout_rate
    FROM v_daily_sales GROUP BY month ORDER BY month
""").df())

print("\nAll checks passed.")
