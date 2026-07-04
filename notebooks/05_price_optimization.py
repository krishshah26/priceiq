# PriceIQ - Week 4: price optimization, the margin bridge, and the efficient frontier
#
# Goal: turn elasticity estimates into actual price recommendations, and be
# honest about how much of the margin gain is real vs. an artifact of
# elasticity estimation error.
#
# REQUIRES: `results` and `weekly` from 04_elasticity_estimation.py in the
# same Colab session (per-SKU naive/clean/shrunk elasticity estimates and
# the SKU-week sales panel). This script doesn't reload from CSV because
# recomputing the elasticity regressions here would duplicate 04's logic -
# run 04 first if you're in a fresh session.
#
# WHAT WE DELIBERATELY DON'T BUILD: cross-SKU constraints (e.g. "Good tier
# must stay cheaper than Better tier"). Our schema has no product-family
# grouping linking SKUs together, so that constraint has nothing to attach
# to here - see roadmap 4.2 "cross-SKU / joint optimization" for the v2 version.

import numpy as np
import pandas as pd

if "results" not in dir() or "weekly" not in dir():
    raise RuntimeError(
        "Missing `results`/`weekly` from 04_elasticity_estimation.py. "
        "Run that script first in this same Colab session."
    )

# ---- SECTION 1: current state per SKU ---------------------------------------
# Anchor "current price" and "current demand" using only cost_driven weeks -
# using promo or markdown weeks here would bake a temporary distortion into
# what's supposed to be the normal baseline we're optimizing away from.

current_state = (
    weekly[weekly["price_reason"] == "cost_driven"]
    .groupby("sku_id")
    .agg(current_price=("price", "mean"), current_demand=("total_units", "mean"))
    .reset_index()
)

state = current_state.merge(
    results[["sku_id", "brand_tier", "true_elasticity", "shrunk_elasticity"]], on="sku_id"
).merge(
    dim_products[["sku_id", "base_cost"]], on="sku_id"
)
state["current_margin"] = (state["current_price"] - state["base_cost"]) * state["current_demand"]

print(f"[checkpoint 1] Current state built for {len(state)} SKUs. "
      f"Current total weekly margin (all SKUs, national): ${state['current_margin'].sum():,.0f}")

# ---- SECTION 2: the optimizer ------------------------------------------------
# Grid search over a bounded price range: constant-elasticity demand curve,
# margin-floor guardrail (never optimize into a near-cost or loss-making
# price), and a max-move guardrail (no 300% overnight repricing). The
# current price is always included as a candidate, so optimization can
# never recommend something worse than doing nothing.

def optimize_sku(row, elasticity_col, max_down=0.20, max_up=0.30, margin_floor_mult=1.05):
    p0, q0, cost = row["current_price"], row["current_demand"], row["base_cost"]
    e = row[elasticity_col]
    lo = max(p0 * (1 - max_down), cost * margin_floor_mult)
    hi = p0 * (1 + max_up)
    if hi <= lo:
        return p0, q0, (p0 - cost) * q0
    candidates = np.append(np.linspace(lo, hi, 101), p0)
    demand = q0 * (candidates / p0) ** e
    margin = (candidates - cost) * demand
    best = np.argmax(margin)
    return candidates[best], demand[best], margin[best]

opt_rows = []
for _, row in state.iterrows():
    p_opt, q_opt_believed, m_opt_believed = optimize_sku(row, "shrunk_elasticity")
    p_oracle, q_oracle, m_oracle = optimize_sku(row, "true_elasticity")

    # m_opt_believed is what the optimizer THINKS it gets, using its own
    # (possibly wrong) elasticity to grade its own recommendation - that's
    # not trustworthy. What actually happens if we charge p_opt is governed
    # by the TRUE elasticity, which we only get to check because we secretly
    # have the answer key.
    q_opt_real = row["current_demand"] * (p_opt / row["current_price"]) ** row["true_elasticity"]
    m_opt_real = (p_opt - row["base_cost"]) * q_opt_real

    opt_rows.append({
        "sku_id": row["sku_id"], "brand_tier": row["brand_tier"],
        "current_price": row["current_price"], "current_margin": row["current_margin"],
        "opt_price": round(p_opt, 2),
        "opt_margin_believed": m_opt_believed,  # what the model thinks it gets - don't trust this
        "opt_margin_real": m_opt_real,          # what actually happens - the honest number
        "oracle_price": round(p_oracle, 2), "oracle_margin": m_oracle,
    })
opt_df = pd.DataFrame(opt_rows)

# sanity: since both optimizers search the identical price grid (the bounds
# depend only on current price/cost, not on elasticity), the oracle - which
# maximizes true-elasticity margin over that grid - can never be beaten by
# any other point in it, including our recommended price evaluated for real.
assert (opt_df["opt_margin_real"] <= opt_df["oracle_margin"] + 1e-6).all(), \
    "Our real-world outcome beat the oracle - that's impossible, something is wrong."

print(f"[checkpoint 2] Optimized {len(opt_df)} SKUs against shrunk elasticity, "
      f"and against the oracle (true elasticity) for comparison.")

n_worse_than_doing_nothing = (opt_df["opt_margin_real"] < opt_df["current_margin"]).sum()
print(f"[checkpoint 2b] Of {len(opt_df)} SKUs, {n_worse_than_doing_nothing} would actually end up "
      f"WORSE OFF in reality if we acted on our shrunk-elasticity recommendation - "
      f"a bad elasticity estimate can make optimization actively destroy margin, "
      f"not just leave money on the table.")

# ---- SECTION 3: the margin bridge --------------------------------------------
# Current -> REAL outcome of implementing our recommendation -> oracle. The
# first gap is real money the pricing exercise unlocks (or, for the SKUs
# above, actually destroys). The second gap is the cost of elasticity
# estimation error - money left on the table purely because we don't know
# the true elasticity, not because of anything wrong with the optimizer.

total_now = opt_df["current_margin"].sum()
total_opt_real = opt_df["opt_margin_real"].sum()
total_oracle = opt_df["oracle_margin"].sum()

print("\n=== MARGIN BRIDGE (weekly, national, all SKUs) ===")
print(f"Current margin:                                    ${total_now:,.0f}")
print(f"+ REAL gain from acting on our estimates:           ${total_opt_real - total_now:,.0f}  "
      f"-> ${total_opt_real:,.0f}  ({(total_opt_real/total_now - 1):+.1%})")
print(f"+ additional gain if elasticity were known perfectly: "
      f"${total_oracle - total_opt_real:,.0f}  -> ${total_oracle:,.0f}  ({(total_oracle/total_now - 1):+.1%})")

direction = np.where(opt_df["opt_price"] > opt_df["current_price"], "price increase",
                      np.where(opt_df["opt_price"] < opt_df["current_price"], "price decrease", "no change"))
opt_df["direction"] = direction
print("\n=== Where the REAL margin change comes from ===")
print(opt_df.groupby("direction").agg(
    n_skus=("sku_id", "count"),
    margin_gain=("opt_margin_real", lambda s: (s - opt_df.loc[s.index, "current_margin"]).sum()),
).round(0))

print("\n=== By brand tier ===")
tier_bridge = opt_df.groupby("brand_tier").agg(
    current_margin=("current_margin", "sum"),
    opt_margin=("opt_margin_real", "sum"),
    oracle_margin=("oracle_margin", "sum"),
)
tier_bridge["pct_gain_ours"] = (tier_bridge["opt_margin"] / tier_bridge["current_margin"] - 1).round(3)
for col in ["current_margin", "opt_margin", "oracle_margin"]:
    tier_bridge[col] = tier_bridge[col].round(0)
print(tier_bridge)

# ---- SECTION 4: efficient frontier -------------------------------------------
# Same optimizer, sweeping the "how big a price move do we allow" cap.
# This is the tradeoff curve: instead of a single recommended price change,
# leadership sees how much margin each level of pricing aggressiveness buys.

# Track both the BELIEVED margin (what the shrunk-elasticity model projects
# for itself - what you'd actually show leadership as "expected outcome")
# and the REAL margin (what true elasticity says would really happen for
# that same recommended price). Believed is what you'd deploy; real is only
# knowable here because we're grading against the hidden answer key.

frontier_rows = []
for max_move in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    believed_total, real_total = 0.0, 0.0
    for _, row in state.iterrows():
        p, q_believed, m_believed = optimize_sku(row, "shrunk_elasticity", max_down=max_move, max_up=max_move)
        q_real = row["current_demand"] * (p / row["current_price"]) ** row["true_elasticity"]
        believed_total += m_believed
        real_total += (p - row["base_cost"]) * q_real
    frontier_rows.append({
        "max_price_move": max_move,
        "believed_margin": believed_total,
        "real_margin": real_total,
        "pct_gain_believed": believed_total / total_now - 1,
        "pct_gain_real": real_total / total_now - 1,
    })
frontier_df = pd.DataFrame(frontier_rows)

print("\n=== Efficient frontier: BELIEVED vs REAL margin at each pricing-latitude level ===")
print(frontier_df.round(3).to_string(index=False))
print("\nWatch whether the believed/real gap widens as latitude grows - that's estimation-error "
      "risk compounding with how aggressively pricing is allowed to move.")

# ---- SECTION 5: sanity checks -------------------------------------------------

print("\n=== SANITY CHECKS ===")
zero_move = frontier_df[frontier_df["max_price_move"] == 0.0].iloc[0]
assert abs(zero_move["believed_margin"] - total_now) < 1e-6, \
    "max_price_move=0 should reproduce current margin exactly (believed)."
assert abs(zero_move["real_margin"] - total_now) < 1e-6, \
    "max_price_move=0 should reproduce current margin exactly (real)."
print("max_price_move=0.0 reproduces current total margin exactly, believed and real (as it must).")
assert (frontier_df["believed_margin"].diff().dropna() >= -1e-6).all(), \
    "Believed margin should be non-decreasing - more latitude can only help under the model's own belief."
print("Believed margin is monotonically non-decreasing in allowed price move (as it must be).")
print("(Real margin is NOT asserted monotonic - more latitude can amplify a bad estimate's damage.)")
print("\nAll checks passed.")
