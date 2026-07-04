# PriceIQ - a case study for Meridian Goods, a fictional mid-size retailer
#
# This file renders a single, scrolling story: the problem, the data, the
# forecasting model, the price sensitivity analysis, the pricing
# recommendations, the experiment that tested them, and a conversational
# question and answer tool at the end.
#
# All of the analysis (forecasting, elasticity, optimization, the CUPED
# experiment) was already computed in notebooks 01 through 07 and exported
# to dashboard_data/. This file only reads and presents that work, it never
# retrains a model or re-runs a regression.
#
# The question and answer tool uses one shared API key configured through
# Streamlit secrets, never a key typed in by a visitor. See
# .streamlit/secrets.toml.example for setup, and the README for how to
# configure the same secret on Streamlit Community Cloud.

import html
import json
import os

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = os.environ.get("PRICEIQ_DATA_DIR", os.path.join(os.path.dirname(__file__), "dashboard_data"))
GITHUB_URL = "https://github.com/krishshah26/priceiq"  # TODO: update if the actual repo name/owner differs
CONTACT_EMAIL = "krishshah712@gmail.com"
BUILDER_NAME = "Krish Shah"

COLORS = {
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4", "orange": "#eb6834",
    "good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b",
    "muted": "#898781", "grid": "#e1e0d9", "surface": "#fcfcfb",
}

st.set_page_config(page_title="PriceIQ | Meridian Goods", layout="wide", page_icon="\U0001F4B2")

# ---- visual design system ----------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Source Serif 4', serif !important; font-weight: 600 !important; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

.block-container { max-width: 980px; padding-top: 2.5rem; padding-bottom: 3rem; }

.kicker {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.78rem;
    font-weight: 600;
    color: #2a78d6;
    margin-bottom: 0.3rem;
}
.hero-title { font-size: 2.6rem; line-height: 1.15; margin-bottom: 0.6rem; }
.hero-subtitle { font-size: 1.15rem; color: #52514e; max-width: 700px; line-height: 1.55; }

[data-testid="stMetric"] {
    background-color: #f4f3ee;
    border: 1px solid #e1e0d9;
    border-radius: 10px;
    padding: 1rem 1.1rem;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}
[data-testid="stMetricLabel"] p { font-size: 0.85rem; color: #52514e; white-space: normal !important; }
[data-testid="stMetricValue"] div { white-space: normal !important; overflow: visible !important; }

.section-space { margin-top: 3.2rem; margin-bottom: 0.4rem; }
hr.divider { border: none; border-top: 1px solid #e1e0d9; margin: 3rem 0; }

.chat-answer {
    background-color: #f4f3ee;
    border-left: 3px solid #2a78d6;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-size: 1.02rem;
    line-height: 1.6;
    margin-top: 0.8rem;
}
.chat-answer p { margin: 0 0 0.9rem 0; }
.chat-answer p:last-child { margin-bottom: 0;
}
.footer-note { text-align: center; color: #898781; font-size: 0.9rem; padding: 1rem 0 0.5rem 0; }
.footer-note a { color: #2a78d6; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


def section(kicker, title):
    st.markdown(f'<div class="section-space"><p class="kicker">{kicker}</p></div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")


# ---- data loading -------------------------------------------------------------

@st.cache_data
def load_data():
    d = {}
    d["opt"] = pd.read_csv(f"{DATA_DIR}/optimization_results.csv")
    d["frontier"] = pd.read_csv(f"{DATA_DIR}/optimization_frontier.csv")
    d["tier_bridge"] = pd.read_csv(f"{DATA_DIR}/optimization_tier_bridge.csv")
    d["elasticity"] = pd.read_csv(f"{DATA_DIR}/elasticity_results.csv")
    d["elasticity_mae"] = pd.read_csv(f"{DATA_DIR}/elasticity_mae.csv")
    d["elasticity_tier_mae"] = pd.read_csv(f"{DATA_DIR}/elasticity_tier_mae.csv")
    d["forecast_folds"] = pd.read_csv(f"{DATA_DIR}/forecast_fold_results.csv")
    d["forecast_monthly"] = pd.read_csv(f"{DATA_DIR}/forecast_monthly.csv")
    d["experiment_summary"] = pd.read_csv(f"{DATA_DIR}/experiment_summary.csv")
    with open(f"{DATA_DIR}/experiment_stats.json") as f:
        d["experiment_stats"] = json.load(f)
    return d


@st.cache_resource
def get_duckdb_con(data_dir):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW fact_sales AS SELECT * FROM read_parquet('{data_dir}/fact_sales.parquet')")
    con.execute(f"CREATE VIEW dim_stores AS SELECT * FROM read_csv_auto('{data_dir}/dim_stores.csv')")
    con.execute(f"CREATE VIEW dim_products AS SELECT * FROM read_csv_auto('{data_dir}/dim_products.csv')")
    con.execute(f"CREATE VIEW dim_calendar AS SELECT * FROM read_csv_auto('{data_dir}/dim_calendar.csv')")
    con.execute("""
        CREATE VIEW v_daily_sales AS
        SELECT f.date, f.store_id, s.store_name, s.region, s.store_size,
               f.sku_id, p.sku_name, p.category, p.brand_tier,
               f.units_sold, f.price, f.cost,
               f.units_sold * f.price AS revenue,
               f.units_sold * (f.price - f.cost) AS margin,
               f.promo_flag, f.price_reason, f.stockout_flag,
               c.day_of_week, c.is_weekend, c.month, c.year, c.seasonal_index
        FROM fact_sales f
        JOIN dim_stores s ON f.store_id = s.store_id
        JOIN dim_products p ON f.sku_id = p.sku_id
        JOIN dim_calendar c ON f.date = c.date
    """)
    return con


try:
    data = load_data()
    con = get_duckdb_con(DATA_DIR)
except FileNotFoundError as e:
    st.error(f"Dashboard data not found in `{DATA_DIR}`. Run 07_export_for_dashboard.py, then place the "
             f"exported folder next to app.py as `dashboard_data/`.\n\nMissing: {e}")
    st.stop()

opt = data["opt"]
stats = data["experiment_stats"]
monthly = data["forecast_monthly"]
mae_df = data["elasticity_mae"]

total_now = opt["current_margin"].sum()
total_opt_real = opt["opt_margin_real"].sum()
total_oracle = opt["oracle_margin"].sum()
margin_gain_pct = total_opt_real / total_now - 1

overall_wmape_baseline = (monthly["wmape_baseline"] * monthly["n"]).sum() / monthly["n"].sum()
overall_wmape_lgb = (monthly["wmape_lgb"] * monthly["n"]).sum() / monthly["n"].sum()

naive_mae = mae_df.loc[mae_df["approach"] == "naive (all weeks)", "MAE_vs_true"].iloc[0]
shrunk_mae = mae_df.loc[mae_df["approach"] == "shrunk (clean + tier pooling)", "MAE_vs_true"].iloc[0]

raw_significant = not (stats["raw_ci_lo"] < 0 < stats["raw_ci_hi"])
guardrail_ok = not stats["guardrail_breached"]

# ---- hero -----------------------------------------------------------------

st.markdown('<p class="kicker">A pricing case study</p>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">How Meridian Goods stopped guessing at prices</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">For years, prices at Meridian Goods were set once a quarter by instinct, '
    'with no real way to know what a change actually did to sales or profit. This project replaces that '
    'guesswork with a system that forecasts demand, measures how customers really respond to price changes, '
    'recommends better prices for every product, and checks the recommendation with a real controlled '
    'experiment before anything ships.</div>',
    unsafe_allow_html=True,
)

st.write("")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Margin today", f"${total_now:,.0f}")
c2.metric("Margin, new pricing", f"${total_opt_real:,.0f}", f"{margin_gain_pct:+.1%}")
c3.metric("Confirmed lift", f"${stats['cuped_diff']:,.0f}",
          "significant" if raw_significant or stats["cuped_p"] < 0.05 else None)
c4.metric("Ready to roll out?", "Not yet" if stats["guardrail_breached"] else "Yes",
          f"{stats['units_pct_change']:+.1%} volume")

if stats["guardrail_breached"]:
    st.error(
        f"The pricing model's recommendation was not shipped as-is. It raises margin by "
        f"{margin_gain_pct:.1%} and that lift held up in a real experiment, but it also costs "
        f"{abs(stats['units_pct_change']):.1%} of unit volume, more than the business is willing to "
        f"give up for a margin gain. The honest fix is a more conservative version of the same "
        f"recommendation, not throwing out the work."
    )
else:
    st.success(
        f"The pricing model's recommendation raises margin by {margin_gain_pct:.1%}, that lift held up "
        f"in a real experiment, and it does not put an unacceptable amount of sales volume at risk."
    )

# ---- the problem and the data ----------------------------------------------

section("The data behind this", "A realistic simulation, built to be tested against the truth")
st.markdown(
    "Meridian Goods is a simulated retailer: 20 stores selling 100 products, every day, for two "
    "years straight. Everything on this page, every chart and every number, comes from that "
    "history: close to 1.5 million individual daily sales records. It was built as a simulation "
    "on purpose, because when the true demand and price sensitivity are known, every model here "
    "can be graded against the real answer, something live store data never allows. Here is a "
    "small sample of what the raw data actually looks like."
)
sample = con.execute(
    "SELECT date, store_name, category, brand_tier, units_sold, price, revenue, margin "
    "FROM v_daily_sales ORDER BY date, store_id, sku_id LIMIT 8"
).df()
st.dataframe(sample, hide_index=True, width='stretch')

# ---- chapter 1: forecasting -------------------------------------------------

section("Chapter one", "Forecasting demand")
st.markdown(
    "Before recommending a single price, the system needs a forecast of how much of each "
    "product will actually sell. A forecast only earns its keep once it beats the simplest "
    "possible alternative: guessing that next week looks like the same week a year earlier, or "
    "more simply, that a given day looks like the same day last week. Both approaches were "
    "tested side by side at seven different points across the two years of history, not just "
    "one convenient split, so the comparison below can be trusted."
)

fig = go.Figure()
fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["wmape_baseline"], name="Simple guess",
                          line=dict(color=COLORS["muted"], width=2)))
fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["wmape_lgb"], name="Forecasting model",
                          line=dict(color=COLORS["blue"], width=2)))
fig.update_layout(title="Average forecast error by month, lower is better", xaxis_title="Month",
                   yaxis_title="Average error", plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"])
st.plotly_chart(fig, width='stretch')
st.caption(
    f"Across the full test, the forecasting model's average error was {overall_wmape_lgb:.0%} versus "
    f"{overall_wmape_baseline:.0%} for the simple guess, a real and repeatable improvement, not a lucky run."
)

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=monthly["month"], y=monthly["bias_baseline"], name="Simple guess", marker_color=COLORS["muted"]))
fig2.add_trace(go.Bar(x=monthly["month"], y=monthly["bias_lgb"], name="Forecasting model", marker_color=COLORS["blue"]))
fig2.add_hline(y=0, line_color=COLORS["grid"])
fig2.update_layout(title="Forecast bias by month (below zero means under-forecasting)", barmode="group",
                    plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"])
st.plotly_chart(fig2, width='stretch')
st.caption(
    "December looks only mildly under-forecast here because it is measured against recorded sales, "
    "and recorded sales were themselves capped by empty shelves that month. Checked against real "
    "uncensored demand, the December shortfall is much larger, a limitation worth knowing about "
    "rather than hiding."
)

with st.expander("See the forecast tested at each point in time"):
    st.dataframe(data["forecast_folds"], hide_index=True, width='stretch')

# ---- chapter 2: price sensitivity -------------------------------------------

section("Chapter two", "Understanding price sensitivity")
st.markdown(
    "Not every product responds to a price change the same way. A cheap, easily substituted "
    "item loses customers quickly when its price rises. A premium item barely moves. Measuring "
    "this correctly is harder than it sounds, because store managers tend to cut prices exactly "
    "when a product is already selling poorly, and a naive analysis can mistake that coincidence "
    "for the truth, sometimes concluding the exact opposite of how customers really behave. The "
    "chart below compares a naive estimate against a corrected one, checked against the real, "
    "otherwise hidden, answer."
)

tier_mae = data["elasticity_tier_mae"]
fig3 = go.Figure()
for col, color, name in [
    ("naive_error", COLORS["muted"], "Naive estimate"),
    ("clean_error", COLORS["yellow"], "Cleaned-up estimate"),
    ("shrunk_error", COLORS["blue"], "Corrected estimate"),
]:
    fig3.add_trace(go.Bar(x=tier_mae["brand_tier"], y=tier_mae[col], name=name, marker_color=color))
fig3.update_layout(title="How far off each estimate was from the truth, by product tier", barmode="group",
                    plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"])
st.plotly_chart(fig3, width='stretch')
st.caption(
    f"The naive estimate was off by an average of {naive_mae:.2f} points. The corrected estimate "
    f"brought that down to {shrunk_mae:.2f}, a meaningfully more trustworthy number to price against."
)

elas = data["elasticity"]
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=elas["true_elasticity"], y=elas["shrunk_elasticity"], mode="markers",
                           marker=dict(color=COLORS["blue"], size=9), name="Product"))
lo, hi = elas["true_elasticity"].min(), elas["true_elasticity"].max()
fig4.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                           line=dict(color=COLORS["muted"], dash="dash"), name="Perfect estimate"))
fig4.update_layout(title="Corrected estimate versus the true, hidden answer, per product",
                    xaxis_title="True price sensitivity", yaxis_title="Estimated price sensitivity",
                    plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"])
st.plotly_chart(fig4, width='stretch')

# ---- chapter 3: price optimization ------------------------------------------

section("Chapter three", "Turning insight into price recommendations")
st.markdown(
    "Using the corrected price sensitivity numbers, the system searches for the price on every "
    "product that maximizes profit, inside safety limits: no price moves further than an agreed "
    "amount, and margins never fall below a floor. The chart below shows what the model expects "
    "to gain at different levels of pricing freedom, next to what would actually happen once real "
    "customer behavior is accounted for. The two lines are not supposed to match. The gap between "
    "them is the whole point, it shows exactly how much to trust the model's own optimism before "
    "acting on it."
)

frontier = data["frontier"]
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=frontier["max_price_move"], y=frontier["believed_margin"], name="What the model expects",
                           line=dict(color=COLORS["blue"], width=2)))
fig5.add_trace(go.Scatter(x=frontier["max_price_move"], y=frontier["real_margin"], name="What would really happen",
                           line=dict(color=COLORS["red"], width=2)))
fig5.update_layout(title="Expected versus real margin, by how much pricing freedom is allowed",
                    xaxis_title="Maximum allowed price move", yaxis_title="Weekly margin ($)",
                    plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"])
st.plotly_chart(fig5, width='stretch')
st.caption(
    "The wider that gap grows, the more the model is extrapolating on a shaky estimate. Pricing "
    "decisions here were made off the real curve, not the optimistic one."
)

with st.expander("See the recommendation for every product"):
    st.dataframe(opt.sort_values("opt_margin_real", ascending=False), hide_index=True, width='stretch')

# ---- chapter 4: the experiment -----------------------------------------------

section("Chapter four", "Proving it with a real experiment")
st.markdown(
    "A model's recommendation is only a hypothesis until it has been tested for real. Twenty "
    "stores were split at random into two even groups, one kept its current prices, the other "
    "received the new recommended prices, and both were tracked for eight weeks. Because some "
    "stores are naturally busier than others, comparing the two groups directly would bury the "
    "real effect in noise, so the analysis adjusted for each store's own recent history before "
    "comparing them. That single adjustment turned a small, inconclusive test into a "
    "statistically solid one."
)

fig6 = go.Figure()
fig6.add_trace(go.Bar(
    x=["Without adjusting for each store's history"], y=[stats["raw_diff"]], marker_color=COLORS["muted"],
    error_y=dict(type="data", symmetric=False,
                  array=[stats["raw_ci_hi"] - stats["raw_diff"]], arrayminus=[stats["raw_diff"] - stats["raw_ci_lo"]]),
))
fig6.add_trace(go.Bar(
    x=["Adjusted for each store's history"], y=[stats["cuped_diff"]], marker_color=COLORS["blue"],
    error_y=dict(type="data", symmetric=False,
                  array=[stats["cuped_ci_hi"] - stats["cuped_diff"]], arrayminus=[stats["cuped_diff"] - stats["cuped_ci_lo"]]),
))
fig6.add_hline(y=0, line_dash="dash", line_color=COLORS["grid"])
fig6.update_layout(title="Weekly margin lift, with the range of likely true values", yaxis_title="Lift ($ / week)",
                    showlegend=False, plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"])
st.plotly_chart(fig6, width='stretch')
st.caption(
    f"Without the adjustment, the result could not be told apart from no effect at all. After "
    f"adjusting for each store's own history, the range of likely true values narrowed by "
    f"{stats['variance_reduction']:.0%}, turning a shrug into a confident answer."
)

with st.expander("See the result for each store in the experiment"):
    st.dataframe(data["experiment_summary"], hide_index=True, width='stretch')

# ---- the bottom line ---------------------------------------------------------

section("The bottom line", "Where this leaves Meridian Goods")
if stats["guardrail_breached"]:
    st.markdown(
        f"The forecasting model beats a simple guess by a wide margin. The corrected price "
        f"sensitivity numbers are far more trustworthy than a naive analysis would produce. The "
        f"recommended prices genuinely raise margin, by {margin_gain_pct:.1%}, and a real "
        f"experiment confirmed that lift is not a fluke. But the same experiment caught something "
        f"the model's own numbers could not: this particular recommendation also costs "
        f"{abs(stats['units_pct_change']):.1%} of sales volume, a tradeoff most retailers would "
        f"reject. The honest recommendation is to dial back how aggressively prices move and test "
        f"again, not to ship this version or throw the whole approach away."
    )
else:
    st.markdown(
        f"The forecasting model beats a simple guess by a wide margin. The corrected price "
        f"sensitivity numbers are far more trustworthy than a naive analysis would produce. The "
        f"recommended prices raise margin by {margin_gain_pct:.1%}, a real experiment confirmed "
        f"that lift holds up, and the same experiment found no unacceptable cost in sales volume. "
        f"This recommendation is ready to roll out."
    )

# ---- ask a question -----------------------------------------------------------

st.markdown('<hr class="divider">', unsafe_allow_html=True)
section("Ask anything", "Have a question about this project?")
st.markdown(
    "Type a question below, about the business results, how any of this works, or the project "
    "itself, and get a straight, plain-English answer."
)


def get_secret(name, default=None):
    try:
        value = st.secrets[name]
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-flash-lite-latest")

FRIENDLY_FALLBACK = (
    "I could not put together a reliable answer to that just now. Feel free to try asking it a "
    "different way."
)

FORBIDDEN_SQL_WORDS = ["insert", "update", "delete", "drop", "alter", "create", "attach", "copy", "pragma", "call", ";"]


def clean_sql(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("sql"):
            text = text[3:]
    return text.strip().rstrip(";")


def is_safe_select(sql):
    lowered = sql.lower().strip()
    return lowered.startswith("select") and not any(w in lowered for w in FORBIDDEN_SQL_WORDS)


def format_answer_html(text):
    # The model's answers often come back as several newline-separated paragraphs, but raw
    # newlines are invisible in HTML - wrap each one in its own <p> so structure survives.
    paragraphs = [html.escape(p.strip()) for p in text.split("\n") if p.strip()]
    return "".join(f"<p>{p}</p>" for p in paragraphs)


def build_context_brief():
    guardrail_line = (
        f"The same test's volume guardrail was breached: unit volume dropped "
        f"{abs(stats['units_pct_change']):.1%}, more than the accepted tolerance, so this specific "
        f"recommendation was not shipped as-is, even though the margin gain is real."
        if stats["guardrail_breached"] else
        "The volume guardrail also passed, meaning this recommendation is safe to roll out."
    )
    return f"""
PROJECT BACKGROUND

This project is called PriceIQ, built for a fictional mid-size retailer called Meridian Goods.
Meridian Goods used to set prices once a quarter by instinct, with no real measurement of how a
price change affected sales or profit. This project replaces that guesswork with a system that
forecasts demand, measures how customers actually respond to price changes, recommends better
prices for every product, and checks the recommendation with a real controlled experiment before
anything ships.

It was built by {BUILDER_NAME}. If asked who built this, who made it, or how to get in touch,
always answer with that name and mention {CONTACT_EMAIL} for more details.

Scale of the data: close to 1.5 million daily records across 20 stores and 100 products over two
years. The data is a realistic simulation, built on purpose, because when the true demand and
price sensitivity are known ahead of time, every model in this project can be graded against the
real answer, something live store data never allows.

Tech stack: Python, pandas and numpy for data handling, LightGBM and statsmodels for the
statistical models, DuckDB for the underlying data queries, Streamlit for this website, Plotly
for the charts, and the Gemini API for this question and answer tool.

What it found, stage by stage:
1. Demand forecasting: a machine learning model was compared honestly against the simplest
   reasonable guess, tested at seven different points in time. The model's average forecast
   error was {overall_wmape_lgb:.0%} versus {overall_wmape_baseline:.0%} for the simple guess.
2. Price sensitivity: a naive estimate of how customers respond to price changes was off from
   the true answer by {naive_mae:.2f} points on average. A corrected estimate, that accounts for
   managers cutting prices on already-struggling products, brought that error down to
   {shrunk_mae:.2f} points.
3. Price recommendations: current weekly margin across all products and stores is
   ${total_now:,.0f}. The recommended prices would realistically bring that to
   ${total_opt_real:,.0f}, a gain of {margin_gain_pct:.1%}. If price sensitivity were known
   perfectly, the ceiling would be ${total_oracle:,.0f}.
4. The experiment: twenty stores were randomly split into two groups and tracked for eight
   weeks, with the analysis adjusted for each store's own history to make a small test
   statistically solid. The adjusted result showed a lift of ${stats['cuped_diff']:,.0f} per
   week. {guardrail_line}

HOW TO RESPOND

You are the friendly assistant on the PriceIQ website. Speak warmly and conversationally, like a
knowledgeable colleague who is proud of this work and happy to explain any part of it in detail,
in plain language, never dodging or over-simplifying a real question. Never use em dashes.

If a question is unrelated to this project (general knowledge, other companies, personal advice,
anything you cannot ground in the background above), do not answer it from your own knowledge.
Warmly say you are only able to help with questions about this project, and invite them to ask
something about it instead.
"""


WHY_HIRE_ANSWER = """You should hire Krish Shah for a pricing, revenue-management, or applied-data-science role because he turns messy data into pricing decisions a business can trust and act on.
Krish built PriceIQ, an end-to-end pricing and demand decision-intelligence platform, to demonstrate that he can own the full loop from raw data to a defensible dollar recommendation, which is the core of what a pricing or revenue-management team does every day.
Here is how the project maps to these roles:
Pricing and revenue management. The entire project is denominated in margin rather than surface-level metrics. Krish estimates price elasticity, optimizes prices under realistic merchandising constraints such as competitor guardrails, price-ladder ordering, and psychological price endings, and quantifies the margin impact with an honest uncertainty range rather than a single false-precision figure. He also handled one of the harder problems in pricing analytics, endogeneity. He recognized that historical prices are biased because managers tend to cut prices on already-slow products, which distorts naive elasticity estimates. By isolating cost-driven price changes as clean signal, he reduced his estimation error substantially and recovered the true price sensitivity.
Data science and experimentation. Krish validates before he claims. Before trusting his own optimizer, he designed a store-randomized A/B test with matched pairs and CUPED variance reduction to make a small sample statistically valid. He reports lift with confidence intervals and tracks guardrail metrics, on the principle that a margin gain that reduces store traffic can be a hidden loss. This reflects the experimentation rigor these teams rely on.
Analyst fundamentals. The project sits on a clean SQL pipeline over 1.5 million rows, honest rolling-origin backtesting that avoids data leakage, and a forecasting model that improved on its baseline by roughly 20 percent across every test fold. The results are presented in a dashboard designed for an executive audience to use directly.
Krish also uses AI the way many companies aim to deploy it, as a productivity layer for natural-language queries that is deliberately kept separate from the numbers, so the pricing math stays deterministic and auditable.
The consistent theme across the project is judgment: baseline discipline, causal awareness, a habit of validating before claiming, and a focus on translating analysis into the dollars a business cares about.
If you would like to discuss how he could bring this approach to your team, you can reach him at krishshah712@gmail.com."""

ROUTER_INSTRUCTION_TEMPLATE = """
{context_brief}

SPECIAL CASE: if the question asks why someone should hire the person who built this, why Krish
should be hired, what makes him a good fit, or anything closely equivalent, reply with exactly
this, unmodified, right after the ANSWER: prefix:

{why_hire_answer}

You can also query one database view called v_daily_sales for questions that need a specific
live number not already covered above (a particular store, category, or month, for example).
Its columns are: date, store_id, store_name, region, store_size, sku_id, sku_name, category,
brand_tier, units_sold, price, cost, revenue, margin, promo_flag, price_reason, stockout_flag,
day_of_week, is_weekend, month, year, seasonal_index.

Reply with EXACTLY ONE of these three formats, and nothing else:

SQL: <one DuckDB SELECT statement>
Use this only when the question needs a specific number from the data that is not already given
to you above.

ANSWER: <a warm, detailed, plain-English answer>
Use this for anything about the project itself: why it was built, the methodology, the tech
stack, who built it, or any of the numbers already given to you above.

DECLINE: <a short, warm redirect>
Use this for anything unrelated to this project.

Always start your reply with one of these three words followed by a colon and a space.
"""

ANSWER_INSTRUCTION = """
You are the friendly assistant on the PriceIQ website. You were given a question and the exact
data that answers it. Write a short, warm, confident answer in plain English, two to four
sentences, referencing the specific numbers in the data. Never mention SQL, databases, code, or
that you are an AI. Never use em dashes.
"""

question = st.text_input("Ask something", placeholder="Why should someone hire the person who built this?",
                          label_visibility="collapsed")
ask_clicked = st.button("Ask")

if ask_clicked and question:
    if not GEMINI_API_KEY:
        st.info("This tool is still getting set up, check back soon.")
    else:
        with st.spinner("Thinking..."):
            try:
                from google import genai as google_genai
                from google.genai import types as genai_types

                client = google_genai.Client(api_key=GEMINI_API_KEY)
                router_instruction = ROUTER_INSTRUCTION_TEMPLATE.format(
                    context_brief=build_context_brief(), why_hire_answer=WHY_HIRE_ANSWER
                )

                first = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=question,
                    config=genai_types.GenerateContentConfig(system_instruction=router_instruction, temperature=0.0),
                )
                raw = first.text.strip()
                upper = raw.upper()

                final_answer = None
                if upper.startswith("SQL:"):
                    sql = clean_sql(raw[4:])
                    if is_safe_select(sql):
                        try:
                            result_df = con.execute(sql).df()
                            records = result_df.head(20).to_dict(orient="records")
                            second = client.models.generate_content(
                                model=GEMINI_MODEL,
                                contents=f"Question: {question}\n\nData: {records}",
                                config=genai_types.GenerateContentConfig(
                                    system_instruction=ANSWER_INSTRUCTION, temperature=0.2
                                ),
                            )
                            final_answer = second.text.strip()
                        except Exception:
                            final_answer = FRIENDLY_FALLBACK
                    else:
                        final_answer = FRIENDLY_FALLBACK
                elif upper.startswith("ANSWER:"):
                    final_answer = raw[7:].strip()
                elif upper.startswith("DECLINE:"):
                    final_answer = raw[8:].strip()
                else:
                    final_answer = FRIENDLY_FALLBACK

                st.markdown(f'<div class="chat-answer">{format_answer_html(final_answer)}</div>', unsafe_allow_html=True)
            except Exception:
                st.markdown(f'<div class="chat-answer">{format_answer_html(FRIENDLY_FALLBACK)}</div>', unsafe_allow_html=True)

# ---- how this was built -------------------------------------------------------

with st.expander("How this was built"):
    st.markdown(f"""
**Methodology**

- **Forecasting**: LightGBM trained with rolling-origin backtesting across seven cutoff dates, compared
  against a seasonal-naive baseline, scored on both average error (WMAPE) and bias.
- **Price sensitivity**: per-product regressions restricted to cost-driven price changes to avoid
  mistaking manager markdowns for customer behavior, then pooled toward each product tier's average
  using an empirical Bayes shrinkage estimator to stabilize noisy individual estimates.
- **Price optimization**: a constrained grid search over feasible prices per product, maximizing
  expected margin subject to a maximum price move and a minimum margin floor, always evaluated
  against both the model's own belief and the true price sensitivity for an honest read.
- **The experiment**: a store-randomized controlled test with CUPED variance reduction, using each
  store's pre-experiment margin as a covariate, plus a volume guardrail check before declaring a win.

**Tech stack**: Python, pandas, numpy, LightGBM, statsmodels, DuckDB, Streamlit, Plotly, and the
Gemini API.

**About the question and answer tool above**: when it needs a specific number that is not already
summarized on this page, it works out a database query behind the scenes, runs it against the real
data, and then explains the result in plain language. It never invents a number, and it never
answers a question outside of this project from general knowledge.
""")

# ---- footer ---------------------------------------------------------------

st.markdown(
    f'<div class="footer-note">Built by {BUILDER_NAME} &middot; '
    f'<a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a> &middot; '
    f'<a href="{GITHUB_URL}" target="_blank">View the code on GitHub</a></div>',
    unsafe_allow_html=True,
)
