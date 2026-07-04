# PriceIQ

Meridian Goods is a fictional mid-size retailer that priced its shelves the way a lot of real ones still do: once a quarter, by gut feel, with no real way to know what a price change actually did to sales or profit. This project replaces that guesswork with a system that forecasts demand, measures how customers actually respond to price changes, recommends better prices, and checks the recommendation with a real controlled experiment before anything ships.

**[Try the live demo](#)** (link goes here once deployed)

## The result, up front

Current weekly margin across Meridian Goods' 20 stores and 100 products is about $74,000. The pricing model recommends changes that would raise that to roughly $88,700, a 19.7% gain, and a real randomized experiment confirmed that lift holds up. The same experiment also caught something the model's own numbers could not see on their own: this particular recommendation costs close to 30% of unit volume, more than most retailers would accept for a margin gain. The honest conclusion isn't "ship it" or "the model failed," it's "dial back how aggressively prices move and test again." That's the kind of judgment call this whole project is built to support.

## What's inside

**Forecasting demand.** A LightGBM model was tested against the simplest reasonable baseline (assume a day looks like the same day last week), backtested at seven different points across two years of history rather than a single lucky split. The model brought average forecast error down from 76.6% to 61.0%, a real improvement, not a coincidence.

**Understanding price sensitivity.** Measuring how customers respond to a price change sounds simple and rarely is, because store managers tend to cut prices exactly when a product is already struggling. A naive analysis mistakes that coincidence for the truth, sometimes concluding the opposite of how customers actually behave. Restricting the analysis to price changes driven by cost rather than by weak sales, then pooling noisy individual estimates toward their category average, brought the average error down from 1.90 to 0.95, roughly half.

**Turning insight into price recommendations.** Using the corrected price sensitivity numbers, the system searches for the price on every product that maximizes profit, inside safety limits: no price moves further than an agreed amount, and margins never fall below a floor. It also tracks the gap between what the model expects to gain and what would actually happen once real customer behavior is accounted for, because that gap is exactly how much to trust the model's optimism before acting on it.

**Proving it with a real experiment.** Twenty stores were split at random into two groups, one kept current prices, the other received the new recommendations, tracked for eight weeks. With only 20 stores, a naive comparison couldn't tell the effect apart from noise. Adjusting for each store's own recent history (a technique called CUPED) narrowed the range of likely outcomes by 98%, turning an inconclusive test into a confident one, and also surfaced the volume tradeoff mentioned above.

**A dashboard that explains itself.** The live site walks through all of the above as a single narrative, and ends with a conversational question and answer tool that can explain any number, any stage of the methodology, or the project itself in plain language, without ever exposing the query it ran behind the scenes to get there.

## Tech stack

Python, pandas and numpy for data handling, LightGBM and statsmodels for the statistical models, DuckDB for the underlying data queries, Streamlit for the website, Plotly for the charts, and the Gemini API for the question and answer tool.

## Running it locally

```
python -m venv .venv
.venv\Scripts\activate          # or source .venv/bin/activate on macOS/Linux
pip install -r dashboard/requirements.txt
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in a real Gemini API key if you want the question and answer tool to work locally.

```
streamlit run dashboard/app.py
```

The dashboard reads precomputed results from `dashboard/dashboard_data/`, it does not retrain anything at runtime. That data is produced by the pipeline in `notebooks/`, run in order from `01_generate_data.py` through `07_export_for_dashboard.py`, which generates the synthetic transaction history, builds the SQL views, backtests the forecast, estimates elasticity, optimizes prices, runs the experiment, and exports everything the dashboard needs.

## Contact

Built by Krish Shah. For more details, reach out at krishshah712@gmail.com.
