# ForecastIQ — Technical Documentation

## 1. Forecasting Methodology

ForecastIQ uses a **weighted ensemble of four models** to produce probabilistic revenue and ROAS forecasts across 30, 60, and 90-day planning horizons.

### Ensemble Architecture

| Model | Weight | Role |
|---|---|---|
| Prophet | 35% | Seasonal decomposition, trend + weekly/annual cycles |
| XGBoost | 25% | Non-linear spend-to-revenue mapping, feature interactions |
| LightGBM | 20% | Gradient boosting with fast training on high-cardinality campaign features |
| CatBoost | 20% | Categorical feature handling (channel, campaign type, season) |

The weighted average of all four model predictions forms the raw ensemble signal. This raw signal is then blended with a **recency-anchor baseline** — the trailing 30-day daily average (90% weight) plus trailing 90-day average (10% weight) — at a **25% raw-ensemble / 75% recency-anchor** split (`model_blend_weight` in `models.py`) before being projected forward as the final P50 forecast. This blending step exists to stabilize the ensemble against short-horizon noise, but it also means the recency anchor's own accuracy dominates the final number — relevant context for Section 5's backtest discussion below, where this weighting is identified as a driver of the Revenue-vs-naive-baseline gap. Probabilistic ranges (P10/P90) are derived from historical residual standard deviation scaled by planning horizon — wider intervals at 90 days reflect genuine macro uncertainty, not arbitrary padding.

### Monte Carlo Simulation

Separately from the ensemble's own P10/P50/P90 bands, a Monte Carlo simulation (10,000 iterations, seed=42 for reproducibility) samples from a volatility distribution derived from the ensemble's residual std to produce the Risk tab's portfolio-level revenue/ROAS distributions, worst/expected/best-case classifications, and histogram visualizations. It does not feed into `predictions.csv` — the P10/P90 values there are computed analytically from residual standard deviation scaled by horizon (see above), independent of the Monte Carlo paths.

### Forecast Dimensions

Forecasts are produced at four levels of granularity:
- **Overall**: blended portfolio across all channels
- **Channel**: Google Ads, Meta Ads, Bing Ads independently
- **Campaign Type**: SEARCH, SOCIAL, PERFORMANCE_MAX, SHOPPING, DISPLAY, VIDEO, DEMAND_GEN, AUDIENCE (derived from the actual channel data at prediction time, not a fixed list — see Section 5)
- **Campaign**: top individual campaigns by historical revenue contribution

---

## 2. Model Selection Rationale

**Why an ensemble and not a single model?**

Each model captures a different signal in marketing data:

- Prophet excels at capturing weekly purchase intent cycles and seasonal shopping events (Black Friday, Q4) that are invisible to tree-based models without explicit feature engineering.
- XGBoost/LightGBM/CatBoost capture non-linear diminishing returns on ad spend — the relationship between spend and revenue is not linear at high budget levels, and tree-based models learn this directly from data.
- Ensemble weighting (Prophet higher at 35%) reflects the strong seasonality present in e-commerce marketing data, where day-of-week and month effects are among the most predictive features.

**Why not a neural network / LSTM?**

The dataset (887 daily rows across ~2.5 years) is too small for deep learning to outperform well-tuned gradient boosting. Ensemble tree models + Prophet deliver superior out-of-sample accuracy on marketing time-series at this data volume.

---

## 3. Data Preprocessing Logic

### Ingestion

Three channel CSV files are ingested independently:
- `google_ads_campaign_stats.csv` — 19,272 records
- `meta_ads_campaign_stats.csv` — 3,417 records  
- `bing_campaign_stats.csv` — 2,873 records

Each file undergoes channel-specific validation (column existence, dtype enforcement, spend/revenue range checks) before being unified into a single cross-channel dataframe (25,562 records total).

### Normalization

- **Google Ads spend**: converted from micros (divide by 1,000,000) to dollars
- **Meta Ads revenue**: sourced from the `conversion` column (represents conversion value)
- **Bing Ads columns**: normalized from PascalCase (`Spend`, `Revenue`) to lowercase

### Date Handling

Malformed or missing dates are forward-filled using `.bfill()` then defaulted to `2025-01-01` if the entire column is null. Date parsing uses `errors='coerce'` to prevent ingestion failures on format inconsistencies.

### Feature Engineering

From the unified dataframe, the following features are constructed:
- **Time features**: month, quarter, ISO week, day_of_week, is_weekend
- **Season encoding**: WINTER(0), SPRING(1), SUMMER(2), FALL(3)
- **Rolling performance**: 7/14/30-day rolling mean and std of revenue and spend per channel
- **Efficiency features**: spend-to-revenue ratio, rolling ROAS, volatility index
- **Channel encoding**: one-hot encoded channel and campaign_type columns

---

## 4. Assumptions

1. **Attribution is treated as source of truth.** No custom attribution model or Media Mix Modeling (MMM) is applied. Channel-reported revenue figures are used as-is per challenge specification.

2. **Stationarity of channel mix.** The ensemble assumes the relative contribution of Google Ads, Meta Ads, and Bing Ads to total revenue remains broadly stable over the forecast horizon. Structural shifts (e.g. a channel being paused entirely) are outside the model's scope.

3. **Spend continuity.** Forecasts assume ad spend continues at approximately the historical run rate unless explicitly modified via the budget simulator. The model does not account for future spend changes autonomously.

4. **No external macro variables.** The model does not incorporate macroeconomic signals (inflation, consumer confidence) or competitor activity. Seasonality is captured via date features derived from historical patterns only.

5. **Campaign structure stability.** Which channels, campaign types, and campaigns get scored is derived from the actual held-out data at prediction time (not a fixed list baked in at training time), so new campaign types or campaign names introduced after the training cutoff are still scored — via the Universal Dimension Fallback, which scales down the current overall-portfolio forecast for that entity (see Section 5) rather than requiring an entity-specific trained model.

---

## 5. Limitations

- **Short Bing Ads history**: Bing Ads has 2,873 records across fewer unique campaigns, making campaign-level Bing forecasts less reliable than Google Ads equivalents.
- **Cold start for new campaigns**: Campaigns not seen during training use a scaled-down version of the overall forecast. This is disclosed in the fallback path.
- **90-day horizon uncertainty**: P10/P90 intervals widen significantly at 90 days. The P50 point estimate is reliable; the tails should be treated as scenario bounds rather than precise predictions.
- **No intra-day modeling**: All forecasts are aggregate-period (30/60/90 days) as specified. Daily granularity trajectories shown in the UI are smoothed interpolations for visualization only and are not scored outputs.
- **Revenue MAPE vs. a naive baseline**: In the offline rolling-origin backtest (3 folds, `output/backtest_summary.json`), ROAS forecasts beat a naive baseline (flat trailing-30-day average) at the 60-day horizon and trail narrowly at 30/90 days; Revenue forecasts trail the naive baseline across all three horizons. We're disclosing this rather than omitting it — see the full backtest breakdown and root-cause analysis (recency-anchoring weight, limited history at the earliest backtest origin, and a Nov–Dec demand spike with only two historical instances in the ~2.5-year dataset) in `executive_forecast_report.pdf`, Section 4.

---

## 6. AI Integration Strategy

### Primary LLM: Gemini (gemini-2.5-flash)

When a `GEMINI_API_KEY` is present in the environment, all causal summaries, anomaly interpretations, and chat responses are generated by Gemini via the `/v1beta/models/gemini-2.5-flash:generateContent` endpoint. The LLM receives a structured analytics context payload containing:
- Historical spend, revenue, and ROAS by channel
- Recent 30-day vs prior 30-day trend deltas
- 90-day P50 forecast values
- Risk profile classification

This grounds every LLM response in actual computed numbers rather than general marketing knowledge.

### Offline Fallback: Rule-Based Causal Engine

When no API key is present (e.g. on the automated scoring pipeline), the system falls back to a fully data-driven rule engine (`rule_engine.py`) and a data-driven chat engine (`chat_engine.py`). These compute the same analytical statistics from the data and generate structured causal narratives — with real numbers from the actual dataset — without any network calls.

This design ensures the forecasting utility functions completely offline while maintaining high-quality AI-assisted insights in production environments where an API key is available.

### LLM Use Cases

| Use Case | Implementation |
|---|---|
| Executive summary generation | `RuleInsightEngine` (offline) / Gemini (online) |
| Anomaly interpretation | `RuleInsightEngine` ROAS trend detection |
| Causal Q&A | `ForecastChatBot` with 10 intent classifiers |
| Risk narrative | `RiskIntelligenceEngine` factor scoring |
| Budget recommendation rationale | `BudgetOptimizer` + Gemini explanation |
