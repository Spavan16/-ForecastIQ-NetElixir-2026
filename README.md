# ForecastIQ — AI-Powered Revenue Intelligence Platform
**Tagline:** "From Marketing Spend to Revenue Certainty — 30, 60, and 90-Day Enterprise Outlook"

**Team:** Team ForecastIQ  
**Members:** Pavan Kumar S (ENG24AD0047), Rohindth  
**College / Institution:** Dayananda Sagar University, Bengaluru  
**Challenge:** NetElixir AIgnition 2026 Hackathon Challenge  
**Python Version:** 3.11+

---

## 🏆 Challenge & Executive Overview
In modern eCommerce digital marketing, agencies and brands deploy capital across fragmented acquisition channels (Google Ads, Meta Ads, Microsoft Bing Ads). Allocating budgets without predicting marginal returns or evaluating ROAS constraints often leads to severe ad waste. 

**ForecastIQ** is an elite, production-grade SaaS forecasting platform and automated command-line utility designed to solve this exact problem. Moving far beyond rudimentary CSV plotting, ForecastIQ incorporates a multi-model weighted statistical ensemble (Prophet, XGBoost, LightGBM, CatBoost) coupled with stochastic **Monte Carlo Risk Simulations**, an algorithmic **Optuna Budget Optimizer**, **TreeSHAP Causal Explanations**, and an **Executive AI Analyst** reasoning engine.

The entire forecasting architecture functions **100% Offline** with no external network dependencies, fulfilling the strictest hackathon automation contracts while looking and behaving like an investor-ready SaaS startup.

---

## 🚀 Key Features & Hackathon Deliverables

### 1. Master Automated Failsafe Pipeline (`./run.sh`)
Conforming perfectly to the **Hackathon Submission Guide** contract, our root `./run.sh` script runs entirely offline, accepts standard paths, dynamically parses cross-channel datasets, builds features, evaluates pickled models, and writes pristine predictions:
```bash
./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
```
* **Universal Dimension Fallback:** If evaluators drop in entirely unseen held-out test data containing new campaign IDs or custom channels, our model unpickling pipeline automatically evaluates their spend shares and derives mathematically rigorous P10-P50-P90 projections with zero crashes.

### 2. Digital Marketing Data Validation & Ingestion Engine
* Dynamically detects cross-channel schemas across Google, Meta, and Bing Ads.
* Standardizes inconsistent campaign naming prefixes and maps time periods to rigorous ISO timestamps.
* Detects missing values, clips negative spend/revenue anomalies, and identifies outlier auction spikes.
* Computes an overarching **Data Quality Score** (e.g. `98.2 / 100`) and outputs detailed audit logs.

### 3. Multi-Model Weighted Ensemble Forecasting
Instead of relying on a single volatile model, ForecastIQ implements an elite weighted ensemble combining:
1. **Facebook Prophet (35% Weight):** Best-in-class handling of yearly and weekly seasonal time-series trends.
2. **XGBoost (25% Weight):** Evaluates deep non-linear multi-channel feature interactions.
3. **LightGBM (20% Weight):** Lightning fast gradient boosting capturing recent performance lag indicators.
4. **CatBoost (20% Weight):** Premium categorical and tabular splitting.
* **Probabilistic Confidence Intervals:** Mathematically evaluates historical residual variance to output **P10 (Worst-Case 90% Floor)**, **P50 (Expected Target)**, and **P90 (Best-Case Upside)** revenue and ROAS ranges across **30, 60, and 90-day** forecast windows.

### 4. Enterprise Monte Carlo Risk Intelligence
* Executes **10,000 rigorous stochastic paths** to model cash flow distribution uncertainties.
* Synthesizes Portfolio Risk Classification (**Low Risk**, **Medium Risk**, **High Risk**) by analyzing Revenue Volatility, Channel Dependency (HHI index), and ROAS Instability.

### 5. Algorithmic Optuna Budget Optimizer & Real-Time Simulator
* **Live Budget Simulator:** Interactive sliders allowing marketers to tweak Google, Meta, and Bing spend run-rates (e.g. Google +20%, Meta -10%, Bing +15%) with instant recalculated business outcomes.
* **Optuna Target ROAS Optimizer:** Runs 300 non-linear optimization trials to discover the exact global media spend split that maximizes revenue subject to hard Max Budget and Target ROAS floors.

### 6. TreeSHAP Causal Inference Layer
* Leverages **SHAP (SHapley Additive exPlanations)** to decode the exact marginal contribution of marketing features.
* Visually highlights **Top Revenue Drivers** and **Top ROAS Drivers**, explaining exactly the "Why" behind the numbers.

### 7. Strategic Business Scenario Generator
Generates multi-dimensional forecasts across 7 core enterprise operational scenarios:
1. `Expected Case (Baseline)` | 2. `Conservative Plan` | 3. `Aggressive Scale` | 4. `Recessionary Slump`  
5. `Q4 Holiday Demand Surge` | 6. `Black Friday / Cyber Week Blitz` | 7. `Aggressive Competitor Conquesting`

### 8. Production Enterprise PDF Reporting (`ReportLab`)
Automated generation of a beautifully styled, multi-page professional PDF report (`Executive AI Forecast & Revenue Intelligence Report`) complete with running headers, numbered footers (`Page X of Y`), Optuna spend splits, and SHAP causal breakdowns.

---

## 🏗️ System Architecture & Folder Structure

```
/home/user/
├── run.sh                       # Core hackathon automated execution entry point (Required)
├── requirements.txt             # Pinned production Python dependencies (Required)
├── README.md                    # System contract, technical documentation & demo walkthrough
├── data/                        # Input cross-channel analytics CSV directory
│   ├── google_ads_campaign_stats.csv
│   ├── meta_ads_campaign_stats.csv
│   └── bing_campaign_stats.csv
├── pickle/                      # Pickle artifact directory
│   └── model.pkl                # Committed trained multi-model ensemble artifact (Required)
├── output/                      # Pipeline outputs directory
│   ├── predictions.csv          # Conforming master P10-P50-P90 predictions table
│   └── executive_forecast_report.pdf # Beautifully formatted ReportLab PDF report
├── src/                         # Unified modular Core Python AI Intelligence package
│   ├── utils.py                 # Failsafe logging, paths, configs
│   ├── llm_provider.py          # AI Abstraction Layer (BaseLLM, Mock, OpenAI, Gemini, Claude)
│   ├── validation.py            # Multi-channel validation engine & Data Quality Score
│   ├── features.py              # Advanced feature engineering (Lag, volatility, season, shares)
│   ├── models.py                # Ensemble system (XGBoost, LightGBM, CatBoost, Prophet)
│   ├── monte_carlo.py           # 10,000-run Monte Carlo stochastic simulation engine
│   ├── budget_optimizer.py      # Optuna revenue maximizing solver
│   ├── scenarios.py             # 7 core eCommerce strategic business scenario generator
│   ├── explainability.py        # TreeSHAP causal feature importance extractor
│   ├── risk_engine.py           # Overarching Enterprise Risk Score (0-100) meter
│   ├── rule_engine.py           # Rule-based offline executive insight briefing
│   ├── chat_engine.py           # Contextual forecast chatbot routing layer
│   ├── pdf_reporting.py         # Premium ReportLab professional PDF builder
│   ├── database.py              # SQLAlchemy persistent run & user models
│   ├── generate_features.py     # CLI script 1: Ingests CSVs and builds features.parquet
│   └── predict.py               # CLI script 2: Loads model.pkl and exports predictions.csv
├── backend/                     # Production FastAPI REST Application
│   └── src/main.py              # Serves highly polished Next.js API endpoints
└── frontend/                    # Modern Next.js 14 App Router App (@TailwindCSS / Recharts)
    ├── package.json
    ├── tailwind.config.ts
    └── src/app/
        ├── layout.tsx
        └── page.tsx             # Master Single-Page 10-Tab SaaS Analytics Dashboard
```

---

## 🛠️ Technical Documentation

### 1. Forecasting Methodology & Model Selection
Digital marketing revenue forecasting exhibits high auto-correlation, non-linear spend saturation, and weekly shopping variance. A pure linear regression fails to capture diminishing returns, while pure deep learning suffers from overfitting on short analytical windows.

We selected an **Ensemble Architecture**:
* **Prophet** provides an incredibly robust baseline that does not degrade during unexpected short ad outages.
* **XGBoost & LightGBM** ingest our engineered marketing ratios (`CPC`, `CTR`, `Spend Share`, `Rolling STD`) to model non-linear auction elasticity.
* **Weighted Averaging** provides optimal stability, ensuring our P50 projections remain exceptionally authentic.

### 2. Data Preprocessing & Advanced Feature Engineering Suite
1. **Monetary Normalization:** Google Ads `metrics_cost_micros` is divided by $10^6$. Meta Ads `conversion` is mapped to true sales revenue. Bing Ads `TimePeriod` is cast to standard ISO timelines.
2. **Lag & Volatility Features:** Rolling $7$, $14$, and $30$-day means and standard deviations model campaign momentum. Variance is tracked to determine our overarching **Risk Classification**.
3. **Calendar Ratios:** Iso-weeks, quarters, weekend binary flags, and meteorological encoded seasons account for shopping cycles.

### 3. Rigorous Uncertainty Handling (P10, P50, P90)
To compute aggregate multi-horizon P10, P50, and P90 sums over $30$, $60$, and $90$ days, we derive the daily standard deviation of model residuals $\sigma$. Recognizing that marketing performance across multi-day horizons exhibits auto-correlation, our horizon uncertainty scales proportionally by $N^{0.75}$ (rather than a pure independent $\sqrt{N}$):
$$\text{Horizon STD} = \sigma \cdot N^{0.75}$$
$$\text{P10 (Floor)} = \text{P50} - 1.28 \cdot \text{Horizon STD}$$
$$\text{P90 (Upside)} = \text{P50} + 1.28 \cdot \text{Horizon STD}$$

### 4. Strategic Assumptions & Limitations
* **Assumptions:** Existing channel attribution is treated as the factual source of truth. Marketers rationalizing Optuna budgets will pace allocations evenly across the planning window.
* **Limitations:** The engine currently assumes stable macroeconomic base interest rates; drastic sudden supply chain shocks require manually toggling our pre-baked `Recessionary Slump` scenario.

### 5. AI Integration & Failsafe Abstraction Strategy
Our `BaseLLMProvider` interface connects to **Google Gemini (2.5 Flash)** (primary), **OpenAI (GPT-4o-Mini)**, or **Anthropic Claude (3 Haiku)** via `.env` API keys. 

**Absolute Failsafe Guarantee:** If no API key is present or if network calls timeout, the application seamlessly routes to an elite **MockLLMProvider**. This offline engine generates pristine, SaaS-grade executive summaries and causal chat insights incorporating exact live metrics with zero downtime.

---

## 💻 Execution & Demo Walkthrough

### Option 1: Standalone Hackathon Automated Evaluation CLI
To run the full hackathon testing sequence exactly as the NetElixir evaluation pipeline executes:
```bash
# 1. Clone repo and ensure run.sh is executable
chmod +x run.sh

# 2. Run the master evaluation pipeline
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv

# 3. Inspect the resulting multi-dimensional probabilistic CSV table
head -n 15 output/predictions.csv
```

### Option 2: Launch the SaaS FastAPI Backend & Next.js Frontend
To experience the jaw-dropping production SaaS startup prototype:

**Step 1: Start FastAPI Backend**
```bash
# From repository root
python3 backend/src/main.py
```
*(Runs Uvicorn on `http://localhost:8000`)*

**Step 2: Start Next.js Frontend**
```bash
# Open a new terminal window
cd frontend/
npm run dev
```
*(Runs Next.js on `http://localhost:3000`)*

**Demo Workflow:**
1. Open `http://localhost:3000` in your web browser.
2. **Executive Overview:** Explore your live audited $\$2.18\text{M}$ ad spend, multi-horizon P10-P50-P90 shaded Area charts, and cross-channel attribution shares.
3. **Data Validation Engine:** Inspect live multi-channel ingestion logs, anomaly detection flags, and Data Quality Score meters.
4. **Budget Simulator & Optuna Optimizer:** Tweak live media sliders to simulate diminishing returns or run the Optuna Algorithmic Solver to goal-seek your exact Target ROAS.
5. **Scenario Intelligence:** Click through your 7 core strategic enterprise scenarios.
6. **Explainability Engine:** Decode your exact Shapley feature importance bar charts.
7. **Executive Chatbot:** Type marketing questions to receive live contextual executive answers.
8. **Export Reports:** Click the `Export Executive PDF` button to instantly download your pristine ReportLab PDF document.

---
*Built with analytical rigor by Pavan Kumar S & Rohindth, Dayananda Sagar University — NetElixir AIgnition 2026.*
