# ForecastIQ — Architecture Overview

## System Architecture

```
data/ (CSV inputs)
    │
    ▼
[ValidationEngine]          ← ingestion, schema enforcement, quality scoring
    │
    ▼
[FeatureEngineer]           ← time, performance, marketing, channel features
    │
    ▼
[EnsembleForecaster]        ← Prophet + XGBoost + LightGBM + CatBoost
    │         │
    │    pickle/model.pkl   ← pre-trained, committed to repo
    │
    ▼
[MonteCarloSimulator]       ← 1000-iteration portfolio simulation → P10/P50/P90
    │
    ▼
output/predictions.csv      ← scored deliverable
    │
    ▼
[FastAPI Backend]           ← 13 REST endpoints serving live analytics
    │
    ▼
[Next.js Frontend]          ← executive dashboard UI
```

---

## Frontend Stack

| Component | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Charts | Recharts |
| State | React useState / useEffect |
| API communication | fetch() with `NEXT_PUBLIC_API_BASE` env var |

The frontend is a single-page executive dashboard with six views: Overview, Forecasts, Budget Optimizer, Scenarios, Explainability, and Risk Intelligence. All data is fetched from the FastAPI backend at startup and cached client-side.

---

## Backend Stack

| Component | Technology |
|---|---|
| Framework | FastAPI |
| Server | Uvicorn |
| Language | Python 3.11+ |
| Data layer | SQLite via SQLAlchemy |
| Serialization | Pydantic v2 |
| PDF generation | ReportLab |

The backend exposes 13 REST endpoints under `/api/`. All heavy computation (forecasting, SHAP, Monte Carlo, optimization) runs once on startup and is cached in-memory for sub-millisecond subsequent API responses. Cache invalidation is triggered by file modification timestamps on the `data/` CSVs.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | System health, LLM provider, data quality score |
| `/api/overview` | GET | Executive KPIs, 90-day trajectory, channel attribution |
| `/api/forecasts` | GET | P10/P50/P90 by dimension (Overall/Channel/CampaignType/Campaign) |
| `/api/trajectory` | GET | 90-day daily revenue trajectory for area chart |
| `/api/validation` | GET | Full data quality audit report |
| `/api/simulations` | GET | Monte Carlo portfolio simulation results |
| `/api/simulate-budget` | POST | What-if budget change simulation |
| `/api/optimize-budget` | POST | Optuna budget allocation optimizer |
| `/api/scenarios` | GET | Bull/Base/Bear scenario projections |
| `/api/explainability` | GET | SHAP feature importance for revenue and ROAS |
| `/api/risk` | GET | Risk profile with factor scores and mitigations |
| `/api/insights` | GET | Rule-based executive insights and recommendations |
| `/api/chat` | POST | Conversational AI query answering |
| `/api/report/pdf` | GET | Download executive PDF report |

---

## Forecasting Pipeline

```
run.sh
  ├── src/generate_features.py
  │     ├── ValidationEngine.run_full_ingestion()   ← reads data/*.csv
  │     ├── FeatureEngineer.generate_all_features()
  │     └── writes features.pkl
  │
  └── src/predict.py
        ├── loads features.pkl → determines forecast start_date
        ├── EnsembleForecaster.load_models()         ← loads pickle/model.pkl
        ├── EnsembleForecaster.produce_full_predictions_table()
        └── writes output/predictions.csv
```

The pipeline is fully offline and requires no network access. All model weights are pre-trained and committed to the repository under `pickle/model.pkl`.

---

## LLM Integration Workflow

```
User Question / Insight Request
        │
        ▼
  GEMINI_API_KEY set?
   YES ──────────────────────────────────────────────────────┐
        │                                                     │
        NO                                              GeminiProvider
        │                                               (gemini-2.0-flash)
        ▼                                                     │
  MockLLMProvider                                             │
  (fully data-driven,                                         │
   no network calls)                                          │
        │                                                     │
        └───────────────────┬─────────────────────────────────┘
                            │
                            ▼
              Structured analytics context payload:
              - total spend / revenue / ROAS
              - channel breakdown (per-channel stats)
              - 30d vs prior 30d trend deltas
              - 90-day P50 forecast values
              - risk classification
                            │
                            ▼
              Natural language causal narrative response
              (grounded in actual computed dataset statistics)
```

The dual-provider architecture ensures the system delivers high-quality AI-assisted insights in production (Gemini) while remaining fully functional in offline/automated evaluation environments (MockLLMProvider).
