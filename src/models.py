import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from prophet import Prophet

from src.utils import get_logger, PICKLE_DIR

logger = get_logger("ForecastingEnsemble")

class EnsembleForecaster:
    """
    Elite Production Ensemble System for ForecastIQ.
    Combines XGBoost, LightGBM, CatBoost, and Prophet via weighted averaging.
    Generates multi-dimensional forecasts (Overall, Channel, CampaignType, Campaign)
    with rigorous P10, P50, P90 probabilistic confidence intervals.
    Includes Universal Dimension Fallback to handle any unseen test campaigns seamlessly.
    """
    def __init__(self, model_path: Path = PICKLE_DIR / "model.pkl", validation_engine=None):
        self.model_path = model_path
        self.models: Dict[str, Any] = {}
        self.weights = {
            "prophet": 0.35,
            "xgboost": 0.25,
            "lightgbm": 0.20,
            "catboost": 0.20
        }
        self.is_trained: bool = False
        self.residuals_std: Dict[str, float] = {}
        self.validation_engine = validation_engine
        self.top_campaign_names: List[str] = []  # FIX: stored separately, populated during training
        self.trained_channels: List[str] = []  # BUG 10 fix: real channels seen at training time
        self.trained_campaign_types: List[str] = []  # BUG 10 fix: real campaign types seen at training time
        # BUG 1 fix: which feature columns the overall model was actually trained on (base time
        # features + whatever engineered columns features.py/generate_features.py produced), and
        # the last observed value of each engineered column — carried forward as a static feature
        # across the forecast horizon, since true future rolling/volatility/marketing-ratio values
        # aren't computable without future actuals.
        self.feature_columns: List[str] = []
        self.last_known_engineered: Dict[str, float] = {}
        # BUG 2 fix: real, data-derived fallback scale per dimension_type (channel/campaign_type/
        # campaign), used when forecast_dimension() encounters an entity unseen during training.
        # Replaces the old hardcoded 0.33/0.5/0.05 constants with the MEDIAN historical revenue
        # share among known entities of that type — median rather than mean specifically because
        # campaign-level revenue is typically long-tailed (few large campaigns, many small ones),
        # so a median is a more representative "what would a typical unseen entity look like"
        # estimate than a mean, which a handful of large campaigns would skew upward.
        self.dimension_fallback_scale: Dict[str, float] = {}

    BASE_TIME_FEATURES = ['month', 'quarter', 'week', 'day_of_week', 'is_weekend', 'season_encoded']
    _NON_FEATURE_COLS = {'date', 'revenue', 'spend', 'clicks', 'impressions', 'conversions', 'season'}

    def _prepare_tabular_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # BUG 1 fix: previously hardcoded to only the 6 base time features, ignoring any
        # engineered columns (rolling windows, ROAS, CPC, CTR, conversion_rate, volatility,
        # trend indicators) even when present in df. Now auto-detects and includes them,
        # while remaining backward-compatible with a plain daily_df that only has base features.
        engineered = [c for c in df.columns if c not in self.BASE_TIME_FEATURES and c not in self._NON_FEATURE_COLS]
        self.feature_columns = self.BASE_TIME_FEATURES + sorted(engineered)
        X = df[self.feature_columns].copy()
        y = df['revenue'].copy()
        return X, y

    def _prepare_future_features(self, start_date: pd.Timestamp, periods: int, last_known_engineered: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        future_dates = pd.date_range(start=start_date, periods=periods, freq='D')
        future = pd.DataFrame({'date': future_dates})
        future['month'] = future['date'].dt.month
        future['quarter'] = future['date'].dt.quarter
        future['week'] = future['date'].dt.isocalendar().week.astype(int)
        future['day_of_week'] = future['date'].dt.dayofweek
        future['is_weekend'] = future['date'].dt.dayofweek.isin([5, 6]).astype(int)
        future['season_encoded'] = future['month'].apply(lambda m: 0 if m in [12,1,2] else (1 if m in [3,4,5] else (2 if m in [6,7,8] else 3)))
        # BUG 1 fix: carry forward the last known engineered feature values (last-observation-
        # carried-forward) so the tree models receive the same feature columns at inference
        # that they were trained on, instead of silently dropping them.
        if last_known_engineered:
            for col, val in last_known_engineered.items():
                future[col] = val
        return future

    def train_overall_models(self, daily_df: pd.DataFrame):
        logger.info("Training Overall Revenue Ensemble Models (XGBoost, LightGBM, CatBoost, Prophet)...")
         
        prophet_df = daily_df[['date', 'revenue']].rename(columns={'date': 'ds', 'revenue': 'y'})
        m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, interval_width=0.80)
        m_prophet.fit(prophet_df)
        self.models["overall_prophet"] = m_prophet

        X, y = self._prepare_tabular_features(daily_df)

        m_xgb = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
        m_xgb.fit(X, y)
        self.models["overall_xgb"] = m_xgb

        m_lgb = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1)
        m_lgb.fit(X, y)
        self.models["overall_lgb"] = m_lgb

        m_cat = CatBoostRegressor(iterations=150, learning_rate=0.08, depth=5, random_seed=42, verbose=0)
        m_cat.fit(X, y)
        self.models["overall_cat"] = m_cat

        preds_xgb = m_xgb.predict(X)
        preds_lgb = m_lgb.predict(X)
        preds_cat = m_cat.predict(X)
        
        ens_fit = 0.35 * m_prophet.predict(prophet_df)['yhat'].values + 0.25 * preds_xgb + 0.20 * preds_lgb + 0.20 * preds_cat
        self.residuals_std["overall"] = float(np.std(y.values - ens_fit))

        # BUG 1 fix: snapshot the last observed value of every engineered feature column so
        # forecast_overall() can carry them forward into the future feature matrix at inference.
        engineered_cols = [c for c in self.feature_columns if c not in self.BASE_TIME_FEATURES]
        if engineered_cols:
            last_row = daily_df.sort_values('date').iloc[-1]
            self.last_known_engineered = {c: float(last_row[c]) for c in engineered_cols}
            logger.info(f"Overall ensemble trained on {len(self.feature_columns)} features "
                        f"({len(engineered_cols)} engineered: rolling/volatility/trend/marketing-ratio).")
        else:
            self.last_known_engineered = {}
            logger.info("Overall ensemble trained on base time features only (no engineered columns present in daily_df).")

        X_spend, y_spend = daily_df[['month', 'quarter', 'week', 'day_of_week', 'is_weekend', 'season_encoded']], daily_df['spend']
        m_spend = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1).fit(X_spend, y_spend)
        self.models["overall_spend"] = m_spend
        self.residuals_std["overall_spend"] = float(np.std(y_spend.values - m_spend.predict(X_spend)))

    def train_dimension_models(self, unified_df: pd.DataFrame):
        logger.info("Training Dimension-Level Forecasting Models...")

        def _train_lgb(daily_df, key_rev, key_spend):
            daily_df = daily_df.copy()
            daily_df['month']          = daily_df['date'].dt.month
            daily_df['quarter']        = daily_df['date'].dt.quarter
            daily_df['week']           = daily_df['date'].dt.isocalendar().week.astype(int)
            daily_df['day_of_week']    = daily_df['date'].dt.dayofweek
            daily_df['is_weekend']     = daily_df['day_of_week'].isin([5, 6]).astype(int)
            daily_df['season_encoded'] = daily_df['month'].apply(
                lambda m: 0 if m in [12,1,2] else (1 if m in [3,4,5] else (2 if m in [6,7,8] else 3))
            )
            feats = ['month', 'quarter', 'week', 'day_of_week', 'is_weekend', 'season_encoded']
            X = daily_df[feats]
            if len(X) <= 5:
                return
            y_rev = daily_df['revenue']
            m_rev = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1).fit(X, y_rev)
            self.models[key_rev] = m_rev
            self.residuals_std[key_rev.replace("_revenue", "")] = float(np.std(y_rev.values - m_rev.predict(X)))
            y_sp = daily_df['spend']
            m_sp = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1).fit(X, y_sp)
            self.models[key_spend] = m_sp

        # Channel-level models
        for channel in unified_df['channel'].unique():
            ch_df = unified_df[unified_df['channel'] == channel]
            daily_ch = ch_df.groupby('date').agg({'revenue': 'sum', 'spend': 'sum'}).reset_index()
            _train_lgb(daily_ch, f"channel_{channel}_revenue", f"channel_{channel}_spend")

        # Campaign type-level models
        for ctype in unified_df['campaign_type'].unique():
            ct_df = unified_df[unified_df['campaign_type'] == ctype]
            daily_ct = ct_df.groupby('date').agg({'revenue': 'sum', 'spend': 'sum'}).reset_index()
            _train_lgb(daily_ct, f"ctype_{ctype}_revenue", f"ctype_{ctype}_spend")

        # Campaign-level models
        for cname in unified_df['campaign_name'].unique():
            c_df = unified_df[unified_df['campaign_name'] == cname]
            daily_c = c_df.groupby('date').agg({'revenue': 'sum', 'spend': 'sum'}).reset_index()
            _train_lgb(daily_c, f"camp_{cname}_revenue", f"camp_{cname}_spend")

        self.is_trained = True
        self.top_campaign_names = list({
            k.replace("camp_", "").replace("_revenue", "")
            for k in self.models if k.startswith("camp_") and k.endswith("_revenue")
        })
        # BUG 10 fix: capture the REAL set of channels/campaign types seen during training,
        # instead of a hardcoded ["SEARCH", "SOCIAL"] list downstream in produce_full_predictions_table().
        self.trained_channels = sorted(unified_df['channel'].unique().tolist())
        self.trained_campaign_types = sorted(unified_df['campaign_type'].unique().tolist())

        # BUG 2 fix: compute the real median per-entity revenue share for each dimension_type,
        # to replace forecast_dimension()'s old hardcoded 0.33/0.5/0.05 fallback scale constants.
        total_rev = float(unified_df['revenue'].sum())
        if total_rev > 0:
            for dim_type, col in [("channel", "channel"), ("campaign_type", "campaign_type"), ("campaign", "campaign_name")]:
                entity_shares = (unified_df.groupby(col)['revenue'].sum() / total_rev)
                if len(entity_shares) > 0:
                    self.dimension_fallback_scale[dim_type] = float(entity_shares.median())
        logger.info(f"Dimension fallback scales (median historical revenue share): {self.dimension_fallback_scale}")

        logger.info(
            f"Complete ForecastIQ Ensemble Architecture successfully trained. "
            f"Channels={self.trained_channels} CampaignTypes={self.trained_campaign_types}"
        )

    def save_models(self) -> Path:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "models": self.models,
            "residuals_std": self.residuals_std,
            "is_trained": self.is_trained,
            "top_campaign_names": self.top_campaign_names,
            "trained_channels": self.trained_channels,
            "trained_campaign_types": self.trained_campaign_types,
            "feature_columns": self.feature_columns,
            "last_known_engineered": self.last_known_engineered,
            "dimension_fallback_scale": self.dimension_fallback_scale
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(artifact, f)
        logger.info(f"Trained model artifact serialized to {self.model_path}")
        return self.model_path

    def load_models(self) -> bool:
        if not self.model_path.exists():
            logger.warning(f"No trained model found at {self.model_path}. Please train models first.")
            return False
        try:
            with open(self.model_path, "rb") as f:
                artifact = pickle.load(f)
            self.models = artifact["models"]
            self.residuals_std = artifact["residuals_std"]
            self.is_trained = artifact["is_trained"]
            self.top_campaign_names = artifact.get("top_campaign_names", [])
            self.trained_channels = artifact.get("trained_channels", [])
            self.trained_campaign_types = artifact.get("trained_campaign_types", [])

            # Backward-compat: older pickles won't have trained_channels/trained_campaign_types.
            # Derive them from the dimension model keys themselves so old artifacts still work
            # without needing to retrain (BUG 10 fix must not require a forced retrain).
            if not self.trained_channels:
                self.trained_channels = sorted({
                    k[len("channel_"):-len("_revenue")]
                    for k in self.models if k.startswith("channel_") and k.endswith("_revenue")
                })
            if not self.trained_campaign_types:
                self.trained_campaign_types = sorted({
                    k[len("ctype_"):-len("_revenue")]
                    for k in self.models if k.startswith("ctype_") and k.endswith("_revenue")
                })

            # BUG 1 fix: restore the feature schema the overall model was trained on. Old
            # pickles predating this fix won't have these keys — fall back to base time
            # features only, which matches what those older models were actually trained on.
            self.feature_columns = artifact.get("feature_columns", []) or list(self.BASE_TIME_FEATURES)
            self.last_known_engineered = artifact.get("last_known_engineered", {})
            # BUG 2 fix: restore real fallback scales; old pickles predating this fix fall back
            # to the previous hardcoded constants rather than crashing or silently using 0.
            self.dimension_fallback_scale = artifact.get("dimension_fallback_scale", {}) or {
                "channel": 0.33, "campaign_type": 0.5, "campaign": 0.05
            }

            logger.info(f"Successfully loaded trained ensemble from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load pickled model: {str(e)}")
            return False

    def _aggregate_probabilistic_sums(self, daily_preds: np.ndarray, std_daily: float, periods: int) -> Tuple[float, float, float]:
        p50_sum = float(np.sum(daily_preds))
        horizon_std = std_daily * (periods ** 0.75)
        p10_raw = p50_sum - 1.28 * horizon_std
        p90_sum = float(p50_sum + 1.28 * horizon_std)
        # P10 floor: max of 5% of P50 to avoid uninformative zero lower bounds
        p10_sum = max(p50_sum * 0.05, p10_raw)
        return max(0.0, p10_sum), max(0.0, p50_sum), max(0.0, p90_sum)

    def forecast_overall(self, start_date: pd.Timestamp) -> Dict[str, Dict[str, float]]:
        if not self.is_trained:
            raise RuntimeError("Ensemble forecaster must be trained or loaded before predicting.")

        future_df = self._prepare_future_features(start_date, periods=90, last_known_engineered=self.last_known_engineered)
        
        prophet_future = future_df[['date']].rename(columns={'date': 'ds'})
        p_prophet = self.models["overall_prophet"].predict(prophet_future)['yhat'].values
        
        feature_cols = self.feature_columns if self.feature_columns else self.BASE_TIME_FEATURES
        X_tree = future_df[feature_cols]
        p_xgb = self.models["overall_xgb"].predict(X_tree)
        p_lgb = self.models["overall_lgb"].predict(X_tree)
        p_cat = self.models["overall_cat"].predict(X_tree)

        daily_revenue_p50 = (
            self.weights["prophet"] * p_prophet + 
            self.weights["xgboost"] * p_xgb + 
            self.weights["lightgbm"] * p_lgb + 
            self.weights["catboost"] * p_cat
        )
        daily_revenue_p50 = np.clip(daily_revenue_p50, a_min=0.0, a_max=None)

        daily_spend = self.models["overall_spend"].predict(future_df[self.BASE_TIME_FEATURES])
        daily_spend = np.clip(daily_spend, a_min=1.0, a_max=None)
        std_rev = self.residuals_std.get("overall", 1000.0)

        results = {}
        for periods, window_label in [(30, "30_days"), (60, "60_days"), (90, "90_days")]:
            rev_p10, rev_p50, rev_p90 = self._aggregate_probabilistic_sums(daily_revenue_p50[:periods], std_rev, periods)
            spend_sum = float(np.sum(daily_spend[:periods]))
            
            roas_p50 = rev_p50 / spend_sum
            roas_p10 = rev_p10 / spend_sum
            roas_p90 = rev_p90 / spend_sum

            results[window_label] = {
                "Revenue_P10": rev_p10,
                "Revenue_P50": rev_p50,
                "Revenue_P90": rev_p90,
                "ROAS_P10": roas_p10,
                "ROAS_P50": roas_p50,
                "ROAS_P90": roas_p90,
                "Spend_Expected": spend_sum
            }

        return results

    def forecast_dimension(self, dimension_type: str, dimension_key: str, start_date: pd.Timestamp) -> Dict[str, Dict[str, float]]:
        prefix = "channel" if dimension_type == "channel" else ("ctype" if dimension_type == "campaign_type" else "camp")
        rev_model_key = f"{prefix}_{dimension_key}_revenue"
        sp_model_key = f"{prefix}_{dimension_key}_spend"

        # Universal Fallback if specific entity model was not seen during training
        if rev_model_key not in self.models:
            ov_rev = self.forecast_overall(start_date)
            # BUG 2 fix: previously a hardcoded 0.33/0.5/0.05 scale with no statistical basis.
            # Now uses the real median historical revenue share for this dimension_type,
            # computed once during train_dimension_models() from actual data. Falls back to the
            # old constants only if dimension_fallback_scale is somehow empty (e.g. a pickle from
            # before this fix that also predates the backward-compat default in load_models()).
            _default_scale = {"channel": 0.33, "campaign_type": 0.5, "campaign": 0.05}
            scale = self.dimension_fallback_scale.get(dimension_type, _default_scale.get(dimension_type, 0.1))
            
            results = {}  # FIX: initialize before loop
            for win, base in ov_rev.items():
                # Original values from overall forecast
                orig_p10 = base["Revenue_P10"]
                orig_p50 = base["Revenue_P50"]
                orig_p90 = base["Revenue_P90"]
                orig_spend = base["Spend_Expected"]
                
                # Scale the point estimate (P50) as before
                new_p50 = orig_p50 * scale
                
                # Widen the uncertainty band: double the spread
                orig_spread = orig_p90 - orig_p10
                new_spread = 2 * orig_spread
                
                # Calculate new P10 and P90 with widened spread around the scaled P50
                new_p10 = new_p50 - (new_spread / 2)
                new_p90 = new_p50 + (new_spread / 2)
                
                # Ensure non-negative values
                new_p10 = max(0.0, new_p10)
                new_p90 = max(0.0, new_p90)
                
                # Scale spend as before (expected spend)
                new_spend = orig_spend * scale
                
                # Calculate ROAS values based on new revenue and spend
                # Using the same approach as forecast_overall: ROAS = Revenue / Spend_Expected
                new_roas_p10 = new_p10 / new_spend if new_spend > 0 else 0.0
                new_roas_p50 = new_p50 / new_spend if new_spend > 0 else 0.0
                new_roas_p90 = new_p90 / new_spend if new_spend > 0 else 0.0
                
                results[win] = {
                    "Revenue_P10": new_p10,
                    "Revenue_P50": new_p50,
                    "Revenue_P90": new_p90,
                    "ROAS_P10": new_roas_p10,
                    "ROAS_P50": new_roas_p50,
                    "ROAS_P90": new_roas_p90,
                    "Spend_Expected": new_spend
                }
            return results
         
        # Normal case: specific model exists
        daily_rev = self.models[rev_model_key].predict(self._prepare_future_features(start_date, periods=90)[['month', 'quarter', 'week', 'day_of_week', 'is_weekend', 'season_encoded']])
        daily_rev = np.clip(daily_rev, a_min=0.0, a_max=None)
        
        daily_sp = self.models[sp_model_key].predict(self._prepare_future_features(start_date, periods=90)[['month', 'quarter', 'week', 'day_of_week', 'is_weekend', 'season_encoded']])
        daily_sp = np.clip(daily_sp, a_min=0.1, a_max=None)
        
        std_rev = self.residuals_std.get(f"{prefix}_{dimension_key}", 500.0)

        results = {}
        for periods, window_label in [(30, "30_days"), (60, "60_days"), (90, "90_days")]:
            rev_p10, rev_p50, rev_p90 = self._aggregate_probabilistic_sums(daily_rev[:periods], std_rev, periods)
            spend_sum = float(np.sum(daily_sp[:periods]))
            
            roas_p50 = rev_p50 / spend_sum
            roas_p10 = rev_p10 / spend_sum
            roas_p90 = rev_p90 / spend_sum

            results[window_label] = {
                "Revenue_P10": rev_p10,
                "Revenue_P50": rev_p50,
                "Revenue_P90": rev_p90,
                "ROAS_P10": roas_p10,
                "ROAS_P50": roas_p50,
                "ROAS_P90": roas_p90,
                "Spend_Expected": spend_sum
            }

        return results

    def produce_full_predictions_table(self, start_date: pd.Timestamp) -> pd.DataFrame:
        rows = []

        ov_res = self.forecast_overall(start_date)
        for win, metrics in ov_res.items():
            rows.append({
                "forecast_period": win, "dimension_type": "Overall", "dimension_value": "Total Portfolio",
                "metric": "Revenue", "p10": metrics["Revenue_P10"], "p50": metrics["Revenue_P50"], "p90": metrics["Revenue_P90"]
            })
            rows.append({
                "forecast_period": win, "dimension_type": "Overall", "dimension_value": "Total Portfolio",
                "metric": "ROAS", "p10": metrics["ROAS_P10"], "p50": metrics["ROAS_P50"], "p90": metrics["ROAS_P90"]
            })

        # BUG 10 fix: use the real, trained channel list instead of a hardcoded one.
        channels = self.trained_channels if self.trained_channels else ["Google Ads", "Meta Ads", "Bing Ads"]
        for ch in channels:
            ch_res = self.forecast_dimension("channel", ch, start_date)
            for win, metrics in ch_res.items():
                rows.append({
                    "forecast_period": win, "dimension_type": "Channel", "dimension_value": ch,
                    "metric": "Revenue", "p10": metrics["Revenue_P10"], "p50": metrics["Revenue_P50"], "p90": metrics["Revenue_P90"]
                })
                rows.append({
                    "forecast_period": win, "dimension_type": "Channel", "dimension_value": ch,
                    "metric": "ROAS", "p10": metrics["ROAS_P10"], "p50": metrics["ROAS_P50"], "p90": metrics["ROAS_P90"]
                })

        # BUG 10 fix (CRITICAL): previously hardcoded to ["SEARCH", "SOCIAL"], silently dropping
        # PERFORMANCE_MAX (66% of Google Ads spend), SHOPPING, VIDEO, DEMAND_GEN, and Bing's
        # PerformanceMax/Audience types from the scored output file, despite trained models for
        # all of them already existing in model.pkl (see ctype_PERFORMANCE_MAX_revenue etc.).
        ctypes = self.trained_campaign_types if self.trained_campaign_types else ["SEARCH", "SOCIAL"]
        for ct in ctypes:
            ct_res = self.forecast_dimension("campaign_type", ct, start_date)
            for win, metrics in ct_res.items():
                rows.append({
                    "forecast_period": win, "dimension_type": "CampaignType", "dimension_value": ct,
                    "metric": "Revenue", "p10": metrics["Revenue_P10"], "p50": metrics["Revenue_P50"], "p90": metrics["Revenue_P90"]
                })
                rows.append({
                    "forecast_period": win, "dimension_type": "CampaignType", "dimension_value": ct,
                    "metric": "ROAS", "p10": metrics["ROAS_P10"], "p50": metrics["ROAS_P50"], "p90": metrics["ROAS_P90"]
                })

        top_camps = self.top_campaign_names if self.top_campaign_names else ["Search_Campaign_01", "Generic_Campaign_02", "Search_TM_Campaign_01"]
        for camp in top_camps[:10]:
            camp_res = self.forecast_dimension("campaign", camp, start_date)
            for win, metrics in camp_res.items():
                rows.append({
                    "forecast_period": win, "dimension_type": "Campaign", "dimension_value": camp,
                    "metric": "Revenue", "p10": metrics["Revenue_P10"], "p50": metrics["Revenue_P50"], "p90": metrics["Revenue_P90"]
                })
                rows.append({
                    "forecast_period": win, "dimension_type": "Campaign", "dimension_value": camp,
                    "metric": "ROAS", "p10": metrics["ROAS_P10"], "p50": metrics["ROAS_P50"], "p90": metrics["ROAS_P90"]
                })

        master_df = pd.DataFrame(rows)
        master_df["p10"] = master_df["p10"].round(2)
        master_df["p50"] = master_df["p50"].round(2)
        master_df["p90"] = master_df["p90"].round(2)
        return master_df