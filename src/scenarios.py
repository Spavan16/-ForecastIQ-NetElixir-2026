import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.utils import get_logger

logger = get_logger("ScenarioGenerator")

class ScenarioGenerator:
    """
    Scenario Generator for ForecastIQ.
    Generates 7 core eCommerce business scenarios and recalculates multi-horizon revenue & ROAS forecasts.
    """
    def __init__(self, base_predictions_30d: Dict[str, float], base_predictions_60d: Dict[str, float], base_predictions_90d: Dict[str, float], base_revenue_volatility: float = 0.15):
        self.base_forecasts = {
            "30_days": base_predictions_30d,
            "60_days": base_predictions_60d,
            "90_days": base_predictions_90d
        }
        # BUG 6 fix: previously every scenario's P10/P90 band was a flat +/-15% regardless of
        # how extreme the scenario is. base_revenue_volatility is the same residual-derived
        # fraction Monte Carlo (BUG 4) now uses, passed in by the caller instead of guessed here.
        self.base_revenue_volatility = base_revenue_volatility

    def generate_all_scenarios(self) -> List[Dict[str, Any]]:
        logger.info("Generating 7 Core eCommerce Strategic Scenarios...")

        scenarios_config = [
            {
                "id": "expected", "name": "Expected Case (Baseline)",
                "description": "Business as usual modeled from P50 historical trajectory.",
                "cpc_change": "0%", "conv_rate_change": "0%", "revenue_multiplier": 1.0, "roas_multiplier": 1.0, "tag": "Baseline"
            },
            {
                "id": "conservative", "name": "Conservative Strategic Plan",
                "description": "Accounts for slight macroeconomic softening and lower auction win rates.",
                "cpc_change": "+5%", "conv_rate_change": "-8%", "revenue_multiplier": 0.88, "roas_multiplier": 0.85, "tag": "Low Risk"
            },
            {
                "id": "aggressive", "name": "Aggressive Scale & Capture",
                "description": "Simulates aggressive top-of-funnel prospecting and market share expansion.",
                "cpc_change": "+15%", "conv_rate_change": "+5%", "revenue_multiplier": 1.25, "roas_multiplier": 1.08, "tag": "High Growth"
            },
            {
                "id": "recession", "name": "Recessionary Slump",
                "description": "Severe consumer spending contraction and prolonged consideration cycles.",
                "cpc_change": "-10%", "conv_rate_change": "-20%", "revenue_multiplier": 0.75, "roas_multiplier": 0.80, "tag": "Warning"
            },
            {
                "id": "holiday", "name": "Q4 Holiday Season Multiplier",
                "description": "Sustained seasonal demand surge accompanied by elevated CPM/CPC ad inflation.",
                "cpc_change": "+25%", "conv_rate_change": "+30%", "revenue_multiplier": 1.45, "roas_multiplier": 1.16, "tag": "Seasonal"
            },
            {
                "id": "black_friday", "name": "Black Friday / Cyber Week Blitz",
                "description": "Extreme high-intent conversion spikes over compressed auction windows.",
                "cpc_change": "+50%", "conv_rate_change": "+75%", "revenue_multiplier": 1.85, "roas_multiplier": 1.25, "tag": "Peak Event"
            },
            {
                "id": "high_competition", "name": "Aggressive Competitor Bidding",
                "description": "Direct competitor conquesting inflating search CPCs and eroding impression share.",
                "cpc_change": "+35%", "conv_rate_change": "-10%", "revenue_multiplier": 0.90, "roas_multiplier": 0.72, "tag": "Threat"
            }
        ]

        scenario_results = []
        for s in scenarios_config:
            # BUG 6 fix: band width now scales with how far this scenario's multiplier deviates
            # from the 1.0 baseline. Black Friday (1.85x) and Recession (0.75x) are genuine
            # extrapolations beyond what the model has typically seen and carry real additional
            # uncertainty; Conservative/Aggressive are modest, closer-to-historical deviations.
            # Floor/ceiling keep the band credible (never falsely tight, never absurdly wide).
            extremeness = abs(s["revenue_multiplier"] - 1.0)
            band_pct = float(np.clip(self.base_revenue_volatility * (1.0 + 2.0 * extremeness), 0.06, 0.45))

            simulated_windows = {}
            for win, base in self.base_forecasts.items():
                rev_p50 = base.get("Revenue_P50", 250000.0) * s["revenue_multiplier"]
                roas_p50 = base.get("ROAS_P50", 4.5) * s["roas_multiplier"]
                
                # We compute realistic P10 and P90 uncertainty bands around the modified P50,
                # scaled per-scenario via band_pct instead of a flat 0.15 (see BUG 6 fix above).
                simulated_windows[win] = {
                    "Revenue_P10": float(rev_p50 * (1.0 - band_pct)),
                    "Revenue_P50": float(rev_p50),
                    "Revenue_P90": float(rev_p50 * (1.0 + band_pct)),
                    "ROAS_P10": float(roas_p50 * (1.0 - band_pct)),
                    "ROAS_P50": float(roas_p50),
                    "ROAS_P90": float(roas_p50 * (1.0 + band_pct)),
                    "Spend_Expected": float(base.get("Spend_Expected", 50000.0) * (s["revenue_multiplier"] / s["roas_multiplier"]))
                }

            s_copy = s.copy()
            s_copy["forecasts"] = simulated_windows
            scenario_results.append(s_copy)

        logger.info("Scenario Generation successfully completed.")
        return scenario_results
