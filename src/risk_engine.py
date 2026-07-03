import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.utils import get_logger

logger = get_logger("RiskIntelligenceEngine")

class RiskIntelligenceEngine:
    """
    Elite Production Risk Intelligence Engine for ForecastIQ.
    Computes an overarching Risk Score (0-100) by analyzing Revenue Volatility,
    Channel Dependency, ROAS Instability, and Data Quality Issues.
    Returns executive Risk Classification (Low Risk, Medium Risk, High Risk) and tactical mitigation strategies.
    """
    def __init__(self, historical_df: pd.DataFrame, data_quality_score: float):
        self.historical_df = historical_df
        self.data_quality_score = data_quality_score

    def evaluate_risk(self) -> Dict[str, Any]:
        logger.info("Executing Enterprise Risk Intelligence Evaluation...")
        
        # 1. Revenue Volatility Score — weekly granularity (daily is too noisy for ad spend)
        weekly_rev = self.historical_df.groupby(pd.Grouper(key='date', freq='W'))['revenue'].sum()
        weekly_rev = weekly_rev[weekly_rev > 0]
        cv_rev = float(weekly_rev.std() / (weekly_rev.mean() + 1e-5))
        rev_vol_score = min(100.0, max(0.0, cv_rev * 100.0))

        # 2. Channel Dependency Score (0-100)
        # Herfindahl-Hirschman Index (HHI) of channel revenue concentration
        ch_sums = self.historical_df.groupby('channel')['revenue'].sum()
        ch_shares = ch_sums / ch_sums.sum()
        hhi = float((ch_shares ** 2).sum())
        # HHI of 0.33 (perfect 3-way split) = Score 20. HHI of 1.0 (100% one channel) = Score 100.
        ch_dep_score = min(100.0, max(0.0, (hhi - 0.33) * 150.0))

        # BUG fix (P3): top-channel name/share used to be hardcoded to "Google Ads" / "50%"
        # in the mitigation text below regardless of which channel actually dominates. Derive
        # it from the real ch_shares computed above so the copy stays correct if a different
        # channel becomes the leader (e.g. on a different dataset or after reallocation).
        top_channel = str(ch_shares.idxmax())
        top_channel_share_pct = round(float(ch_shares.max()) * 100.0, 1)

        # 3. ROAS Instability Score — weekly granularity
        weekly_agg = self.historical_df.groupby(pd.Grouper(key='date', freq='W')).agg(
            revenue=('revenue', 'sum'), spend=('spend', 'sum')
        )
        weekly_roas = weekly_agg['revenue'] / (weekly_agg['spend'] + 1e-5)
        weekly_roas = weekly_roas[weekly_roas > 0]
        cv_roas = float(weekly_roas.std() / (weekly_roas.mean() + 1e-5))
        roas_inst_score = min(100.0, max(0.0, cv_roas * 120.0))

        # 4. Data Quality Issues Score (0-100)
        # Inversely proportional to Data Quality Score
        data_risk_score = 100.0 - self.data_quality_score

        # Overarching Risk Score (Weighted Average)
        weights = {
            "revenue_volatility": 0.30,
            "channel_dependency": 0.25,
            "roas_instability": 0.30,
            "data_quality_issues": 0.15
        }
        overall_risk_score = (
            weights["revenue_volatility"] * rev_vol_score +
            weights["channel_dependency"] * ch_dep_score +
            weights["roas_instability"] * roas_inst_score +
            weights["data_quality_issues"] * data_risk_score
        )
        overall_risk_score = round(float(overall_risk_score), 1)

        # Risk Classification Badge
        if overall_risk_score < 35.0:
            risk_level = "Low Risk"
            badge_color = "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
            summary_text = "Highly dependable portfolio with predictable daily cash flows, diversified channels, and excellent data fidelity."
        elif overall_risk_score < 65.0:
            risk_level = "Medium Risk"
            badge_color = "bg-amber-500/10 text-amber-500 border-amber-500/20"
            summary_text = f"Stable overarching performance with moderate sensitivity to seasonal CPC spikes and slight channel concentration on {top_channel}."
        else:
            risk_level = "High Risk"
            badge_color = "bg-rose-500/10 text-rose-500 border-rose-500/20"
            summary_text = "Elevated portfolio exposure driven by severe day-to-day ROAS variance and heavy dependence on single campaign auctions."

        # Actionable Mitigation Factors
        factors_breakdown = [
            {
                "name": "Revenue Volatility",
                "score": round(rev_vol_score, 1),
                "status": "Healthy" if rev_vol_score < 40 else "Monitor",
                "impact": "High",
                "mitigation": "Utilize automated pacing tools to smooth out mid-week revenue depressions and maintain consistent top-of-funnel impression share."
            },
            {
                "name": "Channel Dependency",
                "score": round(ch_dep_score, 1),
                "status": "Concentrated" if ch_dep_score > 50 else "Diversified",
                "impact": "Medium",
                "mitigation": f"{top_channel} represents {top_channel_share_pct}% of revenue capture. " + (
                    "Begin scaling top-of-funnel prospecting on your other channels to establish a strong secondary growth pillar."
                    if ch_dep_score > 50 else
                    "Concentration is within a healthy range; continue monitoring as spend allocation shifts."
                )
            },
            {
                "name": "ROAS Instability",
                "score": round(roas_inst_score, 1),
                "status": "Volatile" if roas_inst_score > 50 else "Stable",
                "impact": "High",
                "mitigation": "Enforce rigorous Target CPA / Target ROAS bid floor rules across all Meta generic and non-brand campaigns to prevent runaway spend."
            },
            {
                "name": "Data Quality Audit",
                "score": round(data_risk_score, 1),
                "status": "Excellent" if data_risk_score < 10 else "Action Required",
                "impact": "Low",
                "mitigation": f"Validation engine verified a {self.data_quality_score}/100 schema score. Ensure Bing campaign UTM tracking tags are consistently appended."
            }
        ]

        logger.info(f"Risk Evaluation finished. Overall Classification: {risk_level} (Score: {overall_risk_score}/100).")
        return {
            "risk_score": overall_risk_score,
            "risk_classification": risk_level,
            "badge_color": badge_color,
            "executive_risk_summary": summary_text,
            "risk_factors": factors_breakdown
        }
