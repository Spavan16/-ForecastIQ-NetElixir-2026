import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from src.utils import get_logger

logger = get_logger("MonteCarloSimulation")


def derive_revenue_volatility(residual_std_daily: Optional[float], base_revenue: float, horizon_days: int = 30,
                               floor: float = 0.03, ceiling: float = 0.60, fallback: float = 0.15) -> float:
    """
    Shared volatility-fraction derivation used by both MonteCarloSimulator (BUG 4) and
    ScenarioGenerator (BUG 6), so the two engines stay consistent with each other instead of
    each guessing its own static assumption. Aggregates a daily residual std to the given
    horizon (period ** 0.75, matching _aggregate_probabilistic_sums()) and expresses it as a
    fraction of the baseline revenue.
    """
    if residual_std_daily is not None and base_revenue and base_revenue > 0:
        horizon_std = residual_std_daily * (horizon_days ** 0.75)
        return float(np.clip(horizon_std / base_revenue, floor, ceiling))
    logger.warning("No residual_std_daily/base_revenue available for volatility derivation; "
                    f"falling back to static {fallback:.0%} assumption.")
    return fallback

class MonteCarloSimulator:
    """
    Elite Production Monte Carlo Engine for ForecastIQ.
    Executes 10,000 rigorous stochastic paths to model exact revenue, ROAS, 
    and channel distribution uncertainties. Produces Worst, Expected, and Best Case intelligence.
    """
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations

    def run_portfolio_simulation(self, base_revenue_30d: float, base_spend_30d: float,
                                 channel_splits: Dict[str, float], revenue_volatility: Optional[float] = None,
                                 residual_std_daily: Optional[float] = None, horizon_days: int = 30) -> Dict[str, Any]:
        logger.info(f"Executing {self.n_simulations} Monte Carlo stochastic paths...")
        np.random.seed(42)

        # BUG 4 fix: previously hardcoded revenue_volatility=0.15 regardless of how volatile
        # the portfolio actually is, so the simulation's spread had no connection to the
        # ensemble's own residuals_std (the same number forecast_overall() uses for its
        # P10/P50/P90 bands). A high-risk, high-variance portfolio and a low-risk, stable one
        # produced identically-shaped distributions. Derive volatility from the model's actual
        # daily residual std instead, scaled to the horizon and expressed as a fraction of the
        # P50 baseline, so a genuinely noisier revenue history produces a genuinely wider
        # Monte Carlo spread. Callers can still pass revenue_volatility explicitly to override
        # (e.g. for unit tests), in which case that explicit value wins.
        if revenue_volatility is None:
            # BUG 4 fix: delegates to the shared derive_revenue_volatility() helper (also used
            # by ScenarioGenerator for BUG 6) instead of a private inline calculation, so both
            # engines derive volatility the same way from the ensemble's actual residuals.
            revenue_volatility = derive_revenue_volatility(residual_std_daily, base_revenue_30d, horizon_days)

        # We simulate 10,000 possible revenue realizations using a normal distribution around
        # base expectations, with spread now derived from the ensemble's actual residual
        # variance rather than a fixed assumption (see fix note above).
        # BUG fix: previously used np.clip() as a hard boundary, which doesn't discard
        # out-of-range draws — it snaps every one of them to exactly the floor/ceiling value.
        # At realistic (higher) volatility levels this piles up a large fraction of the left
        # tail onto a single point, producing an artificial spike in the revenue histogram
        # instead of a smooth distribution tail. Resample out-of-bounds values instead, which
        # preserves the true shape of the tail near the bounds.
        rev_lo, rev_hi = base_revenue_30d * 0.4, base_revenue_30d * 2.5
        sim_revenues = np.random.normal(loc=base_revenue_30d, scale=base_revenue_30d * revenue_volatility, size=self.n_simulations)
        rev_mask = (sim_revenues < rev_lo) | (sim_revenues > rev_hi)
        while rev_mask.any():
            sim_revenues[rev_mask] = np.random.normal(loc=base_revenue_30d, scale=base_revenue_30d * revenue_volatility, size=int(rev_mask.sum()))
            rev_mask = (sim_revenues < rev_lo) | (sim_revenues > rev_hi)

        # We simulate spend with slightly lower volatility (since budgets are generally controlled)
        spend_lo, spend_hi = base_spend_30d * 0.7, base_spend_30d * 1.4
        sim_spends = np.random.normal(loc=base_spend_30d, scale=base_spend_30d * (revenue_volatility * 0.4), size=self.n_simulations)
        spend_mask = (sim_spends < spend_lo) | (sim_spends > spend_hi)
        while spend_mask.any():
            sim_spends[spend_mask] = np.random.normal(loc=base_spend_30d, scale=base_spend_30d * (revenue_volatility * 0.4), size=int(spend_mask.sum()))
            spend_mask = (sim_spends < spend_lo) | (sim_spends > spend_hi)

        sim_roas = sim_revenues / sim_spends

        # Channel distributions
        sim_channels = {}
        for ch, share in channel_splits.items():
            ch_revs = sim_revenues * np.random.normal(loc=share, scale=share * 0.08, size=self.n_simulations)
            sim_channels[ch] = {
                "worst_case": float(np.percentile(ch_revs, 10)),
                "expected_case": float(np.median(ch_revs)),
                "best_case": float(np.percentile(ch_revs, 90))
            }

        # Produce complete uncertainty profile
        results = {
            "n_simulations": self.n_simulations,
            "worst_case_revenue": float(np.percentile(sim_revenues, 10)),
            "expected_revenue": float(np.median(sim_revenues)),
            "best_case_revenue": float(np.percentile(sim_revenues, 90)),
            "worst_case_roas": float(np.percentile(sim_roas, 10)),
            "expected_roas": float(np.median(sim_roas)),
            "best_case_roas": float(np.percentile(sim_roas, 90)),
            "channel_distributions": sim_channels,
            # Generate histogram distribution data for frontend Recharts rendering
            "revenue_histogram": self._generate_histogram(sim_revenues, bins=25),
            "roas_histogram": self._generate_histogram(sim_roas, bins=25)
        }

        logger.info("Monte Carlo Risk Simulation successfully completed.")
        return results

    def _generate_histogram(self, data: np.ndarray, bins: int) -> List[Dict[str, Any]]:
        counts, bin_edges = np.histogram(data, bins=bins)
        hist = []
        for i in range(len(counts)):
            hist.append({
                "bin_center": float((bin_edges[i] + bin_edges[i+1]) / 2),
                "bin_min": float(bin_edges[i]),
                "bin_max": float(bin_edges[i+1]),
                "frequency": int(counts[i])
            })
        return hist
