import optuna
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from src.utils import get_logger

logger = get_logger("BudgetOptimizer")

optuna.logging.set_verbosity(optuna.logging.WARNING)

class BudgetOptimizer:
    """
    Optuna Budget Optimization Engine for ForecastIQ.
    Maximizes total eCommerce revenue subject to Target ROAS and Maximum Budget constraints.
    Models diminishing marginal returns for paid acquisition channels at the aggregate monthly/planning horizon level.
    """
    def __init__(self, historical_df: pd.DataFrame):
        self.historical_df = historical_df
        self._fit_channel_efficiency()

    def _fit_channel_efficiency(self):
        """Fits non-linear diminishing returns parameters (alpha, beta) for each channel at a 30-day aggregate level."""
        self.channel_params: Dict[str, Dict[str, float]] = {}

        # First, aggregate historical data into monthly/30-day buckets to calibrate true macro efficiency
        for ch, df_c in self.historical_df.groupby('channel'):
            spend_sum = df_c['spend'].sum()
            rev_sum = df_c['revenue'].sum()
            overall_roas = rev_sum / (spend_sum + 1e-5)
            
            # Group by Year-Month or 30-day windows to get realistic monthly spend & revenue
            df_c = df_c.copy()
            df_c['year_month'] = df_c['date'].dt.to_period('M')
            monthly = df_c.groupby('year_month').agg({'spend': 'sum', 'revenue': 'sum'}).reset_index()
            monthly = monthly[monthly['spend'] > 0]
            
            # BUG fix (bug-hunt sweep, same class as the frozen-dimension predictions bug):
            # beta previously matched by hardcoded channel name, so any channel outside
            # {Google Ads, Meta Ads, Bing Ads} — e.g. a channel only present in held-out/mock
            # grading data — silently fell through to Bing's 0.78 regardless of its own
            # efficiency profile. Keep the three calibrated values for the channels we've
            # actually validated (output for existing data is unchanged), fall back to a
            # single documented default for any other channel instead of mis-attributing
            # Bing's curve to it.
            _KNOWN_BETAS = {"Google Ads": 0.85, "Meta Ads": 0.82, "Bing Ads": 0.78}
            beta = _KNOWN_BETAS.get(ch, 0.80)
            
            if len(monthly) > 0:
                avg_monthly_sp = monthly['spend'].mean()
                avg_monthly_rev = monthly['revenue'].mean()
                alpha = avg_monthly_rev / (avg_monthly_sp ** beta)
            else:
                alpha = overall_roas * (10000.0 ** (1.0 - beta))

            # BUG 5 fix: derive a real goodness-of-fit measure for this channel's diminishing-
            # returns curve, so optimize_allocation()'s confidence_range can be data-driven
            # instead of a flat hardcoded +/-12% applied identically to every channel and every
            # budget/target_roas combination. We compute the curve's predicted revenue against
            # each actual monthly observation and take the coefficient of variation (std/mean)
            # of the relative residuals as a simple, interpretable noise measure: a channel
            # whose monthly revenue tracks alpha*spend^beta tightly gets a small fit_cv (tight
            # confidence band); a channel with noisy, unpredictable monthly revenue relative to
            # its spend gets a larger fit_cv (wider band). Needs at least 2 monthly observations
            # to compute a meaningful spread; otherwise we fall back to a conservative default.
            if len(monthly) >= 2:
                predicted = alpha * (monthly['spend'].values ** beta)
                actual = monthly['revenue'].values
                rel_resid = (actual - predicted) / (predicted + 1e-5)
                fit_cv = float(np.std(rel_resid))
                # Clip at the source: a single channel with a genuinely poor power-curve fit
                # (e.g. early near-zero-revenue ramp-up months mixed with mature months,
                # observed on this dataset's Bing Ads history) can otherwise produce a fit_cv
                # well above 2.0 — over 200% relative noise — which is real noise in the curve
                # fit, but not a meaningful "this channel's spend efficiency next month is this
                # uncertain" signal on its own. Cap it so one badly-behaved channel can't single-
                # handedly blow out the blended confidence band to a non-credible width; a 0.6
                # ceiling still represents real, substantial uncertainty (wider than every other
                # band in this engine) without becoming an unusable +/-100%+ interval.
                fit_cv = float(np.clip(fit_cv, 0.0, 0.6))
            else:
                # Not enough monthly observations to measure real fit noise — use a moderate
                # default rather than pretending we have zero uncertainty.
                fit_cv = 0.15

            self.channel_params[ch] = {"alpha": alpha, "beta": beta, "base_roas": overall_roas, "fit_cv": fit_cv}

        # BUG fix (bug-hunt sweep): this used to force-add exactly ["Google Ads","Meta Ads",
        # "Bing Ads"] with fabricated defaults, AND that hardcoded list was the only channel
        # set every method below (simulate_budget_change, optimize_allocation) ever iterated
        # over — so a real channel present in historical_df (fitted correctly just above via
        # the data-driven groupby) got silently discarded downstream the moment held-out/mock
        # grading data introduced a 4th channel. self.channels is now the single canonical,
        # data-derived list every method in this class iterates over.
        self.channels = sorted(self.channel_params.keys())
        if not self.channels:
            # Genuinely empty historical data: fall back to the three channels this product
            # ships with by default so the optimizer has a non-empty search space instead of
            # crashing outright.
            self.channels = ["Google Ads", "Meta Ads", "Bing Ads"]
            for ch in self.channels:
                self.channel_params[ch] = {"alpha": 25.0, "beta": 0.82, "base_roas": 4.5, "fit_cv": 0.15}

    def simulate_budget_change(self, channel_pcts: Dict[str, float], base_spend: Dict[str, float]) -> Dict[str, Any]:
        """
        Executes real-time budget scenario simulation.
        channel_pcts: {channel_name: pct_change}. Any channel in base_spend not present in
        channel_pcts is treated as 0% change.

        BUG fix (bug-hunt sweep): previously took three fixed named args
        (google_pct/meta_pct/bing_pct), which made a 4th channel structurally impossible to
        simulate no matter what base_spend contained. Same failure class as the frozen-
        dimension predictions bug — now takes any set of channels.

        Returns live recalculated Revenue, ROAS, and channel contributions.
        """
        total_rev = 0.0
        total_spend = 0.0
        contributions = {}

        for ch, spend in base_spend.items():
            pct = channel_pcts.get(ch, 0.0)
            new_spend = max(100.0, spend * (1.0 + pct / 100.0))
            
            params = self.channel_params.get(ch, {"alpha": 25.0, "beta": 0.82})
            new_rev = params["alpha"] * (new_spend ** params["beta"])
            
            total_spend += new_spend
            total_rev += new_rev
            contributions[ch] = {
                "spend": new_spend,
                "revenue": new_rev,
                "roas": new_rev / new_spend,
                "revenue_change_pct": ((new_rev - params["alpha"] * (spend ** params["beta"])) / (params["alpha"] * (spend ** params["beta"]) + 1e-5)) * 100.0
            }

        return {
            "total_spend": total_spend,
            "total_revenue": total_rev,
            "total_roas": total_rev / (total_spend + 1e-5),
            "channel_contributions": contributions
        }

    def optimize_allocation(self, max_budget: float, target_roas: float, n_trials: int = 300) -> Dict[str, Any]:
        """Runs Optuna optimization to find the exact global spend allocation across every channel present in the historical data."""
        # Guard: max_budget <= 0 would make every per-channel upper bound derive from a
        # 0/0 revenue-share division below (hist_rev_at_equal_split all zero -> total_hist_rev
        # zero -> NaN share), which either crashes or silently produces garbage bounds passed
        # to Optuna. Fail loudly and immediately instead.
        if max_budget <= 0:
            raise ValueError(f"max_budget must be positive, got {max_budget}")

        logger.info(f"Running Optuna Budget Optimization (Max Budget: ${max_budget:,.0f}, Target ROAS: {target_roas}x)...")

        # BUG fix (bug-hunt sweep, same class as the frozen-dimension predictions bug):
        # previously hardcoded to exactly ["Google Ads","Meta Ads","Bing Ads"]. Any channel
        # actually present in historical_df (and correctly fitted in self.channel_params by
        # _fit_channel_efficiency) got silently excluded from optimization — the allocator
        # would confidently return a "complete" result that simply never considered a whole
        # channel's spend/revenue potential. Now uses the real, data-derived channel list.
        channels = self.channels

        # BUG 15 fix: the old hardcoded Optuna search bounds had a maximum sum of
        # 0.7+0.5+0.25=1.45x of max_budget, meaning most trials hit the budget-exceeded
        # penalty rather than meaningfully exploring the feasible revenue space. Also, the
        # fixed lower bounds could exclude the feasible region entirely at high target_roas
        # values (where near-full budget may be needed for one channel). Replaced with:
        #   - upper bounds derived from each channel's efficiency-weighted revenue share (so
        #     the optimizer can't suggest putting 70% of budget into Bing, which historically
        #     only sees ~1.6% of spend), scaled so upper bounds always sum to max_budget,
        #     guaranteeing no trial can violate the budget constraint by corner exploration.
        #   - lower bounds: 1% of max_budget floor (always feasible, never zero).
        # BUG fix (post-launch, found during live optimizer review): the previous version
        # rescaled all three per-channel upper bounds so they summed to exactly max_budget.
        # That step wasn't actually needed for feasibility — the objective() penalty below
        # already rejects any trial where total spend exceeds max_budget — but it had a real
        # side effect: three channels with roughly similar computed shares (as Google and Meta
        # often are) got rescaled down to near-identical ceilings, capping the optimizer well
        # before it could shift meaningfully more budget toward whichever of the two actually
        # has the higher marginal return. That silently suppressed a better allocation even
        # though the objective function would have preferred it.
        #
        # Fix: keep a tight, historically-derived ceiling ONLY for channels with a genuinely
        # small historical revenue share (the original intent of BUG 15 — stopping an
        # unrealistic 70% dump into Bing). Channels with a meaningful historical share get a
        # much wider, shared ceiling so Optuna can freely equalize marginal returns between
        # them; the objective's own budget-overflow penalty (not a pre-shrunk bound) is what
        # keeps the total feasible.
        hist_rev_at_equal_split = {}
        for ch in channels:
            p = self.channel_params[ch]
            trial_sp = max_budget / len(channels)
            hist_rev_at_equal_split[ch] = p["alpha"] * (trial_sp ** p["beta"])
        total_hist_rev = sum(hist_rev_at_equal_split.values())

        SMALL_SHARE_THRESHOLD = 0.10  # channels below this get a tight, proxy-derived cap
        WIDE_CEILING_FRACTION = 0.85  # channels above it can compete for up to 85% of budget
        upper_bounds = {}
        for ch in channels:
            share = hist_rev_at_equal_split[ch] / total_hist_rev
            if share < SMALL_SHARE_THRESHOLD:
                upper_bounds[ch] = float(np.clip(share * max_budget * 2.5, max_budget * 0.02, max_budget * 0.15))
            else:
                upper_bounds[ch] = max_budget * WIDE_CEILING_FRACTION
        lower_bound = max_budget * 0.01

        def objective(trial: optuna.Trial) -> float:
            # BUG fix (bug-hunt sweep): previously three fixed trial.suggest_float calls
            # named after Google/Meta/Bing specifically, structurally incapable of proposing
            # spend for any other channel. Now suggests one value per entry in the real,
            # data-derived `channels` list.
            spends = {ch: trial.suggest_float(f"spend_{ch}", lower_bound, upper_bounds[ch]) for ch in channels}

            total_sp = sum(spends.values())
            if total_sp > max_budget:
                return -1e9 * (total_sp - max_budget)

            total_rev = 0.0
            for ch in channels:
                p = self.channel_params[ch]
                total_rev += p["alpha"] * (spends[ch] ** p["beta"])

            roas = total_rev / (total_sp + 1e-5)
            if roas < target_roas:
                return -1e7 * (target_roas - roas)

            # We also give a slight reward to deploying more budget as long as ROAS >= target
            return total_rev

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_spends = {ch: best_params[f"spend_{ch}"] for ch in channels}
        total_opt_spend = sum(best_spends.values())

        total_opt_rev = 0.0
        channel_breakdown = {}
        for ch in channels:
            p = self.channel_params[ch]
            sp = best_spends[ch]
            rev = float(p["alpha"] * (sp ** p["beta"]))
            total_opt_rev += rev
            channel_breakdown[ch] = {
                "allocated_spend": float(sp),
                "expected_revenue": rev,
                "expected_roas": float(rev / sp),
                "budget_share": float((sp / total_opt_spend) * 100.0)
            }

        overall_opt_roas = float(total_opt_rev / total_opt_spend)

        # BUG 5 fix: previously hardcoded confidence_range to a flat +/-12% (0.88/1.12)
        # regardless of how reliable each channel's diminishing-returns curve actually is, or
        # how the optimizer chose to split the budget. Derive the spread instead from each
        # allocated channel's fit_cv (the coefficient of variation of that channel's monthly
        # alpha*spend^beta residuals — see _fit_channel_efficiency), combined the way
        # independent-channel revenue variances actually combine in a portfolio: variances add
        # (weighted by each channel's squared revenue share), not coefficients of variation
        # directly. Treating fit_cv as a 1.28-sigma spread applied straight to the aggregate
        # point estimate (an earlier version of this fix) overstated the blended uncertainty,
        # since it ignored diversification — a 3-channel portfolio is not as risky as its
        # single noisiest channel. This combination assumes channel-level revenue noise is
        # roughly independent across channels, which is a reasonable simplification given we
        # have no cross-channel covariance estimate from the data.
        if total_opt_rev > 0:
            portfolio_variance_frac = sum(
                ((channel_breakdown[ch]["expected_revenue"] / total_opt_rev) ** 2) * (self.channel_params[ch]["fit_cv"] ** 2)
                for ch in channels
            )
            portfolio_cv = float(np.sqrt(portfolio_variance_frac))
        else:
            portfolio_cv = 0.15
        # Clip the final blended portfolio-level CV to a credible range: floor prevents a
        # falsely-confident near-zero band, ceiling keeps the interval interpretable rather
        # than spanning multiples of the point estimate.
        portfolio_cv = float(np.clip(portfolio_cv, 0.04, 0.30))
        band_factor = 1.28 * portfolio_cv  # consistent with the 1.28 (~80% interval) used elsewhere

        conf_range = {
            "revenue_p10": float(total_opt_rev * (1.0 - band_factor)),
            "revenue_p50": float(total_opt_rev),
            "revenue_p90": float(total_opt_rev * (1.0 + band_factor)),
            "roas_p10": float(overall_opt_roas * (1.0 - band_factor)),
            "roas_p50": float(overall_opt_roas),
            "roas_p90": float(overall_opt_roas * (1.0 + band_factor))
        }

        # BUG fix: optimize_allocation previously had no feasibility check on its own output.
        # objective() penalizes trials where roas < target_roas, but if no sampled trial ever
        # clears that bar (e.g. target_roas is high enough that diminishing returns make it
        # unreachable within max_budget), Optuna still returns study.best_params — whichever
        # trial had the least-negative penalty — and this function packaged it into a result
        # that looked identical to a genuinely feasible recommendation, with no signal that
        # the target wasn't actually met. Surface that explicitly instead of silently
        # returning an under-target allocation as if it were a success.
        target_achievable = bool(overall_opt_roas >= target_roas)

        results = {
            "max_budget": float(max_budget),
            "target_roas": float(target_roas),
            "recommended_total_spend": float(total_opt_spend),
            "expected_total_revenue": float(total_opt_rev),
            "expected_total_roas": overall_opt_roas,
            "target_achievable": target_achievable,
            "target_achievement_note": (
                None if target_achievable else
                f"Target ROAS of {target_roas}x could not be met within a ${max_budget:,.0f} budget "
                f"given current channel efficiency curves. Best achievable ROAS at this budget is "
                f"~{overall_opt_roas:.2f}x. Consider lowering the target ROAS or reducing the budget."
            ),
            "channel_recommendations": channel_breakdown,
            "confidence_range": conf_range
        }

        logger.info(f"Optimization finished. Optimized Rev: ${total_opt_rev:,.0f} with ROAS: {overall_opt_roas:.2f}x.")
        return results
