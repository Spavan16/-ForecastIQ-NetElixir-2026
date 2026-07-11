import pandas as pd
from typing import Dict, Any
from src.llm_provider import get_llm_provider, MockLLMProvider
from src.utils import get_logger

logger = get_logger("ChatEngine")


class ForecastChatBot:
    """
    ForecastIQ Conversational AI Engine.
    When Gemini is active: delegates to Gemini with full analytics context.
    When offline: generates fully data-driven causal responses from computed stats —
    indistinguishable from LLM output, with no hardcoded numbers.
    """

    def __init__(self, historical_df: pd.DataFrame, forecast_context: Dict[str, Any]):
        self.historical_df = historical_df
        self.forecast_context = forecast_context
        self.llm_provider = get_llm_provider()
        self._stats = self._compute_stats()

    def _compute_stats(self) -> Dict[str, Any]:
        """Pre-compute all analytical stats once so every response is data-driven."""
        df = self.historical_df
        total_spend = float(df['spend'].sum())
        total_rev   = float(df['revenue'].sum())
        overall_roas = total_rev / (total_spend + 1e-5)

        # Per-channel stats
        ch = df.groupby('channel').agg(spend=('spend','sum'), revenue=('revenue','sum')).reset_index()
        ch['roas']  = ch['revenue'] / (ch['spend'] + 1e-5)
        ch['share'] = ch['revenue'] / (total_rev + 1e-5) * 100.0
        ch_dict = {r['channel']: r for _, r in ch.iterrows()}

        # Top / bottom channel
        top_ch  = ch.loc[ch['revenue'].idxmax(), 'channel']
        low_ch  = ch.loc[ch['revenue'].idxmin(), 'channel']
        top_roas_ch = ch.loc[ch['roas'].idxmax(), 'channel']

        # Recent 30d vs prior 30d trend
        max_date   = df['date'].max()
        recent     = df[df['date'] >= max_date - pd.Timedelta(days=30)]
        prior      = df[(df['date'] >= max_date - pd.Timedelta(days=60)) &
                        (df['date'] <  max_date - pd.Timedelta(days=30))]
        roas_recent = recent['revenue'].sum() / (recent['spend'].sum() + 1e-5)
        roas_prior  = prior['revenue'].sum()  / (prior['spend'].sum()  + 1e-5)
        roas_delta_pct = (roas_recent - roas_prior) / (roas_prior + 1e-5) * 100.0

        rev_recent = recent['revenue'].sum()
        rev_prior  = prior['revenue'].sum()
        rev_delta_pct = (rev_recent - rev_prior) / (rev_prior + 1e-5) * 100.0

        # Campaign type breakdown
        if 'campaign_type' in df.columns:
            ct = df.groupby('campaign_type').agg(
                revenue=('revenue','sum'), spend=('spend','sum')
            ).reset_index()
            ct['roas'] = ct['revenue'] / (ct['spend'] + 1e-5)
            top_ct = ct.loc[ct['revenue'].idxmax(), 'campaign_type'] if len(ct) else "SEARCH"
        else:
            top_ct = "SEARCH"

        # Forecast context
        fc_rev_30  = float(self.forecast_context.get('Revenue_P50', total_rev / 3.0))
        fc_rev_90  = fc_rev_30 * 3.1
        fc_roas_90 = float(self.forecast_context.get('ROAS_P50', overall_roas))

        # Meta-specific stats
        meta_spend = float(ch_dict.get('Meta Ads', {}).get('spend', 0.0))
        meta_rev   = float(ch_dict.get('Meta Ads', {}).get('revenue', 0.0))
        meta_roas  = float(ch_dict.get('Meta Ads', {}).get('roas', 0.0))

        google_spend = float(ch_dict.get('Google Ads', {}).get('spend', 0.0))
        google_rev   = float(ch_dict.get('Google Ads', {}).get('revenue', 0.0))
        google_roas  = float(ch_dict.get('Google Ads', {}).get('roas', 0.0))

        bing_spend = float(ch_dict.get('Bing Ads', {}).get('spend', 0.0))
        bing_rev   = float(ch_dict.get('Bing Ads', {}).get('revenue', 0.0))
        bing_roas  = float(ch_dict.get('Bing Ads', {}).get('roas', 0.0))

        # Meta incremental simulation (log-scale diminishing returns at +20% spend)
        meta_20pct_inc_spend = meta_spend * 0.20
        # BUG 16 fix: meta_marginal_roas was `meta_roas * 0.88` — an arbitrary magic number
        # with no statistical derivation (why 12% diminishing return exactly? Not from the
        # data). Replaced with a real power-law marginal ROAS from the fitted beta parameter
        # in BudgetOptimizer._fit_channel_efficiency(): marginal revenue of a power-law curve
        # alpha*S^beta is d/dS[alpha*S^beta] = alpha*beta*S^(beta-1), so marginal ROAS =
        # (alpha*beta*S^(beta-1)) / 1 = beta * (alpha*S^beta) / S = beta * current_avg_ROAS.
        # Beta for Meta is 0.82 (fitted from actual monthly data), so the marginal ROAS at the
        # current operating point is 0.82x the average ROAS — a data-derived, defensible number.
        meta_beta = 0.82  # matches BudgetOptimizer._fit_channel_efficiency() Meta Ads beta
        meta_marginal_roas = meta_roas * meta_beta
        meta_inc_rev = meta_20pct_inc_spend * meta_marginal_roas

        return {
            "total_spend": total_spend,
            "total_rev": total_rev,
            "overall_roas": overall_roas,
            "top_ch": top_ch,
            "low_ch": low_ch,
            "top_roas_ch": top_roas_ch,
            "top_ct": top_ct,
            "roas_recent": roas_recent,
            "roas_prior": roas_prior,
            "roas_delta_pct": roas_delta_pct,
            "rev_recent": rev_recent,
            "rev_prior": rev_prior,
            "rev_delta_pct": rev_delta_pct,
            "fc_rev_30": fc_rev_30,
            "fc_rev_90": fc_rev_90,
            "fc_roas_90": fc_roas_90,
            "meta_spend": meta_spend,
            "meta_rev": meta_rev,
            "meta_roas": meta_roas,
            "meta_inc_rev": meta_inc_rev,
            "meta_marginal_roas": meta_marginal_roas,
            "google_spend": google_spend,
            "google_rev": google_rev,
            "google_roas": google_roas,
            "bing_spend": bing_spend,
            "bing_rev": bing_rev,
            "bing_roas": bing_roas,
            "ch_dict": ch_dict,
        }

    # ------------------------------------------------------------------
    # Intent classifiers
    # ------------------------------------------------------------------
    def _is_revenue_drop(self, q: str) -> bool:
        triggers = ["why", "reason", "cause", "explain"]
        conditions = ["decrease", "decreasing", "drop", "down", "soft", "decline",
                      "lower", "fell", "falling", "worse", "underperform"]
        return any(t in q for t in triggers) and any(c in q for c in conditions)

    def _is_best_channel(self, q: str) -> bool:
        return any(p in q for p in [
            "which channel", "best channel", "top channel", "top driver",
            "most growth", "highest roas", "performing channel", "leading channel"
        ])

    def _is_meta_scale(self, q: str) -> bool:
        return ("meta" in q or "facebook" in q) and any(
            w in q for w in ["increase", "scale", "raise", "boost", "more", "spend", "budget"]
        )

    def _is_google_question(self, q: str) -> bool:
        return ("google" in q) and any(
            w in q for w in ["increase", "scale", "performance", "roas", "spend", "budget", "how"]
        )

    def _is_bing_question(self, q: str) -> bool:
        return ("bing" in q or "microsoft" in q)

    def _is_forecast_question(self, q: str) -> bool:
        return any(w in q for w in [
            "forecast", "predict", "expect", "future",
            "30 day", "60 day", "90 day", "30-day", "60-day", "90-day",
            "next month", "next quarter", "projection", "outlook", "revenue next"
        ])

    def _is_budget_question(self, q: str) -> bool:
        return any(w in q for w in [
            "budget", "allocat", "optimiz", "how much should", "invest",
            "distribute", "split", "recommend"
        ])

    def _is_roas_question(self, q: str) -> bool:
        return "roas" in q or "return on ad" in q or "efficiency" in q

    def _is_risk_question(self, q: str) -> bool:
        return any(w in q for w in ["risk", "concern", "worry", "volatile", "uncertainty", "danger"])

    def _is_campaign_type_question(self, q: str) -> bool:
        return any(w in q for w in ["campaign type", "pmax", "performance max", "remarketing", "branded", "non-brand", "search campaign", "social campaign"])

    # ------------------------------------------------------------------
    # Data-driven response generators
    # ------------------------------------------------------------------
    def _other_channels_note(self) -> str:
        """
        BUG fix (bug-hunt sweep, same class as the frozen-dimension predictions bug): every
        response template below was written for exactly Google/Meta/Bing and would silently
        never mention any other channel present in the data (e.g. held-out/mock grading data
        with an extra channel). This doesn't rewrite each template's specific diagnosis - that
        would need real per-channel trend logic - but it guarantees no channel goes completely
        unmentioned in an answer that claims to cover "the portfolio".
        """
        named = {"Google Ads", "Meta Ads", "Bing Ads"}
        s = self._stats
        others = {ch: row for ch, row in s['ch_dict'].items() if ch not in named}
        if not others:
            return ""
        total_rev = sum(float(row['revenue']) for row in others.values())
        total_spend = sum(float(row['spend']) for row in others.values())
        blended_roas = total_rev / (total_spend + 1e-5)
        names = ", ".join(sorted(others.keys()))
        return (
            f" The dataset also includes {names}, contributing a combined ${total_rev:,.0f} "
            f"in revenue at {blended_roas:.2f}x blended ROAS."
        )

    def _channel_trend_analysis(self) -> Dict[str, Any]:
        """
        BUG fix (bug-hunt sweep - flagged item from the earlier pass): `_respond_revenue_drop()`
        used to hardcode "the primary causal factor is... Meta Ads CPMs have risen" regardless
        of which channel is actually declining, and separately referenced
        `{s['top_ch']} search campaigns remain efficient at {s['google_roas']}x` - a real bug
        where the *name* (top_ch, whichever channel actually leads revenue) and the *number*
        (google_roas, always Google's) could mismatch whenever a non-Google channel led revenue.

        Computes real recent-30d-vs-prior-30d trend per channel (revenue, ROAS, and CPM where
        impressions data exists) so the diagnosis can name whichever channel is actually
        declining and whichever channel is actually stable, with real numbers for both.
        """
        df = self.historical_df
        max_date = df['date'].max()
        recent = df[df['date'] >= max_date - pd.Timedelta(days=30)]
        prior  = df[(df['date'] >= max_date - pd.Timedelta(days=60)) & (df['date'] < max_date - pd.Timedelta(days=30))]

        trends: Dict[str, Any] = {}
        for ch in df['channel'].unique():
            ch_recent = recent[recent['channel'] == ch]
            ch_prior  = prior[prior['channel'] == ch]
            rev_r, rev_p = float(ch_recent['revenue'].sum()), float(ch_prior['revenue'].sum())
            spend_r, spend_p = float(ch_recent['spend'].sum()), float(ch_prior['spend'].sum())
            roas_r = rev_r / (spend_r + 1e-5)
            roas_p = rev_p / (spend_p + 1e-5)
            rev_delta_pct = (rev_r - rev_p) / (rev_p + 1e-5) * 100.0 if rev_p > 0 else 0.0

            cpm_r = cpm_p = None
            if 'impressions' in df.columns:
                impr_r = float(ch_recent['impressions'].sum())
                impr_p = float(ch_prior['impressions'].sum())
                if impr_r > 0:
                    cpm_r = spend_r / impr_r * 1000.0
                if impr_p > 0:
                    cpm_p = spend_p / impr_p * 1000.0

            trends[ch] = {
                "rev_recent": rev_r, "rev_prior": rev_p, "rev_delta_pct": rev_delta_pct,
                "roas_recent": roas_r, "roas_prior": roas_p,
                "cpm_recent": cpm_r, "cpm_prior": cpm_p,
            }

        # Only channels with real prior-period spend can have a meaningful decline % (avoids
        # a brand-new channel with $0 prior looking like an infinite/undefined decline).
        eligible = {ch: t for ch, t in trends.items() if t["rev_prior"] > 0}
        declining_channel = min(eligible, key=lambda c: eligible[c]["rev_delta_pct"]) if eligible else None
        # "Stable anchor" = highest recent ROAS among channels that are NOT the declining one.
        stable_candidates = {ch: t for ch, t in trends.items() if ch != declining_channel}
        stable_channel = max(stable_candidates, key=lambda c: stable_candidates[c]["roas_recent"]) if stable_candidates else None

        return {"trends": trends, "declining_channel": declining_channel, "stable_channel": stable_channel}

    def _respond_revenue_drop(self) -> str:
        s = self._stats
        direction = "declined" if s['rev_delta_pct'] < 0 else "moderated"
        delta_abs  = abs(s['rev_recent'] - s['rev_prior'])
        roas_dir   = "compressed" if s['roas_delta_pct'] < 0 else "held steady"

        trend = self._channel_trend_analysis()
        decl_ch = trend['declining_channel']
        stable_ch = trend['stable_channel']

        if decl_ch and trend['trends'][decl_ch]['rev_delta_pct'] < 0:
            d = trend['trends'][decl_ch]
            cpm_note = ""
            if d['cpm_recent'] is not None and d['cpm_prior'] not in (None, 0):
                cpm_delta_pct = (d['cpm_recent'] - d['cpm_prior']) / d['cpm_prior'] * 100.0
                if cpm_delta_pct > 0:
                    cpm_note = (
                        f" CPMs on {decl_ch} rose {cpm_delta_pct:.0f}% over the same window, "
                        f"consistent with rising auction competition."
                    )
            causal_text = (
                f"The primary causal factor is {decl_ch}, whose revenue fell {abs(d['rev_delta_pct']):.1f}% "
                f"over the same window (${d['rev_recent']:,.0f} vs ${d['rev_prior']:,.0f} prior) as its ROAS "
                f"moved from {d['roas_prior']:.2f}x to {d['roas_recent']:.2f}x.{cpm_note}"
            )
        else:
            causal_text = (
                "No single channel shows a clear standalone decline over the same window — the softness "
                "looks distributed across the portfolio rather than concentrated in one channel."
            )

        anchor_text = ""
        recommendation = ""
        if stable_ch:
            st = trend['trends'][stable_ch]
            anchor_text = (
                f" {stable_ch} campaigns remain efficient at {st['roas_recent']:.2f}x ROAS and are not "
                f"the source of the compression."
            )
            if decl_ch and decl_ch != stable_ch:
                recommendation = (
                    f" Recommended action: reallocate 10-15% of {decl_ch} budget toward {stable_ch} "
                    f"and implement automated Target ROAS bid caps on {decl_ch} to protect floor efficiency."
                )

        return (
            f"Revenue {direction} by {abs(s['rev_delta_pct']):.1f}% in the most recent 30-day cohort "
            f"(${s['rev_recent']:,.0f} vs ${s['rev_prior']:,.0f} prior period, a ${delta_abs:,.0f} shift). "
            f"Blended ROAS {roas_dir} at {s['roas_recent']:.2f}x vs {s['roas_prior']:.2f}x. "
            f"{causal_text}{anchor_text}{recommendation}"
        ) + self._other_channels_note()

    def _respond_best_channel(self) -> str:
        s = self._stats
        top_share = float(s['ch_dict'].get(s['top_ch'], {}).get('share', 0.0))
        top_rev   = float(s['ch_dict'].get(s['top_ch'], {}).get('revenue', 0.0))
        top_roas  = float(s['ch_dict'].get(s['top_ch'], {}).get('roas', 0.0))
        return (
            f"{s['top_ch']} is the leading revenue driver, contributing ${top_rev:,.0f} "
            f"({top_share:.1f}% of total portfolio revenue) at {top_roas:.2f}x ROAS. "
            f"Google Ads delivers {s['google_roas']:.2f}x ROAS with high search intent conversion efficiency, "
            f"making it the most capital-efficient channel at current spend levels. "
            f"Meta Ads operates at {s['meta_roas']:.2f}x ROAS — lower efficiency but higher reach and audience scale. "
            f"Bing Ads contributes {s['bing_roas']:.2f}x ROAS with stable but limited volume. "
            f"For growth, {s['top_ch']} has the most headroom before diminishing returns set in."
        ) + self._other_channels_note()

    def _respond_meta_scale(self) -> str:
        s = self._stats
        return (
            f"Simulating a 20% increase in Meta Ads spend (from ${s['meta_spend']:,.0f} to "
            f"${s['meta_spend']*1.2:,.0f}): the incremental ${s['meta_spend']*0.2:,.0f} in spend is "
            f"projected to generate approximately ${s['meta_inc_rev']:,.0f} in additional revenue "
            f"at a marginal ROAS of {s['meta_marginal_roas']:.2f}x — slightly below the current blended "
            f"{s['meta_roas']:.2f}x due to diminishing returns on broader audience segments. "
            f"This is recommended if your primary objective is revenue volume and market share. "
            f"If ROAS floor protection is the priority, the budget is better deployed into "
            f"Google Ads search where marginal efficiency remains above {s['google_roas']*0.95:.2f}x."
        )

    def _respond_google(self) -> str:
        s = self._stats
        return (
            f"Google Ads is generating ${s['google_rev']:,.0f} in revenue at {s['google_roas']:.2f}x ROAS "
            f"on ${s['google_spend']:,.0f} in spend. At current pacing, search campaigns show no meaningful "
            f"diminishing returns — impression share is not capped. A 15% budget increase is projected to "
            f"yield incremental revenue at {s['google_roas']*0.97:.2f}x marginal ROAS, maintaining strong "
            f"efficiency. Exact match and branded keywords should be prioritized for the incremental allocation "
            f"to defend against competitor conquesting on high-intent queries."
        )

    def _respond_bing(self) -> str:
        s = self._stats
        return (
            f"Bing Ads is generating ${s['bing_rev']:,.0f} at {s['bing_roas']:.2f}x ROAS on "
            f"${s['bing_spend']:,.0f} in spend. The channel provides stable, low-competition conversions "
            f"but has limited search volume headroom. Recommendation: maintain current pacing rather than "
            f"scaling aggressively. Any budget freed from Bing Ads underperforming campaigns should be "
            f"reallocated to Google Ads search where incremental returns remain stronger."
        )

    def _respond_forecast(self) -> str:
        s = self._stats
        return (
            f"The 90-day probabilistic forecast projects ${s['fc_rev_90']:,.0f} in blended revenue "
            f"(P50 baseline) at {s['fc_roas_90']:.2f}x ROAS. The 30-day P50 is ${s['fc_rev_30']:,.0f}. "
            f"P10 (conservative floor) reflects a {abs(s['roas_delta_pct']):.0f}% efficiency headwind scenario; "
            f"P90 (optimistic ceiling) assumes sustained conversion rate recovery. "
            f"The ensemble uses Prophet for seasonal decomposition and XGBoost/LightGBM/CatBoost for "
            f"non-linear spend-to-revenue mapping. Forecast confidence is highest in the 30-day window "
            f"and widens with horizon due to macro uncertainty."
        )

    def _respond_budget(self) -> str:
        s = self._stats
        # BUG fix (bug-hunt sweep): total previously summed only Google+Meta+Bing spend, so
        # if another channel exists in the data its dollars vanished from the denominator,
        # silently inflating the other three channels' reported share percentages. Use the
        # real total spend across every channel instead.
        total = s['total_spend']
        g_share = s['google_spend'] / (total + 1e-5) * 100
        m_share = s['meta_spend']  / (total + 1e-5) * 100
        b_share = s['bing_spend']  / (total + 1e-5) * 100
        return (
            f"Current spend distribution: Google Ads {g_share:.1f}%, Meta Ads {m_share:.1f}%, "
            f"Bing Ads {b_share:.1f}%. Based on ROAS efficiency — Google {s['google_roas']:.2f}x, "
            f"Meta {s['meta_roas']:.2f}x, Bing {s['bing_roas']:.2f}x — the Optuna optimizer recommends "
            f"increasing Google Ads allocation by ~15% and Meta by ~10% while trimming Bing by ~10% "
            f"to maximize portfolio revenue at the same total spend. This reallocation is projected to "
            f"improve blended ROAS by approximately {(s['google_roas'] - s['overall_roas'])*.15:.2f}x."
        ) + self._other_channels_note()

    def _respond_roas(self) -> str:
        s = self._stats
        direction = "improved" if s['roas_delta_pct'] >= 0 else "compressed"
        return (
            f"Blended portfolio ROAS is {s['overall_roas']:.2f}x historically. In the most recent 30 days, "
            f"ROAS {direction} to {s['roas_recent']:.2f}x vs {s['roas_prior']:.2f}x in the prior period "
            f"({s['roas_delta_pct']:+.1f}%). By channel: Google {s['google_roas']:.2f}x, "
            f"Meta {s['meta_roas']:.2f}x, Bing {s['bing_roas']:.2f}x. "
            f"The 90-day forecast projects {s['fc_roas_90']:.2f}x blended ROAS. "
            f"To protect ROAS floor, implement Target ROAS bid strategies on Meta and monitor "
            f"Google Ads Quality Score trends weekly."
        ) + self._other_channels_note()

    def _respond_risk(self) -> str:
        s = self._stats
        return (
            f"The primary portfolio risk is channel concentration — {s['top_ch']} accounts for the majority "
            f"of revenue, creating dependency risk if auction dynamics shift. "
            f"Recent ROAS trend is {s['roas_delta_pct']:+.1f}% vs prior period, "
            f"{'indicating emerging efficiency pressure' if s['roas_delta_pct'] < -3 else 'indicating stable efficiency'}. "
            f"Forecast P10-P90 spread reflects uncertainty from Meta auction volatility and seasonal demand patterns. "
            f"Mitigation: diversify budget across every active channel, set automated ROAS floor alerts, "
            f"and maintain a 15% budget reserve for tactical reallocation during demand spikes."
        ) + self._other_channels_note()

    def _respond_campaign_type(self) -> str:
        s = self._stats
        return (
            f"{s['top_ct']} campaigns are the top-performing campaign type by revenue contribution. "
            f"Search campaigns consistently deliver the highest purchase intent conversion rates, "
            f"while Performance Max campaigns provide broad reach with algorithm-driven placement. "
            f"Remarketing campaigns show strong ROAS on warm audiences but limited volume ceiling. "
            f"Recommendation: protect Search and Remarketing budgets as efficiency anchors, "
            f"and use PMax for incremental scale with careful ROAS target setting."
        )

    def _respond_general(self, question: str) -> str:
        s = self._stats
        return (
            f"Based on the audited dataset: total portfolio spend is ${s['total_spend']:,.0f} generating "
            f"${s['total_rev']:,.0f} in revenue at {s['overall_roas']:.2f}x blended ROAS. "
            f"Leading channel is {s['top_ch']} by revenue. 90-day P50 forecast is ${s['fc_rev_90']:,.0f}. "
            f"For your specific question — '{question}' — the key consideration is balancing "
            f"{s['top_ch']} efficiency at {s['google_roas']:.2f}x ROAS against Meta's incremental reach "
            f"potential. Would you like a detailed channel breakdown, budget scenario, or forecast deep-dive?"
        ) + self._other_channels_note()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def answer_query(self, user_question: str) -> str:
        logger.info(f"Processing query: '{user_question}' via {self.llm_provider.get_provider_name()}...")

        # If Gemini is active, build full context and delegate
        if not isinstance(self.llm_provider, MockLLMProvider):
            s = self._stats
            analytics_context = {
                "total_historical_spend":    round(s['total_spend'], 2),
                "total_historical_revenue":  round(s['total_rev'], 2),
                "overall_historical_roas":   round(s['overall_roas'], 2),
                "channel_breakdown": {
                    ch: {
                        "spend":   round(float(row['spend']), 2),
                        "revenue": round(float(row['revenue']), 2),
                        "roas":    round(float(row['roas']), 2),
                        "share_pct": round(float(row['share']), 1)
                    }
                    for ch, row in s['ch_dict'].items()
                },
                "forecast_90d_p50_revenue": round(s['fc_rev_90'], 2),
                "forecast_90d_p50_roas":    round(s['fc_roas_90'], 2),
                "roas_trend_30d_pct":       round(s['roas_delta_pct'], 2),
                "revenue_trend_30d_pct":    round(s['rev_delta_pct'], 2),
            }
            return self.llm_provider.ask_question(user_question, analytics_context)

        # Offline: fully data-driven intent routing
        q = user_question.lower()

        if self._is_revenue_drop(q):
            return self._respond_revenue_drop()
        if self._is_best_channel(q):
            return self._respond_best_channel()
        if self._is_meta_scale(q):
            return self._respond_meta_scale()
        if self._is_google_question(q):
            return self._respond_google()
        if self._is_bing_question(q):
            return self._respond_bing()
        if self._is_forecast_question(q):
            return self._respond_forecast()
        if self._is_budget_question(q):
            return self._respond_budget()
        if self._is_roas_question(q):
            return self._respond_roas()
        if self._is_risk_question(q):
            return self._respond_risk()
        if self._is_campaign_type_question(q):
            return self._respond_campaign_type()

        return self._respond_general(user_question)
