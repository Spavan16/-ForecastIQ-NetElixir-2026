"""
Unit/smoke tests for the five analytics modules that had zero automated test coverage:
budget_optimizer.py, monte_carlo.py, risk_engine.py, rule_engine.py, scenarios.py.

These sit downstream of the core forecasting engine (already covered by
test_unit_forecasting.py) and the graded run.sh pipeline (already covered by
test_run_pipeline.py) -- they power the SaaS backend's budget optimizer, risk dashboard,
Monte Carlo simulator, rule-based insights, and scenario planner. None of them are exercised
by any existing test, and several have had real, documented bugs fixed in them (see the
BUG-fix comments in each module) without any regression test to keep those fixes from
silently reverting.

None of these five need an API key or network access (that's chat_engine.py/llm_provider.py/
pdf_reporting.py, deliberately NOT covered here -- testing those meaningfully needs mocking
an LLM response, a bigger job than fits this pass). All five are pure computation over a
historical_df / forecast dict, so a small synthetic dataset with a known, sane shape is
enough to catch "does this crash" and "are the basic invariants (ROAS positive, P10<=P50<=P90,
budget constraint respected, risk score in [0,100]) actually true," the same level of rigor
as the existing two test files.

Same no-pytest, plain-assert convention as the rest of tests/ (requirements.txt is locked
this close to the deadline). Run directly:

    python tests/test_analytics_modules.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.budget_optimizer import BudgetOptimizer
from src.monte_carlo import MonteCarloSimulator, derive_revenue_volatility
from src.risk_engine import RiskIntelligenceEngine
from src.rule_engine import RuleInsightEngine
from src.scenarios import ScenarioGenerator


def _synthetic_channel_df(n_days: int = 400, seed: int = 7) -> pd.DataFrame:
    """A small, sane, multi-channel/multi-month synthetic dataset -- real enough to exercise
    every groupby/date-window code path in these modules (weekly resampling, month-over-month
    aggregation, per-channel ROAS), without depending on the actual project data/ files (so
    these tests still catch a regression even if the real dataset changes shape)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    channel_config = [("Google Ads", 500.0, 4.0), ("Meta Ads", 300.0, 3.0), ("Bing Ads", 100.0, 2.5)]
    rows = []
    for ch, base_spend, roas in channel_config:
        noise = rng.normal(1.0, 0.08, size=n_days)
        spend = base_spend * (1.0 + 0.15 * np.sin(np.arange(n_days) / 30.0)) * noise
        spend = np.clip(spend, 10.0, None)
        revenue = spend * roas * rng.normal(1.0, 0.05, size=n_days)
        for d, sp, rv in zip(dates, spend, revenue):
            rows.append({"date": d, "channel": ch, "spend": float(sp), "revenue": float(rv)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# budget_optimizer.py
# ---------------------------------------------------------------------------

def test_budget_optimizer_fits_and_simulates():
    df = _synthetic_channel_df()
    opt = BudgetOptimizer(df)
    assert set(opt.channel_params.keys()) == {"Google Ads", "Meta Ads", "Bing Ads"}
    for ch, p in opt.channel_params.items():
        assert p["alpha"] > 0, f"{ch}: alpha should be positive, got {p['alpha']}"
        assert 0.0 < p["beta"] <= 1.0, f"{ch}: beta should be in (0, 1], got {p['beta']}"
        assert 0.0 <= p["fit_cv"] <= 0.6, f"{ch}: fit_cv out of documented [0, 0.6] clip range: {p['fit_cv']}"

    base_spend = {ch: 10000.0 for ch in opt.channels}
    sim_up = opt.simulate_budget_change({"Google Ads": 20.0}, base_spend)
    sim_flat = opt.simulate_budget_change({}, base_spend)
    assert sim_up["total_revenue"] > 0 and sim_up["total_spend"] > 0
    # Diminishing-returns curve is monotonic increasing in spend (alpha*spend^beta, beta>0):
    # more spend on Google should mean strictly more total revenue than the 0%-change case.
    assert sim_up["total_revenue"] > sim_flat["total_revenue"], (
        "Increasing Google Ads spend by 20% did not increase simulated total revenue -- "
        "diminishing-returns curve may be broken."
    )
    assert sim_up["channel_contributions"]["Google Ads"]["revenue_change_pct"] > 0
    print("PASS: BudgetOptimizer fits positive, in-range channel params and simulate_budget_change "
          "responds monotonically to a spend increase.")


def test_budget_optimizer_excludes_zero_revenue_ramp_up_months():
    """Regression test for the zero-revenue-ramp-up fix in _fit_channel_efficiency (see the
    'BUG fix (Budget Optimizer audit, July 2026)' comment in budget_optimizer.py). This module
    has no other backtest harness, unlike models.py -- this is the lightweight substitute:
    construct a channel with a real launch/tracking-ramp period (genuine spend, zero revenue)
    followed by a known, stable alpha*spend^beta relationship, and verify the fitted alpha
    tracks the TRUE post-ramp efficiency rather than being dragged down by the contaminated
    zero-revenue months averaged in unfiltered.
    """
    beta_true = 0.78
    alpha_true = 30.0
    rng = np.random.default_rng(11)

    rows = []
    # Months 1-3: real spend, genuinely zero revenue (launch/tracking-ramp period) --
    # structurally identical to the documented Bing Ads 2024-05/06/07 case.
    ramp_months = pd.date_range("2024-01-01", periods=3, freq="MS")
    ramp_daily_spend = [28.0 / 30, 3436.0 / 30, 4396.0 / 30]
    for m, daily_sp in zip(ramp_months, ramp_daily_spend):
        for d in pd.date_range(m, periods=28, freq="D"):
            rows.append({"date": d, "channel": "Bing Ads", "spend": daily_sp, "revenue": 0.0})

    # Months 4-15: genuine, stable efficiency curve with light noise -- this is the "true"
    # relationship the fit should recover once the zero-revenue months are excluded.
    mature_months = pd.date_range("2024-04-01", periods=12, freq="MS")
    for m in mature_months:
        monthly_spend = float(rng.uniform(2000.0, 6000.0))
        monthly_revenue = alpha_true * (monthly_spend ** beta_true) * float(rng.normal(1.0, 0.03))
        daily_sp = monthly_spend / 28.0
        daily_rev = monthly_revenue / 28.0
        for d in pd.date_range(m, periods=28, freq="D"):
            rows.append({"date": d, "channel": "Bing Ads", "spend": daily_sp, "revenue": daily_rev})

    df = pd.DataFrame(rows)
    opt = BudgetOptimizer(df)
    fitted_alpha = opt.channel_params["Bing Ads"]["alpha"]

    # Sanity replica of the OLD buggy behavior (spend>0 filter only, no revenue>0 filter) to
    # confirm this synthetic setup actually exercises the bug, not just a trivially-passing
    # dataset. If the old filter were still in place, the 3 zero-revenue ramp months would be
    # averaged into avg_monthly_rev/avg_monthly_sp and drag alpha down noticeably.
    df_c = df.copy()
    df_c["year_month"] = df_c["date"].dt.to_period("M")
    monthly_old_buggy = df_c.groupby("year_month").agg({"spend": "sum", "revenue": "sum"}).reset_index()
    monthly_old_buggy = monthly_old_buggy[monthly_old_buggy["spend"] > 0]  # old filter only
    old_buggy_alpha = monthly_old_buggy["revenue"].mean() / (monthly_old_buggy["spend"].mean() ** beta_true)

    assert old_buggy_alpha < fitted_alpha * 0.9, (
        "Synthetic setup should reproduce the bug's contamination (old-filter alpha "
        f"{old_buggy_alpha:.2f} should be meaningfully lower than the fixed fit {fitted_alpha:.2f}) "
        "-- if not, this test isn't actually exercising the fix."
    )
    # The fix should recover something close to the true post-ramp alpha, not the
    # zero-revenue-contaminated one.
    assert abs(fitted_alpha - alpha_true) / alpha_true < 0.15, (
        f"Fitted alpha {fitted_alpha:.2f} should be within 15% of the true post-ramp alpha "
        f"{alpha_true} once zero-revenue ramp-up months are excluded, got {fitted_alpha:.2f}"
    )
    print(f"PASS: BudgetOptimizer excludes zero-revenue ramp-up months from efficiency fitting -- "
          f"fitted alpha {fitted_alpha:.2f} tracks true {alpha_true} (old buggy filter would have "
          f"given {old_buggy_alpha:.2f}).")


def test_budget_optimizer_allocation_respects_budget_and_reports_feasibility():
    df = _synthetic_channel_df()
    opt = BudgetOptimizer(df)
    max_budget = 50_000.0
    # Low target_roas: should be easily achievable given synthetic ROAS is 2.5x-4x.
    easy = opt.optimize_allocation(max_budget=max_budget, target_roas=1.0, n_trials=25)
    assert easy["recommended_total_spend"] <= max_budget * 1.01, (
        f"Recommended spend {easy['recommended_total_spend']} exceeds max_budget {max_budget}"
    )
    assert easy["expected_total_roas"] > 0
    assert easy["target_achievable"] is True, "A target_roas of 1.0x should be trivially achievable here"
    assert easy["confidence_range"]["revenue_p10"] <= easy["confidence_range"]["revenue_p50"] <= easy["confidence_range"]["revenue_p90"]
    assert abs(sum(c["budget_share"] for c in easy["channel_recommendations"].values()) - 100.0) < 1.0

    # Impossibly high target_roas: should be reported as infeasible, not silently returned
    # as if it were a success (see the BUG fix this guards in budget_optimizer.py).
    hard = opt.optimize_allocation(max_budget=max_budget, target_roas=500.0, n_trials=25)
    assert hard["target_achievable"] is False
    assert hard["target_achievement_note"] is not None

    try:
        opt.optimize_allocation(max_budget=-100.0, target_roas=2.0, n_trials=5)
        assert False, "optimize_allocation should raise ValueError on a non-positive max_budget"
    except ValueError:
        pass
    print("PASS: optimize_allocation respects the budget constraint, reports feasibility honestly, "
          "and fails loudly on an invalid max_budget.")


# ---------------------------------------------------------------------------
# monte_carlo.py
# ---------------------------------------------------------------------------

def test_derive_revenue_volatility_bounds_and_fallback():
    # Real residual/base_revenue input should land inside the documented [floor, ceiling] clip.
    v = derive_revenue_volatility(residual_std_daily=5000.0, base_revenue=100_000.0, horizon_days=30)
    assert 0.03 <= v <= 0.60
    # Missing residual_std_daily should fall back to the documented static default.
    v_fallback = derive_revenue_volatility(residual_std_daily=None, base_revenue=100_000.0)
    assert v_fallback == 0.15
    print("PASS: derive_revenue_volatility stays within its documented clip range and falls back "
          "to the static default when no residual std is available.")


def test_monte_carlo_simulation_invariants():
    sim = MonteCarloSimulator(n_simulations=2000)  # smaller than production 10,000 for test speed
    result = sim.run_portfolio_simulation(
        base_revenue_30d=150_000.0,
        base_spend_30d=40_000.0,
        channel_splits={"Google Ads": 0.5, "Meta Ads": 0.3, "Bing Ads": 0.2},
        residual_std_daily=3000.0,
    )
    assert result["worst_case_revenue"] <= result["expected_revenue"] <= result["best_case_revenue"]
    assert result["worst_case_roas"] <= result["expected_roas"] <= result["best_case_roas"]
    assert result["expected_revenue"] > 0 and result["expected_roas"] > 0
    assert set(result["channel_distributions"].keys()) == {"Google Ads", "Meta Ads", "Bing Ads"}
    for ch, dist in result["channel_distributions"].items():
        assert dist["worst_case"] <= dist["expected_case"] <= dist["best_case"], f"{ch} distribution not ordered"
    assert len(result["revenue_histogram"]) == 25
    assert sum(b["frequency"] for b in result["revenue_histogram"]) == 2000, (
        "Histogram bin frequencies should sum to n_simulations -- every simulated draw should "
        "land in exactly one bin."
    )
    print("PASS: Monte Carlo worst<=expected<=best holds for revenue/ROAS/every channel, and "
          "histogram bins account for every simulated path.")


# ---------------------------------------------------------------------------
# risk_engine.py
# ---------------------------------------------------------------------------

def test_risk_engine_score_in_range_and_classification_consistent():
    df = _synthetic_channel_df()
    risk = RiskIntelligenceEngine(df, data_quality_score=95.0).evaluate_risk()
    assert 0.0 <= risk["risk_score"] <= 100.0, f"risk_score out of [0, 100]: {risk['risk_score']}"
    assert risk["risk_classification"] in {"Low Risk", "Medium Risk", "High Risk"}
    # Classification bands are supposed to exactly match the documented thresholds --
    # verifying the mapping itself, not just that SOME string came back.
    s = risk["risk_score"]
    expected = "Low Risk" if s < 35.0 else ("Medium Risk" if s < 65.0 else "High Risk")
    assert risk["risk_classification"] == expected, (
        f"risk_score {s} should map to '{expected}' per the documented thresholds, "
        f"got '{risk['risk_classification']}'"
    )
    assert len(risk["risk_factors"]) == 4
    for factor in risk["risk_factors"]:
        assert 0.0 <= factor["score"] <= 100.0, f"{factor['name']} score out of range: {factor['score']}"
    print(f"PASS: RiskIntelligenceEngine score ({s}) is in [0, 100] and its classification label "
          "matches the documented threshold bands.")


def test_risk_engine_handles_a_dominant_single_channel():
    """A portfolio where one channel is ~99% of spend should score meaningfully higher on
    Channel Dependency than the balanced 3-way-split synthetic dataset above -- verifying the
    HHI-based dependency score actually responds to concentration, not just returning a
    constant."""
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "channel": ["Google Ads"] * 200 + ["Bing Ads"] * 200,
        "spend": [1000.0] * 200 + [10.0] * 200,
        "revenue": [4000.0] * 200 + [25.0] * 200,
    })
    risk = RiskIntelligenceEngine(df, data_quality_score=95.0).evaluate_risk()
    dep_factor = next(f for f in risk["risk_factors"] if f["name"] == "Channel Dependency")
    assert dep_factor["status"] == "Concentrated", (
        f"A 99%-single-channel portfolio should be flagged 'Concentrated', got '{dep_factor['status']}' "
        f"(score {dep_factor['score']})"
    )
    print("PASS: RiskIntelligenceEngine correctly flags a dominant-single-channel portfolio as "
          "'Concentrated' rather than a constant/generic score.")


# ---------------------------------------------------------------------------
# rule_engine.py
# ---------------------------------------------------------------------------

def test_rule_engine_generates_all_five_sections_without_crashing():
    df = _synthetic_channel_df()
    risk = RiskIntelligenceEngine(df, data_quality_score=95.0).evaluate_risk()
    forecast_90d = {"Revenue_P50": 400_000.0, "ROAS_P50": 3.2}
    insights = RuleInsightEngine(df, forecast_90d, risk).generate_all_insights()

    for key in ["executive_summary", "growth_opportunities", "risk_assessment",
                "budget_recommendations", "forecast_explanation"]:
        assert key in insights, f"Missing expected section: {key}"

    assert isinstance(insights["executive_summary"], str) and len(insights["executive_summary"]) > 0
    assert len(insights["growth_opportunities"]) > 0
    assert len(insights["risk_assessment"]) > 0
    # One budget recommendation per channel actually present in historical_df -- verifies the
    # BUG fix that stopped this from returning a fixed, hardcoded Google/Meta/Bing triplet
    # regardless of what's actually in the data.
    recommended_channels = {r["channel"] for r in insights["budget_recommendations"]}
    assert recommended_channels == set(df["channel"].unique()), (
        f"budget_recommendations should cover exactly the channels present in the data, "
        f"got {recommended_channels} vs {set(df['channel'].unique())}"
    )
    print("PASS: RuleInsightEngine produces all 5 documented sections, and budget recommendations "
          "trace to the real channels present in the data (not a hardcoded triplet).")


def test_rule_engine_only_mentions_channels_actually_present():
    """A dataset with no Meta Ads at all should never generate a Meta-specific growth
    opportunity -- guards the BUG fix that used to fire Meta/Google-specific insights
    unconditionally regardless of whether those channels exist in the data."""
    dates = pd.date_range("2024-01-01", periods=150, freq="D")
    df = pd.DataFrame({
        "date": dates, "channel": ["Bing Ads"] * 150,
        "spend": [200.0] * 150, "revenue": [500.0] * 150,
    })
    risk = RiskIntelligenceEngine(df, data_quality_score=90.0).evaluate_risk()
    insights = RuleInsightEngine(df, {"Revenue_P50": 50_000.0}, risk).generate_all_insights()
    titles = " ".join(g["title"] for g in insights["growth_opportunities"])
    assert "Meta" not in titles and "Google" not in titles, (
        f"Growth opportunities mention a channel not present in the data: {titles}"
    )
    print("PASS: RuleInsightEngine doesn't fabricate insights about channels absent from the data.")


# ---------------------------------------------------------------------------
# scenarios.py
# ---------------------------------------------------------------------------

def test_scenario_generator_produces_seven_ordered_scenarios():
    base = {"30_days": {"Revenue_P50": 100_000.0, "ROAS_P50": 4.0, "Spend_Expected": 25_000.0},
            "60_days": {"Revenue_P50": 200_000.0, "ROAS_P50": 4.0, "Spend_Expected": 50_000.0},
            "90_days": {"Revenue_P50": 300_000.0, "ROAS_P50": 4.0, "Spend_Expected": 75_000.0}}
    scenarios = ScenarioGenerator(base["30_days"], base["60_days"], base["90_days"],
                                   base_revenue_volatility=0.15).generate_all_scenarios()
    assert len(scenarios) == 7
    ids = [s["id"] for s in scenarios]
    assert "expected" in ids and len(set(ids)) == 7, "Scenario ids should be unique"

    for s in scenarios:
        for window in ["30_days", "60_days", "90_days"]:
            f = s["forecasts"][window]
            assert f["Revenue_P10"] <= f["Revenue_P50"] <= f["Revenue_P90"], (
                f"Scenario '{s['id']}' {window}: P10/P50/P90 out of order for Revenue"
            )
            assert f["ROAS_P10"] <= f["ROAS_P50"] <= f["ROAS_P90"], (
                f"Scenario '{s['id']}' {window}: P10/P50/P90 out of order for ROAS"
            )
            assert f["Revenue_P50"] > 0 and f["Spend_Expected"] > 0

    expected_scenario = next(s for s in scenarios if s["id"] == "expected")
    assert abs(expected_scenario["forecasts"]["30_days"]["Revenue_P50"] - 100_000.0) < 1.0, (
        "The 'expected' scenario (1.0x multiplier) should reproduce the base P50 forecast unchanged"
    )
    print("PASS: ScenarioGenerator produces 7 uniquely-identified scenarios, each with correctly "
          "ordered P10/P50/P90 bands, and the baseline 'expected' scenario reproduces the input P50.")


def test_scenario_generator_handles_missing_keys_with_documented_defaults():
    """base_predictions dicts use .get() with documented fallback defaults (Revenue_P50:
    250000.0, ROAS_P50: 4.5, Spend_Expected: 50000.0) -- verifying an incomplete/empty input
    dict doesn't crash, matching how a caller with a partially-populated forecast would use
    this in practice."""
    empty = {}
    scenarios = ScenarioGenerator(empty, empty, empty).generate_all_scenarios()
    assert len(scenarios) == 7
    for s in scenarios:
        assert s["forecasts"]["30_days"]["Revenue_P50"] > 0
    print("PASS: ScenarioGenerator falls back to documented defaults on an empty input dict "
          "instead of crashing.")


if __name__ == "__main__":
    test_budget_optimizer_fits_and_simulates()
    test_budget_optimizer_excludes_zero_revenue_ramp_up_months()
    test_budget_optimizer_allocation_respects_budget_and_reports_feasibility()
    test_derive_revenue_volatility_bounds_and_fallback()
    test_monte_carlo_simulation_invariants()
    test_risk_engine_score_in_range_and_classification_consistent()
    test_risk_engine_handles_a_dominant_single_channel()
    test_rule_engine_generates_all_five_sections_without_crashing()
    test_rule_engine_only_mentions_channels_actually_present()
    test_scenario_generator_produces_seven_ordered_scenarios()
    test_scenario_generator_handles_missing_keys_with_documented_defaults()
    print("\nAll analytics-module tests passed.")
