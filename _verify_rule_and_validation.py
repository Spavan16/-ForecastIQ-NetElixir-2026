"""Verification for the two pending uncommitted fixes:
1. rule_engine.py - growth_opportunities now covers every channel, not just Meta/Google.
2. validation.py - an unrecognized CSV in data/ is now logged with a penalty, not silently dropped.
"""
import sys, shutil
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
from src.validation import ValidationEngine
from src.risk_engine import RiskIntelligenceEngine
from src.rule_engine import RuleInsightEngine

print("=" * 70)
print("PART 1: rule_engine.py growth_opportunities coverage")
print("=" * 70)
val = ValidationEngine()
df, summary = val.run_full_ingestion()
risk = RiskIntelligenceEngine(df, data_quality_score=summary['data_quality_score']).evaluate_risk()

# Real data run
rule = RuleInsightEngine(df, {"Revenue_P50": 500000}, risk)
insights = rule.generate_all_insights()
titles = [g['title'] for g in insights['growth_opportunities']]
print("Real data growth_opportunities titles:", titles)
assert any('Bing Ads' in t for t in titles), f"FAIL: Bing Ads missing from growth opportunities: {titles}"
print("PASS: Bing Ads present in growth_opportunities on real data.")

# Synthetic extra channel
tiktok = df[df['channel'] == 'Meta Ads'].copy()
tiktok['channel'] = 'TikTok Ads'
df2 = pd.concat([df, tiktok], ignore_index=True)
risk2 = RiskIntelligenceEngine(df2, data_quality_score=summary['data_quality_score']).evaluate_risk()
rule2 = RuleInsightEngine(df2, {"Revenue_P50": 500000}, risk2)
insights2 = rule2.generate_all_insights()
titles2 = [g['title'] for g in insights2['growth_opportunities']]
print("Synthetic 4-channel growth_opportunities titles:", titles2)
assert any('TikTok Ads' in t for t in titles2), f"FAIL: TikTok Ads missing: {titles2}"
assert any('Bing Ads' in t for t in titles2), f"FAIL: Bing Ads missing: {titles2}"
print("PASS: Both Bing Ads and synthetic TikTok Ads present, no exceptions.")

print()
print("=" * 70)
print("PART 2: validation.py unrecognized-CSV detection")
print("=" * 70)
DATA_DIR = Path(__file__).resolve().parent / "data"
temp_csv = DATA_DIR / "_verify_tiktok_ads_temp.csv"
try:
    df[df['channel'] == 'Meta Ads'].head(20).to_csv(temp_csv, index=False)
    val2 = ValidationEngine()
    _, summary2 = val2.run_full_ingestion()
    matches = [w for w in summary2['critical_warnings'] if '_verify_tiktok_ads_temp.csv' in w['message']]
    print("Critical warnings containing the unrecognized file:", matches)
    assert len(matches) == 1, f"FAIL: expected exactly 1 warning about the unrecognized CSV, got {len(matches)}"
    assert matches[0]['penalty'] == 10.0, f"FAIL: expected penalty 10.0, got {matches[0]['penalty']}"
    print("PASS: unrecognized CSV correctly flagged as a critical warning with a 10.0 penalty.")
finally:
    if temp_csv.exists():
        temp_csv.unlink()
    print("Cleaned up temp CSV.")

print()
print("ALL CHECKS PASSED")
