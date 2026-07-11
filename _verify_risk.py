"""Verification script for risk_engine.py mitigation-copy fix.
Run 1: real 3-channel data (should still work, no exceptions, sane channel names).
Run 2: synthetic 4-channel data with one channel engineered to have wildly volatile
weekly ROAS, confirming the mitigation text names THAT channel, not a hardcoded one.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from src.validation import ValidationEngine
from src.risk_engine import RiskIntelligenceEngine

print("=" * 70)
print("RUN 1: REAL DATA")
print("=" * 70)
val = ValidationEngine()
df, summary = val.run_full_ingestion()
risk = RiskIntelligenceEngine(df, data_quality_score=summary['data_quality_score']).evaluate_risk()
for f in risk['risk_factors']:
    print(f"[{f['name']}] {f['mitigation']}")
print(f"\nRisk score: {risk['risk_score']} ({risk['risk_classification']})")

print("\n" + "=" * 70)
print("RUN 2: SYNTHETIC 4-CHANNEL, ENGINEERED VOLATILE 'TikTok Ads'")
print("=" * 70)
df2 = df.copy()
tiktok = df[df['channel'] == 'Meta Ads'].copy()
tiktok['channel'] = 'TikTok Ads'
# Engineer wild week-to-week swings in revenue (spend held flat) so its ROAS CV is
# far higher than Google/Meta/Bing's real historical CV.
rng = np.random.default_rng(42)
week_num = tiktok['date'].dt.isocalendar().week.astype(int)
multiplier = np.where(week_num % 2 == 0, 4.0, 0.15)  # violent swings by week parity
tiktok['revenue'] = tiktok['revenue'] * multiplier
df2 = pd.concat([df2, tiktok], ignore_index=True)

risk2 = RiskIntelligenceEngine(df2, data_quality_score=summary['data_quality_score']).evaluate_risk()
for f in risk2['risk_factors']:
    print(f"[{f['name']}] {f['mitigation']}")
print(f"\nRisk score: {risk2['risk_score']} ({risk2['risk_classification']})")

roas_mitigation = [f for f in risk2['risk_factors'] if f['name'] == 'ROAS Instability'][0]['mitigation']
assert 'TikTok Ads' in roas_mitigation, f"FAIL: expected TikTok Ads to be named as most volatile, got: {roas_mitigation}"
assert 'Meta' not in roas_mitigation, f"FAIL: old hardcoded 'Meta' still present: {roas_mitigation}"
print("\nPASS: ROAS Instability mitigation correctly names the engineered volatile channel (TikTok Ads), not the old hardcoded 'Meta'.")

dq_mitigation = [f for f in risk2['risk_factors'] if f['name'] == 'Data Quality Audit'][0]['mitigation']
assert 'Bing' not in dq_mitigation, f"FAIL: old hardcoded 'Bing' still present: {dq_mitigation}"
print("PASS: Data Quality mitigation no longer hardcodes 'Bing'.")

print("\nALL CHECKS PASSED")
