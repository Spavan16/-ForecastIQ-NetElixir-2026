"""Verification for chat_engine.py _respond_revenue_drop fix - the last flagged item
from the bug-hunt sweep. Confirms the causal diagnosis names whichever channel is
ACTUALLY declining, not a hardcoded 'Meta'. Calls the offline method directly since
Gemini is currently live and answer_query() would otherwise route around the fix.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
from src.validation import ValidationEngine
from src.chat_engine import ForecastChatBot

val = ValidationEngine()
df, summary = val.run_full_ingestion()
max_date = df['date'].max()

print("=" * 70)
print("RUN 1: REAL DATA (offline path)")
print("=" * 70)
bot = ForecastChatBot(df, {"Revenue_P50": 500000, "ROAS_P50": 4.0})
answer = bot._respond_revenue_drop()
print(answer)
print()

print("=" * 70)
print("RUN 2: SYNTHETIC - ENGINEER GOOGLE ADS (not Meta) TO BE THE DECLINING CHANNEL")
print("=" * 70)
df2 = df.copy()
is_recent_google = (df2['date'] >= max_date - pd.Timedelta(days=30)) & (df2['channel'] == 'Google Ads')
df2.loc[is_recent_google, 'revenue'] = df2.loc[is_recent_google, 'revenue'] * 0.2
bot2 = ForecastChatBot(df2, {"Revenue_P50": 500000, "ROAS_P50": 4.0})
answer2 = bot2._respond_revenue_drop()
print(answer2)

assert "primary causal factor is Google Ads" in answer2, \
    f"FAIL: expected Google Ads to be named as the declining channel. Got:\n{answer2}"
assert "Meta Ads CPMs have risen" not in answer2, "FAIL: old hardcoded Meta sentence still present"
print("\nPASS: diagnosis correctly names Google Ads (the actually-declining channel), old hardcoded Meta sentence is gone.")

print("\n" + "=" * 70)
print("RUN 3: SYNTHETIC - Meta actually IS still the declining channel (sanity check)")
print("=" * 70)
df3 = df.copy()
is_recent_meta = (df3['date'] >= max_date - pd.Timedelta(days=30)) & (df3['channel'] == 'Meta Ads')
df3.loc[is_recent_meta, 'revenue'] = df3.loc[is_recent_meta, 'revenue'] * 0.2
bot3 = ForecastChatBot(df3, {"Revenue_P50": 500000, "ROAS_P50": 4.0})
answer3 = bot3._respond_revenue_drop()
print(answer3)
assert "primary causal factor is Meta Ads" in answer3, \
    f"FAIL: expected Meta Ads to still be correctly named when it IS the real decline. Got:\n{answer3}"
print("\nPASS: correctly still names Meta Ads when Meta genuinely is the declining channel.")

print("\n" + "=" * 70)
print("RUN 4: SYNTHETIC - extra channel (TikTok Ads) present, engineered to be the decliner")
print("=" * 70)
tiktok = df[df['channel'] == 'Meta Ads'].copy()
tiktok['channel'] = 'TikTok Ads'
df4 = pd.concat([df, tiktok], ignore_index=True)
is_recent_tt = (df4['date'] >= max_date - pd.Timedelta(days=30)) & (df4['channel'] == 'TikTok Ads')
df4.loc[is_recent_tt, 'revenue'] = df4.loc[is_recent_tt, 'revenue'] * 0.2
bot4 = ForecastChatBot(df4, {"Revenue_P50": 500000, "ROAS_P50": 4.0})
answer4 = bot4._respond_revenue_drop()
print(answer4)
assert "primary causal factor is TikTok Ads" in answer4, \
    f"FAIL: expected TikTok Ads to be named. Got:\n{answer4}"
print("\nPASS: works correctly for a channel unseen in local training data too.")

print("\nALL CHECKS PASSED")
