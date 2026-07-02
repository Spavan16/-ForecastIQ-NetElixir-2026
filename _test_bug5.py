import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "src"))

import main as backend_main

state = backend_main.get_analytics_state()
opt = state["budget_opt"]

print("=== Per-channel fit_cv (real, computed from actual monthly residuals) ===")
for ch, params in opt.channel_params.items():
    print(f"  {ch}: alpha={params['alpha']:.3f} beta={params['beta']:.3f} fit_cv={params['fit_cv']:.4f}")

print()
print("=== Real optimize_allocation() output at two different budgets ===")
res_100k = opt.optimize_allocation(100000, 4.5)
res_500k = opt.optimize_allocation(500000, 4.5)

for label, res in [("100k budget", res_100k), ("500k budget", res_500k)]:
    cr = res["confidence_range"]
    spread_pct = (cr["revenue_p90"] - cr["revenue_p10"]) / cr["revenue_p50"] * 100
    print(f"{label}: revenue_p10={cr['revenue_p10']:.0f} p50={cr['revenue_p50']:.0f} p90={cr['revenue_p90']:.0f}  (spread={spread_pct:.1f}% of P50)")

print()
print("=== Sanity checks ===")
print("Is spread no longer flat 24% (old 0.88-1.12) for both budgets?")
for label, res in [("100k", res_100k), ("500k", res_500k)]:
    cr = res["confidence_range"]
    spread_pct = (cr["revenue_p90"] - cr["revenue_p10"]) / cr["revenue_p50"] * 100
    print(f"  {label}: {spread_pct:.2f}% (old hardcoded value was always 24.00%)")

print()
print("Ordering check: p10 <= p50 <= p90 for both?")
for label, res in [("100k", res_100k), ("500k", res_500k)]:
    cr = res["confidence_range"]
    ok = cr["revenue_p10"] <= cr["revenue_p50"] <= cr["revenue_p90"]
    print(f"  {label}: {ok}")

print()
print("=== Targeted unit test: does a manually-injected noisy channel widen the band vs a clean one? ===")
import pandas as pd
import numpy as np

# Build a tiny synthetic historical_df: one channel with a near-perfect alpha*spend^beta fit,
# one channel with the same average spend/revenue but wildly noisy month-to-month.
dates = pd.date_range("2025-01-01", periods=180, freq="D")
rows = []
rng = np.random.default_rng(42)
for d in dates:
    # Clean channel: revenue almost exactly tracks a fixed daily spend with tiny noise
    rows.append({"date": d, "channel": "Clean Channel", "spend": 1000.0, "revenue": 1000.0 * 4.0 + rng.normal(0, 5)})
    # Noisy channel: same average spend/revenue, but huge month-to-month swings
    noisy_mult = 1.0 + rng.normal(0, 0.6)
    rows.append({"date": d, "channel": "Noisy Channel", "spend": 1000.0, "revenue": max(0, 1000.0 * 4.0 * noisy_mult)})
synthetic_df = pd.DataFrame(rows)

from src.budget_optimizer import BudgetOptimizer
synth_opt = BudgetOptimizer(synthetic_df)
print("Clean Channel fit_cv:", round(synth_opt.channel_params["Clean Channel"]["fit_cv"], 4))
print("Noisy Channel fit_cv:", round(synth_opt.channel_params["Noisy Channel"]["fit_cv"], 4))
print("Noisy > Clean (expected True):", synth_opt.channel_params["Noisy Channel"]["fit_cv"] > synth_opt.channel_params["Clean Channel"]["fit_cv"])
