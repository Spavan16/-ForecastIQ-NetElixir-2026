import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "src"))

import main as backend_main
state = backend_main.get_analytics_state()
df = state["df"]

bing = df[df["channel"] == "Bing Ads"].copy()
bing["year_month"] = bing["date"].dt.to_period("M")
monthly = bing.groupby("year_month").agg({"spend": "sum", "revenue": "sum"}).reset_index()
monthly = monthly[monthly["spend"] > 0]
print(monthly.to_string())

print()
opt = state["budget_opt"]
p = opt.channel_params["Bing Ads"]
print("alpha:", p["alpha"], "beta:", p["beta"])
import numpy as np
predicted = p["alpha"] * (monthly["spend"].values ** p["beta"])
actual = monthly["revenue"].values
print()
print("predicted:", predicted)
print("actual:   ", actual)
rel_resid = (actual - predicted) / (predicted + 1e-5)
print("rel_resid:", rel_resid)
