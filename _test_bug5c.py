import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "src"))

import main as backend_main
state = backend_main.get_analytics_state()
opt = state["budget_opt"]

res_100k = opt.optimize_allocation(100000, 4.5)
res_500k = opt.optimize_allocation(500000, 4.5)

for label, res in [("100k", res_100k), ("500k", res_500k)]:
    print(label)
    for ch, data in res["channel_recommendations"].items():
        print(f"  {ch}: spend={data['allocated_spend']:.0f} rev={data['expected_revenue']:.0f} share={data['budget_share']:.1f}%")
    print()
