"""disp_impl step 3b - POST-HOC stress of the 6 STRAT configs (no new configs: same 12-grid members under
2x cost and signal lag 1), because the pre-registered pick (G=1.0 STRAT D) showed 2x-cost Sharpe 0.57 in run A
driven by membership churn (15,989 orders). Also fee drag per year. Not used to re-rank unless the pick fails the
brief's bar (2x cost and lag1 Sharpe >= ~0.6 in both runs).
Outputs -> s3b_out.txt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import s3_design_lib as L

OUT = os.path.dirname(os.path.abspath(__file__))
log = open(os.path.join(OUT, "s3b_out.txt"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.write(s + "\n"); log.flush()


for G in [1.0, 1.25, 1.5]:
    for R in ["D", "W"]:
        res = []
        for lbl, kw in [("base", {}), ("2x", dict(cost=2 * L.lab.HYPE_COST)), ("lag1", dict(lag=1))]:
            _, rA, sA, _ = L.sim(G, "STRAT", R, "2020-03-01", **kw)
            _, rB, sB, _ = L.sim(G, "STRAT", R, "2023-01-01", **kw)
            res.append(f"{lbl}: A {sA['sharpe']:.2f} B {sB['sharpe']:.2f}")
            if lbl == "base":
                feeA = sA["fees"] / (len(rA) / 365); feeB = sB["fees"] / (len(rB) / 365)
                fb = f"fees/yr A ${feeA:.1f} (end eq ${sA['eq_end']:.0f}) B ${feeB:.1f} (end eq ${sB['eq_end']:.0f}); orders B/yr {sB['orders']/(len(rB)/365):.0f}"
        P(f"G={G} STRAT {R}: " + " | ".join(res) + " | " + fb)
log.close()
