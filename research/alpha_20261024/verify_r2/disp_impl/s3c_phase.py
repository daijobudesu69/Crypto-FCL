"""disp_impl step 3c - POST-HOC diagnostics of the re-ranked pick G=1.0 STRAT W (not selection):
 - weekly re-target weekday phase (Mon..Sun close) -> is Monday lucky?
 - starting equity 150/250/400 on run B (N step function)
 - drawdown location run A/B
Outputs -> s3c_out.txt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import s3_design_lib as L
OUT = os.path.dirname(os.path.abspath(__file__))
log = open(os.path.join(OUT, "s3c_out.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n"); log.flush()
base_mon = L.monday_close.copy()
for wd in range(7):
    L.monday_close = np.array([(ts + pd.Timedelta("1D")).weekday() == wd for ts in L.C.index])
    oA, rA, sA, _ = L.sim(1.0, "STRAT", "W", "2020-03-01")
    oB, rB, sB, _ = L.sim(1.0, "STRAT", "W", "2023-01-01")
    P(f"re-target weekday {wd} (0=Mon): A SR {sA['sharpe']:.2f} DD {sA['maxdd']:.1%} eq ${sA['eq_end']:.0f} | B SR {sB['sharpe']:.2f} DD {sB['maxdd']:.1%} eq ${sB['eq_end']:.0f}")
L.monday_close = base_mon
for e0 in [150, 250, 400]:
    oB, rB, sB, _ = L.sim(1.0, "STRAT", "W", "2023-01-01", eq0=e0)
    P(f"eq0 ${e0} run B: SR {sB['sharpe']:.2f} DD {sB['maxdd']:.1%} tot {sB['tot']:.1%} npos {sB['avg_npos']:.1f}")
for nm, st in [("A", "2020-03-01"), ("B", "2023-01-01")]:
    o, r, s, _ = L.sim(1.0, "STRAT", "W", st)
    e = o["equity"]; dd = e / e.cummax() - 1
    t = dd.idxmin(); pk = e[:t].idxmax()
    P(f"run {nm}: maxDD {dd.min():.1%} peak {pk.date()} trough {t.date()}; yearly {dict((int(k), round(v,3)) for k,v in ((1+r).groupby(r.index.year).prod()-1).items())}")
    o.to_csv(os.path.join(OUT, f"s3c_equity_{nm}.csv"))
# same diagnostics for the pre-registered pick D (DD location only)
for nm, st in [("A", "2020-03-01"), ("B", "2023-01-01")]:
    o, r, s, _ = L.sim(1.0, "STRAT", "D", st)
    e = o["equity"]; dd = e / e.cummax() - 1; t = dd.idxmin(); pk = e[:t].idxmax()
    P(f"[STRAT D] run {nm}: maxDD {dd.min():.1%} peak {pk.date()} trough {t.date()}")
log.close()
