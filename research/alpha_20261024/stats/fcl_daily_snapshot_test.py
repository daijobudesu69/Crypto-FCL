"""Q1 (IS only, 2020-03..2024-12): can FCL run on ONE funding snapshot per day (a plain daily GitHub
Actions cron, no watcher loop)? F14 is rebuilt from the 00:00 UTC settlement only, scaled by settlements/day,
and with random missed days (cron failures) forward-filled."""
import os, sys
assert os.environ.get("ALPHA_FORWARD") != "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, lab, fcl
f1 = lab.load("fund", "1h")
nset = (f1 != 0).groupby(f1.index.floor("1D")).sum()            # settlements per day (live: fundingIntervalHours)
snap = f1[f1.index.hour == 23]; snap.index = snap.index.floor("1D")  # the settlement at 00:00 of the next day
daily_proxy = (snap * nset.reindex(snap.index).clip(lower=1)).reindex(lab.load("fund", "1d").index)
orig = fcl.panels
def make(miss):
    rng = np.random.default_rng(7)
    d = daily_proxy.copy()
    if miss:
        mask = rng.random(len(d)) < miss
        d[mask] = np.nan; d = d.ffill()
    def panels():
        C, F, U, F14, mom = orig()
        return C, F, U, d.rolling(14, min_periods=14).sum(), mom
    return panels
real = lab.load("fund", "1d").rolling(14).sum(); prox = daily_proxy.rolling(14).sum()
rc = [real.loc[t].rank().corr(prox.loc[t].rank()) for t in real.index[400::7]]
print("XS rank corr F14 real vs 1-snapshot/day:", round(float(np.nanmedian(rc)), 3))
for name, miss in [("all settlements (baseline)", None), ("1 snapshot/day", 0.0), ("1 snapshot/day, 20% days missed", 0.2), ("1 snapshot/day, 40% days missed", 0.4)]:
    fcl.panels = orig if miss is None else make(miss)
    d, s, _ = fcl.run("2020-03-01", "2024-12-31", venue="binance")
    print(f"{name:34s} ${s['equity_end']:.0f}  SR {s['sharpe']:.2f}  DD {s['maxdd']:.1%}", flush=True)
