"""Diagnostic (not selection): what is ATSlvl- (long small avg-trade-size / short large)? Compare to a plain
size factor (long low 30d qv / short high) - counted as 1 extra config (total 79)."""
import os
import numpy as np, pandas as pd
import r2common as rc, lab
import famB_grid as B
LOG = os.path.join(rc.HERE, "checks.csv")
Um = rc.U(50); p = rc.panels()
S = B.build_scores()
lq = np.log(p["Q"].rolling(30, min_periods=24).mean())
m = Um & S["ATSlvl"].notna() & lq.notna()
ra, rq = lab.xs_rank(S["ATSlvl"], m), lab.xs_rank(lq, m)
c = [ra.iloc[i].corr(rq.iloc[i]) for i in range(len(ra)) if m.iloc[i].sum() >= 10]
print("XS rank corr ATSlvl vs log qv30: median %.2f" % np.nanmedian(c))
row, res = rc.evaluate(rc.qls(-lq, Um), "SIZE_smallqv+", "diag", LOG)
print(rc.fmt(row))
d = rc.ctrl()
a = lab.backtest(rc.qls(-S["ATSlvl"], Um), "1d", cost=rc.COST)["df"]["r"]
print("corr ATSlvl- vs size factor daily:", round(a.corr(res["df"]["r"]), 2))
W = rc.qls(-S["ATSlvl"], Um)
sh = (W < 0).sum().sort_values(ascending=False).head(10)
lo = (W > 0).sum().sort_values(ascending=False).head(10)
print("most frequent shorts (days):", sh.to_dict())
print("most frequent longs (days):", lo.to_dict())
