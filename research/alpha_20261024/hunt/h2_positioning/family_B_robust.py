"""FAMILY B robustness (pre-registered in family_B.py): best 2 by IS Sharpe = B3_-dlev7 (0.70) and
B3_-zlev90 (0.37). Variants: reb3, top100, q33, funding-neutral (resid on XS rank of 7d funding). 8 configs.
Optional 4h step skipped: every fast (k=1) OI signal was strongly negative, nothing to refine.
Also: control funding factor (short high 7d funding, quintile) for correlation checks - control, counted."""
from h2lib import *
U50 = lab.universe("1d", top_k=50); U100 = lab.universe("1d", top_k=100)
C = lab.load("close", "1d"); QV = lab.load("qv", "1d"); F = lab.load("fund", "1d")
OI = lab.load("oi_usd", "1d"); OI = OI.where(OI > 0)
lev = np.log(OI / QV.rolling(30, min_periods=15).mean())
fund7 = F.rolling(7, min_periods=5).sum()
def hold(W, k):
    keep = np.arange(len(W)) % k == 0
    return W.where(pd.Series(keep, index=W.index), np.nan).ffill().fillna(0.0)
rets = {}
row, res = evaluate("CTRL_fund7_short_high", "B-ctrl", qls(-fund7, U50)); print(fmt(row)); rets["fund"] = res["df"]["r"]
base = {"B3_-dlev7": -(lev - lev.shift(7)), "B3_-zlev90": -tsz(lev, 90)}
for n, s in base.items():
    for tag, W in [("reb3", hold(qls(s, U50), 3)), ("top100", qls(s, U100)), ("q33", qls(s, U50, q=0.33)),
                   ("fundneutral", qls(xs_resid(s, fund7, U50), U50))]:
        row, res = evaluate(f"{n}|{tag}", "B", W); print(fmt(row))
    r = lab.backtest(qls(s, U50), "1d")["df"]["r"]
    print(f"   corr({n} base, funding factor) = {r.corr(rets['fund'].reindex(r.index)):.2f}")
