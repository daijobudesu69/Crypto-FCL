"""Diagnostic (not a strategy config): why do rank-IC-positive signals lose money in quintile L/S?
Per-quintile mean vs median next-day return (top-50, days with >=20 names)."""
from h2lib import *
U = lab.universe("1d", top_k=50)
C = lab.load("close", "1d"); OI = lab.load("oi_usd", "1d"); OI = OI.where(OI > 0)
R1 = C.pct_change(fill_method=None).shift(-1)
sigs = {"-ret7": -(C / C.shift(7) - 1), "-dOI7": -lchg(OI, 7), "-dOI3": -lchg(OI, 3), "-ret1": -C.pct_change(fill_method=None)}
for n, s in sigs.items():
    x = s.where(U & R1.notna()); ok = x.notna().sum(axis=1) >= 20
    x = x[ok]; r = R1[ok].where(x.notna())
    q = x.rank(axis=1, pct=True)
    out = {}
    for i, (a, b) in enumerate([(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]):
        m = (q > a) & (q <= b)
        v = r.where(m).stack()
        out[f"Q{i+1}"] = {"mean_bp": v.mean() * 1e4, "median_bp": v.median() * 1e4, "p99_bp": v.quantile(.99) * 1e4, "p01_bp": v.quantile(.01) * 1e4}
    print(n); print(pd.DataFrame(out).round(1).to_string())
