"""Concentration / outlier diagnostics for A4_L14_top50_t_rb1 and A1_L14_top50_r_rb1 (no new configs)."""
from common import *
F = lab.load("fund", "1d"); C = lab.load("close", "1d")
R = C.pct_change(fill_method=None)
U = lab.universe("1d", top_k=50)
s = -F.rolling(14, min_periods=14).sum()
# funding outliers
fu = F.where(U)
st = fu.stack()
print("most extreme daily funding in U:"); print(st.sort_values().head(8)); print(st.sort_values().tail(5))
for name, W in [("tercile", lab.quantile_ls_weights(s, U, q=0.33)), ("rank", lab.xs_demean_weights(lab.xs_rank(s, U), U))]:
    Wv = W.where(R.shift(-1).notna(), 0)
    pr = (Wv * R.shift(-1).fillna(0)).sum()
    fr = (-(Wv * F.shift(-1).fillna(0))).sum()
    tot = (pr + fr).sort_values()
    print(f"\n== {name}: per-coin sum contribution (price+fund), top/bottom 8")
    print(pd.DataFrame({"tot": tot, "price": pr[tot.index], "fund": fr[tot.index]}).round(3).head(8).to_string())
    print(pd.DataFrame({"tot": tot, "price": pr[tot.index], "fund": fr[tot.index]}).round(3).tail(8).to_string())
    print("share of total from top-5 coins:", tot.tail(5).sum() / tot.sum())
    # drop top 5 contributing coins (hindsight exclusion = harsh test)
    drop = tot.tail(5).index
    U2 = U.copy(); U2[drop] = False
    W2 = lab.quantile_ls_weights(s, U2, q=0.33) if name == "tercile" else lab.xs_demean_weights(lab.xs_rank(s, U2), U2)
    r2 = lab.backtest(W2, "1d")
    print("drop top5 contributors", list(drop), "-> Sharpe %.2f" % r2["metrics"]["sharpe"], subperiods(r2["df"]["r"]))
    # cap funding at +-0.3%/day in signal and pnl? just signal winsor check
    # sign split: long-leg funding vs short-leg funding
    fl = (-(Wv.clip(lower=0) * F.shift(-1).fillna(0))).sum(axis=1).mean() * 365
    fs = (-(Wv.clip(upper=0) * F.shift(-1).fillna(0))).sum(axis=1).mean() * 365
    print("funding/yr from long leg %.3f, from short leg %.3f" % (fl, fs))
    # how many days long leg avg funding negative
    lf = (Wv.clip(lower=0) * F).sum(axis=1)
