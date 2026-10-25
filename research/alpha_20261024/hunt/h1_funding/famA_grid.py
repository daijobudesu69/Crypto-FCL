"""FAMILY A - cross-sectional funding factor (1d bars, 2020-2024 IS only).

PRE-REGISTERED GRID (written before running):
 A1 base: score = -(trailing funding sum over L days), L in {1,3,7,14}
          universe in {top50, top100} (lab.universe, min_hist 60d, 30d median qv rank)
          weighting in {quintile q=0.2 EW, rank-weighted xs_demean(rank)}
          rebalance_every in {1, 3}
          -> 4*2*2*2 = 32 configs
 A2 delta: score = -(F3(t)-F3(t-3)) and -(F7(t)-F7(t-7)); top50/top100; quintile/rank; daily
          -> 8 configs
 A3 residual vs momentum (diagnostic): score = -(residual of XS-ranked F7 after XS-regressing
          on XS-ranked 14d return), top50, quintile/rank, daily -> 2 configs
 Note: "funding relative to BTC funding" (F_i - F_BTC) is a constant shift per day, so its
 cross-sectional rank is IDENTICAL to A1; not run separately (0 extra configs).
 TOTAL = 42 configs. All net of 0.07%/side + actual funding.
"""
from common import *

F = lab.load("fund", "1d")
C = lab.load("close", "1d")
Us = {50: lab.universe("1d", top_k=50), 100: lab.universe("1d", top_k=100)}


def weights(score, U, wt):
    if wt == "q":
        return lab.quantile_ls_weights(score, U, q=0.2)
    return lab.xs_demean_weights(lab.xs_rank(score, U), U)


rows = []
for L in [1, 3, 7, 14]:
    s = -F.rolling(L, min_periods=L).sum()
    for k, U in Us.items():
        for wt in ["q", "r"]:
            W = weights(s, U, wt)
            for rb in [1, 3]:
                name = f"A1_L{L}_top{k}_{wt}_rb{rb}"
                d, _ = row(W, name, U, rebalance_every=rb, legs=True)
                d.update(fam="A1", L=L, k=k, wt=wt, rb=rb)
                rows.append(d); print(name, f"{d['sharpe']:.2f} price_net {d['sh_price_net']:.2f} fund_yr {d['fund_yr']:+.3f}", flush=True)
for L in [3, 7]:
    f = F.rolling(L, min_periods=L).sum()
    s = -(f - f.shift(L))
    for k, U in Us.items():
        for wt in ["q", "r"]:
            W = weights(s, U, wt)
            name = f"A2_d{L}_top{k}_{wt}"
            d, _ = row(W, name, U)
            d.update(fam="A2", L=L, k=k, wt=wt, rb=1)
            rows.append(d); print(name, f"{d['sharpe']:.2f} price_net {d['sh_price_net']:.2f}", flush=True)
# A3 residual
U = Us[50]
rf = lab.xs_rank(F.rolling(7, min_periods=7).sum(), U)
rm = lab.xs_rank(C / C.shift(14) - 1, U)
rf_d = rf.sub(rf.mean(axis=1), axis=0); rm_d = rm.sub(rm.mean(axis=1), axis=0)
beta = (rf_d * rm_d).sum(axis=1) / (rm_d ** 2).sum(axis=1)
resid = rf_d - rm_d.mul(beta, axis=0)
for wt in ["q", "r"]:
    W = weights(-resid, U, wt)
    name = f"A3_resid7_top50_{wt}"
    d, _ = row(W, name, U)
    d.update(fam="A3", L=7, k=50, wt=wt, rb=1)
    rows.append(d); print(name, f"{d['sharpe']:.2f} price_net {d['sh_price_net']:.2f}", flush=True)

out = pd.DataFrame(rows)
out.to_csv("famA_grid.csv", index=False)
pd.set_option("display.width", 250)
cols = ["name", "sharpe", "cagr", "maxdd", "sh_price_net", "sh_fund", "fund_yr", "cost_yr", "sh_2020-21", "sh_2022",
        "sh_2023-24", "sh_long_vsEW", "sh_short_vsEW", "npos"]
print(out[cols].round(2).to_string())
print("N configs:", len(out))
