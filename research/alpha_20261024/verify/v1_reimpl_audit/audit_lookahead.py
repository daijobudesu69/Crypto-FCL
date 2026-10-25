"""Audit (b)(c)(d): truncation test for lookahead + lab.backtest accounting checks.

Truncation test: rebuild universe, signal, weights and backtest returns from panels cut at date X
(future rows physically removed) and compare with the full-sample objects on rows <= X.
Any difference = something at row t depends on rows > t.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab

Cf = lab.load("close", "1d").copy(); Qf = lab.load("qv", "1d").copy(); Ff = lab.load("fund", "1d").copy()


def pipeline(q):
    U = lab.universe("1d", min_hist_days=60, top_k=50)
    S = -lab.load("fund", "1d").rolling(14, min_periods=14).sum()
    W_t = lab.quantile_ls_weights(S, U, q=q)
    W_r = lab.xs_demean_weights(lab.xs_rank(S, U), U)
    return U, S, W_t, W_r


full = pipeline(0.33)
rb_full = lab.backtest(full[2], "1d")["df"]["r"]
for X in ["2020-09-15", "2021-06-30", "2022-11-15", "2023-08-01", "2024-06-30"]:
    Xt = pd.Timestamp(X, tz="UTC")
    lab._mem[("close", "1d")] = Cf[Cf.index <= Xt]
    lab._mem[("qv", "1d")] = Qf[Qf.index <= Xt]
    lab._mem[("fund", "1d")] = Ff[Ff.index <= Xt]
    tr = pipeline(0.33)
    diffs = []
    for name, a, b in zip(["U", "F14", "W_tercile", "W_rank"], full, tr):
        aa = a[a.index <= Xt].astype("float64"); bb = b.astype("float64")
        d = (aa - bb).abs().max().max()
        diffs.append(f"{name}:{d:.2e}")
    rb_tr = lab.backtest(tr[2], "1d")["df"]["r"]
    common = rb_tr.index[:-1]                    # the last row of the truncated run has no next bar
    dr = (rb_full.reindex(common) - rb_tr.reindex(common)).abs().max()
    print(f"cut {X}: max |full - truncated| on rows<=cut  " + "  ".join(diffs) + f"  backtest r: {dr:.2e}")
lab._mem[("close", "1d")] = Cf; lab._mem[("qv", "1d")] = Qf; lab._mem[("fund", "1d")] = Ff

# rolling windows used anywhere: confirm pandas rolling default is trailing (center=False)
x = pd.Series(np.arange(10.0)); print("rolling(3).sum() at i=2:", x.rolling(3).sum().iloc[2], "(trailing => 0+1+2=3)")

# drift-adjusted turnover check on a toy 2-asset case vs hand calculation
ix = Cf.index[:6]
C2 = pd.DataFrame({"A": [100, 110, 121, 121, 100, 100.0], "B": [100, 100, 100, 90, 90, 90.0]}, index=ix)
F2 = pd.DataFrame(0.0, index=ix, columns=["A", "B"]); F2.iloc[2, 0] = 0.001; F2.iloc[2, 1] = -0.002
save = {k: lab._mem.get(k) for k in [("close", "1d"), ("fund", "1d")]}
lab._mem[("close", "1d")] = C2; lab._mem[("fund", "1d")] = F2
W2 = pd.DataFrame({"A": [0.5, 0.5, 0.5, 0, 0, 0], "B": [-0.5, -0.5, -0.5, 0, 0, 0.0]}, index=ix)
r2 = lab.backtest(W2, "1d", cost=0.001)["df"]
print(r2[["r", "gross_r", "fund_r", "cost_r", "turnover"]].round(6).to_string())
# hand: t0: buy 0.5A, short 0.5B -> turnover 1.0. bar1: A +10%, B 0 -> gross +0.05; drift A=0.55/1.05, B=-0.5/1.05
# t1 turnover = |0.5-0.5238|+|-0.5+0.4762| = 0.0476 ; fund on bar2 (row t1 -> F[2]): -(0.5*0.001 + -0.5*-0.002) = -0.0015
lab._mem[("close", "1d")] = save[("close", "1d")]; lab._mem[("fund", "1d")] = save[("fund", "1d")]
print("hand: t0 turnover 1.0, gross +0.05; t1 turnover 0.0476, fund -0.0015 (long pays +0.1%, short pays when F<0: -0.2% -> short pays 0.1%)")
