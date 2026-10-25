"""Follow-up 2 (evaluations of configs already in the grids, no new selection):
  Family A: canonical George-Hwang PH N=365 rb=1 (best PH), PHR365 rb7 re-residualised inside
            top-30/75 universes (fixes the universe check, which reused the top-50 residual).
  Family B: best RESID (L30 skip7 raw rb7), best FIP-type (FIP L30 rb7): full stress + spanning
            vs xsmom14, mom90, matched plain momentum (same L/skip/rebal) and beta-neutral matched
            momentum (separates 'residualisation' from 'beta-neutral leg scaling').
"""
import json
import numpy as np, pandas as pd
import common as cm
import lab
import famA_grid as A
import famB_grid as B

ctrl = cm.controls()
out = {}

# ---- A: PH365 rb1
S = A.scores(365)
u50 = cm.U(50)
W = lab.quantile_ls_weights(S["PH"], u50, q=0.2)
o, _ = cm.full_eval(W, "PH365_rb1", u50, rebal=1, n_trials=36, ctrl=ctrl, mom_match=cm.mom_ctrl(365, 0, 1))
W16 = cm.topn_weights(S["PH"], u50, n=8)
o16, _ = cm.full_eval(W16, "16", u50, 1, 36, legs=False, ctrl=ctrl)
o["pos16"] = {k: o16[k] for k in ["sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "sh_cost2x", "sh_lag1",
                                  "alpha_t_xsmom14_btc", "mineq_med"]}
out["PH365_rb1"] = o

# ---- A: PHR365 rb7 inside other universes
c, h = A.c, A.h
mx = h.rolling(365, min_periods=60).max()
ph = np.log(c / mx)
lag = c.shift(364)
hist = c.notna().cumsum()
lag = lag.where(lag.notna(), pd.DataFrame(np.broadcast_to(A.first_close.values, c.shape),
                                          index=c.index, columns=c.columns).where(hist >= 60))
ret = np.log(c / lag)
for k in (30, 75):
    uk = cm.U(k)
    y = ph.where(uk & ret.notna()); x = ret.where(y.notna())
    xd = x.sub(x.mean(axis=1), axis=0); yd = y.sub(y.mean(axis=1), axis=0)
    phr = yd - xd.mul((xd * yd).sum(axis=1) / (xd ** 2).sum(axis=1), axis=0)
    ok, _ = cm.quick(lab.quantile_ls_weights(phr, uk, q=0.2), 7)
    out[f"PHR365_rb7_u{k}"] = {kk: ok[kk] for kk in ["sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "npos"]}

# ---- B: RESID L30 sk7 raw rb7
Wr = B.resid_weights(30, 7, "raw")
mm = cm.mom_ctrl(30, 7, rebal=7)
o, _ = cm.full_eval(Wr, "RESID30_sk7_raw_rb7", u50, rebal=7, n_trials=42, ctrl=ctrl, mom_match=mm)
mom = c.shift(7) / c.shift(37) - 1
Wbn = cm.beta_neutral(lab.quantile_ls_weights(mom, B.u_nobtc, q=0.2), B.beta)
rbn = cm.run(Wbn, 7)["df"]["r"]
o["betaneutral_plain_mom_sharpe"] = cm.sh(rbn)
cc = ctrl.copy(); cc["bnm"] = rbn
rr = cm.run(Wr, 7)["df"]["r"]
o["alpha_t_vs_betaneutral_mom"] = cm.spanning(rr, cc, ["bnm", "btc"])["alpha_t"]
o["corr_betaneutral_mom"] = cm.corr(rr, rbn)
W16 = cm.beta_neutral(cm.topn_weights(B.resid_score(30, 7, "raw"), B.u_nobtc, n=8), B.beta)
o16, _ = cm.full_eval(W16, "16", u50, 7, 42, legs=False, ctrl=ctrl)
o["pos16"] = {k: o16[k] for k in ["sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "sh_cost2x", "sh_lag1",
                                  "alpha_t_xsmom14_btc", "mineq_med"]}
for k in (30, 75):
    ok, _ = cm.quick(B.resid_weights(30, 7, "raw", u=cm.U(k)), 7)
    o[f"u{k}"] = {kk: ok[kk] for kk in ["sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "npos"]}
g = pd.read_csv("famB_grid.csv")
sub = g[(g["sub"] == "RESID") & (
    ((g.L == 30) & (g.skip == 7) & (g.kind == "raw") & (g.rebal == 1)) |
    ((g.L == 30) & (g.skip == 3) & (g.kind == "raw") & (g.rebal == 7)) |
    ((g.L == 60) & (g.skip == 7) & (g.kind == "raw") & (g.rebal == 7)) |
    ((g.L == 30) & (g.skip == 7) & (g.kind == "scaled") & (g.rebal == 7)))]
o["neigh_median"] = float(sub.sharpe.median())
out["RESID30_sk7_raw_rb7"] = o

# ---- B: FIP L30 rb7
Wf = B.fip_weights(30, 3, "FIP")
o, _ = cm.full_eval(Wf, "FIP30_rb7", u50, rebal=7, n_trials=42, ctrl=ctrl, mom_match=cm.mom_ctrl(30, 3, 7))
sub = g[(g["sub"] == "FIP") & (((g.L == 30) & (g.rebal == 1)) | ((g.L == 60) & (g.rebal == 7)))]
o["neigh_median"] = float(sub.sharpe.median())
out["FIP30_rb7"] = o

for k, v in out.items():
    print(k, json.dumps(v, default=float))
json.dump(out, open("followup2.json", "w"), indent=1, default=float)
