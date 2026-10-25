"""Diagnostics for the family-A survivor RP (range position) - no new signal selection.
  1. variance decomposition vs [xsmom14, btc]: R2, beta, appraisal ratio of the residual.
  2. 16-position (8L/8S) neighbours: RP N in {90,180,365} rb1 (RP180 16pos = +1 trial).
  3. short-leg resting-stop overlay (risk sensitivity, +1 trial): every short carries an exchange
     stop at +30% above the previous daily close; if day-high crosses it the short is closed at
     the stop with 1% extra slippage and re-entered at that day's close (extra round-trip cost).
  4. worst events + Sharpe without the single worst day.
"""
import json
import numpy as np, pandas as pd
import common as cm
import lab
from famA_grid import scores

ctrl = cm.controls()
c = lab.load("close", "1d"); h = lab.load("high", "1d")
u50 = cm.U(50)
out = {}


def decomp(r):
    d = pd.concat([r.rename("y"), ctrl[["xsmom14", "btc"]]], axis=1).dropna()
    X = np.column_stack([np.ones(len(d)), d[["xsmom14", "btc"]].values])
    b = np.linalg.lstsq(X, d["y"].values, rcond=None)[0]
    e = d["y"].values - X @ b
    r2 = 1 - e.var() / d["y"].var()
    return {"r2": float(r2), "b_xsmom14": float(b[1]), "b_btc": float(b[2]),
            "alpha_ann": float(b[0] * 365), "appraisal_ratio": float(b[0] / e.std() * np.sqrt(365))}


def stop_overlay(W, stop=0.30, slip=0.01, cost=cm.COST):
    """approximate resting stop on shorts using daily high (daily rebalance books only)."""
    res = cm.run(W, 1)
    df = res["df"].copy()
    R = cm.R()
    Wv = W.reindex(index=c.index, columns=c.columns).fillna(0.0)
    hi_ret = (h / c.shift(1) - 1).shift(-1)           # next-bar high vs this close
    r_next = R.shift(-1)
    hit = (Wv < 0) & (hi_ret >= stop)
    # replace realised short return r_next with (stop+slip) for stopped names
    adj = (Wv.where(hit, 0.0) * ((stop + slip) - r_next.where(hit, 0.0))).sum(axis=1)  # w<0 -> loss smaller
    extra_cost = (Wv.abs().where(hit, 0.0).sum(axis=1)) * 2 * cost
    delta = adj - extra_cost
    delta.index = delta.index + pd.Timedelta("1D")
    r2 = df["r"] + delta.reindex(df.index).fillna(0.0)
    return r2, int(hit.sum().sum())


for N in (365, 90):
    S = scores(N)["RP"]
    W = lab.quantile_ls_weights(S, u50, q=0.2)
    r = cm.run(W, 1)["df"]["r"]
    o = {"decomp": decomp(r)}
    rs, nh = stop_overlay(W)
    o["stop30"] = {"sharpe": cm.sh(rs), "n_stops": nh, **{f"sh_{k}": v for k, v in cm.subperiods(rs).items()},
                   "maxdd": float(((1 + rs).cumprod() / (1 + rs).cumprod().cummax() - 1).min()),
                   "skew": float(rs.skew()), "alpha_t_xsmom14_btc": cm.spanning(rs, ctrl, ["xsmom14", "btc"])["alpha_t"]}
    o["worst5"] = {str(k.date()): float(v) for k, v in r.nsmallest(5).items()}
    o["sharpe_ex_worst_day"] = cm.sh(r.drop(r.nsmallest(1).index))
    W16 = cm.topn_weights(S, u50, n=8)
    r16 = cm.run(W16, 1)["df"]["r"]
    o["pos16_decomp"] = decomp(r16)
    rs16, nh16 = stop_overlay(W16)
    o["pos16_stop30"] = {"sharpe": cm.sh(rs16), "n_stops": nh16, **{f"sh_{k}": v for k, v in cm.subperiods(rs16).items()},
                         "sh_lag1_nostop": cm.run(W16, 1, extra_lag=1)["metrics"]["sharpe"],
                         "alpha_t_xsmom14_btc": cm.spanning(rs16, ctrl, ["xsmom14", "btc"])["alpha_t"]}
    out[f"RP{N}"] = o

S = scores(180)["RP"]
W16 = cm.topn_weights(S, u50, n=8)
o16, _ = cm.full_eval(W16, "RP180_16pos", u50, 1, 38, legs=False, ctrl=ctrl)
out["RP180_16pos"] = {k: o16[k] for k in ["sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "sh_cost2x", "sh_lag1",
                                          "alpha_t_xsmom14_btc"]}
print(json.dumps(out, indent=1, default=float))
json.dump(out, open("diag_rp.json", "w"), indent=1, default=float)
