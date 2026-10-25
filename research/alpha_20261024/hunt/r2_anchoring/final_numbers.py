"""Final numbers for the structured report (no new selection).
Candidate presentations:
  A1 = RP365 rb1 quintile (hype top-50)                 (grid config)
  A2 = RP365 rb1 8L/8S (16 positions) + 30% resting short stop (deployable form)
  A3 = RP90  rb1 quintile                                (grid best by Sharpe)
Trial counts: family A = 24 grid + 12 follow-up variants + 1 (RP180 16pos) + 4 stop overlays (diag_rp)
  + 2 stop levels (20/40%) + 2 neighbour 16pos+stop books = 45.
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import common as cm, lab
from famA_grid import scores

N_A = 45
ctrl = cm.controls()
c = lab.load("close", "1d"); h = lab.load("high", "1d")
u50 = cm.U(50)


def overlay_run(W, stop=0.30, slip=0.01, cost=cm.COST, extra_lag=0):
    Wl = W.shift(extra_lag).fillna(0.0) if extra_lag else W
    res = cm.run(Wl, 1, cost=cost)
    df = res["df"].copy()
    R = cm.R()
    Wv = Wl.reindex(index=c.index, columns=c.columns).fillna(0.0)
    hi_ret = (h / c.shift(1) - 1).shift(-1)
    r_next = R.shift(-1)
    hit = (Wv < 0) & (hi_ret >= stop)
    adj = (Wv.where(hit, 0.0) * ((stop + slip) - r_next.where(hit, 0.0))).sum(axis=1)
    extra = Wv.abs().where(hit, 0.0).sum(axis=1) * 2 * cost
    d = (adj - extra)
    d.index = d.index + pd.Timedelta("1D")
    extra.index = extra.index + pd.Timedelta("1D")
    df["r"] = df["r"] + d.reindex(df.index).fillna(0.0)
    df["cost_r"] = df["cost_r"] + extra.reindex(df.index).fillna(0.0)
    return df, int(hit.sum().sum())


def summarize(df, W, name, legs_fn=None):
    r = df["r"]
    m = lab.metrics(r, "1d", df)
    o = {"name": name, "sharpe": m["sharpe"], "cagr": m["cagr"], "maxdd": m["maxdd"], "t_nw": m["t_nw"],
         "skew": m["skew"], "kurt": m["kurt"], "turn": m["turnover_per_year"], "cost_yr": m["cost_drag_per_year"],
         "fund_yr": m["funding_per_year"], "npos": m["avg_n_pos"], "days": m["days"]}
    o.update({"sh_" + k: v for k, v in cm.subperiods(r).items()})
    o["yearly"] = cm.yearly_str(r)
    o["corr_btc"] = cm.corr(r, ctrl["btc"]); o["corr_xsmom14"] = cm.corr(r, ctrl["xsmom14"])
    o["alpha_t_xsmom14_btc"] = cm.spanning(r, ctrl, ["xsmom14", "btc"])["alpha_t"]
    o["alpha_t_hmom14_btc"] = cm.spanning(r, ctrl, ["hmom14", "btc"])["alpha_t"]
    o["alpha_t_mom90_btc"] = cm.spanning(r, ctrl, ["hmom90", "btc"])["alpha_t"]
    o["dsr"] = lab.deflated_sharpe(m["sharpe"], N_A, m["days"], m["skew"], m["kurt"] + 3)
    o["mineq_med"], o["mineq_p90"] = cm.min_equity(W)
    return o


out = {}
# A2: 16 positions + stop
S = scores(365)["RP"]
W16 = cm.topn_weights(S, u50, n=8)
df, nst = overlay_run(W16)
o = summarize(df, W16, "RP365_16pos_stop30")
o["n_stops"] = nst
o["sh_cost2x"] = cm.sh(overlay_run(W16, cost=2 * cm.COST)[0]["r"])
o["sh_lag1"] = cm.sh(overlay_run(W16, extra_lag=1)[0]["r"])
ew = cm.ew_univ(u50)
for side, Wl in [("long", W16.clip(lower=0)), ("short", W16.clip(upper=0))]:
    dl, _ = overlay_run(Wl * 2)
    e = ew.reindex(dl.index).fillna(0)
    o["sh_" + side] = cm.sh(dl["r"])
    o["sh_" + side + "_vsEW"] = cm.sh(dl["r"] - e) if side == "long" else cm.sh(dl["r"] + e)
# neighbours: same overlay on 16pos RP90 / RP180, and stop 20% / 40% (sensitivity, not selection)
nb = {}
for N in (90, 180):
    nb[f"RP{N}_16pos_stop30"] = cm.sh(overlay_run(cm.topn_weights(scores(N)["RP"], u50, n=8))[0]["r"])
for st in (0.2, 0.4):
    nb[f"RP365_16pos_stop{int(st*100)}"] = cm.sh(overlay_run(W16, stop=st)[0]["r"])
nb["RP365_16pos_nostop"] = cm.run(W16, 1)["metrics"]["sharpe"]
o["neighbours"] = nb
o["neigh_median"] = float(np.median([nb["RP90_16pos_stop30"], nb["RP180_16pos_stop30"]]))
for k in (30, 75):
    o[f"sh_u{k}"] = cm.sh(overlay_run(cm.topn_weights(S, cm.U(k), n=8))[0]["r"])
out["A2"] = o
print(json.dumps(o, indent=1, default=float))
json.dump(out, open("final_numbers.json", "w"), indent=1, default=float)
