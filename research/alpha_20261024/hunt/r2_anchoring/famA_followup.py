"""Family A follow-up (pre-registered in famA_grid.py header): full stress + spanning for
  (1) best net-Sharpe config  = RP N=90 rb=1
  (2) best PHR (anchoring-net-of-momentum) config = PHR N=365 rb=7
  (3) extra (most consistent sub-periods)          = RP N=365 rb=1
plus 8L/8S version, universe top_k 30/75, full Binance top-50. Extra trials counted: 3 x 4 = 12.
"""
import json
import numpy as np, pandas as pd
import common as cm
import lab
from famA_grid import scores

N_TRIALS = 24 + 12
ctrl = cm.controls()
g = pd.read_csv("famA_grid.csv")


def neigh(sc, N, rb):
    sub = g[(g.score == sc) & (((g.N == N) & (g.rebal != rb)) | ((g.N != N) & (g.rebal == rb)))]
    return float(sub.sharpe.median()), sub[["N", "rebal", "sharpe"]].values.tolist()


out = {}
S_cache = {}
for sc, N, rb in [("RP", 90, 1), ("PHR", 365, 7), ("RP", 365, 1)]:
    if N not in S_cache:
        S_cache[N] = scores(N)
    S = S_cache[N][sc]
    u50 = cm.U(50)
    W = lab.quantile_ls_weights(S, u50, q=0.2)
    mm = cm.mom_ctrl(N, 0, rebal=rb)
    o, res = cm.full_eval(W, f"{sc}{N}_rb{rb}", u50, rebal=rb, n_trials=N_TRIALS, ctrl=ctrl, mom_match=mm)
    o["neigh_median"], o["neigh"] = neigh(sc, N, rb)
    # short-horizon momentum spanning (daily rebal signals may just be 1-14d continuation)
    s = cm.spanning(res["df"]["r"], ctrl, ["hmom14", "btc"])
    o["alpha_t_hmom14_btc"] = s["alpha_t"]
    # 16-position version
    W16 = cm.topn_weights(S, u50, n=8)
    o16, _ = cm.full_eval(W16, "16pos", u50, rebal=rb, n_trials=N_TRIALS, legs=False, ctrl=ctrl)
    o["pos16"] = {k: o16[k] for k in ["sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "sh_cost2x", "sh_lag1",
                                      "npos", "mineq_med", "alpha_t_xsmom14_btc", "alpha_t_mom90_btc", "turn"]}
    # universe robustness
    for k in (30, 75):
        uk = cm.U(k)
        Sk = S if sc != "PHR" else S  # PHR residualised on top-50 cross-section (kept fixed)
        ok, _ = cm.quick(lab.quantile_ls_weights(Sk, uk, q=0.2), rb)
        o[f"u{k}"] = {kk: ok[kk] for kk in ["sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "npos"]}
    of, _ = cm.quick(lab.quantile_ls_weights(S, cm.U(50, hype=False), q=0.2), rb)
    o["binance50"] = {kk: of[kk] for kk in ["sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "npos"]}
    out[o["name"]] = o
    print(json.dumps(o, indent=1, default=float), flush=True)

json.dump(out, open("famA_followup.json", "w"), indent=1, default=float)
