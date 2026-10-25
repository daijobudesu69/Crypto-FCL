"""v2_overfit / step 2: parameter surface around the candidate.

PRE-REGISTERED SURFACE (written before running; ALL cells are reported, none is selected):
  score     = -(sum of the last L days of Binance funding settlements), known at close t
  L         in {5, 7, 10, 14, 21, 30}
  fraction  q per side in {1/5, 1/4, 1/3, 1/2}, implemented as lab.quantile_ls_weights(q) with
            q in {0.20, 0.25, 0.33, 0.50}  (candidate uses q=0.33; q=0.5 -> the median name can sit in
            both legs and nets to ~0)
  universe  lab.universe("1d", min_hist_days=60, top_k=k), k in {30, 50, 75, 100}
  fixed     equal weight per leg, each leg 0.5 gross, daily rebalance, cost 0.07%/side + actual funding
  -> 6 x 4 x 4 = 96 cells. Per cell: Sharpe, CAGR, MaxDD, sub-period Sharpe (2020-21, 2022, 2023-24),
     Sharpe ex-2021, Sharpe at 2x cost. These are verification cells, not new candidates.
  Also: effective number of independent configs (participation ratio of the eigenvalues of the
  daily-return correlation matrix) to judge how much 'n_trials' overstates independent tries.
"""
from vlib import *
import json

Ls = [5, 7, 10, 14, 21, 30]
qs = [0.20, 0.25, 0.33, 0.50]
ks = [30, 50, 75, 100]
Us = {k: lab.universe("1d", top_k=k) for k in ks}
S = {L: signal(L) for L in Ls}
rows, rets = [], {}
for L in Ls:
    for q in qs:
        for k in ks:
            W = lab.quantile_ls_weights(S[L], Us[k], q=q)
            res = lab.backtest(W, "1d")
            r = res["df"]["r"]; m = res["metrics"]
            r = r[r.index >= pd.Timestamp("2020-03-15", tz="UTC")]     # common start for comparability
            d = {"L": L, "q": q, "k": k, "sharpe": sh(r), "cagr_full": m["cagr"], "maxdd_full": m["maxdd"],
                 "sh_2020-21": sh(between(r, "2020-01-01", "2022-01-01")), "sh_2022": sh(between(r, "2022-01-01", "2023-01-01")),
                 "sh_2023-24": sh(between(r, "2023-01-01", "2025-01-01")), "sh_ex2021": sh(r[r.index.year != 2021]),
                 "sh_cost2x": lab.backtest(W, "1d", cost=2 * lab.COST)["metrics"]["sharpe"], "npos": m["avg_n_pos"],
                 "turn": m["turnover_per_year"], "fund_yr": m["funding_per_year"]}
            rows.append(d); rets[f"L{L}_q{q}_k{k}"] = r
            print(f"L{L:>2} q{q:.2f} k{k:>3}  SR {d['sharpe']:.2f}  20-21 {d['sh_2020-21']:.2f}  22 {d['sh_2022']:.2f}  "
                  f"23-24 {d['sh_2023-24']:.2f}  ex21 {d['sh_ex2021']:.2f}  2xc {d['sh_cost2x']:.2f}", flush=True)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "surface.csv"), index=False)
R = pd.DataFrame(rets).dropna()
C = R.corr().values
ev = np.linalg.eigvalsh(C)
neff = float(ev.sum() ** 2 / (ev ** 2).sum())
pd.set_option("display.width", 250)
summ = {"n_cells": len(out), "effective_n_participation": neff,
        "mean_pairwise_corr": float((C.sum() - len(C)) / (len(C) * (len(C) - 1)))}
for col in ["sharpe", "sh_2020-21", "sh_2022", "sh_2023-24", "sh_ex2021", "sh_cost2x"]:
    summ[col] = {"median": float(out[col].median()), "min": float(out[col].min()), "max": float(out[col].max()),
                 "pct_gt_1": float((out[col] > 1).mean()), "pct_gt_0.5": float((out[col] > 0.5).mean()),
                 "pct_lt_0": float((out[col] < 0).mean())}
c = out[(out.L == 14) & (out.q == 0.33) & (out.k == 50)].iloc[0]
summ["candidate_cell"] = c.to_dict()
summ["candidate_rank_of_96"] = int((out["sharpe"] > c["sharpe"]).sum() + 1)
summ["candidate_rank_2023-24_of_96"] = int((out["sh_2023-24"] > c["sh_2023-24"]).sum() + 1)
# immediate neighbours: one step in exactly one dimension
iL, iq, ik = Ls.index(14), qs.index(0.33), ks.index(50)
nb = []
for dL, dq, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
    a, b, e = iL + dL, iq + dq, ik + dk
    if 0 <= a < len(Ls) and 0 <= b < len(qs) and 0 <= e < len(ks):
        nb.append(out[(out.L == Ls[a]) & (out.q == qs[b]) & (out.k == ks[e])].iloc[0])
nb = pd.DataFrame(nb)
summ["neighbours"] = nb[["L", "q", "k", "sharpe", "sh_2023-24", "sh_ex2021"]].round(3).to_dict(orient="records")
summ["neighbour_median_sharpe"] = float(nb["sharpe"].median())
summ["neighbour_median_2023-24"] = float(nb["sh_2023-24"].median())
# marginal medians
for dim in ["L", "q", "k"]:
    summ[f"median_by_{dim}"] = out.groupby(dim)[["sharpe", "sh_2023-24", "sh_ex2021"]].median().round(3).to_dict(orient="index")
print(json.dumps(summ, indent=1, default=float))
for col in ["sharpe", "sh_2023-24", "sh_ex2021"]:
    print(f"\n== {col}: rows L, cols (q,k)")
    print(out.pivot_table(index="L", columns=["q", "k"], values=col).round(2).to_string())
json.dump(summ, open(os.path.join(HERE, "s2_surface.json"), "w"), indent=1, default=float)
