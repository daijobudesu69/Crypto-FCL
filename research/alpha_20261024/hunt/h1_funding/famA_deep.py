"""Family A deep-dive on a few grid configs (no new signal configs here, just diagnostics):
stress (2x cost, lag1), correlations to controls, yearly price/funding/cost split, min equity, DSR.
Also A4 implementation variants (POST-HOC, counted as extra configs: 6):
  tercile q=0.33 EW, L in {7,14}, top50, rb in {1,3}  (4)
  rank-weighted on top-30 universe, L in {7,14}, rb 1   (2)
"""
from common import *
F = lab.load("fund", "1d")
Us = {30: lab.universe("1d", top_k=30), 50: lab.universe("1d", top_k=50), 100: lab.universe("1d", top_k=100)}


def W_of(L, k, wt, q=0.2):
    s = -F.rolling(L, min_periods=L).sum()
    U = Us[k]
    if wt == "q":
        return lab.quantile_ls_weights(s, U, q=q), U
    return lab.xs_demean_weights(lab.xs_rank(s, U), U), U


cands = [("A1_L14_top100_r_rb1", 14, 100, "r", 0.2, 1), ("A1_L14_top50_r_rb1", 14, 50, "r", 0.2, 1),
         ("A1_L7_top50_r_rb1", 7, 50, "r", 0.2, 1), ("A1_L14_top50_q_rb3", 14, 50, "q", 0.2, 3),
         ("A1_L7_top50_q_rb1", 7, 50, "q", 0.2, 1),
         ("A4_L7_top50_t_rb1", 7, 50, "q", 0.33, 1), ("A4_L7_top50_t_rb3", 7, 50, "q", 0.33, 3),
         ("A4_L14_top50_t_rb1", 14, 50, "q", 0.33, 1), ("A4_L14_top50_t_rb3", 14, 50, "q", 0.33, 3),
         ("A4_L7_top30_r_rb1", 7, 30, "r", 0.2, 1), ("A4_L14_top30_r_rb1", 14, 30, "r", 0.2, 1)]
rows = []
for name, L, k, wt, q, rb in cands:
    W, U = W_of(L, k, wt, q)
    d, res = row(W, name, U, rebalance_every=rb)
    c2, l1 = stress(W, rb)
    cb, cm = corr_ctrl(res["df"]["r"])
    d.update(sh_cost2x=c2, sh_lag1=l1, corr_btc=cb, corr_xsmom=cm, min_eq=min_equity(W),
             dsr42=lab.deflated_sharpe(d["sharpe"], 48, d["days"], d["skew"], d["kurt"] + 3))
    rows.append(d)
    print("\n==", name)
    lab.report(res, name)
    print(yearly_decomp(res).round(3).to_string())
out = pd.DataFrame(rows)
out.to_csv("famA_deep.csv", index=False)
pd.set_option("display.width", 250)
print(out[["name", "sharpe", "cagr", "maxdd", "sh_cost2x", "sh_lag1", "corr_btc", "corr_xsmom", "sh_long_vsEW",
           "sh_short_vsEW", "sh_long", "sh_short", "min_eq", "npos", "turn", "dsr42", "sh_2020-21", "sh_2022", "sh_2023-24"]].round(2).to_string())
