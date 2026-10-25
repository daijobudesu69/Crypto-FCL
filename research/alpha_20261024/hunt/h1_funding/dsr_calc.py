from common import *
F = lab.load("fund", "1d"); U = lab.universe("1d", top_k=50)
s = -F.rolling(14, min_periods=14).sum()
Wsh = lab.quantile_ls_weights(s, U, q=0.33, short_only=True) * 0.5
ewl = U.astype(float).div(U.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5
cands = {"A4_tercile": lab.quantile_ls_weights(s, U, q=0.33), "A1_rank": lab.xs_demean_weights(lab.xs_rank(s, U), U), "A6b": Wsh + ewl}
for k, W in cands.items():
    m = lab.backtest(W, "1d")["metrics"]
    print(k, "SR %.2f t %.2f DSR52 %.3f DSR71 %.3f minEq %.0f turn %.1f cost %.3f fund %.3f mdd %.3f cagr %.3f" % (
        m["sharpe"], m["t_nw"], lab.deflated_sharpe(m["sharpe"], 52, m["days"], m["skew"], m["kurt"] + 3),
        lab.deflated_sharpe(m["sharpe"], 71, m["days"], m["skew"], m["kurt"] + 3), min_equity(W),
        m["turnover_per_year"], m["cost_drag_per_year"], m["funding_per_year"], m["maxdd"], m["cagr"]))
