"""POST-HOC variants (counted, +2 configs -> family A total 52):
 A6a short top-tercile 14d funding (top50), hedge with equal-$ long BTC+ETH (50/50), daily, net costs+funding
 A6b short top-tercile 14d funding (top50), hedge with equal-$ long EW of whole top50 universe, daily
Also: when does the candidate's rolling-365d Sharpe bottom out?"""
from common import *
F = lab.load("fund", "1d")
U = lab.universe("1d", top_k=50)
s = -F.rolling(14, min_periods=14).sum()
Wc = lab.quantile_ls_weights(s, U, q=0.33)
r = lab.backtest(Wc, "1d")["df"]["r"]
rs = r.rolling(365).apply(lambda x: x.mean() / x.std() * np.sqrt(365))
print("rolling-365 Sharpe min at", rs.idxmin(), round(rs.min(), 2), "; periods with rolling<0:",
      (rs < 0).groupby(rs.index.year).mean().round(2).to_dict())
Wsh = lab.quantile_ls_weights(s, U, q=0.33, short_only=True) * 0.5      # -0.5 gross on shorts
hedge_btc = pd.DataFrame(0.0, index=Wsh.index, columns=Wsh.columns)
on = (Wsh.abs().sum(axis=1) > 0)
hedge_btc.loc[on, "BTCUSDT"] = 0.25; hedge_btc.loc[on, "ETHUSDT"] = 0.25
ewl = U.astype(float).div(U.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5
for name, W in [("A6a_short_hedgeBTCETH", Wsh + hedge_btc), ("A6b_short_hedgeEW", Wsh + ewl)]:
    d, res = row(W, name, U)
    c2, l1 = stress(W)
    cb, cm = corr_ctrl(res["df"]["r"])
    print(name, "Sh %.2f CAGR %.3f MDD %.2f c2x %.2f lag1 %.2f corrBTC %.2f corrMom %.2f" % (d["sharpe"], d["cagr"], d["maxdd"], c2, l1, cb, cm),
          {k: round(d[k], 2) for k in ["sh_2020-21", "sh_2022", "sh_2023-24", "fund_yr", "sh_price_net", "npos"]})
    print(yearly_decomp(res).round(3).to_string())
