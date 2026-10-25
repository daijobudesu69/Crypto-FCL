"""Collect final candidate numbers (no new configs)."""
from common import *
import json
F = lab.load("fund", "1d"); P = lab.load("close", "1d"); S = lab.load("spot_close", "1d")
U = lab.universe("1d", top_k=50)
s = -F.rolling(14, min_periods=14).sum()
out = {}
for name, W, rb in [("A4_L14_top50_tercile_rb1", lab.quantile_ls_weights(s, U, q=0.33), 1),
                    ("A1_L14_top50_rank_rb1", lab.xs_demean_weights(lab.xs_rank(s, U), U), 1)]:
    res = lab.backtest(W, "1d", rebalance_every=rb); m = res["metrics"]
    y = lab.yearly(res["df"]["r"]); out[name] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()}
    out[name]["yearly"] = {int(k): round(float(v), 3) for k, v in y["ret"].items()}
    out[name]["yearly_sharpe"] = {int(k): round(float(v), 2) for k, v in y["sharpe"].items()}
# B3 always-on BTC+ETH carry
rp = P.pct_change(fill_method=None); rs = S.pct_change(fill_method=None)
cols = ["BTCUSDT", "ETHUSDT"]
ok = (rp[cols].shift(-1).notna() & rs[cols].shift(-1).notna() & S[cols].notna())
Wt = ok.astype(float).div(ok.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
for tag, cm, lag in [("base", 1, 0), ("cost2x", 2, 0), ("lag1", 1, 1)]:
    W_ = Wt.shift(lag).fillna(0) if lag else Wt
    W_ = W_.where(ok, 0)
    b = (W_ * (rs[cols] - rp[cols]).shift(-1).fillna(0)).sum(axis=1)
    f = (W_ * F[cols].shift(-1).fillna(0)).sum(axis=1)
    t = W_.diff().abs().sum(axis=1); t.iloc[0] = W_.iloc[0].sum()
    r = (b + f - t * 0.0017 * cm); r.index = r.index + pd.Timedelta("1D"); r = r.iloc[:-1]
    r = r[r.index >= pd.Timestamp("2020-01-02", tz="UTC")]
    eq = (1 + r).cumprod()
    out.setdefault("B3", {})[tag] = {"sharpe": sh(r), "cagr": float(eq.iloc[-1] ** (365 / len(r)) - 1),
                                     "maxdd": float((eq / eq.cummax() - 1).min())}
    if tag == "base":
        cb, cmo = corr_ctrl(r)
        out["B3"].update(corr_btc=cb, corr_xsmom=cmo, t_nw=lab.newey_west_t(r.values, 10),
                         skew=float(r.skew()), kurt=float(r.kurt()), days=len(r),
                         yearly={int(k): round(float(v), 3) for k, v in ((1 + r).groupby(r.index.year).prod() - 1).items()},
                         sub=subperiods(r), fund_yr=float(f.mean() * 365), basis_yr=float(b.mean() * 365),
                         dsr19=lab.deflated_sharpe(sh(r), 19, len(r), float(r.skew()), float(r.kurt()) + 3))
print(json.dumps(out, indent=1, default=float))
json.dump(out, open("final_numbers.json", "w"), indent=1, default=float)
