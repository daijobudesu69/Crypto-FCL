"""v4_risk part 4: correlation of the candidate with proxies of the user's existing books (IS 2020-03..2024-12),
plus half-year Sharpe (decay check) and realised per-episode loss of single shorts scaled to 5-8 names per side.

Proxies (defined before running, daily bars, top-50 point-in-time universe, net of 0.07%/side + funding):
  RMF  : 14d XS momentum long-only, top-10 by 14d return, equal weight, gross 1, flat when BTC < EMA50.
  RMFnf: same without the BTC filter.
  CLRs : Donchian-20 breakdown short: trigger at close t if close < min(low[t-20..t-1]) and EMA50 < EMA200 and
         that day's funding sum > 0.0003 (> 0.01%/8h); hold 5 days after the last trigger; -0.2 per name,
         total short gross capped at 1.0.
  RNT  : risk-adjusted RS long/short on top-20: score = 30d return / 30d stdev, long top-4 / short bottom-4,
         equal weight, 0.5 gross per side, daily.
  XSM  : lab.controls() 14d XS momentum quintile L/S (brief's standard control).
"""
from common4 import *
import json

C = lab.load("close", "1d"); L = lab.load("low", "1d"); F = lab.load("fund", "1d")
R = C.pct_change(fill_method=None)
U = lab.universe("1d", top_k=50); U20 = lab.universe("1d", top_k=20)
W, s = candidate_weights(U, F)
cand = lab.backtest(W, "1d")["df"]["r"]
out = {}

btc = C["BTCUSDT"]; ema50b = btc.ewm(span=50, adjust=False).mean()
mom = C / C.shift(14) - 1
rk = mom.where(U).rank(axis=1, ascending=False)
top10 = (rk <= 10).astype(float)
Wrmf_nf = top10.div(top10.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
Wrmf = Wrmf_nf.mul((btc > ema50b).astype(float), axis=0)

e50 = C.ewm(span=50, adjust=False).mean(); e200 = C.ewm(span=200, adjust=False).mean()
don = L.shift(1).rolling(20, min_periods=20).min()
trig = (C < don) & (e50 < e200) & (F > 0.0003) & U
active = trig.astype(float).rolling(5, min_periods=1).max().astype(bool) & U
na = active.sum(axis=1)
Wclr = -active.astype(float).mul(np.minimum(0.2, 1.0 / na.replace(0, np.nan)), axis=0).fillna(0)

vol30 = R.rolling(30, min_periods=20).std()
sc = (C / C.shift(30) - 1) / vol30
r20 = sc.where(U20).rank(axis=1, ascending=False); n20 = sc.where(U20).notna().sum(axis=1)
lo = (r20 <= 4).astype(float); shn = (r20.gt(n20 - 4, axis=0) & sc.where(U20).notna()).astype(float)
Wrnt = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5 - shn.div(shn.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5

books = {"RMF_top10_btcfilter": Wrmf, "RMF_top10_nofilter": Wrmf_nf, "CLR_donchian_short": Wclr, "RNT_rs_ls_top20": Wrnt}
rets = {"candidate": cand}
for k, Wx in books.items():
    res = lab.backtest(Wx, "1d")
    rets[k] = res["df"]["r"]
    out.setdefault("proxy_metrics", {})[k] = {"sharpe": round(res["metrics"]["sharpe"], 2), "cagr": round(res["metrics"]["cagr"], 3), "npos": round(res["metrics"]["avg_n_pos"], 1),
                                              "share_days_active": round(float((res["df"]["gross_exposure"] > 0).mean()), 3)}
ctrl = lab.controls()
rets["XSM_quintile_ls"] = ctrl["xsmom14"]; rets["BTC"] = ctrl["btc"]
D = pd.DataFrame(rets).loc[cand.index]
D = D[D.index >= pd.Timestamp("2020-03-01", tz="UTC")]
print(out["proxy_metrics"])
cm = D.corr()
out["daily_corr_with_candidate"] = {k: round(float(v), 3) for k, v in cm["candidate"].items()}
Wk = np.expm1(np.log1p(D.fillna(0)).resample("W").sum())
out["weekly_corr_with_candidate"] = {k: round(float(v), 3) for k, v in Wk.corr()["candidate"].items()}
# tail-conditional: on the candidate's worst 5% days, what did the other books do?
bad = D["candidate"] <= D["candidate"].quantile(0.05)
out["mean_ret_on_candidate_worst5pct_days"] = {k: round(float(D.loc[bad, k].mean()), 4) for k in D.columns}
badm = D["RMF_top10_btcfilter"] <= D["RMF_top10_btcfilter"].quantile(0.05)
out["mean_ret_on_RMF_worst5pct_days"] = {k: round(float(D.loc[badm, k].mean()), 4) for k in D.columns}
# by year correlations with RMF and CLR proxies
for k in ["RMF_top10_btcfilter", "RMF_top10_nofilter", "CLR_donchian_short", "RNT_rs_ls_top20", "XSM_quintile_ls", "BTC"]:
    out.setdefault("corr_by_year", {})[k] = {int(y): round(float(g["candidate"].corr(g[k])), 3) for y, g in D.groupby(D.index.year)}
print("daily corr", out["daily_corr_with_candidate"]); print("weekly corr", out["weekly_corr_with_candidate"])
print("on candidate worst 5% days", out["mean_ret_on_candidate_worst5pct_days"]); print("on RMF worst 5% days", out["mean_ret_on_RMF_worst5pct_days"])
print("corr by year", out["corr_by_year"])
# short-leg vs CLR short: name overlap
cs = (W < 0); clr = (Wclr < 0)
out["clr_names_also_in_candidate_short_leg"] = round(float((cs & clr).sum().sum() / max(clr.sum().sum(), 1)), 3)
out["clr_names_in_candidate_long_leg"] = round(float(((W > 0) & clr).sum().sum() / max(clr.sum().sum(), 1)), 3)
rm = (Wrmf_nf > 0)
out["rmf_names_in_candidate_short_leg"] = round(float((cs & rm).sum().sum() / rm.sum().sum()), 3)
out["rmf_names_in_candidate_long_leg"] = round(float(((W > 0) & rm).sum().sum() / rm.sum().sum()), 3)
print({k: out[k] for k in out if "names" in k})
# simple blend: 50/50 capital candidate + RMF proxy, and inverse-vol blend
for nm, wts in [("50/50 cand+RMF", (0.5, 0.5)), ("RMF only", (0.0, 1.0)), ("cand only", (1.0, 0.0))]:
    b = wts[0] * D["candidate"].fillna(0) + wts[1] * D["RMF_top10_btcfilter"].fillna(0)
    e = (1 + b).cumprod()
    out.setdefault("blend", {})[nm] = {"sharpe": round(sh(b), 3), "maxdd": round(float((e / e.cummax() - 1).min()), 4), "cagr": round(float(e.iloc[-1] ** (365 / len(b)) - 1), 4)}
print(out["blend"])

# half-year Sharpe of candidate (decay)
hy = cand.groupby([cand.index.year, (cand.index.month > 6).astype(int) + 1]).apply(sh)
out["halfyear_sharpe"] = {f"{a}H{b}": round(float(v), 2) for (a, b), v in hy.items()}
hr = cand.groupby([cand.index.year, (cand.index.month > 6).astype(int) + 1]).apply(lambda x: (1 + x).prod() - 1)
out["halfyear_ret"] = {f"{a}H{b}": round(float(v), 3) for (a, b), v in hr.items()}
print("half-year Sharpe", out["halfyear_sharpe"]); print("half-year ret", out["halfyear_ret"])

# realised per-episode contribution of single shorts in the actual book, and scaled to 5 / 8 names per side
Rn = R.shift(-1)
contrib = (W * Rn).where(W < 0)
scaled5 = (-0.1 * Rn).where(W < 0)       # constant 10% short (5 names/side at gross 1), rebalanced daily
scaled8 = (-0.0625 * Rn).where(W < 0)
inS = (W < 0)
epid = (inS != inS.shift(1)).cumsum().where(inS)
rows = []
for sym in W.columns:
    if not inS[sym].any():
        continue
    g = pd.DataFrame({"ep": epid[sym], "c": contrib[sym], "s5": scaled5[sym], "s8": scaled8[sym]}).dropna(subset=["ep"])
    agg = g.groupby("ep").agg(c=("c", "sum"), s5=("s5", "sum"), s8=("s8", "sum"))
    starts = g.reset_index().groupby("ep")["ts"].first()
    for ep_, rw in agg.iterrows():
        rows.append((sym, str(starts[ep_].date()), float(rw["c"]), float(rw["s5"]), float(rw["s8"])))
E = pd.DataFrame(rows, columns=["sym", "start", "actual", "w10pct", "w6p25pct"]).sort_values("w10pct")
out["worst_short_episodes_realised"] = E.head(10).round(4).values.tolist()
print(E.head(10).round(4).to_string())
json.dump(out, open(os.path.join(HERE, "t4_corr.json"), "w"), indent=1, default=str)
