"""v4_risk part 2: regime dependence of the h1 candidate (IS 2020-03..2024-12, top-50 full universe, lab.backtest).

Descriptive splits (no trading rule): BTC above/below its 200d SMA (known at decision close),
universe-median F14 terciles (full-sample breakpoints - DESCRIPTIVE ONLY), calendar month, year-month table,
cross-sectional F14 dispersion terciles, and share of names pinned at the funding clamp.

PRE-REGISTERED regime switches (written before running; thresholds fixed ex ante, not tuned):
  S1 BTC trend gate : hold the book only when BTC close > its 200d SMA at the decision close, else flat.
  S2 leverage gate  : hold the book only when the universe-median F14 > 0.0042 (= 14 d x 3 settlements x the
                      0.01%/8h Binance base-interest clamp, i.e. longs are paying more than the neutral rate),
                      else flat. Rationale: no aggregate long crowding -> nothing to unwind, ranks are clamp noise.
  Decision rule: a switch is "justified" only if Sharpe improves >= 0.10, no sub-period (2020-21/2022/2023-24)
  gets worse, and the OFF-state standalone return of the base book is negative with NW t < -1.
"""
from common4 import *
import json

C = lab.load("close", "1d"); F = lab.load("fund", "1d")
U = lab.universe("1d", top_k=50)
W, s = candidate_weights(U, F)
F14 = -s
res = lab.backtest(W, "1d"); df = res["df"]; r = df["r"]
out = {}

def shift_state(x):
    """state known at decision close t applies to the return realised at t+1 (= backtest index t+1d)."""
    y = x.copy(); y.index = y.index + pd.Timedelta("1D"); return y.reindex(r.index)

def grp_stats(state, name):
    g = {}
    for k, x in r.groupby(state):
        g[str(k)] = {"days": int(len(x)), "ann_ret": round(float(x.mean() * 365), 4), "sharpe": round(sh(x), 2),
                     "t_nw": round(float(lab.newey_west_t(x.values, 10)), 2), "share_of_pnl": round(float(x.sum() / r.sum()), 3)}
    out[name] = g
    print(name, g)

btc = C["BTCUSDT"]; sma = btc.rolling(200, min_periods=200).mean()
bull = shift_state((btc > sma).where(sma.notna()))
grp_stats(bull.map({True: "BTC>SMA200", False: "BTC<SMA200"}).fillna("no_sma"), "by_btc_sma200")

medF = F14.where(U).median(axis=1)
mF = shift_state(medF)
q = mF.quantile([1 / 3, 2 / 3]).values
terc = pd.cut(mF, [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
out["medF14_tercile_breaks"] = [float(q[0]), float(q[1])]
grp_stats(terc.astype(str), "by_median_F14_tercile")
grp_stats(shift_state((medF > 0.0042)).map({True: "medF14>0.42%", False: "medF14<=0.42%"}), "by_median_F14_vs_clamp")

disp = shift_state(F14.where(U).quantile(0.833, axis=1) - F14.where(U).quantile(0.167, axis=1))
qd = disp.quantile([1 / 3, 2 / 3]).values
grp_stats(pd.cut(disp, [-np.inf, qd[0], qd[1], np.inf], labels=["low", "mid", "high"]).astype(str), "by_F14_dispersion_tercile")
clamp_share = shift_state(((F14 - 0.0042).abs() < 0.0003).where(U).mean(axis=1))
out["clamp_share_by_year"] = {int(k): round(float(v), 3) for k, v in clamp_share.groupby(clamp_share.index.year).mean().items()}
print("share of universe with F14 within +-0.03% of the 0.42% clamp:", out["clamp_share_by_year"])

# BTC 30d return regime
b30 = shift_state(btc / btc.shift(30) - 1)
grp_stats(pd.cut(b30, [-np.inf, -0.15, 0, 0.15, np.inf], labels=["<-15%", "-15..0", "0..15%", ">15%"]).astype(str), "by_btc_30d_ret")
# equal-weight alt-season proxy: EW universe 30d return minus BTC 30d
R = C.pct_change(fill_method=None)
ew = R.where(U.shift(1)).mean(axis=1)
alt30 = shift_state(np.expm1(np.log1p(ew).rolling(30).sum()) - (btc / btc.shift(30) - 1))
grp_stats(pd.cut(alt30, [-np.inf, -0.1, 0, 0.1, np.inf], labels=["alts<<BTC", "alts<BTC", "alts>BTC", "alts>>BTC"]).astype(str), "by_alt_vs_btc_30d")

# calendar month and year-month
mret = (1 + r).groupby(r.index.to_period("M") if r.index.tz is None else r.index.tz_localize(None).to_period("M")).prod() - 1
out["monthly"] = {str(k): round(float(v), 4) for k, v in mret.items()}
out["pct_pos_months"] = round(float((mret > 0).mean()), 3)
out["worst_months"] = [(str(k), round(float(v), 4)) for k, v in mret.nsmallest(8).items()]
cal = mret.groupby(mret.index.month).agg(["mean", "median", lambda x: (x > 0).mean()])
cal.columns = ["mean", "median", "pct_pos"]
out["calendar_month"] = {int(k): {c_: round(float(v), 4) for c_, v in row.items()} for k, row in cal.iterrows()}
print("worst months", out["worst_months"], "pct pos months", out["pct_pos_months"])
ym = mret.copy(); ym.index = pd.MultiIndex.from_arrays([ym.index.year, ym.index.month])
print((ym.unstack() * 100).round(1).to_string())
# losing streaks: rolling 90d / 180d returns
for wdw in (90, 180, 365):
    x = np.expm1(np.log1p(r).rolling(wdw).sum())
    out[f"rolling_{wdw}d"] = {"min": round(float(x.min()), 4), "min_end": str(x.idxmin().date()), "share_negative": round(float((x < 0).mean()), 3)}
print({k: out[k] for k in out if k.startswith("rolling")})

# leg split by BTC regime
Rn = R.shift(-1)
legL = (W.clip(lower=0) * Rn).sum(axis=1) + 0; legS = (W.clip(upper=0) * Rn).sum(axis=1)
legL.index = legL.index + pd.Timedelta("1D"); legS.index = legS.index + pd.Timedelta("1D")
st = bull.map({True: "bull", False: "bear"}).fillna("na")
out["leg_price_ann_by_btc"] = {k: {"long_leg": round(float(legL.reindex(r.index)[st == k].mean() * 365), 4),
                                   "short_leg": round(float(legS.reindex(r.index)[st == k].mean() * 365), 4)} for k in ("bull", "bear")}
print("leg price PnL (ann, before cost/funding) by BTC regime", out["leg_price_ann_by_btc"])

# pre-registered switches
def sub(rr):
    o = {}
    for nm, a, b in [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"), ("2023-24", "2023-01-01", "2025-01-01")]:
        o[nm] = round(sh(rr[(rr.index >= pd.Timestamp(a, tz="UTC")) & (rr.index < pd.Timestamp(b, tz="UTC"))]), 3)
    return o
out["base"] = {"sharpe": round(sh(r), 3), "sub": sub(r)}
gates = {"S1_btc_sma200": (btc > sma), "S2_medF14_gt_clamp": (medF > 0.0042)}
for nm, g in gates.items():
    Wg = W.mul(g.reindex(W.index).fillna(False).astype(float), axis=0)
    rg = lab.backtest(Wg, "1d")
    rr = rg["df"]["r"].reindex(r.index).fillna(0.0)
    off = shift_state(~g.reindex(W.index).fillna(False))
    xo = r[off.fillna(False).astype(bool)]
    out[nm] = {"sharpe": round(sh(rr), 3), "cagr": round(float((1 + rr).prod() ** (365 / len(rr)) - 1), 4),
               "maxdd": round(float(((1 + rr).cumprod() / (1 + rr).cumprod().cummax() - 1).min()), 4), "sub": sub(rr),
               "share_time_on": round(float(1 - off.fillna(False).astype(bool).mean()), 3),
               "off_state_base_ann": round(float(xo.mean() * 365), 4), "off_state_t_nw": round(float(lab.newey_west_t(xo.values, 10)), 2)}
    print(nm, out[nm])
json.dump(out, open(os.path.join(HERE, "t2_regime.json"), "w"), indent=1, default=str)
