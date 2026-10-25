"""FAMILY B follow-up: stress + plateau + decomposition of the family-best configs.

Pre-registered selection: best of B1/B2 by Sharpe = B1_disp7_MED (1.39; B2_disp7_MED ties at 1.39 but
has MaxDD -77%, so the market-neutral spread is the one we stress); best overlay B3/B4 = B4_trend_AND_disp7.
POST-HOC configs (counted, family total 39 + 9 = 48):
 neighbours of disp7 MED: dispersion lookback in {3, 5, 10, 14}d; percentile window in {180, 730}d   = 6
 disp7 computed only on members with >= 180d history (new listings excluded from the signal)          = 1
 control: spread trade timed by BTC 30d realised-vol percentile (high vol = favourable)               = 1
 control: spread time-series momentum 7d                                                              = 1
"""
import numpy as np, pandas as pd, json, warnings
import r2lib
from famB_regime import (C, R, LR, U50, BTC, signals, trail_pct, state, spread_w, overlay_w, ew_index,
                         rotation_w, ew_weights)
warnings.filterwarnings("ignore")
lab = r2lib.lab
N_TRIALS = 48
out = {}


def disp(U, L=7, min_hist=None):
    m = U
    if min_hist is not None:
        m = U & (C.notna().cumsum() >= min_hist)
    return np.log(C / C.shift(L)).where(m).std(axis=1)


def st_from(sig, win=365, kind="MED", sign=1):
    p = trail_pct(sig, win=win, minp=min(120, win // 2))
    if sign < 0:
        p = 1 - p
    return state(p, kind)


def align(s, idx):
    s2 = s.copy(); s2.index = s2.index + pd.Timedelta("1D")
    return s2.reindex(idx)


def show(o, keys=None):
    print(r2lib.fmt(o) if keys is None else r2lib.fmt(o, keys=keys), flush=True)


K = ("sharpe", "cagr", "maxdd", "sh_2020-21", "sh_2022", "sh_2023-24", "turnover_per_year")

# ---------------------------------------------------------------- B1_disp7 core
st7 = st_from(disp(U50, 7))
st7t = st_from(disp(U50, 7), kind="TERC")
W = spread_w(st7, U50)
Wt = spread_w(st7t, U50)
for nm, WW in (("B1_disp7_MED", W), ("B1_disp7_TERC", Wt)):
    o, res = r2lib.evaluate(WW, nm, U=U50, full=True, n_trials=N_TRIALS)
    show(o); print("   yearly:", o["yearly"]); out[nm] = o
    d = res["df"]
    sa = align(st7 if nm.endswith("MED") else st7t, d.index)
    for sv in (1, -1):
        x = d["r"][sa == sv]
        print(f"   state {sv:+d}: days {len(x)}  mean/day {x.mean()*1e4:.1f}bp  ann {x.mean()*365:+.1%}  SR {r2lib.sharpe(x):.2f}")
        out[nm][f"state{sv}_ann"] = float(x.mean() * 365); out[nm][f"state{sv}_sr"] = r2lib.sharpe(x)
    # alt leg vs BTC leg
    Walt = WW.copy(); Walt[BTC] = 0.0
    Wbtc = WW * 0.0; Wbtc[BTC] = WW[BTC]
    ra = lab.backtest(Walt, "1d", cost=r2lib.COST)["metrics"]; rb = lab.backtest(Wbtc, "1d", cost=r2lib.COST)["metrics"]
    print(f"   alt-leg SR {ra['sharpe']:.2f} CAGR {ra['cagr']:+.1%} | BTC-leg SR {rb['sharpe']:.2f} CAGR {rb['cagr']:+.1%}")
    out[nm]["alt_leg_sharpe"] = ra["sharpe"]; out[nm]["btc_leg_sharpe"] = rb["sharpe"]
    if nm.endswith("MED"):
        r_core = d["r"]

# ---------------------------------------------------------------- plateau / neighbours (post-hoc, counted)
nb = {}
for L in (3, 5, 10, 14):
    o, _ = r2lib.evaluate(spread_w(st_from(disp(U50, L)), U50), f"B1_disp{L}_MED", U=U50, full=False)
    r2lib.log(o); show(o, K); nb[o["name"]] = o["sharpe"]
for win in (180, 730):
    o, _ = r2lib.evaluate(spread_w(st_from(disp(U50, 7), win=win), U50), f"B1_disp7_MED_pctwin{win}", U=U50, full=False)
    r2lib.log(o); show(o, K); nb[o["name"]] = o["sharpe"]
o, _ = r2lib.evaluate(spread_w(st_from(disp(U50, 7, min_hist=180)), U50), "B1_disp7_MED_ex_new_listings", U=U50, full=False)
r2lib.log(o); show(o, K); out["ex_new"] = o
nb["B1_disp7_TERC"] = out["B1_disp7_TERC"]["sharpe"]
out["neighbour_sharpes"] = nb
out["neighbour_median"] = float(np.median(list(nb.values())))
print("   neighbour median Sharpe:", round(out["neighbour_median"], 2))

# ---------------------------------------------------------------- controls (post-hoc, counted)
btcvol = R[BTC].rolling(30, min_periods=20).std()
o, rv = r2lib.evaluate(spread_w(st_from(btcvol), U50), "B1c_btcvol30_high", U=U50, full=False)
r2lib.log(o); show(o, K); out["ctl_btcvol"] = o
alt_idx = ew_index(U50, ex_btc=True)
ctl = {}
for L in (7, 30):
    spr = np.log(alt_idx / alt_idx.shift(L)) - np.log(C[BTC] / C[BTC].shift(L))
    s = np.sign(spr).replace(0, -1.0).where(spr.notna())
    o, rr = r2lib.evaluate(spread_w(s, U50), f"B1c_spread_tsmom{L}", U=U50, full=False)
    if L == 7:
        r2lib.log(o)
    show(o, K); ctl[f"tsmom{L}"] = rr["df"]["r"]
    print(f"   state agreement disp7 vs spread-tsmom{L}: {(s == st7).where(s.notna() & st7.notna()).mean():.2f}")
ctl["btcvol"] = rv["df"]["r"]
print("   corr(disp7 pct, BTC vol30 pct):", round(trail_pct(disp(U50, 7)).corr(trail_pct(btcvol)), 2))
cc = r2lib.controls()
X = pd.concat([r_core.rename("r"), cc, pd.DataFrame(ctl)], axis=1, join="inner").dropna()
b, t = r2lib.ols_nw(X["r"].values, X[["xsmom14", "btc", "tsmom7", "tsmom30", "btcvol"]].values)
print("   regression of B1_disp7_MED on [xsmom14, btc, tsmom7, tsmom30, btcvol]: alpha/yr "
      f"{b[0]*365:+.1%} t {t[0]:.2f}; betas " + " ".join(f"{v:+.2f}(t{tt:.1f})" for v, tt in zip(b[1:], t[1:])))
out["alpha_vs_all_controls"] = {"alpha_ann": float(b[0] * 365), "t": float(t[0]),
                                "betas": [float(v) for v in b[1:]], "t_betas": [float(v) for v in t[1:]]}
print("   corr with controls:", X.corr()["r"].round(2).to_dict())

# ---------------------------------------------------------------- universe robustness + 16 positions
for k in (30, 75):
    Uk = lab.hype_universe("1d", top_k=k)
    o, _ = r2lib.evaluate(spread_w(st_from(disp(Uk, 7)), Uk), f"B1_disp7_MED_top{k}", U=Uk, full=False)
    show(o, K); out[f"top{k}"] = o
U16 = lab.hype_universe("1d", top_k=16)
for nm, stt in (("B1_disp7_MED_16pos", st7), ("B1_disp7_TERC_16pos", st7t)):
    o, _ = r2lib.evaluate(spread_w(stt, U16), nm, U=U16, full=True, n_trials=N_TRIALS)
    show(o); out[nm] = o
    print("   yearly:", o["yearly"])
# full-universe (all Binance) version for reference
Ufull = lab.universe("1d", top_k=50)
o, _ = r2lib.evaluate(spread_w(st_from(disp(Ufull, 7)), Ufull), "B1_disp7_MED_binance_top50", U=Ufull, full=False)
show(o, K); out["binance_top50"] = o

# ---------------------------------------------------------------- overlay family best
trend_idx = ew_index(U50)
trend = (trend_idx > trend_idx.rolling(50, min_periods=50).mean()).astype(float).where(trend_idx.rolling(50).mean().notna())
Wtr = overlay_w(trend, U50)
Wov = overlay_w((st7 == 1).astype(float).where(st7.notna()), U50)
Wcomb = overlay_w((trend * (st7 == 1).astype(float)).where(st7.notna() & trend.notna()), U50)
rtr = lab.backtest(Wtr, "1d", cost=r2lib.COST)["df"]["r"]
for nm, WW in (("B3_disp7_MED", Wov), ("B4_trend_AND_disp7", Wcomb), ("B3t_trend_EWidx_sma50", Wtr)):
    o, res = r2lib.evaluate(WW, nm, U=U50, full=True, n_trials=N_TRIALS)
    Z = pd.concat([res["df"]["r"].rename("r"), rtr.rename("tr")], axis=1, join="inner").dropna()
    if nm != "B3t_trend_EWidx_sma50":
        bb, tt = r2lib.ols_nw(Z["r"].values, Z[["tr"]].values)
        o["alpha_vs_trend_ann"] = float(bb[0] * 365); o["alpha_vs_trend_t"] = float(tt[0]); o["corr_trend"] = float(Z.corr().iloc[0, 1])
    show(o); print("   yearly:", o["yearly"], "| vs trend:", {k: round(o.get(k, np.nan), 2) for k in ("alpha_vs_trend_ann", "alpha_vs_trend_t", "corr_trend")})
    out[nm] = o
    expo = {"B3_disp7_MED": (st7 == 1).astype(float).where(st7.notna()),
            "B4_trend_AND_disp7": (trend * (st7 == 1).astype(float)).where(st7.notna() & trend.notna()),
            "B3t_trend_EWidx_sma50": trend}[nm]
    o16, _ = r2lib.evaluate(overlay_w(expo, U16), nm + "_16pos", U=U16, full=False)
    show(o16, K); out[nm + "_16pos"] = o16
json.dump(out, open(r2lib.os.path.join(r2lib.HERE, "famB_post.json"), "w"), indent=1, default=str)
