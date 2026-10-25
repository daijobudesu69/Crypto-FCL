"""v4_risk part 1: short-leg tail risk + pre-registered overlays. IS 2020-01..2024-12 only.

PRE-REGISTRATION (written before running; at most 4 overlays, judged vs base on the SAME universe):
  O1 stop25 : intraday stop on each short at +25% above the episode entry close (fill at level +0.5% slip),
              name banned from the short leg until the signal drops it. Freed weight stays in cash.
  O2 stop40 : same with +40%.
  O3 skipmom: drop from the short leg any name whose trailing 14d return is in the top decile of the
              top-50 universe that day (do not short parabolic names); remaining shorts re-normalised to -0.5.
  O4 invvol : inside each leg weight names by 1/(30d trailing stdev of daily returns), each leg still 0.5 gross
              (caps risk contributed by a single high-vol name).
  Decision rule (pre-registered): an overlay is only recommended if it raises full-period Sharpe by >= 0.10
  AND does not lower Sharpe in any of 2020-21 / 2022 / 2023-24 AND reduces the worst day or maxDD materially.
  Otherwise it is "not worth adding". Run on full top-50 universe (primary) and HYPE-restricted top-50.
"""
from common4 import *
import json

C = lab.load("close", "1d"); H = lab.load("high", "1d"); F = lab.load("fund", "1d")
R = C.pct_change(fill_method=None)
U = lab.universe("1d", top_k=50)
UH = lab.hype_universe("1d", top_k=50)
W, s = candidate_weights(U, F)
out = {}

# ---- 0. reproduce
res = lab.backtest(W, "1d"); m = res["metrics"]; df = res["df"]
out["base_lab"] = {k: m[k] for k in ["sharpe", "cagr", "maxdd", "vol", "skew", "kurt", "avg_n_pos", "turnover_per_year", "funding_per_year", "cost_drag_per_year"]}
out["base_lab"]["start"] = m["start"]
d0 = sim(W)
out["sim_check_sharpe"] = sh(d0["r"])
print("lab Sharpe %.3f  sim Sharpe %.3f  CAGR %.3f MDD %.3f npos %.1f start %s" % (m["sharpe"], sh(d0["r"]), m["cagr"], m["maxdd"], m["avg_n_pos"], m["start"]))
print(lab.yearly(df["r"]).round(3).T.to_string())
r = df["r"]

# names per side
ns = (W < 0).sum(axis=1); nl = (W > 0).sum(axis=1)
act = ns > 0
out["names_per_side"] = {"short_median": float(ns[act].median()), "short_min": int(ns[act].min()), "long_median": float(nl[act].median())}
WH, _ = candidate_weights(UH, F)
nsh = (WH < 0).sum(axis=1)
out["names_per_side_hype"] = {"short_median": float(nsh[nsh > 0].median()), "short_min": int(nsh[nsh > 0].min()),
                              "by_year_median": {int(k): float(v) for k, v in nsh[nsh > 0].groupby(nsh[nsh > 0].index.year).median().items()}}
print("names/side", out["names_per_side"], "HYPE", out["names_per_side_hype"])

# ---- 1. single coin-day contributions (position decided at close t, earns bar t+1)
Rn = R.shift(-1); Hn = H.shift(-1)
contrib = (W * Rn)                      # fraction of equity
cs = contrib.stack(); cs = cs[cs != 0]
worst = cs.sort_values().head(20)
rows = []
for (t, sym), v in worst.items():
    rows.append({"date_held": str((t + pd.Timedelta("1D")).date()), "sym": sym, "side": "short" if W.at[t, sym] < 0 else "long",
                 "w": round(float(W.at[t, sym]), 4), "coin_ret": round(float(Rn.at[t, sym]), 4),
                 "intraday_hi_vs_entry": round(float(Hn.at[t, sym] / C.at[t, sym] - 1), 4), "contrib": round(float(v), 4),
                 "contrib_if_w_0.0625": round(float(np.sign(W.at[t, sym]) * 0.0625 * Rn.at[t, sym]), 4),
                 "contrib_if_w_0.10": round(float(np.sign(W.at[t, sym]) * 0.10 * Rn.at[t, sym]), 4)})
out["worst20_coin_days"] = rows
print(pd.DataFrame(rows).to_string())

# short leg raw moves
shmask = (W < 0) & Rn.notna()
sr = Rn.where(shmask).stack()
shi = (Hn / C - 1).where(shmask).stack()
out["short_leg_moves"] = {"n_coin_days": int(len(sr)),
                          "close_gt_20": int((sr > 0.2).sum()), "close_gt_30": int((sr > 0.3).sum()), "close_gt_50": int((sr > 0.5).sum()),
                          "high_gt_20": int((shi > 0.2).sum()), "high_gt_25": int((shi > 0.25).sum()), "high_gt_30": int((shi > 0.3).sum()), "high_gt_50": int((shi > 0.5).sum()),
                          "max_close_move": float(sr.max()), "max_intraday_high": float(shi.max()),
                          "p99_close": float(sr.quantile(0.99)), "p999_close": float(sr.quantile(0.999))}
yrs = sr.index.get_level_values(0).year
out["short_leg_moves"]["close_gt_30_by_year"] = {int(k): int(v) for k, v in (sr > 0.3).groupby(yrs).sum().items()}
out["short_leg_moves"]["high_gt_25_by_year"] = {int(k): int(v) for k, v in (shi > 0.25).groupby(shi.index.get_level_values(0).year).sum().items()}
out["short_leg_moves"]["days_per_year_any_short_high_gt_25"] = float(((Hn / C - 1).where(shmask) > 0.25).any(axis=1).sum() / (len(r) / 365))
print(out["short_leg_moves"])

# episode cumulative adverse move (consecutive days in short leg)
ep_rows = []
inS = (W < 0)
for sym in W.columns:
    x = inS[sym].values
    if not x.any():
        continue
    c = C[sym].values
    hh = H[sym].values
    i = 0; n = len(x)
    while i < n:
        if x[i]:
            j = i
            while j + 1 < n and x[j + 1]:
                j += 1
            # held from close i to close j+1
            e = c[i]; ex = c[min(j + 1, n - 1)]
            mx = np.nanmax(hh[i + 1:min(j + 2, n)]) if j + 1 < n else np.nan
            ep_rows.append((sym, C.index[i], j - i + 1, ex / e - 1, mx / e - 1))
            i = j + 1
        else:
            i += 1
ep = pd.DataFrame(ep_rows, columns=["sym", "start", "days", "ret", "max_hi"])
out["episodes"] = {"n": int(len(ep)), "median_days": float(ep["days"].median()),
                   "max_hi_gt_25": int((ep["max_hi"] > 0.25).sum()), "max_hi_gt_40": int((ep["max_hi"] > 0.40).sum()),
                   "ret_gt_25": int((ep["ret"] > 0.25).sum()), "ret_gt_40": int((ep["ret"] > 0.40).sum()),
                   "ret_gt_100": int((ep["ret"] > 1.0).sum()),
                   "worst10": [(a, str(b.date()), int(c_), round(float(d_), 3), round(float(e_), 3)) for a, b, c_, d_, e_ in ep.sort_values("ret", ascending=False).head(10).values]}
# of episodes that touched +25 / +40 at some point, how did they finish?
for X in (0.25, 0.40):
    sub = ep[ep["max_hi"] > X]
    out["episodes"][f"touched_{int(X*100)}_final_ret_median"] = float(sub["ret"].median()) if len(sub) else np.nan
    out["episodes"][f"touched_{int(X*100)}_final_gt_X"] = float((sub["ret"] > X).mean()) if len(sub) else np.nan
print(out["episodes"])

# ---- 2. worst portfolio days / weeks / months
out["worst_days"] = [(str(k.date()), round(float(v), 4)) for k, v in r.nsmallest(10).items()]
out["best_days"] = [(str(k.date()), round(float(v), 4)) for k, v in r.nlargest(5).items()]
lr = np.log1p(r)
w7 = np.expm1(lr.rolling(7).sum())
w30 = np.expm1(lr.rolling(30).sum())
out["worst_7d"] = float(w7.min()); out["worst_7d_end"] = str(w7.idxmin().date())
out["worst_30d"] = float(w30.min()); out["worst_30d_end"] = str(w30.idxmin().date())
eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
trough = dd.idxmin(); peak = eq[:trough].idxmax()
rec = eq[trough:][eq[trough:] >= eq[peak]]
out["maxdd_detail"] = {"maxdd": float(dd.min()), "peak": str(peak.date()), "trough": str(trough.date()),
                       "recovered": str(rec.index[0].date()) if len(rec) else "not by 2024-12-31"}
# longest underwater
uw = (dd < 0).astype(int)
grp = (uw.diff() != 0).cumsum()
lens = uw.groupby(grp).sum()
out["longest_underwater_days"] = int(lens.max())
# daily VaR / ES
out["var_es"] = {"p1": float(r.quantile(0.01)), "es1": float(r[r <= r.quantile(0.01)].mean()), "p5": float(r.quantile(0.05))}
# leg attribution on worst days
Wl, Ws = W.clip(lower=0), W.clip(upper=0)
legL = (Wl * Rn).sum(axis=1); legS = (Ws * Rn).sum(axis=1)
legL.index = legL.index + pd.Timedelta("1D"); legS.index = legS.index + pd.Timedelta("1D")
out["worst_days_leg_split"] = [(str(k.date()), round(float(v), 4), round(float(legL.get(k, np.nan)), 4), round(float(legS.get(k, np.nan)), 4)) for k, v in r.nsmallest(10).items()]
print({k: out[k] for k in ["worst_days", "worst_7d", "worst_7d_end", "worst_30d", "worst_30d_end", "maxdd_detail", "longest_underwater_days", "var_es"]})
print(out["worst_days_leg_split"])

# ---- 3. hypothetical concentration: 5-8 names per side at gross 1.0
# single position loss = (0.5/n) * worst move; use observed short-leg close moves and intraday highs
for n_ in (5, 6, 8, 12, 16):
    wpos = 0.5 / n_
    out.setdefault("single_pos_loss", {})[n_] = {"w": wpos, "worst_close_day": round(wpos * float(sr.max()), 4),
                                                 "worst_intraday": round(wpos * float(shi.max()), 4),
                                                 "p999_close": round(wpos * float(sr.quantile(0.999)), 4),
                                                 "worst_episode_cum": round(wpos * float(ep["ret"].max()), 4)}
print(out["single_pos_loss"])

# ---- 4. overlays
def leg_norm(Wx, side_gross=0.5):
    L = Wx.clip(lower=0); S = Wx.clip(upper=0)
    L = L.div(L.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * side_gross
    S = S.div(S.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * side_gross
    return L + S

def overlays(Wb, Ub):
    mom14 = C / C.shift(14) - 1
    rk = mom14.where(Ub).rank(axis=1, pct=True)
    W3 = Wb.copy(); W3[(Wb < 0) & (rk > 0.9)] = 0.0
    W3 = leg_norm(W3)
    vol = R.rolling(30, min_periods=20).std()
    iv = (1 / vol).where(Wb != 0)
    W4 = leg_norm(np.sign(Wb) * iv.fillna(0))
    return {"base": (Wb, None), "O1_stop25": (Wb, 0.25), "O2_stop40": (Wb, 0.40), "O3_skipmom": (W3, None), "O4_invvol": (W4, None)}

def sub(rr):
    o = {}
    for nm, a, b in [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"), ("2023-24", "2023-01-01", "2025-01-01")]:
        o[nm] = sh(rr[(rr.index >= pd.Timestamp(a, tz="UTC")) & (rr.index < pd.Timestamp(b, tz="UTC"))])
    return o

ov = {}
for uname, Ub in [("top50", U), ("hype_top50", UH)]:
    Wb, _ = candidate_weights(Ub, F)
    for name, (Wx, stp) in overlays(Wb, Ub).items():
        d = sim(Wx, stop=stp)
        d2 = sim(Wx, stop=stp, cost=2 * lab.COST)
        rr = d["r"]; e = (1 + rr).cumprod()
        ov[f"{uname}:{name}"] = {"sharpe": sh(rr), "sharpe_2x": sh(d2["r"]), "cagr": float(e.iloc[-1] ** (365 / len(rr)) - 1),
                                 "maxdd": float((e / e.cummax() - 1).min()), "worst_day": float(rr.min()),
                                 "worst_7d": float(np.expm1(np.log1p(rr).rolling(7).sum()).min()),
                                 "skew": float(rr.skew()), "nstops": d.attrs["nstops"], "sub": sub(rr),
                                 "npos": float(d["n_pos"].mean())}
        print(uname, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ov[f"{uname}:{name}"].items()})
out["overlays"] = ov
json.dump(out, open(os.path.join(HERE, "t1_tail.json"), "w"), indent=1, default=str)
print("saved")
