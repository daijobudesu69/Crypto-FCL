"""Step 3: impact on Sharpe of every issue found + reproduction of the hunter's exact numbers.
All runs are IS 2020-2024 via lab.py; no new signal family, only implementation/bug variants of the
frozen candidate (counted in REPORT.md as verification runs, not as alpha configs)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab
OUT = os.path.dirname(os.path.abspath(__file__))
C = lab.load("close", "1d"); F = lab.load("fund", "1d")
U = lab.universe("1d", min_hist_days=60, top_k=50)
S = -F.rolling(14, min_periods=14).sum()
res = {}


def sub(r):
    o = {}
    for k, a, b in [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"), ("2023-24", "2023-01-01", "2025-01-01")]:
        x = r[(r.index >= a) & (r.index < b)]; o[k] = float(x.mean() / x.std() * np.sqrt(365))
    return o


def run(name, W, start=None, **kw):
    b = lab.backtest(W, "1d", **kw); r = b["df"]["r"]
    if start: r = r[r.index >= start]
    m = lab.metrics(r, "1d")
    y = lab.yearly(r)
    d = {"sharpe": m["sharpe"], "cagr": m["cagr"], "maxdd": m["maxdd"], "skew": m["skew"], "kurt": m["kurt"], "days": m["days"],
         "start": m["start"], "yearly": {int(k): round(float(v), 3) for k, v in y["ret"].items()}, "sub": sub(r),
         "npos": float(b["df"]["n_pos"].mean())}
    res[name] = d
    yr = " ".join(f"{k}:{v:+.1%}" for k, v in d["yearly"].items())
    print(f"{name:58s} SR {d['sharpe']:.2f} CAGR {d['cagr']:+.1%} DD {d['maxdd']:.1%} sub {d['sub']['2020-21']:.2f}/{d['sub']['2022']:.2f}/{d['sub']['2023-24']:.2f} | {yr}")
    return b


print("== reproduction of hunter numbers and tercile-cut sensitivity")
WH = lab.quantile_ls_weights(S, U, q=0.33)
run("HUNTER spec q=0.33 (A4_L14_top50_tercile_rb1)", WH)
run("  2x cost", WH, cost=0.0014)
run("  lag1", WH, extra_lag=1)
for q in [0.30, 0.32, 1 / 3, 0.34, 0.35]:
    run(f"q={q:.4f}", lab.quantile_ls_weights(S, U, q=q))
run("rank twin A1_L14_top50_rank_rb1", lab.xs_demean_weights(lab.xs_rank(S, U), U))

print("== start-date / tiny-universe effect (universe < 12 names until 2020-06)")
n = U.sum(axis=1)
print("first day universe >= 12 / >= 30 / >= 45 names:", n[n >= 12].index[0].date(), n[n >= 30].index[0].date(), n[n >= 45].index[0].date())
run("HUNTER spec, from 2020-07-01", WH, start="2020-07-01")
run("HUNTER spec, from 2021-01-01", WH, start="2021-01-01")
run("HUNTER spec, ex-2021 not possible as a backtest -> 2022-01-01..", WH, start="2022-01-01")
# share of 2020 return made in Mar-Jun 2020 with < 12 names
b = lab.backtest(WH, "1d")["df"]["r"]
print("HUNTER spec cum return 2020-03..2020-06 (<=11 names):", float((1 + b["2020-03-01":"2020-06-30"]).prod() - 1),
      " 2020-07..12:", float((1 + b["2020-07-01":"2020-12-31"]).prod() - 1))

print("== bug 1: ICPUSDT funding missing in raw data 2021-05-11..2022-08-31 (panel shows 0 -> F14=0)")
icp_mask = pd.DataFrame(False, index=U.index, columns=U.columns)
icp_mask.loc[(U.index >= "2021-05-11") & (U.index < "2022-09-15"), "ICPUSDT"] = True  # incl. 14d warm-up after data resumes
print("ICP coin-days in top-50 universe during hole:", int((U & icp_mask).values.sum()),
      " in LONG leg:", int(((WH > 0) & icp_mask).values.sum()), " in SHORT leg:", int(((WH < 0) & icp_mask).values.sum()))
U1 = U & ~icp_mask
run("fix1: ICP excluded while its funding is missing", lab.quantile_ls_weights(S, U1, q=0.33))

print("== bug 2: stale prices (close unchanged >= 3 consecutive days while 'listed')")
same = (C.diff() == 0).astype(float)
stale_run = same.rolling(3).sum() >= 3
st = stale_run.sum(); print("coins with stale runs:", st[st > 0].to_dict())
U2 = U1 & ~stale_run
run("fix1+2: also exclude stale-price coin-days", lab.quantile_ls_weights(S, U2, q=0.33))

print("== issue 3: BNB funding has 0% interest component on Binance (median rate 0 vs 0.01%/8h for others)")
Ud = U.where(U)  # bool
fz = ((F == 0) & C.notna()).where(U).mean()
f3 = (pd.DataFrame(np.isclose(F.values, 0.0003), index=F.index, columns=F.columns) & C.notna()).where(U).mean()
tab = pd.DataFrame({"share_days_fund0": fz, "share_days_fund_0.03%": f3}).dropna()
print(tab[tab.share_days_fund0 > 0.05].round(2).to_string())
inlong = ((WH > 0).where(U).mean()).dropna()
print("share of universe days in LONG leg: BNB %.2f  (median coin %.2f)" % (inlong["BNBUSDT"], inlong.median()))
U3 = U & ~pd.DataFrame(np.broadcast_to(U.columns == "BNBUSDT", U.shape), index=U.index, columns=U.columns)
run("excl. BNB (structural 0%-interest funding)", lab.quantile_ls_weights(S, U3, q=0.33))
# fix: add back the missing 0.01%/8h interest to BNB so its baseline matches the other coins (venue-neutral signal)
Fadj = F.copy(); Fadj["BNBUSDT"] = Fadj["BNBUSDT"] + 0.0003 * C["BNBUSDT"].notna()
Sadj = -Fadj.rolling(14, min_periods=14).sum()
run("BNB signal baseline-adjusted (+0.03%/day), actual funding paid", lab.quantile_ls_weights(Sadj, U, q=0.33))
run("all fixes: ICP hole + stale + BNB adj", lab.quantile_ls_weights(Sadj, U2, q=0.33))
run("all fixes, from 2020-07-01", lab.quantile_ls_weights(Sadj, U2, q=0.33), start="2020-07-01")

print("== leg attribution with a properly costed hedge (EW top-50, incl. its funding and costs)")
ew = U.astype(float).div(U.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
Ws = WH.clip(upper=0); Wl = WH.clip(lower=0)
run("short leg (-0.5) + EW universe long (+0.5)", Ws + 0.5 * ew)
run("long leg (+0.5) + EW universe short (-0.5)", Wl - 0.5 * ew)
run("short leg alone (-0.5)", Ws); run("long leg alone (+0.5)", Wl)

print("== HYPE-tradable subset (Binance data, HYPE cost) for context")
UH = lab.hype_universe("1d", min_hist_days=60, top_k=50)
run("HYPE subset of top-50, q=0.33, HYPE cost", lab.quantile_ls_weights(S, UH, q=0.33), cost=lab.HYPE_COST)

print("== deflated Sharpe")
for nm in ["HUNTER spec q=0.33 (A4_L14_top50_tercile_rb1)", "q=0.3333", "all fixes: ICP hole + stale + BNB adj"]:
    d = res[nm]
    for ntr in [52, 71, 433]:
        p = lab.deflated_sharpe(d["sharpe"], ntr, d["days"], d["skew"], d["kurt"] + 3)
        res[nm][f"dsr_{ntr}"] = p
    print(f"{nm:58s} DSR n=52 {res[nm]['dsr_52']:.3f}  n=71 {res[nm]['dsr_71']:.3f}  n=433 {res[nm]['dsr_433']:.3f}")
json.dump(res, open(os.path.join(OUT, "impact.json"), "w"), indent=1, default=float)
