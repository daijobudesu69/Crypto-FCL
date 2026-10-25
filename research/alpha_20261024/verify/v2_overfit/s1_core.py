"""v2_overfit / step 1: reproduce the candidate, multiple-testing haircuts, block bootstrap, stability.

Candidate (h1 A4_L14_top50_tercile_rb1): score = -(14d funding sum), lab.universe top-50 (60d hist,
30d median qv), lab.quantile_ls_weights(q=0.33), daily rebalance, 0.07%/side + actual funding.
No new strategy configs are selected here (diagnostics on one fixed series only).
"""
from vlib import *
import json, glob
from scipy.stats import norm

out = {}
U = lab.universe("1d", top_k=50)
W = cand_weights(14, 0.33, 50, U)
res = lab.backtest(W, "1d")
df, m = res["df"], res["metrics"]
r = df["r"]
out["metrics"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()}
lab.report(res, "candidate A4_L14_top50_tercile")
y = lab.yearly(r)
out["yearly"] = y.round(4).to_dict()
df.to_parquet(os.path.join(HERE, "cand_daily_IS.parquet"))

T = m["days"]; yrs = T / 365
SR = m["sharpe"]; skew = m["skew"]; kurt_raw = m["kurt"] + 3
srd = SR / np.sqrt(365)

# ---------------- 1. multiple testing ----------------
# empirical dispersion of annual Sharpe across the desk's recorded trials (IS result CSVs only)
H = os.path.join(os.path.dirname(os.path.dirname(HERE)), "hunt")
files = {"h1A": "h1_funding/famA_grid.csv", "h1B": "h1_funding/famB_carry.csv", "h2": "h2_positioning/runlog.csv",
         "h3": "h3_basis_flow/registry.csv", "h4": "h4_xs_anomalies/grid_log.csv"}
for f in ["a1_dow_results", "a2_hourly_results", "a3_weekend_results", "b1_leadlag_results", "b1r_shockrev_results",
          "b2_volmanaged_results"]:
    files["h5_" + f[:3]] = f"h5_timeseries/{f}.csv"
tr = []
for k, f in files.items():
    d = pd.read_csv(os.path.join(H, f))
    s = pd.to_numeric(d["sharpe"], errors="coerce").dropna()
    tr.append(pd.DataFrame({"src": k, "sharpe": s.values}))
tr = pd.concat(tr, ignore_index=True)
tr_nc = tr[tr["src"] != "h1B"]                       # carry family (SR ~9) is a different object
tr.to_csv(os.path.join(HERE, "trial_sharpes.csv"), index=False)
q75, q25 = np.percentile(tr_nc["sharpe"], [75, 25])
sd_rob = float((q75 - q25) / 1.349)
print("trial Sharpe quantiles excl carry", np.percentile(tr_nc["sharpe"], [1, 5, 25, 50, 75, 95, 99]).round(2))
fam = tr[tr["src"] == "h1A"]["sharpe"]
mt = {"n_recorded_trials": int(len(tr)), "n_recorded_excl_carry": int(len(tr_nc)),
      "sd_sharpe_all_recorded": float(tr["sharpe"].std()), "sd_sharpe_excl_carry": float(tr_nc["sharpe"].std()),
      "sd_sharpe_h1_famA": float(fam.std()), "mean_sharpe_excl_carry": float(tr_nc["sharpe"].mean()),
      "rank_of_candidate_excl_carry": int((tr_nc["sharpe"] > SR).sum() + 1),
      "n_trials_sharpe_ge_1.5_excl_carry": int((tr_nc["sharpe"] >= 1.5).sum()), "sd_sharpe_robust_IQR": sd_rob,
      "top10_sharpes_excl_carry": sorted(tr_nc["sharpe"].round(2).tolist())[-10:]}
print("recorded trials", mt)
emc = 0.5772156649


def sr0(n, sd_ann):
    return sd_ann * ((1 - emc) * norm.ppf(1 - 1 / n) + emc * norm.ppf(1 - 1 / (n * np.e)))


dsr = {}
for n in [433, 71, 52]:
    for tag, sd in [("sd0.5", 0.5), ("sd_emp_all", mt["sd_sharpe_excl_carry"]), ("sd_emp_famA", mt["sd_sharpe_h1_famA"]), ("sd_robust_all", sd_rob), ("sd1.0", 1.0)]:
        dsr[f"n{n}_{tag}"] = {"SR0_expected_max_ann": float(sr0(n, sd)),
                              "DSR": lab.deflated_sharpe(SR, n, T, skew, kurt_raw, sr_var_trials=(sd / np.sqrt(365)) ** 2)}
psr0 = float(norm.cdf(srd * np.sqrt(T - 1) / np.sqrt(1 - skew * srd + (kurt_raw - 1) / 4 * srd ** 2)))
mt["PSR_vs_0"] = psr0
# minimum track record length (days) to have 95% confidence SR>0 and SR>0.5
for tgt in [0.0, 0.5]:
    s0 = tgt / np.sqrt(365)
    mt[f"MinTRL_days_SR>{tgt}"] = float(1 + (1 - skew * srd + (kurt_raw - 1) / 4 * srd ** 2) * (norm.ppf(0.95) / (srd - s0)) ** 2)
mt["dsr"] = dsr
# t-stats
t_iid = SR * np.sqrt(yrs)
t_sk = srd * np.sqrt(T - 1) / np.sqrt(1 - skew * srd + (kurt_raw - 1) / 4 * srd ** 2)
mt.update(t_nw=m["t_nw"], t_iid=float(t_iid), t_skew_kurt_adj=float(t_sk))
hc = {}
for tname, t in [("t_nw", m["t_nw"]), ("t_iid", t_iid)]:
    p = 2 * (1 - norm.cdf(t))
    for n in [433, 71]:
        pb = min(1.0, p * n)
        th = norm.ppf(1 - pb / 2) if pb < 1 else 0.0
        hc[f"{tname}_bonf_n{n}"] = {"p_raw": float(p), "p_bonf": float(pb), "t_haircut": float(th),
                                    "SR_haircut": float(th / np.sqrt(yrs)), "haircut_pct": float(1 - th / t)}
# BHY using recorded trial Sharpes as t = SR*sqrt(yrs) (approx; unrecorded trials set to p=1)
tt = np.r_[tr_nc["sharpe"].values * np.sqrt(yrs)]
pv = np.sort(np.r_[2 * (1 - norm.cdf(np.abs(tt))), np.ones(max(0, 433 - len(tt)))])
N = len(pv); cN = np.sum(1 / np.arange(1, N + 1))
padj = np.minimum.accumulate((N * cN / np.arange(1, N + 1) * pv)[::-1])[::-1]
p_c = 2 * (1 - norm.cdf(m["t_nw"]))
k = np.searchsorted(pv, p_c, side="right")
p_bhy = float(min(1, padj[max(k - 1, 0)]))
th = norm.ppf(1 - p_bhy / 2) if p_bhy < 1 else 0.0
hc["t_nw_BHY_n433"] = {"p_bhy": p_bhy, "t_haircut": float(th), "SR_haircut": float(th / np.sqrt(yrs))}
mt["haircuts"] = hc
mt["passes_t>3"] = {"t_nw": bool(m["t_nw"] > 3), "t_iid": bool(t_iid > 3), "t_skew_kurt_adj": bool(t_sk > 3)}
# t by sub-period
mt["t_nw_by_period"] = {lab_: float(lab.newey_west_t(between(r, a, b).values, 10)) for lab_, a, b in
                        [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"),
                         ("2023-24", "2023-01-01", "2025-01-01"), ("ex2021", None, None)] if a} | \
                       {"ex2021": float(lab.newey_west_t(r[r.index.year != 2021].values, 10))}
out["multiple_testing"] = mt
print(json.dumps(mt, indent=1, default=float))

# ---------------- 2. stationary block bootstrap ----------------
bs = {}
for blk in [5, 20, 60]:
    b = stationary_bootstrap_sharpes(r, blk, 2000, seed=11)
    bs[f"block{blk}"] = {"ci90": [float(np.percentile(b, 5)), float(np.percentile(b, 95))], "median": float(np.median(b)),
                         "p_sr_lt_0": float((b < 0).mean()), "p_sr_lt_0.5": float((b < 0.5).mean()),
                         "p_sr_lt_1": float((b < 1.0).mean())}
for lab_, a, b_ in [("2023-24", "2023-01-01", "2025-01-01"), ("2022-24", "2022-01-01", "2025-01-01")]:
    rr = between(r, a, b_)
    b = stationary_bootstrap_sharpes(rr, 20, 2000, seed=12)
    bs[f"block20_{lab_}"] = {"point": sh(rr), "ci90": [float(np.percentile(b, 5)), float(np.percentile(b, 95))],
                             "p_sr_lt_0": float((b < 0).mean()), "p_sr_lt_0.5": float((b < 0.5).mean())}
rx = r[r.index.year != 2021]
b = stationary_bootstrap_sharpes(rx, 20, 2000, seed=13)
bs["block20_ex2021"] = {"point": sh(rx), "ci90": [float(np.percentile(b, 5)), float(np.percentile(b, 95))],
                        "p_sr_lt_0": float((b < 0).mean()), "p_sr_lt_0.5": float((b < 0.5).mean())}
out["bootstrap"] = bs
print(json.dumps(bs, indent=1))

# ---------------- 3. stability ----------------
st = {}
roll = r.rolling(365).apply(lambda x: x.mean() / x.std() * np.sqrt(365), raw=True).dropna()
st["rolling365"] = {"min": float(roll.min()), "min_date": str(roll.idxmin().date()), "max": float(roll.max()),
                    "median": float(roll.median()), "pct_lt_0": float((roll < 0).mean()),
                    "pct_lt_0.5": float((roll < 0.5).mean()), "pct_gt_1": float((roll > 1).mean()),
                    "last": float(roll.iloc[-1])}
roll.to_frame("sharpe365").to_csv(os.path.join(HERE, "rolling365.csv"))
# rolling 365d by quarter-end for the report
st["rolling365_quarterly"] = {str(k.date()): round(float(v), 2) for k, v in roll.resample("QE").last().items()}
years = sorted(set(r.index.year))
st["year_alone"] = {int(yv): sh(r[r.index.year == yv]) for yv in years}
st["leave_one_year_out"] = {int(yv): sh(r[r.index.year != yv]) for yv in years}
st["ex2021"] = sh(r[r.index.year != 2021])
st["ex2020_2021"] = sh(r[r.index.year >= 2022])
st["only_2023_2024"] = sh(between(r, "2023-01-01", "2025-01-01"))
st["only_2024"] = sh(between(r, "2024-01-01", "2025-01-01"))
st["half_years"] = {f"{yv}H{h}": sh(r[(r.index.year == yv) & ((r.index.month <= 6) == (h == 1))]) for yv in years for h in (1, 2)}
# share of log return from 2021
lr = np.log1p(r)
st["share_logret_2021"] = float(lr[lr.index.year == 2021].sum() / lr.sum())
# decomposition by year: price, funding, cost; price-net Sharpe
g = df.groupby(df.index.year)
st["decomp_by_year"] = pd.DataFrame({"price": g["gross_r"].sum(), "fund": g["fund_r"].sum(), "cost": -g["cost_r"].sum(),
                                     "sh_price_net": g.apply(lambda x: sh(x["gross_r"] - x["cost_r"])),
                                     "sh_total": g["r"].apply(sh)}).round(3).to_dict(orient="index")
# leg contributions (additive): split per-coin pnl by the sign of the position
price, fund, costc, tot = coin_pnl(W)
Wl = W.reindex(index=lab.load("close", "1d").index).fillna(0.0)
sg = np.sign(Wl).where(Wl != 0, np.sign(Wl.shift(1).fillna(0.0)))   # exits are charged to the leg being closed
Ws = sg.copy(); Ws.index = Ws.index + pd.Timedelta("1D"); Ws = Ws.reindex(tot.index)
legL = tot.where(Ws > 0, 0).sum(axis=1).reindex(r.index); legS = tot.where(Ws < 0, 0).sum(axis=1).reindex(r.index)
assert np.allclose((legL + legS).values, r.values, atol=1e-9), "decomposition mismatch"
st["legs_by_year_sum"] = pd.DataFrame({"long": legL.groupby(legL.index.year).sum(), "short": legS.groupby(legS.index.year).sum()}).round(3).to_dict(orient="index")
# leg alpha vs equal-weight universe (hedged leg returns, 0.5 gross each side)
R = lab.returns("1d"); Rn = R.shift(-1)
ew = Rn.where(U & Rn.notna()).mean(axis=1); ew.index = ew.index + pd.Timedelta("1D"); ew = ew.reindex(r.index).fillna(0)
hl = legL - 0.5 * ew; hs = legS + 0.5 * ew
st["leg_hedged_sharpe_by_year"] = {int(yv): {"long_vsEW": sh(hl[hl.index.year == yv]), "short_vsEW": sh(hs[hs.index.year == yv])} for yv in years}
st["leg_hedged_sharpe_all"] = {"long_vsEW": sh(hl), "short_vsEW": sh(hs), "long_vsEW_2023-24": sh(between(hl, "2023-01-01", "2025-01-01")),
                               "short_vsEW_2023-24": sh(between(hs, "2023-01-01", "2025-01-01"))}
pd.DataFrame({"r": r, "legL": legL, "legS": legS, "ew": ew}).to_parquet(os.path.join(HERE, "cand_legs_IS.parquet"))
out["stability"] = st
print(json.dumps(st, indent=1, default=float))
# stress
out["stress"] = {"cost2x": lab.backtest(W, "1d", cost=2 * lab.COST)["metrics"]["sharpe"],
                 "lag1": lab.backtest(W, "1d", extra_lag=1)["metrics"]["sharpe"],
                 "lag2": lab.backtest(W, "1d", extra_lag=2)["metrics"]["sharpe"],
                 "hype_cost": lab.backtest(W, "1d", cost=lab.HYPE_COST)["metrics"]["sharpe"],
                 "no_funding_pnl": lab.backtest(W, "1d", funding=False)["metrics"]["sharpe"]}
print(out["stress"])
json.dump(out, open(os.path.join(HERE, "s1_core.json"), "w"), indent=1, default=float)
print("saved")
