"""v2_overfit / step 4: survivorship sensitivity + calibration checks. No config selection.

Pre-registered runs (candidate rules unchanged, only the coin pool changes):
  (a) pool = coins whose first Binance daily close is before 2022-01-01
      a1: top-50 re-ranked inside the pool;  a2: lab top-50 of all coins, intersected with the pool
  (b) pool = lab.hype_symbols()
      b1: lab.hype_universe(top_k=50) (top-50 of all, intersected; = what the lab offers)
      b2: top-50 re-ranked inside the HYPE pool
      each at lab.COST and lab.HYPE_COST
  (c) contribution of coins listed 2022+ vs older coins in 2023-24 (decomposition, not a new config)
  (d) collapse-stress: analytic number of -90%/5-day collapses per year landing in the long leg that
      would push the Sharpe to 0.5 (full sample and 2023-24)
  (e) calibration: share of PnL in the 10 best days for iid normal returns with the same mean/std
  (f) Binance-vs-HYPE F14 rank agreement where HYPE funding exists (2023-05..2024-12), descriptive only
"""
from vlib import *
import json

out = {}
C = lab.load("close", "1d")
fd = C.apply(lambda x: x.first_valid_index())
old = [c for c, d in fd.items() if d is not None and pd.notna(d) and d < pd.Timestamp("2022-01-01", tz="UTC")]
hs = [c for c in lab.hype_symbols() if c in C.columns]
U50 = lab.universe("1d", top_k=50)


def run(U, cost=lab.COST, tag=""):
    W = cand_weights(14, 0.33, 50, U)
    res = lab.backtest(W, "1d", cost=cost)
    r, m = res["df"]["r"], res["metrics"]
    d = {"sharpe": m["sharpe"], "cagr": m["cagr"], "maxdd": m["maxdd"], "npos": m["avg_n_pos"],
         "sh_2020-21": sh(between(r, "2020-01-01", "2022-01-01")), "sh_2022": sh(between(r, "2022-01-01", "2023-01-01")),
         "sh_2023-24": sh(between(r, "2023-01-01", "2025-01-01")), "sh_ex2021": sh(r[r.index.year != 2021]),
         "avg_univ_2023-24": float(U.sum(axis=1)[U.index >= pd.Timestamp("2023-01-01", tz="UTC")].mean())}
    # leg hedged vs EW of the same universe
    price, fund, costc, tot = coin_pnl(W, cost)
    tot = tot.reindex(r.index)
    Wl = W.reindex(index=C.index).fillna(0.0)
    sg = np.sign(Wl).where(Wl != 0, np.sign(Wl.shift(1).fillna(0.0)))
    sg.index = sg.index + pd.Timedelta("1D"); sg = sg.reindex(r.index)
    Rn = C.pct_change(fill_method=None).shift(-1)
    ew = Rn.where(U & Rn.notna()).mean(axis=1); ew.index = ew.index + pd.Timedelta("1D"); ew = ew.reindex(r.index).fillna(0)
    hl = tot.where(sg > 0, 0).sum(axis=1) - 0.5 * ew
    hsr = tot.where(sg < 0, 0).sum(axis=1) + 0.5 * ew
    d.update(long_vsEW=sh(hl), short_vsEW=sh(hsr), long_vsEW_2023_24=sh(between(hl, "2023-01-01", "2025-01-01")),
             short_vsEW_2023_24=sh(between(hsr, "2023-01-01", "2025-01-01")))
    print(f"{tag:28s}", {k: round(v, 3) for k, v in d.items()}, flush=True)
    return d, r, tot


surv = {}
surv["base"], rb, totb = run(U50, tag="base")
surv["a1_pre2022_reranked"], _, _ = run(universe_pool(50, old), tag="a1 pre-2022 re-ranked")
surv["a2_pre2022_intersect"], _, _ = run(U50 & pd.Series(U50.columns.isin(old), index=U50.columns), tag="a2 pre-2022 intersect")
Uh = lab.hype_universe("1d", top_k=50)
Uh2 = universe_pool(50, hs)
surv["b1_hype_intersect"], _, _ = run(Uh, tag="b1 hype intersect")
surv["b1_hype_intersect_hypecost"], _, _ = run(Uh, lab.HYPE_COST, tag="b1 hype intersect HYPEcost")
surv["b2_hype_reranked"], _, _ = run(Uh2, tag="b2 hype re-ranked")
surv["b2_hype_reranked_hypecost"], _, _ = run(Uh2, lab.HYPE_COST, tag="b2 hype re-ranked HYPEcost")
out["survivorship_runs"] = surv
out["pools"] = {"n_old_pre2022": len(old), "n_hype": len(hs), "n_with_data_pre2025": int(fd.notna().sum()),
                "non_hype_coins": [c.replace("USDT", "") for c in C.columns if c not in hs and pd.notna(fd[c])]}

# (c) 2023-24 contribution: coins listed 2022+ vs older
m = totb.index >= pd.Timestamp("2023-01-01", tz="UTC")
t34 = totb[m].sum()
out["contrib_2023-24"] = {"old_pre2022": float(t34[t34.index.isin(old)].sum()), "new_2022plus": float(t34[~t34.index.isin(old)].sum())}
print(out["contrib_2023-24"])

# (d) collapse stress (analytic): one coin in the long leg at weight w = 0.5/n_long, daily rebalanced to w,
#     falls 90% over 5 days -> loss = w * sum(daily r) with r = 0.1**(1/5)-1 each day
Wc = cand_weights(14, 0.33, 50, U50)
nl = (Wc > 0).sum(axis=1); nl = nl[nl > 0]
w_avg = float((0.5 / nl).mean()); w_2324 = float((0.5 / nl[nl.index >= pd.Timestamp("2023-01-01", tz="UTC")]).mean())
dr = 0.1 ** (1 / 5) - 1
st = {}
for lab_, rr, w in [("full", rb, w_avg), ("2023-24", between(rb, "2023-01-01", "2025-01-01"), w_2324)]:
    loss = -w * 5 * dr
    mu, sd = rr.mean() * 365, rr.std() * np.sqrt(365)
    st[lab_] = {"w_long_name": w, "loss_per_event": float(loss), "mu_ann": float(mu), "vol_ann": float(sd),
                "events_per_year_to_SR0.5": float((mu - 0.5 * sd) / loss), "events_per_year_to_SR0": float(mu / loss)}
out["collapse_stress"] = st
print(json.dumps(st, indent=1))

# (e) calibration of the best-days statistic
rng = np.random.default_rng(3)
cal = {}
for lab_, rr in [("full", rb), ("2023-24", between(rb, "2023-01-01", "2025-01-01"))]:
    sims = []
    for _ in range(2000):
        x = rng.normal(rr.mean(), rr.std(), len(rr))
        sims.append(np.sort(x)[-10:].sum() / x.sum())
    # t-distributed (fat tails, df=3) with same mean/std
    sims_t = []
    for _ in range(2000):
        x = rng.standard_t(3, len(rr)) / np.sqrt(3) * rr.std() + rr.mean()
        sims_t.append(np.sort(x)[-10:].sum() / x.sum())
    rs = rr.sort_values()
    cal[lab_] = {"actual": float(rs.tail(10).sum() / rr.sum()), "normal_median": float(np.median(sims)),
                 "t3_median": float(np.median(sims_t))}
out["best_days_calibration"] = cal
print(cal)

# (f) Binance vs HYPE F14 rank agreement (descriptive)
try:
    Fh = lab.load("fund", "hype_1d")
    Fb = lab.load("fund", "1d")
    sb = Fb.rolling(14, min_periods=14).sum()
    shh = Fh.rolling(14, min_periods=14).sum()
    m = Uh & sb.notna() & shh.notna()
    m = m[m.index >= pd.Timestamp("2023-06-01", tz="UTC")]
    rc = []
    for t in m.index:
        cols = m.columns[m.loc[t].values]
        if len(cols) >= 10:
            rc.append(sb.loc[t, cols].rank().corr(shh.loc[t, cols].rank()))
    rc = pd.Series(rc)
    out["binance_vs_hype_F14_rankcorr"] = {"days": int(len(rc)), "median": float(rc.median()), "p10": float(rc.quantile(0.1)),
                                           "p90": float(rc.quantile(0.9))}
    print(out["binance_vs_hype_F14_rankcorr"])
except Exception as e:
    out["binance_vs_hype_F14_rankcorr"] = {"error": str(e)}
json.dump(out, open(os.path.join(HERE, "s4_surv.json"), "w"), indent=1, default=float)
