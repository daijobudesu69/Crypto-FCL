"""v4_risk part 3: execution realism on HYPE (IS only: HYPE data 2023-05..2024-12 exists; nothing after 2024-12-31
is loaded because lab.load truncates at IS_END).
 a) HYPE liquidity of the names the candidate would trade (2024 HYPE daily quote volume)
 b) HYPE funding paid by the book: worst coin-days and portfolio days (shorts pay when funding < 0)
 c) Binance-F14 vs HYPE-F14 rank agreement (signal must be rebuilt from HYPE funding live)
 d) liquidation proxy for isolated margin: how often a short episode moved +X% from entry (1h closes)
 e) failure-mode estimate (NOT a selectable overlay): isolated 4x ~= forced exit at about +20% from entry with
    the remaining margin lost -> approximated by an exit at +20% with 4% extra slippage.
 f) $200 account replay with $10 minimum order (account_sim), Binance prices 2020-03.., HYPE prices 2023-07..
"""
from common4 import *
import json
import account_sim

C = lab.load("close", "1d"); F = lab.load("fund", "1d")
CH = lab.load("close", "hype_1d"); FH = lab.load("fund", "hype_1d"); QH = lab.load("qv", "hype_1d")
assert CH.index.equals(C.index) and CH.columns.equals(C.columns)
U = lab.universe("1d", top_k=50); UH = lab.hype_universe("1d", top_k=50)
WH, sH = candidate_weights(UH, F)
out = {}

# a) liquidity in 2024
y24 = (C.index >= pd.Timestamp("2024-01-01", tz="UTC"))
traded = (WH != 0)[y24]
names = traded.columns[traded.any()]
qv24 = QH[y24][names]
first_hype = CH.notna().idxmax().where(CH.notna().any())
liq = pd.DataFrame({"days_in_book_2024": traded[names].sum(), "days_short_2024": (WH[y24][names] < 0).sum(),
                    "hype_median_qv_2024_musd": (qv24.median() / 1e6).round(2),
                    "hype_p10_qv_2024_musd": (qv24.quantile(0.1) / 1e6).round(2),
                    "hype_first_candle": first_hype[names].astype(str).str[:10],
                    "days_in_book_without_hype_market": (traded[names] & CH[y24][names].isna()).sum()}).sort_values("hype_median_qv_2024_musd")
out["liquidity_2024_lowest15"] = liq.head(15).to_dict(orient="index")
out["liquidity_2024_summary"] = {"n_names_traded_2024": int(len(names)),
                                 "median_of_medians_musd": float(liq["hype_median_qv_2024_musd"].median()),
                                 "names_median_qv_lt_5m": int((liq["hype_median_qv_2024_musd"] < 5).sum()),
                                 "names_median_qv_lt_1m": int((liq["hype_median_qv_2024_musd"] < 1).sum()),
                                 "coin_days_in_book_2024_without_hype_market": int(liq["days_in_book_without_hype_market"].sum()),
                                 "share_coin_days_without_hype_market_2024": round(float(liq["days_in_book_without_hype_market"].sum() / liq["days_in_book_2024"].sum()), 3)}
print(liq.head(15).to_string()); print(out["liquidity_2024_summary"])
# coin-days in book without a HYPE market, by year (2023-05 onwards)
for yr in (2023, 2024):
    m_ = (C.index.year == yr) & (C.index >= pd.Timestamp("2023-05-15", tz="UTC"))
    tr = (WH != 0)[m_]
    out.setdefault("share_book_without_hype_market", {})[yr] = round(float((tr & CH[m_].isna()).sum().sum() / tr.sum().sum()), 3)
print("share of book coin-days with no HYPE market yet:", out["share_book_without_hype_market"])

# b) HYPE funding paid (same weights, HYPE funding of bar t+1)
FHn = FH.shift(-1)
fpay = -(WH * FHn)                                   # + = received, - = paid (fraction of equity)
hy = C.index >= pd.Timestamp("2023-06-01", tz="UTC")
fs = fpay[hy].stack(); fs = fs[fs != 0]
rates = FHn.where(WH < 0)[hy].stack()                # daily HYPE funding rate on shorted names (negative = short pays)
ratesL = FHn.where(WH > 0)[hy].stack()
out["hype_funding"] = {"short_names_worst_daily_rate": [(str((k[0] + pd.Timedelta('1D')).date()), k[1], round(float(v), 5)) for k, v in rates.nsmallest(10).items()],
                       "short_names_daily_rate_p1": round(float(rates.quantile(0.01)), 5),
                       "short_names_days_rate_lt_-0.5pct": int((rates < -0.005).sum()), "short_names_days_rate_lt_-1pct": int((rates < -0.01).sum()),
                       "long_names_worst_daily_rate_paid": [(str((k[0] + pd.Timedelta('1D')).date()), k[1], round(float(v), 5)) for k, v in ratesL.nlargest(5).items()],
                       "portfolio_worst_funding_days": [(str((k + pd.Timedelta('1D')).date()), round(float(v), 5)) for k, v in fpay[hy].sum(axis=1).nsmallest(5).items()],
                       "portfolio_funding_ann_hype": round(float(fpay[hy].sum(axis=1).mean() * 365), 4),
                       "portfolio_funding_ann_binance_same_period": round(float((-(WH * F.shift(-1)))[hy].sum(axis=1).mean() * 365), 4),
                       "short_names_mean_daily_rate_hype": round(float(rates.mean()), 6),
                       "short_names_mean_daily_rate_binance": round(float(F.shift(-1).where(WH < 0)[hy].stack().mean()), 6)}
print(out["hype_funding"])

# c) Binance F14 vs HYPE F14 ranks
FH14 = FH.rolling(14, min_periods=14).sum()
FB14 = F.rolling(14, min_periods=14).sum()
mask = UH & FH14.notna() & FB14.notna() & CH.notna()
rb = FB14.where(mask).rank(axis=1, pct=True); rh = FH14.where(mask).rank(axis=1, pct=True)
dd = mask.sum(axis=1) >= 15
cor = rb[dd].T.corrwith(rh[dd].T)                  # spearman per day (ranks)
topB = (rb >= 0.67) & mask; topH = (rh >= 0.67) & mask
ov = ((topB & topH).sum(axis=1) / topB.sum(axis=1).replace(0, np.nan))[dd]
out["rank_agreement"] = {"days": int(dd.sum()), "spearman_median": round(float(cor.median()), 3), "spearman_p10": round(float(cor.quantile(0.1)), 3),
                         "short_tercile_overlap_median": round(float(ov.median()), 3), "short_tercile_overlap_p10": round(float(ov.quantile(0.1)), 3),
                         "first_day": str(cor.index[0].date())}
print(out["rank_agreement"])

# venue replay 2023-07..2024-12: same universe (HYPE-listed, Binance top-50, HYPE market exists), HYPE prices+funding+cost
UV = UH & CH.notna() & (CH.notna().cumsum() >= 30)
for tag, sig in [("sig_binanceF14", -FB14), ("sig_hypeF14", -FH14)]:
    Wv = lab.quantile_ls_weights(sig, UV & sig.notna(), q=0.33)
    rv = lab.backtest(Wv, "hype_1d", cost=lab.HYPE_COST, start="2023-07-01")
    rvb = lab.backtest(Wv, "1d", cost=lab.COST, start="2023-07-01")
    out.setdefault("venue_replay_2023H2_2024", {})[tag] = {"hype_px_fund_sharpe": round(rv["metrics"]["sharpe"], 3), "hype_cagr": round(rv["metrics"]["cagr"], 4),
                                                         "hype_maxdd": round(rv["metrics"]["maxdd"], 4), "hype_fund_yr": round(rv["metrics"]["funding_per_year"], 4),
                                                         "binance_px_fund_sharpe_same_weights": round(rvb["metrics"]["sharpe"], 3), "npos": round(rv["metrics"]["avg_n_pos"], 1)}
    print(tag, out["venue_replay_2023H2_2024"][tag])

# d) liquidation proxy: per short episode, max adverse 1h-close move from entry
cube = hourly_cube(C)
W, _ = candidate_weights(U, F)
inS = (W < 0).values
cvals = C.values
eps = []
for j in range(inS.shape[1]):
    x = inS[:, j]
    i = 0; n = len(x)
    while i < n:
        if x[i]:
            k = i
            while k + 1 < n and x[k + 1]:
                k += 1
            e = cvals[i, j]
            seg = cube[i + 1:min(k + 2, n), :, j]
            mx = np.nanmax(seg) / e - 1 if seg.size and np.isfinite(seg).any() else np.nan
            eps.append((C.columns[j], C.index[i], k - i + 1, mx))
            i = k + 1
        else:
            i += 1
ep = pd.DataFrame(eps, columns=["sym", "start", "days", "max_adverse_1h"])
out["episodes_1h"] = {"n": int(len(ep)), **{f"share_touch_{int(X*100)}pct": round(float((ep['max_adverse_1h'] > X).mean()), 3) for X in (0.10, 0.15, 0.20, 0.28, 0.45, 1.0)},
                      "per_year_touch_20pct": {int(k): int(v) for k, v in (ep["max_adverse_1h"] > 0.20).groupby(ep["start"].dt.year).sum().items()}}
print(out["episodes_1h"])

# e) isolated-4x failure-mode estimate
for tag, X, slip in [("liq_4x_iso_~+20pct", 0.20, 0.04), ("liq_3x_iso_~+28pct", 0.28, 0.04), ("liq_2x_iso_~+45pct", 0.45, 0.04)]:
    d = sim(W, stop=X, stop_slip=slip, cube=cube)
    rr = d["r"]; e_ = (1 + rr).cumprod()
    sub = {nm: round(sh(rr[(rr.index >= pd.Timestamp(a, tz='UTC')) & (rr.index < pd.Timestamp(b, tz='UTC'))]), 3)
           for nm, a, b in [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"), ("2023-24", "2023-01-01", "2025-01-01")]}
    out.setdefault("isolated_margin_estimate", {})[tag] = {"sharpe": round(sh(rr), 3), "cagr": round(float(e_.iloc[-1] ** (365 / len(rr)) - 1), 4),
                                                         "maxdd": round(float((e_ / e_.cummax() - 1).min()), 4), "n_liquidations": d.attrs["nstops"], "sub": sub}
    print(tag, out["isolated_margin_estimate"][tag])

# f) $200 account replays
for tag, Wx, tf, cost, st in [("binance_px_top50_from_2020-03", W, "1d", lab.COST, "2020-03-01"),
                              ("binance_px_hype_univ_from_2020-03", WH, "1d", lab.HYPE_COST, "2020-03-01"),
                              ("hype_px_hype_univ_from_2023-07", lab.quantile_ls_weights(-FB14, UV & FB14.notna(), q=0.33), "hype_1d", lab.HYPE_COST, "2023-07-01"),
                              ("binance_px_hype_univ_from_2023-07", WH, "1d", lab.HYPE_COST, "2023-07-01")]:
    eqdf, summ = account_sim.simulate(Wx, tf=tf, equity0=200.0, min_order=10.0, cost=cost, start=st)
    summ["sharpe"] = round(sh(eqdf["equity"].pct_change().dropna()), 3)
    summ["skip_rate_per_day"] = round(summ["skipped_below_min"] / len(eqdf), 2)
    out.setdefault("account_200", {})[tag] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()}
    print(tag, out["account_200"][tag])
json.dump(out, open(os.path.join(HERE, "t3_hype.json"), "w"), indent=1, default=str)
