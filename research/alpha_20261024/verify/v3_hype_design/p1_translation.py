"""Part 1: signal translation Binance -> HYPE on the overlap 2023-06-01 .. 2024-12-31 (IS only).

PRE-REGISTERED (written before running):
  coins: lab.hype_symbols(); F14_bin = 14d sum of Binance daily funding (tf 1d);
  F14_hyp = 14d sum of HYPE daily funding (tf hype_1d, min 14 days of HYPE funding).
  1a) per-day cross-sectional Spearman(F14_hyp, F14_bin) on (i) all HYPE coins with both values and a
      HYPE close, (ii) the same restricted to hype_universe(top_k=50). Also top/bottom-K set overlap, K=6.
  1b) candidate (tercile L/S, q=0.33, -0.5/+0.5) on common universe
      U_ov = hype_universe("1d", top_k=50) & HYPE close & F14_hyp valid & F14_bin valid:
        (a) Binance signal + Binance returns/funding (tf 1d)
        (b) Binance signal + HYPE returns/funding    (tf hype_1d)
        (c) HYPE signal    + HYPE returns/funding    (tf hype_1d)
      cost = lab.HYPE_COST for all three (venue cost; isolates signal/return source). Also at 0.07%.
      Same three for the K=6 gross 1.0 daily top/bottom-K design (one design-relevant check).
  window for metrics: backtest df index in [2023-06-01, 2025-01-01).
  Total new configs here: 2 designs x 3 sources = 6 (diagnostic, not selection).
"""
from common_v3 import *
import json

A, B = "2023-06-01", "2025-01-01"
hs = lab.hype_symbols()
Fb = lab.load("fund", "1d"); Fh = lab.load("fund", "hype_1d")
Cb = lab.load("close", "1d"); Ch = lab.load("close", "hype_1d")
F14b = f14(Fb); F14h = f14(Fh)
Uh = lab.hype_universe("1d", top_k=50)
out = {}

# sanity: are HYPE closes a different series from Binance? (return correlation per year, BTC & SOL)
Rb = Cb.pct_change(fill_method=None); Rh = Ch.pct_change(fill_method=None)
san = {}
for c in ["BTCUSDT", "SOLUSDT", "LINKUSDT"]:
    for y in ["2021", "2023", "2024"]:
        x = pd.concat([Rb.loc[y, c], Rh.loc[y, c]], axis=1).dropna()
        if len(x) > 20:
            san[f"{c}_{y}"] = {"corr": round(float(x.corr().iloc[0, 1]), 4), "n": len(x),
                               "mean_abs_diff_bp": round(float((x.iloc[:, 0] - x.iloc[:, 1]).abs().mean() * 1e4), 1)}
out["hype_vs_binance_return_sanity"] = san
print(json.dumps(san, indent=0))

ix = Fb.index[(Fb.index >= pd.Timestamp(A, tz="UTC")) & (Fb.index < pd.Timestamp(B, tz="UTC"))]
hcols = [c for c in Fb.columns if c in hs]


def spearman_rows(X, Y, mask):
    X = X.where(mask); Y = Y.where(mask)
    ok = X.notna() & Y.notna()
    X = X.where(ok).rank(axis=1); Y = Y.where(ok).rank(axis=1)
    n = ok.sum(axis=1)
    xm = X.sub(X.mean(axis=1), axis=0); ym = Y.sub(Y.mean(axis=1), axis=0)
    r = (xm * ym).sum(axis=1) / np.sqrt((xm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
    return r.where(n >= 8), n


m_all = pd.DataFrame(False, index=Fb.index, columns=Fb.columns)
m_all[hcols] = True
m_all &= Ch.notna()
res_sp = {}
for tag, mask in [("all_hype_coins", m_all), ("hype_universe_top50", m_all & Uh)]:
    r, n = spearman_rows(F14h, F14b, mask)
    r = r.loc[ix]; n = n.loc[ix]
    res_sp[tag] = {"days": int(r.notna().sum()), "mean": float(r.mean()), "median": float(r.median()),
                   "p10": float(r.quantile(0.1)), "min": float(r.min()), "frac_below_0.5": float((r < 0.5).mean()),
                   "median_n_coins": float(n.median()), "n_coins_first": int(n.iloc[0]), "n_coins_last": int(n.iloc[-1]),
                   "mean_2023H2": float(r.loc["2023"].mean()), "mean_2024": float(r.loc["2024"].mean())}
    q = r.resample("QE").mean()
    res_sp[tag]["quarterly_mean"] = {str(k.date()): round(float(v), 3) for k, v in q.items()}
out["spearman_F14"] = res_sp
print(json.dumps(res_sp, indent=1))

# common universe for the three variants
Uov = Uh & Ch.notna() & F14h.notna() & F14b.notna()
nU = Uov.sum(axis=1).loc[ix]
out["Uov_size"] = {"min": int(nU.min()), "median": float(nU.median()), "max": int(nU.max())}
print("Uov size", out["Uov_size"])

# top/bottom-K overlap (K=6)
K = 6
Wb6 = topk_weights(F14b, Uov, K); Wh6 = topk_weights(F14h, Uov, K)
sb, sh_ = (Wb6 < 0), (Wh6 < 0); lb, lh = (Wb6 > 0), (Wh6 > 0)
ov_s = ((sb & sh_).sum(axis=1) / sb.sum(axis=1).replace(0, np.nan)).loc[ix]
ov_l = ((lb & lh).sum(axis=1) / lb.sum(axis=1).replace(0, np.nan)).loc[ix]
out["topK6_overlap"] = {"short_mean": float(ov_s.mean()), "long_mean": float(ov_l.mean()),
                        "short_2024": float(ov_s.loc["2024"].mean()), "long_2024": float(ov_l.loc["2024"].mean())}
print("K6 overlap", out["topK6_overlap"])

# ties at the K boundary (constant-baseline funding) on HYPE vs Binance
def tie_frac(F14, U, K):
    s = F14.where(U)
    hi = s.rank(axis=1, ascending=False, method="min"); lo = s.rank(axis=1, ascending=True, method="min")
    hi2 = s.rank(axis=1, ascending=False, method="max"); lo2 = s.rank(axis=1, ascending=True, method="max")
    tie_s = ((hi <= K) & (hi2 > K)).any(axis=1); tie_l = ((lo <= K) & (lo2 > K)).any(axis=1)
    return float(tie_s.loc[ix].mean()), float(tie_l.loc[ix].mean())
out["boundary_ties_K6"] = {"binance": tie_frac(F14b, Uov, 6), "hype": tie_frac(F14h, Uov, 6)}
print("ties", out["boundary_ties_K6"])

rows = []
for design in ["tercile", "K6_g1.0"]:
    for src, sig, tf in [("a_binSig_binRet", F14b, "1d"), ("b_binSig_hypRet", F14b, "hype_1d"),
                         ("c_hypSig_hypRet", F14h, "hype_1d")]:
        if design == "tercile":
            W = lab.quantile_ls_weights(-sig, Uov, q=0.33)
        else:
            W = topk_weights(sig, Uov, 6, gross=1.0)
        for cost in [lab.HYPE_COST, 0.0007]:
            res = lab.backtest(W, tf, cost=cost)
            df = window(res["df"], A, B)
            st = wstats(df["r"])
            d = {"design": design, "source": src, "cost": cost, **st,
                 "sh_2023H2": sh(window(df["r"], "2023-06-01", "2024-01-01")), "sh_2024": sh(window(df["r"], "2024-01-01", "2025-01-01")),
                 "fund_yr": float(df["fund_r"].mean() * 365), "price_yr": float(df["gross_r"].mean() * 365),
                 "cost_yr": float(df["cost_r"].mean() * 365), "turn_yr": float(df["turnover"].mean() * 365),
                 "npos": float(df["n_pos"].mean())}
            if cost == lab.HYPE_COST:
                d["sh_lag1"] = sh(window(lab.backtest(W, tf, cost=cost, extra_lag=1)["df"]["r"], A, B))
                d["_r"] = df["r"]
            rows.append(d)
            print(design, src, cost, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "_r"})
# correlation of daily returns between sources
for design in ["tercile", "K6_g1.0"]:
    rr = {d["source"]: d["_r"] for d in rows if d["design"] == design and "_r" in d}
    cm = pd.DataFrame(rr).corr()
    out[f"ret_corr_{design}"] = cm.round(3).to_dict()
    print(design, "\n", cm.round(3))
out["variants"] = [{k: v for k, v in d.items() if k != "_r"} for d in rows]
json.dump(out, open("p1_translation.json", "w"), indent=1, default=float)
