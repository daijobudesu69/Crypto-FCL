# Dijalankan 25 Okt 2026 di PC dengan data lake + cache alpha_20261024 (PYTHONUTF8=1). Hasil: archive_signal_study.json.
"""Study case: HYPE predictedFundings mati -> F14 dari rekonstruksi arsip premium index Binance.
Rumus sama dengan fclbot/bootstrap.py. Bandingkan dengan F14 dari funding settled asli."""
import os, sys, json, numpy as np, pandas as pd
os.environ["ALPHA_FORWARD"] = "1"
LAB = r"C:\Crypto data\backtest data and more\alpha_20261024"; sys.path.insert(0, LAB); os.chdir(LAB)
import lab, fcl
DATA = r"C:\Crypto data\backtest data and more\data\binance\um"
C, F, U, F14, mom = fcl.panels()
ix = C.index
rec = {}
for s in C.columns:
    pp, fp = os.path.join(DATA, "premium_index", "1h", f"{s}.parquet"), os.path.join(DATA, "funding", f"{s}.parquet")
    if not (os.path.exists(pp) and os.path.exists(fp)):
        continue
    p = pd.read_parquet(pp).set_index("ts")["prem_close"].sort_index()
    p = p[~p.index.duplicated()]
    p = p.reindex(pd.date_range(p.index[0], p.index[-1], freq="1h"))
    f = pd.read_parquet(fp)[["ts", "funding_interval_hours"]].dropna()
    T = f["ts"].astype("datetime64[ns, UTC]").dt.round("1h")
    ih = f["funding_interval_hours"].astype(int).values
    roll = {h: p.rolling(h, min_periods=max(1, h // 2)).mean() for h in set(ih)}
    P = np.array([roll[h].get(t - pd.Timedelta("1h"), np.nan) for t, h in zip(T, ih)])
    I = 0.0001 * ih / 8
    rate = P + np.clip(I - P, -0.0005, 0.0005)
    bar = (T - pd.Timedelta("1ms")).dt.floor("1D")
    g = pd.Series(rate, index=bar.values).groupby(level=0).sum(min_count=1)
    g.index = pd.DatetimeIndex(g.index).tz_convert("UTC") if g.index.tz is not None else pd.DatetimeIndex(g.index).tz_localize("UTC")
    fs = g.reindex(ix)
    rec[s] = fs.where(fs.notna() | C[s].isna(), 0.0)
Fr = pd.DataFrame(rec).reindex(columns=C.columns)
F14r = Fr.rolling(14, min_periods=14).sum()
F14r_lag2 = Fr.shift(2).rolling(12, min_periods=12).sum() * 14 / 12        # arsip terbit 1-2 hari kemudian

def xs(A, B, a, b):
    rho, ov_s, ov_l = [], [], []
    for t in range(ix.get_loc(pd.Timestamp(a, tz="UTC")), ix.get_loc(pd.Timestamp(b, tz="UTC"))):
        ok = U.iloc[t].values & A.iloc[t].notna().values & B.iloc[t].notna().values
        if ok.sum() < 24: continue
        x, y = A.iloc[t].values[ok], B.iloc[t].values[ok]
        rho.append(pd.Series(x).rank().corr(pd.Series(y).rank()))
        k = 8; ox, oy = np.argsort(x), np.argsort(y)
        ov_s.append(len(set(ox[-k:]) & set(oy[-k:])) / k); ov_l.append(len(set(ox[:k]) & set(oy[:k])) / k)
    r = np.array(rho)
    return dict(days=len(r), spearman_mean=round(r.mean(), 3), p10=round(np.percentile(r, 10), 3), min=round(r.min(), 3),
                overlap_short8=round(np.mean(ov_s), 3), overlap_long8=round(np.mean(ov_l), 3))

out = {"xs": {}, "bt": {}}
for nm, B in [("rec", F14r), ("rec_lag2", F14r_lag2)]:
    for per, a, b in [("IS", "2020-03-15", "2024-12-31"), ("FWD", "2025-01-01", "2026-10-01")]:
        out["xs"][f"{nm}|{per}"] = xs(F14, B, a, b); print(nm, per, out["xs"][f"{nm}|{per}"], flush=True)

orig = fcl.panels
for nm, B in [("true", F14), ("rec", F14r), ("rec_lag2", F14r_lag2)]:
    fcl.panels = lambda B=B: (C, F, U, B, mom)
    for per, st, en, ven in [("FWD hype", "2025-01-01", None, "hype"), ("IS binance", "2020-03-01", "2024-12-31", "binance")]:
        _, s, _ = fcl.run(st, en, venue=ven)
        out["bt"][f"{nm}|{per}"] = {k: round(s[k], 4) for k in ["equity_end", "sharpe", "maxdd", "cagr"]}
        print(nm, per, out["bt"][f"{nm}|{per}"], flush=True)
fcl.panels = orig
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive_signal_study.json"), "w"), indent=1)
