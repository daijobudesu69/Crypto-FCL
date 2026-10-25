# Dijalankan 25 Okt 2026 di PC dengan data lake + cache alpha_20261024 (PYTHONUTF8=1). Butuh gate/oi_logs/: salinan oi_logs dari repo
# daijobudesu69/Crypto-Funding-Strategy (git archive origin/main oi_logs, data 2026-07-12 -> 2026-10-24; tidak di repo ini).
# gate/gate_intervals.json = funding_interval kontrak Gate per 25 Okt 2026 (api.gateio.ws /futures/usdt/contracts).
"""Validasi logger Gate.io (repo Crypto-Funding-Strategy, oi_logs/) sebagai cadangan sinyal F14 FCL.
F14 Gate vs F14 Binance settled (data lake), dibandingkan juga dengan rekonstruksi arsip premium index."""
import os, sys, glob, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "archive_signal_study.py")).read().split("def xs")[0]
exec(src)                                        # -> C, F, U, F14, mom, F14r, F14r_lag2, ix
iv = json.load(open(os.path.join(HERE, "gate", "gate_intervals.json")))
g = pd.concat([pd.read_parquet(f, columns=["time", "symbol", "contract", "last_funding_rate"])
               for f in sorted(glob.glob(os.path.join(HERE, "gate", "oi_logs", "*", "gate_contract_stats.parquet")))])
g["last_funding_rate"] = pd.to_numeric(g["last_funding_rate"], errors="coerce")
g["ts"] = pd.to_datetime(g["time"], unit="s", utc=True).dt.floor("1h")
g = g.drop_duplicates(["contract", "ts"]).sort_values("ts")
print("gate rows", len(g), "contracts", g.contract.nunique(), g.ts.min(), g.ts.max(), flush=True)
fg = {}
for (con, sym), d in g.groupby(["contract", "symbol"]):
    if sym not in C.columns or not iv.get(con):
        continue
    ih = int(iv[con]) // 3600
    s = d.set_index("ts")["last_funding_rate"]
    T = s.index[(s.index.hour % ih == 0)]           # snapshot di jam settlement = rate settlement itu (jam ke-0 setelah settle)
    rate = s.reindex(T)
    bar = (rate.index - pd.Timedelta("1ms")).floor("1D")
    fg[sym] = rate.groupby(bar).sum(min_count=1).reindex(ix)
Fg = pd.DataFrame(fg).reindex(columns=C.columns)
first = g.ts.min().ceil("1D")
Fg.loc[Fg.index < first] = np.nan
F14g = Fg.rolling(14, min_periods=14).sum()
a, b = (first + pd.Timedelta("14D")).strftime("%Y-%m-%d"), "2026-10-01"
def xs(A, B):
    rho, ovs, ovl, n = [], [], [], []
    for t in range(ix.get_loc(pd.Timestamp(a, tz="UTC")), ix.get_loc(pd.Timestamp(b, tz="UTC"))):
        ok = U.iloc[t].values & A.iloc[t].notna().values & B.iloc[t].notna().values
        if ok.sum() < 24: continue
        x, y = A.iloc[t].values[ok], B.iloc[t].values[ok]
        rho.append(pd.Series(x).rank().corr(pd.Series(y).rank())); n.append(ok.sum())
        ox, oy = np.argsort(x), np.argsort(y)
        ovs.append(len(set(ox[-8:]) & set(oy[-8:])) / 8); ovl.append(len(set(ox[:8]) & set(oy[:8])) / 8)
    r = np.array(rho)
    return dict(window=f"{a}..{b}", days=len(r), coins_avg=round(float(np.mean(n)), 1), spearman_mean=round(float(r.mean()), 3),
                p10=round(float(np.percentile(r, 10)), 3), min=round(float(r.min()), 3),
                overlap_short8=round(float(np.mean(ovs)), 3), overlap_long8=round(float(np.mean(ovl)), 3))
out = {"gate_vs_binance": xs(F14, F14g), "archive_vs_binance_same_window": xs(F14, F14r),
       "archive_lag2_vs_binance_same_window": xs(F14, F14r_lag2), "gate_vs_archive": xs(F14r, F14g)}
for k, v in out.items(): print(k, v)
json.dump(out, open(os.path.join(HERE, "gate_signal_study.json"), "w"), indent=1)
