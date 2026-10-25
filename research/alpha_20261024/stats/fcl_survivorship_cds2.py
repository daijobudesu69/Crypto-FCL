"""Survivorship check for FCL on the user's Crypto-Funding-Strategy panel (C:\Crypto data 2\data\panel.parquet):
776 Binance USDT perps INCLUDING delisted ones, daily 2024-08 -> 2026-07. Core FCL (tercile and k=8 per side,
momentum skip top-10%, no stop), top-50 by 30d ADV point-in-time, cost 0.065%/side + funding."""
import numpy as np, pandas as pd, json, os
p = pd.read_parquet(r"C:\Crypto data 2\data\panel.parquet", columns=["date", "symbol", "close_t", "adv_usd_30", "funding_24h_sum", "n_hist_days"])
C = p.pivot(index="date", columns="symbol", values="close_t").sort_index()
ADV = p.pivot(index="date", columns="symbol", values="adv_usd_30").reindex_like(C)
FD = p.pivot(index="date", columns="symbol", values="funding_24h_sum").reindex_like(C)
H = p.pivot(index="date", columns="symbol", values="n_hist_days").reindex_like(C)
last = C.apply(lambda s: s.last_valid_index())
dead = set(last[last < C.index[-1] - pd.Timedelta("3D")].index)
alive150 = set(pd.read_csv(r"C:\Crypto data\backtest data and more\data\universe\binance_um_top150.csv")["symbol"])
hype = set(pd.read_csv(r"C:\Crypto data\backtest data and more\alpha_20261024\hype_map.csv").dropna()["binance"])
F14 = FD.rolling(14, min_periods=14).sum()
MOM = C / C.shift(14) - 1
R = C.pct_change(fill_method=None).shift(-1)         # return earned after trading at close t
FN = FD.shift(-1)                                    # funding paid over the next day


def run(allowed, k_fixed=None, cost=0.00065, mom_skip=0.9):
    cols = [c for c in C.columns if c in allowed]
    elig = (H[cols] >= 60) & C[cols].notna() & F14[cols].notna() & R[cols].notna()
    adv = ADV[cols].where(elig)
    rk = adv.rank(axis=1, ascending=False)
    U = elig & (rk <= 50)
    W = pd.DataFrame(0.0, index=C.index, columns=cols)
    for t in C.index[30:]:
        u = U.loc[t]; names = u[u].index
        N = len(names)
        if N < 12:
            continue
        k = N // 3 if k_fixed is None else min(k_fixed, N // 3)
        f = F14.loc[t, names]; m = MOM.loc[t, names].rank(pct=True)
        cs = f[(m <= mom_skip) | m.isna()].sort_values(ascending=False).index[:k]
        cl = f.drop(cs).sort_values().index[:k]
        W.loc[t, cs] = -0.5 / len(cs); W.loc[t, cl] = 0.5 / len(cl)
    gross = (W * R[cols].fillna(0)).sum(axis=1)
    fund = -(W * FN[cols].fillna(0)).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    r = (gross + fund - turn * cost)
    r = r[W.abs().sum(axis=1).shift(0) > 0]
    eq = (1 + r).cumprod()
    sr = r.mean() / r.std() * np.sqrt(365)
    yr = (1 + r).groupby(r.index.year).prod() - 1
    short_dead = (W.lt(0) & W.columns.isin(list(dead))).sum().sum() / max(W.lt(0).sum().sum(), 1)
    long_dead = (W.gt(0) & W.columns.isin(list(dead))).sum().sum() / max(W.gt(0).sum().sum(), 1)
    return {"sharpe": round(float(sr), 2), "total": round(float(eq.iloc[-1] - 1), 3), "maxdd": round(float((eq / eq.cummax() - 1).min()), 3),
            "yearly": {int(k): round(float(v), 3) for k, v in yr.items()}, "share_short_dead": round(float(short_dead), 3),
            "share_long_dead": round(float(long_dead), 3), "days": len(r), "start": str(r.index[0].date())}


out = {"n_symbols": C.shape[1], "n_dead": len(dead), "period": [str(C.index[0].date()), str(C.index[-1].date())]}
for name, allowed in [("all_incl_delisted", set(C.columns)), ("alive_today_150_only", alive150), ("hype_listed_only", hype),
                      ("all_minus_hype_coin", set(C.columns) - {"HYPEUSDT"})]:
    for kname, kk in [("tercile", None), ("k8", 8)]:
        out[f"{name}|{kname}"] = run(allowed, kk)
        print(name, kname, out[f"{name}|{kname}"], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fcl_survivorship_cds2.json"), "w"), indent=1)
