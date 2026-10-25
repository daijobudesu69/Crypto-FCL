"""pkcc_impl verifier helpers -- written from the rules text only (hunter code NOT read).

Rules (from task text):
  ratio = sqrt( mean_n[(ln H/L)^2 / (4 ln2)] / mean_n[(ln C_t/C_{t-1})^2] ), n=30, >=24 obs
  universe = lab.hype_universe("1d", top_k=50)
  long bottom 20% by ratio, short top 20%, equal weight, 0.5 gross per side, daily at 00:00 UTC close.
Own return/cost/funding loop (dollar-equity based) independent of lab.backtest.
"""
import sys
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import numpy as np, pandas as pd
import lab

LN2x4 = 4.0 * np.log(2.0)


def panels():
    H = lab.load("high", "1d").astype("float64")
    L = lab.load("low", "1d").astype("float64")
    C = lab.load("close", "1d").astype("float64")
    return H, L, C


def ratio_signal(H, L, C, n=30, minp=None):
    """Trailing-only Parkinson / close-to-close variance ratio (sqrt)."""
    if minp is None:
        minp = int(np.ceil(0.8 * n))
    pk = (np.log(H / L)) ** 2 / LN2x4
    cc = (np.log(C / C.shift(1))) ** 2
    # use only rows where both terms exist (joint observations)
    both = pk.notna() & cc.notna()
    pk = pk.where(both)
    cc = cc.where(both)
    mpk = pk.rolling(n, min_periods=minp).mean()
    mcc = cc.rolling(n, min_periods=minp).mean()
    r = np.sqrt(mpk / mcc)
    r = r.replace([np.inf, -np.inf], np.nan)
    return r


def ls_weights(score, U, q=0.2, k=None, gross=1.0, side="both"):
    """Long LOW score (bottom q or bottom k), short HIGH score; equal weight, gross/2 per side.
    Implemented with explicit per-row loops (own code, not lab.quantile_ls_weights)."""
    s = score.where(U)
    W = np.zeros(s.shape)
    vals = s.values
    for i in range(vals.shape[0]):
        row = vals[i]
        ok = np.flatnonzero(~np.isnan(row))
        N = len(ok)
        if N < 5:
            continue
        order = ok[np.argsort(row[ok], kind="mergesort")]
        if k is None:
            # percentile rank like pandas rank(pct=True, method='average'); count names with rank<=q
            ranks = pd.Series(row[ok]).rank(pct=True).values
            lo_idx = ok[ranks <= q]
            hi_idx = ok[ranks >= 1 - q]
        else:
            kk = min(k, N // 2)
            lo_idx = order[:kk]
            hi_idx = order[-kk:]
        if side in ("both", "long") and len(lo_idx):
            W[i, lo_idx] += (gross / 2 if side == "both" else gross) / len(lo_idx)
        if side in ("both", "short") and len(hi_idx):
            W[i, hi_idx] -= (gross / 2 if side == "both" else gross) / len(hi_idx)
    return pd.DataFrame(W, index=score.index, columns=score.columns)


def own_bt(W, cost=lab.HYPE_COST, extra_lag=0, funding=True, start=None, end=None, C=None):
    """Own dollar-equity loop. W[t] decided at close t (lagged by extra_lag), traded at close t,
    earns close t -> close t+1 and pays funding settled during bar t+1. Cost on |target - drifted|.
    Returns daily net return series indexed by the END of the holding bar (t+1 open ts).
    C: optional execution-price panel (daily index) to test delayed execution."""
    if C is None:
        C = lab.load("close", "1d").astype("float64")
    F = lab.load("fund", "1d").astype("float64")
    W = W.reindex(index=C.index, columns=C.columns).fillna(0.0)
    if extra_lag:
        W = W.shift(extra_lag).fillna(0.0)
    R = (C / C.shift(1) - 1).values  # R[t] = close t / close t-1 - 1
    Fv = F.fillna(0.0).values
    Wv = W.values
    T, N = Wv.shape
    eq = 1.0
    hold = np.zeros(N)  # dollar notional held after drift, before trading at close t
    out = np.full(T, np.nan)
    gr = np.full(T, np.nan); fr = np.full(T, np.nan); cr = np.full(T, np.nan); tu = np.full(T, np.nan)
    gx = np.full(T, np.nan)
    for t in range(T - 1):
        rn = R[t + 1]
        tgt = Wv[t] * eq
        # cannot hold a name without a next-bar price
        bad = np.isnan(rn)
        tgt = np.where(bad, 0.0, tgt)
        traded = np.abs(tgt - hold).sum()
        fee = traded * cost
        rn0 = np.where(bad, 0.0, rn)
        pnl = (tgt * rn0).sum()
        fund = -(tgt * Fv[t + 1]).sum() if funding else 0.0
        e0 = eq
        eq = eq + pnl + fund - fee
        out[t + 1] = eq / e0 - 1
        gr[t + 1] = pnl / e0; fr[t + 1] = fund / e0; cr[t + 1] = fee / e0; tu[t + 1] = traded / e0
        gx[t + 1] = np.abs(tgt).sum() / e0
        hold = tgt * (1 + rn0)
    df = pd.DataFrame({"r": out, "gross_r": gr, "fund_r": fr, "cost_r": cr, "turnover": tu, "gx": gx},
                      index=C.index)
    df = df.iloc[1:]
    nz = np.flatnonzero(df["gx"].values > 0)
    df = df.iloc[nz[0]:] if len(nz) else df
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df


def sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else np.nan


def maxdd(r):
    e = (1 + r.dropna()).cumprod()
    return float((e / e.cummax() - 1).min())


def cagr(r):
    r = r.dropna()
    e = (1 + r).prod()
    return float(e ** (365 / len(r)) - 1)


def subp(r):
    r = r.dropna()
    y = r.index.year
    return {"2020-21": sharpe(r[(y >= 2020) & (y <= 2021)]), "2022": sharpe(r[y == 2022]),
            "2023-24": sharpe(r[(y >= 2023) & (y <= 2024)])}


def yearly(r):
    r = r.dropna()
    g = r.groupby(r.index.year)
    return pd.DataFrame({"ret": g.apply(lambda x: (1 + x).prod() - 1), "sharpe": g.apply(sharpe),
                         "maxdd": g.apply(maxdd)})
