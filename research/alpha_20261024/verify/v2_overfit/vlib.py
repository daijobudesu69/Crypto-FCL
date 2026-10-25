"""Helpers for verifier v2_overfit (research data 2020-2024 only, via lab.py; ALPHA_FORWARD never set)."""
import os, sys
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab
import numpy as np, pandas as pd

assert os.environ.get("ALPHA_FORWARD") != "1", "forward data must never be enabled"
HERE = os.path.dirname(os.path.abspath(__file__))


def sh(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) > 20 and x.std() > 0 else np.nan


def between(r, a, b):
    return r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]


def signal(L):
    F = lab.load("fund", "1d")
    return -F.rolling(L, min_periods=L).sum()


def universe_pool(top_k, pool=None, min_hist_days=60, lookback_days=30):
    """Same logic as lab.universe('1d'), but the liquidity ranking is done only inside `pool` columns
    (pool=None -> identical to lab.universe)."""
    c = lab.load("close", "1d")
    qv = lab.load("qv", "1d")
    if pool is not None:
        c = c[[x for x in c.columns if x in set(pool)]]
        qv = qv[c.columns]
    hist = c.notna().cumsum()
    elig = c.notna() & (hist >= min_hist_days)
    dqv = qv.rolling(lookback_days, min_periods=max(1, lookback_days // 2)).median()
    rk = dqv.where(elig).rank(axis=1, ascending=False)
    elig &= rk <= top_k
    return elig.reindex(columns=lab.load("close", "1d").columns, fill_value=False)


def cand_weights(L=14, q=0.33, top_k=50, U=None):
    if U is None:
        U = lab.universe("1d", top_k=top_k)
    return lab.quantile_ls_weights(signal(L), U, q=q)


def coin_pnl(W, cost=lab.COST):
    """Per-coin decomposition replicating lab.backtest exactly: price, funding, cost (drift-adjusted)."""
    C = lab.load("close", "1d")
    R = C.pct_change(fill_method=None)
    F = lab.load("fund", "1d")
    W = W.reindex(index=C.index, columns=C.columns).fillna(0.0)
    Rn = R.shift(-1)
    Wv = W.where(Rn.notna() | (W == 0), 0.0)
    Rn0 = Rn.fillna(0.0)
    Fn0 = F.shift(-1).fillna(0.0)
    price = Wv * Rn0
    fund = -(Wv * Fn0)
    gross_r = price.sum(axis=1)
    prev = Wv.shift(1).fillna(0.0)
    Rcur = Rn0.shift(1).fillna(0.0)
    port_prev = gross_r.shift(1).fillna(0.0)
    drift = prev * (1 + Rcur) / (1 + port_prev).values[:, None]
    costc = (Wv - drift).abs() * cost
    tot = price + fund - costc
    for d in (price, fund, costc, tot):
        d.index = d.index + pd.Timedelta("1D")
    return price.iloc[:-1], fund.iloc[:-1], costc.iloc[:-1], tot.iloc[:-1]


def stationary_bootstrap_sharpes(r, mean_block=20, reps=2000, seed=7):
    x = np.asarray(pd.Series(r).dropna().values, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    ar = np.arange(n)
    out = np.empty(reps)
    for k in range(reps):
        s = rng.random(n) < p
        s[0] = True
        new = rng.integers(0, n, size=n)
        bs = np.maximum.accumulate(np.where(s, ar, 0))
        idx = (new[bs] + (ar - bs)) % n
        y = x[idx]
        out[k] = y.mean() / y.std(ddof=1) * np.sqrt(365)
    return out
