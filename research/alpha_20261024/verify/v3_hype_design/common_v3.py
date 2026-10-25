"""v3_hype_design helpers. IS only (lab.py truncates at 2025-01-01; ALPHA_FORWARD never set)."""
import sys, os
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab
import numpy as np, pandas as pd

ROOT = r"C:\Crypto data\backtest data and more\alpha_20261024"


def sh(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) > 20 and x.std() > 0 else np.nan


def window(r, a, b):
    return r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]


def subperiods(r):
    return {k: sh(window(r, a, b)) for k, a, b in [("2020-21", "2020-01-01", "2022-01-01"),
                                                    ("2022", "2022-01-01", "2023-01-01"),
                                                    ("2023-24", "2023-01-01", "2025-01-01"),
                                                    ("2021-24", "2021-01-01", "2025-01-01")]}


def wstats(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    return {"sharpe": sh(r), "tot": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "maxdd": float((eq / eq.cummax() - 1).min()), "days": len(r)}


def f14(F, L=14):
    """Sum of the last L daily funding sums known at the close of bar t (trailing only)."""
    return F.rolling(L, min_periods=L).sum()


def topk_weights(F14, U, K, gross=1.0, min_k=2, third_cap=True):
    """Short the K highest-F14 names, long the K lowest-F14 names, equal weight, gross/2 per side.
    K_eff = min(K, floor(N/3)) where N = names in U with valid F14 (keeps a neutral middle);
    flat if K_eff < min_k. Ties broken by column order (deterministic)."""
    s = F14.where(U)
    N = s.notna().sum(axis=1)
    Keff = np.minimum(K, (N // 3) if third_cap else (N // 2)).astype(int)
    Keff = Keff.where(Keff >= min_k, 0)
    hi = s.rank(axis=1, ascending=False, method="first")   # 1 = highest funding
    lo = s.rank(axis=1, ascending=True, method="first")    # 1 = lowest funding
    shrt = hi.le(Keff, axis=0) & s.notna()
    lng = lo.le(Keff, axis=0) & s.notna()
    k = Keff.replace(0, np.nan)
    W = lng.astype(float).div(k, axis=0) - shrt.astype(float).div(k, axis=0)
    return (W * gross / 2).fillna(0.0)


def band_weights(F14, U, K, buf=2, gross=1.0, min_k=2, rebalance_every=1):
    """No-trade band: a held short stays while its funding rank (desc) <= K_eff+buf and it is in U;
    a held long stays while its ascending rank <= K_eff+buf. Empty slots are filled with the
    best-ranked names not already held. Only evaluated on rebalance days; held otherwise."""
    s = F14.where(U)
    N = s.notna().sum(axis=1).values
    hi = s.rank(axis=1, ascending=False, method="first").values
    lo = s.rank(axis=1, ascending=True, method="first").values
    cols = s.columns
    T, M = s.shape
    W = np.zeros((T, M))
    S, L = set(), set()
    for t in range(T):
        Keff = min(K, N[t] // 3)
        if Keff < min_k:
            S, L = set(), set()
            continue
        if t % rebalance_every == 0 or (not S and not L):
            h, l = hi[t], lo[t]
            S = {j for j in S if not np.isnan(h[j]) and h[j] <= Keff + buf}
            L = {j for j in L if not np.isnan(l[j]) and l[j] <= Keff + buf}
            # trim if K shrank
            if len(S) > Keff:
                S = set(sorted(S, key=lambda j: h[j])[:Keff])
            if len(L) > Keff:
                L = set(sorted(L, key=lambda j: l[j])[:Keff])
            for j in np.argsort(np.where(np.isnan(h), 1e9, h)):
                if len(S) >= Keff or np.isnan(h[j]):
                    break
                if j not in S and j not in L:
                    S.add(j)
            for j in np.argsort(np.where(np.isnan(l), 1e9, l)):
                if len(L) >= Keff or np.isnan(l[j]):
                    break
                if j not in L and j not in S:
                    L.add(j)
        else:
            # between rebalances drop names that left the universe
            S = {j for j in S if not np.isnan(hi[t][j])}
            L = {j for j in L if not np.isnan(lo[t][j])}
        for j in S:
            W[t, j] = -gross / 2 / max(len(S), 1)
        for j in L:
            W[t, j] = gross / 2 / max(len(L), 1)
    return pd.DataFrame(W, index=s.index, columns=cols)


def hold_every(W, k):
    if k <= 1:
        return W
    keep = np.arange(len(W)) % k == 0
    return W.where(pd.Series(keep, index=W.index), np.nan).ffill().fillna(0.0)
