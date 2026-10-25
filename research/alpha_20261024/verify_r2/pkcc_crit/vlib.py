"""Verifier pkcc_crit helpers. Independent re-implementation of PKCC30 (does not import hunter code).
Data ONLY via lab.py (IS 2020-01-01..2024-12-31). ALPHA_FORWARD is never set.
Signal inputs: high, low, close (1d) only; universe ranking uses qv through lab.hype_universe."""
import os, sys
import numpy as np, pandas as pd

ROOT = r"C:\Crypto data\backtest data and more\alpha_20261024"
sys.path.insert(0, ROOT)
import lab  # noqa: E402

assert os.environ.get("ALPHA_FORWARD") != "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "REPORT.md")
COST = lab.HYPE_COST
SUBS = [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"),
        ("2023-24", "2023-01-01", "2025-01-01")]


def append(txt):
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(txt.rstrip() + "\n\n")


def hlc():
    h = lab.load("high", "1d").astype("float64")
    l = lab.load("low", "1d").astype("float64")
    c = lab.load("close", "1d").astype("float64")
    return h, l, c


def comps():
    h, l, c = hlc()
    pc = c.shift(1)
    lr = np.log((h / l).clip(lower=1.0 + 1e-5))
    pk2 = lr ** 2 / (4 * np.log(2))
    cc2 = np.log(c / pc) ** 2
    return pk2, cc2


def pkcc(W=30, minp=None, pk2=None, cc2=None):
    if pk2 is None:
        pk2, cc2 = comps()
    mp = minp if minp is not None else max(3, int(0.8 * W))
    return np.sqrt(pk2.rolling(W, min_periods=mp).mean() / cc2.rolling(W, min_periods=mp).mean())


def qls(score, mask, q=0.2, n=None, gross=1.0):
    """long top-q of score, short bottom-q, EW legs, gross/2 per side; zero when a leg is empty."""
    s = score.where(mask)
    if n is None:
        r = s.rank(axis=1, pct=True)
        lo = (r >= 1 - q); sh = (r <= q)
    else:
        cnt = s.notna().sum(axis=1)
        rd = s.rank(axis=1, ascending=False, method="first")
        ra = s.rank(axis=1, ascending=True, method="first")
        nn = np.minimum(n, cnt // 2).values[:, None]
        lo = pd.DataFrame(rd.values <= nn, index=s.index, columns=s.columns) & s.notna()
        sh = pd.DataFrame(ra.values <= nn, index=s.index, columns=s.columns) & s.notna()
    lo = lo.astype(float); sh = sh.astype(float)
    nl = lo.sum(axis=1).replace(0, np.nan); ns = sh.sum(axis=1).replace(0, np.nan)
    W = lo.div(nl, axis=0) - sh.div(ns, axis=0)
    ok = nl.notna() & ns.notna()
    return W.where(ok, 0.0).fillna(0.0) * gross / 2


def sharpe(r):
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * np.sqrt(365)) if len(r) > 10 and r.std() > 0 else np.nan


def win(r, a, b):
    return r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))]


def subs(r):
    return {nm: sharpe(win(r, a, b)) for nm, a, b in SUBS}


def bt(W, cost=COST, lag=0, reb=1):
    return lab.backtest(W, "1d", cost=cost, extra_lag=lag, rebalance_every=reb)


def base_weights(W_=30, k=50, q=0.2, n=None, mask=None):
    U = lab.hype_universe("1d", top_k=k) if mask is None else mask
    return qls(-pkcc(W_), U, q=q, n=n), U


def fmt(x, p=2):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"
