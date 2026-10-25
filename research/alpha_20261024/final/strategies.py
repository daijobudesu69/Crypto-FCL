"""Frozen strategy implementations for the forward test (specs: ../SPEC_CHOP30_v1.md, verify_r2 DISP7-200 v1).

venue="binance": Binance daily OHLC/volume + Binance funding, universe = lab.hype_universe (Binance-ranked)
venue="hype":    HYPE daily OHLC/volume + HYPE funding, universe ranked by HYPE volume (coin must have traded
                 on HYPE, qv > 0, on >= 60 days) - what the live GitHub Actions bot would see
All signals use only open/high/low/close/volume. Funding enters as a cost only.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lab

MIN_ORDER = 10.0
SLOT = 12.5


def venue_panels(venue):
    if venue == "binance":
        P = {f: lab.load(f, "1d") for f in ["open", "high", "low", "close", "qv", "fund"]}
        U = lab.hype_universe("1d", top_k=50)
    else:
        P = {f: lab.load(f, "hype_1d") for f in ["open", "high", "low", "close", "qv", "fund"]}
        traded = P["qv"] > 0
        hist = traded.cumsum()
        dqv = P["qv"].where(traded).rolling(30, min_periods=15).median()
        elig = traded & (hist >= 60) & P["fund"].notna()
        rk = dqv.where(elig).rank(axis=1, ascending=False)
        U = elig & (rk <= 50)
        P["close"] = P["close"].where(traded | P["close"].isna())
    return P, U.fillna(False)


# ---------------------------------------------------------------- CHOP-30 (Parkinson / close-to-close ratio)
def chop_ratio(P, window=30, min_obs=24):
    H, L, C = P["high"], P["low"], P["close"]
    pk = np.log(H / L) ** 2 / (4 * np.log(2))
    cc = np.log(C / C.shift(1)) ** 2
    both = pk.notna() & cc.notna()
    pk, cc = pk.where(both), cc.where(both)
    n = both.astype(float).rolling(window).sum()
    mpk = pk.rolling(window, min_periods=1).mean()
    mcc = cc.rolling(window, min_periods=1).mean()
    ratio = np.sqrt(mpk / mcc)
    ratio = ratio.where((n >= min_obs) & (mcc > 0) & np.isfinite(ratio))
    return ratio


def chop_targets(P, U):
    ratio = chop_ratio(P).where(U)
    cols = np.array(P["close"].columns)
    order = np.argsort(cols)                       # ties broken A-Z

    def target(t, equity, state):
        r = ratio.iloc[t].values
        valid = np.flatnonzero(~np.isnan(r))
        N = len(valid)
        if equity is None:                         # ideal (fractional) book: quintile-capped k
            k = min(int(0.2 * N), N // 4)
        else:
            k = min(int(np.floor(equity * 1.0 / 2 / SLOT)), int(0.2 * N), N // 4)
        w = np.zeros(len(cols))
        if k < 3:
            return w, state
        rank_key = np.lexsort((np.argsort(order)[valid], r[valid]))   # ratio asc, then symbol A-Z
        srt = valid[rank_key]
        w[srt[:k]] = 0.5 / k
        w[srt[-k:]] = -0.5 / k
        return w, state
    return target, (lambda t, state: True)


# ---------------------------------------------------------------- DISP7-200 (dispersion-timed alts vs BTC)
def disp_targets(P, U, stratified=True):
    C = P["close"]
    lr7 = np.log(C / C.shift(7)).where(U)
    disp = lr7.std(axis=1, ddof=1)
    disp = disp.where(lr7.notna().sum(axis=1) >= 3)
    pct = disp.rolling(365, min_periods=120).apply(lambda a: np.mean(a[~np.isnan(a)] <= a[-1]) if not np.isnan(a[-1]) else np.nan, raw=True)
    st = np.zeros(len(C)); s = 0
    for i, p in enumerate(pct.values):
        if not np.isnan(p):
            s = 1 if p > 2 / 3 else (-1 if p < 1 / 3 else s)
        st[i] = s
    cols = list(C.columns); bi = cols.index("BTCUSDT")
    qv = P["qv"].where(P["qv"] > 0).rolling(30, min_periods=15).median()
    mondays = C.index.dayofweek == 0

    def target(t, equity, state):
        s = st[t]
        w = np.zeros(len(cols))
        if s == 0:
            return w, s
        alts = [j for j in np.flatnonzero(U.iloc[t].values) if j != bi and not np.isnan(C.iat[t, j])]
        if not alts:
            return w, s
        alts = sorted(alts, key=lambda j: -np.nan_to_num(qv.iat[t, j]))
        M = len(alts)
        if equity is None or not stratified:
            sel = alts
        else:
            N = min(M, max(4, int(np.floor(equity * 1.0 / 2 / SLOT))))
            sel = [alts[int(np.floor((i + 0.5) * M / N))] for i in range(N)]
        w[sel] = s * 0.5 / len(sel)
        w[bi] = -s * 0.5
        return w, s

    def retarget(t, state):
        return st[t] != state or mondays[t]
    return target, retarget


# ---------------------------------------------------------------- shared small-account replay
def replay(P, target, retarget, start, end=None, equity0=200.0, cost=lab.HYPE_COST, ideal=False):
    """ideal=True: fractional book, no minimum order, daily re-target (lab-like reference)."""
    C = P["close"]; Fd = P["fund"]
    ix = C.index
    t0 = ix.get_indexer([pd.Timestamp(start, tz="UTC")])[0]
    t1 = len(ix) - 1 if end is None else ix.get_indexer([pd.Timestamp(end, tz="UTC")])[0]
    c = C.values.astype(float); f = Fd.fillna(0.0).values.astype(float)
    m = c.shape[1]
    pos = np.zeros(m); eq = equity0; state = None
    rows = []; fees = fund_tot = 0.0; orders = 0
    for t in range(t0, t1):
        nxt_ok = ~np.isnan(c[t + 1]) & ~np.isnan(c[t])
        if ideal or retarget(t, state):
            w, state = target(t, None if ideal else eq, state)
            tgt = w * eq
            tgt[~nxt_ok] = 0.0
            if ideal:
                new = tgt
            else:
                diff = tgt - pos
                send = (np.abs(diff) >= MIN_ORDER) | ((tgt == 0) & (pos != 0)) | ((pos == 0) & (np.abs(tgt) >= MIN_ORDER))
                new = np.where(send, tgt, pos)
                new[(pos == 0) & (np.abs(tgt) < MIN_ORDER)] = 0.0
        else:
            new = pos.copy()
        new[~nxt_ok] = 0.0                          # cannot hold through a missing bar
        traded = np.abs(new - pos)
        orders += int((traded > 1e-9).sum())
        fee = traded.sum() * cost
        r = np.where(nxt_ok, c[t + 1] / c[t] - 1, 0.0)
        pnl = (new * r).sum()
        fund = -(new * f[t + 1]).sum()
        eq = eq + pnl + fund - fee
        fees += fee; fund_tot += fund
        pos = new * (1 + r)
        rows.append((ix[t + 1], eq, np.abs(new).sum() / max(eq, 1e-9), new.sum() / max(eq, 1e-9),
                     int((np.abs(new) > 0).sum()), fee, fund))
        if eq <= 0:
            break
    d = pd.DataFrame(rows, columns=["ts", "equity", "gross", "net", "n_pos", "fee", "fund"]).set_index("ts")
    d["r"] = d["equity"].pct_change()
    d.iloc[0, d.columns.get_loc("r")] = d["equity"].iloc[0] / equity0 - 1
    mt = lab.metrics(d["r"], "1d")
    s = {"start": str(d.index[0].date()), "end": str(d.index[-1].date()), "equity_start": equity0,
         "equity_end": float(d["equity"].iloc[-1]), "total_return": float(d["equity"].iloc[-1] / equity0 - 1),
         "cagr": mt["cagr"], "sharpe": mt["sharpe"], "maxdd": mt["maxdd"], "t_nw": mt["t_nw"], "vol": mt["vol"],
         "worst_month": mt["worst_month"], "best_month": mt["best_month"], "pct_pos_months": mt["pct_pos_months"],
         "fees": fees, "funding": fund_tot, "orders": orders, "avg_n_pos": float(d["n_pos"].mean()),
         "avg_gross": float(d["gross"].mean())}
    return d, s
