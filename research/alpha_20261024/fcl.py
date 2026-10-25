"""FCL — Funding-Crowding Long/Short. Frozen spec v1.0 (see SPEC_FCL_v1.md). Daily book, executed on HYPE.

Each day at 00:00 UTC:
  universe  coins listed on HYPE, >= 60 days of Binance history, top 50 by trailing 30d median Binance
            daily quote volume (point in time)                                   -> N names
  signal    F14 = sum of Binance USDT-M funding settlements over the last 14 days
  K         names per side = min(floor(N/3), floor(equity * GROSS / 2 / 12.5)), at least 4
            ("tercile" mode ignores equity: K = floor(N/3))
  short     the K highest-F14 names, after removing names whose 14d return is in the universe's top 10%
  long      the K lowest-F14 names (never a name that is short)
  weights   equal weight, +GROSS/2 long side, -GROSS/2 short side (dollar neutral), GROSS = 1.0
  stop      a short is closed if a 1h close reaches entry * 1.40 (fill +0.5% slippage); it stays out of
            the short side until the signal itself drops it
  orders    trade to target; skip changes smaller than MIN_ORDER (except full exits / new entries)
"""
import numpy as np, pandas as pd
import lab

GROSS = 1.0
MIN_ORDER = 10.0
SLOT_USD = 12.5
K_MIN = 4
MOM_SKIP = 0.90
STOP = 0.40
STOP_SLIP = 0.005


def panels():
    C = lab.load("close", "1d")
    F = lab.load("fund", "1d")
    U = lab.hype_universe("1d", top_k=50)
    F14 = F.rolling(14, min_periods=14).sum()
    mom = C / C.shift(14) - 1
    return C, F, U, F14, mom


def hourly_cube(index, cols):
    c1 = lab.load("close", "1h").reindex(columns=cols)
    ix = pd.date_range(index[0], index[-1] + pd.Timedelta("23h"), freq="1h")
    return c1.reindex(ix).values.astype("float32").reshape(len(index), 24, -1)


def run(start, end=None, venue="binance", equity0=200.0, mode="capacity", stop=STOP, mom_skip=MOM_SKIP,
        cost=lab.HYPE_COST, min_order=MIN_ORDER, gross=GROSS):
    """Day-by-day replay. venue='binance': Binance closes + Binance funding; 'hype': HYPE closes + HYPE funding
    (signal is Binance in both). mode='capacity' (equity-limited K, $ min order) or 'tercile' (fractional,
    no min order: the infinite-capital reference). Returns (daily DataFrame, summary dict, trades list)."""
    C, F, U, F14, mom = panels()
    if venue == "hype":
        PX = lab.load("close", "hype_1d").reindex_like(C)
        FV = lab.load("fund", "hype_1d").reindex_like(C)
        U = U & PX.notna() & FV.notna()
    else:
        PX, FV = C, F
    ix = C.index
    t0 = ix.get_indexer([pd.Timestamp(start, tz="UTC")])[0]
    t1 = len(ix) - 1 if end is None else ix.get_indexer([pd.Timestamp(end, tz="UTC")])[0]
    cube = hourly_cube(ix, C.columns) if stop is not None else None
    px = PX.values.astype(float); cb = C.values.astype(float); fv = FV.fillna(0.0).values.astype(float)
    f14 = F14.values; mo = mom.values; u = U.values
    m = len(C.columns)
    pos = np.zeros(m)                     # USD notional
    entry = np.full(m, np.nan); banned = np.zeros(m, bool); was_short = np.zeros(m, bool)
    eq = equity0
    rows, trades = [], []
    fees = fundsum = 0.0; nstops = 0
    for t in range(t0, t1):
        ok = u[t] & ~np.isnan(f14[t]) & ~np.isnan(px[t]) & ~np.isnan(px[t + 1])
        idx = np.flatnonzero(ok)
        N = len(idx)
        tgt = np.zeros(m)
        if N >= 3 * K_MIN or (mode == "tercile" and N >= 6):
            if mode == "tercile":
                K = N // 3
            else:
                K = int(min(N // 3, max(K_MIN, np.floor(eq * gross / 2 / SLOT_USD))))
            # short side: highest F14 after the momentum filter
            mrank = pd.Series(mo[t, idx]).rank(pct=True).values
            cand_s = idx[(mrank <= mom_skip) | np.isnan(mrank)]
            cand_s = cand_s[np.argsort(-f14[t, cand_s], kind="stable")]
            short = [j for j in cand_s if not banned[j]][:K]
            cand_l = idx[np.argsort(f14[t, idx], kind="stable")]
            long_ = [j for j in cand_l if j not in set(short)][:K]
            if len(short) and len(long_):
                tgt[short] = -gross / 2 / len(short)
                tgt[long_] = gross / 2 / len(long_)
        tgt_usd = tgt * eq
        if mode == "capacity":
            small = (np.abs(tgt_usd) < min_order) & (tgt_usd != 0)
            tgt_usd[small] = 0.0
            diff = tgt_usd - pos
            trade = (np.abs(diff) >= min_order) | ((tgt_usd == 0) & (pos != 0)) | ((pos == 0) & (tgt_usd != 0))
            new = np.where(trade, tgt_usd, pos)
        else:
            new = tgt_usd
        traded = np.abs(new - pos)
        fee = traded.sum() * cost
        # short episode bookkeeping (entry = close at which the short was opened)
        is_short = new < 0
        start_ep = is_short & ~was_short
        entry[start_ep] = cb[t, start_ep]
        entry[~is_short] = np.nan
        banned[~(tgt < 0) & ~is_short] = False         # ban ends once the signal drops the name
        r = px[t + 1] / px[t] - 1
        r = np.where(np.isnan(r), 0.0, r)
        hit = np.zeros(m, bool)
        if stop is not None:
            lvl = entry * (1 + stop)
            hc = cube[t + 1]
            with np.errstate(invalid="ignore"):
                cross = hc >= lvl[None, :]
            anyc = cross.any(axis=0) & is_short
            if anyc.any():
                first = np.argmax(cross, axis=0)
                fill = hc[first, np.arange(m)].astype(float) * (1 + STOP_SLIP)
                r = r.copy()
                r[anyc] = fill[anyc] / cb[t, anyc] - 1     # Binance trigger/fill expressed as a return
                hit = anyc
                nstops += int(hit.sum())
        pnl = (new * r).sum()
        fund = -(new * np.where(hit, 0.5, 1.0) * fv[t + 1]).sum()
        stop_fee = (np.abs(new[hit]) * (1 + r[hit])).sum() * cost
        eq = eq + pnl + fund - fee - stop_fee
        fees += fee + stop_fee; fundsum += fund
        for j in np.flatnonzero(traded > 0):
            trades.append((ix[t], C.columns[j], float(new[j] - pos[j])))
        pos = new * (1 + r)
        pos[hit] = 0.0
        banned |= hit
        was_short = pos < 0
        rows.append((ix[t + 1], eq, np.abs(new).sum() / max(eq, 1e-9), int((new != 0).sum()), N,
                     fee + stop_fee, fund, pnl, int(hit.sum())))
        if eq <= 0:
            break
    d = pd.DataFrame(rows, columns=["ts", "equity", "gross", "n_pos", "n_univ", "fee", "fund", "pnl", "stops"]).set_index("ts")
    d["r"] = d["equity"].pct_change().fillna(d["equity"].iloc[0] / equity0 - 1)
    mt = lab.metrics(d["r"], "1d")
    summary = {"venue": venue, "mode": mode, "start": str(d.index[0].date()), "end": str(d.index[-1].date()),
               "equity_start": equity0, "equity_end": float(d["equity"].iloc[-1]),
               "total_return": float(d["equity"].iloc[-1] / equity0 - 1), "cagr": mt["cagr"], "sharpe": mt["sharpe"],
               "maxdd": mt["maxdd"], "t_nw": mt["t_nw"], "vol": mt["vol"], "worst_month": mt["worst_month"],
               "pct_pos_months": mt["pct_pos_months"], "fees": fees, "funding": fundsum, "stops": nstops,
               "avg_n_pos": float(d["n_pos"].mean()), "avg_gross": float(d["gross"].mean()), "orders": len(trades)}
    return d, summary, trades
