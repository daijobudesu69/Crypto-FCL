"""AUTO-EXTRACTED function library from s3_design.py (definitions only, no runs)."""

import sys, os
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import numpy as np, pandas as pd
import lab, account_sim

OUT = os.path.dirname(os.path.abspath(__file__))
log = open(os.devnull, "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.write(s + "\n"); log.flush()


BTC = "BTCUSDT"
C = lab.load("close", "1d")
QV = lab.load("qv", "1d")
F = lab.load("fund", "1d")
U = lab.hype_universe("1d", top_k=50)
cols = list(C.columns); ib = cols.index(BTC)
dqv = QV.rolling(30, min_periods=15).median()
disp = np.log(C / C.shift(7)).where(U).std(axis=1)
pctl = disp.rolling(365, min_periods=120).apply(lambda x: (x[np.isfinite(x)] <= x[-1]).mean() if np.isfinite(x[-1]) else np.nan, raw=True)


def make_state(p):
    st = np.zeros(len(p)); cur = 0.0
    for i, v in enumerate(p.values):
        if np.isfinite(v):
            if v > 2 / 3: cur = 1.0
            elif v < 1 / 3: cur = -1.0
        st[i] = cur
    return pd.Series(st, index=p.index)


STATE = make_state(pctl)
Uv = U.values; Dv = dqv.values
Rn = C.pct_change(fill_method=None).shift(-1).values
Fn = F.shift(-1).values
monday_close = np.array([(ts + pd.Timedelta("1D")).weekday() == 0 for ts in C.index])


def target_w(i, eq, st, G, S):
    w = np.zeros(len(cols))
    if st == 0:
        return w
    m = Uv[i].copy(); m[ib] = False
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        return w
    order = idx[np.argsort(-np.nan_to_num(Dv[i, idx], nan=-1.0), kind="stable")]
    M = len(order)
    N = int(min(M, max(4, np.floor(eq * G / 2 / 12.5))))
    if S == "LIQ":
        sel = order[:N]
    else:
        sel = order[np.floor((np.arange(N) + 0.5) * M / N).astype(int)]
    w[sel] = st * (G / 2) / N
    w[ib] = -st * G / 2
    return w


def sim(G, S, R, start, end="2025-01-01", eq0=200.0, cost=lab.HYPE_COST, min_order=10.0, lag=0, fixedW=None, track=False):
    """Mirror of account_sim.simulate with targets computed from current equity."""
    st_all = STATE.shift(lag).fillna(0.0).values
    ix = np.flatnonzero((C.index >= pd.Timestamp(start, tz="UTC")) & (C.index < pd.Timestamp(end, tz="UTC")))
    pos = np.zeros(len(cols)); eq = eq0
    rows = []; n_orders = 0; fees = 0.0; fund_t = 0.0; skipped = 0
    entry = np.full(len(cols), np.nan); mae = np.zeros(len(cols)); side = np.zeros(len(cols))
    exc = []  # (coin, side, max adverse excursion)
    open_sizes = []
    prev_st = None
    for i in ix:
        r = Rn[i]; f = Fn[i]
        st = st_all[i]
        if fixedW is not None:
            do_trade = True
            tgt = np.nan_to_num(fixedW[i]) * eq
        else:
            do_trade = (R == "D") or (prev_st is None) or (st != prev_st) or monday_close[i]
            tgt = target_w(i, eq, st, G, S) * eq if do_trade else pos.copy()
        prev_st = st
        small = (np.abs(tgt) < min_order) & (tgt != 0)
        skipped += int(small.sum()); tgt[small] = 0.0
        tgt[np.isnan(r)] = 0.0
        diff = tgt - pos
        trade = np.abs(diff) >= min_order
        trade |= (tgt == 0) & (pos != 0)
        if not do_trade:
            trade = (tgt == 0) & (pos != 0)
        new = np.where(trade, tgt, pos)
        d = np.abs(new - pos)
        opened = (pos == 0) & (new != 0) | (np.sign(pos) * np.sign(new) < 0)
        open_sizes.extend(np.abs(new[opened]).tolist())
        n_orders += int((d > 0).sum())
        fee = d.sum() * cost
        rr = np.nan_to_num(r)
        pnl = (new * rr).sum(); fu = -(new * np.nan_to_num(f)).sum()
        eq = eq + pnl + fu - fee
        fees += fee; fund_t += fu
        if track:
            px = C.values[i]
            closed = (side != 0) & (np.sign(new) != side)
            for j in np.flatnonzero(closed):
                exc.append((cols[j], side[j], mae[j]))
            side[closed] = 0; mae[closed] = 0
            newo = (new != 0) & (side == 0)
            side[newo] = np.sign(new[newo]); entry[newo] = px[newo]
            # adverse excursion measured on next close (daily closes only; intraday wicks not covered)
            nx = C.values[i + 1] if i + 1 < len(C) else px
            held = side != 0
            ae = np.where(side > 0, entry / nx - 1, nx / entry - 1)  # loss fraction (>0 = adverse) relative to entry
            ae = np.where(side > 0, 1 - nx / entry, nx / entry - 1)
            mae[held] = np.maximum(mae[held], np.nan_to_num(ae[held]))
        pos = new * (1 + rr)
        rows.append((C.index[i] + pd.Timedelta("1D"), eq, np.abs(new).sum() / max(eq, 1e-9), int((new != 0).sum())))
        if eq <= 0:
            break
    out = pd.DataFrame(rows, columns=["ts", "equity", "gross", "n_pos"]).set_index("ts")
    e = out["equity"]
    dr = e.pct_change().fillna(e.iloc[0] / eq0 - 1)
    traded = dr[out["gross"] > 0]
    first = np.flatnonzero(out["gross"].values > 0)
    dr2 = dr.iloc[first[0]:] if len(first) else dr
    shp = dr2.mean() / dr2.std() * np.sqrt(365)
    yrs = len(dr2) / 365
    summ = {"eq_end": float(e.iloc[-1]), "tot": float(e.iloc[-1] / eq0 - 1), "cagr": float((e.iloc[-1] / eq0) ** (1 / yrs) - 1),
            "sharpe": float(shp), "maxdd": float((e / e.cummax() - 1).min()), "orders": n_orders, "fees": fees, "fund": fund_t,
            "skipped": skipped, "avg_npos": float(out["n_pos"][out["gross"] > 0].mean()), "avg_gross": float(traded.size and out["gross"][out["gross"] > 0].mean()),
            "min_open": float(np.min(open_sizes)) if open_sizes else np.nan}
    return out, dr2, summ, exc


