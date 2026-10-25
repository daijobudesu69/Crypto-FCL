"""disp_impl step 3 - $200 design for DISP7. PRE-REGISTERED (written before any step-3 run).

Signal (unchanged, replicated in step 2): U = lab.hype_universe("1d", top_k=50); disp7 = XS std (ddof 1) of
ln(C_t/C_{t-7}) over U_t; p = percentile of disp7_t in trailing 365 bars incl. today (min 120 obs, "<=" count);
state +1 if p > 2/3, -1 if p < 1/3, else previous (0 before first trigger).

Book at close t with account equity E_t (USD):
  M_t   = alts in U_t excluding BTC, ranked by 30d median daily quote volume (lab's dqv)
  N_t   = min(M_t, max(4, floor(E_t * G / 2 / 12.5)))      (each alt target >= $12.5 -> >= $10 after drift)
  alt weights  = state * (G/2) / N_t on the N_t selected alts;  BTC weight = -state * G/2
  selection S:  LIQ   = the N_t most liquid alts
                STRAT = stratified over the liquidity ranking: indices floor((i+0.5)*M_t/N_t), i=0..N_t-1
  rebalance R:  D = re-target every daily close (orders < $10 not sent, = account_sim min_trade)
                W = re-target only on state-change closes and on the Monday 00:00 UTC close; no orders otherwise
Grid (12 configs): G in {1.0, 1.25, 1.5} x S in {LIQ, STRAT} x R in {D, W}
Simulator: account_sim.simulate logic re-coded with equity-dependent targets (validated to reproduce
account_sim.simulate exactly on a fixed-W book first). equity0=200, min_order=10, cost=lab.HYPE_COST, funding on.
Runs: A = 2020-03-01 .. 2024-12-31 ($200 at 2020-03-01; state 0 until 2020-06-29),  B = fresh $200 at 2023-01-01.
Selection rule (robustness, fixed in advance):
  score = min(Sharpe_A, Sharpe_B), eligible only if MaxDD_A > -35%;
  pick max score; if the top two are within 0.05, prefer lower G, then W over D (fewer orders), then LIQ.
Post-selection diagnostics on the pick only (not used for selection): 2x cost, signal lag 1 day,
yearly returns, order sizes, isolated-margin adverse excursions per position.
Outputs -> s3_out.txt, s3_grid.csv
"""
import sys, os
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import numpy as np, pandas as pd
import lab, account_sim

OUT = os.path.dirname(os.path.abspath(__file__))
log = open(os.path.join(OUT, "s3_out.txt"), "w")


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


# ---- validation vs account_sim.simulate on a fixed-W book (top-8 liquid alts, G=1, equity-independent)
Wfix = np.zeros((len(C), len(cols)))
for i in range(len(C)):
    Wfix[i] = target_w(i, 200.0, STATE.values[i], 1.0, "LIQ")
Wdf = pd.DataFrame(Wfix, index=C.index, columns=cols)
o_ref, s_ref = account_sim.simulate(Wdf, "1d", equity0=200.0, min_order=10.0, cost=lab.HYPE_COST, start="2020-03-01", end="2025-01-01")
o_me, _, s_me, _ = sim(1.0, "LIQ", "D", "2020-03-01", fixedW=Wfix)
P(f"VALIDATION fixed-W: account_sim eq_end {s_ref['equity_end']:.4f} orders {s_ref['orders']} fees {s_ref['fees_usd']:.4f} | "
  f"mine eq_end {s_me['eq_end']:.4f} orders {s_me['orders']} fees {s_me['fees']:.4f} | max|d equity| {np.abs(o_ref['equity'].values - o_me['equity'].values).max():.2e}")

# ---- ideal lab reference
Wideal = U.copy(); Wideal[BTC] = False
Wideal = Wideal.astype(float).div(Wideal.sum(1).replace(0, np.nan), axis=0).fillna(0.0).mul(STATE, axis=0) * 0.5
Wideal[BTC] = -0.5 * STATE
ideal = lab.backtest(Wideal, "1d", cost=lab.HYPE_COST)["df"]["r"]


def shp(x):
    x = x.dropna(); return x.mean() / x.std() * np.sqrt(365)


P(f"ideal lab (G=1.0, full basket): Sharpe 2020-06..2024 {shp(ideal):.3f}, 2023-24 {shp(ideal[ideal.index >= pd.Timestamp('2023-01-02', tz='UTC')]):.3f}")

rows = []
for G in [1.0, 1.25, 1.5]:
    for S in ["LIQ", "STRAT"]:
        for R in ["D", "W"]:
            oA, rA, sA, _ = sim(G, S, R, "2020-03-01")
            oB, rB, sB, _ = sim(G, S, R, "2023-01-01")
            yA = (1 + rA).groupby(rA.index.year).prod() - 1
            corr = rA.corr(ideal.reindex(rA.index))
            row = {"G": G, "S": S, "R": R, "SR_A": sA["sharpe"], "eqA": sA["eq_end"], "cagrA": sA["cagr"], "ddA": sA["maxdd"],
                   "SR_B": sB["sharpe"], "eqB": sB["eq_end"], "totB": sB["tot"], "ddB": sB["maxdd"],
                   "ordersA": sA["orders"], "feesA": sA["fees"], "fundA": sA["fund"], "skipA": sA["skipped"], "nposA": sA["avg_npos"],
                   "nposB": sB["avg_npos"], "ordersB": sB["orders"], "min_openA": sA["min_open"], "min_openB": sB["min_open"],
                   "corr_ideal": corr, **{f"y{k}": v for k, v in yA.items()}}
            row["score"] = min(sA["sharpe"], sB["sharpe"]) if sA["maxdd"] > -0.35 else -9
            rows.append(row)
            P(f"G={G} {S:5s} {R}: A SR {sA['sharpe']:.2f} eq ${sA['eq_end']:.0f} DD {sA['maxdd']:.1%} | B SR {sB['sharpe']:.2f} eq ${sB['eq_end']:.0f} DD {sB['maxdd']:.1%} "
              f"| orders A {sA['orders']} B {sB['orders']} npos B {sB['avg_npos']:.1f} min_open {sB['min_open']:.1f} corr_ideal {corr:.2f} | "
              f"yearly {dict((int(k), round(v, 3)) for k, v in yA.items())}")
grid = pd.DataFrame(rows)
grid.to_csv(os.path.join(OUT, "s3_grid.csv"), index=False)
g = grid.sort_values("score", ascending=False).reset_index(drop=True)
best = g.iloc[0]
cands = g[g["score"] >= best["score"] - 0.05].copy()
cands["Rk"] = (cands["R"] != "W").astype(int); cands["Sk"] = (cands["S"] != "LIQ").astype(int)
pick = cands.sort_values(["G", "Rk", "Sk"]).iloc[0]
P("\nranking by score:\n" + g[["G", "S", "R", "SR_A", "SR_B", "ddA", "ddB", "score", "corr_ideal"]].round(3).to_string())
P(f"\nPICK (pre-registered rule): G={pick['G']} S={pick['S']} R={pick['R']} score {pick['score']:.3f}")
P("grid median SR_A %.2f SR_B %.2f" % (grid["SR_A"].median(), grid["SR_B"].median()))

# ---- post-selection diagnostics on the pick
G, S, R = float(pick["G"]), pick["S"], pick["R"]
for lbl, kw in [("2x cost", dict(cost=2 * lab.HYPE_COST)), ("lag1", dict(lag=1))]:
    _, rA, sA, _ = sim(G, S, R, "2020-03-01", **kw)
    _, rB, sB, _ = sim(G, S, R, "2023-01-01", **kw)
    P(f"pick {lbl}: A SR {sA['sharpe']:.2f} eq ${sA['eq_end']:.0f} DD {sA['maxdd']:.1%} | B SR {sB['sharpe']:.2f} eq ${sB['eq_end']:.0f} DD {sB['maxdd']:.1%}")
oB, rB, sB, exc = sim(G, S, R, "2023-01-01", track=True)
yB = (1 + rB).groupby(rB.index.year).prod() - 1
P(f"pick fresh-2023: yearly {dict((int(k), round(v, 3)) for k, v in yB.items())}, orders {sB['orders']}, fees ${sB['fees']:.2f}, funding ${sB['fund']:.2f}, "
  f"avg npos {sB['avg_npos']:.1f}, avg gross {sB['avg_gross']:.2f}, min opened ${sB['min_open']:.2f}")
oA, rA, sA, excA = sim(G, S, R, "2020-03-01", track=True)
for nm, E in [("A", excA), ("B", exc)]:
    ex = pd.DataFrame(E, columns=["coin", "side", "mae"])
    alt = ex[ex["coin"] != BTC]
    sh_ = alt[alt["side"] < 0]["mae"]; lo_ = alt[alt["side"] > 0]["mae"]
    P(f"isolated-margin check run {nm}: alt positions {len(alt)}; shorts: max adverse (close-to-close from entry) {sh_.max():.1%}, "
      f">25% {int((sh_ > .25).sum())}, >40% {int((sh_ > .40).sum())}, >90% {int((sh_ > .9).sum())}; longs: max adverse {lo_.max():.1%}, >25% {int((lo_ > .25).sum())}, >40% {int((lo_ > .40).sum())}")
    P("   worst 5 short excursions:", alt[alt['side'] < 0].sort_values('mae', ascending=False).head(5).round(3).values.tolist())
mo = (1 + rB).groupby(rB.index.to_period("M") if rB.index.tz is None else rB.index.tz_localize(None).to_period("M")).prod() - 1
P(f"pick fresh-2023 monthly: pos months {(mo > 0).mean():.0%}, worst {mo.min():.1%}, best {mo.max():.1%}")
log.close()
