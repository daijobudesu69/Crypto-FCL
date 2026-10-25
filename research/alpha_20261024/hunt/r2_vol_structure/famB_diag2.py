"""FAMILY B diagnostics 2 for BASE = B_pkcc30_FLIP_R1 (no new selection; only checks of the frozen cell).
 D1 signed decomposition: within up-movers (5d ret>0) and down-movers, next-day excess return of low-vs-high
    pkcc30 quintile -> is the effect unsigned, or just disguised continuation?
 D2 spanning on extra controls: 1d / 5d XS continuation, signed close-location value (CLV5) L/S
 D3 drop the 3 biggest PnL contributors (DOGE, BCH, CRV) from the universe
 D4 exclude coin-days whose 30d window contains a degenerate range (H/L < 1.0005)
 D5 execution delay on 4h bars: trade 0 / 4h / 8h / 12h after the 00:00 UTC daily close
 D6 rolling 365d Sharpe (min / share of windows > 0)
 D7 $200 account sim of the 16-pos (8/8) book: units held between rebalances (no drift trading),
    entries sized equity/16, each order must be >= $10, taker cost HYPE_COST, funding charged
"""
import numpy as np, pandas as pd
from common import (universe, ls_weights, evaluate, fmt_row, ols_nw, controls, lab, COST, returns, sharpe,
                    sub_sharpes)
from signals import ohlc

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    U = universe(50)
    o_, h, l, c = ohlc()
    R = returns()
    pc = c.shift(1)
    lrange = np.log((h / l).clip(lower=1.0 + 1e-5))
    pk2 = lrange ** 2 / (4 * np.log(2))
    cc2 = np.log(c / pc) ** 2
    pkcc30 = np.sqrt(pk2.rolling(30, min_periods=24).mean() / cc2.rolling(30, min_periods=24).mean())
    score = -pkcc30
    Wb = ls_weights(score, U, q=0.2)
    base = lab.backtest(Wb, "1d", cost=COST)
    rb = base["df"]["r"]
    print("BASE", round(base["metrics"]["sharpe"], 3))

    # D1 signed decomposition
    Rn = R.shift(-1)
    ex = Rn.sub(Rn.where(U).mean(axis=1), axis=0)              # next-day excess vs EW universe
    r5 = c / c.shift(5) - 1
    q = pkcc30.where(U).rank(axis=1, pct=True)
    up = r5.where(U).rank(axis=1, pct=True) > 0.5
    for nm, sel in (("5d winners (top half)", up), ("5d losers (bottom half)", ~up & U)):
        lo = ex.where(sel & (q <= 0.2)).mean(axis=1); hi = ex.where(sel & (q >= 0.8)).mean(axis=1)
        d = (lo - hi).dropna()
        print(f"D1 {nm:24s}: low-pkcc minus high-pkcc next-day excess {d.mean() * 1e4:+.1f} bp/day, "
              f"t {d.mean() / d.std() * np.sqrt(len(d)):.2f}, ann Sharpe {sharpe(d):.2f}")
    # also by sign of the most recent day's return
    up1 = R.where(U).rank(axis=1, pct=True) > 0.5
    for nm, sel in (("1d winners", up1), ("1d losers", ~up1 & U)):
        lo = ex.where(sel & (q <= 0.2)).mean(axis=1); hi = ex.where(sel & (q >= 0.8)).mean(axis=1)
        d = (lo - hi).dropna()
        print(f"D1 {nm:24s}: low-pkcc minus high-pkcc {d.mean() * 1e4:+.1f} bp/day, t {d.mean() / d.std() * np.sqrt(len(d)):.2f}")

    # D2 spanning
    def book(s):
        return lab.backtest(ls_weights(s, U, q=0.2), "1d", cost=COST)["df"]["r"]
    clv = ((2 * c - h - l) / (h - l).where((h - l) > 0)).rolling(5, min_periods=4).mean()
    f = pd.DataFrame({"cont1d": book(R), "cont5d": book(r5), "clv5": book(clv)})
    j = pd.concat([rb.rename("s"), controls(), f], axis=1, sort=True).dropna()
    print("D2 corr with base:", j.corr()["s"].round(2).to_dict())
    print("D2 standalone Sharpe of controls:", {k: round(sharpe(j[k]), 2) for k in f.columns})
    b, t, n = ols_nw(j["s"], j[["xsmom14", "btc", "cont1d", "cont5d", "clv5"]])
    print("D2 span: alpha/yr {:+.1%} t {:.2f} | ".format(b["const"] * 365, t["const"]) +
          " ".join(f"{k} {b[k]:+.2f}(t{t[k]:.1f})" for k in ["xsmom14", "btc", "cont1d", "cont5d", "clv5"]))

    # D3 drop top contributors
    U3 = U.copy()
    for s in ("DOGEUSDT", "BCHUSDT", "CRVUSDT"):
        U3[s] = False
    o3, _ = evaluate(ls_weights(score, U3, q=0.2), "D3_drop_DOGE_BCH_CRV", mask=U3)
    print(o3["name"], fmt_row(o3), o3["yearly"])

    # D4 degenerate windows
    degen = ((h / l) < 1.0005).astype(float).rolling(30, min_periods=1).max() > 0
    U4 = U & ~degen
    o4, _ = evaluate(ls_weights(score, U4, q=0.2), "D4_no_degenerate", mask=U4)
    print(o4["name"], fmt_row(o4), "| coin-days removed:", int((U & degen).sum().sum()))

    # D6 rolling 365d Sharpe
    rs = rb.rolling(365).apply(lambda x: x.mean() / x.std() * np.sqrt(365), raw=True).dropna()
    print(f"D6 rolling-365d Sharpe: min {rs.min():.2f}  median {rs.median():.2f}  share>0 {(rs > 0).mean():.0%}")

    # D7 $200 account sim, 16-pos book (8/8), units held between changes
    W16 = ls_weights(score, U, n=8)
    F = lab.load("fund", "1d").astype("float64")
    C = c.values; Fv = F.shift(-1).fillna(0).values
    tgt = np.sign(W16.values)
    eq = 200.0; units = np.zeros(C.shape[1]); hist = []; skipped = 0; ntr = 0
    for t in range(len(C) - 1):
        px = C[t]
        # mark-to-market from t to t+1 handled below; rebalance at close t
        want = tgt[t]
        have = np.sign(units)
        for i in np.flatnonzero((want != have)):
            if not np.isfinite(px[i]):
                continue
            if units[i] != 0:                                   # exit (reduce-only close is always allowed)
                eq -= abs(units[i]) * px[i] * COST; units[i] = 0.0; ntr += 1
            if want[i] != 0:
                notional = eq / 16
                if notional < 10:
                    skipped += 1; continue
                units[i] = want[i] * notional / px[i]; eq -= notional * COST; ntr += 1
        nxt = C[t + 1]
        ok = np.isfinite(nxt) & np.isfinite(px)
        pnl = np.nansum(np.where(ok, units * (nxt - px), 0.0))
        fund = -np.nansum(np.where(ok, units * px * Fv[t], 0.0))
        # a coin without a next price is force-closed at the last price
        bad = (units != 0) & ~ok
        units[bad] = 0.0
        eq += pnl + fund
        hist.append((c.index[t + 1], eq))
    E = pd.Series(dict(hist)); E = E[E.index >= rb.index[0]]
    d = E.pct_change().dropna()
    print(f"D7 $200 sim 16-pos units-held: final ${E.iloc[-1]:.0f}, Sharpe {sharpe(d):.2f}, "
          f"maxDD {(E / E.cummax() - 1).min():.0%}, trades {ntr}, skipped(<$10) {skipped}")
    print("   sub-period Sharpe:", {k: round(v, 2) for k, v in sub_sharpes(d).items()})
    print("   yearly:", lab.yearly(d)["ret"].round(2).to_dict())

    # D5 execution delay on 4h bars (last, since it loads the 4h panels)
    lab.clear()
    c4 = lab.load("close", "4h")
    W4 = pd.DataFrame(np.nan, index=c4.index, columns=c4.columns)
    # daily row D (known at close D = D+24h) -> place at the 4h bar whose CLOSE is D+24h (open D+20h)
    Wd = Wb.copy(); Wd.index = Wd.index + pd.Timedelta("20h")
    W4 = Wd.reindex(c4.index).ffill(limit=5).fillna(0.0)
    for k, lbl in ((0, "0h"), (1, "4h"), (2, "8h"), (3, "12h"), (6, "24h")):
        m = lab.backtest(W4, "4h", cost=COST, extra_lag=k)["metrics"]
        print(f"D5 execution delay {lbl:>4s}: Sharpe {m['sharpe']:.2f}  CAGR {m['cagr']:+.1%}")
