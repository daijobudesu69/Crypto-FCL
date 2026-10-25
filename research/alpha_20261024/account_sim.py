"""Small-account simulator: replays target weights on a real-sized account.

Differences vs lab.backtest (which assumes infinitely divisible positions):
- starting equity (default 200 USD), compounding on actual equity
- every order must be >= min_order USD (Hyperliquid minimum is 10 USD); target positions whose
  notional is below min_order are skipped (set to 0); trades smaller than min_trade are not sent
  (no-trade band), so small drifts are left alone
- fees per side on traded notional, funding on held notional, liquidation-free (gross <= 1.5x)
Positions are tracked in USD notional, marked to market bar by bar.
"""
import numpy as np, pandas as pd
import lab


def simulate(W, tf="1d", equity0=200.0, min_order=10.0, min_trade=None, cost=lab.COST, start=None, end=None):
    min_trade = min_order if min_trade is None else min_trade
    C = lab.load("close", tf)
    R = C.pct_change(fill_method=None).shift(-1)   # return earned after trading at close t
    F = lab.load("fund", tf).shift(-1)
    W = W.reindex(index=C.index, columns=C.columns).fillna(0.0)
    ix = C.index
    if start is not None:
        keep = ix >= pd.Timestamp(start, tz="UTC")
        if end is not None:
            keep &= ix < pd.Timestamp(end, tz="UTC")
        ix = ix[keep]
    cols = C.columns
    pos = pd.Series(0.0, index=cols)           # USD notional, + long / - short
    eq = equity0
    rows = []
    n_orders = 0
    fees_total = 0.0
    fund_total = 0.0
    skipped = 0
    for t in ix:
        w = W.loc[t]
        r = R.loc[t]
        f = F.loc[t]
        tgt = w * eq
        # positions below the venue minimum cannot be opened
        small = (tgt.abs() < min_order) & (tgt != 0)
        skipped += int(small.sum())
        tgt[small] = 0.0
        # cannot hold a name with no next-bar price
        tgt[r.isna()] = 0.0
        diff = tgt - pos
        trade = diff.abs() >= min_trade
        closing = (tgt == 0) & (pos != 0)          # full exits always allowed
        trade |= closing
        new_pos = pos.where(~trade, tgt)
        traded = (new_pos - pos).abs().sum()
        n_orders += int(((new_pos - pos).abs() > 0).sum())
        fee = traded * cost
        pnl = (new_pos * r.fillna(0.0)).sum()
        fund = -(new_pos * f.fillna(0.0)).sum()
        eq = eq + pnl + fund - fee
        fees_total += fee
        fund_total += fund
        pos = new_pos * (1 + r.fillna(0.0))
        rows.append((t, eq, new_pos.abs().sum() / max(eq, 1e-9), int((new_pos != 0).sum()), fee, fund))
        if eq <= 0:
            break
    out = pd.DataFrame(rows, columns=["ts", "equity", "gross", "n_pos", "fee", "fund"]).set_index("ts")
    out.index = out.index + pd.Timedelta(lab.TF_DELTA[tf])
    summary = {"equity_start": equity0, "equity_end": float(out["equity"].iloc[-1]),
               "total_return": float(out["equity"].iloc[-1] / equity0 - 1),
               "maxdd": float((out["equity"] / out["equity"].cummax() - 1).min()),
               "fees_usd": fees_total, "funding_usd": fund_total, "orders": n_orders,
               "skipped_below_min": skipped, "avg_gross": float(out["gross"].mean()),
               "avg_n_pos": float(out["n_pos"].mean())}
    return out, summary
