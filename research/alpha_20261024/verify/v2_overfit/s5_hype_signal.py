"""v2_overfit / step 5 (descriptive, 4 fixed runs, no selection): does the edge survive when the signal
is built from HYPE funding and PnL is booked on HYPE prices/funding? Window 2023-07-01..2024-12-31 only
(HYPE funding exists from 2023-05; 14d warm-up). Candidate rules otherwise unchanged (q=0.33, EW legs).
Common eligibility mask for all 4 runs: lab.hype_universe top-50 & Binance F14 & HYPE F14 & HYPE close.
  R1 Binance F14 signal, Binance prices+funding, lab.COST
  R2 HYPE    F14 signal, Binance prices+funding, lab.COST
  R3 Binance F14 signal, HYPE prices+funding, lab.HYPE_COST
  R4 HYPE    F14 signal, HYPE prices+funding, lab.HYPE_COST   (= what live trading would do)
"""
from vlib import *
import json

A, B = "2023-07-01", "2025-01-01"
Uh = lab.hype_universe("1d", top_k=50)
sb = -lab.load("fund", "1d").rolling(14, min_periods=14).sum()
shh = -lab.load("fund", "hype_1d").rolling(14, min_periods=14).sum()
ch = lab.load("close", "hype_1d")
mask = Uh & sb.notna() & shh.notna() & ch.notna()
out = {"avg_names_in_mask": float(mask.sum(axis=1)[mask.index >= pd.Timestamp(A, tz="UTC")].mean())}
for tag, s, tf, cost in [("R1_binSig_binPx", sb, "1d", lab.COST), ("R2_hypeSig_binPx", shh, "1d", lab.COST),
                         ("R3_binSig_hypePx", sb, "hype_1d", lab.HYPE_COST), ("R4_hypeSig_hypePx", shh, "hype_1d", lab.HYPE_COST)]:
    W = lab.quantile_ls_weights(s, mask, q=0.33)
    W = W[W.index >= pd.Timestamp(A, tz="UTC")].reindex(W.index).fillna(0.0)
    res = lab.backtest(W, tf, cost=cost, start=A, end=B)
    r, m = res["df"]["r"], res["metrics"]
    rp = res["df"]["gross_r"] - res["df"]["cost_r"]
    out[tag] = {"sharpe": m["sharpe"], "cagr": m["cagr"], "maxdd": m["maxdd"], "fund_yr": m["funding_per_year"],
                "cost_yr": m["cost_drag_per_year"], "sh_price_net": sh(rp), "npos": m["avg_n_pos"],
                "sh_2023H2": sh(between(r, "2023-07-01", "2024-01-01")), "sh_2024": sh(between(r, "2024-01-01", "2025-01-01"))}
    print(tag, {k: round(float(v), 3) for k, v in out[tag].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "s5_hype_signal.json"), "w"), indent=1, default=float)
