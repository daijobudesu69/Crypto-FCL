"""Part 2: small-account design grid (IS 2020-03-01 .. 2024-12-31 only, via lab.py).

PRE-REGISTERED GRID (written before running; 16 configs + 1 reference):
  universe  U = lab.hype_universe("1d", top_k=50)  (Binance-ranked point-in-time liquidity, HYPE-listed only)
  signal    F14 = 14-day trailing sum of Binance daily funding (tf "1d"), known at the 00:00 UTC close.
            (Binance signal chosen from Part 1; HYPE funding does not exist before 2023-05.)
  K         names per side in {4, 5, 6, 8}; K_eff = min(K, floor(N/3)), N = names in U with valid F14;
            flat if K_eff < 2 (only affects Mar-Apr 2020). Ties broken by column order.
  gross     {1.0, 1.5}: long K lowest-F14 names, short K highest-F14 names, equal weight, gross/2 per side.
  rebalance every 1 day vs every 3 days (lab.backtest rebalance_every; account_sim gets the held weights).
  cost      lab.HYPE_COST = 0.065%/side, actual Binance funding; returns = Binance closes (tf "1d").
            (HYPE close panel equals Binance before 2023, HYPE funding is 0/NaN before 2023-05 -> a
            2020-24 replay on tf hype_1d would silently drop the carry; Binance used for the full IS.)
  outputs   lab.backtest metrics, sub-periods, Sharpe at 2x cost and extra_lag=1;
            account_sim.simulate(W, "1d", equity0=200, min_order=10, cost=HYPE_COST) on 2020-03..2024-12
            and (robustness) a fresh $200 start on 2023-01-01..2024-12-31;
            diagnostic venue-true replay on tf hype_1d over 2023-06-01..2024-12-31 (HYPE returns+funding).
  reference pure tercile (q=0.33) on the same U, same cost.
"""
from common_v3 import *
import json, time
sys.path.insert(0, ROOT)
import account_sim

A0 = "2020-03-01"
F = lab.load("fund", "1d"); S14 = f14(F)
U = lab.hype_universe("1d", top_k=50)
Ch = lab.load("close", "hype_1d")
Uh_live = U & Ch.notna()                    # for the HYPE-return diagnostic only
DSR_N = 433


def evaluate(W, name, rb, extra=None):
    t0 = time.time()
    res = lab.backtest(W, "1d", cost=lab.HYPE_COST, rebalance_every=rb, start=A0)
    df, m = res["df"], res["metrics"]
    d = {"name": name, "rb": rb, "sharpe": m["sharpe"], "cagr": m["cagr"], "maxdd": m["maxdd"], "vol": m["vol"],
         "t_nw": m["t_nw"], "turn_yr": m["turnover_per_year"], "cost_yr": m["cost_drag_per_year"],
         "fund_yr": m["funding_per_year"], "npos": m["avg_n_pos"], "skew": m["skew"], "kurt": m["kurt"], "days": m["days"]}
    for k, v in subperiods(df["r"]).items():
        d["sh_" + k] = v
    y = lab.yearly(df["r"]); d["yearly_ret"] = {int(k): round(float(v), 3) for k, v in y["ret"].items()}
    d["yearly_sh"] = {int(k): round(float(v), 2) for k, v in y["sharpe"].items()}
    d["sh_2xcost"] = lab.backtest(W, "1d", cost=2 * lab.HYPE_COST, rebalance_every=rb, start=A0)["metrics"]["sharpe"]
    d["sh_lag1"] = lab.backtest(W, "1d", cost=lab.HYPE_COST, rebalance_every=rb, extra_lag=1, start=A0)["metrics"]["sharpe"]
    d["dsr_433"] = lab.deflated_sharpe(m["sharpe"], DSR_N, m["days"], m["skew"], m["kurt"] + 3)
    # venue-true diagnostic (HYPE closes + HYPE funding), overlap only
    rh = lab.backtest(W.where(Uh_live | (W == 0), 0.0), "hype_1d", cost=lab.HYPE_COST, rebalance_every=rb)["df"]["r"]
    rb_ = lab.backtest(W.where(Uh_live | (W == 0), 0.0), "1d", cost=lab.HYPE_COST, rebalance_every=rb)["df"]["r"]
    d["sh_ovl_hypRet"] = sh(window(rh, "2023-06-01", "2025-01-01"))
    d["sh_ovl_binRet"] = sh(window(rb_, "2023-06-01", "2025-01-01"))
    # $200 account
    Wh = hold_every(W, rb)
    for tag, a, b in [("full", A0, None), ("2023-24", "2023-01-01", None)]:
        o, s = account_sim.simulate(Wh, "1d", equity0=200.0, min_order=10.0, cost=lab.HYPE_COST, start=a, end=b)
        d[f"acct_{tag}"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
        d[f"acct_{tag}"]["sharpe_daily"] = round(sh(o["equity"].pct_change().dropna()), 2)
        d[f"acct_{tag}"]["days_flat"] = int((o["n_pos"] == 0).sum())
        d[f"acct_{tag}"]["min_equity"] = round(float(o["equity"].min()), 1)
        if tag == "full":
            ye = o["equity"].groupby(o.index.year).last()
            d["acct_full"]["equity_by_year_end"] = {int(k): round(float(v), 1) for k, v in ye.items()}
    d["secs"] = round(time.time() - t0, 1)
    return d, res


rows = []
for K in [4, 5, 6, 8]:
    for g in [1.0, 1.5]:
        W = topk_weights(S14, U, K, gross=g)
        for rb in [1, 3]:
            d, _ = evaluate(W, f"K{K}_g{g}_rb{rb}", rb)
            d.update(K=K, gross=g)
            rows.append(d)
            a = d["acct_full"]
            print(f"{d['name']:14s} Sh {d['sharpe']:.2f} CAGR {d['cagr']:+.1%} DD {d['maxdd']:.1%} 20-21 {d['sh_2020-21']:.2f} "
                  f"22 {d['sh_2022']:.2f} 23-24 {d['sh_2023-24']:.2f} 2x {d['sh_2xcost']:.2f} lag1 {d['sh_lag1']:.2f} "
                  f"turn {d['turn_yr']:.0f} cost {d['cost_yr']:.1%} fund {d['fund_yr']:+.1%} | ovl hyp {d['sh_ovl_hypRet']:.2f} bin {d['sh_ovl_binRet']:.2f} | "
                  f"$200-> {a['equity_end']:.0f} dd {a['maxdd']:.1%} fees {a['fees_usd']:.0f} skip {a['skipped_below_min']} "
                  f"ord {a['orders']} | 23-24 $200-> {d['acct_2023-24']['equity_end']:.0f} dd {d['acct_2023-24']['maxdd']:.1%} ({d['secs']}s)", flush=True)
# reference: pure tercile on the HYPE universe
W = lab.quantile_ls_weights(-S14, U, q=0.33)
for g in [1.0, 1.5]:
    d, _ = evaluate(W * g, f"tercile_g{g}_rb1", 1); d.update(K="tercile", gross=g); rows.append(d)
    a = d["acct_full"]
    print(f"{d['name']:14s} Sh {d['sharpe']:.2f} CAGR {d['cagr']:+.1%} DD {d['maxdd']:.1%} 20-21 {d['sh_2020-21']:.2f} 22 {d['sh_2022']:.2f} "
          f"23-24 {d['sh_2023-24']:.2f} 2x {d['sh_2xcost']:.2f} lag1 {d['sh_lag1']:.2f} npos {d['npos']:.1f} | ovl hyp {d['sh_ovl_hypRet']:.2f} | "
          f"$200-> {a['equity_end']:.0f} dd {a['maxdd']:.1%} fees {a['fees_usd']:.0f} skip {a['skipped_below_min']} ord {a['orders']}", flush=True)
json.dump(rows, open("p2_grid.json", "w"), indent=1, default=float)
