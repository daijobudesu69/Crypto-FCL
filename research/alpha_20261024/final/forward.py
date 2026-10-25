"""ONE-TIME forward test 2025-01-01 -> last data day, frozen specs only (SPEC_CHOP30_v1.md).
Writes final/forward_results.json, final/forward_*.csv and the charts in ../report_assets/."""
import os, sys, json
os.environ["ALPHA_FORWARD"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import lab, strategies as S, fcl

FWD = "2025-01-01"
OUT = {}
EQ = {}


def yearly(d):
    return {int(k): round(float(v), 4) for k, v in lab.yearly(d["r"])["ret"].items()}


def keep(name, d, s):
    s = dict(s); s["yearly"] = yearly(d)
    m = (1 + d["r"]).groupby(d.index.tz_localize(None).to_period("M")).prod() - 1
    s["monthly"] = {str(k): round(float(v), 4) for k, v in m.items()}
    q = (1 + d["r"]).groupby(d.index.tz_localize(None).to_period("Q")).prod() - 1
    s["quarterly"] = {str(k): round(float(v), 4) for k, v in q.items()}
    OUT[name] = s; EQ[name] = d["equity"]
    print(f"{name:34s} ${s['equity_start']:.0f}->${s['equity_end']:8.2f}  ret {s['total_return']:+.1%}  SR {s['sharpe']:.2f}  "
          f"DD {s['maxdd']:.1%}  npos {s.get('avg_n_pos', float('nan')):.1f}  fees ${s.get('fees', 0):.1f}  fund ${s.get('funding', 0):+.1f}  yrs {s['yearly']}", flush=True)


# ------------------------------------------------ CHOP-30 (primary) and DISP7
for venue in ["hype", "binance"]:
    P, U = S.venue_panels(venue)
    for name, mk in [("CHOP30", S.chop_targets), ("DISP7", S.disp_targets)]:
        tg, rt = mk(P, U)
        d, s = S.replay(P, tg, rt, FWD)
        keep(f"{name}_{venue}_acct200", d, s)
        d, s = S.replay(P, tg, rt, FWD, ideal=True)
        keep(f"{name}_{venue}_ideal", d, s)
    if venue == "hype":
        # CHOP-30 leg decomposition on the ideal book (HYPE prices + funding)
        tg, rt = S.chop_targets(P, U)
        ix = P["close"].index
        t0 = ix.get_indexer([pd.Timestamp(FWD, tz="UTC")])[0]
        W = pd.DataFrame(0.0, index=ix, columns=P["close"].columns)
        for t in range(t0 - 1, len(ix)):
            W.iloc[t] = tg(t, None, None)[0]
        for leg, Wl in [("long_leg", W.clip(lower=0)), ("short_leg", W.clip(upper=0))]:
            res = lab.backtest(Wl, "hype_1d", cost=lab.HYPE_COST, start=FWD)
            OUT[f"CHOP30_hype_ideal_{leg}"] = {k: res["metrics"][k] for k in ["cagr", "sharpe", "maxdd", "funding_per_year", "cost_drag_per_year"]}
            print(leg, OUT[f"CHOP30_hype_ideal_{leg}"])
        ew = U.astype(float).div(U.sum(axis=1), axis=0).fillna(0.0)
        res = lab.backtest(ew, "hype_1d", cost=lab.HYPE_COST, start=FWD)
        dd = res["df"]; eq = 200 * (1 + dd["r"]).cumprod()
        keep("BENCH_EW_hype_universe_long", dd.assign(equity=eq), {"equity_start": 200.0, "equity_end": float(eq.iloc[-1]),
             "total_return": float(eq.iloc[-1] / 200 - 1), **{k: res["metrics"][k] for k in ["sharpe", "maxdd", "cagr"]}, "avg_n_pos": res["metrics"]["avg_n_pos"]})
        c = P["close"]["BTCUSDT"]; c = c[c.index >= pd.Timestamp(FWD, tz="UTC") - pd.Timedelta("1D")]
        r = c.pct_change().dropna(); eq = 200 * (1 + r).cumprod(); m = lab.metrics(r)
        keep("BENCH_BTC_buy_hold", pd.DataFrame({"r": r, "equity": eq}), {"equity_start": 200.0, "equity_end": float(eq.iloc[-1]),
             "total_return": float(eq.iloc[-1] / 200 - 1), "sharpe": m["sharpe"], "maxdd": m["maxdd"], "cagr": m["cagr"]})

# ------------------------------------------------ FCL v1.0 (Binance funding signal; not deployable for this user)
for venue in ["hype", "binance"]:
    d, s, _ = fcl.run(FWD, venue=venue, mode="capacity")
    keep(f"FCL_{venue}_acct200", d, s)

# ------------------------------------------------ IS curve of the primary for the chart (same code, ends 2024-12-31)
P, U = S.venue_panels("binance")
tg, rt = S.chop_targets(P, U)
d_is, s_is = S.replay(P, tg, rt, "2020-03-01", end="2024-12-31")
EQ["CHOP30_IS_binance_acct200"] = d_is["equity"]

json.dump(OUT, open(os.path.join(HERE, "forward_results.json"), "w"), indent=1, default=str)
pd.DataFrame(EQ).to_csv(os.path.join(HERE, "forward_equity.csv"))
print("saved")
