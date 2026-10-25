"""v4_risk part 1b: re-run the pre-registered stop overlays O1/O2 with a robust trigger (1h closes, fill at
the first 1h close beyond the level + 0.5% slip) because the 1d high panel has bad prints
(ETH/ATOM 2020-03-13 high = 2.4x max(open, close)). Also: robustness of O3 to its decile choice
(2 neighbours, reported as robustness only - NOT new selectable variants), and base metrics after the
early-2020 tiny-universe period (universe < 10 names in Q1 2020)."""
from common4 import *
import json

C = lab.load("close", "1d"); F = lab.load("fund", "1d"); R = C.pct_change(fill_method=None)
U = lab.universe("1d", top_k=50); UH = lab.hype_universe("1d", top_k=50)
cube = hourly_cube(C)
print("ETH 1h max on 2020-03-13:", np.nanmax(cube[C.index.get_loc(pd.Timestamp("2020-03-13", tz="UTC")), :, C.columns.get_loc("ETHUSDT")]))
out = {}

def sub(rr):
    o = {}
    for nm, a, b in [("2020-21", "2020-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"), ("2023-24", "2023-01-01", "2025-01-01")]:
        o[nm] = round(sh(rr[(rr.index >= pd.Timestamp(a, tz="UTC")) & (rr.index < pd.Timestamp(b, tz="UTC"))]), 3)
    return o

def stats(d, d2=None):
    rr = d["r"]; e = (1 + rr).cumprod()
    o = {"sharpe": round(sh(rr), 3), "cagr": round(float(e.iloc[-1] ** (365 / len(rr)) - 1), 4), "maxdd": round(float((e / e.cummax() - 1).min()), 4),
         "worst_day": round(float(rr.min()), 4), "worst_7d": round(float(np.expm1(np.log1p(rr).rolling(7).sum()).min()), 4),
         "skew": round(float(rr.skew()), 2), "nstops": d.attrs.get("nstops", 0), "sub": sub(rr), "npos": round(float(d["n_pos"].mean()), 1)}
    if d2 is not None:
        o["sharpe_2x"] = round(sh(d2["r"]), 3)
    return o

def leg_norm(Wx, side_gross=0.5):
    L = Wx.clip(lower=0); S = Wx.clip(upper=0)
    L = L.div(L.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * side_gross
    S = S.div(S.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * side_gross
    return L + S

mom14 = C / C.shift(14) - 1
for uname, Ub in [("top50", U), ("hype_top50", UH)]:
    Wb, _ = candidate_weights(Ub, F)
    for name, stp in [("base", None), ("O1_stop25_1h", 0.25), ("O2_stop40_1h", 0.40)]:
        d = sim(Wb, stop=stp, cube=cube); d2 = sim(Wb, stop=stp, cube=cube, cost=2 * lab.COST)
        out[f"{uname}:{name}"] = stats(d, d2); print(uname, name, out[f"{uname}:{name}"])
    rk = mom14.where(Ub).rank(axis=1, pct=True)
    for thr in (0.8, 0.9, 0.95):
        W3 = Wb.copy(); W3[(Wb < 0) & (rk > thr)] = 0.0; W3 = leg_norm(W3)
        d = sim(W3); d2 = sim(W3, cost=2 * lab.COST)
        tag = f"O3_skipmom_top{int(round((1 - thr) * 100))}pct" + ("" if thr == 0.9 else "_robust")
        out[f"{uname}:{tag}"] = stats(d, d2); print(uname, tag, out[f"{uname}:{tag}"])
        # lag-1 for the pre-registered O3
        if thr == 0.9:
            rl = lab.backtest(W3, "1d", extra_lag=1)["metrics"]["sharpe"]
            out[f"{uname}:{tag}"]["sharpe_lag1"] = round(rl, 3)
            # how much did the skipped shorts actually move the next day? (the overlay's mechanism)
            skipped = (Wb < 0) & (rk > thr)
            Rn = R.shift(-1)
            kept = (Wb < 0) & ~(rk > thr)
            out[f"{uname}:{tag}"]["mean_next_day_ret_skipped_shorts"] = round(float(Rn.where(skipped).stack().mean()), 5)
            out[f"{uname}:{tag}"]["mean_next_day_ret_kept_shorts"] = round(float(Rn.where(kept).stack().mean()), 5)
            out[f"{uname}:{tag}"]["share_short_names_skipped"] = round(float(skipped.sum().sum() / (Wb < 0).sum().sum()), 3)
            print("  skipped vs kept next-day ret", out[f"{uname}:{tag}"]["mean_next_day_ret_skipped_shorts"], out[f"{uname}:{tag}"]["mean_next_day_ret_kept_shorts"])
            # yearly
            y = lab.yearly(d["r"]); out[f"{uname}:{tag}"]["yearly_ret"] = {int(k): round(float(v), 3) for k, v in y["ret"].items()}

# base after tiny-universe period
W, _ = candidate_weights(U, F)
for st in ("2020-07-01", "2021-01-01", "2022-01-01"):
    res = lab.backtest(W, "1d", start=st); m = res["metrics"]
    out[f"base_from_{st}"] = {"sharpe": round(m["sharpe"], 3), "cagr": round(m["cagr"], 4), "maxdd": round(m["maxdd"], 4)}
    print("base from", st, out[f"base_from_{st}"])
json.dump(out, open(os.path.join(HERE, "t1b_stops.json"), "w"), indent=1, default=str)
