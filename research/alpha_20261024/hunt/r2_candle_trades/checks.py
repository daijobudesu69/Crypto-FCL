"""Robustness / stress for the best attempt(s) of each family (NOT selection - specs fixed from the grids).

For each spec: base (hype top-50 quintile), 2x cost, extra_lag=1, long/short legs (raw and excess vs EW
universe), alpha t vs xsmom14+BTC, 16-position (8L/8S) version, top_k 30 and 75, neighbour plateau
(same feature+direction at the other grid windows), deflated Sharpe (n_trials = all configs this hunter ran).
Extra configs counted here: ATSlvl window 10/60 plateau neighbours (+2) -> total 78 configs.
Also: venue-translation check for family A (XS rank correlation of the candle-shape features computed from
Binance vs HYPE daily candles, 2024 only, where both exist) and the mechanical price link of ATSchg.
"""
import os, json
import numpy as np, pandas as pd
import r2common as rc
import lab
import famA_grid as A
import famB_grid as B

LOG = os.path.join(rc.HERE, "checks.csv")
N_TRIALS = 42 + 34 + 2


def spec_scores():
    SA = A.build_scores()
    SB = B.build_scores()
    p = rc.panels()
    ats = p["Q"] / p["T"]
    for w in (10, 60):
        SB[f"ATSlvl_w{w}"] = np.log(ats.rolling(w, min_periods=int(0.8 * w)).mean())
    return SA, SB


def resid_clv10(SA, Um):
    return rc.xs_resid(SA["CLV10"], [rc.mom(10), rc.mom(14)], Um)


def main():
    if os.path.exists(LOG):
        os.remove(LOG)
    SA, SB = spec_scores()
    U50, U30, U75 = rc.U(50), rc.U(30), rc.U(75)
    grids = pd.concat([pd.read_csv(os.path.join(rc.HERE, "famA_grid.csv")),
                       pd.read_csv(os.path.join(rc.HERE, "famB_grid.csv"))])
    gsh = dict(zip(grids["name"], grids["sharpe"]))
    specs = [
        # name, family, score builder(U), neighbours (grid names)
        ("CLV20+", "A", lambda U: SA["CLV20"], ["CLV5+", "CLV10+"]),
        ("LW5-", "A", lambda U: -SA["LW5"], ["LW10-", "LW20-"]),
        ("RPOS20+", "A", lambda U: SA["RPOS20"], ["RPOS5+", "RPOS10+"]),
        ("CLV10res+", "A", lambda U: rc.xs_resid(SA["CLV10"], [rc.mom(10), rc.mom(14)], U), ["CLV5res+", "CLV20res+"]),
        ("ATSchg7+", "B", lambda U: SB["ATSchg7"], ["ATSchg14+", "ATSchg30+"]),
        ("ATSlvl-", "B", lambda U: -SB["ATSlvl"], ["ATSlvl_w10-", "ATSlvl_w60-"]),
        ("FOMO14res-", "B", lambda U: -rc.xs_resid(SB["FOMO14"], [rc.mom(14), rc.mom(14)], U), ["FOMO7res-", "FOMO30res-"]),
    ]
    # plateau neighbours for ATSlvl (extra configs, counted)
    for w in (10, 60):
        row, _ = rc.evaluate(rc.qls(-SB[f"ATSlvl_w{w}"], U50), f"ATSlvl_w{w}-", "B_plateau", LOG)
        gsh[row["name"]] = row["sharpe"]
        print(rc.fmt(row))
    out = {}
    for nm, fam, fn, nb in specs:
        s50 = fn(U50)
        W = rc.qls(s50, U50)
        row, res = rc.evaluate(W, nm, fam, LOG, Um=U50, full=True)
        m = res["metrics"]
        row["deflated_sharpe_prob"] = lab.deflated_sharpe(m["sharpe"], N_TRIALS, m["days"], m["skew"], m["kurt"] + 3)
        row["neighbour_median_sharpe"] = float(np.median([gsh[k] for k in nb]))
        r16, res16 = rc.evaluate(rc.topn_weights(s50, U50, n=8), nm + "_8L8S", fam, LOG, Um=U50, full=True)
        r30, _ = rc.evaluate(rc.qls(fn(U30), U30), nm + "_top30", fam, LOG)
        r75, _ = rc.evaluate(rc.qls(fn(U75), U75), nm + "_top75", fam, LOG)
        row["sharpe_8L8S"] = r16["sharpe"]
        row["sharpe_8L8S_lag1"] = r16["sharpe_lag1"]
        row["sharpe_8L8S_cost2x"] = r16["sharpe_cost2x"]
        row["min_equity_8L8S"] = r16["min_equity_usd"]
        row["sharpe_top30"] = r30["sharpe"]
        row["sharpe_top75"] = r75["sharpe"]
        row["subp_8L8S"] = f"{r16['sh_2020-21']:.2f}/{r16['sh_2022']:.2f}/{r16['sh_2023-24']:.2f}"
        out[nm] = row
        print(rc.fmt(row))
        print(f"{'':34s} DSR {row['deflated_sharpe_prob']:.3f} neighbours {row['neighbour_median_sharpe']:.2f} | "
              f"8L8S {r16['sharpe']:.2f} (lag1 {r16['sharpe_lag1']:.2f}, 2x {r16['sharpe_cost2x']:.2f}, subp {row['subp_8L8S']}, "
              f"minEq ${r16['min_equity_usd']:.0f}) | top30 {r30['sharpe']:.2f} top75 {r75['sharpe']:.2f} | "
              f"alpha_ann {row['alpha_ann_vs_mom_btc']:+.1%} beta_mom {row['beta_xsmom']:.2f}", flush=True)
    with open(os.path.join(rc.HERE, "checks.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    # ---- venue translation check, family A (2024 only: HYPE candles with real activity)
    hp = {k: lab.load(k, "hype_1d") for k in ("open", "high", "low", "close")}
    rng = hp["high"] - hp["low"]
    rng = rng.where(rng > 0)
    hclv = ((hp["close"] - hp["low"]) - (hp["high"] - hp["close"])) / rng
    hlw = (np.minimum(hp["open"], hp["close"]) - hp["low"]) / rng
    f = A.features()
    pairs = {"CLV20": (f["CLV"].rolling(20, min_periods=16).mean(), hclv.rolling(20, min_periods=16).mean()),
             "LW5": (f["LW"].rolling(5, min_periods=4).mean(), hlw.rolling(5, min_periods=4).mean()),
             "CLV1": (f["CLV"], hclv)}
    sel = (U50.index >= pd.Timestamp("2024-01-01", tz="UTC"))
    for k, (b, h) in pairs.items():
        m = U50 & b.notna() & h.notna()
        rb = lab.xs_rank(b, m)[sel]; rh = lab.xs_rank(h, m)[sel]
        cors = [rb.iloc[i].corr(rh.iloc[i]) for i in range(len(rb)) if m[sel].iloc[i].sum() >= 10]
        print(f"venue rank corr Binance vs HYPE 2024, {k}: median {np.nanmedian(cors):.2f}  p10 {np.nanpercentile(cors, 10):.2f}  days {len(cors)}")
    # ---- mechanical price link of ATSchg (USD trade size scales with price)
    for h in (7, 14):
        m = U50 & SB[f"ATSchg{h}"].notna() & rc.mom(h).notna()
        ra = lab.xs_rank(SB[f"ATSchg{h}"], m); rr = lab.xs_rank(rc.mom(h), m)
        c = [ra.iloc[i].corr(rr.iloc[i]) for i in range(len(ra)) if m.iloc[i].sum() >= 10]
        print(f"XS rank corr ATSchg{h} vs ret_{h}: median {np.nanmedian(c):.2f}")


if __name__ == "__main__":
    main()
