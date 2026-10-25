"""FAMILY B robustness for the best grid cell B_pkcc30_FLIP_R1
   (long LOW Parkinson/close-to-close vol ratio over 30d, short HIGH; $-neutral quintiles, daily rebalance).

PRE-REGISTERED ROBUSTNESS SET (written before running; every line is counted as a trial for DSR):
 R1 window plateau  pkcc W in {10, 20, 45, 60} FLIP R1                                    4
 R2 holding         W=30, rebalance every {2, 3, 5} days                                  3
 R3 universe        hype top_k 30, hype top_k 75, Binance (non-HYPE-restricted) top 50    3
 R4 book size       16-pos (8 long / 8 short), 12-pos (6/6)                               2
 R5 neutralization  beta-neutral legs (60d beta); vol-neutral: XS residual of log(pkcc30) on
                    log(rv60) and log(30d Parkinson vol)                                  2
 R6 recency         pkcc over last 5 days only; pkcc30 lagged 5 days (days t-34..t-5)     2
 R7 variant         efficiency ratio: 30d mean of |log(C/O)| / log(H/L), long high        1
 stress (not selection): extra_lag 2 and 3; cost 3x                                       -
 => 17 trials -> family B total 32 + 17 = 49.
Diagnostics (no trials): spanning regression on xsmom14 + btc + 3d XS continuation + low-vol (rv60) +
 4h-skew30 L/S + BAB-BN, per-coin PnL concentration, per-year leg decomposition, XS correlation of the
 signal with vol level / 30d return / liquidity, degenerate-range check.
"""
import os
import numpy as np, pandas as pd
from common import (HERE, universe, ls_weights, bn_weights, evaluate, fmt_row, ols_nw, controls, lab, COST,
                    returns, sharpe, sub_sharpes)
from signals import vol_structure, skew4h, ohlc, beta_family
from famA_diag import xs_resid

N_TRIALS_B = 49

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    U = universe(50)
    o_, h, l, c = ohlc()
    R = returns()
    pc = c.shift(1)
    lrange = np.log((h / l).clip(lower=1.0 + 1e-5))
    pk2 = lrange ** 2 / (4 * np.log(2))
    cc2 = np.log(c / pc) ** 2

    def pkcc(W, shift=0):
        mp = max(3, int(0.8 * W))
        x = np.sqrt(pk2.rolling(W, min_periods=mp).mean() / cc2.rolling(W, min_periods=mp).mean())
        return x.shift(shift)

    rows = []

    def run(score, name, mask=U, reb=1, n=None, W=None):
        Wt = W if W is not None else ls_weights(score, mask, q=0.2, n=n)
        o, res = evaluate(Wt, name, rebalance=reb, mask=mask, n_trials=N_TRIALS_B)
        rows.append(o)
        print(name.ljust(28), fmt_row(o), f"c2x={o['sharpe_cost2x']:.2f} lag1={o['sharpe_lag1']:.2f} "
              f"aT={o['alpha_t']:.2f} cXS={o['corr_xsmom']:.2f} L={o.get('long_leg_xs_sharpe', np.nan):.2f} "
              f"S={o.get('short_leg_xs_sharpe', np.nan):.2f} mineq={o['min_eq_med']:.0f}/{o['min_eq_p90']:.0f}",
              o["yearly"], flush=True)
        return o, res

    base_sig = -pkcc(30)
    ob, rb = run(base_sig, "BASE_pkcc30_FLIP_R1")
    print("--- R1 window plateau")
    for W in (10, 20, 45, 60):
        run(-pkcc(W), f"R1_pkcc{W}_FLIP_R1")
    print("--- R2 holding")
    for k in (2, 3, 5):
        run(base_sig, f"R2_pkcc30_FLIP_R{k}", reb=k)
    print("--- stress lags / 3x cost (base)")
    Wb = ls_weights(base_sig, U, q=0.2)
    for lg in (2, 3):
        m = lab.backtest(Wb, "1d", cost=COST, extra_lag=lg)["metrics"]
        print(f"  extra_lag={lg}: Sharpe {m['sharpe']:.2f}")
    m = lab.backtest(Wb, "1d", cost=3 * COST)["metrics"]; print(f"  cost 3x: Sharpe {m['sharpe']:.2f}")
    print("--- R3 universe")
    for k in (30, 75):
        Uk = universe(k)
        run(base_sig, f"R3_hype_top{k}", mask=Uk)
    Ub = lab.universe("1d", min_hist_days=60, top_k=50)
    run(base_sig, "R3_binance_top50", mask=Ub)
    print("--- R4 book size")
    o16, r16 = run(base_sig, "R4_16pos_8x8", n=8)
    run(base_sig, "R4_12pos_6x6", n=6)
    print("--- R5 neutralization")
    S = beta_family(60)
    run(None, "R5_betaneutral", W=bn_weights(base_sig, U, S["beta"]))
    pkvol = np.sqrt(pk2.rolling(30, min_periods=24).mean())
    VS = vol_structure()
    rv60 = VS["rv60"]
    # two-regressor XS residual: do it sequentially (orthogonalize on log rv60, then on log pk30)
    lp = np.log(pkcc(30))
    res1 = xs_resid(lp, np.log(rv60), U)
    res2 = xs_resid(res1, np.log(pkvol), U)
    run(-res2, "R5_volneutral")
    print("--- R6 recency")
    run(-pkcc(5), "R6_pkcc5_only")
    run(-pkcc(30, shift=5), "R6_pkcc30_lag5")
    print("--- R7 efficiency ratio")
    er = (np.log(c / o_).abs() / lrange).rolling(30, min_periods=24).mean()
    run(er, "R7_effratio30_long_high")
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "famB_robust.csv"), index=False)

    # ---------------- diagnostics on BASE
    print("\n=== diagnostics on BASE_pkcc30_FLIP_R1")
    r = rb["df"]["r"]
    Ctl = controls()
    # factor books (same universe, R1, HYPE cost)
    def book(score, reb=1, Wt=None):
        Wt = Wt if Wt is not None else ls_weights(score, U, q=0.2)
        return lab.backtest(Wt, "1d", cost=COST, rebalance_every=reb)["df"]["r"]
    f = pd.DataFrame({
        "cont3d": book(c / c.shift(3) - 1),
        "lowvol": book(-rv60),
        "skew4h30": book(skew4h(30).reindex(c.index)),
        "bab_bn": book(None, reb=7, Wt=bn_weights(-S["beta"], U, S["beta"])),
    })
    j = pd.concat([r.rename("s"), Ctl, f], axis=1, sort=True).dropna()
    print("corr matrix:\n", j.corr().round(2).to_string())
    for cols in (["xsmom14", "btc"], ["xsmom14", "btc", "cont3d", "lowvol", "skew4h30", "bab_bn"]):
        b, t, n = ols_nw(j["s"], j[cols])
        print(f"  span on {cols}: alpha/yr {b['const'] * 365:+.1%} t {t['const']:.2f} | " +
              " ".join(f"{k} {b[k]:+.2f}(t{t[k]:.1f})" for k in cols))
    # XS correlation of signal with characteristics
    sig = pkcc(30).where(U)
    def xsc(x):
        return sig.rank(axis=1).corrwith(x.where(U).rank(axis=1), axis=1).mean()
    qv = lab.load("qv", "1d")
    print(f"  XS Spearman pkcc30 vs: rv60 {xsc(rv60):.2f}  ret30 {xsc(c / c.shift(30) - 1):.2f}  "
          f"|ret30| {xsc((c / c.shift(30) - 1).abs()):.2f}  qv30 {xsc(qv.rolling(30).median()):.2f}  "
          f"beta60 {xsc(S['beta']):.2f}  skew4h30 {xsc(skew4h(30).reindex(c.index)):.2f}")
    print(f"  signal autocorr (XS rank, 1d / 5d / 20d): "
          f"{sig.rank(axis=1).corrwith(sig.rank(axis=1).shift(1), axis=1).mean():.2f} / "
          f"{sig.rank(axis=1).corrwith(sig.rank(axis=1).shift(5), axis=1).mean():.2f} / "
          f"{sig.rank(axis=1).corrwith(sig.rank(axis=1).shift(20), axis=1).mean():.2f}")
    # degenerate ranges
    flat = ((h / l) < 1.0005) & U
    print(f"  universe-days with H/L < 1.0005 (degenerate range): {int(flat.sum().sum())}")
    # per-coin PnL concentration & legs per year
    Wt = Wb
    pnl = (Wt * R.shift(-1)).sum()
    tot = pnl.sum(); srt = pnl.sort_values(ascending=False)
    print(f"  price PnL total {tot:+.2f}; top5 {srt.head(5).round(2).to_dict()} share {srt.head(5).sum() / tot:.0%}; "
          f"top10 share {srt.head(10).sum() / tot:.0%}; bottom5 {srt.tail(5).round(2).to_dict()}")
    pl = (Wt.clip(lower=0) * R.shift(-1)).sum(axis=1); ps = (Wt.clip(upper=0) * R.shift(-1)).sum(axis=1)
    ew = (lab.xs_rank(c, U).notna().astype(float))
    ew = ew.div(ew.sum(axis=1), axis=0); rew = (ew * R.shift(-1)).sum(axis=1)
    yl = pd.DataFrame({"long_px": pl, "short_px": ps, "long_minus_halfEW": pl - 0.5 * rew,
                       "halfEW_minus_shortabs": -ps - 0.5 * rew})
    print("  yearly sums (price pnl, before cost):\n", yl.groupby(yl.index.year).sum().round(3).T.to_string())
    print("  16-pos sub-periods:", {k: round(v, 2) for k, v in sub_sharpes(r16["df"]["r"]).items()})
