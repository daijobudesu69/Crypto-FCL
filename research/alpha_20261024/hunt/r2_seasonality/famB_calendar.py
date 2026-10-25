"""Family B - CALENDAR EFFECTS AT LOW TURNOVER (long/flat, few trades).

Hypotheses / rationale
  TOM   turn-of-the-month: month-start inflows (salary-day DCA buys, monthly exchange/DCA plans,
        new-month allocations) and month-end window dressing / rebalancing concentrate demand around
        the month boundary (Ariel 1987; Lakonishok-Smidt 1988; McConnell-Xu 2008 in equities).
        Other side: liquidity providers / anyone who must sell into the scheduled flow.
  LAST3 month-end rebalancing flows alone; FIRST3 month-start inflows alone.
  MEREV month-end CONDITIONAL rebalancing: fixed-weight allocators sell what went up this month and
        buy what went down in the last days of the month (pension-style rebalancing, Harvey et al.)
        -> position over the last 3 days = -sign(month-to-date return known 4 days before month end).
  MON   'Monday open vs Sunday close': TradFi re-opens on Monday (CME gap, ETF/desk flows) ->
        hold the Monday UTC bar (Sunday close -> Monday close). NOTE round 1 (h5) already found
        weekday timing loses to buy&hold; included only because it is assigned, kept to 4 configs.

PRE-REGISTERED GRID (24 configs = 6 windows x 4 assets)
  windows: TOM23 {last 2 + first 3 days}, TOM33 {last 3 + first 3}, FIRST3, LAST3, MEREV, MON
  assets : BTC, ETH, EW = equal-weight lab.hype_universe("1d", top_k=50) basket,
           SPREAD = 0.5 EW long / 0.5 BTC short ($-neutral long-alt / short-BTC)
  position: long/flat (MEREV: +-1 by contrarian sign, long/flat split shown as diagnostic)
  ESTIMATE on 2020-01..2022-12, VALIDATE on 2023-01..2024-12 (both inside IS).
  Pre-registered pass rule: EST diff t >= 1.5 AND VAL same sign with t >= 1.0 (diff = mean daily
  return in window minus outside window, Welch t; MEREV: t of mean signed 3-day return per month).
  Only passing configs get the full candidate treatment; everything is still backtested at
  HYPE cost (0.065%/side) + funding, with 2x cost and extra_lag=1.
  Robustness for passers: 16-name basket (hype top_k=16), top_k 30 / 75.
Signal inputs: calendar + close (MTD return) only.
"""
import sys, os
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm
from common import lab

N_TRIALS_B = 24 + 6
C = lab.load("close", "1d")
R = C.pct_change(fill_method=None)
IDX = C.index
BTC, ETH = "BTCUSDT", "ETHUSDT"


def held_day_features(idx):
    """features of the day D held when W[t] is set (D = idx[t] + 1 day)."""
    D = idx + pd.Timedelta("1D")
    dom = D.day
    dim = D.days_in_month
    dte = dim - dom + 1                       # 1 = last day of month
    return pd.DataFrame({"dom": dom, "dte": dte, "wd": D.weekday}, index=idx)


F = held_day_features(IDX)
WIN = {
    "TOM23": (F["dte"] <= 2) | (F["dom"] <= 3),
    "TOM33": (F["dte"] <= 3) | (F["dom"] <= 3),
    "FIRST3": F["dom"] <= 3,
    "LAST3": F["dte"] <= 3,
    "MON": F["wd"] == 0,
}


def ew_base(top_k=50):
    U = cm.hype_U(top_k).reindex(index=IDX, columns=C.columns).fillna(False)
    return U.astype(float).div(U.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def base_weights(asset, EWb):
    W = pd.DataFrame(0.0, index=IDX, columns=C.columns)
    if asset == "BTC":
        W[BTC] = 1.0
    elif asset == "ETH":
        W[ETH] = 1.0
    elif asset == "EW":
        W = EWb.copy()
    elif asset == "SPREAD":
        W = 0.5 * EWb.copy()
        W[BTC] = W[BTC] - 0.5
    return W


def asset_daily_ret(asset, EWb):
    """descriptive daily return of the asset for held day (aligned to row t = return of bar t+1)."""
    Rn = R.shift(-1)
    if asset == "BTC":
        return Rn[BTC]
    if asset == "ETH":
        return Rn[ETH]
    ew = (EWb * Rn.fillna(0.0)).sum(axis=1).where(EWb.sum(axis=1) > 0)
    if asset == "EW":
        return ew
    return 0.5 * ew - 0.5 * Rn[BTC]


def mtd_signal(asset, EWb):
    """-sign(MTD return) known at close of the 4th-last day, held over the last 3 days."""
    r = asset_daily_ret(asset, EWb).shift(1)     # r at row t = return of bar t (known at close t)
    month = IDX.to_period("M") if IDX.tz is None else IDX.tz_localize(None).to_period("M")
    lr = np.log1p(r.fillna(0.0))
    mtd = np.expm1(lr.groupby(month).cumsum())
    entry = F["dte"] == 3
    sig = pd.Series(np.nan, index=IDX)
    sig[entry] = -np.sign(mtd[entry])
    sig = sig.ffill(limit=2).where(F["dte"] <= 3)
    return sig.fillna(0.0)


def period_mask(idx, a, b):
    return (idx >= cm.ts(a)) & (idx < cm.ts(b))


def stats_inout(x, win, a, b):
    m = period_mask(x.index, a, b) & x.notna()
    xi, xo = x[m & win], x[m & ~win]
    t = stats.ttest_ind(xi, xo, equal_var=False).statistic
    return float(xi.mean()), float(xo.mean()), float(t), int(len(xi))


def stats_merev(x, sig, a, b):
    """per-month signed 3-day return."""
    m = period_mask(x.index, a, b) & (sig != 0) & x.notna()
    s = (x * sig)[m]
    month = s.index.tz_localize(None).to_period("M")
    pm = np.expm1(np.log1p(s).groupby(month).sum())
    t = pm.mean() / pm.std() * np.sqrt(len(pm)) if pm.std() > 0 else np.nan
    return float(pm.mean()), float(t), int(len(pm)), float((pm > 0).mean())


def alpha_vs_asset(d, asset, EWb):
    """HAC alpha of strategy daily returns vs the asset's own buy&hold daily return."""
    ar = asset_daily_ret(asset, EWb)
    ar.index = ar.index + pd.Timedelta("1D")       # align to backtest df index (bar end)
    j = pd.concat([d.rename("s"), ar.rename("a")], axis=1, join="inner").dropna()
    b, t, n = cm.hac_ols(j["s"].values, j[["a"]].values)
    return float(b[0] * 365), float(t[0]), float(b[1])


def build_W(win_name, asset, EWb):
    base = base_weights(asset, EWb)
    if win_name == "MEREV":
        sig = mtd_signal(asset, EWb)
    else:
        sig = WIN[win_name].astype(float)
    return base.mul(sig, axis=0), sig


if __name__ == "__main__":
    EWb = ew_base(50)
    rows = []
    # buy & hold references
    for asset in ["BTC", "ETH", "EW", "SPREAD"]:
        Wb = base_weights(asset, EWb)
        rb = lab.backtest(Wb, "1d", cost=cm.COST)
        d = cm.daily(rb)
        print(f"B&H {asset:6s} Sh {rb['metrics']['sharpe']:+.2f} CAGR {rb['metrics']['cagr']:+.1%} | "
              f"{cm.fmt_sub(cm.sub_sharpes(d))}")
    print()
    for wn in ["TOM23", "TOM33", "FIRST3", "LAST3", "MEREV", "MON"]:
        for asset in ["BTC", "ETH", "EW", "SPREAD"]:
            W, sig = build_W(wn, asset, EWb)
            x = asset_daily_ret(asset, EWb)
            if wn == "MEREV":
                e_m, e_t, e_n, e_hit = stats_merev(x, sig, *cm.EST)
                v_m, v_t, v_n, v_hit = stats_merev(x, sig, *cm.VAL)
                st = {"est_mean": e_m, "est_t": e_t, "val_mean": v_m, "val_t": v_t,
                      "est_hit": e_hit, "val_hit": v_hit}
            else:
                win = WIN[wn]
                ei, eo, e_t, _ = stats_inout(x, win, *cm.EST)
                vi, vo, v_t, _ = stats_inout(x, win, *cm.VAL)
                st = {"est_in": ei, "est_out": eo, "est_t": e_t, "val_in": vi, "val_out": vo, "val_t": v_t}
            passed = (st["est_t"] >= 1.5) and (st["val_t"] >= 1.0)
            name = f"B_{wn}_{asset}"
            o = cm.evaluate(W, "1d", name, "B_" + wn, N_TRIALS_B, extra=dict(st, passed=passed,
                                                                            exposure=float((sig != 0).mean())))
            a_ann, a_t, a_b = alpha_vs_asset(cm.daily(lab.backtest(W, "1d", cost=cm.COST)), asset, EWb)
            o.update({"alpha_vs_asset_ann": a_ann, "alpha_vs_asset_t": a_t, "beta_vs_asset": a_b})
            rows.append(o)
            if wn == "MEREV":
                tag = f"EST t {e_t:+.2f} (hit {e_hit:.0%})  VAL t {v_t:+.2f} (hit {v_hit:.0%})"
            else:
                tag = (f"EST in {ei*1e4:+.0f}bp out {eo*1e4:+.0f}bp t {e_t:+.2f} | "
                       f"VAL in {vi*1e4:+.0f}bp out {vo*1e4:+.0f}bp t {v_t:+.2f}")
            print(f"{name:18s} {tag} | PASS={passed}")
            print("   " + cm.short_line(o) + f" | a_vs_asset t {a_t:+.2f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(cm.HERE, "famB_grid.csv"), index=False)
    print("done")
