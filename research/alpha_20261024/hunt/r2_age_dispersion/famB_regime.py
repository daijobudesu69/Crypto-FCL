"""FAMILY B - MARKET-LEVEL REGIME / DISPERSION TIMING (round 2, HYPE candles only).

Hypotheses (directions fixed BEFORE running):
 - "Idiosyncratic / broad-participation regime favours alts": alts beat BTC (and an alt basket earns
   its premium) when breadth is high, average pairwise correlation is LOW (market not driven by one
   macro/liquidation factor) and cross-sectional dispersion is HIGH (capital rotating into alt
   stories). When correlation is high / breadth low, the market is in a BTC-led or deleveraging
   regime where alts (beta > 1, thin books, forced-selling of leveraged alt longs) underperform.
 - Other side: slow-moving alt holders / late retail who do not de-risk; the regime is observable
   only at market level so there is no single-name arbitrage that removes it.

Universe U50 = lab.hype_universe("1d", top_k=50) (min_hist 60). Alts = U50 ex BTC.
Signals at close t (members of U50 at t, trailing windows only):
  breadth50  = fraction of members with close > SMA50                        (favourable = high)
  dbreadth10 = breadth50 - breadth50 10 days earlier ("breadth thrust")      (favourable = high)
  corr30     = mean pairwise correlation of daily log returns, 30d window    (favourable = LOW)
  corr90     = same, 90d window                                              (favourable = LOW)
  disp7      = cross-sectional std of 7d log returns                         (favourable = high)
  disp30     = 30d mean of the daily cross-sectional std of 1d log returns   (favourable = high)
Normalisation: p_t = percentile of S_t within its trailing 365 days (min 120 obs) - trailing only.
Thresholds: MED  -> favourable iff favourable-oriented p > 0.5
            TERC -> favourable when p > 2/3, unfavourable when p < 1/3, else keep previous state.

PRE-REGISTERED GRID (39 configs):
 B1 spread (alts EW +0.5 / BTC -0.5 when favourable, reversed when unfavourable): 6 signals x {MED,TERC} = 12
 B1c control: spread time-series momentum (sign of 30d EW-alt minus BTC log return)                = 1
 B2 long-only rotation: hold EW alts (gross 1) when favourable else BTC (gross 1), MED               = 6
 B3 overlay on EW long basket of U50 (gross 1 when favourable, 0 else): 6 signals x {MED,TERC}      = 12
 B3t control: trend overlay (EW index > its SMA50)                                                   = 1
 B4 combined overlay: trend AND regime-favourable (MED), 6 signals                                    = 6
 (always-on EW basket = reference, counted as 1)                                                       = 1
Rebalance daily, cost lab.HYPE_COST + funding. Best of B1/B2 and best of B3/B4 stressed fully
(2x cost, lag 1, top_k 30/75, 16-position version = HYPE top-16 basket incl. BTC).
"""
import numpy as np, pandas as pd, json, warnings
import r2lib
warnings.filterwarnings("ignore")
lab = r2lib.lab
BTC = r2lib.BTC
N_TRIALS = 39

C = lab.load("close", "1d")
R = C.pct_change(fill_method=None)
LR = np.log(C).diff()
U50 = lab.hype_universe("1d", top_k=50)
DIRS = {"breadth50": 1, "dbreadth10": 1, "corr30": -1, "corr90": -1, "disp7": 1, "disp30": 1}


def avg_corr(win):
    X = LR.values; M = U50.values
    out = np.full(len(C), np.nan)
    for t in range(win, len(C)):
        cols = M[t] & np.isfinite(X[t - win + 1:t + 1]).all(0)
        if cols.sum() < 5:
            continue
        cm = np.corrcoef(X[t - win + 1:t + 1, cols].T)
        n = cm.shape[0]
        out[t] = (cm.sum() - n) / (n * (n - 1))
    return pd.Series(out, index=C.index)


def signals():
    sma = C.rolling(50, min_periods=50).mean()
    above = (C > sma).where(U50 & sma.notna())
    b50 = above.mean(axis=1)
    S = {"breadth50": b50, "dbreadth10": b50 - b50.shift(10),
         "corr30": avg_corr(30), "corr90": avg_corr(90),
         "disp7": np.log(C / C.shift(7)).where(U50).std(axis=1),
         "disp30": LR.where(U50.shift(1).fillna(False).astype(bool) & U50).std(axis=1).rolling(30, min_periods=20).mean()}
    return pd.DataFrame(S)


def trail_pct(s, win=365, minp=120):
    return s.rolling(win, min_periods=minp).apply(lambda x: (x[np.isfinite(x)] <= x[-1]).mean() if np.isfinite(x[-1]) else np.nan, raw=True)


def state(pf, kind):
    """+1 favourable / -1 unfavourable / NaN unknown. pf already oriented (high = favourable)."""
    if kind == "MED":
        s = np.sign(pf - 0.5).replace(0, -1.0)
        return s.where(pf.notna())
    s = pd.Series(np.nan, index=pf.index)
    cur = np.nan
    for i, v in enumerate(pf.values):
        if np.isfinite(v):
            if v > 2 / 3:
                cur = 1.0
            elif v < 1 / 3:
                cur = -1.0
            elif not np.isfinite(cur):
                cur = np.nan
        s.iloc[i] = cur
    return s


def ew_weights(mask):
    return mask.astype(float).div(mask.sum(1).replace(0, np.nan), axis=0).fillna(0.0)


def spread_w(st, U):
    alts = U.copy(); alts[BTC] = False
    Wa = ew_weights(alts) * 0.5
    Wb = pd.DataFrame(0.0, index=U.index, columns=U.columns); Wb[BTC] = 0.5
    s = st.fillna(0.0)
    return Wa.mul(s, axis=0) - Wb.mul(s, axis=0)


def rotation_w(st, U):
    alts = U.copy(); alts[BTC] = False
    Wa = ew_weights(alts)
    Wb = pd.DataFrame(0.0, index=U.index, columns=U.columns); Wb[BTC] = 1.0
    on = (st == 1).astype(float); off = (st == -1).astype(float)
    return Wa.mul(on, axis=0) + Wb.mul(off, axis=0)


def overlay_w(expo, U):
    return ew_weights(U).mul(expo.fillna(0.0), axis=0)


def ew_index(U, ex_btc=False):
    m = U.shift(1).fillna(False).astype(bool)
    if ex_btc:
        m[BTC] = False
    r = R.where(m).mean(axis=1).fillna(0.0)
    return (1 + r).cumprod()


def build_all(U_trade):
    """Return dict name -> (W, meta) for the full pre-registered grid on trading universe U_trade."""
    S = signals()
    P = {k: trail_pct(S[k]) for k in S}
    out = {}
    for k in S:
        pf = P[k] if DIRS[k] == 1 else 1 - P[k]
        for kind in ("MED", "TERC"):
            st = state(pf, kind)
            out[f"B1_{k}_{kind}"] = spread_w(st, U_trade)
            out[f"B3_{k}_{kind}"] = overlay_w((st == 1).astype(float).where(st.notna()), U_trade)
        stm = state(pf, "MED")
        out[f"B2_{k}_MED"] = rotation_w(stm, U_trade)
        out[f"_st_{k}"] = stm
    alt_idx = ew_index(U50, ex_btc=True)
    spr = np.log(alt_idx / alt_idx.shift(30)) - np.log(C[BTC] / C[BTC].shift(30))
    st_mom = np.sign(spr).replace(0, -1.0).where(spr.notna())
    out["B1c_spread_tsmom30"] = spread_w(st_mom, U_trade)
    idx = ew_index(U50)
    trend = (idx > idx.rolling(50, min_periods=50).mean()).astype(float).where(idx.rolling(50).mean().notna())
    out["B3t_trend_EWidx_sma50"] = overlay_w(trend, U_trade)
    for k in S:
        stm = out[f"_st_{k}"]
        out[f"B4_trend_AND_{k}"] = overlay_w((trend * (stm == 1).astype(float)).where(stm.notna() & trend.notna()), U_trade)
    out["B0_EW_always_on"] = overlay_w(pd.Series(1.0, index=C.index), U_trade)
    out["_trend"] = trend; out["_st_mom"] = st_mom
    out["_S"] = S; out["_P"] = P
    return out


def run_grid():
    G = build_all(U50)
    rows = []
    S = G["_S"]
    print("signal summary (2020-2024):"); print(S.describe().round(3).to_string())
    for name, W in G.items():
        if name.startswith("_"):
            continue
        o, res = r2lib.evaluate(W, name, U=U50, full=False)
        o["family"] = "B"
        # time in favourable state / number of switches
        g = res["df"]["gross_exposure"]
        o["time_in_mkt"] = float((g > 0).mean())
        rows.append(o); r2lib.log(o)
        print(r2lib.fmt(o, keys=("sharpe", "cagr", "maxdd", "sh_2020-21", "sh_2022", "sh_2023-24",
                                  "turnover_per_year", "time_in_mkt")), flush=True)
    for k in S:
        st = G[f"_st_{k}"]; sw = int((st.diff().abs() > 0).sum())
        agree = float((st == G["_st_mom"]).where(st.notna() & G["_st_mom"].notna()).mean())
        agree_t = float(((st == 1).astype(float) == G["_trend"]).where(st.notna() & G["_trend"].notna()).mean())
        print(f"state {k}: switches(MED) {sw}, agreement with spread-tsmom {agree:.2f}, with EW trend {agree_t:.2f}")
    pd.DataFrame(rows).to_csv(r2lib.os.path.join(r2lib.HERE, "famB_grid.csv"), index=False)


if __name__ == "__main__":
    run_grid()
