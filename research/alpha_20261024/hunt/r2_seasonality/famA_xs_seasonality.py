"""Family A - CROSS-SECTIONAL SEASONALITY (Heston-Sadka 2008; Keloharju-Linnainmaa-Nyberg 2016;
crypto: Long, Zaremba et al. 2020 FRL "Seasonality in the cross-section of cryptocurrency returns").

Hypothesis: coins that historically did well (relative to peers) on a given weekday / 4h slot of the
day keep doing so, because of recurring coin-specific calendar flows (fixed-weekday unlocks / reward
distributions / buy-backs, region-specific trading hours: Asia-heavy coins vs US-heavy coins).
Other side: liquidity providers absorbing predictable flows; it persists only if it is smaller than
the cost of harvesting it (the book turns over completely every bar) -> cost is the key test.

PRE-REGISTERED GRID (written before running; 32 core configs + <=6 robustness = <=38)
  A1  1d bars, weekday seasonality. score(i, day D) = mean of coin i's returns on the last N same
      weekdays before D.  N in {8, 13, 26, 52} weeks.
  A2  4h bars, slot-of-day seasonality. score(i, bar B) = mean of coin i's returns in the same 4h
      slot over the last N days.  N in {7, 14, 30, 60} days.
  variant  raw    = same-slot mean
           excess = same-slot mean - mean of ALL bars over the same trailing window
                    (removes the momentum/reversal component; pure seasonal deviation)
  selection n8   = long top 8 / short bottom 8 (the 16-position live version), $-neutral, gross 1
            rank = rank-proportional $-neutral over all names (breadth version), gross 1
  universe  lab.hype_universe("1d", top_k=50) (4h: daily universe known at the previous daily close)
  min obs   same-slot count >= max(4, N//2)
  Robustness (only for the best config of A1 and of A2, by net Sharpe): top_k 30, top_k 75, q=0.2.
  Diagnostics (cost-free, not trading configs): mean daily rank-IC of raw / excess / "other-slot"
  placebo (mean of the non-same-slot bars of the same window) -> tells seasonality vs momentum.
Cost lab.HYPE_COST (0.065%/side) + funding as cost; 2x cost and extra_lag=1 reported.
Signal inputs: close only.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm
from common import lab

N_TRIALS_A = 38


def seasonal_scores(R, p, N):
    """same-slot mean (bars t+1-p*k, k=1..N) known at close t; all-bar trailing mean; other-slot mean."""
    acc = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    cnt = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    for k in range(1, N + 1):
        x = R.shift(p * k - 1)
        acc += x.fillna(0.0)
        cnt += x.notna()
    minobs = max(4, N // 2)
    same = (acc / cnt).where(cnt >= minobs)
    tot_s = R.rolling(p * N, min_periods=p * N // 2).sum()
    tot_c = R.rolling(p * N, min_periods=p * N // 2).count()
    allm = tot_s / tot_c
    # other-slot placebo: window bars that are not the same slot (window = t-pN+1..t)
    # same-slot bars inside that window are k=1..N -> acc/cnt already; other = (tot - acc)/(cnt_all - cnt)
    other = ((tot_s - acc) / (tot_c - cnt)).where(cnt >= minobs)
    return same.astype("float32"), (same - allm).astype("float32"), other.astype("float32")


def rank_ic(score, R, U):
    """mean cross-sectional Spearman IC between score[t] and R[t+1] over names in U[t]."""
    Rn = R.shift(-1)
    m = U & score.notna() & Rn.notna()
    a = score.where(m).rank(axis=1)
    b = Rn.where(m).rank(axis=1)
    a = a.sub(a.mean(axis=1), axis=0); b = b.sub(b.mean(axis=1), axis=0)
    ic = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    ic = ic[m.sum(axis=1) >= 8].dropna()
    t = lab.newey_west_t(ic.values, lags=10)
    return float(ic.mean()), float(t), len(ic)


def run_tf(tf, p, Ns, label, rows, icrows):
    R = lab.returns(tf).astype("float32")
    Ud = cm.hype_U(50)
    U = Ud if tf == "1d" else cm.daily_to_4h(Ud, R.index)
    U = U.reindex(index=R.index, columns=R.columns).fillna(False)
    for N in Ns:
        same, exc, other = seasonal_scores(R, p, N)
        for vn, sc in [("raw", same), ("excess", exc), ("placebo_other", other)]:
            ic, t, n = rank_ic(sc, R, U)
            icrows.append({"family": label, "N": N, "score": vn, "ic_mean": ic, "ic_t": t, "n": n})
            print(f"  IC {label} N={N:3d} {vn:14s} IC {ic:+.4f}  t {t:+.2f}")
        for vn, sc in [("raw", same), ("excess", exc)]:
            for sel in ["n8", "rank"]:
                W = cm.topn_ls(sc, U, 8) if sel == "n8" else cm.rank_ls(sc, U)
                name = f"{label}_N{N}_{vn}_{sel}"
                o = cm.evaluate(W, tf, name, label, N_TRIALS_A, extra={"N": N, "variant": vn, "sel": sel,
                                                                        "top_k": 50})
                rows.append(o)
                print(cm.short_line(o), flush=True)
        del same, exc, other
    return R


def robustness(tf, p, best, label, rows):
    R = lab.returns(tf).astype("float32")
    N, vn = best["N"], best["variant"]
    same, exc, _ = seasonal_scores(R, p, N)
    sc = same if vn == "raw" else exc
    for tk, sel in [(30, best["sel"]), (75, best["sel"]), (50, "q20")]:
        Ud = cm.hype_U(tk)
        U = Ud if tf == "1d" else cm.daily_to_4h(Ud, R.index)
        U = U.reindex(index=R.index, columns=R.columns).fillna(False)
        if sel == "n8":
            W = cm.topn_ls(sc, U, 8)
        elif sel == "rank":
            W = cm.rank_ls(sc, U)
        else:
            W = cm.quint_ls(sc, U, 0.2)
        name = f"{label}_N{N}_{vn}_{sel}_top{tk}_ROB"
        o = cm.evaluate(W, tf, name, label + "_rob", N_TRIALS_A, extra={"N": N, "variant": vn, "sel": sel,
                                                                         "top_k": tk})
        rows.append(o)
        print(cm.short_line(o), flush=True)


if __name__ == "__main__":
    rows, icrows = [], []
    print("== A1 weekday seasonality (1d)")
    run_tf("1d", 7, [8, 13, 26, 52], "A1_weekday", rows, icrows)
    lab.clear()
    print("== A2 4h-slot seasonality (4h)")
    run_tf("4h", 6, [7, 14, 30, 60], "A2_slot4h", rows, icrows)
    df = pd.DataFrame(rows)
    for lbl, tf, p in [("A1_weekday", "1d", 7), ("A2_slot4h", "4h", 6)]:
        sub = df[df["family"] == lbl]
        best = sub.sort_values("sharpe", ascending=False).iloc[0]
        print(f"== robustness for best {lbl}: {best['name']} Sh {best['sharpe']:.2f}")
        robustness(tf, p, best, lbl, rows)
        lab.clear()
    pd.DataFrame(rows).to_csv(os.path.join(cm.HERE, "famA_grid.csv"), index=False)
    pd.DataFrame(icrows).to_csv(os.path.join(cm.HERE, "famA_ic.csv"), index=False)
    print("done")
