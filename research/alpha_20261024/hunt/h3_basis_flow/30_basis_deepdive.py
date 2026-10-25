"""Deep-dive on the only family-A survivor: Binance spot-perp BASIS contrarian (A3/A4).
Research data 2020-2024 only via lab.py.

Why it needs a deep-dive: basis = perp close / Binance spot close - 1 at the SAME instant. It can
contain (i) the perp premium vs the multi-exchange index (= funding driver), (ii) Binance spot's
own premium/discount vs the index, and (iii) last-trade noise in the perp close, which would mean-
revert next bar and create a FAKE reversal profit. We decompose before believing anything.

PRE-REGISTERED ROBUSTNESS CONFIGS (counted as trials; quintile L/S, top-50, daily rebal):
 R1 prem_snap_7       score=-rm(prem,7)    prem = premium index last hourly close of the day
 R2 basis_x_prem_7    score=-rm(basis - prem, 7)   (Binance spot vs index + last-trade noise)
 R3 basis_lvl_7_skip  score=-rm(basis.shift(1), 7)  (drop today's same-instant snapshot)
 R4 basis_z_7_skip    trailing z of the skipped version
 R5 basis_lvl_7_top30 / R6 basis_z_7_top30   universe top_k=30
 R7 basis_lvl_14 / R8 basis_z_14             plateau neighbours on the long side
 R9 basis_lvl_7_top80 / R10 basis_z_7_top80  wider, less liquid universe
 TOTAL 10.
DIAGNOSTICS (no new trading rules): spot-price PnL of the same weights, per-coin contribution,
drop-top-3-contributors, XS rank correlations, rolling 1y Sharpe, $200 equity path.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import h3lib as H
import lab

U = lab.universe("1d", min_hist_days=60, top_k=50)
C = lab.load("close", "1d").astype("float64")
SC = lab.load("spot_close", "1d").astype("float64")
PR = lab.load("prem", "1d").astype("float64")
PM = lab.load("prem_mean", "1d").astype("float64")
F = lab.load("fund", "1d").astype("float64")
BAS = (C / SC - 1).where(SC > 0)
BAS = BAS.where(BAS.abs() < 0.05)


def rm(x, L):
    return x.rolling(L, min_periods=L // 2 + 1).mean() if L > 1 else x


def bz(B, L):
    return (rm(B, L) - B.rolling(90, min_periods=30).mean()) / B.rolling(90, min_periods=30).std()


def xs_sp(a, b, mask):
    a = a.where(mask).rank(axis=1); b = b.where(mask & a.notna()).rank(axis=1); a = a.where(b.notna())
    ac = a.sub(a.mean(axis=1), axis=0); bc = b.sub(b.mean(axis=1), axis=0)
    return ((ac * bc).sum(axis=1) / np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))).mean()


print("=== XS rank correlations (mean daily spearman, top-50) ===")
sigs = {"basis_7": rm(BAS, 7), "basis_z_7": bz(BAS, 7), "prem_snap_7": rm(PR, 7), "prem_mean_7": rm(PM, 7),
        "basis_x_prem_7": rm(BAS - PR, 7), "fund_7": F.rolling(7, min_periods=4).sum(),
        "ret_7": C / C.shift(7) - 1, "ret_14": C / C.shift(14) - 1}
ks = list(sigs)
M = pd.DataFrame(index=ks, columns=ks, dtype=float)
for i, a in enumerate(ks):
    for b in ks[i:]:
        M.loc[a, b] = M.loc[b, a] = xs_sp(sigs[a], sigs[b], U)
print(M.round(2).to_string())
print("\nlevels: median |basis| %.5f, median |prem| %.5f, median |basis-prem| %.5f" % (
    BAS.where(U).abs().stack().median(), PR.where(U).abs().stack().median(), (BAS - PR).where(U).abs().stack().median()))

rows = []; rets = {}
def go(score, name, Uu=U):
    W = lab.quantile_ls_weights(score, Uu, q=0.2)
    o, res = H.evaluate(W, name, "A_basis_robust")
    print(H.fmt(o)); rows.append(o); rets[name] = res
    return W, o, res

print("\n=== robustness configs ===")
go(-rm(PR, 7), "R1_prem_snap_7")
go(-rm(BAS - PR, 7), "R2_basis_x_prem_7")
go(-rm(BAS.shift(1), 7), "R3_basis_lvl_7_skip")
go(-bz(BAS.shift(1), 7), "R4_basis_z_7_skip")
U30 = lab.universe("1d", min_hist_days=60, top_k=30)
go(-rm(BAS, 7), "R5_basis_lvl_7_top30", U30)
go(-bz(BAS, 7), "R6_basis_z_7_top30", U30)
go(-rm(BAS, 14), "R7_basis_lvl_14")
go(-bz(BAS, 14), "R8_basis_z_14")
U80 = lab.universe("1d", min_hist_days=60, top_k=80)
go(-rm(BAS, 7), "R9_basis_lvl_7_top80", U80)
go(-bz(BAS, 7), "R10_basis_z_7_top80", U80)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(H.REG), "robust_results.csv"), index=False)

# ---------------- diagnostics on the two main candidates (re-built, save=False) ----------------
SR = SC.pct_change(fill_method=None)
PRt = C.pct_change(fill_method=None)
for nm, score in {"A3_basis_lvl_7": -rm(BAS, 7), "A4_basis_z_7": -bz(BAS, 7)}.items():
    W = lab.quantile_ls_weights(score, U, q=0.2)
    o, res = H.evaluate(W, nm, "diag", save=False)
    print(f"\n##### {nm} #####\n" + H.fmt(o))
    m = res["metrics"]
    print(f"CAGR {m['cagr']:.1%} vol {m['vol']:.1%} maxDD {m['maxdd']:.1%} t_NW {m['t_nw']:.2f} skew {m['skew']:.2f} "
          f"kurt {m['kurt']:.1f} pos_months {m['pct_pos_months']:.0%} turnover/yr {m['turnover_per_year']:.0f} "
          f"npos {m['avg_n_pos']:.1f} min_equity {o['min_equity']:.0f}")
    print("yearly:", o["yearly"], o["yearly_sh"])
    # spot vs perp price PnL with identical weights (gross, no cost)
    Wn = W.fillna(0.0)
    both = SR.shift(-1).notna() & PRt.shift(-1).notna()
    Wb = Wn.where(both, 0.0)
    g_perp = (Wb * PRt.shift(-1)).sum(axis=1); g_spot = (Wb * SR.shift(-1)).sum(axis=1)
    g_perp = g_perp[g_perp.index >= "2020-03-01"]; g_spot = g_spot[g_spot.index >= "2020-03-01"]
    print(f"gross price PnL same weights: perp {g_perp.mean()*365:+.1%}/yr (Sh {H.sh(g_perp):.2f})  "
          f"spot {g_spot.mean()*365:+.1%}/yr (Sh {H.sh(g_spot):.2f})")
    # lagged-by-one: does spot-return PnL survive when entering one bar later?
    g_spot_l1 = (Wb.shift(1).fillna(0.0) * SR.shift(-1)).sum(axis=1)
    g_spot_l1 = g_spot_l1[g_spot_l1.index >= "2020-03-01"]
    print(f"  spot PnL with 1-bar delay: {g_spot_l1.mean()*365:+.1%}/yr (Sh {H.sh(g_spot_l1):.2f})")
    # per-coin contribution of gross perp PnL
    contrib = (Wn * PRt.shift(-1)).sum(axis=0).sort_values()
    print("bottom-5 coins:", contrib.head(5).round(3).to_dict())
    print("top-5 coins   :", contrib.tail(5).round(3).to_dict())
    print(f"total gross sum {contrib.sum():.2f}; top-5 share {contrib.tail(5).sum()/contrib.sum():.0%}")
    drop = contrib.tail(3).index.tolist()
    Ud = U.copy(); Ud[drop] = False
    Wd = lab.quantile_ls_weights(score, Ud, q=0.2)
    rd = lab.backtest(Wd, "1d")["metrics"]
    print(f"drop top-3 contributors {drop}: Sharpe {rd['sharpe']:.2f} CAGR {rd['cagr']:.1%}")
    # rolling 1y sharpe
    d = res["df"]["r"]
    roll = d.rolling(365).mean() / d.rolling(365).std() * np.sqrt(365)
    print("rolling-1y Sharpe quantiles (5/25/50/75/95%):", roll.dropna().quantile([.05, .25, .5, .75, .95]).round(2).tolist(),
          " frac<0:", round(float((roll.dropna() < 0).mean()), 2))
    # $200 path
    eq = 200 * (1 + d).cumprod()
    print(f"$200 -> ${eq.iloc[-1]:.0f} by {eq.index[-1].date()}; min ${eq.min():.0f}; "
          + " ".join(f"{y}: ${eq[eq.index.year == y].iloc[-1]:.0f}" for y in sorted(set(eq.index.year))))
    # signal decay: IC of score vs fwd return over horizon 1..10
    ics = []
    for h in (1, 2, 3, 5, 7, 10):
        f = C.shift(-h) / C.shift(-h + 1) - 1  # the single-day return h days ahead
        ics.append(round(float(H.rank_ic(score, f, U).mean()), 4))
    print("IC of day-h return (h=1,2,3,5,7,10):", ics)
print("done")
