"""Final candidate diagnostics (h3_basis_flow). Research data 2020-2024 only via lab.py.

One extra trading config is declared here, POST-HOC (chosen after seeing A3/A4), counted as trial 73:
 C1 basis_combo_7  score = xsrank(-rm(basis,7)) + xsrank(-z90(rm(basis,7)))  quintile L/S top-50 daily
Everything else here is diagnostics on already-registered configs (no new rules):
 regression of daily net returns on BTC, 14d XS-mom control and trailing-funding control (alpha, t_NW),
 Sharpe without the 5 best days, deflated Sharpe with the full trial count, neighbour medians.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import h3lib as H
import lab

U = lab.universe("1d", min_hist_days=60, top_k=50)
C = lab.load("close", "1d").astype("float64")
SC = lab.load("spot_close", "1d").astype("float64")
PM = lab.load("prem_mean", "1d").astype("float64")
F = lab.load("fund", "1d").astype("float64")
TB = lab.load("tbqv", "1d").astype("float64"); QV = lab.load("qv", "1d").astype("float64")
QV = QV.where(QV > 0)
BAS = (C / SC - 1).where(SC > 0); BAS = BAS.where(BAS.abs() < 0.05)


def rm(x, L):
    return x.rolling(L, min_periods=L // 2 + 1).mean() if L > 1 else x


def bz(B, L):
    return (rm(B, L) - B.rolling(90, min_periods=30).mean()) / B.rolling(90, min_periods=30).std()


_r = TB / QV; TBR1 = _r.where((_r > 0.05) & (_r < 0.95))
f7 = F.rolling(7, min_periods=4).sum()
cands = {
    "A3_basis_lvl_7": -rm(BAS, 7),
    "A4_basis_z_7": -bz(BAS, 7),
    "C1_basis_combo_7": lab.xs_rank(-rm(BAS, 7), U) + lab.xs_rank(-bz(BAS, 7), U),
    "A1_pm_lvl_7": -rm(PM, 7),
    "A0_fund7_ctrl": -f7,
    "B1_tbr_lvl_7_cont": (TB.rolling(7, min_periods=7).sum() / QV.rolling(7, min_periods=7).sum()).where(TBR1.notna()),
}
fund_ctrl = lab.backtest(lab.quantile_ls_weights(-f7, U, q=0.2), "1d")["df"]["r"].rename("fund7")
ctrl = H.controls().join(fund_ctrl, how="outer")

reg = pd.read_csv(H.REG).drop_duplicates("name", keep="last")
N_TRIALS = reg["name"].nunique() + 1   # + C1 below
print("trials registered so far:", reg["name"].nunique(), "-> with C1:", N_TRIALS)

out = {}
for nm, score in cands.items():
    W = lab.quantile_ls_weights(score, U, q=0.2)
    save = nm.startswith("C1")
    o, res = H.evaluate(W, nm, "A_basis_posthoc" if save else "diag", save=save)
    d = res["df"]["r"]
    j = pd.concat([d.rename("y"), ctrl], axis=1, sort=True).dropna()
    X = np.column_stack([np.ones(len(j)), j["btc"], j["xsmom14"], j["fund7"]])
    beta, *_ = np.linalg.lstsq(X, j["y"].values, rcond=None)
    resid_alpha = j["y"].values - X[:, 1:] @ beta[1:]
    t_alpha = lab.newey_west_t(resid_alpha, 10)
    a_ann = beta[0] * 365
    best = d.nlargest(5).index
    sh_ex5 = H.sh(d.drop(best))
    m = res["metrics"]
    dsr = lab.deflated_sharpe(m["sharpe"], N_TRIALS, m["days"], m["skew"], m["kurt"] + 3)
    corr_f = float(j["y"].corr(j["fund7"]))
    print(f"\n##### {nm}\n" + H.fmt(o))
    print(f"  alpha vs [btc, xsmom14, fund7] = {a_ann:+.1%}/yr t_NW {t_alpha:.2f}; betas btc {beta[1]:+.3f} "
          f"mom {beta[2]:+.3f} fund7 {beta[3]:+.3f}; corr fund7 {corr_f:+.2f}")
    print(f"  Sharpe ex-5-best-days {sh_ex5:.2f}; best days {[str(x.date()) for x in best]} "
          f"sum {d.loc[best].sum():+.1%}")
    print(f"  DSR(n={N_TRIALS}) = {dsr:.3f}   days {m['days']} skew {m['skew']:.2f} kurt_ex {m['kurt']:.1f} "
          f"t_NW {m['t_nw']:.2f} CAGR {m['cagr']:.1%} maxDD {m['maxdd']:.1%} min_equity {o['min_equity']:.0f} "
          f"npos {m['avg_n_pos']:.1f} turnover {m['turnover_per_year']:.0f}")
    print("  yearly ret", o["yearly"], " yearly Sh", o["yearly_sh"])
    out[nm] = dict(o, alpha_ann=a_ann, alpha_t=t_alpha, sh_ex5=sh_ex5, dsr=dsr, corr_fund7=corr_f)

# neighbour medians from registry (plateau check)
reg = pd.read_csv(H.REG).drop_duplicates("name", keep="last").set_index("name")
nb = {"A3_basis_lvl_7": ["A3_basis_lvl_3", "R7_basis_lvl_14", "A4_basis_z_7", "R5_basis_lvl_7_top30", "R9_basis_lvl_7_top80"],
      "A4_basis_z_7": ["A4_basis_z_3", "R8_basis_z_14", "A3_basis_lvl_7", "R6_basis_z_7_top30", "R10_basis_z_7_top80"],
      "C1_basis_combo_7": ["A3_basis_lvl_7", "A4_basis_z_7", "R3_basis_lvl_7_skip", "R4_basis_z_7_skip"],
      "A1_pm_lvl_7": ["A1_pm_lvl_3", "A1_pm_lvl_14", "A2_pm_z_7", "A6_pm_lvl_7_demean", "A7_pm_lvl_7_rebal3"],
      "B1_tbr_lvl_7_cont": ["B1_tbr_lvl_3_cont", "B2_tbr_z_7_cont"]}
print("\n=== plateau (neighbour) medians ===")
for k, v in nb.items():
    vals = reg.loc[v, "sharpe"]
    print(f"{k:22s} neighbours {dict(vals.round(2))} median {vals.median():.2f}")
    if k in out:
        out[k]["neighbour_median"] = float(vals.median())
pd.DataFrame(out).T.to_csv(os.path.join(os.path.dirname(H.REG), "candidates.csv"))
print("\nfinal trial count:", pd.read_csv(H.REG)["name"].nunique())
