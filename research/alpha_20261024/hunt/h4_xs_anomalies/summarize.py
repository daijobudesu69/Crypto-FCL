"""Summary diagnostics for the candidates (no new selection): DSR, plateau, BAB-BN decomposition."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import lab, rolling_beta, shrink_beta, beta_neutral_weights, BTC, controls

HERE = os.path.dirname(os.path.abspath(__file__))
g = pd.read_csv(os.path.join(HERE, "grid_log.csv")).drop_duplicates("name", keep="last").set_index("name")
g["days"] = (pd.to_datetime(g["end"]) - pd.to_datetime(g["start"])).dt.days + 1
print("configs in log:", len(g), g["family"].value_counts().to_dict())

# effective trial counts (sign flips chosen ex post count as extra trials)
N_A = 40 + 40 + 4
N_B = 36 + 36 + 13
N_ALL = N_A + N_B
cands = {"B_BAB_b90_BN_R7": (N_B, ["B_BAB_b60_BN_R7", "B_BAB_b90_BN_R1", "B_BAB_b90_DN_R7"]),
         "BR_SKEWrev_4h30_DN_R1": (N_B, ["BR_SKEWrev_4h30_DN_R7", "BR_SKEWrev_d30_DN_R1", "BR_SKEWrev_4h30_DN_R1_rob_U30",
                                         "BR_SKEWrev_4h30_DN_R1_rob_U100", "BR_SKEWrev_4h30_DN_R1_rob_q30"]),
         "A4_h3_RAW_LOWAV": (N_A, ["A4_h6_RAW_LOWAV", "A3_h1_RAW_B", "A3_h3_RAW_B", "A4_h3_RAW_HIGHAV"]),
         "AF2_mom3_RESID_S3": (N_A, ["AF1_mom3_RESID_D", "AF3_mom3_RAW_HIGHAV", "AF4_mom3_RESID_HIGHAV"]),
         "A2_L3_RAW_LOWAV": (N_A, ["A2_L1_RAW_LOWAV", "A2_L3_RESID_LOWAV", "A2_L3_RAW_HIGHAV"]),
         "B_SKEW_d90_BN_R7": (N_B, ["B_SKEW_d90_BN_R1", "B_SKEW_d90_DN_R7", "B_SKEW_d30_BN_R7"])}
cols = ["sharpe", "sharpe_gross_nocost", "sharpe_cost2x", "sharpe_lag1", "sh_2020-21", "sh_2022", "sh_2023-24", "cagr", "maxdd",
        "t_nw", "turnover_per_year", "cost_drag_per_year", "funding_per_year", "gross_price_per_year", "avg_gross", "avg_net",
        "avg_n_pos", "long_leg_sharpe", "short_leg_sharpe", "long_leg_excess_vs_EW_sharpe", "short_leg_excess_vs_EW_sharpe",
        "long_leg_excess_ann", "short_leg_excess_ann", "corr_btc", "corr_xsmom", "min_equity_usd", "min_equity_usd_p90", "start", "end", "days"]
for c, (n, nb) in cands.items():
    r = g.loc[c]
    dsr_f = lab.deflated_sharpe(r["sharpe"], n, r["days"], r["skew"], r["kurt"] + 3)
    dsr_a = lab.deflated_sharpe(r["sharpe"], N_ALL, r["days"], r["skew"], r["kurt"] + 3)
    print(f"\n=== {c}  DSR(n={n}) {dsr_f:.3f}  DSR(n={N_ALL}) {dsr_a:.3f}  neighbour median SR {g.loc[nb, 'sharpe'].median():.2f} "
          f"({', '.join(f'{x}:{g.loc[x, 'sharpe']:.2f}' for x in nb)})")
    print(r[cols].to_string())
    print("yearly:", r["yearly"])

# ---- BAB-BN decomposition: is it beta sorting, or just a net-long beta-hedged alt basket?
C = lab.load("close", "1d"); R = C.pct_change(fill_method=None)
U = lab.universe("1d", min_hist_days=60, top_k=50); U[BTC] = False
b60, _, _ = rolling_beta(R, R[BTC], 60); b90, _, _ = rolling_beta(R, R[BTC], 90)
bshr = shrink_beta(b60.where(U))
W = beta_neutral_weights(-b90, U, bshr, q=0.2)
keep = pd.Series(np.arange(len(W)) % 7 == 0, index=W.index); W = W.where(keep, np.nan).ffill().fillna(0.0)
bab = lab.backtest(W, "1d")["df"]["r"]
# beta-hedged EW alt basket: long EW universe (ex BTC) gross 1, short its avg shrunk beta in BTC
m = U & bshr.notna()
Wew = m.astype(float).div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
Wew[BTC] = -(Wew * bshr.fillna(0)).sum(axis=1)
alt = lab.backtest(Wew, "1d")["df"]["r"]
j = pd.concat([bab.rename("bab"), alt.rename("alt_hedged"), controls()], axis=1).dropna()
X = np.column_stack([np.ones(len(j)), j[["alt_hedged", "btc"]].values])
b, *_ = np.linalg.lstsq(X, j["bab"].values, rcond=None)
e = j["bab"].values - X @ b
print("\nBAB_b90_BN_R7 on beta-hedged EW alt basket + BTC: alpha/yr "
      f"{b[0]*365:+.1%} t~{lab.newey_west_t(e + b[0], 10):.2f}  b_alt {b[1]:+.2f}  b_btc {b[2]:+.2f}  "
      f"corr {j['bab'].corr(j['alt_hedged']):+.2f}  R2 {1 - e.var() / j['bab'].var():.2f}")
ra = lab.backtest(Wew, "1d")
print("beta-hedged EW alt basket itself: SR", round(ra["metrics"]["sharpe"], 2))
print(lab.yearly(ra["df"]["r"]).round(2).T.to_string())
