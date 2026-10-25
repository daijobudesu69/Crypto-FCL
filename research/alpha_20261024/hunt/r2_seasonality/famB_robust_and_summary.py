"""Family B robustness for the best IS config (no config passed the pre-registered EST->VAL rule, so this
is descriptive only): TOM33 on the EW basket with 16 names (live-sized), top-30, top-75.
Counts 3 more configs toward family B (total 27). Then prints the summary table of the best attempt per
family from runlog.csv (incl. neighbour medians).
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm
from common import lab
import famB_calendar as fb

rows = []
for tk in [16, 30, 75]:
    EWb = fb.ew_base(tk)
    W, sig = fb.build_W("TOM33", "EW", EWb)
    o = cm.evaluate(W, "1d", f"B_TOM33_EW_top{tk}_ROB", "B_TOM33_rob", 27, extra={"top_k": tk})
    x = fb.asset_daily_ret("EW", EWb)
    ei, eo, e_t, _ = fb.stats_inout(x, fb.WIN["TOM33"], *cm.EST)
    vi, vo, v_t, _ = fb.stats_inout(x, fb.WIN["TOM33"], *cm.VAL)
    print(cm.short_line(o) + f" | EST t {e_t:+.2f} VAL t {v_t:+.2f}")
    rows.append(o)

pd.DataFrame(rows).to_csv(os.path.join(cm.HERE, "famB_robust.csv"), index=False)
log = pd.concat([pd.read_csv(os.path.join(cm.HERE, f)) for f in
                 ["famA_grid.csv", "famA3_grid.csv", "famB_grid.csv", "famB_robust.csv"]], ignore_index=True)
best = ["A1_weekday_N13_raw_n8", "A2_slot4h_N60_raw_n8", "A3_blk12h_N30_raw_n8", "B_TOM33_ETH", "B_TOM33_EW",
        "B_LAST3_BTC"]
nb = {"A1_weekday_N13_raw_n8": ["A1_weekday_N8_raw_n8", "A1_weekday_N26_raw_n8", "A1_weekday_N13_raw_rank",
                                "A1_weekday_N13_excess_n8"],
      "A2_slot4h_N60_raw_n8": ["A2_slot4h_N30_raw_n8", "A2_slot4h_N60_raw_rank", "A2_slot4h_N60_excess_n8"],
      "A3_blk12h_N30_raw_n8": ["A3_blk12h_N60_raw_n8", "A3_blk8h_N30_raw_n8", "A3_blk12h_N30_excess_n8"],
      "B_TOM33_ETH": ["B_TOM23_ETH", "B_LAST3_ETH", "B_FIRST3_ETH", "B_TOM33_EW", "B_TOM33_BTC"],
      "B_TOM33_EW": ["B_TOM23_EW", "B_LAST3_EW", "B_FIRST3_EW", "B_TOM33_ETH", "B_TOM33_BTC"],
      "B_LAST3_BTC": ["B_LAST3_ETH", "B_LAST3_EW", "B_TOM33_BTC", "B_TOM23_BTC"]}
L = log.drop_duplicates("name", keep="last").set_index("name")
cols = ["sharpe", "sharpe_gross_nocost", "cagr", "maxdd", "t_nw", "sh_2020-21", "sh_2022", "sh_2023-24",
        "sharpe_cost2x", "sharpe_lag1", "long_leg_sharpe", "short_leg_sharpe", "corr_btc", "corr_xsmom",
        "alpha_t_vs_mom_btc", "alpha_t_vs_mom", "avg_n_pos", "min_equity_usd", "turnover_per_year",
        "cost_drag_per_year", "funding_per_year", "dsr", "yearly"]
for b in best:
    r = L.loc[b]
    med = float(np.nanmedian([L.loc[n, "sharpe"] for n in nb[b]]))
    print(f"\n== {b}  (neighbour median Sharpe {med:+.2f})")
    for c in cols:
        print(f"   {c:22s} {r[c]}")
print("\nconfig counts:", log.groupby(log["family"].str[:2]).size().to_dict())
