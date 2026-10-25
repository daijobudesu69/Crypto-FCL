"""Collect final candidate numbers from the json outputs; DSR with family / all-family trial counts."""
import json, numpy as np, pandas as pd, warnings
import r2lib
warnings.filterwarnings("ignore")
lab = r2lib.lab
# dedupe run log (famB_post was executed twice once by accident; identical rows)
rl = pd.read_csv(r2lib.RUNLOG); n0 = len(rl); rl = rl.drop_duplicates("name", keep="first"); rl.to_csv(r2lib.RUNLOG, index=False)
print(f"runlog rows {n0} -> {len(rl)} unique configs logged")
A = json.load(open("famA_post.json")); B = json.load(open("famB_post.json")); B2 = json.load(open("famB_post2.json"))
keys = ["sharpe", "cagr", "maxdd", "t_nw", "sh_2020-21", "sh_2022", "sh_2023-24", "sharpe_cost2x", "sharpe_lag1",
        "long_leg_sharpe", "short_leg_sharpe", "long_leg_active_sharpe", "short_leg_active_sharpe", "corr_btc", "corr_xsmom",
        "alpha_t_vs_xsmom", "alpha_t_vs_xsmom_btc", "alpha_ann", "avg_n_pos", "min_equity_median", "min_equity_p90",
        "turnover_per_year", "cost_drag_per_year", "funding_per_year", "avg_gross", "skew", "kurt", "days", "yearly", "dsr"]
for nm, o in (("A_best_none", A["best"]), ("A_best_16pos", A["best_16pos"]), ("A_flip", A["flip"]),
              ("B1_disp7_TERC", B["B1_disp7_TERC"]), ("B1_disp7_MED", B["B1_disp7_MED"]),
              ("B1_disp7_TERC_16pos", B["B1_disp7_TERC_16pos"]), ("B1_disp7_MED_16pos", B["B1_disp7_MED_16pos"]),
              ("B4_trend_AND_disp7", B["B4_trend_AND_disp7"]), ("B3_disp7_MED", B["B3_disp7_MED"]),
              ("B3t_trend", B["B3t_trend_EWidx_sma50"]), ("B1_TERC_11pos", B2["B1_disp7_TERC_11pos"]),
              ("B1_TERC_betaneutral", B2["betaneutral"]), ("B1_TERC_oldmajors", B2["oldmaj_sigU50"])):
    print("\n==", nm)
    print({k: (round(o[k], 3) if isinstance(o.get(k), float) else o.get(k)) for k in keys if k in o})
    if "skew" in o:
        for n in (42, 53, 95):
            print(f"   DSR n={n}: {lab.deflated_sharpe(o['sharpe'], n, o['days'], o['skew'], o['kurt'] + 3):.3f}")
    for k in ("alpha_vs_trend_ann", "alpha_vs_trend_t", "corr_trend", "state1_ann", "state1_sr", "state-1_ann", "state-1_sr",
              "alt_leg_sharpe", "btc_leg_sharpe"):
        if k in o:
            print(f"   {k}: {o[k]:.3f}")
print("\nB neighbours:", B["neighbour_sharpes"], "median", B["neighbour_median"])
print("A neighbour median:", A["best_neighbour_median"])
# number of TERC / MED state switches and average spell length
from famB_regime import U50
from bhelp import disp, st_from
for kind in ("TERC", "MED"):
    st = st_from(disp(U50, 7), kind=kind).dropna()
    sw = int((st.diff().abs() > 0).sum())
    print(f"{kind}: switches {sw}, avg spell {len(st)/(sw+1):.1f} days, frac favourable {(st==1).mean():.2f}")
