"""Spanning of RP vs short-horizon continuation controls (1-7d plain momentum L/S, HYPE top-50,
daily rebal, HYPE cost). Controls only - no new signal selection."""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import common as cm, lab
from famA_grid import scores
ctrl = cm.controls().copy()
c = lab.load("close", "1d"); u50 = cm.U(50)
for L in (1, 3, 5, 7):
    ctrl[f"hmom{L}"] = cm.run(lab.quantile_ls_weights(c / c.shift(L) - 1, u50, q=0.2), 1)["df"]["r"]
out = {}
for N in (365, 90):
    S = scores(N)["RP"]
    for nm, W in [("q20", lab.quantile_ls_weights(S, u50, q=0.2)), ("16pos", cm.topn_weights(S, u50, n=8))]:
        r = cm.run(W, 1)["df"]["r"]
        o = {}
        for cols in (["hmom1", "btc"], ["hmom3", "btc"], ["hmom5", "btc"], ["hmom7", "btc"], ["hmom14", "btc"],
                     ["xsmom14", "hmom5", "hmom90", "btc"], ["xsmom14", "hmom1", "hmom3", "hmom5", "hmom30", "hmom90", "btc"]):
            s = cm.spanning(r, ctrl, cols)
            o["+".join(cols)] = round(s["alpha_t"], 2)
        o["corr_hmom5"] = round(cm.corr(r, ctrl["hmom5"]), 2)
        o["corr_hmom1"] = round(cm.corr(r, ctrl["hmom1"]), 2)
        out[f"RP{N}_{nm}"] = o
for k in ("hmom1", "hmom3", "hmom5", "hmom7", "hmom14", "hmom30", "hmom90"):
    out[k + "_sharpe"] = round(cm.sh(ctrl[k]), 2)
print(json.dumps(out, indent=1))
json.dump(out, open("span_short.json", "w"), indent=1)
