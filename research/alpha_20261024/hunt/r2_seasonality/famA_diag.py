"""Diagnostics for family A (no new trading configs): where does the 4h-slot gross come from,
break-even cost, gross by slot-of-day, gross lag-1, gross by year, momentum overlap.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm
from common import lab
from famA_xs_seasonality import seasonal_scores

R = lab.returns("4h").astype("float32")
U = cm.daily_to_4h(cm.hype_U(50), R.index).reindex(index=R.index, columns=R.columns).fillna(False)
for N, vn in [(60, "raw"), (60, "excess"), (30, "raw")]:
    same, exc, _ = seasonal_scores(R, 6, N)
    sc = same if vn == "raw" else exc
    W = cm.topn_ls(sc, U, 8)
    for lag in [0, 1, 2, 6]:
        res = lab.backtest(W, "4h", cost=0.0, extra_lag=lag)
        df = res["df"]
        g = df["gross_r"] + df["fund_r"]
        ann = g.mean() * 2190
        turn = df["turnover"].mean() * 2190
        print(f"N{N} {vn:6s} lag{lag}: gross ann {ann:+.1%} Sharpe(d) {cm.sharpe(lab.to_daily(g,'4h')):+.2f} "
              f"turn/yr {turn:.0f} break-even cost/side {ann/turn*1e4:.2f} bp")
        if lag == 0:
            slot = (df.index - pd.Timedelta("4h")).hour  # df index = bar end; held bar start hour
            # held bar = bar t+1 whose END is df.index -> start = index - 4h
            by = g.groupby(slot).mean() * 2190 / 6
            print("   gross ann contribution by held-bar start hour (UTC):",
                  ", ".join(f"{h:02d}h {v:+.1%}" for h, v in by.items()))
            yr = g.groupby(df.index.year).mean() * 2190
            print("   gross ann by year:", ", ".join(f"{k} {v:+.1%}" for k, v in yr.items()))
            d = lab.to_daily(g, "4h")
            c = cm.controls()
            j = pd.concat([d.rename("s"), c], axis=1, join="inner").dropna()
            b, t, n = cm.hac_ols(j["s"].values, j[["xsmom14", "btc"]].values)
            print(f"   gross daily vs xsmom/btc: alpha ann {b[0]*365:+.1%} t {t[0]:+.2f} beta_mom {b[1]:+.2f} "
                  f"corr_mom {j['s'].corr(j['xsmom14']):+.2f}")
    del same, exc
