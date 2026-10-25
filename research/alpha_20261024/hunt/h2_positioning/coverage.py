"""Coverage check of Binance metrics fields (1d) inside the point-in-time top-50/100 universes."""
import sys; sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab, pandas as pd, numpy as np
C = lab.load("close", "1d")
print("panel", C.shape, C.index[0], C.index[-1])
for k in (50, 100):
    U = lab.universe("1d", top_k=k)
    print(f"\n== top{k}: avg names in U per year and fraction with field present ==")
    rows = {}
    for f in ["oi_usd", "top_pos_ls", "top_acct_ls", "global_ls", "taker_ls_vol", "fund"]:
        X = lab.load(f, "1d")
        have = (X.notna() & U).sum(axis=1)
        nU = U.sum(axis=1)
        rows[f] = (have.groupby(have.index.year).mean()).round(1)
    rows["nU"] = U.sum(axis=1).groupby(U.index.year).mean().round(1)
    print(pd.DataFrame(rows).to_string())
# raw distribution snapshot
for f in ["top_pos_ls", "top_acct_ls", "global_ls", "taker_ls_vol"]:
    X = lab.load(f, "1d")
    print(f, X.stack().describe().round(3).to_dict())
print("first valid date for a few:", {c: str(lab.load('global_ls','1d')[c].first_valid_index()) for c in ['BTCUSDT','ETHUSDT','SOLUSDT','DOGEUSDT','LINKUSDT']})
