"""Diagnostic: quantile_ls_weights is not sign-symmetric (pct-rank cut gives 8 long / 7 short names),
so a flipped score is NOT the exact negative of the original book; the marginal name matters."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import lab, rolling_beta, shrink_beta, BTC
C = lab.load("close", "1d"); R = C.pct_change(fill_method=None)
U = lab.universe("1d", min_hist_days=60, top_k=50); U[BTC] = False
beta60, _, _ = rolling_beta(R, R[BTC], 60); bshr = shrink_beta(beta60.where(U))
sk = R.rolling(30, min_periods=20).skew()
m = U & sk.notna() & bshr.notna()
Wo = lab.quantile_ls_weights(-sk, m, q=0.2); Wr = lab.quantile_ls_weights(sk, m, q=0.2)
print("n long / n short (median):", (Wo>0).sum(1).median(), (Wo<0).sum(1).median())
for nm, W in [("orig", Wo), ("rev", Wr), ("neg_orig", -Wo)]:
    d = lab.backtest(W, "1d", cost=0.0)["df"]; g = d["gross_r"] + d["fund_r"]
    print(nm, g.groupby(g.index.year).apply(lambda x: x.mean()/x.std()*np.sqrt(365)).round(2).to_dict(), round(g.mean()/g.std()*np.sqrt(365),2))
