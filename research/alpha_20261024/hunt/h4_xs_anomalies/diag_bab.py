"""Diagnostic only (no selection): what is inside the BAB-BN long leg and where its PnL comes from."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import lab, rolling_beta, shrink_beta, beta_neutral_weights, BTC
C = lab.load("close", "1d"); R = C.pct_change(fill_method=None)
U = lab.universe("1d", min_hist_days=60, top_k=50); U[BTC] = False
b60, _, vx60 = rolling_beta(R, R[BTC], 60); b90, rv90, vx90 = rolling_beta(R, R[BTC], 90)
bshr = shrink_beta(b60.where(U))
W = beta_neutral_weights(-b90, U, bshr, q=0.2)
keep = pd.Series(np.arange(len(W)) % 7 == 0, index=W.index); W = W.where(keep, np.nan).ffill().fillna(0.0)
vol = np.sqrt(vx90 * 365)
corr = b90 * np.sqrt(R[BTC].rolling(90, min_periods=60).var()).values[:, None] / np.sqrt(vx90)
L, S = W > 0, W < 0
print("avg ann vol long leg %.2f  short leg %.2f  universe %.2f" % (vol.where(L).stack().mean(), vol.where(S).stack().mean(), vol.where(U).stack().mean()))
print("avg corr-to-BTC long %.2f short %.2f ; avg beta long %.2f short %.2f" % (corr.where(L).stack().mean(), corr.where(S).stack().mean(), b90.where(L).stack().mean(), b90.where(S).stack().mean()))
print("avg leg gross long %.2f short %.2f" % (W.clip(lower=0).sum(1)[W.abs().sum(1)>0].mean(), -W.clip(upper=0).sum(1)[W.abs().sum(1)>0].mean()))
contrib = (W * R.shift(-1)).fillna(0)
for y in [2021, 2024]:
    cy = contrib[contrib.index.year == y].sum().sort_values()
    print(y, "top +:", cy.tail(6).round(3).to_dict()); print(y, "top -:", cy.head(4).round(3).to_dict())
tot = contrib.sum().sort_values()
print("share of total price PnL from top-5 coins: %.2f" % (tot.tail(5).sum() / tot.sum()))
