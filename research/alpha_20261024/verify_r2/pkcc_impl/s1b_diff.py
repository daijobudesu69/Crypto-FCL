"""Step 1b: explain gap vs hunter. Build hunter's score exactly as final_numbers.py (copied formula) and diff vs own."""
import numpy as np, pandas as pd
from common_pk import *
H, L, C = panels()
U = lab.hype_universe("1d", top_k=50)
pc = C.shift(1)
lr_h = np.log((H / L).clip(lower=1.0 + 1e-5))
pk2 = lr_h ** 2 / (4 * np.log(2)); cc2 = np.log(C / pc) ** 2
Sh = np.sqrt(pk2.rolling(30, min_periods=24).mean() / cc2.rolling(30, min_periods=24).mean())
So = ratio_signal(H, L, C, 30, 24)
d = (Sh - So).abs()
print("signal abs diff max", np.nanmax(d.values), "n cells >1e-6 within U:", int(((d > 1e-6) & U).sum().sum()))
print("cells where one is NaN and other not, within U:", int(((Sh.isna() != So.isna()) & U).sum().sum()))
print("inf in hunter score within U:", int((np.isinf(Sh) & U).sum().sum()))
bad = ((d > 1e-6) & U)
print("columns with diffs:", bad.sum()[bad.sum() > 0].sort_values(ascending=False).head(15).to_dict())
# data sanity: H==L days, H<max(O,C), L>min(O,C), zero-return streaks
O = lab.load("open", "1d").astype("float64")
inU = U
print("days H==L inside U:", int(((H == L) & inU).sum().sum()))
print("days H<C or L>C inside U:", int((((H < C) | (L > C)) & inU).sum().sum()))
# where do rows differ
rows = bad.any(axis=1)
print("dates with diff:", rows.sum(), rows[rows].index[:5].tolist(), rows[rows].index[-5:].tolist())
