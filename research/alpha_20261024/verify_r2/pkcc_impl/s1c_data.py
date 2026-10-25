import numpy as np, pandas as pd
from common_pk import *
H, L, C = panels()
O = lab.load("open", "1d").astype("float64")
U = lab.hype_universe("1d", top_k=50)
pd.set_option("display.width", 200)
s = "SOLUSDT"
x = pd.DataFrame({"o": O[s], "h": H[s], "l": L[s], "c": C[s], "U": U[s]}).loc["2022-01-25":"2022-03-05"]
print(x.to_string())
# H==L inside U
m = (H == L) & U
print(m.sum()[m.sum() > 0].to_dict())
print(m.any(axis=1)[m.any(axis=1)].index.tolist()[:20])
pc = C.shift(1)
cc0 = (C == pc) & U
print("zero cc ret days in U per coin top:", cc0.sum().sort_values(ascending=False).head(8).to_dict())
lr_h = np.log((H / L).clip(lower=1.0 + 1e-5)); cc2 = np.log(C / pc) ** 2
Sh = np.sqrt((lr_h**2/(4*np.log(2))).rolling(30, min_periods=24).mean() / cc2.rolling(30, min_periods=24).mean())
inf = np.isinf(Sh) & U
print("inf cells:", inf.sum()[inf.sum() > 0].to_dict(), inf.any(axis=1)[inf.any(axis=1)].index[[0, -1]].tolist())
