"""disp_impl step 2c - independent check of correlations of replicated DISP7 (lab, ideal) with BTC and 14d XS momentum,
alpha t (NW, own OLS), and DSR at several trial counts. Outputs -> s2c_out.txt"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import s3_design_lib as L
lab = L.lab
OUT = os.path.dirname(os.path.abspath(__file__)); log = open(os.path.join(OUT, "s2c_out.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n"); log.flush()
U = L.U; BTC = L.BTC; ST = L.STATE
W = U.copy(); W[BTC] = False
W = W.astype(float).div(W.sum(1).replace(0, np.nan), axis=0).fillna(0.0).mul(ST, axis=0) * 0.5
W[BTC] = -0.5 * ST
res = lab.backtest(W, "1d", cost=lab.HYPE_COST); r = res["df"]["r"]; m = res["metrics"]
c = lab.controls()
df = pd.concat([r.rename("r"), c], axis=1, join="inner").dropna()
P("corr BTC %.3f corr xsmom14 %.3f" % (df.r.corr(df.btc), df.r.corr(df.xsmom14)))
y = df.r.values; X = np.column_stack([np.ones(len(y)), df.xsmom14.values])
b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; n = len(y)
Xe = X * e[:, None]; S = Xe.T @ Xe / n
for Lg in range(1, 11):
    G = Xe[Lg:].T @ Xe[:-Lg] / n; S += (1 - Lg / 11) * (G + G.T)
XtXi = np.linalg.inv(X.T @ X); V = n * XtXi @ S @ XtXi
P("alpha ann %.3f  NW t %.2f (vs xsmom14)" % (b[0] * 365, b[0] / np.sqrt(V[0, 0])))
for k in [95, 300, 855, 900]:
    P(f"DSR n={k}: {lab.deflated_sharpe(m['sharpe'], k, m['days'], m['skew'], m['kurt'] + 3):.3f}")
# BTC state-conditional correlation
st_r = ST.shift(1).reindex(df.index)
for s in [1, -1]:
    d = df[st_r == s]; P(f"state {s:+d}: corr BTC {d.r.corr(d.btc):.2f}")
log.close()
