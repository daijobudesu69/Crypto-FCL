"""Size / liquidity confound check for the funding factor (diagnostic; +2 configs counted:
 A5_resid_size_L14_t: funding-14 rank residualised on liquidity rank, tercile, top50
 size control itself is a control, not a candidate)."""
from common import *
F = lab.load("fund", "1d"); C = lab.load("close", "1d"); QV = lab.load("qv", "1d")
U = lab.universe("1d", top_k=50)
f14 = F.rolling(14, min_periods=14).sum()
sz = np.log(QV.rolling(30, min_periods=15).median())
rf = lab.xs_rank(f14, U); rs = lab.xs_rank(sz, U)
print("avg XS rank corr(funding14, liquidity):", rf.T.corrwith(rs.T).mean() if hasattr(rf.T, 'corrwith') else None)
cc = []
for t in rf.index[::5]:
    a, b = rf.loc[t], rs.loc[t]
    m = a.notna() & b.notna()
    if m.sum() > 10:
        cc.append(np.corrcoef(a[m], b[m])[0, 1])
cc = pd.Series(cc, index=rf.index[::5][:len(cc)]) if False else pd.Series(cc)
print("spearman(funding14, liq) mean %.3f" % cc.mean())
Wl = lab.quantile_ls_weights(sz, U, q=0.33)   # long big, short small
rsz = lab.backtest(Wl, "1d")
lab.report(rsz, "SIZE control: long liquid tercile / short illiquid")
Wf = lab.quantile_ls_weights(-f14, U, q=0.33)
rfac = lab.backtest(Wf, "1d")
print("corr(funding tercile, size control) daily:", rfac["df"]["r"].corr(rsz["df"]["r"].reindex(rfac["df"].index)))
# leg composition
print("avg liquidity rank of long leg %.2f, short leg %.2f (1=most liquid)" % (rs.where(Wf > 0).stack().mean(), rs.where(Wf < 0).stack().mean()))
# residualise
a = rf.sub(rf.mean(axis=1), axis=0); b = rs.sub(rs.mean(axis=1), axis=0)
beta = (a * b).sum(axis=1) / (b ** 2).sum(axis=1)
res_ = a - b.mul(beta, axis=0)
for q, nm in [(0.33, "t"), (None, "r")]:
    W = lab.quantile_ls_weights(-res_, U, q=q) if q else lab.xs_demean_weights(lab.xs_rank(-res_, U), U)
    d, r = row(W, "A5_resid_size_L14_" + nm, U)
    print(nm, {k: round(v, 2) for k, v in d.items() if isinstance(v, float)})
# also the BTC-relative check (should equal raw): sanity
