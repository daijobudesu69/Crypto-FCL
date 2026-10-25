"""Extra diagnostics for the Family A candidate (no new signal configs):
 (a) rolling 365d Sharpe; (b) price PnL measured on SPOT returns (is it just perp-basis convergence?);
 (c) correlation with 3d short-term reversal control; (d) $200 equity path; (e) yearly leg-vs-EW Sharpe.
"""
from common import *
F = lab.load("fund", "1d"); C = lab.load("close", "1d"); S = lab.load("spot_close", "1d")
U = lab.universe("1d", top_k=50)
s = -F.rolling(14, min_periods=14).sum()
W = lab.quantile_ls_weights(s, U, q=0.33)
res = lab.backtest(W, "1d"); r = res["df"]["r"]
rs = r.rolling(365).apply(lambda x: x.mean() / x.std() * np.sqrt(365))
print("rolling 365d Sharpe: min %.2f  10pct %.2f  median %.2f  max %.2f" % (rs.min(), rs.quantile(.1), rs.median(), rs.max()))
print("rolling Sharpe at year ends:", rs.resample("YE").last().round(2).to_dict())
# (b) spot-based price pnl on coins with spot
Rp = C.pct_change(fill_method=None).shift(-1); Rs = S.pct_change(fill_method=None).shift(-1)
ok = Rp.notna() & Rs.notna()
Wv = W.where(ok, 0)
pp = (Wv * Rp.fillna(0)).sum(axis=1); ps = (Wv * Rs.fillna(0)).sum(axis=1)
print("coverage of |W| with spot: %.2f" % (Wv.abs().sum(axis=1).sum() / W.abs().sum(axis=1).sum()))
print("price pnl/yr perp-returns %.3f (Sh %.2f) vs spot-returns %.3f (Sh %.2f)" % (pp.mean() * 365, sh(pp), ps.mean() * 365, sh(ps)))
# (c) reversal control
rev = lab.backtest(lab.quantile_ls_weights(-(C / C.shift(3) - 1), U, q=0.2), "1d")["df"]["r"]
print("corr with 3d reversal control: %.2f ; reversal control Sharpe %.2f" % (r.corr(rev.reindex(r.index)), sh(rev)))
cb, cm = corr_ctrl(r); print("corr btc %.2f, xsmom14 %.2f" % (cb, cm))
# (d) $200 path
eq = 200 * (1 + r).cumprod()
print("$200 -> year-end equity:", eq.resample("YE").last().round(0).to_dict())
# (e) yearly legs vs EW
ew = ew_univ(U)
for side, Wl in [("long", W.clip(lower=0)), ("short", W.clip(upper=0))]:
    rl = lab.backtest(Wl * 2, "1d")["df"]["r"]
    e = ew.reindex(rl.index).fillna(0)
    x = rl - e if side == "long" else rl + e
    print(side, "leg vs EW yearly Sharpe:", x.groupby(x.index.year).apply(sh).round(2).to_dict(),
          " ann ret:", (x.groupby(x.index.year).sum()).round(3).to_dict())
m = res["metrics"]
print("DSR (n=48 famA configs): %.3f ; (n=69 all h1 configs): %.3f" % (
    lab.deflated_sharpe(m["sharpe"], 48, m["days"], m["skew"], m["kurt"] + 3),
    lab.deflated_sharpe(m["sharpe"], 69, m["days"], m["skew"], m["kurt"] + 3)))
print("worst month %.3f best month %.3f pct pos months %.2f skew %.2f kurt %.2f" % (m["worst_month"], m["best_month"], m["pct_pos_months"], m["skew"], m["kurt"]))
# gross leverage needed so every position >= $10 at $200
a = W.abs().where(W.abs() > 0); mn = a.min(axis=1).dropna()
print("median #pos %.1f; gross needed for $10 min at $200 equity (median day): %.2f" % ((W != 0).sum(axis=1)[W.abs().sum(axis=1) > 0].median(), 10 / (200 * mn.median())))
