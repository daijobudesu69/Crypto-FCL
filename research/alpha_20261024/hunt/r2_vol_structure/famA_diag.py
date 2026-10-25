"""FAMILY A diagnostics (post-hoc, counted as extra trials):
 Is the best grid cell (A_bminus_90_FLIP_BN = long LOW downside beta, beta-neutral) just BAB?
 D1  BAB-BN reference books (score = -beta_W, BN, R7) for W=90,180         (2 trials)
 D2  pure downside component: bminus residualized cross-sectionally on beta each day, FLIP, BN, W=90/180 (2 trials)
 D3  spanning regression of the best cell on BAB-BN(90) + xsmom14 + btc; daily return corr
 D4  per-coin price-PnL concentration of the best cell; avg XS Spearman(bminus, beta)
"""
import numpy as np, pandas as pd
from common import universe, ls_weights, bn_weights, evaluate, fmt_row, ols_nw, controls, lab, COST, returns
from signals import beta_family


def xs_resid(y, x, mask):
    """Daily cross-sectional OLS residual of y on x (within mask)."""
    y = y.where(mask); x = x.where(mask)
    ok = y.notna() & x.notna()
    y = y.where(ok); x = x.where(ok)
    xm = x.sub(x.mean(axis=1), axis=0); ym = y.sub(y.mean(axis=1), axis=0)
    b = (xm * ym).sum(axis=1) / (xm * xm).sum(axis=1)
    return ym - xm.mul(b, axis=0)


if __name__ == "__main__":
    U = universe(50, ex_btc=True)
    R = returns()
    out = {}
    for W in (90, 180):
        S = beta_family(W)
        bab = bn_weights(-S["beta"], U, S["beta"])
        o, r_bab = evaluate(bab, f"BAB_BN_{W}", rebalance=7, mask=U)
        print(o["name"].ljust(24), fmt_row(o), f"alpha_t={o['alpha_t']:.2f}", o["yearly"])
        res_bm = xs_resid(S["bminus"], S["beta"], U)
        pure = bn_weights(-res_bm, U, S["beta"])
        o2, r_pure = evaluate(pure, f"PURE_bminus_resid_{W}_FLIP_BN", rebalance=7, mask=U)
        print(o2["name"].ljust(24), fmt_row(o2), f"alpha_t={o2['alpha_t']:.2f}", o2["yearly"])
        best = bn_weights(-S["bminus"], U, S["beta"])
        ob, r_best = evaluate(best, f"A_bminus_{W}_FLIP_BN", rebalance=7, mask=U)
        print(ob["name"].ljust(24), fmt_row(ob), ob["yearly"])
        # rank correlation of bminus with beta
        sp = S["bminus"].where(U).rank(axis=1).corrwith(S["beta"].where(U).rank(axis=1), axis=1)
        print(f"  W={W}: mean daily XS Spearman(bminus, beta) = {sp.mean():.2f};"
              f" (bplus,beta) = {S['bplus'].where(U).rank(axis=1).corrwith(S['beta'].where(U).rank(axis=1), axis=1).mean():.2f}")
        # spanning
        j = pd.concat([r_best['df']['r'].rename('s'), r_bab['df']['r'].rename('bab'), controls()], axis=1).dropna()
        print(f"  corr(best, BAB_BN) = {j['s'].corr(j['bab']):.2f}")
        b, t, n = ols_nw(j['s'], j[['bab', 'xsmom14', 'btc']])
        print("  span on BAB+xsmom+btc: alpha/yr {:+.1%} t {:.2f} | b_bab {:.2f} (t {:.1f}) b_xs {:.2f} b_btc {:.2f}".format(
            b['const'] * 365, t['const'], b['bab'], t['bab'], b['xsmom14'], b['btc']))
        # concentration of price pnl (target weights held 7d)
        Wt = best.copy(); keep = np.arange(len(Wt)) % 7 == 0
        Wt = Wt.where(pd.Series(keep, index=Wt.index), np.nan).ffill().fillna(0)
        pnl = (Wt * R.shift(-1)).sum()
        tot = pnl.sum()
        top = pnl.sort_values(ascending=False)
        print(f"  price PnL total {tot:+.2f}; top5 {top.head(5).round(2).to_dict()} share {top.head(5).sum() / tot:.0%};"
              f" bottom5 {top.tail(5).round(2).to_dict()}")
        # long vs short contributions
        pl = (Wt.clip(lower=0) * R.shift(-1)).sum().sum(); ps = (Wt.clip(upper=0) * R.shift(-1)).sum().sum()
        print(f"  long-leg price pnl {pl:+.2f}  short-leg price pnl {ps:+.2f}")
        del S
