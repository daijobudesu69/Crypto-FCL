"""FAMILY A robustness (pre-registered in family_A.py header): best 2 by IS Sharpe = A7_-levelG (0.88)
and A6_zTpos/acct30 (0.51). Variants: rebalance_every=3, top_k=100, q=0.33, momentum-neutral (resid
on XS rank of 14d return). 8 configs.  + 1 counted sign-flip trial: -tsz(top_pos_ls,90) (A2 was -1.26:
hypothesis 'top-trader long tilt = crowded/high-funding longs'), checked against funding.
"""
from h2lib import *
U50 = lab.universe("1d", top_k=50); U100 = lab.universe("1d", top_k=100)
C = lab.load("close", "1d")
G = lab.load("global_ls", "1d"); T = lab.load("top_pos_ls", "1d"); A = lab.load("top_acct_ls", "1d")
T = T.where(T > 0); A = A.where(A > 0); G = G.where(G > 0)
ret14 = C / C.shift(14) - 1
base = {"A7_-levelG": -np.log(G), "A6_zTpos/acct30": tsz(np.log(T / A), 30)}
for n, s in base.items():
    for tag, W, kw in [("reb3", qls(s, U50), {"rebalance_every": 3}),
                       ("top100", qls(s, U100), {}),
                       ("q33", qls(s, U50, q=0.33), {}),
                       ("momneutral", qls(xs_resid(s, ret14, U50), U50), {})]:
        if kw:
            # rebalance_every handled by passing a pre-held weight panel
            keep = np.arange(len(W)) % 3 == 0
            W = W.where(pd.Series(keep, index=W.index), np.nan).ffill().fillna(0.0)
        row, res = evaluate(f"{n}|{tag}", "A", W)
        print(fmt(row))
s = -tsz(T, 90)
row, res = evaluate("A2flip_-zT90", "A", qls(s, U50), n_note="sign flip, counted")
print(fmt(row))
