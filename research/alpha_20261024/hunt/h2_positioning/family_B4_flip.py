"""POST-HOC (declared, counted): pre-registered B4 'taker imbalance is contrarian' was strongly NEGATIVE in
every year 2021-2024 (Sharpe -1.4..-2.7) -> test the continuation direction (aggressive taker buying =
informed / persistent demand, permanent price impact), and whether it is just XS momentum.
  B4f_imb_k   +imb_k = sum_k tbqv / sum_k qv                        k in {1,3,7}         3
  B4f_imbdm7  +(imb_7 - trailing 90d mean of imb_1)                                      1
  B4f_imb_k|momneutral  resid of rank(imb_k) on rank(ret_k), then on rank(ret_14)  k in {3,7}  2
Total 6 (family B -> 35). Also prints the xsmom14 control stats."""
from h2lib import *
U = lab.universe("1d", top_k=50)
C = lab.load("close", "1d"); QV = lab.load("qv", "1d"); TB = lab.load("tbqv", "1d")
ret14 = C / C.shift(14) - 1
ctrl = controls(); x = ctrl["xsmom14"].dropna()
print("CONTROL xsmom14:", round(lab.metrics(x)["sharpe"], 2), lab.yearly(x).round(2)["sharpe"].to_dict())
imb1m = (TB / QV).rolling(90, min_periods=45).mean()
for k in (1, 3, 7):
    imb = TB.rolling(k).sum() / QV.rolling(k).sum()
    row, _ = evaluate(f"B4f_imb{k}", "B", qls(imb, U), n_note="post-hoc flip"); print(fmt(row))
    if k == 7:
        row, _ = evaluate("B4f_imbdm7", "B", qls(imb - imb1m, U), n_note="post-hoc flip"); print(fmt(row))
    if k in (3, 7):
        rk = C / C.shift(k) - 1
        sres = xs_resid(xs_resid(imb, rk, U), ret14, U)
        row, _ = evaluate(f"B4f_imb{k}|momneutral", "B", qls(sres, U), n_note="post-hoc"); print(fmt(row))
