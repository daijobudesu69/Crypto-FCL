"""v2_overfit / step 3: concentration of the candidate's PnL (coins and days). No config selection.

Pre-registered checks:
  - per-coin PnL contribution (price + funding - cost, exact decomposition of lab.backtest), share of the
    top-5 / top-10 coins in total PnL, full sample and 2023-24; split by leg
  - Sharpe after dropping each of the top-5 contributors individually (coin removed from the ranking
    pool, so top-50 refills), and after dropping top-5 / top-10 together (hindsight, harsh)
  - share of total (simple-sum) PnL from the 10 best days; Sharpe without the 10 best days;
    Sharpe without the 10 best and 10 worst days
"""
from vlib import *
import json

out = {}
C = lab.load("close", "1d")
U = lab.universe("1d", top_k=50)
W = cand_weights(14, 0.33, 50, U)
r = lab.backtest(W, "1d")["df"]["r"]
price, fund, costc, tot = coin_pnl(W)
tot = tot.reindex(r.index)
assert np.allclose(tot.sum(axis=1).values, r.values, atol=1e-9)
Wl = W.reindex(index=C.index).fillna(0.0)
sg = np.sign(Wl).where(Wl != 0, np.sign(Wl.shift(1).fillna(0.0)))
sg.index = sg.index + pd.Timedelta("1D"); sg = sg.reindex(r.index)


def conc(tt, label):
    c = tt.sum().sort_values(ascending=False)
    total = c.sum()
    d = {"total_pnl_sum": float(total), "top5": {k.replace("USDT", ""): round(float(v), 3) for k, v in c.head(5).items()},
         "top5_share": float(c.head(5).sum() / total), "top10_share": float(c.head(10).sum() / total),
         "bottom5": {k.replace("USDT", ""): round(float(v), 3) for k, v in c.tail(5).items()},
         "n_coins_positive": int((c > 0).sum()), "n_coins_traded": int((tt.abs().sum() > 0).sum())}
    print(label, json.dumps(d))
    return d, c


out["coins_all"], call = conc(tot, "all")
m2324 = (tot.index >= pd.Timestamp("2023-01-01", tz="UTC"))
out["coins_2023-24"], c2324 = conc(tot[m2324], "2023-24")
out["coins_ex2021"], _ = conc(tot[tot.index.year != 2021], "ex2021")
out["coins_long_leg"], _ = conc(tot.where(sg > 0, 0), "long leg")
out["coins_short_leg"], _ = conc(tot.where(sg < 0, 0), "short leg")
pd.DataFrame({"all": call, "2023-24": c2324}).to_csv(os.path.join(HERE, "coin_contrib.csv"))

# drop each of the top-5 contributors (re-ranked pool so top-50 refills)
allc = list(C.columns)
drops = {}
for coin in list(call.head(5).index):
    Ud = universe_pool(50, [c for c in allc if c != coin])
    rd = lab.backtest(cand_weights(14, 0.33, 50, Ud), "1d")["df"]["r"]
    drops[coin.replace("USDT", "")] = {"sharpe": sh(rd), "sh_2023-24": sh(between(rd, "2023-01-01", "2025-01-01")),
                                        "sh_ex2021": sh(rd[rd.index.year != 2021])}
for n in [5, 10]:
    dl = set(call.head(n).index)
    Ud = universe_pool(50, [c for c in allc if c not in dl])
    rd = lab.backtest(cand_weights(14, 0.33, 50, Ud), "1d")["df"]["r"]
    drops[f"top{n}_together"] = {"sharpe": sh(rd), "sh_2023-24": sh(between(rd, "2023-01-01", "2025-01-01")),
                                 "sh_ex2021": sh(rd[rd.index.year != 2021])}
# drop the top-5 contributors of 2023-24 (harsh, in-period hindsight)
dl = set(c2324.head(5).index)
Ud = universe_pool(50, [c for c in allc if c not in dl])
rd = lab.backtest(cand_weights(14, 0.33, 50, Ud), "1d")["df"]["r"]
drops["top5_of_2023-24_together"] = {"coins": [c.replace("USDT", "") for c in dl], "sharpe": sh(rd),
                                     "sh_2023-24": sh(between(rd, "2023-01-01", "2025-01-01"))}
out["drop_contributors"] = drops
print(json.dumps(drops, indent=1))

# best days
rs = r.sort_values(ascending=False)
out["days"] = {"share_of_sum_from_10_best_days": float(rs.head(10).sum() / r.sum()),
               "share_of_sum_from_20_best_days": float(rs.head(20).sum() / r.sum()),
               "best10_dates": [str(d.date()) for d in rs.head(10).index], "best10_ret": [round(float(x), 4) for x in rs.head(10)],
               "sharpe_ex_10_best": sh(r.drop(rs.head(10).index)),
               "sharpe_ex_10_best_10_worst": sh(r.drop(rs.head(10).index).drop(rs.tail(10).index)),
               "cagr_ex_10_best": float(np.prod(1 + r.drop(rs.head(10).index)) ** (365 / (len(r) - 10)) - 1)}
r34 = between(r, "2023-01-01", "2025-01-01").sort_values(ascending=False)
out["days"]["2023-24_share_from_10_best"] = float(r34.head(10).sum() / r34.sum())
out["days"]["2023-24_sharpe_ex_10_best"] = sh(r34.drop(r34.head(10).index))
print(json.dumps(out["days"], indent=1))
json.dump(out, open(os.path.join(HERE, "s3_conc.json"), "w"), indent=1, default=float)
