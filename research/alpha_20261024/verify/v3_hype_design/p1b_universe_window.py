"""Part 1b diagnostic: is the overlap-window weakness a universe effect (HYPE-listed subset) or a time
effect? Candidate tercile (Binance signal, Binance returns, cost 0.07%) on: full top-50, HYPE-listed
top-50 subset (hype_universe), non-HYPE subset of top-50. Windows: yearly + 2023H1/2023H2.
Diagnostic only (3 universes of the same rule, no selection)."""
from common_v3 import *
import json
F = lab.load("fund", "1d"); s = -f14(F)
U50 = lab.universe("1d", top_k=50); Uh = lab.hype_universe("1d", top_k=50); Un = U50 & ~Uh
wins = [("2020", "2020-03-01", "2021-01-01"), ("2021", "2021-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"),
        ("2023H1", "2023-01-01", "2023-06-01"), ("2023H2", "2023-06-01", "2024-01-01"), ("2024", "2024-01-01", "2025-01-01"),
        ("ovl_2023-06..24", "2023-06-01", "2025-01-01"), ("full", "2020-03-01", "2025-01-01")]
out = {}
for name, U in [("top50_all", U50), ("top50_hype", Uh), ("top50_nonhype", Un)]:
    W = lab.quantile_ls_weights(s, U, q=0.33)
    res = lab.backtest(W, "1d", cost=0.0007); df = res["df"]
    d = {}
    for w, a, b in wins:
        x = window(df, a, b)
        d[w] = {"sh": round(sh(x["r"]), 2), "ret": round(float((1 + x["r"]).prod() - 1), 3),
                "fund": round(float(x["fund_r"].sum()), 3), "price": round(float(x["gross_r"].sum()), 3)}
    # legs on the overlap window (vs EW of the same universe)
    R = lab.returns("1d").shift(-1); ew = R.where(U & R.notna()).mean(axis=1); ew.index = ew.index + pd.Timedelta("1D")
    for side, Wl in [("short", W.clip(upper=0)), ("long", W.clip(lower=0))]:
        rl = lab.backtest(Wl * 2, "1d", cost=0.0007)["df"]["r"]
        e = ew.reindex(rl.index).fillna(0); ex = rl + e if side == "short" else rl - e
        d[f"{side}_vsEW_full"] = round(sh(ex), 2); d[f"{side}_vsEW_ovl"] = round(sh(window(ex, "2023-06-01", "2025-01-01")), 2)
        d[f"{side}_vsEW_2021-24"] = round(sh(window(ex, "2021-01-01", "2025-01-01")), 2)
    d["npos"] = round(res["metrics"]["avg_n_pos"], 1)
    out[name] = d
    print(name, json.dumps(d))
json.dump(out, open("p1b_universe_window.json", "w"), indent=1)
