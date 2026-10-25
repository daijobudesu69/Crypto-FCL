"""v2_overfit / step 6: DSR of sub-series (ex-2021, 2022-24, 2023-24) and the pre-registered rank twin.
Fixed diagnostics, no selection. Twin = h1 A1_L14_top50_rank_rb1 (lab.xs_demean_weights of the F14 rank)."""
from vlib import *
import json
out = {}
r = pd.read_parquet(os.path.join(HERE, "cand_daily_IS.parquet"))["r"]
U = lab.universe("1d", top_k=50)
rt = lab.backtest(lab.xs_demean_weights(lab.xs_rank(signal(14), U), U), "1d")["df"]["r"]
for name, x in [("cand", r), ("twin_rank", rt)]:
    for tag, y in [("full", x), ("ex2021", x[x.index.year != 2021]), ("2022-24", between(x, "2022-01-01", "2025-01-01")),
                   ("2023-24", between(x, "2023-01-01", "2025-01-01"))]:
        s = sh(y); sk = float(y.skew()); ku = float(y.kurt()) + 3
        out[f"{name}_{tag}"] = {"sharpe": s, "days": len(y), "t_nw": float(lab.newey_west_t(y.values, 10)),
                                "skew": sk, "kurt_raw": ku,
                                **{f"DSR_n{n}_sd{sd}": lab.deflated_sharpe(s, n, len(y), sk, ku, (sd / np.sqrt(365)) ** 2)
                                   for n in [433, 71] for sd in [0.5, 0.61]}}
    b = stationary_bootstrap_sharpes(x, 20, 2000, seed=21)
    out[f"{name}_boot20"] = {"ci90": [float(np.percentile(b, 5)), float(np.percentile(b, 95))], "p_lt_0": float((b < 0).mean()),
                             "p_lt_0.5": float((b < 0.5).mean())}
out["corr_cand_twin"] = float(r.corr(rt.reindex(r.index)))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "s6_extra.json"), "w"), indent=1, default=float)
