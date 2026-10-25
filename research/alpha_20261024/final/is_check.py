"""IS-only check of the frozen FCL v1.0 spec and its ablations (2020-03-01 .. 2024-12-31). No ALPHA_FORWARD."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
assert os.environ.get("ALPHA_FORWARD") != "1"
import lab, fcl
out = {}
def sub(d):
    r = d["r"]; o = {}
    for nm, a, b in [("2020-21", "2020", "2022"), ("2022", "2022", "2023"), ("2023-24", "2023", "2025")]:
        x = r[(r.index >= pd.Timestamp(a + "-01-01", tz="UTC")) & (r.index < pd.Timestamp(b + "-01-01", tz="UTC"))]
        o[nm] = round(float(x.mean() / x.std() * 365 ** .5), 2)
    return o
import pandas as pd
for name, kw in [("tercile_base", dict(mode="tercile", stop=None, mom_skip=1.0)),
                 ("tercile_FCLv1", dict(mode="tercile")),
                 ("tercile_FCLv1_cost2x", dict(mode="tercile", cost=2 * lab.HYPE_COST)),
                 ("cap200_base", dict(mode="capacity", stop=None, mom_skip=1.0)),
                 ("cap200_mom_only", dict(mode="capacity", stop=None)),
                 ("cap200_stop_only", dict(mode="capacity", mom_skip=1.0)),
                 ("cap200_FCLv1", dict(mode="capacity")),
                 ("cap200_FCLv1_2023start", dict(mode="capacity", start="2023-01-01"))]:
    st = kw.pop("start", "2020-03-01")
    d, s, _ = fcl.run(st, "2024-12-31", venue="binance", **kw)
    s["sub"] = sub(d); s["yearly"] = {int(k): round(float(v), 3) for k, v in lab.yearly(d["r"])["ret"].items()}
    out[name] = s
    print(f"{name:26s} end ${s['equity_end']:8.0f} SR {s['sharpe']:.2f} CAGR {s['cagr']:+.1%} DD {s['maxdd']:.1%} sub {s['sub']} yrs {s['yearly']} npos {s['avg_n_pos']:.1f} stops {s['stops']} fees ${s['fees']:.0f} fund ${s['funding']:.0f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "is_check.json"), "w"), indent=1, default=str)
